"""(d) Sizing: volatility forecasts (20-night realized vs EWMA vs GARCH(1,1) vs a VIX regression), inverse-variance
(Moreira-Muir) and fractional Kelly with Bayesian shrinkage of the mean, and a gross-exposure cap that keeps
the plausible worst night within $100 on the $1,000 sleeve. All with the 200-day trend filter, cap 3x per leg.

Fitted pieces are walk-forward: GARCH and the VIX regression refit every Jan 1 on prior data; the inverse-variance
constant is calibrated on 2001-2007 only; the Kelly mean is an expanding-window posterior (prior N(0, tau^2)).
Forecast accuracy: QLIKE loss on overnight returns, OOS 2008-2026, Diebold-Mariano t vs the 20-night sd.
"""
import sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, signals as sg, evalx  # noqa: E402

STRESS_LIMIT = 0.10   # $100 on $1,000


def vix_reg_vol(P):
    """sigma forecast for the next night = k * VIX / sqrt(252), with k^2 = sum(r_on^2) / sum(VIX^2/252) estimated
    walk-forward (Jan 1 refits, all prior nights): the share of implied variance realized overnight."""
    v = sg._align(sg.market_series("^VIX"), P["c"].index) / 100.0
    out = pd.DataFrame(index=P["c"].index, columns=P["c"].columns, dtype=float)
    for s in out.columns:
        y = P["r_on"][s] ** 2
        x = (v ** 2 / 252).shift(1)                        # VIX at the previous close vs the next night
        col = pd.Series(np.nan, index=out.index)
        for k, rd in enumerate(sg.REFITS):
            rd = pd.Timestamp(rd)
            tr = (y.index < rd) & y.notna() & x.notna()
            if tr.sum() < 750:
                continue
            k2 = y[tr].sum() / x[tr].sum()
            end = pd.Timestamp(sg.REFITS[k + 1]) if k + 1 < len(sg.REFITS) else pd.Timestamp("2100-01-01")
            use = (out.index >= rd) & (out.index < end)
            col[use] = np.sqrt(k2 * v[use] ** 2 / 252)     # row d: VIX at close d -> night d+1
        out[s] = col
    return out


def qlike(sig, r):
    s2 = sig ** 2
    return np.log(s2) + r ** 2 / s2


def stress_gaps(P, sma_n=200):
    """Point-in-time worst overnight gap seen so far on nights the index was above its 200-day average."""
    up = (P["c"] > core.sma(P["c"], sma_n)).shift(1)
    g = P["r_on"].where(up.astype(bool))
    return (-g).cummax().ffill().shift(1).clip(lower=0.02)   # known before the night; floor 2%


def gross_cap(L, G, limit=STRESS_LIMIT, w=1 / 3):
    """Scale all legs down so the stress loss sum_i w * L_i * G_i <= limit."""
    loss = (L.fillna(0) * w * G).sum(axis=1)
    k = (limit / loss).clip(upper=1.0).fillna(1.0)
    return L.mul(k, axis=0)


def main():
    P = core.panel(("SPY", "QQQ", "IWM"))
    trend = sg.sma_trend(P, 200)
    r0, ff0 = core.sleeve_returns(P, core.spec_exposure(P))
    vols = {"sd20": sg.vol_sd(P, 20), "ewma94": sg.vol_ewma(P, 0.94), "ewma97": sg.vol_ewma(P, 0.97)}
    vg, gparams = sg.vol_garch(P)
    vols["garch11"] = vg
    vols["vix_reg"] = vix_reg_vol(P)
    print("GARCH params (2008, 2026 refits):", {k: tuple(round(x, 4) for x in v) for k, v in gparams.items()
                                                 if k[1] in ("2008", "2026")})

    # ---- forecast accuracy, OOS 2008+
    print("\nQLIKE (lower is better) of the next-night sigma forecast, OOS 2008-2026; DM t vs sd20 (neg = better)")
    acc = {}
    for s in P["c"].columns:
        r_next = P["r_on"][s].shift(-1)
        base = None
        for k, v in vols.items():
            q = qlike(v[s], r_next)
            q = q[(q.index >= evalx.OOS) & q.notna() & np.isfinite(q)]
            if k == "sd20":
                base = q
            d = (q - base).dropna()
            dm = d.mean() / (d.std() / math.sqrt(len(d))) if len(d) > 10 and d.std() > 0 else 0.0
            acc[(s, k)] = (float(q.mean()), float(dm))
        print(f"  {s}: " + "  ".join(f"{k} {acc[(s, k)][0]:+.3f} (DM {acc[(s, k)][1]:+.1f})" for k in vols))

    rows, variants = [], {}
    def run(name, L):
        r, ff = core.sleeve_returns(P, L)
        variants[name] = r
        o = evalx.evaluate(name, r, r0, ff)
        pre = (r.index >= "2003-01-01") & (r.index < "2008-01-01")
        o["pre_sr"] = core.stats(r[pre])["sharpe"]
        G = stress_gaps(P)
        o["stress_worst"] = float(-(ff * 3 * G).sum(axis=1)[ff.index >= evalx.OOS].max())
        rows.append(o)
        return r, ff

    for k, v in vols.items():
        if k == "sd20":
            continue
        run(f"vol={k}", core.spec_exposure(P, vol=v))

    # inverse variance (Moreira-Muir): L = min(3, kappa / sigma^2); kappa calibrated on 2001-2007 to match the
    # spec's average exposure there
    sd = vols["sd20"]
    L_spec = core.spec_exposure(P)
    pre = (L_spec.index >= "2001-03-15") & (L_spec.index < "2008-01-01")
    kappas = np.linspace(1e-5, 4e-4, 400)
    target_avg = L_spec[pre].fillna(0).values.mean()
    best = min(kappas, key=lambda kk: abs(((kk / sd ** 2).clip(upper=3) * trend).shift(1)[pre].fillna(0).values.mean()
                                          - target_avg))
    run("inverse_variance (kappa from 2001-07)", ((best / sd ** 2).clip(upper=3) * trend).shift(1))
    print(f"\ninverse-variance kappa {best:.2e} (avg L pre {target_avg:.2f})")

    # fractional Kelly with Bayesian shrinkage of the mean (trend-on nights, expanding window, prior N(0, tau^2))
    for tau_bp, frac in ((3.0, 0.25), (3.0, 0.5), (10.0, 0.25)):
        tau = tau_bp / 1e4
        Lk = pd.DataFrame(index=P["c"].index, columns=P["c"].columns, dtype=float)
        for s in P["c"].columns:
            f = sg.own(s)
            up = (f["c"] > core.sma(f["c"], 200)).shift(1).astype(bool)
            x = f["r_on"].where(up)
            n = x.notna().cumsum(); m = x.expanding().mean(); v2 = x.expanding().var()
            se2 = v2 / n
            mu_post = (m / se2) / (1 / se2 + 1 / tau ** 2)          # prior mean 0
            mu_post = mu_post.reindex(P["c"].index)                 # known at close t (uses nights <= t)
            Lk[s] = (frac * mu_post / sd[s] ** 2).clip(lower=0, upper=3) * trend[s]
        run(f"kelly frac {frac} tau {tau_bp:.0f}bp", Lk.shift(1))

    # gross cap for the worst-night constraint, on the spec and on the TSMOM ensemble
    G = stress_gaps(P)
    Lc = gross_cap(L_spec, G)
    run("spec + gross cap (stress <= $100)", Lc)
    tm = sg.tsmom(P, mode="majority")
    Lt = core.spec_exposure(P, trend=tm)
    run("tsmom_majority + gross cap", gross_cap(Lt, G))
    for cap in (1.5, 1.75, 2.0, 2.25, 2.5):
        run(f"spec cap {cap} per leg", core.spec_exposure(P, cap=cap))
    for tgt in (0.010, 0.0125):
        run(f"spec target {tgt:.2%} cap 3", core.spec_exposure(P, target=tgt))

    base = evalx.evaluate("spec (benchmark)", r0, r0, ff0, boot=False)
    base["stress_worst"] = float(-(ff0 * 3 * G).sum(axis=1)[ff0.index >= evalx.OOS].max())
    pre = (r0.index >= "2003-01-01") & (r0.index < "2008-01-01")
    base["pre_sr"] = core.stats(r0[pre])["sharpe"]
    print("\nSizing variants, OOS 2008-01 -> 2026-11, realistic costs; stress = worst sleeve loss if every leg took "
          "its point-in-time worst trend-on gap")
    print(evalx.show(base) + f" | stress {base['stress_worst']:.1%} | pre03-07 SR {base['pre_sr']:+.2f}")
    for o in sorted(rows, key=lambda o: -o["sharpe"]):
        print(evalx.show(o) + f" | stress {o['stress_worst']:.1%} | pre03-07 SR {o['pre_sr']:+.2f}")
    print("current stress gaps (worst trend-on overnight gap so far):", (100 * G.iloc[-1]).round(2).to_dict())
    evalx.save_variants("sizing", variants)
    json.dump({"rows": rows, "base": base, "qlike": {f"{a}|{b}": v for (a, b), v in acc.items()},
               "kappa": best, "stress_now": G.iloc[-1].to_dict()},
              open(evalx.STORE / "exp_sizing.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
