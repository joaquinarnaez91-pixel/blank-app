"""(a) Validate the current spec and the whole family of variants tried.

1. Stationary block bootstrap (Politis-Romano) CIs for the spec's Sharpe and CAGR (mean block 5/20/60 nights)
   on three windows: the spec's own (2017-09 -> 2026-11), out of sample 2008-2026, and 2001-2026.
2. CSCV probability of backtest overfitting (Bailey et al. 2017) for the spec's own parameter family
   (SMA 100/150/200/250 x sigma window 10/20/40/60 x target 1.0/1.5/2.0% x cap 2/3 = 96 variants, 2001-2026, S=16),
   and for all variants tried in this program (2008-2026).
3. White's Reality Check and Hansen's SPA: does any variant beat the spec (vol-matched, i.e. on Sharpe)? And does
   the spec family beat cash?
4. Deflated Sharpe probability for the spec and the best candidates, with the overnight family's total trial count.
"""
import sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
sys.path.insert(0, "tools/research")
import core, evalx, validate  # noqa: E402
from registry import trials  # noqa: E402


def main():
    P = core.panel(("SPY", "QQQ", "IWM"))
    r0, ff0 = core.sleeve_returns(P, core.spec_exposure(P))
    out = {}

    # 1. bootstrap CIs
    print("1. Stationary bootstrap 95% CIs for the current spec (realistic costs)")
    wins = {"spec window 2017-09..2026-11": ("2017-09-08", None), "OOS 2008-2026": ("2008-01-01", None),
            "full 2001-03..2026-11": ("2001-03-15", None)}
    out["boot"] = {}
    for wn, (a, b) in wins.items():
        r = evalx.window(r0, a)
        for blk in (5, 20, 60):
            ci = validate.boot_ci(r.values, B=4000, b=blk, seed=blk)
            out["boot"][f"{wn}|b{blk}"] = ci
            print(f"  {wn:<30} b={blk:2d}: SR {ci['sr']:.2f} [{ci['sr_ci'][0]:.2f}, {ci['sr_ci'][2]:.2f}] "
                  f"P(SR<=0) {ci['p_sr_le_0']:.3f} | CAGR {ci['cagr']:+.1%} [{ci['cagr_ci'][0]:+.1%}, "
                  f"{ci['cagr_ci'][2]:+.1%}]")

    # 2a. CSCV on the spec's parameter family
    grid = {}
    for n, vw, tg, cap in itertools.product((100, 150, 200, 250), (10, 20, 40, 60), (0.010, 0.015, 0.020), (2.0, 3.0)):
        L = core.spec_exposure(P, sma_n=n, vol_n=vw, target=tg, cap=cap)
        grid[(n, vw, tg, cap)] = core.sleeve_returns(P, L)[0]
    G = pd.DataFrame({str(k): v for k, v in grid.items()})
    G = G[G.index >= "2001-03-15"].dropna()
    pbo = validate.cscv_pbo(G.values, S=16)
    srs = G.mean() / G.std() * np.sqrt(252)
    rank_spec = int((srs > srs[str((200, 20, 0.015, 3.0))]).sum()) + 1
    out["cscv_spec_family"] = {**pbo, "n_variants": G.shape[1], "spec_rank": rank_spec,
                               "best": srs.idxmax(), "best_sr": float(srs.max()), "spec_sr": float(srs[str((200, 20, 0.015, 3.0))])}
    print(f"\n2a. CSCV, spec parameter family (96 variants, 2001-2026, S=16, {pbo['combos']} splits): PBO {pbo['pbo']:.2f}, "
          f"median logit {pbo['median_logit']:+.2f}, P(IS-best has OOS SR<0) {pbo['p_oos_best_neg']:.2f}, "
          f"IS-best SR {pbo['mean_is_sr_best']:.2f} -> OOS {pbo['mean_oos_sr_best']:.2f}, slope {pbo['degradation_slope']:+.2f}")
    print(f"    spec (200, 20, 1.5%, 3) full-sample SR {srs[str((200, 20, 0.015, 3.0))]:.2f}, rank {rank_spec}/96; best {srs.idxmax()} SR {srs.max():.2f}")
    cap_eff = pd.Series({k: float(v) for k, v in srs.items()})
    print("    mean SR by cap:", {c: round(float(np.mean([srs[str(k)] for k in grid if k[3] == c])), 3) for c in (2.0, 3.0)},
          " by SMA:", {n: round(float(np.mean([srs[str(k)] for k in grid if k[0] == n])), 3) for n in (100, 150, 200, 250)},
          " by sigma window:", {w: round(float(np.mean([srs[str(k)] for k in grid if k[1] == w])), 3) for w in (10, 20, 40, 60)},
          " by target:", {t: round(float(np.mean([srs[str(k)] for k in grid if k[2] == t])), 3) for t in (0.01, 0.015, 0.02)})

    # 2b/3. the whole family tried in this program (proxy-based, 2008-2026)
    V = evalx.load_variants()
    fam = {}
    for g in ("signals", "calendar", "sizing", "assets"):
        for k, v in V.get(g, {}).items():
            if k in ("spec_sma200",):
                continue
            fam[f"{g}:{k}"] = v
    F = pd.DataFrame(fam)
    F = F[F.index >= evalx.OOS]
    F = F.loc[:, F.notna().mean() > 0.99].fillna(0.0)
    b0 = evalx.window(r0).reindex(F.index).fillna(0.0)
    pbo2 = validate.cscv_pbo(np.column_stack([b0.values, F.values]), S=16)
    out["cscv_my_family"] = {**pbo2, "n_variants": F.shape[1] + 1}
    print(f"\n2b. CSCV, all {F.shape[1] + 1} variants of this program (2008-2026): PBO {pbo2['pbo']:.2f}, "
          f"median logit {pbo2['median_logit']:+.2f}, IS-best SR {pbo2['mean_is_sr_best']:.2f} -> OOS {pbo2['mean_oos_sr_best']:.2f}, "
          f"P(OOS SR<0) {pbo2['p_oos_best_neg']:.2f}")
    spa = validate.spa_test(F.values, b0.values, B=2000, b=20)
    spa["best_name"] = F.columns[spa["best"]]
    out["spa_vs_spec"] = spa
    print(f"\n3a. SPA / Reality Check, H0: no variant beats the spec (vol-matched), {F.shape[1]} variants: "
          f"best {spa['best_name']}, T_SPA {spa['t_spa']:.2f}, p lower/consistent/upper {spa['p_lower']:.3f}/"
          f"{spa['p_consistent']:.3f}/{spa['p_upper']:.3f}, White RC p {spa['p_rc']:.3f}")
    # SPA for the subfamilies separately
    for g in ("signals", "calendar", "sizing", "assets"):
        cols = [c for c in F.columns if c.startswith(g + ":")]
        if not cols:
            continue
        s = validate.spa_test(F[cols].values, b0.values, B=2000, b=20)
        out[f"spa_{g}"] = {**s, "best_name": cols[s["best"]], "k": len(cols)}
        print(f"    {g:<9} ({len(cols):2d} variants): best {cols[s['best']]:<42} SPA p (consistent) {s['p_consistent']:.3f}, RC p {s['p_rc']:.3f}")
    # spec family vs cash (zero benchmark), not vol matched
    G8 = G[G.index >= "2001-03-15"]
    s0 = validate.spa_test(G8.values, np.zeros(len(G8)), B=2000, b=20, vol_match=False)
    out["spa_family_vs_cash"] = s0
    print(f"3b. Spec family (96) vs cash, 2001-2026: SPA p consistent {s0['p_consistent']:.4f}, RC p {s0['p_rc']:.4f}")

    # 4. deflated Sharpe
    n_prev = trials("overnight")
    n_total = n_prev + F.shape[1] + 96
    allsr = pd.concat([F, G.reindex(F.index)], axis=1).apply(lambda x: x.mean() / x.std() * np.sqrt(252))
    v_sr = float(allsr.var())
    out["dsr"] = {"n_trials": n_total, "sr_var": v_sr}
    print(f"\n4. Deflated Sharpe, trials {n_total} (registry overnight {n_prev} + this program {F.shape[1]} + spec grid 96), "
          f"var of trial SRs {v_sr:.3f}")
    cands = {"spec (OOS 2008-26)": b0}
    for k in ("sizing:spec cap 2.0 per leg", "assets:SMH+XLF legs, cap 2", "assets:SMH legs, cap 2",
              "signals:tsmom_majority|replace", "assets:SMH+XLF legs, cap 3"):
        if k in F:
            cands[k] = F[k]
    for k, r in cands.items():
        p, emax = validate.dsr(r.values, n_total, v_sr)
        out["dsr"][k] = {"p": p, "sr": float(r.mean() / r.std() * np.sqrt(252)), "e_max_sr": emax}
        print(f"   {k:<36} SR {r.mean() / r.std() * np.sqrt(252):.2f}  E[max SR | null] {emax:.2f}  DSR prob {p:.3f}")
    json.dump(out, open(evalx.STORE / "exp_validate.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
