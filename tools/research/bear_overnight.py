"""Bear-regime overnight test (Nov 25, 2026). Is there a long-only close -> next-open edge while the index is
BELOW its 200-day average, when the live overnight model (tools/overnight.py) sits in cash?

  python3 tools/research/bear_overnight.py            # prints the tables, writes tools/research/bear/bear_overnight.csv
  python3 tools/research/bear_overnight.py --log      # also appends the registry line (family "bear_overnight")

Pre-declared hypotheses (fixed parameters from the literature, nothing fitted; written before any result was seen).
Every rule trades only legs whose index (SPY->UPRO/SPXU, QQQ->TQQQ/SQQQ, IWM->TNA/TZA) closed BELOW its 200-day SMA
on day t-1, buys the fund at ~15:58 on day t-1 and sells at the 9:30 open on day t.
  H1  long the 3x BULL fund after a short-term washout (Connors-style mean reversion):
      H1a below 200d, no further condition (baseline)     H1b 2+ consecutive down closes
      H1c 3+ consecutive down closes                       H1d close < lower Bollinger band (20 d, 2 sd)
      H1e RSI(2) < 10
  H2  long the 3x INVERSE fund:
      H2a below 200d, no further condition                 H2b after an up close (fade the bounce)
      H2c after 2+ consecutive down closes (momentum)
  H3  long the 3x BULL fund, conditioned on the volatility regime (thresholds from expanding windows, point in time):
      H3a VIX in the top third of its below-200d history    H3b VIX in the bottom third
      H3c VIX term structure inverted (VIX > VIX3M)
Each rule is run as the equal-weight 3-index sleeve AND on each index alone: 11 rules x 4 = 44 variants, plus one
walk-forward selector (each Jan 1 pick the rule with the best Sharpe on all earlier nights, trade it for the year).

Sizing (same as the live spec): capital $450 (half the sleeve), one third per leg, index exposure
L = min(2, 0.015 / sigma20) with sigma20 = std of the last 20 overnight returns, fund dollars = capital/3 * L/3.
Stress guard (point in time): every held leg gaps against us by max(worst adverse gap seen so far while below the
200-day, 5 x sigma20); the 3x loss must be <= $100, else all legs scale down. Fractional dollars are used for the
statistics; whole-share rounding is checked for the best rule.

Returns (tools/research/overnight/core.py conventions): bull fund = 3 r_on - carry (2.4 x T-bill), inverse fund
= -3 r_on - 1 bp/yr-ish fee (we charge 0.95%/yr expense; financing roughly nets out for inverse funds); r_on
includes the ETF dividend (the inverse fund owes it). 4 bp round trip on fund dollars. Real-fund check: actual
UPRO/TQQQ/TNA (2010-) and SPXU/SQQQ/TZA (2016-) opens and closes.
Out of sample = 2008-01-01 onward (the overnight program's OOS window); 2000-2007 shown only as a pre-sample check
and used as burn-in for the walk-forward selector and the expanding VIX thresholds.
"""
import sys, math, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
sys.path.insert(0, "tools/research/overnight")
import core  # noqa: E402
from stats import deflated_sharpe  # noqa: E402

IDX = ("SPY", "QQQ", "IWM")
BULL = {"SPY": "UPRO", "QQQ": "TQQQ", "IWM": "TNA"}
BEAR = {"SPY": "SPXU", "QQQ": "SQQQ", "IWM": "TZA"}
CAPITAL, MAX_LOSS, TARGET, CAP = 450.0, 100.0, 0.015, 2.0
OOS = pd.Timestamp("2008-01-01")
INV_FEE = 0.0095
PRIOR_TRIALS = 30   # family "bear_inverse" (Nov 20) tested the same question for inverse funds


def rsi2(c):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def down_streak(c):
    dn = (c.diff() < 0).astype(int)
    return dn.groupby((dn == 0).cumsum()).cumsum()


def build():
    P = core.panel(IDX, start="2000-06-01")
    c, r_on = P["c"], P["r_on"]
    idx = c.index
    vix = core.daily("^VIX")["c"].reindex(idx)
    vix3 = core.daily("^VIX3M")["c"].reindex(idx)
    below = c < core.sma(c, 200)                          # at close t-1 (shifted below)
    sig20 = r_on.rolling(20, min_periods=20).std()
    streak = c.apply(down_streak)
    up = c.diff() > 0
    bb = c < core.sma(c, 20) - 2 * c.rolling(20).std()
    rsi = c.apply(rsi2)
    # VIX terciles among below-200d days, expanding (point in time): threshold uses only earlier days
    anyb = below.any(axis=1)
    vb = vix.where(anyb)
    hi_thr = vb.expanding(min_periods=250).quantile(2 / 3).shift(1)
    lo_thr = vb.expanding(min_periods=250).quantile(1 / 3).shift(1)
    vhi = pd.DataFrame({s: vix > hi_thr for s in IDX})
    vlo = pd.DataFrame({s: vix < lo_thr for s in IDX})
    inv = pd.DataFrame({s: (vix / vix3) > 1 for s in IDX})
    T = pd.DataFrame(True, index=idx, columns=list(IDX))
    rules = {
        "H1a bull | below200": ("bull", T),
        "H1b bull | 2+ down closes": ("bull", streak >= 2),
        "H1c bull | 3+ down closes": ("bull", streak >= 3),
        "H1d bull | close < lower BB(20,2)": ("bull", bb),
        "H1e bull | RSI(2) < 10": ("bull", rsi < 10),
        "H2a inverse | below200": ("bear", T),
        "H2b inverse | after up close": ("bear", up),
        "H2c inverse | 2+ down closes": ("bear", streak >= 2),
        "H3a bull | VIX top third": ("bull", vhi),
        "H3b bull | VIX bottom third": ("bull", vlo),
        "H3c bull | VIX > VIX3M": ("bull", inv),
    }
    # point-in-time worst adverse gap while below the 200-day (for the stress guard), known at close t-1
    bel_on = below.shift(1).fillna(False).astype(bool)   # night t was a below-200d night
    worst_dn = (-r_on.where(bel_on)).expanding().max().shift(1).fillna(0.05).clip(lower=0.05)
    worst_up = (r_on.where(bel_on)).expanding().max().shift(1).fillna(0.05).clip(lower=0.05)
    return P, below, sig20, rules, worst_dn, worst_up


def run(P, below, sig20, side, cond, worst_dn, worst_up, legs=IDX, real=None):
    """Nightly sleeve return (fraction of $450) and $ P&L for one rule. Signals at close t-1 -> night t."""
    legs = list(legs)
    on = (below & cond.fillna(False))[legs].shift(1).fillna(False).astype(bool)
    L = (TARGET / sig20[legs]).clip(upper=CAP).shift(1) * on
    frac = L.fillna(0.0) / len(legs) / 3.0                  # fraction of capital in each 3x fund
    sgn = 1.0 if side == "bull" else -1.0
    gap = pd.concat([(worst_dn if side == "bull" else worst_up)[legs],
                     5 * sig20[legs].shift(1)], axis=1, keys=["w", "s"])
    g = np.maximum(gap["w"], gap["s"])
    stress = (CAPITAL * frac * 3 * g).sum(axis=1)
    scale = (MAX_LOSS / stress).clip(upper=1.0).fillna(1.0)
    frac = frac.mul(scale, axis=0)
    if real is not None:
        fr = real[legs]
    elif side == "bull":
        fr = (3 * P["r_on"][legs]).sub(core.carry(P), axis=0)
    else:
        fr = (-3 * P["r_on"][legs]).sub(INV_FEE * P["fwd_days"] / 365, axis=0)
    r = (frac * fr.fillna(0.0)).sum(axis=1) - core.COST_RT * frac.sum(axis=1)
    return r, frac, stress * scale


def st(r, active_mask):
    r = r.dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    act = r[active_mask.reindex(r.index).fillna(False)]
    h = n // 2
    m1, m2 = r.iloc[:h].mean(), r.iloc[h:].mean()
    return {"nights": n, "active": len(act), "bp_active": 1e4 * act.mean() if len(act) else np.nan,
            "usd_active": CAPITAL * act.mean() if len(act) else np.nan,
            "t": mu / sd * math.sqrt(n) if sd > 0 else np.nan,
            "sharpe": mu / sd * math.sqrt(252) if sd > 0 else np.nan,
            "h1_bp": 1e4 * m1, "h2_bp": 1e4 * m2, "worst_usd": CAPITAL * r.min(),
            "skew": float(((r - mu) ** 3).mean() / sd ** 3) if sd > 0 else 0.0,
            "kurt": float(((r - mu) ** 4).mean() / sd ** 4) if sd > 0 else 3.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    P, below, sig20, rules, wdn, wup = build()
    rows, series = [], {}
    for name, (side, cond) in rules.items():
        for legs in (IDX, ("SPY",), ("QQQ",), ("IWM",)):
            r, frac, sl = run(P, below, sig20, side, cond, wdn, wup, legs)
            tag = f"{name} [{'3-index' if len(legs) == 3 else legs[0]}]"
            series[tag] = (r, frac.sum(axis=1) > 0)
            for win, lo, hi in (("OOS", OOS, None), ("pre", pd.Timestamp("2000-06-01"), OOS)):
                rr = r[(r.index >= lo) & ((r.index < hi) if hi is not None else True)]
                s = st(rr, frac.sum(axis=1) > 0)
                rows.append({"rule": name, "legs": tag.split("[")[1][:-1], "window": win, **s,
                             "max_stress_usd": float(sl[rr.index].max())})
    df = pd.DataFrame(rows)
    n_trials = len(series)
    oos = df[df.window == "OOS"].copy()
    srv = float(np.nanvar(oos.sharpe.values))
    pt = n_trials + PRIOR_TRIALS
    oos["dsr"] = [deflated_sharpe(s.sharpe, s.nights, pt, srv, s.skew, s.kurt)[0] if s.sharpe == s.sharpe else np.nan
                  for s in oos.itertuples()]
    df = df.merge(oos[["rule", "legs", "window", "dsr"]], how="left", on=["rule", "legs", "window"])

    # walk-forward selector over the 11 three-index rules: each Jan 1 pick the best prior Sharpe (>= 500 nights)
    three = {k: v for k, v in series.items() if "3-index" in k}
    M = pd.DataFrame({k: v[0] for k, v in three.items()})
    wf, picks = [], {}
    for y in range(2008, M.index[-1].year + 1):
        hist = M[M.index < pd.Timestamp(f"{y}-01-01")]
        sr = hist.mean() / hist.std(ddof=1)
        best = sr.idxmax()
        picks[y] = best.split(" |")[0]
        wf.append(M.loc[M.index.year == y, best])
    wf = pd.concat(wf)
    wfs = st(wf, wf != 0)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cols = ["rule", "legs", "active", "bp_active", "usd_active", "t", "sharpe", "h1_bp", "h2_bp", "worst_usd",
            "max_stress_usd", "dsr"]
    print(f"OUT OF SAMPLE 2008-01 -> {P['c'].index[-1].date()}, $450 capital, net of 4 bp + carry. "
          f"t on the full nightly series (0 when flat). DSR trials = {n_trials} + {PRIOR_TRIALS} prior.")
    print(df[df.window == "OOS"][cols].round(2).to_string(index=False))
    print("\nPRE-SAMPLE 2000-06 -> 2007-12 (check only)")
    print(df[(df.window == "pre") & (df.legs == "3-index")][cols[:-1]].round(2).to_string(index=False))
    print(f"\nWalk-forward selector (11 three-index rules): active {wfs['active']} bp/active {wfs['bp_active']:+.1f} "
          f"t {wfs['t']:+.2f} SR {wfs['sharpe']:+.2f} halves {wfs['h1_bp']:+.2f}/{wfs['h2_bp']:+.2f} bp "
          f"worst ${wfs['worst_usd']:.0f}\n  picks: {picks}")

    # best OOS three-index rule by t: real funds and whole shares
    o3 = df[(df.window == "OOS") & (df.legs == "3-index")].sort_values("t", ascending=False)
    best = o3.iloc[0]
    side, cond = rules[best.rule]
    fund = BULL if side == "bull" else BEAR
    real = pd.DataFrame({s: core.index_frame(fund[s])["r_on"].reindex(P["c"].index) for s in IDX})
    start = real.dropna().index[0]
    rr, fr, _ = run(P, below, sig20, side, cond, wdn, wup, IDX, real=real)
    rp, fp, _ = run(P, below, sig20, side, cond, wdn, wup, IDX)
    m = rr.index >= start
    sr_, sp_ = st(rr[m], fr.sum(axis=1) > 0), st(rp[m], fp.sum(axis=1) > 0)
    print(f"\nBest OOS 3-index rule: {best.rule}. Real funds from {start.date()}: bp/active {sr_['bp_active']:+.1f} "
          f"t {sr_['t']:+.2f} worst ${sr_['worst_usd']:.0f} | proxy same nights: bp/active {sp_['bp_active']:+.1f} "
          f"t {sp_['t']:+.2f}")
    # whole shares at the real fund's previous close (proxy for the 15:58 ask)
    px = pd.DataFrame({s: core.daily(fund[s])["c"].reindex(P["c"].index).shift(1) for s in IDX})
    sh = np.floor(CAPITAL * fr / px).where(px > 0, 0)
    pnl = (sh * px * real.fillna(0.0)).sum(axis=1) - core.COST_RT * (sh * px).sum(axis=1)
    pnl = pnl[m]
    ws = {"mean_usd_active": float(pnl[pnl != 0].mean()), "t": float(pnl.mean() / pnl.std() * math.sqrt(len(pnl))),
          "worst_usd": float(pnl.min()), "nights_with_shares": int((pnl != 0).sum())}
    print(f"  whole shares: {ws}")

    df.to_csv("tools/research/bear/bear_overnight.csv", index=False)
    passes = o3[(o3.t > 3) & (o3.dsr > 0.9) & (o3.h1_bp > 0) & (o3.h2_bp > 0) & (o3.worst_usd > -MAX_LOSS)]
    allp = df[(df.window == "OOS") & (df.t > 3) & (df.dsr > 0.9) & (df.h1_bp > 0) & (df.h2_bp > 0)
              & (df.worst_usd > -MAX_LOSS)]
    print(f"\nvariants passing every hurdle: {len(allp)}")
    if a.log:
        from registry import log
        log(family="bear_overnight",
            name="Long-only overnight below the 200-day: washout bull 3x, inverse 3x, VIX regimes",
            hypothesis="While SPY/QQQ/IWM are below their 200-day SMA, some pre-declared long-only close->open hold "
                       "(3x bull after a washout, 3x inverse, or 3x bull in a VIX regime) earns a positive net edge",
            data=f"Yahoo daily SPY/QQQ/IWM 2000-06..{P['c'].index[-1].date()}, ^VIX, ^VIX3M, ^IRX; real UPRO/TQQQ/TNA "
                 "2010-, SPXU/SQQQ/TZA 2016-; OOS 2008-01 on",
            method="11 fixed-parameter rules (H1a-e washout bull, H2a-c inverse, H3a-c VIX regime) x (3-index sleeve "
                   "+ each index) = 44 variants + walk-forward selector; live sizing L=min(2,0.015/sigma20), $450, "
                   "PIT stress guard <= $100; proxy 3x r_on - carry, 4 bp; t on the nightly series; DSR with "
                   f"{pt} trials; halves of OOS",
            variants=n_trials + 1,
            metrics={"best_rule": best.rule, "best_bp_active": round(best.bp_active, 1),
                     "best_usd_active": round(best.usd_active, 2), "best_t": round(best.t, 2),
                     "best_sharpe": round(best.sharpe, 2), "best_dsr": round(best.dsr, 3),
                     "best_h1_bp": round(best.h1_bp, 2), "best_h2_bp": round(best.h2_bp, 2),
                     "best_worst_usd": round(best.worst_usd, 1), "best_active_nights": int(best.active),
                     "best_real_bp_active": round(sr_["bp_active"], 1), "best_real_t": round(sr_["t"], 2),
                     "best_wholeshare_worst_usd": round(ws["worst_usd"], 1),
                     "wf_selector_bp_active": round(wfs["bp_active"], 1), "wf_selector_t": round(wfs["t"], 2),
                     "inverse_best_t": round(float(df[(df.window == "OOS") & df.rule.str.startswith("H2")].t.max()), 2),
                     "n_pass": int(len(allp))},
            verdict="accepted" if len(passes) else "rejected",
            notes="see tools/research/reports/bear_overnight.md",
            code="tools/research/bear_overnight.py", author="agent-bear-overnight")
        print("logged to registry")


if __name__ == "__main__":
    main()
