"""Deep dive: buy the largest stock gap-downs at the open, sell at the close, ONLY on days the overnight model is in
cash (SPY's prior close below its 200-day SMA). Daily bars Nov 2016 - Nov 2026, same 1,651-stock universe.

Prior: returns to liquidity provision / short-term reversal are much higher in turbulent markets (Nagel 2012,
"Evaporating liquidity", RFS); overnight gaps partially reverse intraday (Berkman, Koch, Tuttle & Zhang 2012;
Lou, Polk & Skouras 2019 for the overnight/intraday tug of war).
Grid (every cell counted): universe {base: prior close >= $5 & 14d volume >= 1M; large: prior close >= $10 & 14d
dollar volume >= $50M} x gap band {<= -2%, -2..-10%, <= -4%} x K {3, 5, 10} x regime {SPY<200d, any index<200d};
costs 10 bp and 20 bp. Report: by episode, holdout (2017-2023) vs 2024-26, trimmed means, top-day share, DSR.
Run: python3 tools/research/intraday_hourly/gd_regime.py [--log]
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from sim import topk
from common import evaluate, fmt, nw_t, log_exp, regime_below, cash_fraction, OUT, CODE
from gaps_daily_holdout import build, SPLIT

EPIS = [("2018-19", "2018-01-01", "2019-06-01"), ("2020", "2020-01-01", "2021-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023", "2023-01-01", "2024-01-01"), ("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2027-01-01")]


def main(do_log=False):
    syms, cal, A = build()
    o, h, l, c, v = A["o"], A["h"], A["l"], A["c"], A["v"]
    N, T = c.shape
    dates = np.array(list(cal))
    pC = np.concatenate([np.full((N, 1), np.nan), c[:, :-1]], 1)
    adv = pd.DataFrame(v.T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    dv = pd.DataFrame((v * c).T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    gap = o / pC - 1
    y = c / o - 1
    ok = np.isfinite(gap) & np.isfinite(y) & (np.abs(gap) < 0.6) & (o > 0) & (np.abs(y) < 0.6)
    ok[:, :20] = False
    UNI = {"base": ok & (pC >= 5) & (adv >= 1e6), "large": ok & (pC >= 10) & (dv >= 5e7)}
    BAND = {"<=-2%": (-0.6, -0.02), "-10..-2%": (-0.10, -0.02), "<=-4%": (-0.6, -0.04)}
    REG = {"SPY<200d": regime_below(dates, "SPY"), "any<200d": cash_fraction(dates) > 0}
    pre = np.array([d < SPLIT for d in dates])
    res, series = [], {}
    for un, U in UNI.items():
        for bn, (lo, hi) in BAND.items():
            el = U & (gap > lo) & (gap <= hi)
            for K in (3, 5, 10):
                for cost in (0.0010, 0.0020):
                    w, fr, npos = topk(np.where(el, -gap, -np.inf), el, o, c, dates, K, cost=cost)
                    for rn, rg in REG.items():
                        r = pd.Series(np.where(rg, w.to_numpy(), 0.0), index=dates)[20:]
                        series[(un, bn, K, cost, rn)] = r
    # trial count: every (universe, band, K, regime) cell at 10 bp (the 20 bp runs are sensitivities, not trials)
    keys10 = [k for k in series if k[3] == 0.0010]
    n_tr = len(keys10) + 16 + 34   # + the earlier unconditional GD rules and gap cells in gaps.py / gaps_daily_holdout
    full_srs = [series[k].mean() / series[k].std() * math.sqrt(252) for k in keys10]
    var = float(np.var(full_srs, ddof=1))
    print(f"{len(keys10)} regime cells at 10 bp (+50 earlier gap trials) = {n_tr} trials; SR var {var:.3f}")
    for k, r in series.items():
        un, bn, K, cost, rn = k
        rg = REG[rn][20:]
        act = r[rg]
        hold = r[pre[20:]]
        post = r[~pre[20:]]
        e = evaluate(r, f"{un} {bn} K={K} {rn} {cost*1e4:.0f}bp", n_tr, var)
        eh = evaluate(hold, "", n_tr, var); ep = evaluate(post, "", n_tr, var)
        # regime-day statistics (the days the capital is actually free)
        a_h = act[pre[20:][rg]]; a_p = act[~pre[20:][rg]]
        srt = np.sort(act.to_numpy())
        trim = srt[int(0.02 * len(srt)):int(0.98 * len(srt))].mean() if len(srt) > 50 else np.nan
        top5 = srt[-5:].sum() / srt.sum() if srt.sum() > 0 else np.nan
        row = {"universe": un, "band": bn, "K": K, "cost_bp": cost * 1e4, "regime": rn, "regime_days": int(rg.sum()),
               "full_mean_bp": e["mean_bp"], "full_t": e["t"], "full_nw_t": e["nw_t"], "full_SR": e["sharpe"], "DSR": e["dsr_p"],
               "act_mean_bp": 1e4 * act.mean(), "act_median_bp": 1e4 * act.median(), "act_trim2_bp": 1e4 * trim,
               "act_hit": (act > 0).mean(), "top5_share": top5,
               "hold_act_bp": 1e4 * a_h.mean(), "hold_t": nw_t(hold), "post_act_bp": 1e4 * a_p.mean(), "post_t": nw_t(post),
               "post_days": int(len(a_p))}
        for en, a, b in EPIS:
            m = (act.index >= pd.Timestamp(a).date()) & (act.index < pd.Timestamp(b).date())
            row[f"{en}_bp"] = 1e4 * act[m].mean() if m.sum() else np.nan
            row[f"{en}_n"] = int(m.sum())
        res.append(row)
    R = pd.DataFrame(res)
    R.to_csv(OUT / "gd_regime.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 40)
    cols = ["universe", "band", "K", "cost_bp", "regime", "regime_days", "act_mean_bp", "act_median_bp", "act_trim2_bp",
            "act_hit", "top5_share", "full_t", "full_nw_t", "full_SR", "DSR", "hold_act_bp", "hold_t", "post_act_bp", "post_t"]
    print(R[cols].round(2).to_string(index=False))
    print("\nBy episode (mean bp per regime day):")
    ec = ["universe", "band", "K", "cost_bp", "regime"] + [f"{e}_bp" for e, _, _ in EPIS] + [f"{e}_n" for e, _, _ in EPIS]
    print(R[R.cost_bp == 10][ec].round(1).to_string(index=False))
    if do_log:
        m = R[(R.cost_bp == 10)]
        b = m.loc[m.full_SR.idxmax()]
        p = R[(R.universe == "base") & (R.band == "<=-2%") & (R.K == 5) & (R.cost_bp == 10) & (R.regime == "SPY<200d")].iloc[0]
        log_exp(family="gaps_intraday", name="Gap-down reversal only when SPY < 200-day (overnight model in cash), daily bars 2017-2026",
                hypothesis="Liquidity-provision returns are higher in turbulent markets (Nagel 2012): buying the largest "
                           "stock gap-downs at the open and selling at the close pays on below-200-day days",
                data=f"Yahoo daily, {N} stocks, {cal[0]}..{cal[-1]}; holdout before 2023-12-21; regime from SPY daily closes",
                method="universe {base, large} x band {<=-2%, -10..-2%, <=-4%} x K {3,5,10} x regime {SPY<200d, any<200d}; "
                       "whole shares $1000; 10 bp (20 bp sensitivity); NW t; DSR with all gap-family trials",
                variants=len(keys10),
                metrics={"pre_spec_base_K5_act_bp": float(p.act_mean_bp), "pre_spec_full_t": float(p.full_t),
                         "pre_spec_nw_t": float(p.full_nw_t), "pre_spec_SR": float(p.full_SR), "pre_spec_DSR": float(p.DSR),
                         "pre_spec_hold_t": float(p.hold_t), "pre_spec_post_t": float(p.post_t),
                         "best_SR": float(b.full_SR), "best_t": float(b.full_t), "best_DSR": float(b.DSR)},
                verdict="promising", notes=f"all 36 cells positive on regime days, every episode positive (base); large-cap "
                      f"universe about half the effect (survivorship and/or size). best cell {b.universe} {b.band} K={b.K} {b.regime}. "
                      "Hurdle check in final_eval.py; report tools/research/reports/intraday_hourly.md",
                code=CODE + "gd_regime.py")
    return R


if __name__ == "__main__":
    main("--log" in sys.argv)
