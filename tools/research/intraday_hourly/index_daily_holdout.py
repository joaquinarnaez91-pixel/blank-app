"""Holdout check for the index-level day trades on DAILY bars (SPY from 1993, QQQ from 1999, IWM from 2000, to Nov 2026).

The hourly sample (Dec 2023 on) was used to look at many index rules (index_day.py). Rules that use only the
9:30 open, the daily high/low and the close can be re-run on daily bars back to 1993; everything before Dec 21 2023 never
entered any selection, so it is a genuine holdout. 3x funds are approximated from the index: a fund bought at the
open and sold at price P returns (1 + 3*(P/prevC - 1)) / (1 + 3*(O/prevC - 1)) - 1 (daily reset, no fees), minus 5 bp.
Rules re-tested (as selected from the hourly sample, plus their 9:30 variants):
  G1  fade |gap| > 1.0% (gap down -> bull fund, gap up -> bear fund), exit at the close
  G2  gap < -1.0% -> bull fund, exit at the close           G3  same, exit at the gap fill (prior close) or close
  G4  fade |gap| > 0.5%, exit at the close
  R1  below the 200-day & prior day down -> bull fund 9:30 -> close
  R2  below the 200-day -> bull fund 9:30 -> close
Run: python3 tools/research/intraday_hourly/index_daily_holdout.py [--log]
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from common import evaluate, fmt, log_exp, CODE

COST = 0.0005
SPLIT = pd.Timestamp("2023-12-21").date()


def fund(P, O, pC, lev):
    return (1 + lev * (P / pC - 1)) / (1 + lev * (O / pC - 1)) - 1


def series(sym):
    d = frame(sym, "1d")
    d = d[~d.index.duplicated()].set_index("date")
    d = d[d.v > 0]
    pC = d.c.shift(1)
    below = (d.c < d.c.rolling(200).mean()).shift(1).fillna(False).astype(bool)
    gap = d.o / pC - 1
    bull_oc = fund(d.c, d.o, pC, 3) - COST
    bear_oc = fund(d.c, d.o, pC, -3) - COST
    bull_fill = np.where(d.h >= pC, fund(pC, d.o, pC, 3), fund(d.c, d.o, pC, 3)) - COST     # gap-down fill
    prev_down = (d.c.shift(1) < d.c.shift(2))
    R = {}
    R["G1 fade |gap|>1%, close"] = np.where(gap < -0.01, bull_oc, np.where(gap > 0.01, bear_oc, 0.0))
    R["G2 gap<-1% bull, close"] = np.where(gap < -0.01, bull_oc, 0.0)
    R["G3 gap<-1% bull, fill/close"] = np.where(gap < -0.01, bull_fill, 0.0)
    R["G4 fade |gap|>0.5%, close"] = np.where(gap < -0.005, bull_oc, np.where(gap > 0.005, bear_oc, 0.0))
    R["R1 below200 & prev down, bull 9:30"] = np.where(below & prev_down, bull_oc, 0.0)
    R["R2 below200, bull 9:30"] = np.where(below, bull_oc, 0.0)
    out = pd.DataFrame(R, index=d.index).iloc[201:]           # after the 200-day warm-up
    return out.fillna(0.0), below.iloc[201:]


def main(do_log=False):
    allres = {}
    for sym in ("SPY", "QQQ", "IWM"):
        R, below = series(sym)
        for k in R:
            allres[(k, sym)] = R[k]
    names = sorted(set(k for k, _ in allres))
    rows = []
    for k in names:
        ew = pd.concat([allres[(k, s)] for s in ("SPY", "QQQ", "IWM")], axis=1).fillna(0).mean(axis=1)
        allres[(k, "EW3")] = ew
    for k in names:
        for s in ("SPY", "QQQ", "IWM", "EW3"):
            r = allres[(k, s)]
            pre, post = r[r.index < SPLIT], r[r.index >= SPLIT]
            a = evaluate(pre, f"{k} [{s}] holdout to 2023-12-20", 1, 0.25)
            b = evaluate(post, f"{k} [{s}] hourly-era 2023-2026", 1, 0.25)
            rows.append({"rule": k, "idx": s, "pre_days": a["days"], "pre_act": a["active"], "pre_bp": a["mean_bp"],
                         "pre_t": a["t"], "pre_SR": a["sharpe"], "post_bp": b["mean_bp"], "post_t": b["t"],
                         "post_SR": b["sharpe"]})
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print(D.round(2).to_string(index=False))
    if do_log:
        g = D[(D.rule.str.startswith("G1")) & (D.idx == "IWM")].iloc[0]
        log_exp(family="index_daytime_hourly", name="Holdout (daily bars to Dec 2023) of the best index day rules",
                hypothesis="Index gap fades / rebound rules picked on the 2024-26 hourly sample hold in 2017-2023",
                data="Yahoo daily SPY 1993+/QQQ 1999+/IWM 2000+ to Nov 2026; holdout = before 2023-12-21",
                method="3x fund returns approximated from the index (daily reset), 5 bp; same rules, no refitting",
                variants=len(names) * 4,
                metrics={"IWM_G1_pre_bp": float(g.pre_bp), "IWM_G1_pre_t": float(g.pre_t), "IWM_G1_post_t": float(g.post_t),
                         **{f"{r.rule[:2]}_{r.idx}_pre_t": float(r.pre_t) for r in D.itertuples() if r.idx in ("IWM", "EW3")}},
                verdict="rejected",
                notes="IWM large-gap fade (best hourly-sample index rule, t 2.3) fails the 2000-2023 holdout (t -0.6); gap-down fill looked strong only with touch fills (see gapfill_long record). Report tools/research/reports/intraday_hourly.md", code=CODE + "index_daily_holdout.py")
    return D


if __name__ == "__main__":
    main("--log" in sys.argv)
