"""Holdout for the stock gap rules that need only daily bars (9:30 open -> close), Nov 2016 - Nov 2026.

The hourly sample starts Dec 21 2023; everything before it never entered any selection here, so 2017-2023 is a
7-year holdout. Daily bars (Yahoo, cached by another session in trader/data/1d/) for the same 1,651-stock universe.
Point in time: prior close >= $5, 14-day mean volume (days t-14..t-1) >= 1M shares, |gap| < 60%.
Rules (long only, whole shares, $1,000, 10 bp; 20 bp sensitivity since opening spreads of gappers are wider):
  GD-930  gap <= -2%: buy the K largest gap-downs at the 9:30 open, sell at the close.
Diagnostics: gap buckets (9:30 -> close), daily EW means raw and in excess of the universe, NW t, by period.
Survivorship: the universe is today's listing; stocks that gapped down and later delisted are missing, which
probably flatters a gap-down-reversal rule.
Run: python3 tools/research/intraday_hourly/gaps_daily_holdout.py [--log]
"""
import sys, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from sim import topk
from common import evaluate, fmt, nw_t, log_exp, regime_below, OUT, CODE

SPLIT = pd.Timestamp("2023-12-21").date()
ETFS = {"SPY", "QQQ", "IWM", "SMH", "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "TNA", "TZA", "FAS", "FAZ"}
GB = [(-1, -0.08, "<=-8%"), (-0.08, -0.04, "-8..-4%"), (-0.04, -0.02, "-4..-2%"), (-0.02, -0.005, "-2..-0.5%"),
      (-0.005, 0.005, "flat"), (0.005, 0.02, "0.5..2%"), (0.02, 0.04, "2..4%"), (0.04, 0.08, "4..8%"), (0.08, 9, ">=8%")]


def build():
    syms = [x["symbol"] for x in json.load(open("trader/data/superset.json")) if x["symbol"] not in ETFS]
    spy = frame("SPY", "1d"); spy = spy[~spy.index.duplicated()]
    cal = pd.Index(sorted(d for d in spy.date.unique() if d >= pd.Timestamp("2016-11-21").date()))
    T = len(cal)
    A = {k: np.full((len(syms), T), np.nan) for k in ("o", "h", "l", "c", "v")}
    for n, s in enumerate(syms):
        d = frame(s, "1d")
        if d is None or len(d) < 30:
            continue
        d = d[~d.index.duplicated()]
        i = cal.get_indexer(d.date)
        ok = i >= 0
        for k in A:
            A[k][n, i[ok]] = d[k].to_numpy()[ok]
    return syms, cal, A


def main(do_log=False):
    t0 = time.time()
    syms, cal, A = build()
    o, h, l, c, v = A["o"], A["h"], A["l"], A["c"], A["v"]
    N, T = c.shape
    pC = np.concatenate([np.full((N, 1), np.nan), c[:, :-1]], 1)
    adv = pd.DataFrame(v.T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    gap = o / pC - 1
    y = c / o - 1
    U = (pC >= 5) & (adv >= 1e6) & np.isfinite(gap) & np.isfinite(y) & (np.abs(gap) < 0.6) & (o > 0) & (np.abs(y) < 0.6)
    U[:, :20] = False
    dates = np.array(list(cal))
    print(f"daily panel {N} stocks x {T} days ({cal[0]}..{cal[-1]}), universe stock-days {int(U.sum())}, "
          f"median/day {int(np.median(U.sum(0)[20:]))}, {time.time()-t0:.0f}s")
    ym = np.nanmean(np.where(U, y, np.nan), axis=0)
    pre = np.array([d < SPLIT for d in dates])
    rows = []
    for lo, hi, gl in GB:
        m = U & (gap > lo) & (gap <= hi)
        for per, pm in (("2017-2023 holdout", pre), ("2024-2026", ~pre)):
            raw, exc = [], []
            for t in range(20, T):
                if pm[t] and m[:, t].sum() >= 3:
                    raw.append(y[m[:, t], t].mean()); exc.append(y[m[:, t], t].mean() - ym[t])
            raw, exc = pd.Series(raw), pd.Series(exc)
            rows.append({"gap": gl, "period": per, "days": len(raw), "raw_bp": 1e4 * raw.mean(), "raw_nw_t": nw_t(raw),
                         "net10_bp": 1e4 * (raw.mean() - 0.001), "exc_bp": 1e4 * exc.mean(), "exc_nw_t": nw_t(exc)})
    G = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print("\nGap buckets, 9:30 open -> close, daily EW means (raw = long before costs):")
    print(G.round(2).to_string(index=False))
    below = regime_below(dates, "SPY")
    res = []
    for K in (1, 3, 5, 10):
        for cost in (0.0010, 0.0020):
            sc = np.where(U & (gap <= -0.02), -gap, -np.inf)
            w, fr, npos = topk(sc, U & (gap <= -0.02), o, c, dates, K, cost=cost)
            w = w.iloc[20:]
            for per, pm in (("holdout 2017-2023", pre[20:]), ("2024-2026", ~pre[20:])):
                s = evaluate(w[pm], f"GD-930 K={K} cost {cost*1e4:.0f}bp {per}", 1, 0.25)
                wb = w[pm & below[20:]]
                s.update({"K": K, "cost_bp": cost * 1e4, "period": per, "below200_bp": 1e4 * wb.mean(),
                          "below200_t": wb.mean() / wb.std() * math.sqrt(len(wb)) if len(wb) > 2 and wb.std() > 0 else 0,
                          "below200_n": int(len(wb))})
                res.append(s)
                print(fmt(s), f"| SPY<200d {s['below200_bp']:+.1f}bp t {s['below200_t']:+.2f} (n {s['below200_n']})")
    R = pd.DataFrame(res)
    G.to_csv(OUT / "gaps_daily_buckets.csv", index=False); R.to_csv(OUT / "gaps_daily_gd930.csv", index=False)
    if do_log:
        h = R[(R.period == "holdout 2017-2023") & (R.cost_bp == 10)]
        log_exp(family="gaps_intraday", name="Holdout 2017-2023 (daily bars): buy the largest stock gap-downs at the open, sell at the close",
                hypothesis="Large overnight gap-downs in liquid stocks partially reverse intraday (overnight-intraday reversal)",
                data=f"Yahoo daily, {N} stocks of the superset, {cal[0]}..{cal[-1]}; holdout before 2023-12-21",
                method="gap <= -2%, top-K by gap size, K in {1,3,5,10}, whole shares $1000, 10/20 bp; gap-bucket diagnostics",
                variants=8 + len(GB) * 2,
                metrics={**{f"K{int(r.K)}_holdout_bp": float(r.mean_bp) for r in h.itertuples()},
                         **{f"K{int(r.K)}_holdout_t": float(r.t) for r in h.itertuples()}},
                verdict="rejected", notes="as an every-day rule: K=5/10 holdout t 1.5, ~0 at 20 bp. Gap buckets reverse monotonically "
                      "(gap-downs up, gap-ups down) in both periods. The SPY<200d split is strong (K=5 t 3.1 in the "
                      "holdout) -> see the gd_regime record. Survivorship (today's listings) likely flatters gap-down reversal.",
                code=CODE + "gaps_daily_holdout.py")
    return G, R


if __name__ == "__main__":
    main("--log" in sys.argv)
