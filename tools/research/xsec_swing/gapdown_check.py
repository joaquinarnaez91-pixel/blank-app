"""Survivorship sensitivity of the hourly agent's "cash-day gap-down reversal" on the daily panel.

  python3 tools/research/xsec_swing/gapdown_check.py
Rule (as specified by agent-intraday-hourly): on days when SPY closed below its 200-day average the prior day,
buy the 5 largest opening gap-downs (open / prior close - 1 <= -2%) at the open and sell at the close; 10 bp round
trip; equal weight. Same rule on different candidate sets:
  superset_raw   today's 1,651-name superset (as cached in this panel), no point-in-time filter
  superset_pit   superset names, point-in-time filter: actual close of t-1 >= $5 and 60-day mean volume >= 750k shares
  listed_pit     every name listed today (adds names now below $5 or thin), same point-in-time filter
  long_hist_pit  listed_pit restricted to names with bars since Jan 2017 (long continuous histories)
  recent_pit     listed_pit restricted to names first listed after Jan 2017 (recent listings)
  listed_top500  every listed name, point-in-time top 500 by 60-day dollar volume (large, liquid names)
Truly delisted names (bankrupt or acquired) are missing from every set; listed_pit vs superset_pit measures the part
of the bias that comes from choosing names by today's price and volume.
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, rolling_mean, shift, summary

COST = 0.0010
HOLDOUT_END = pd.Timestamp("2023-12-21")


def main():
    d = Data()
    t0 = 260
    sup = {x["symbol"].upper().replace(".", "-") for x in json.load(open("trader/data/superset.json"))}
    is_sup = np.isin(d.syms, list(sup)) & ~d.is_etf
    stock = ~d.is_etf
    first = np.array([np.argmax(np.isfinite(d.C[:, j])) if np.isfinite(d.C[:, j]).any() else 10 ** 9 for j in range(d.C.shape[1])])
    long_hist = first <= np.searchsorted(d.dates, pd.Timestamp("2017-01-31"))
    avgvol = shift(rolling_mean(d.V, 60, 40), 1)
    pit = (shift(d.CA, 1) >= 5) & (avgvol >= 750_000)
    spy = d.C[:, d.s("SPY")]
    cash_day = shift((spy < rolling_mean(spy[:, None], 200)[:, 0]).astype(float)[:, None], 1)[:, 0] == 1
    gap = d.O / shift(d.C, 1) - 1
    rid = d.C / d.O - 1
    sets = {"superset_raw": np.broadcast_to(is_sup, d.C.shape),
            "superset_pit": pit & is_sup[None, :],
            "listed_pit": pit & stock[None, :],
            "long_hist_pit": pit & (stock & long_hist)[None, :],
            "recent_pit": pit & (stock & ~long_hist)[None, :],
            "listed_top500": d.universe(500)}
    out = {}
    for name, M in sets.items():
        r = np.full(len(d.dates), np.nan)
        for t in range(t0, len(d.dates)):
            if not cash_day[t]:
                continue
            m = M[t] & np.isfinite(gap[t]) & np.isfinite(rid[t]) & (gap[t] <= -0.02)
            ii = np.flatnonzero(m)
            if len(ii) == 0:
                continue
            sel = ii[np.argsort(gap[t, ii])[:5]]
            r[t] = rid[t, sel].mean() - COST
        s = pd.Series(r, index=d.dates).dropna()
        full = pd.Series(r, index=d.dates).iloc[t0:].fillna(0.0)
        tt = lambda x: x.mean() / x.std() * math.sqrt(len(x))
        pre, post = s[s.index < HOLDOUT_END], s[s.index >= HOLDOUT_END]
        sm = summary(full)
        out[name] = {"days": len(s), "bp": 1e4 * s.mean(), "t": tt(s), "pre_bp": 1e4 * pre.mean(), "pre_t": tt(pre),
                     "post_bp": 1e4 * post.mean(), "post_t": tt(post), "SR_daily_series": sm["sharpe"], "worst": s.min()}
        print(f"  {name:<14} active days {len(s):4d}  net {1e4*s.mean():+6.1f}bp/day  t {tt(s):+5.2f} | before {HOLDOUT_END.date()} "
              f"{1e4*pre.mean():+6.1f}bp t {tt(pre):+5.2f} | after {1e4*post.mean():+6.1f}bp t {tt(post):+5.2f} "
              f"| SR (all days) {sm['sharpe']:+.2f} worst day {100*s.min():+.1f}%")
    json.dump(out, open("trader/data/xsec_gapdown_check.json", "w"), default=float)


if __name__ == "__main__":
    main()
