"""Tug of war, part 2: hold 12-1 month momentum winners continuously (close to close, multi-day) instead of
overnight only. Long only, top-500 by dollar volume, 10 bp round trip per trade.

  python3 tools/research/xsec_swing/momentum_hold.py [--superset-only]
Rule: at each close, buy names ranked in the top N by 12-1 month momentum (best first) into free slots;
sell at the close when a held name falls out of the top 2N (a buffer that cuts turnover).
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, index_model, rolling_mean, shift, report, summary, by_year, halves
from swing import simulate, trade_stats

COST = 0.0010


def main():
    superset_only = "--superset-only" in sys.argv
    d = Data()
    names = None
    if superset_only:
        names = {x["symbol"].upper().replace(".", "-") for x in json.load(open("trader/data/superset.json"))}
    U = d.universe(500, names=names)
    t0 = 260
    C = d.C
    mom = shift(C, 21) / shift(C, 252) - 1
    rank = pd.DataFrame(np.where(U & np.isfinite(mom), -mom, np.nan)).rank(axis=1, method="first").values
    idx, _ = index_model(d)
    idx = idx[idx.index >= d.dates[t0]]
    res = {}
    for n in (5, 10):
        entry = rank <= n
        ex = lambda t, j, t_in, n=n: not (rank[t, j] <= 2 * n)
        r, tr = simulate(d, entry, rank, ex, K=n, capital=1e7, cost=COST, whole=False, max_hold=10 ** 9, t0=t0)
        name = f"m_hold_top{n}"
        res[name] = r
        ts = trade_stats(tr)
        report(name + " (fractional)", r)
        print(f"    trades {ts['trades']}, mean hold {ts['days']:.0f} days, mean {ts['mean_pct']:+.1f}%/trade, win {ts['win']:.0%}, "
              f"corr(index model) {r.corr(idx):+.2f}, by year {({y: v[0] for y, v in by_year(r).items()})}")
        r1, tr1 = simulate(d, entry, rank, ex, K=n, capital=1000.0, cost=COST, whole=True, max_hold=10 ** 9, t0=t0)
        report(name + " ($1,000 whole shares)", r1)
    tag = "_superset" if superset_only else ""
    pd.DataFrame(res).to_pickle(f"trader/data/xsec_momhold_series{tag}.pkl")


if __name__ == "__main__":
    main()
