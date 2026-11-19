"""(d) Post-earnings-announcement drift, proxied by high-volume gap-up days (earnings dates were not obtainable:
SEC EDGAR is outside this session's network allowlist, and Yahoo's earnings-date endpoints need a session crumb).

  python3 tools/research/xsec_swing/pead_proxy.py [--superset-only]
Event at the close of day t (top-500 by dollar volume): volume >= 3x its median of the prior 60 days, opening gap
>= g and close-to-close return >= g (the gap held). Buy at the close (largest close-to-close return first),
hold H trading days, sell at the close. 10 slots x 10% (fractional) for statistics, 10 bp round trip.
Most such days are earnings announcements (the announcement-return drift of Brandt et al. 2008; Chan 2003).
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
    medv = shift(pd.DataFrame(d.V).rolling(60, min_periods=40).median().values, 1)
    vr = d.V / medv
    gap = d.O / shift(d.C, 1) - 1
    cc = d.C / shift(d.C, 1) - 1
    idx, _ = index_model(d)
    idx = idx[idx.index >= d.dates[t0]]
    never = lambda t, j, t_in: False
    res = {}
    for g in (0.03, 0.06):
        ev = U & (vr >= 3) & (gap >= g) & (cc >= g)
        print(f"events with gap >= {g:.0%}: {int(ev[t0:].sum())} ({ev[t0:].sum() / (len(d.dates) - t0):.2f} per day)")
        for H in (20, 60):
            r, tr = simulate(d, ev, -cc, never, K=10, capital=1e7, cost=COST, whole=False, max_hold=H, t0=t0)
            name = f"pead_g{int(g*100)}_h{H}"
            res[name] = r
            ts = trade_stats(tr)
            report(name, r)
            print(f"      trades {ts['trades']}, mean {ts['mean_pct']:+.2f}%/trade, win {ts['win']:.0%}, worst {ts['worst_pct']:+.0f}%, "
                  f"corr(index model) {r.corr(idx):+.2f}")
    # event-study style check (no slot limit): mean H-day return after the event vs the universe average
    for g in (0.03, 0.06):
        ev = U & (vr >= 3) & (gap >= g) & (cc >= g)
        for H in (5, 20, 60):
            fwd = shift(d.C, -H) / d.C - 1
            uni = np.nanmean(np.where(U, fwd, np.nan), axis=1)
            ex = (fwd - uni[:, None])[ev & np.isfinite(fwd)]
            # cluster by month for a rough t-stat
            rows = np.argwhere(ev & np.isfinite(fwd))
            mon = pd.Series(ex, index=d.dates[rows[:, 0]]).groupby(lambda x: (x.year, x.month)).mean()
            print(f"  event study gap>={g:.0%} H={H}: mean excess {100*ex.mean():+.2f}% over universe, n {len(ex)}, "
                  f"monthly-clustered t {mon.mean()/mon.std()*math.sqrt(len(mon)):+.2f}")
    tag = "_superset" if superset_only else ""
    pd.DataFrame(res).assign(index_model=idx).to_pickle(f"trader/data/xsec_pead_series{tag}.pkl")


if __name__ == "__main__":
    main()
