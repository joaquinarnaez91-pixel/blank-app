"""Mean-reversion swing trades only while the live index model holds cash (all of SPY, QQQ, IWM below their
200-day averages), combined with the index model under the cash-account rule.

  python3 tools/research/xsec_swing/regime.py
Prior: short-term reversal (liquidity provision) pays more when markets are turbulent (Nagel 2012).
New entries only at closes after which the index model holds nothing; open positions run to their normal exit.
Combined sleeve: swing positions first; whatever is in cash at the close goes into the index model that night
(each dollar is committed once per day: index-model dollars come back at the 9:30 open and can fund a 15:58 buy).
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, index_model, rolling_mean, shift, report, summary, by_year, halves, dsr
from swing import simulate, trade_stats
from meanrev import build_signals, walk_forward

COST = 0.0010


def main():
    d = Data()
    t0 = 260
    V, aux = build_signals(d)
    idx, legs = index_model(d)
    gate = np.zeros(len(d.dates), bool)
    for u in ("SPY", "QQQ", "IWM"):
        c = d.C[:, d.s(u)]
        gate |= c > rolling_mean(c[:, None], 200)[:, 0]
    cash_regime = ~gate                                    # at the close of t: index model holds nothing tonight
    print("closes in the cash regime since start: %d of %d" % (cash_regime[t0:].sum(), len(d.dates) - t0))
    sl = lambda s: s[s.index >= d.dates[t0]]
    idx = sl(idx)
    res, comb = {}, {}
    for name, (entry, prio, ex, H) in V.items():
        r, tr, cf = simulate(d, entry, prio, ex, K=10, capital=1e7, cost=COST, whole=False, max_hold=H, t0=t0,
                             gate=cash_regime, return_frac=True)
        r = sl(r); cf = sl(cf)
        c = r + cf.shift(1).fillna(1.0) * idx
        res[name], comb[name] = r, c
        s, sc = summary(r), summary(c)
        on = pd.Series(cash_regime, index=d.dates).shift(1).reindex(r.index).fillna(False).astype(bool)
        rr = r[on]
        ts = trade_stats(tr)
        print(f"  {name:<20} swing-only SR {s['sharpe']:+5.2f} t {s['t']:+5.2f} | cash-regime days mean {1e4*rr.mean():+6.1f}bp "
              f"t {rr.mean()/rr.std()*math.sqrt(len(rr)):+5.2f} | trades {ts.get('trades',0)} mean {ts.get('mean_pct',0):+.2f}% "
              f"| combined SR {sc['sharpe']:+5.2f} ann {sc['ann_ret']:+6.1%} maxDD {sc['maxdd']:+6.1%}")
    report("index model alone", idx)
    srs = {k: summary(v)["sharpe"] for k, v in res.items()}
    var = float(np.var(list(srs.values()), ddof=1))
    best = max(srs, key=srs.get)
    report("BEST gated swing " + best, res[best], n_trials=len(res), sr_var=var)
    report("BEST combined " + best, comb[best])
    print("  gated swing by year:", {y: v[:2] for y, v in by_year(res[best]).items()})
    wf, picks = walk_forward(res, first_year=2019)
    print("  walk-forward picks:", picks)
    report("walk-forward gated swing OOS", wf, n_trials=len(res), sr_var=var)
    wfc = pd.concat([comb[picks[y]][comb[picks[y]].index.year == y] for y in picks])
    report("walk-forward combined OOS", wfc)
    report("index model, same period", idx[idx.index >= wfc.index[0]])
    diff = wfc - idx[idx.index >= wfc.index[0]]
    print("  combined minus index alone (OOS): %+.2f bp/day t %.2f" % (1e4 * diff.mean(), diff.mean() / diff.std() * math.sqrt(len(diff))))
    # $1,000 whole-share version of the canonical Connors rule and of the best, gated
    for name in dict.fromkeys(["c_rsi10_x5_h5", best]):
        entry, prio, ex, H = V[name]
        r, tr, cf = simulate(d, entry, prio, ex, K=4, capital=1000.0, cost=COST, whole=True, max_hold=H, t0=t0,
                             gate=cash_regime, return_frac=True)
        r = sl(r); cf = sl(cf)
        c = r + cf.shift(1).fillna(1.0) * idx
        ts = trade_stats(tr)
        report(f"$1,000 4 slots gated {name}", r)
        report(f"$1,000 combined with index model", c)
        print(f"      trades {ts['trades']}, mean {ts['mean_pct']:+.2f}%, win {ts['win']:.0%}, hold {ts['days']:.1f}d, worst trade {ts['worst_pct']:+.1f}% "
              f"(${ts['worst_usd']:.0f}), worst day ${1000*r.min():.0f} (non-compounded approx)")
    pd.DataFrame(res).to_pickle("trader/data/xsec_regime_series.pkl")
    json.dump({"srs": srs, "var": var, "best": best, "picks": {str(k): v for k, v in picks.items()}},
              open("trader/data/xsec_regime_summary.json", "w"))


if __name__ == "__main__":
    main()
