"""(c) Short-term mean-reversion swing trades in liquid large caps (long only, daily data).

  python3 tools/research/xsec_swing/meanrev.py [--superset-only]
Variants (all counted):
  Connors RSI(2): buy at the close when RSI(2) < 10 (or 5) and close > 200-day SMA; sell at the close when the
  close > 5-day SMA or after H days; or time exits only (1/3/5 days). 1-week loser reversal: buy the 10 worst
  5-day returns of the day (top-200 or top-500 by dollar volume), hold 5 days, optional 200-day filter.
Signal portfolio for statistics: 10 slots of 10% each, fractional shares, lowest RSI(2) / worst 5-day return first.
Costs: 10 bp round trip (5 bp each side) for top-500 names.
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, index_model, rolling_mean, shift, report, summary, by_year, halves, dsr, fmt
from swing import simulate, rsi, trade_stats

COST = 0.0010


def build_signals(d, names=None):
    U500, U200 = d.universe(500, names=names), d.universe(200, names=names)
    C = d.C
    sma200, sma5 = rolling_mean(C, 200), rolling_mean(C, 5)
    r2 = rsi(C, 2)
    ret5 = C / shift(C, 5) - 1
    above = C > sma200
    rk200 = pd.DataFrame(np.where(U200, ret5, np.nan)).rank(axis=1, method="first").values
    rk500 = pd.DataFrame(np.where(U500, ret5, np.nan)).rank(axis=1, method="first").values
    x5 = lambda t, j, t_in: C[t, j] > sma5[t, j]
    never = lambda t, j, t_in: False
    V = {
        "c_rsi10_x5_h5":      (U500 & (r2 < 10) & above, r2, x5, 5),
        "c_rsi5_x5_h5":       (U500 & (r2 < 5) & above, r2, x5, 5),
        "c_rsi10_x5_h10":     (U500 & (r2 < 10) & above, r2, x5, 10),
        "c_rsi10_t1":         (U500 & (r2 < 10) & above, r2, never, 1),
        "c_rsi10_t3":         (U500 & (r2 < 10) & above, r2, never, 3),
        "c_rsi10_t5":         (U500 & (r2 < 10) & above, r2, never, 5),
        "c_rsi10_x5_h5_u200": (U200 & (r2 < 10) & above, r2, x5, 5),
        "r_wk_u200_t5":       (rk200 <= 10, ret5, never, 5),
        "r_wk_u200_t5_sma":   ((rk200 <= 10) & above, ret5, never, 5),
        "r_wk_u500_t5":       (rk500 <= 10, ret5, never, 5),
    }
    return V, dict(U500=U500, U200=U200, sma200=sma200, sma5=sma5, r2=r2, ret5=ret5)


def walk_forward(series, first_year):
    """Each calendar year, trade the variant with the best Sharpe over all prior data (expanding window)."""
    df = pd.DataFrame(series)
    years = sorted(set(df.index.year))
    out, picks = [], {}
    for y in years:
        if y < first_year:
            continue
        hist = df[df.index.year < y]
        sr = hist.mean() / hist.std() * math.sqrt(252)
        best = sr.idxmax()
        picks[y] = best
        out.append(df[df.index.year == y][best])
    return pd.concat(out), picks


def main():
    superset_only = "--superset-only" in sys.argv
    d = Data()
    names = None
    if superset_only:
        names = {x["symbol"].upper().replace(".", "-") for x in json.load(open("trader/data/superset.json"))}
    V, aux = build_signals(d, names)
    t0 = 260
    start = d.dates[t0]
    print("panel", d.C.shape, "start", start.date(), "superset only" if superset_only else "all listed names",
          "| top-500 ADV cut-off ($M) min/median: %.0f / %.0f" % (
              np.nanmin(np.where(aux["U500"], d.ADV, np.nan).min(1)[t0:]) / 1e6 if False else 0, 0))
    adv_cut = np.nanmin(np.where(aux["U500"], d.ADV, np.inf), axis=1)[t0:]
    print("top-500 dollar-volume cut-off: min $%.0fM, median $%.0fM" % (adv_cut.min() / 1e6, np.median(adv_cut) / 1e6))
    idx, _ = index_model(d)
    idx = idx[idx.index >= start]

    res, trs = {}, {}
    print("\n== Signal portfolios: 10 slots x 10%, fractional, 10 bp round trip ==")
    for name, (entry, prio, ex, H) in V.items():
        r, tr = simulate(d, entry, prio, ex, K=10, capital=1e7, cost=COST, whole=False, max_hold=H, t0=t0)
        res[name], trs[name] = r, tr
        s, ts = summary(r), trade_stats(tr)
        h = halves(r)
        print(f"  {name:<20} SR {s['sharpe']:+5.2f} t {s['t']:+5.2f} ann {s['ann_ret']:+6.1%} maxDD {s['maxdd']:+6.1%} | "
              f"trades {ts['trades']:5d} mean {ts['mean_pct']:+5.2f}% win {ts['win']:.0%} days {ts['days']:.1f} worst {ts['worst_pct']:+.0f}%"
              f" | halves SR {h['first'][1]:+.2f}/{h['second'][1]:+.2f}  corr(idx) {r.corr(idx):+.2f}")
    srs = {k: summary(v)["sharpe"] for k, v in res.items()}
    var = float(np.var(list(srs.values()), ddof=1))
    best = max(srs, key=srs.get)
    print(f"\n{len(res)} variants; best {best}")
    report("BEST " + best, res[best], n_trials=len(res), sr_var=var)
    print("  by year:", {y: v[:2] for y, v in by_year(res[best]).items()})
    wf, picks = walk_forward(res, first_year=2019)
    print("\n== Walk-forward (yearly re-selection by expanding-window Sharpe), out of sample from 2019 ==")
    print("  picks:", picks)
    report("walk-forward OOS", wf, n_trials=len(res), sr_var=var)
    canon = "c_rsi10_x5_h5"
    report("Canonical Connors (published rule, not fitted)", res[canon], n_trials=len(res), sr_var=var)
    print("  canonical by year:", {y: v[:2] for y, v in by_year(res[canon]).items()})
    tag = "_superset" if superset_only else ""
    pd.DataFrame(res).assign(index_model=idx).to_pickle(f"trader/data/xsec_meanrev_series{tag}.pkl")
    json.dump({"srs": srs, "var": var, "n": len(res), "best": best, "wf_picks": {str(k): v for k, v in picks.items()}},
              open(f"trader/data/xsec_meanrev_summary{tag}.json", "w"))
    for k, tr in trs.items():
        tr.to_pickle(f"trader/data/xsec_meanrev_trades_{k}{tag}.pkl")


if __name__ == "__main__":
    main()
