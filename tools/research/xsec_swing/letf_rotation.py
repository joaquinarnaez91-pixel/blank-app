"""(b) extension: a long-only overnight momentum portfolio that is cheap to trade -- rotate the live model's
3x exposure across the allowed bull funds by momentum.

  python3 tools/research/xsec_swing/letf_rotation.py
Funds (underlying used for signals): UPRO (SPY), TQQQ (QQQ), TNA (IWM), SOXL (SMH), FAS (XLF).
Live-model rules per fund: hold only if the underlying closes above its 200-day average; index exposure
min(3, 1.5% / 20-night std of the underlying's overnight return); buy ~15:58, sell at the 9:30 open; 4 bp per night.
Variants: equal fifths of all five (no momentum); top-K (K = 1, 2, 3) of the trend-qualified funds by
12-1 month momentum or by 12-month cumulative overnight return, each holding 1/K of the sleeve.
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, index_model, rolling_mean, shift, report, summary, by_year, halves, dsr

PAIRS = [("SPY", "UPRO"), ("QQQ", "TQQQ"), ("IWM", "TNA"), ("SMH", "SOXL"), ("XLF", "FAS")]
COST = 0.0004


def main():
    d = Data()
    t0 = 260
    T = len(d.dates)
    sig, w_full, fret = {}, {}, {}
    for u, f in PAIRS:
        i, j = d.s(u), d.s(f)
        c = d.C[:, i]
        trend = c > rolling_mean(c[:, None], 200)[:, 0]
        vol = pd.Series(d.RON[:, i]).rolling(20, min_periods=20).std().values
        expo = np.nan_to_num(np.minimum(3.0, 0.015 / vol)) / 3.0             # fraction of a slot in the 3x fund
        w_full[f] = np.where(trend, expo, 0.0)
        mom = c / shift(c[:, None], 0)[:, 0]
        mom = shift(c[:, None], 21)[:, 0] / shift(c[:, None], 252)[:, 0] - 1
        on12 = pd.Series(np.log1p(np.nan_to_num(d.RON[:, i]))).rolling(252, min_periods=200).sum().values
        sig[f] = {"mom12_1": np.where(trend, mom, np.nan), "on12": np.where(trend, on12, np.nan)}
        fret[f] = shift(d.RON[:, j][:, None], -1)[:, 0]                      # fund overnight t -> t+1
    funds = [f for _, f in PAIRS]
    W = np.stack([w_full[f] for f in funds], 1)
    R = np.nan_to_num(np.stack([fret[f] for f in funds], 1))
    idx, legs = index_model(d)
    sl = lambda s: s[s.index >= d.dates[t0]]
    res = {}

    def series(weights):
        night = (weights * (R - COST)).sum(1)
        return pd.Series(shift(night[:, None], 1)[:, 0], index=d.dates).fillna(0.0)

    res["live_thirds"] = sl(idx)
    res["fifths_all"] = sl(series(W / 5.0))
    for name in ("mom12_1", "on12"):
        S = np.stack([sig[f][name] for f in funds], 1)
        for K in (1, 2, 3):
            wt = np.zeros_like(W)
            for t in range(T):
                ok = np.flatnonzero(np.isfinite(S[t]) & (W[t] > 0))
                if len(ok) == 0:
                    continue
                top = ok[np.argsort(-S[t, ok])[:K]]
                wt[t, top] = W[t, top] / K
            res[f"rot_{name}_top{K}"] = sl(series(wt))
    print("== Overnight 3x-fund rotation (net of 4 bp per night) ==")
    for k, r in res.items():
        report(k, r)
        print(f"      worst night {100*r.min():+.1f}% (${10*r.min()*100:.0f} on $1,000), corr(live) {r.corr(res['live_thirds']):+.2f}")
    trials = {k: v for k, v in res.items() if k != "live_thirds"}
    srs = {k: summary(v)["sharpe"] for k, v in trials.items()}
    var = float(np.var(list(srs.values()), ddof=1))
    best = max(srs, key=srs.get)
    print(f"\n{len(trials)} variants; best {best}")
    report("BEST " + best, trials[best], n_trials=len(trials), sr_var=var)
    diff = trials[best] - res["live_thirds"]
    print("  best minus live: %+.1f bp/night, t %.2f" % (1e4 * diff.mean(), diff.mean() / diff.std() * math.sqrt(len(diff))))
    print("  by year best:", {y: v[:2] for y, v in by_year(trials[best]).items()})
    print("  by year live:", {y: v[:2] for y, v in by_year(res['live_thirds']).items()})
    df = pd.DataFrame(trials)
    wf = []
    picks = {}
    for y in sorted(set(df.index.year)):
        if y < 2019:
            continue
        h = df[df.index.year < y]
        b = (h.mean() / h.std()).idxmax(); picks[y] = b
        wf.append(df[df.index.year == y][b])
    wf = pd.concat(wf)
    print("  walk-forward picks:", picks)
    report("walk-forward OOS", wf, n_trials=len(trials), sr_var=var)
    report("live model, same period", res["live_thirds"][res["live_thirds"].index >= wf.index[0]])
    pd.DataFrame(res).assign(walk_forward=wf, index_model=res["live_thirds"], index_expo=sl(legs["_expo"])).to_pickle("trader/data/xsec_letf_rotation.pkl")


if __name__ == "__main__":
    main()
