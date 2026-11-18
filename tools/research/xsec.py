"""Cross-sectional intraday predictability (Fama-MacBeth). Target: 9:45 open -> 15:55 return.
Features known at 9:45: gap, first-15-min return, relative volume (first 5m), prior-day return, 5-day return,
distance to 20-day high, ATR %, dollar volume. Universe each day: price > $5, avg volume >= 500k."""
import sys, math
import numpy as np, pandas as pd
E = pd.read_pickle("trader/data/sip_events.pkl")
U = E[(E.open > 5) & (E.avgv >= 5e5)].dropna(subset=["r_rest", "r15"]).copy()
U["log_rv"] = np.log(U.rv.clip(lower=1e-3)); U["log_dvol"] = np.log(U.dvol)
F = ["gap", "r15", "log_rv", "prev_day", "ret5", "dist20", "atr_pct", "log_dvol"]
COST = 0.0010
print("stock-days", len(U), "days", U.date.nunique(), "avg stocks/day", round(len(U) / U.date.nunique()))
slopes, deciles = [], {f: [] for f in F}
for d, g in U.groupby("date"):
    X = g[F].rank(pct=True) - 0.5
    X.insert(0, "const", 1.0)
    y = g.r_rest.values
    b, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    slopes.append(pd.Series(b, index=["const"] + F, name=d))
    for f in F:
        q = g[f].rank(pct=True)
        deciles[f].append({"date": d, "top": g.r_rest[q > 0.9].mean(), "bottom": g.r_rest[q <= 0.1].mean(), "all": g.r_rest.mean()})
S = pd.DataFrame(slopes)
n = len(S)
print("\nFama-MacBeth (multivariate, rank features; slope = rest-of-day return difference, top vs bottom of the day, in bp):")
for f in F:
    m, sd = S[f].mean(), S[f].std(ddof=1)
    print(f"  {f:<10} {1e4*m:+7.1f}bp  t {m/sd*math.sqrt(n):+5.2f}  positive days {(S[f]>0).mean():.0%}")
print("\nLong-only single-feature deciles (equal weight, net of 10 bp), per day:")
for f in F:
    D = pd.DataFrame(deciles[f])
    for side in ("top", "bottom"):
        r = D[side] - COST
        print(f"  {f:<10} {side:<6} {1e4*r.mean():+7.1f}bp  t {r.mean()/r.std(ddof=1)*math.sqrt(len(r)):+5.2f}   (universe avg gross {1e4*D['all'].mean():+.1f}bp)")
