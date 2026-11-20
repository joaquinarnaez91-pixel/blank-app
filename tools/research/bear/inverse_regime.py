"""Bear-regime inverse-fund test (Nov 20, 2026, user asked about shorting the decline).
H: when the index is below its 200-day SMA, holding an inverse 3x fund over some window
has positive expected return. Proxy: -3 x index return (ignores fund fees/financing, which
cost ~1bp/day and make results look slightly better than reality).
Windows: open->close (day), close->open (night), close->close; conditions: below 200d,
plus prior day down, plus gap down at the open. Daily data, 10y cache."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
from bars import frame
from stats import summary

rows = []
for sym in ["SPY", "QQQ", "IWM"]:
    d = frame(sym, "1d")
    d = d.groupby("date").last()
    c, o = d["c"], d["o"]
    sma = c.rolling(200).mean().shift(1)          # known before today
    below = c.shift(1) < sma
    prev_ret = c.shift(1) / c.shift(2) - 1
    gap = o / c.shift(1) - 1
    day = o.pipe(lambda x: c / x - 1)
    night = o / c.shift(1) - 1
    cc = c / c.shift(1) - 1
    conds = {"all": pd.Series(True, index=d.index), "below200": below,
             "below200 & prev down": below & (prev_ret < 0),
             "below200 & prev down & gap down": below & (prev_ret < 0) & (gap < 0),
             "below200 & prev < -1%": below & (prev_ret < -0.01)}
    for cn, m in conds.items():
        m = m.fillna(False)
        for wn, r in {"day(open->close)": day, "close->close": cc}.items():
            x = (-3 * r[m]).dropna()
            if len(x) < 30: continue
            mu, sd = x.mean(), x.std()
            t = mu / sd * np.sqrt(len(x))
            yrs = sorted(set(pd.Index(x.index).map(lambda z: z.year)))
            h1 = x.iloc[: len(x)//2].mean(); h2 = x.iloc[len(x)//2:].mean()
            rows.append(dict(sym=sym, cond=cn, window=wn, n=len(x), mean_bp=round(mu*1e4,1),
                             t=round(t,2), win=round((x>0).mean(),3), half1_bp=round(h1*1e4,1),
                             half2_bp=round(h2*1e4,1), first=str(x.index[0]), last=str(x.index[-1])))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(out.to_string(index=False))
out.to_csv("tools/research/bear/inverse_regime.csv", index=False)
