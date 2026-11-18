"""What the current rules would have done over the last ~60 sessions (Yahoo 5m bars):
15-minute opening-range breakout on the 3x funds, stop at the range midpoint, 2R / 1R / hold-to-15:55 exits,
with and without the market-direction filter and the 10:30 cutoff. Entry at the open of the bar after the
first 5-minute close above the range high. Stops/targets checked bar by bar, stop first when both touch."""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
from bars import frame

COST = 0.0005
BULL = ["TQQQ", "SOXL", "UPRO", "TNA", "FAS"]
BEAR = ["SQQQ", "SOXS", "SPXU", "TZA"]


def market_state(df):
    """Per date: dict tod -> (above_mid_and_not_down, below_mid_and_down) using closes up to that bar."""
    out = {}
    prev_close = df.groupby("date").c.last().shift(1)
    for d, g in df.groupby("date"):
        orr = g[g.tod < 585]
        if len(orr) < 3 or pd.isna(prev_close.get(d)):
            continue
        mid = (orr.h.max() + orr.l.min()) / 2
        out[d] = {t: (c > mid and c / prev_close[d] - 1 > -0.005, c < mid and c / prev_close[d] - 1 < -0.005)
                  for t, c in zip(g.tod, g.c)}
    return out


def trades_for(sym, mkt, side):
    df = frame(sym, "5m")
    rows = []
    for d, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        orr = g[g.tod < 585]
        if len(orr) < 3:
            continue
        hi, lo = orr.h.max(), orr.l.min()
        mid = (hi + lo) / 2
        post = g[(g.tod >= 585)]
        for i, r in post.iterrows():
            if r.tod > 925:          # entries only through the 15:25 bar (fills by 15:30)
                break
            if r.c > hi:
                if i + 1 >= len(g):
                    break
                entry = g.o[i + 1]
                t_entry = g.tod[i + 1]
                risk = entry - mid
                if risk <= 0:
                    break
                res = {}
                for name, tgt in (("1R", entry + risk), ("2R", entry + 2 * risk), ("EOD", None)):
                    exitp = None
                    for j in range(i + 1, len(g)):
                        if g.tod[j] >= 955:      # flat at 15:55: exit at that bar's open
                            exitp = g.o[j]; break
                        if g.l[j] <= mid:
                            exitp = min(mid, g.o[j]); break
                        if tgt is not None and g.h[j] >= tgt:
                            exitp = max(tgt, g.o[j]) if g.o[j] < tgt else g.o[j]; break
                    if exitp is None:
                        exitp = g.c.iloc[-1]
                    res[name] = (exitp - entry) / risk
                    res[name + "_ret"] = exitp / entry - 1 - COST
                m = mkt.get(d, {}).get(r.tod)
                filt = (m[0] if side == "bull" else m[1]) if m else False
                rows.append({"sym": sym, "date": d, "tod": t_entry, "side": side, "filter": filt, **res})
                break
    return rows


def show(df, label):
    if df.empty:
        print(f"{label:<55} n 0"); return
    daily = df.groupby("date")["EOD_ret"].mean()
    t_day = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 2 else float("nan")
    print(f"{label:<55} n {len(df):3d} days {df.date.nunique():2d} | R 1R {df['1R'].mean():+.2f} 2R {df['2R'].mean():+.2f} "
          f"EOD {df['EOD'].mean():+.2f} | %/trade 2R {100*df['2R_ret'].mean():+.2f} EOD {100*df['EOD_ret'].mean():+.2f} "
          f"| day-clustered t(EOD) {t_day:+.2f}")


if __name__ == "__main__":
    spy, qqq = market_state(frame("SPY", "5m")), market_state(frame("QQQ", "5m"))
    mkt = {d: {t: (spy[d][t][0] and qqq[d].get(t, (False, False))[0], spy[d][t][1] and qqq[d].get(t, (False, False))[1])
               for t in spy[d]} for d in spy if d in qqq}
    rows = []
    for s in BULL:
        rows += trades_for(s, mkt, "bull")
    for s in BEAR:
        rows += trades_for(s, mkt, "bear")
    T = pd.DataFrame(rows)
    T.to_csv("trader/data/letf_orb_trades.csv", index=False)
    print("Sessions:", T.date.nunique(), "from", T.date.min(), "to", T.date.max())
    show(T, "all 3x-fund ORB breakouts (no filter)")
    show(T[T["filter"]], "with market filter (current rule)")
    show(T[~T["filter"]], "filter says no")
    show(T[T["filter"] & (T.tod <= 630)], "filter + entry by 10:30 (new rule)")
    show(T[T["filter"] & (T.tod > 630)], "filter + entry after 10:30")
    show(T[T.tod <= 630], "no filter, entry by 10:30")
    show(T[T.side == "bull"], "bull funds")
    show(T[T.side == "bear"], "inverse funds")
    show(T[(T.side == "bear") & T["filter"] & (T.tod <= 630)], "inverse funds, filter, by 10:30")
