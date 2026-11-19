"""Entry-time sensitivity of the cash-day gap-down reversal (no new data, no new grid).

Picks are always chosen at the open exactly as in the rule: base universe (prior close >= $5, 14-day mean volume
>= 1M shares), gap = 9:30 open / prior close - 1 <= -2%, the 5 largest gap-downs priced <= $200. Only the ENTRY
time changes; exit at the close. Equal-weight mean of the 5 picks, net of 10 bp. Cash days = the overnight model
was in cash for SPY's leg (SPY close t-1 < 200-day SMA), and separately for any leg (SPY, QQQ or IWM).
  hourly bars (2024-26): entry at the 9:30 open vs the 10:30 open.
  5-minute bars (last ~60 sessions): entry at 9:30 / 9:35 / 9:45 / 10:00 / 10:30, whatever cash days exist.
Run: python3 tools/research/intraday_hourly/entry_timing.py [--log]
"""
import sys, glob, os, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from panel import load, features
from common import regime_below, cash_fraction, nw_t, log_exp, CODE

COST, K = 0.0010, 5


def stats(x):
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return f"n {len(x)}"
    return f"{1e4*x.mean():+6.1f} bp/day (t {x.mean()/x.std()*math.sqrt(len(x)):+.2f}, n {len(x)})"


def hourly():
    P = load(); F = features(P)
    o, c = P["o"].astype(float), P["c"].astype(float)
    dates = pd.to_datetime(pd.Series(P["dates"])).dt.date.to_numpy()
    U = (F["full"] & (F["prevC"] >= 5) & (F["adv"] >= 1e6) & np.isfinite(F["gap"]) & (np.abs(F["gap"]) < 0.6)
         & (F["gap"] <= -0.02) & (o[:, :, 0] <= 200))
    U[:, :20] = False
    spy, anyc = regime_below(dates, "SPY"), cash_fraction(dates) > 0
    rows = []
    for t in range(20, len(dates)):
        idx = np.nonzero(U[:, t])[0]
        if len(idx) == 0:
            continue
        p = idx[np.argsort(F["gap"][idx, t])[:K]]
        rows.append({"date": dates[t], "spy": spy[t], "any": anyc[t],
                     "e930": np.mean(c[p, t, 6] / o[p, t, 0] - 1) - COST,
                     "e1030": np.mean(c[p, t, 6] / o[p, t, 1] - 1) - COST,
                     "first_hour": np.mean(o[p, t, 1] / o[p, t, 0] - 1)})
    D = pd.DataFrame(rows)
    print(f"Hourly bars, {P['o'].shape[0]} stocks, {D.date.min()}..{D.date.max()} (top-5 gap-downs, EW, net 10 bp):")
    out = {}
    for lab, m in (("SPY<200d cash days", D.spy), ("any leg in cash", D["any"]), ("all other days", ~D["any"])):
        d = D[m]
        print(f"  {lab:20s} entry 9:30 open {stats(d.e930)} | entry 10:30 open {stats(d.e1030)} | "
              f"picks' 9:30->10:30 move {1e4*d.first_hour.mean():+.1f} bp")
        out[lab] = (d.e930.mean(), d.e1030.mean(), len(d))
    return out


def five_minute():
    rows = []
    for f in glob.glob("trader/data/5m/*.npz"):
        s = os.path.basename(f)[:-4]
        if s in ("SPY", "QQQ", "IWM", "SMH", "^VIX", "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "TNA", "TZA", "FAS"):
            continue
        df = frame(s, "5m")
        if df is None or len(df) < 500:
            continue
        g = df.groupby("date")
        first, last, vol = g.o.first(), g.c.last(), g.v.sum()
        ent = {k: df[df.tod == k].groupby("date").o.first() for k in (575, 585, 600, 630)}
        x = pd.DataFrame({"o": first, "c": last, "pc": last.shift(1), "adv": vol.rolling(14).mean().shift(1),
                          **{f"o{k}": v for k, v in ent.items()}})
        x["gap"] = x.o / x.pc - 1
        x = x[(x.gap <= -0.02) & (x.gap > -0.6) & (x.pc >= 5) & (x.adv >= 1e6) & (x.o <= 200)].dropna()
        for d, r in x.iterrows():
            rows.append({"date": d, "gap": r.gap, "e930": r.c / r.o - 1, **{f"e{k}": r.c / r[f"o{k}"] - 1 for k in ent}})
    X = pd.DataFrame(rows)
    X["rk"] = X.groupby("date").gap.rank(method="first")
    T5 = X[X.rk <= K].groupby("date")[["e930", "e575", "e585", "e600", "e630"]].mean() - COST
    dates = list(T5.index)
    spy, anyc = regime_below(dates, "SPY"), cash_fraction(dates) > 0
    print(f"\n5-minute bars, {len(dates)} sessions {dates[0]}..{dates[-1]}: SPY<200d days {int(spy.sum())}, "
          f"any-leg cash days {int(anyc.sum())}")
    labs = {"e930": "9:30", "e575": "9:35", "e585": "9:45", "e600": "10:00", "e630": "10:30"}
    out = {}
    for lab, m in (("any leg in cash", anyc), ("all other days", ~anyc)):
        if m.sum() == 0:
            print(f"  {lab}: none in this sample")
            continue
        d = T5[m]
        print(f"  {lab:16s} " + " | ".join(f"{labs[k]} {stats(d[k])}" for k in labs))
        out[lab] = {labs[k]: (d[k].mean(), int(d[k].notna().sum())) for k in labs}
    return out


if __name__ == "__main__":
    H = hourly()
    M = five_minute()
    if "--log" in sys.argv:
        s, a = H["SPY<200d cash days"], H["any leg in cash"]
        log_exp(family="gaps_intraday", name="Cash-day gap-down reversal: entry-time sensitivity (hourly 9:30 vs 10:30; 5-min 9:30-10:30)",
                hypothesis="Most of the cash-day reversal accrues after the open, so a later entry (owner's window starts 9:45) keeps most of it",
                data="hourly bars 2024-26 (full universe); 5-minute bars, last ~60 sessions",
                method="same picks chosen at the open; only the entry time changes; EW top-5, net 10 bp",
                variants=0,
                metrics={"h_spy_930_bp": 1e4 * s[0], "h_spy_1030_bp": 1e4 * s[1], "h_spy_n": s[2],
                         "h_any_930_bp": 1e4 * a[0], "h_any_1030_bp": 1e4 * a[1], "h_any_n": a[2],
                         **({f"m5_cash_{k}": 1e4 * v[0] for k, v in M["any leg in cash"].items()} if "any leg in cash" in M else {})},
                verdict="promising", notes="diagnostic of the existing rule (no new trial); see report section 6",
                code=CODE + "entry_timing.py")
