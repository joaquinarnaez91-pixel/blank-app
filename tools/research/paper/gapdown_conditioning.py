"""Does the cash-day gap-down basket depend on SPY's own state at the open? (research only, Nov 23 2026)

Motivation: the live pilot lost on its first two days (Nov 19-20 2026), both days SPY kept falling after the open.
Question: is there a skip rule, known BEFORE entry, that avoids such days?

Basket (same data and rule as intraday_hourly/gd_regime.py, the pilot's backtest): daily Yahoo bars Nov 2016 -
Nov 2026 for trader/data/superset.json (today's listing; survivorship flatters gap-down reversal). Day t is traded
only if SPY's close on t-1 was below its 200-day SMA. Eligible: prior close >= $5, 14-day mean volume (t-14..t-1)
>= 1M, gap = open/prior close - 1 <= -2% (|gap| < 60%), open price <= $95 (pilot's per-slot cap). Buy the 5 most
negative gaps at the 9:30 open (daily bars have no 9:45 price), equal weight, sell at the close, 10 bp round trip.
Sensitivity: the pilot's OCO exits approximated on daily bars (stop -8% / take-profit +20% from the open; if both
are touched the stop is assumed first).

Pre-specified conditioning variables, all known at the 9:30 open (no information after the open is used):
  A. SPY open gap vs prior close: < -1%, -1..0, 0..+1, > +1%
  B. SPY prior-day return (close t-2 -> t-1): down > 1%, down 0..1%, up
  C. SPY distance below its 200-day SMA at t-1: 0-3%, 3-7%, > 7%
Stats per bucket: days, mean bp, t on the daily series (one observation per day = clustered by day), hit rate,
halves (2017-2021 / 2022-2026) and the difference vs the rest of the sample. A descriptive (look-ahead, NOT usable)
row splits by SPY's own open->close direction to show how much of the basket is market beta.
Run: python3 tools/research/paper/gapdown_conditioning.py
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from gaps_daily_holdout import build

COST, K, PXCAP, STOP, TP = 0.0010, 5, 95.0, 0.08, 0.20
SPLIT = pd.Timestamp("2022-01-01").date()


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def main():
    syms, cal, A = build()
    o, h, l, c, v = A["o"], A["h"], A["l"], A["c"], A["v"]
    N, T = c.shape
    dates = np.array(list(cal))
    pC = np.concatenate([np.full((N, 1), np.nan), c[:, :-1]], 1)
    adv = pd.DataFrame(v.T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    gap = o / pC - 1
    y = c / o - 1
    ok = np.isfinite(gap) & np.isfinite(y) & (np.abs(gap) < 0.6) & (o > 0) & (np.abs(y) < 0.6)
    ok[:, :20] = False
    el = ok & (pC >= 5) & (adv >= 1e6) & (gap <= -0.02) & (o <= PXCAP)
    # stop/TP approximation on daily high/low
    lo_r, hi_r = l / o - 1, h / o - 1
    y_oco = np.where(lo_r <= -STOP, -STOP, np.where(hi_r >= TP, TP, y))

    # SPY state, all known at the open of day t
    s = frame("SPY", "1d"); s = s[~s.index.duplicated()].set_index("date")
    sc, so = s.c, s.o
    sma = sc.rolling(200).mean()
    spy = pd.DataFrame({"gap": so / sc.shift(1) - 1, "prev_ret": sc.shift(1) / sc.shift(2) - 1,
                        "dist200": (sc / sma - 1).shift(1), "oc": sc / so - 1}).reindex(pd.Index(dates))
    cash = (spy.dist200 < 0).to_numpy()

    rows = []
    for t in range(20, T):
        if not cash[t]:
            continue
        idx = np.where(el[:, t])[0]
        if len(idx) == 0:
            continue
        p = idx[np.argsort(gap[idx, t])][:K]
        rows.append({"date": dates[t], "n": len(p), "ret": y[p, t].mean() - COST, "ret_oco": y_oco[p, t].mean() - COST,
                     **spy.iloc[t].to_dict()})
    D = pd.DataFrame(rows).set_index("date")
    D["half"] = np.where(D.index < SPLIT, "H1", "H2")

    def bucket(x, edges, labels):
        return pd.cut(x, edges, labels=labels, right=False)
    D["A_spy_gap"] = bucket(D.gap, [-1, -0.01, 0, 0.01, 1], ["< -1%", "-1..0%", "0..+1%", "> +1%"])
    D["B_prev_ret"] = bucket(D.prev_ret, [-1, -0.01, 0, 1], ["down > 1%", "down 0..1%", "up"])
    D["C_dist200"] = bucket(-D.dist200, [0, 0.03, 0.07, 1], ["0-3% below", "3-7% below", "> 7% below"])
    D["X_spy_oc_LOOKAHEAD"] = np.where(D.oc < 0, "SPY falls open->close", "SPY rises open->close")

    def stats(sub, rest, name, var):
        r = sub.ret
        diff = r.mean() - rest.ret.mean()
        se = math.sqrt(r.var(ddof=1) / len(r) + rest.ret.var(ddof=1) / len(rest)) if len(r) > 2 and len(rest) > 2 else np.nan
        return {"variable": var, "bucket": name, "days": len(r), "mean_bp": 1e4 * r.mean(), "t": tstat(r),
                "hit": (r > 0).mean(), "median_bp": 1e4 * r.median(),
                "H1_bp": 1e4 * sub[sub.half == "H1"].ret.mean(), "H1_n": int((sub.half == "H1").sum()),
                "H2_bp": 1e4 * sub[sub.half == "H2"].ret.mean(), "H2_n": int((sub.half == "H2").sum()),
                "diff_vs_rest_bp": 1e4 * diff, "diff_t": diff / se if se else np.nan,
                "oco_mean_bp": 1e4 * sub.ret_oco.mean(), "oco_t": tstat(sub.ret_oco)}

    out = [stats(D, D.iloc[:0], "ALL cash days (baseline)", "-")]
    out[0]["diff_vs_rest_bp"] = out[0]["diff_t"] = np.nan
    for var in ("A_spy_gap", "B_prev_ret", "C_dist200", "X_spy_oc_LOOKAHEAD"):
        for b in (D[var].cat.categories if hasattr(D[var], "cat") else sorted(D[var].unique())):
            m = D[var] == b
            if m.sum():
                out.append(stats(D[m], D[~m], str(b), var))
    R = pd.DataFrame(out)
    # continuous slopes (one regression per variable), daily series
    slopes = {}
    for k in ("gap", "prev_ret", "dist200", "oc"):
        x, yv = D[k].to_numpy(), D.ret.to_numpy()
        xd = x - x.mean(); g = xd @ (yv - yv.mean()) / (xd @ xd)
        e = yv - yv.mean() - g * xd
        se = math.sqrt(e @ e / (len(x) - 2) / (xd @ xd))
        slopes[k] = (g, g / se, np.corrcoef(x, yv)[0, 1])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"cash days traded {len(D)} ({D.index[0]}..{D.index[-1]}), mean picks {D.n.mean():.2f}")
    print(R.round(2).to_string(index=False))
    print("\nslopes of basket return on SPY variable (bp of basket per 1% of variable, OLS t, corr):")
    for k, (g, tt, r) in slopes.items():
        print(f"  {k:9s} {g*100:+7.1f} bp/1%  t {tt:+.2f}  corr {r:+.2f}" + ("   [LOOK-AHEAD, descriptive]" if k == "oc" else ""))
    last = D.tail(6)[["n", "ret", "gap", "prev_ret", "dist200", "oc"]]
    print("\nmost recent cash days in sample:\n", last.round(4).to_string())
    R.to_csv("trader/data/gapdown_conditioning.csv", index=False)
    json.dump({k: [float(a) for a in v] for k, v in slopes.items()}, open("trader/data/gapdown_conditioning_slopes.json", "w"))
    return D, R, slopes


if __name__ == "__main__":
    main()
