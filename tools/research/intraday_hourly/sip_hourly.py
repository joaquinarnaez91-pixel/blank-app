"""(b) Stocks in play at hourly granularity (60-minute opening range), long only, ~710 sessions.

Each day, point in time (daily inputs through t-1, first-hour inputs known at 10:30):
  eligible: prior close >= $5, 14-day mean volume >= 1M shares, ATR14 >= $0.50, full hourly session,
            first-hour relative volume RV = 9:30-bar volume / its 14-day mean >= 1, first-hour candle bullish.
  rank by RV, take the top K whose entry price fits a whole share in a $1,000/K slot.
Entries: E1 = buy at the 10:30 open.  E2 = buy-stop one cent above the first-hour high, triggered 10:30-15:30
         (fills at max(level, bar open)); untriggered slots stay in cash.
Stops:   first-hour low (checked on hourly lows, stop-first, also inside the entry bar) | none.
Exit:    the close (last regular-session trade).  Costs: 10 bp per round trip (20 bp sensitivity).
Sizing:  whole shares; loss at the stop capped at $100 per trade and $200 per day.
Run: python3 tools/research/intraday_hourly/sip_hourly.py [--log]
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from panel import load, features
from sim import topk, stop_exit
from common import evaluate, fmt, passes, log_exp, regime_below, cash_fraction, OUT, CODE

COST = 0.0010
WARM = 20


def main(do_log=False):
    P = load(); F = features(P)
    o, h, l, c = (P[f].astype(float) for f in "ohlc")
    dates = pd.to_datetime(pd.Series(P["dates"])).dt.date.to_numpy()
    N, T = F["y"].shape
    full = F["full"]
    bull = c[:, :, 0] > o[:, :, 0]
    base = (full & (F["prevC"] >= 5) & (F["adv"] >= 1e6) & (F["atr"] >= 0.5) & np.isfinite(F["rv"]) & (F["rv"] >= 1)
            & bull)
    base[:, :WARM] = False
    stop = l[:, :, 0]
    # E1: 10:30 open
    e1 = o[:, :, 1]
    e1_ok = base & (e1 > stop)
    x1s, _ = stop_exit(o, h, l, c, np.ones((N, T), int), e1, stop)
    x1n = c[:, :, 6]
    # E2: buy-stop above the first-hour high
    level = np.round(h[:, :, 0] + 0.01, 2)
    ebar = np.full((N, T), 99)
    for k in range(5, 0, -1):          # first bar 1..5 whose high reaches the level
        ebar = np.where(h[:, :, k] >= level, k, ebar)
    trig = ebar < 99
    kk = np.clip(ebar, 0, 6)
    o_at = np.take_along_axis(o, kk[:, :, None], 2)[:, :, 0]
    e2 = np.where(trig, np.maximum(level, o_at), np.nan)
    x2s, _ = stop_exit(o, h, l, c, np.where(trig, ebar, 99), e2, stop)
    x2n = c[:, :, 6]
    rv = np.where(np.isfinite(F["rv"]), F["rv"], -1)
    print(f"days {T - WARM} ({dates[WARM]} .. {dates[-1]}), stocks {N}; eligible stock-days {int(base.sum())}, "
          f"per day median {int(np.median(base.sum(0)[WARM:]))}")

    below = regime_below(dates, "SPY")
    cf = cash_fraction(dates)
    grid = []
    for K in (1, 3, 5, 10):
        for ename, entry, ok, xs, xn in (("E1 10:30 open", e1, e1_ok, x1s, x1n),
                                          ("E2 buy-stop > 1st-hour high", e2, base, x2s, x2n)):
            for sname, xp, st in (("stop 1st-hour low", xs, stop), ("no stop", xn, None)):
                # E2: rank at 10:30 on the level price; trade only when triggered (exit NaN -> cash)
                ent = np.where(np.isfinite(entry), entry, level) if ename.startswith("E2") else entry
                xx = np.where(np.isfinite(entry), xp, np.nan)
                w, fr, npos = topk(rv, ok, ent, xx, dates, K, stop=st, cost=COST)
                w20, _, _ = topk(rv, ok, ent, xx, dates, K, stop=st, cost=0.0020)
                grid.append({"K": K, "entry": ename, "stop": sname, "whole": w.iloc[WARM:], "frac": fr.iloc[WARM:],
                             "w20": w20.iloc[WARM:], "npos": npos.iloc[WARM:]})
    srs = [g["whole"].mean() / g["whole"].std(ddof=1) * math.sqrt(252) for g in grid]
    var = float(np.var(srs, ddof=1))
    n_tr = len(grid)
    n_fam = n_tr + 40                    # + the 40 variants of the 5-minute stocks-in-play study (same idea)
    print(f"\n{n_tr} variants (family total with the 5-minute study: {n_fam}); SR variance across variants {var:.2f}")
    rows = []
    for g in grid:
        name = f"K={g['K']} {g['entry']}, {g['stop']}"
        s = evaluate(g["whole"], name, n_fam, var)
        s20 = evaluate(g["w20"], name, n_fam, var)
        sf = evaluate(g["frac"], name, n_fam, var)
        wb = g["whole"][below[WARM:]]
        s.update({"K": g["K"], "entry": g["entry"], "stop": g["stop"], "net20_bp": s20["mean_bp"], "net20_t": s20["t"],
                  "frac_bp": sf["mean_bp"], "frac_t": sf["t"], "avg_pos": g["npos"].mean(),
                  "below200_bp": 1e4 * wb.mean(), "below200_t": wb.mean() / wb.std(ddof=1) * math.sqrt(len(wb)) if wb.std() > 0 else 0,
                  "pass": passes(s)})
        rows.append(s)
        print(fmt(s), "| 20bp", f"{s20['mean_bp']:+.2f}", "| frac", f"{sf['mean_bp']:+.2f} t {sf['t']:+.2f}",
              f"| SPY<200d {s['below200_bp']:+.1f}bp t {s['below200_t']:+.2f}", " PASS" if s["pass"] else "")
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "sip_hourly_grid.csv", index=False)
    # Control: does relative volume matter? E1, no stop, per trade, all eligible bullish names by RV bucket
    yb = (c[:, :, 6] / e1 - 1 - COST)
    m = base & np.isfinite(yb)
    rvv = F["rv"][m]; yy = yb[m]
    bk = pd.cut(rvv, [1, 1.5, 2, 3, 5, 1e9], labels=["1-1.5", "1.5-2", "2-3", "3-5", "5+"])
    ctl = pd.DataFrame({"b": bk, "y": yy}).groupby("b", observed=True).y.agg(["size", "mean", lambda x: (x > 0).mean()])
    ctl.columns = ["n", "mean_bp", "hit"]; ctl["mean_bp"] *= 1e4
    print("\nRV control, E1 no stop, per trade net of 10 bp (pooled, descriptive only):")
    print(ctl.round(2).to_string())
    # same, daily-series version: equal-weight all eligible names with RV >= 3 vs 1-1.5
    best = G.loc[G.sharpe.idxmax()]
    print(f"\nBest: {best['name']}: mean {best.mean_bp:+.2f}bp/day t {best.t:+.2f} SR {best.sharpe:+.2f} "
          f"DSR p {best.dsr_p:.2f} (trials {n_fam}); halves {best.h1_bp:+.2f} / {best.h2_bp:+.2f}")
    if do_log:
        log_exp(family="stocks_in_play", name="Hourly stocks-in-play (60-min range), long only, top-K by RV, whole shares",
                hypothesis="High first-hour relative volume + bullish first hour predicts a positive 10:30->close drift "
                           "(Zarattini-Barbon-Aziz style ORB on stocks in play), 730 sessions instead of 45",
                data=f"Yahoo 1h, {N} stocks of trader/data/superset.json, {dates[WARM]}..{dates[-1]} ({T-WARM} days)",
                method="K in {1,3,5,10} x entry {10:30 open, buy-stop above 1st-hour high} x stop {1st-hour low, none}; "
                       "whole shares in $1000/K slots, $100/$200 loss caps, 10 bp (and 20 bp) costs; daily-series t",
                variants=n_tr,
                metrics={"best_SR": float(best.sharpe), "best_t": float(best.t), "best_mean_bp": float(best.mean_bp),
                         "DSR_prob": float(best.dsr_p), "n_pass": int(G["pass"].sum()),
                         "median_SR": float(G.sharpe.median())},
                verdict="promising" if G["pass"].any() else "rejected",
                notes=f"best variant {best['name']}; halves {best.h1_bp:+.1f}/{best.h2_bp:+.1f} bp; DSR uses {n_fam} trials "
                      "(incl. 40 from the 5-minute study). Survivorship: universe = today's liquid listings.",
                code=CODE + "sip_hourly.py")
    return G


if __name__ == "__main__":
    main("--log" in sys.argv)
