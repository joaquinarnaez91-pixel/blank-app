"""Index gap-down 'gap fill' day trade on long daily history (SPY 1996+, QQQ 1999+, IWM 2000+) and on the
hourly 3x-fund bars (2024-26), with the robustness checks the standards ask for.

Rule family (long only, one 3x bull fund per index, 1/3 of the capital each when traded as a basket):
  if index open / prior close - 1 < -theta: buy the bull fund at the 9:30 open;
  sell with a limit at the fund's prior close (= index back at its prior close) if the index trades through
  it by `thru` (5 bp, so a touch is not counted as a fill); otherwise sell at the close.
  Optional protective stop on the fund at -S from entry (stop-first if the day's low reaches it: pessimistic).
Daily-bar version: 3x fund approximated from the index with the daily reset; 5 bp per round trip.
Checks: thresholds, with/without the fill exit, with a stop, by period, by 200-day regime (the overnight model is
in cash below it), and the actual-fund hourly version with stop-first on hourly lows.
Run: python3 tools/research/intraday_hourly/gapfill_long.py [--log]
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from common import evaluate, fmt, log_exp, hourly_pivot, CODE

COST = 0.0005
START = {"SPY": "1996-01-01", "QQQ": "1999-06-01", "IWM": "2000-07-01"}
PERIODS = [("1996-2007", "1996-01-01", "2008-01-01"), ("2008-2015", "2008-01-01", "2016-01-01"),
           ("2016-2023", "2016-01-01", "2023-12-21"), ("2024-2026", "2023-12-21", "2027-01-01")]


def fund(P, O, pC, lev=3):
    return (1 + lev * (P / pC - 1)) / (1 + lev * (O / pC - 1)) - 1


def daily(sym):
    d = frame(sym, "1d")
    d = d[~d.index.duplicated()].set_index("date")
    d = d[(d.v > 0) & (d.o > 0)]
    d = d.assign(pC=d.c.shift(1), below=(d.c < d.c.rolling(200).mean()).shift(1).fillna(False).astype(bool))
    d = d.iloc[201:]
    d = d[d.index >= pd.Timestamp(START[sym]).date()]
    return d


def rule(d, theta, exit_kind="fill", thru=0.0005, stop=None):
    gap = d.o / d.pC - 1
    sig = gap < -theta
    r_close = fund(d.c, d.o, d.pC)
    lim = d.pC * (1 + thru)
    filled = d.h >= lim
    r_fill = fund(d.pC, d.o, d.pC)
    r = r_close.copy()
    if exit_kind == "fill":
        r = np.where(filled, r_fill, r_close)
    if stop is not None:                      # fund stop at -stop from entry; index level that triggers it
        idx_stop = d.pC * (1 + ((1 + 3 * (d.o / d.pC - 1)) * (1 - stop) - 1) / 3)
        hit = d.l <= idx_stop
        r = np.where(hit, -stop, r)           # stop-first (the day's low may have come before the fill)
    return pd.Series(np.where(sig, r - COST, 0.0), index=d.index)


def hourly_version(theta, stop=None, thru=0.0005):
    """Actual fund bars: buy the bull fund at its 9:30 open when the index gaps below -theta; limit at the fund's
    prior close (+thru); optional stop at -stop (stop-first within each hourly bar); else the close."""
    out = {}
    for idx, bull in (("SPY", "UPRO"), ("QQQ", "TQQQ"), ("IWM", "TNA")):
        di, I = hourly_pivot(idx); db, B = hourly_pivot(bull)
        dd = frame(idx, "1d"); dd = dd[~dd.index.duplicated()].set_index("date").c
        fb = frame(bull, "1h").groupby("date").c.last()
        dates = sorted(set(di) & set(db))
        ii = pd.Index(di).get_indexer(dates); ib = pd.Index(db).get_indexer(dates)
        pC = dd.shift(1).reindex(dates).to_numpy(); fpC = fb.shift(1).reindex(dates).to_numpy()
        gap = I["o"][ii, 0] / pC - 1
        o, h, l, c = (B[k][ib] for k in "ohlc")
        entry = o[:, 0]; target = fpC * (1 + thru)
        r = np.zeros(len(dates))
        for j in range(len(dates)):
            if not (gap[j] < -theta) or not np.isfinite(fpC[j]):
                continue
            ex = c[j, 6]
            for k in range(7):
                if stop is not None and l[j, k] <= entry[j] * (1 - stop):
                    ex = min(entry[j] * (1 - stop), o[j, k]) if k > 0 else entry[j] * (1 - stop); break
                if h[j, k] >= target[j]:
                    ex = fpC[j]; break
            r[j] = ex / entry[j] - 1 - COST
        out[idx] = pd.Series(r, index=pd.Index(dates))
    return out


def main(do_log=False):
    D = {s: daily(s) for s in ("SPY", "QQQ", "IWM")}
    variants = []
    for theta in (0.005, 0.010, 0.015, 0.020):
        for exit_kind in ("fill", "close"):
            for stop in (None, 0.10):
                variants.append((theta, exit_kind, stop))
    rows, keep = [], {}
    for (theta, ek, st) in variants:
        per = {s: rule(D[s], theta, ek, stop=st) for s in D}
        ew = pd.concat(per, axis=1).fillna(0.0).mean(axis=1)          # 1/3 per index; missing early QQQ/IWM = cash
        per["EW3"] = ew
        for s, r in per.items():
            keep[(theta, ek, st, s)] = r
    n_tr = len(variants) * 4
    srs = [r.mean() / r.std() * math.sqrt(252) for r in keep.values()]
    var = float(np.var(srs, ddof=1))
    print(f"{len(variants)} rules x 4 series = {n_tr} trials; SR variance {var:.3f}")
    for (theta, ek, st, s), r in keep.items():
        lab = f"gap<-{theta:.1%} bull, exit {ek}{', stop 10%' if st else ''} [{s}]"
        e = evaluate(r, lab, n_tr, var)
        per = {p: r[(r.index >= pd.Timestamp(a).date()) & (r.index < pd.Timestamp(b).date())] for p, a, b in PERIODS}
        below = pd.concat([D[x].below for x in D], axis=1).any(axis=1) if s == "EW3" else D[s].below
        rb = r[below.reindex(r.index).fillna(False).to_numpy(bool)]
        ra = r[~below.reindex(r.index).fillna(False).to_numpy(bool)]
        row = {"theta": theta, "exit": ek, "stop": st or 0, "idx": s, "days": e["days"], "trades": e["active"],
               "mean_bp": e["mean_bp"], "per_trade_bp": e["active_mean_bp"], "hit": e["hit"], "t": e["t"], "nw_t": e["nw_t"],
               "SR": e["sharpe"], "DSR": e["dsr_p"], "h1": e["h1_bp"], "h2": e["h2_bp"],
               **{f"{p}_bp": 1e4 * x.mean() for p, x in per.items()},
               **{f"{p}_t": x.mean() / x.std() * math.sqrt(len(x)) if x.std() > 0 else 0 for p, x in per.items()},
               "below_bp": 1e4 * rb.mean(), "below_t": rb.mean() / rb.std() * math.sqrt(len(rb)) if rb.std() > 0 else 0,
               "below_trades": int((rb != 0).sum()), "above_bp": 1e4 * ra.mean(),
               "above_t": ra.mean() / ra.std() * math.sqrt(len(ra)) if ra.std() > 0 else 0}
        rows.append(row)
    G = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 40)
    cols = ["theta", "exit", "stop", "idx", "trades", "mean_bp", "per_trade_bp", "hit", "t", "nw_t", "SR", "DSR", "h1", "h2"]
    print(G[cols].round(3).to_string(index=False))
    print("\nBy period (bp/day, t) and by the 200-day regime of the prior close:")
    pc = ["theta", "exit", "stop", "idx"] + [f"{p}_bp" for p, _, _ in PERIODS] + [f"{p}_t" for p, _, _ in PERIODS] + \
         ["below_trades", "below_bp", "below_t", "above_bp", "above_t"]
    print(G[pc].round(2).to_string(index=False))
    # actual 3x fund bars, 2024-26
    print("\nHourly bars, actual funds (UPRO/TQQQ/TNA), 2024-2026, 5 bp, stop-first on hourly lows:")
    hrows = []
    for theta in (0.005, 0.010, 0.015, 0.020):
        for st in (None, 0.10):
            per = hourly_version(theta, st)
            per["EW3"] = pd.concat(per, axis=1).fillna(0.0).mean(axis=1)
            for s, r in per.items():
                e = evaluate(r, f"[hourly] gap<-{theta:.1%} bull fill{', stop 10%' if st else ''} [{s}]", n_tr, var)
                print(fmt(e)); hrows.append({"theta": theta, "stop": st or 0, "idx": s, **e})
    H = pd.DataFrame(hrows)
    G.to_csv("trader/data/intraday_hourly/gapfill_long.csv", index=False)
    H.to_csv("trader/data/intraday_hourly/gapfill_hourly.csv", index=False)
    if do_log:
        b = G[(G.theta == 0.010) & (G.exit == "fill") & (G.stop == 0) & (G.idx == "EW3")].iloc[0]
        bs = G[(G.theta == 0.010) & (G.exit == "fill") & (G.stop == 0) & (G.idx == "SPY")].iloc[0]
        log_exp(family="index_daytime_hourly", name="Index gap-down fill (buy 3x bull at open, sell at prior close or close), long history",
                hypothesis="After an overnight index gap below -theta, the index tends to retrace toward the prior close "
                           "intraday; a 3x bull fund bought at the open and sold at the fill/close earns > costs",
                data="Yahoo daily SPY 1996+, QQQ 1999+, IWM 2000+ to Nov 2026 (synthetic 3x, daily reset); hourly "
                     "UPRO/TQQQ/TNA 2024-26 (actual fund bars)",
                method="theta {0.5,1,1.5,2}% x exit {fill-or-close, close} x stop {none, 10% fund}; 4 series each; "
                       "fill needs a 5 bp trade-through; stop-first; by period and 200-day regime; 5 bp costs",
                variants=n_tr,
                metrics={"EW3_1pct_mean_bp": float(b.mean_bp), "EW3_1pct_t": float(b.t), "EW3_1pct_SR": float(b.SR),
                         "EW3_1pct_DSR": float(b.DSR), "SPY_1pct_t": float(bs.t), "SPY_1pct_SR": float(bs.SR)},
                verdict="rejected", notes="t < 2 everywhere with a 5 bp trade-through fill; the t 3.5 seen with touch fills came from 31 of 378 SPY days whose high stopped within 5 bp of the prior close (fill-at-the-high illusion). "
                      "Not stronger below the 200-day. Report: tools/research/reports/intraday_hourly.md",
                code=CODE + "gapfill_long.py")
    return G, H


if __name__ == "__main__":
    main("--log" in sys.argv)
