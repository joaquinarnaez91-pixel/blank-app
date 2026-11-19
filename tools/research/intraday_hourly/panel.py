"""Build the stock x day x hour panel from the hourly cache and the point-in-time feature set.

  python3 tools/research/intraday_hourly/panel.py      -> trader/data/intraday_hourly/panel.npz

Calendar = SPY's hourly dates. Per stock and date: the 7 hourly bars (NaN when missing) and daily OHLCV
aggregated from all regular-session hourly bars of that date (early-close days included).
Features for day t (all known at 10:30 of day t; everything daily uses days <= t-1):
  gap        9:30 open / prior close - 1
  r1h        first-hour return: 10:30 price (close of the 9:30 bar) / 9:30 open - 1
  rv         first-hour volume / mean first-hour volume of the previous 14 sessions
  prev_ret   prior day close-to-close return
  ret5       5-day return to the prior close
  dist20     prior close / max(high of the previous 20 sessions) - 1
  atr_pct    ATR14 (previous 14 sessions) / prior close
  ldvol      log(14-day mean daily volume x prior close)
  mkt1h      SPY first-hour return (same for all stocks on a day); mktgap SPY gap
Targets: y (10:30 open -> close), y930 (9:30 open -> close).
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from common import CHK, OUT

ETFS = {"SPY", "QQQ", "IWM", "SMH", "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "TNA", "TZA", "FAS", "FAZ", "^VIX"}


def build():
    syms = [x["symbol"] for x in json.load(open("trader/data/superset.json"))]
    syms = [s for s in syms if s not in ETFS and (Path("trader/data/1h") / (s.upper().replace(".", "-") + ".npz")).exists()]
    spy = frame("SPY", "1h")
    cal = pd.Index(sorted(spy.date.unique()))
    T, N = len(cal), len(syms)
    bars = {f: np.full((N, T, 7), np.nan, np.float32) for f in "ohlcv"}
    daily = {f: np.full((N, T), np.nan) for f in ("dO", "dH", "dL", "dC", "dV")}
    t0 = time.time()
    keep = np.ones(N, bool)
    for n, s in enumerate(syms):
        df = frame(s, "1h")
        if df is None or len(df) < 500:
            keep[n] = False
            continue
        di = cal.get_indexer(df.date)
        ok = di >= 0
        df, di = df[ok], di[ok]
        g = df.groupby(di)
        idx = g.o.first().index.to_numpy()
        daily["dO"][n, idx] = g.o.first().to_numpy(); daily["dH"][n, idx] = g.h.max().to_numpy()
        daily["dL"][n, idx] = g.l.min().to_numpy(); daily["dC"][n, idx] = g.c.last().to_numpy()
        daily["dV"][n, idx] = g.v.sum().to_numpy()
        k = pd.Index(CHK).get_indexer(df.tod)
        m = k >= 0
        for f in "ohlcv":
            bars[f][n, di[m], k[m]] = df[f].to_numpy()[m]
        if n % 300 == 0:
            print("loaded", n, "of", N, f"{time.time()-t0:.0f}s", flush=True)
    syms = [s for s, k in zip(syms, keep) if k]
    bars = {f: a[keep] for f, a in bars.items()}
    daily = {f: a[keep] for f, a in daily.items()}
    # SPY (market) bars on the same calendar
    sp = {f: np.full((T, 7), np.nan) for f in "ohlcv"}
    di = cal.get_indexer(spy.date); k = pd.Index(CHK).get_indexer(spy.tod); m = (k >= 0) & (di >= 0)
    for f in "ohlcv":
        sp[f][di[m], k[m]] = spy[f].to_numpy()[m]
    spd = spy.groupby("date").c.last().reindex(cal).to_numpy()
    np.savez_compressed(OUT / "panel.npz", syms=np.array(syms), dates=np.array([str(d) for d in cal]),
                        **{f: bars[f] for f in "ohlcv"}, **daily,
                        **{"spy_" + f: sp[f] for f in "ohlcv"}, spy_dC=spd)
    print("panel", len(syms), "stocks x", T, "days", f"{time.time()-t0:.0f}s")


def load():
    z = np.load(OUT / "panel.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def rolling_mean_prev(a, w, minp):
    """a: (N, T). Mean over the previous w columns (t-w .. t-1), NaN-aware, min count minp."""
    df = pd.DataFrame(a.T)
    return df.rolling(w, min_periods=minp).mean().shift(1).to_numpy().T


def features(P):
    o, h, l, c, v = (P[f].astype(float) for f in "ohlcv")
    dO, dH, dL, dC, dV = (P[f] for f in ("dO", "dH", "dL", "dC", "dV"))
    N, T = dC.shape
    full = np.isfinite(o).all(2) & np.isfinite(c).all(2) & np.isfinite(h).all(2) & np.isfinite(l).all(2)
    prevC = np.concatenate([np.full((N, 1), np.nan), dC[:, :-1]], 1)
    prev2C = np.concatenate([np.full((N, 2), np.nan), dC[:, :-2]], 1)
    prev6C = np.concatenate([np.full((N, 6), np.nan), dC[:, :-6]], 1)
    F = {}
    F["gap"] = o[:, :, 0] / prevC - 1
    F["r1h"] = c[:, :, 0] / o[:, :, 0] - 1
    v0 = v[:, :, 0].copy(); v0[v0 <= 0] = np.nan
    F["rv"] = v0 / rolling_mean_prev(v0, 14, 10)
    F["prev_ret"] = prevC / prev2C - 1
    F["ret5"] = prevC / prev6C - 1
    hi20 = pd.DataFrame(dH.T).rolling(20, min_periods=15).max().shift(1).to_numpy().T
    F["dist20"] = prevC / hi20 - 1
    tr = np.fmax(dH - dL, np.fmax(np.abs(dH - prevC), np.abs(dL - prevC)))
    atr = rolling_mean_prev(tr, 14, 10)
    F["atr"] = atr
    F["atr_pct"] = atr / prevC
    adv = rolling_mean_prev(dV, 14, 10)
    F["adv"] = adv
    F["ldvol"] = np.log(adv * prevC)
    F["prevC"] = prevC
    F["v0avg"] = rolling_mean_prev(v0, 14, 10)
    spo, spc = P["spy_o"], P["spy_c"]
    spy_prev = np.concatenate([[np.nan], P["spy_dC"][:-1]])
    F["mkt1h"] = np.broadcast_to(spc[:, 0] / spo[:, 0] - 1, (N, T)).copy()
    F["mktgap"] = np.broadcast_to(spo[:, 0] / spy_prev - 1, (N, T)).copy()
    F["y"] = c[:, :, 6] / o[:, :, 1] - 1           # 10:30 open -> close
    F["y930"] = c[:, :, 6] / o[:, :, 0] - 1        # 9:30 open -> close
    F["mkt_y"] = np.broadcast_to(spc[:, 6] / spo[:, 1] - 1, (N, T)).copy()
    # data sanity: drop sessions with absurd prints (bar range > 60% of price, or |gap| > 60%)
    bad = (np.nanmax(h, 2) / np.nanmin(l, 2) > 1.6) | (np.abs(F["gap"]) > 0.6) | (np.abs(F["y"]) > 0.6)
    F["full"] = full & ~bad
    return F


if __name__ == "__main__":
    build()
