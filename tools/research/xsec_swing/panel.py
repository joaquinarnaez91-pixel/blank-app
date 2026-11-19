"""Build an aligned daily panel (dates x symbols) from the cached Yahoo bars and dividend events.

  python3 tools/research/xsec_swing/panel.py      # writes trader/data/xsec_panel.npz
Arrays: O, H, L, C, V (split-adjusted, as Yahoo serves them), D (cash dividend per share on its ex-date,
split-adjusted). Calendar = SPY trading days. Missing bars stay NaN. Bars with zero volume or broken
OHLC (open or close outside [low, high] by > 1%) are set to NaN and counted.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
sys.path.insert(0, "tools/research/xsec_swing")
from fetch import ETFS
from bars import NY

ROOT = Path("trader/data")
OUT = ROOT / "xsec_panel.npz"


def ny_dates(t):
    return pd.to_datetime(t, unit="s", utc=True).tz_convert(NY).date


def build():
    stocks = [x["symbol"].upper().replace(".", "-") for x in json.load(open(ROOT / "superset.json"))]
    syms = [s for s in dict.fromkeys(ETFS + stocks)]
    spy = np.load(ROOT / "1d" / "SPY.npz")
    dates = np.array(sorted(set(ny_dates(spy["t"]))))
    di = {d: i for i, d in enumerate(dates)}
    T, N = len(dates), len(syms)
    A = {k: np.full((T, N), np.nan) for k in "OHLCVD"}
    A["D"][:] = 0.0
    bad = {"zero_vol": 0, "broken_ohlc": 0, "missing_file": 0}
    keep = np.ones(N, bool)
    for j, s in enumerate(syms):
        p = ROOT / "1d" / (s + ".npz")
        if not p.exists():
            bad["missing_file"] += 1; keep[j] = False; continue
        z = np.load(p)
        ds = ny_dates(z["t"])
        rows = np.array([di.get(d, -1) for d in ds])
        m = rows >= 0
        for k, f in zip("OHLCV", "ohlcv"):
            A[k][rows[m], j] = z[f][m]
        e = ROOT / "1d_events" / (s + ".json")
        if e.exists():
            ev = json.load(open(e))
            for ts, amt in (ev.get("div") or {}).items():
                d = ny_dates(np.array([int(ts)]))[0]
                if d in di and amt:
                    A["D"][di[d], j] += float(amt)
    O, H, L, C, V = (A[k] for k in "OHLCV")
    zero = (V <= 0) & ~np.isnan(C)
    broken = ((O > H * 1.01) | (O < L * 0.99) | (C > H * 1.01) | (C < L * 0.99) | (O <= 0) | (C <= 0)) & ~np.isnan(C)
    bad["zero_vol"] = int(zero.sum()); bad["broken_ohlc"] = int(broken.sum())
    for k in "OHLCV":
        A[k][zero | broken] = np.nan
    syms = np.array(syms)[keep]
    for k in A:
        A[k] = A[k][:, keep]
    np.savez_compressed(OUT, dates=np.array([str(d) for d in dates]), syms=syms,
                        is_etf=np.isin(syms, ETFS), **A)
    print("panel", A["C"].shape, "dates", dates[0], dates[-1], bad)


def load():
    z = np.load(OUT, allow_pickle=False)
    P = {k: z[k] for k in z.files}
    P["dates"] = pd.to_datetime(P["dates"])
    return P


if __name__ == "__main__":
    build()
