"""Build an aligned daily panel (dates x symbols) from the cached Yahoo bars and dividend/split events.

  python3 tools/research/xsec_swing/panel.py      # writes trader/data/xsec_panel.npz
Symbols: the ETFs in fetch.ETFS + every cached name in trader/data/superset.json or trader/universe.txt
(today's listings, including those now below $5 or thinly traded), kept if their 60-day mean dollar volume ever
reached $20M (a pre-filter far below the point-in-time top-500 cut-off; checked in common.py).
Arrays: O, H, L, C, V (split-adjusted, as Yahoo serves them), D (cash dividend per share on its ex-date,
split-adjusted), F (split factor: actual traded price = adjusted price * F). Calendar = SPY trading days of the
10-year window. Bars with zero volume or broken OHLC (open or close outside [low, high] by > 1%) are set to NaN.
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
MIN_MAX_ADV = 20e6


def ny_dates(t):
    return pd.to_datetime(t, unit="s", utc=True).tz_convert(NY).date


def build():
    norm = lambda s: s.upper().replace(".", "-")
    stocks = [norm(x["symbol"]) for x in json.load(open(ROOT / "superset.json"))]
    stocks += [norm(l.split("|")[0].strip()) for l in open("trader/universe.txt") if l.strip() and not l.startswith("#")]
    syms = [s for s in dict.fromkeys(ETFS + stocks) if (ROOT / "1d" / (s + ".npz")).exists()]
    spy = np.load(ROOT / "1d" / "SPY.npz")
    dates = np.array(sorted(set(ny_dates(spy["t"]))))
    dates = dates[dates >= pd.Timestamp("2016-11-21").date()]       # the 10-year cross-section
    di = {d: i for i, d in enumerate(dates)}
    T = len(dates)
    cols = {k: [] for k in "OHLCVDF"}
    kept, bad = [], {"zero_vol": 0, "broken_ohlc": 0, "dropped_small": 0}
    for s in syms:
        z = np.load(ROOT / "1d" / (s + ".npz"))
        ds = ny_dates(z["t"])
        rows = np.array([di.get(d, -1) for d in ds])
        m = rows >= 0
        a = {}
        for k, f in zip("OHLCV", "ohlcv"):
            a[k] = np.full(T, np.nan); a[k][rows[m]] = z[f][m]
        O, H, L, C, V = (a[k] for k in "OHLCV")
        zero = (V <= 0) & ~np.isnan(C)
        broken = ((O > H * 1.01) | (O < L * 0.99) | (C > H * 1.01) | (C < L * 0.99) | (O <= 0) | (C <= 0)) & ~np.isnan(C)
        for k in "OHLCV":
            a[k][zero | broken] = np.nan
        adv = pd.Series(a["C"] * a["V"]).rolling(60, min_periods=40).mean().max()
        if s not in ETFS and not (adv >= MIN_MAX_ADV):
            bad["dropped_small"] += 1
            continue
        bad["zero_vol"] += int(zero.sum()); bad["broken_ohlc"] += int(broken.sum())
        a["D"] = np.zeros(T); a["F"] = np.ones(T)
        e = ROOT / "1d_events" / (s + ".json")
        if e.exists():
            ev = json.load(open(e))
            for ts, amt in (ev.get("div") or {}).items():
                d = ny_dates(np.array([int(ts)]))[0]
                if d in di and amt:
                    a["D"][di[d]] += float(amt)
            for ts, (num, den) in (ev.get("split") or {}).items():
                if not num or not den:
                    continue
                d = ny_dates(np.array([int(ts)]))[0]
                before = np.array([x < d for x in dates])
                a["F"][before] *= float(num) / float(den)
        for k in "OHLCVDF":
            cols[k].append(a[k])
        kept.append(s)
    A = {k: np.stack(v, axis=1) for k, v in cols.items()}
    syms = np.array(kept)
    np.savez_compressed(OUT, dates=np.array([str(d) for d in dates]), syms=syms, is_etf=np.isin(syms, ETFS), **A)
    print("panel", A["C"].shape, "dates", dates[0], dates[-1], bad)


def load():
    z = np.load(OUT, allow_pickle=False)
    P = {k: z[k] for k in z.files}
    P["dates"] = pd.to_datetime(P["dates"])
    return P


if __name__ == "__main__":
    build()
