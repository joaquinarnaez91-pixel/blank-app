"""Shared helpers: load cached bars as a pandas frame indexed by New York time, regular session only."""
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from pathlib import Path
NY = ZoneInfo("America/New_York")
ROOT = Path("trader/data")


def frame(sym, interval):
    p = ROOT / interval / (sym.upper().replace(".", "-") + ".npz")
    if not p.exists():
        return None
    z = np.load(p)
    idx = pd.to_datetime(z["t"], unit="s", utc=True).tz_convert(NY)
    df = pd.DataFrame({"o": z["o"], "h": z["h"], "l": z["l"], "c": z["c"], "v": z["v"]}, index=idx)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    mins = df.index.hour * 60 + df.index.minute
    if interval in ("5m", "15m", "30m", "1h"):
        df = df[(mins >= 570) & (mins < 960)]          # 9:30 <= bar start < 16:00
    df["date"] = df.index.date
    df["tod"] = df.index.hour * 60 + df.index.minute  # minutes after midnight, NY
    return df
