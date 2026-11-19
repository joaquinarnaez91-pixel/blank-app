"""Remove monthly/quarterly bars that a range=max fetch merged into trader/data/1d/ (Yahoo returns 1mo/3mo
bars for range=max even when interval=1d). Keeps only the regular daily bars. Run from the repo root.

  python3 tools/research/overnight/repair_cache.py SYM ...
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("trader/data/1d")


def repair(sym):
    p = ROOT / (sym.upper().replace(".", "-") + ".npz")
    if not p.exists():
        return sym, "missing"
    z = np.load(p)
    d = {k: z[k] for k in z.files}
    t = pd.to_datetime(d["t"], unit="s", utc=True).tz_convert("America/New_York")
    hh, mm, day = t.hour.values, t.minute.values, t.day.values
    monthly = ((hh == 0) | (hh == 1)) & (mm == 0) & (day == 1)
    trailing = hh >= 16
    keep = ~(monthly | trailing)
    if keep.sum() == 0:
        os.remove(p)
        return sym, f"deleted (all {len(keep)} bars were monthly/quarterly)"
    tods = pd.Series((hh * 60 + mm)[keep]).value_counts()
    out = {k: (v[keep] if (hasattr(v, "shape") and v.shape == d["t"].shape) else v) for k, v in d.items()}
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)
    return sym, f"kept {keep.sum()} dropped {(~keep).sum()} tods {tods.to_dict()}"


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print(*repair(s))
