"""Evaluation helpers shared by the experiment scripts: OOS window stats vs the current spec, and a store of
every variant's nightly return series (for SPA / CSCV / DSR across the whole family)."""
import sys, math, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, validate  # noqa: E402

OOS = "2008-01-01"                 # common out-of-sample window start (pre-2008 = training for fitted models)
END = None
STORE = Path("trader/data/overnight"); STORE.mkdir(parents=True, exist_ok=True)
VAR_FILE = STORE / "variants.pkl"


def window(r, start=OOS, end=END):
    r = pd.Series(r).dropna()
    r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index < pd.Timestamp(end)]
    return r


def evaluate(name, r, r0, ff=None, start=OOS, boot=True):
    """Stats of variant r vs benchmark r0 over the window; both-halves; paired bootstrap of the SR difference."""
    a, b = window(r, start).align(window(r0, start), join="inner")
    s = core.stats(a); s0 = core.stats(b)
    h1, h2 = core.halves(a); g1, g2 = core.halves(b)
    out = {"name": name, **{k: s[k] for k in ("n", "mean_bp", "t", "sharpe", "cagr", "maxdd", "worst", "active")},
           "sr_h1": h1["sharpe"], "sr_h2": h2["sharpe"], "bench_sr": s0["sharpe"],
           "bench_h1": g1["sharpe"], "bench_h2": g2["sharpe"], "skew": s["skew"], "kurt": s["kurt"]}
    if ff is not None:
        f = window(ff.sum(axis=1), start)
        out["avg_fund_frac"] = float(f.mean()); out["max_fund_frac"] = float(f.max())
    if boot:
        pdiff = validate.paired_sr_diff(a.values, b.values, B=2000, b=20)
        out["d_sr"] = pdiff["d_sr"]; out["d_sr_ci"] = pdiff["ci"]; out["p_dsr"] = pdiff["p"]
    return out


def show(o):
    ff = f" fund {o['avg_fund_frac']:.2f}/{o['max_fund_frac']:.2f}" if "avg_fund_frac" in o else ""
    ds = (f" dSR {o['d_sr']:+.2f} [{o['d_sr_ci'][0]:+.2f},{o['d_sr_ci'][1]:+.2f}] p {o['p_dsr']:.2f}"
          if "d_sr" in o else "")
    return (f"{o['name']:<40} mean {o['mean_bp']:+5.2f}bp t {o['t']:+5.2f} SR {o['sharpe']:+5.2f} "
            f"({o['sr_h1']:+.2f}/{o['sr_h2']:+.2f}) CAGR {o['cagr']:+6.1%} DD {o['maxdd']:+6.1%} "
            f"worst {o['worst']:+6.2%} act {o['active']:.0%}{ff}{ds}")


def save_variants(group, series):
    """series: dict name -> pd.Series of nightly returns (full history). Stored under the group key."""
    d = pickle.load(open(VAR_FILE, "rb")) if VAR_FILE.exists() else {}
    d[group] = {k: pd.Series(v) for k, v in series.items()}
    pickle.dump(d, open(VAR_FILE, "wb"))


def load_variants():
    return pickle.load(open(VAR_FILE, "rb")) if VAR_FILE.exists() else {}
