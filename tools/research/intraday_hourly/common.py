"""Shared helpers for the hourly intraday program (agent-intraday-hourly).

Hourly Yahoo bars start 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 (last bar 30 min). A FULL session has
exactly these 7 bars; early-close days (13:00 close) and sessions with missing bars are treated as flat days.
Bar index k = 0..6 <-> start time CHK[k]. "10:30 open" = o[:, 1]; "close" = c[:, 6] (last regular-session trade).
"""
import sys, math, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
from bars import frame
from stats import deflated_sharpe

CHK = [570, 630, 690, 750, 810, 870, 930]
OUT = Path("trader/data/intraday_hourly")
OUT.mkdir(parents=True, exist_ok=True)
AUTHOR = "agent-intraday-hourly"
CODE = "tools/research/intraday_hourly/"


def hourly_pivot(sym):
    """Return (dates, dict field -> array[n_days, 7]) for FULL sessions of sym; None if missing."""
    df = frame(sym, "1h")
    if df is None or len(df) == 0:
        return None
    df = df[df.tod.isin(CHK)]
    cnt = df.groupby("date").tod.agg(lambda s: tuple(s) == tuple(CHK))
    df = df[df.date.isin(cnt[cnt].index)]
    if len(df) == 0:
        return None
    out = {}
    for f in ("o", "h", "l", "c", "v"):
        out[f] = df.pivot_table(index="date", columns="tod", values=f).reindex(columns=CHK)
    dates = out["o"].index
    return dates, {f: out[f].to_numpy(float) for f in out}


def nw_t(x, lags=None):
    """Newey-West (Bartlett) t-stat of the mean of a 1-d series."""
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 10:
        return float("nan")
    if lags is None:
        lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(s, 1e-30) / n)
    return float(x.mean() / se)


def moments(r):
    r = np.asarray(pd.Series(r).dropna(), float)
    m, sd = r.mean(), r.std(ddof=0)
    if sd == 0:
        return 0.0, 3.0
    z = (r - m) / sd
    return float((z ** 3).mean()), float((z ** 4).mean())


def evaluate(r, name="", n_trials=1, sr_var=None, per_year=252):
    """Daily strategy return series (0 = flat day), indexed by date. Returns a dict of the standard numbers:
    mean (bp/day), active-day mean, plain and Newey-West t, annualized Sharpe, deflated-Sharpe probability
    (given n_trials and the cross-trial variance of annualized Sharpe), first/second-half means and t."""
    r = pd.Series(r).astype(float).fillna(0.0)
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd * math.sqrt(per_year) if sd > 0 else 0.0
    t = mu / sd * math.sqrt(n) if sd > 0 else 0.0
    act = r[r != 0]
    h = n // 2
    a, b = r.iloc[:h], r.iloc[h:]
    ta = a.mean() / a.std(ddof=1) * math.sqrt(len(a)) if a.std(ddof=1) > 0 else 0.0
    tb = b.mean() / b.std(ddof=1) * math.sqrt(len(b)) if b.std(ddof=1) > 0 else 0.0
    sk, ku = moments(r)
    if sr_var is None:
        sr_var = 0.5 ** 2   # fallback: dispersion of annualized SR across null trials (conservative-ish)
    p, emax = deflated_sharpe(sr, n, max(1, n_trials), sr_var, sk, ku) if sd > 0 else (float("nan"), float("nan"))
    eq = (1 + r).cumprod()
    return {"name": name, "days": n, "active": int((r != 0).sum()), "mean_bp": 1e4 * mu,
            "active_mean_bp": 1e4 * act.mean() if len(act) else 0.0, "hit": float((act > 0).mean()) if len(act) else float("nan"),
            "t": t, "nw_t": nw_t(r), "sharpe": sr, "dsr_p": p, "sr_null_max": emax, "trials": n_trials,
            "h1_bp": 1e4 * a.mean(), "h1_t": ta, "h2_bp": 1e4 * b.mean(), "h2_t": tb,
            "maxdd": float((eq / eq.cummax() - 1).min()), "ann": float(eq.iloc[-1] ** (per_year / n) - 1) if n else float("nan")}


def fmt(s):
    return ("{name:<58} n {days:4d} act {active:4d} mean {mean_bp:+6.2f}bp act {active_mean_bp:+7.1f}bp hit {hit:5.1%} "
            "t {t:+5.2f} nw {nw_t:+5.2f} SR {sharpe:+5.2f} DSR {dsr_p:4.2f} | H1 {h1_bp:+6.2f} ({h1_t:+4.1f}) "
            "H2 {h2_bp:+6.2f} ({h2_t:+4.1f})").format(**s)


def passes(s, prior=False):
    """STANDARDS hurdles: t > 3 (or > 2 with a strong published prior), DSR prob > 0.9, same sign both halves."""
    tt = min(s["t"], s["nw_t"]) if not math.isnan(s["nw_t"]) else s["t"]
    return bool(((tt > 3) or (prior and tt > 2)) and s["dsr_p"] > 0.9 and s["h1_bp"] > 0 and s["h2_bp"] > 0)


def log_exp(**kw):
    from registry import log
    kw.setdefault("author", AUTHOR)
    return log(**kw)
