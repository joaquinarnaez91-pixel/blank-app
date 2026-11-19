"""Overnight program core: data, overnight returns (dividend-corrected), financing carry of 3x funds,
the current live spec, and evaluation helpers. Run scripts from the repository root.

Conventions
- Night t = close of trading day t-1 -> open of trading day t. Decision for night t is made at ~15:58 on
  day t-1, so every signal for night t may use data up to and including the close of day t-1 (as a proxy for
  the 15:58 price), and nothing later. In code: signals are computed on day t-1 and shifted by one row.
- Index overnight total return r_on[t] = (open[t] + div[t]) / close[t-1] - 1, where div[t] is the ETF's cash
  dividend with ex-date t. The 3x fund does not drop on the ETF's ex-date (it tracks 3x the index and earns
  the dividends through its swaps), so the dividend is added back.
- 3x fund overnight return = 3 * r_on[t] - carry[t] (the fund resets at the close, so the overnight leg is
  exactly 3x the index move; carry is the daily financing/fee accrual, which the price at the open already
  reflects -- verified empirically in tracking.py).
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
from bars import frame  # noqa: E402

DIV = Path("trader/data/div")
COST_RT = 0.0004  # round trip on the 3x fund dollars, overnight
# carry model fitted in tracking.py on UPRO/TQQQ/TNA 2009-2026 (per-night slope on rf: 2.22, 2.40, 2.63;
# intercepts ~0): 3x fund overnight return = 3 r_on - CARRY_A * rf * fwd_days / 365, where fwd_days = calendar
# days from the morning's date to the next trading date (the NAV struck that day books those days' accrual,
# so the Thursday->Friday night bears the weekend's financing; data: Friday mornings show ~2.5x the drag).
CARRY_A, CARRY_B = 2.4, 0.0


def daily(sym):
    d = frame(sym, "1d")
    if d is None:
        raise FileNotFoundError(sym)
    d = d.set_index(pd.DatetimeIndex(pd.to_datetime(d["date"])))[["o", "h", "l", "c", "v"]]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def divs(sym):
    p = DIV / (sym.upper().replace(".", "-") + ".csv")
    if not p.exists():
        return pd.Series(dtype=float)
    x = pd.read_csv(p, parse_dates=["exdate"])
    return x.groupby("exdate")["amount"].sum()


def index_frame(sym):
    """Daily frame with overnight/intraday/close-to-close TOTAL returns of an ETF."""
    d = daily(sym)
    dv = divs(sym).reindex(d.index).fillna(0.0)
    out = pd.DataFrame(index=d.index)
    out["o"], out["c"], out["h"], out["l"], out["v"] = d["o"], d["c"], d["h"], d["l"], d["v"]
    out["div"] = dv
    cp = d["c"].shift(1)
    out["r_on"] = (d["o"] + dv) / cp - 1
    out["r_on_raw"] = d["o"] / cp - 1
    out["r_id"] = d["c"] / d["o"] - 1
    out["r_cc"] = (d["c"] + dv) / cp - 1
    out["cal_days"] = pd.Series(d.index, index=d.index).diff().dt.days
    return out


def rf_series(index):
    """3-month T-bill rate (decimal, annual) known at each date's close (^IRX), forward filled."""
    x = daily("^IRX")["c"] / 100.0
    return x.reindex(index.union(x.index)).ffill().reindex(index)


def panel(syms=("SPY", "QQQ", "IWM"), start=None):
    """Aligned panel: dict of DataFrames r_on, r_cc, c, ... with columns = syms, on the common calendar."""
    fr = {s: index_frame(s) for s in syms}
    idx = None
    for s in syms:
        idx = fr[s].index if idx is None else idx.intersection(fr[s].index)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    P = {}
    for k in ("o", "c", "h", "l", "r_on", "r_on_raw", "r_id", "r_cc", "div"):
        P[k] = pd.DataFrame({s: fr[s][k].reindex(idx) for s in syms})
    cd = pd.Series(idx, index=idx).diff().dt.days
    P["cal_days"] = cd
    P["fwd_days"] = (-pd.Series(idx, index=idx).diff(-1).dt.days).fillna(1.0)
    P["rf"] = rf_series(idx).shift(1)  # rate known at the decision (previous close)
    return P


def carry(P, a=CARRY_A, b=CARRY_B):
    """Per-night financing drag of a 3x fund (fraction of fund value) for night t (close t-1 -> open t)."""
    return (a * P["rf"].fillna(0.0) + b) * P["fwd_days"].fillna(1.0) / 365.0


# ---------------------------------------------------------------- the current live spec

def sma(c, n):
    return c.rolling(n, min_periods=n).mean()


def spec_exposure(P, sma_n=200, vol_n=20, target=0.015, cap=3.0, vol=None, trend=None):
    """Index exposure multiple L (0..cap) per index for each NIGHT t (already shifted: uses data to close t-1).
    vol: optional DataFrame of overnight sigma forecasts known at close t-1 (same shape), else rolling std."""
    c, r_on = P["c"], P["r_on"]
    if trend is None:
        trend = (c > sma(c, sma_n)).astype(float)
    if vol is None:
        vol = r_on.rolling(vol_n, min_periods=vol_n).std()
    L = (target / vol).clip(upper=cap) * trend
    return L.shift(1)  # decided at close t-1, applies to night t


def sleeve_returns(P, L, weights=None, cost_rt=COST_RT, use_carry=True, carry_a=CARRY_A, carry_b=CARRY_B):
    """Sleeve return per night. L: index exposure multiple per leg (0..3). weights: sleeve fraction per leg
    (default equal). Dollars in the 3x fund for leg i = w_i * L_i / 3; fund return = 3 r_on - carry."""
    cols = list(L.columns)
    if weights is None:
        weights = pd.Series(1.0 / len(cols), index=cols)
    w = pd.Series(weights).reindex(cols)
    fund_frac = L.fillna(0.0).mul(w, axis=1) / 3.0           # fraction of sleeve in each 3x fund
    fund_ret = 3.0 * P["r_on"][cols]
    if use_carry:
        fund_ret = fund_ret.sub(carry(P, carry_a, carry_b), axis=0)
    r = (fund_frac * fund_ret).sum(axis=1) - cost_rt * fund_frac.sum(axis=1)
    return r, fund_frac


# ---------------------------------------------------------------- statistics

def stats(r, per_year=252):
    r = pd.Series(r).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    yrs = n / per_year
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if n else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    sk = float(((r - mu) ** 3).mean() / sd ** 3) if sd > 0 else float("nan")
    ku = float(((r - mu) ** 4).mean() / sd ** 4) if sd > 0 else float("nan")
    return {"n": n, "mean_bp": 1e4 * mu, "t": mu / sd * math.sqrt(n) if sd > 0 else float("nan"),
            "sharpe": mu / sd * math.sqrt(per_year) if sd > 0 else float("nan"),
            "cagr": cagr, "maxdd": float(dd), "worst": float(r.min()), "best": float(r.max()),
            "skew": sk, "kurt": ku, "active": float((r != 0).mean())}


def halves(r):
    r = pd.Series(r).dropna()
    h = len(r) // 2
    return stats(r.iloc[:h]), stats(r.iloc[h:])


def fmt(s, name=""):
    return (f"{name:<38} n {s['n']:5d} mean {s['mean_bp']:+6.2f}bp t {s['t']:+5.2f} SR {s['sharpe']:+5.2f} "
            f"CAGR {s['cagr']:+6.1%} maxDD {s['maxdd']:+6.1%} worst {s['worst']:+6.2%} act {s['active']:.0%}")


SPEC_START = "2017-09-01"   # the live spec's reported window (Sept 2017 - Nov 2026)
