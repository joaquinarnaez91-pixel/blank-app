"""Regime/trend signals and volatility forecasts for the overnight program.

Every function returns a DataFrame/Series indexed like the panel whose value on row d uses data up to and
including the close of day d only (the 15:58 decision proxy). Exposure functions shift by one row so the value
on row d applies to the night d -> d+1. Fitted models are refit walk-forward on data strictly before each
refit date (Jan 1 each year) and only forecast after it.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, models  # noqa: E402

REFITS = [np.datetime64(f"{y}-01-01") for y in range(2003, 2027)]


def tr_index(P):
    """Total-return price index (dividends reinvested) from close-to-close TR returns."""
    return (1 + P["r_cc"].fillna(0.0)).cumprod()


_OWN = {}


def own(sym):
    """The symbol's own full daily history (longer than the panel's common calendar)."""
    if sym not in _OWN:
        _OWN[sym] = core.index_frame(sym)
    return _OWN[sym]


def sma_trend(P, n=200):
    c = P["c"]
    return (c > c.rolling(n, min_periods=n).mean()).astype(float).where(c.rolling(n).count() >= n)


def tsmom(P, horizons=(21, 63, 126, 252), mode="majority"):
    """Multi-horizon time-series momentum on the TR index. mode: 'majority' (mean sign > 0 -> 1),
    'prop' (max(mean sign, 0)), 'all' (every sign positive)."""
    px = tr_index(P)
    signs = [np.sign(px / px.shift(h) - 1) for h in horizons]
    s = sum(signs) / len(signs)
    valid = px.shift(max(horizons)).notna()
    if mode == "majority":
        out = (s > 0).astype(float)
    elif mode == "prop":
        out = s.clip(lower=0)
    elif mode == "all":
        out = (s == 1).astype(float)
    else:
        raise ValueError(mode)
    return out.where(valid)


def kalman_slope(P, mode="mle", ratio=None, zthr=0.0):
    """Local-linear-trend filtered slope of log TR price > zthr * sd(slope). mode 'mle': variances refit
    walk-forward by MLE (Jan 1 each year, expanding window); mode 'fixed': q_b = ratio * q_l, r = 0.25 q_l."""
    out = pd.DataFrame(index=P["c"].index, columns=P["c"].columns, dtype=float)
    for s in out.columns:
        f = own(s)
        y = np.log((1 + f["r_cc"].fillna(0.0)).cumprod()).values
        dates = f.index.values
        first = 1
        if mode == "fixed":
            v = np.nanvar(np.diff(y[first:first + 750]))
            sl, ss, _ = models.llt_filter(y, v, v * ratio, 0.25 * v)
            col = (sl > zthr * ss).astype(float); col[:first + 252] = np.nan
            out[s] = pd.Series(col, index=f.index).reindex(out.index)
            continue
        col = np.full(len(y), np.nan)
        for k, rd in enumerate(REFITS):
            train = (dates < rd) & (np.arange(len(y)) >= first)
            if train.sum() < 750:
                continue
            q_l, q_b, r = models.llt_fit(y[train][-2520:])    # last <=10y for speed/stability
            end = REFITS[k + 1] if k + 1 < len(REFITS) else np.datetime64("2100-01-01")
            use = (dates >= rd) & (dates < end)
            sl, ss, _ = models.llt_filter(y, q_l, q_b, r)
            col[use] = (sl > zthr * ss)[use].astype(float)
        out[s] = pd.Series(col, index=f.index).reindex(out.index)
    return out


def hmm_calm(P, field="r_cc"):
    """Walk-forward 2-state Gaussian HMM on daily % returns; filtered P(calm state) known at each close."""
    out = pd.DataFrame(index=P[field].index, columns=P[field].columns, dtype=float)
    for s in P[field].columns:
        x = (100 * own(s)[field]).dropna()
        p = models.hmm_walkforward(x.index.values, x.values, REFITS)
        out[s] = pd.Series(p, index=x.index).reindex(out.index)
    return out


def market_series(sym, field="c"):
    return core.daily(sym)[field]


def _align(x, idx):
    return x.reindex(idx.union(x.index)).ffill(limit=3).reindex(idx)


def vix_term(P, thr=1.0):
    """Risk-on when VIX / VIX3M < thr (contango). Same value for every leg."""
    v, v3 = market_series("^VIX"), market_series("^VIX3M")
    ratio = (v / v3).dropna()
    on = (ratio < thr).astype(float)
    on = _align(on, P["c"].index)
    return pd.DataFrame({s: on for s in P["c"].columns})


def vix_ma(P, n=50):
    v = market_series("^VIX")
    on = (v < v.rolling(n, min_periods=n).mean()).astype(float).where(v.rolling(n).count() >= n)
    on = _align(on, P["c"].index)
    return pd.DataFrame({s: on for s in P["c"].columns})


def vix_level(P, thr=25.0):
    v = market_series("^VIX")
    on = _align((v < thr).astype(float), P["c"].index)
    return pd.DataFrame({s: on for s in P["c"].columns})


def credit_trend(P, n=100):
    """Risk-on when HYG/IEF (total return) is above its n-day average."""
    h, i = core.index_frame("HYG"), core.index_frame("IEF")
    idx = h.index.intersection(i.index)
    hp = (1 + h["r_cc"].reindex(idx).fillna(0)).cumprod()
    ip = (1 + i["r_cc"].reindex(idx).fillna(0)).cumprod()
    ratio = hp / ip
    on = (ratio > ratio.rolling(n, min_periods=n).mean()).astype(float).where(ratio.rolling(n).count() >= n)
    on = _align(on, P["c"].index)
    return pd.DataFrame({s: on for s in P["c"].columns})


# ---------------------------------------------------------------- volatility forecasts (sigma of next night)

def vol_sd(P, n=20):
    return P["r_on"].rolling(n, min_periods=n).std()


def vol_ewma(P, lam=0.94):
    out = pd.DataFrame(index=P["r_on"].index, columns=P["r_on"].columns, dtype=float)
    for s in out.columns:
        f = own(s)
        ee = np.nan_to_num(f["r_on"].values)
        h = models.ewma_var(ee, lam, init=np.var(ee[1:61]))
        col = np.sqrt(h[1:])                   # row k -> forecast for night k+1
        col[:60] = np.nan
        out[s] = pd.Series(col, index=f.index).reindex(out.index)
    return out


def vol_garch(P, refits=REFITS):
    """GARCH(1,1) on overnight returns, refit Jan 1 each year on all prior data (expanding)."""
    out = pd.DataFrame(index=P["r_on"].index, columns=P["r_on"].columns, dtype=float)
    params = {}
    for s in out.columns:
        f = own(s)
        ee = np.nan_to_num(f["r_on"].values) * 100.0
        dd = f.index.values
        col = np.full(len(ee), np.nan)
        for k, rd in enumerate(refits):
            train = dd < rd
            if train.sum() < 750:
                continue
            om, al, be = models.garch_fit(ee[train])
            params[(s, str(rd)[:4])] = (om, al, be)
            h = models.garch_filter(ee, om, al, be, s0=np.var(ee[1:61]))
            end = refits[k + 1] if k + 1 < len(refits) else np.datetime64("2100-01-01")
            use = (dd >= rd) & (dd < end)
            col[use] = (np.sqrt(h[1:]) / 100.0)[use]
        out[s] = pd.Series(col, index=f.index).reindex(out.index)
    return out, params
