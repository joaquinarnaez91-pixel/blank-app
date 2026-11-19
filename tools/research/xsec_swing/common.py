"""Shared pieces for the cross-sectional overnight / swing studies.

Conventions
- Panel rows are trading days t. A decision "at the close of t" uses data through C[t] (a 15:58 proxy).
- Overnight return realized on the morning of t+1: RON[t+1] = (O[t+1] + D[t+1]) / C[t] - 1 (dividend on its ex-date).
- Intraday return of day t: RID[t] = C[t] / O[t] - 1. Close-to-close total return RCC[t] = (C[t] + D[t]) / C[t-1] - 1.
- Universe at decision time t: common stocks (no ETFs), top K by mean dollar volume over t-60..t-1, C[t-1] >= $5.
- Daily strategy series are indexed by the date the return is realized (the morning of t+1 for overnight trades).
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
sys.path.insert(0, "tools/research/xsec_swing")
from panel import load as load_panel
from stats import summary, by_year, deflated_sharpe, fmt


def rolling_mean(X, n, minp=None):
    """Column-wise trailing mean over the last n rows (including the current row), NaN-aware."""
    minp = minp or n
    df = pd.DataFrame(X)
    return df.rolling(n, min_periods=minp).mean().values


def shift(X, k=1):
    Y = np.full_like(X, np.nan)
    if k > 0:
        Y[k:] = X[:-k]
    elif k < 0:
        Y[:k] = X[-k:]
    else:
        Y[:] = X
    return Y


class Data:
    def __init__(self):
        P = load_panel()
        self.P = P
        self.dates = P["dates"]
        self.syms = P["syms"]
        self.col = {s: j for j, s in enumerate(self.syms)}
        self.O, self.H, self.L, self.C, self.V, self.D = (P[k] for k in "OHLCVD")
        self.is_etf = P["is_etf"]
        C, O, D = self.C, self.O, self.D
        Cp = shift(C, 1)
        self.RON = (O + D) / Cp - 1            # overnight, realized at the open of row t
        self.RID = C / O - 1                    # intraday of row t
        self.RCC = (C + D) / Cp - 1             # close-to-close total return of row t
        dv = C * self.V
        self.ADV = shift(rolling_mean(dv, 60, 40), 1)   # through t-1
        adv = np.where(self.is_etf[None, :], np.nan, self.ADV)
        adv = np.where(shift(C, 1) >= 5, adv, np.nan)
        self.adv_rank = pd.DataFrame(adv).rank(axis=1, ascending=False, method="first").values  # 1 = most liquid

    def universe(self, k=500, lo=0):
        r = self.adv_rank
        return (r > lo) & (r <= k)

    def s(self, sym):
        return self.col[sym]

    def series(self, x):
        return pd.Series(x, index=self.dates)


def index_model(d, target=0.015, cap=3.0, cost=0.0004, use_funds=True):
    """Live overnight model: equal thirds UPRO/TQQQ/TNA when SPY/QQQ/IWM close > 200-day SMA,
    index exposure = min(cap, target / 20-night std of the index's overnight return), bought at the close,
    sold at the next open. Returns the sleeve's nightly return realized on the morning of t+1, indexed by t+1."""
    out = np.zeros(len(d.dates))
    legs = {}
    for idx, fund in (("SPY", "UPRO"), ("QQQ", "TQQQ"), ("IWM", "TNA")):
        i, f = d.s(idx), d.s(fund)
        c = d.C[:, i]
        sma = rolling_mean(c[:, None], 200)[:, 0]
        on = d.RON[:, i]
        vol = pd.Series(on).rolling(20, min_periods=20).std().values      # nights up to the morning of t
        expo = np.minimum(cap, target / vol)
        w = np.where(c > sma, expo / 3.0, 0.0)                              # fraction of this third in the 3x fund
        w = np.nan_to_num(w)
        fr = d.RON[:, f] if use_funds else 3 * on
        r_next = shift(fr, -1)                                              # night t -> t+1
        leg = w * (r_next - cost)
        leg = np.nan_to_num(leg)
        legs[fund] = shift(leg, 1)                                          # index by realization date t+1
        out += leg / 3.0
    ser = pd.Series(shift(out, 1), index=d.dates).fillna(0.0)
    return ser, {k: pd.Series(v, index=d.dates).fillna(0.0) for k, v in legs.items()}


def halves(r):
    r = r.dropna()
    m = len(r) // 2
    a, b = r.iloc[:m], r.iloc[m:]
    f = lambda x: (1e4 * x.mean(), x.mean() / x.std(ddof=1) * math.sqrt(252) if x.std() > 0 else float("nan"),
                   x.mean() / x.std(ddof=1) * math.sqrt(len(x)) if x.std() > 0 else float("nan"))
    return {"first": f(a), "second": f(b), "split": str(r.index[m].date())}


def dsr(r, n_trials, sr_var):
    s = summary(r)
    p, e = deflated_sharpe(s["sharpe"], s["days"], n_trials, sr_var, skew=float(r.skew()), kurt=float(r.kurt() + 3))
    return p, e


def report(name, r, n_trials=None, sr_var=None):
    s = summary(r, name)
    h = halves(r)
    line = fmt(s) + "  | halves SR %.2f / %.2f (t %.2f / %.2f)" % (h["first"][1], h["second"][1], h["first"][2], h["second"][2])
    if n_trials:
        p, e = dsr(r, n_trials, sr_var)
        line += "  DSR p %.3f (E[max SR] %.2f, %d trials)" % (p, e, n_trials)
    print(line, flush=True)
    return s
