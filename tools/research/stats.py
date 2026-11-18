"""Statistics for strategy evaluation: Sharpe, t-stats, yearly splits, deflated Sharpe (Bailey & Lopez de Prado 2014)."""
import math
from statistics import NormalDist
import numpy as np
N01 = NormalDist()


def summary(r, name="", per_year=252):
    """r: pandas Series of daily strategy returns (0 on flat days), indexed by date."""
    r = r.dropna()
    n = len(r)
    active = r[r != 0]
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd * math.sqrt(per_year) if sd > 0 else float("nan")
    t = mu / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"name": name, "days": n, "trades": int((r != 0).sum()), "mean_bp": 1e4 * mu,
            "hit": float((active > 0).mean()) if len(active) else float("nan"),
            "ann_ret": float(eq.iloc[-1] ** (per_year / n) - 1) if n else float("nan"),
            "sharpe": sr, "t": t, "maxdd": float(dd), "total": float(eq.iloc[-1] - 1)}


def by_year(r):
    out = {}
    for y, g in r.groupby([d.year for d in r.index]):
        sd = g.std(ddof=1)
        out[y] = (round(1e4 * g.mean(), 1), round(g.mean() / sd * math.sqrt(252), 2) if sd > 0 else None, len(g))
    return out


def deflated_sharpe(sr_ann, n_obs, n_trials, sr_ann_trials_var, skew=0.0, kurt=3.0, per_year=252):
    """Probability that the true Sharpe > 0 after picking the best of n_trials (DSR).
    sr inputs annualized; converted to per-period."""
    sr = sr_ann / math.sqrt(per_year)
    v = sr_ann_trials_var / per_year
    g = 0.5772156649
    if n_trials > 1:
        e_max = math.sqrt(v) * ((1 - g) * N01.inv_cdf(1 - 1 / n_trials) + g * N01.inv_cdf(1 - 1 / (n_trials * math.e)))
    else:
        e_max = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - e_max) * math.sqrt(n_obs - 1) / denom
    return N01.cdf(z), e_max * math.sqrt(per_year)


def fmt(s):
    return ("{name:<44} days {days:4d} trades {trades:4d} mean {mean_bp:+6.1f}bp hit {hit:5.1%} "
            "ann {ann_ret:+7.1%} SR {sharpe:+5.2f} t {t:+5.2f} maxDD {maxdd:+6.1%}").format(**s)
