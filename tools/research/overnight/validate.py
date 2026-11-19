"""Validation statistics for the overnight program (numpy only).

- stationary_boot_idx: Politis & Romano (1994) stationary bootstrap index paths.
- boot_ci: bootstrap confidence intervals for Sharpe and CAGR.
- spa_test: White (2000) Reality Check and Hansen (2005) SPA (lower / consistent / upper p-values).
- cscv_pbo: Combinatorially symmetric cross-validation probability of backtest overfitting
  (Bailey, Borwein, Lopez de Prado & Zhu 2017).
- dsr: deflated Sharpe ratio probability (Bailey & Lopez de Prado 2014) with sample skew/kurtosis.
"""
import math, itertools, sys
from statistics import NormalDist
import numpy as np
sys.path.insert(0, "tools/research")
from stats import deflated_sharpe  # noqa: E402

N01 = NormalDist()


def stationary_boot_idx(n, B, b, seed=0):
    rng = np.random.default_rng(seed)
    p = 1.0 / b
    idx = np.empty((B, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n, size=B)
    for t in range(1, n):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, n, size=B), (idx[:, t - 1] + 1) % n)
    return idx


def sharpe(r, axis=-1, per_year=252):
    m = r.mean(axis=axis); s = r.std(axis=axis, ddof=1)
    return m / s * math.sqrt(per_year)


def cagr(r, axis=-1, per_year=252):
    return np.exp(np.log1p(r).mean(axis=axis) * per_year) - 1


def boot_ci(r, B=5000, b=20, seed=0, alpha=0.05):
    r = np.asarray(r, float)
    idx = stationary_boot_idx(len(r), B, b, seed)
    rb = r[idx]
    sr, cg = sharpe(rb), cagr(rb)
    q = [alpha / 2, 0.5, 1 - alpha / 2]
    return {"sr": float(sharpe(r)), "sr_ci": np.quantile(sr, q).tolist(), "p_sr_le_0": float((sr <= 0).mean()),
            "cagr": float(cagr(r)), "cagr_ci": np.quantile(cg, q).tolist(), "p_cagr_le_0": float((cg <= 0).mean())}


def paired_sr_diff(r1, r0, B=5000, b=20, seed=0):
    """Bootstrap distribution of SR(r1) - SR(r0) on the same resampled nights. Returns diff, CI, p(diff<=0)."""
    r1, r0 = np.asarray(r1, float), np.asarray(r0, float)
    idx = stationary_boot_idx(len(r1), B, b, seed)
    d = sharpe(r1[idx]) - sharpe(r0[idx])
    return {"d_sr": float(sharpe(r1) - sharpe(r0)), "ci": np.quantile(d, [0.025, 0.975]).tolist(),
            "p": float((d <= 0).mean())}


def spa_test(R, r0, B=3000, b=20, seed=0, vol_match=True):
    """R: (n, m) variant returns; r0: (n,) benchmark. H0: no variant beats the benchmark.
    With vol_match each variant is rescaled (one constant) to the benchmark's volatility, so the test compares
    risk-adjusted performance (mean at equal volatility ~ Sharpe)."""
    R = np.asarray(R, float); r0 = np.asarray(r0, float)
    n, m = R.shape
    if vol_match:
        R = R * (r0.std(ddof=1) / R.std(axis=0, ddof=1))
    d = R - r0[:, None]
    dbar = d.mean(0)
    idx = stationary_boot_idx(n, B, b, seed)
    dstar = np.stack([d[idx[k]].mean(0) for k in range(B)])         # (B, m)
    omega = np.sqrt(n) * (dstar - dbar).std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, 1e-12)
    t_obs = max(0.0, float(np.max(np.sqrt(n) * dbar / omega)))
    thr = -np.sqrt(omega ** 2 / n * 2 * math.log(math.log(n)))
    g = {"lower": np.maximum(dbar, 0), "consistent": dbar * (dbar >= thr), "upper": dbar}
    out = {"t_spa": t_obs, "best": int(np.argmax(dbar / omega))}
    for k, gk in g.items():
        z = np.sqrt(n) * (dstar - gk) / omega
        tb = np.maximum(z.max(1), 0)
        out["p_" + k] = float((tb >= t_obs).mean())
    t_rc = float(np.max(np.sqrt(n) * dbar))
    trc_b = (np.sqrt(n) * (dstar - dbar)).max(1)
    out["t_rc"] = t_rc
    out["p_rc"] = float((trc_b >= t_rc).mean())
    return out


def cscv_pbo(M, S=16, per_year=252):
    """M: (T, N) returns of N variants. Returns PBO, median OOS relative rank of the IS-best, P(OOS SR < 0)
    for the IS-best, and the slope of OOS SR on IS SR (performance degradation)."""
    M = np.asarray(M, float)
    T, N = M.shape
    T = (T // S) * S
    M = M[-T:]
    blocks = M.reshape(S, T // S, N)
    cnt = np.full(S, T // S)
    s1 = blocks.sum(1); s2 = (blocks ** 2).sum(1)
    lam, oos_best, is_best = [], [], []
    for comb in itertools.combinations(range(S), S // 2):
        a = np.zeros(S, bool); a[list(comb)] = True
        def sr(mask):
            n = cnt[mask].sum(); m1 = s1[mask].sum(0) / n
            v = (s2[mask].sum(0) - n * m1 ** 2) / (n - 1)
            return m1 / np.sqrt(np.maximum(v, 1e-18)) * math.sqrt(per_year)
        ins, oos = sr(a), sr(~a)
        k = int(np.argmax(ins))
        rank = (oos < oos[k]).sum() + 0.5 * ((oos == oos[k]).sum() - 1) + 1     # 1..N
        w = rank / (N + 1)
        lam.append(math.log(w / (1 - w)))
        oos_best.append(oos[k]); is_best.append(ins[k])
    lam = np.array(lam); oos_best = np.array(oos_best); is_best = np.array(is_best)
    slope = np.polyfit(is_best, oos_best, 1)[0] if np.std(is_best) > 0 else float("nan")
    return {"pbo": float((lam <= 0).mean()), "median_logit": float(np.median(lam)),
            "p_oos_best_neg": float((oos_best < 0).mean()), "mean_oos_sr_best": float(oos_best.mean()),
            "mean_is_sr_best": float(is_best.mean()), "degradation_slope": float(slope), "combos": len(lam)}


def dsr(r, n_trials, sr_trials_var, per_year=252):
    r = np.asarray(r, float)
    mu, sd = r.mean(), r.std(ddof=1)
    sk = float(((r - mu) ** 3).mean() / sd ** 3); ku = float(((r - mu) ** 4).mean() / sd ** 4)
    p, emax = deflated_sharpe(mu / sd * math.sqrt(per_year), len(r), n_trials, sr_trials_var, sk, ku, per_year)
    return p, emax
