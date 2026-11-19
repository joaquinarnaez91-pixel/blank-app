"""Numpy-only models for the overnight program: Nelder-Mead, 2-state Gaussian HMM (Baum-Welch, filtered
probabilities), Kalman local-linear-trend (MLE), GARCH(1,1) (MLE), EWMA. All walk-forward helpers fit on data
up to a refit date and produce forecasts only for later dates (no look-ahead)."""
import math
import numpy as np
try:                      # optional speed-up; the code runs (slower) without numba
    from numba import njit
except Exception:         # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


# ---------------------------------------------------------------- optimizer

def nelder_mead(f, x0, step=0.5, iters=400, tol=1e-8):
    n = len(x0)
    pts = [np.asarray(x0, float)]
    for i in range(n):
        x = np.array(x0, float); x[i] += step; pts.append(x)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals); pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (abs(vals[0]) + 1e-12):
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# ---------------------------------------------------------------- HMM

def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


@njit(cache=True)
def _fb(B, pi, A):
    T = B.shape[0]
    alpha = np.empty((T, 2)); beta = np.empty((T, 2)); c = np.empty(T)
    a0 = pi[0] * B[0, 0]; a1 = pi[1] * B[0, 1]; c[0] = a0 + a1
    alpha[0, 0] = a0 / c[0]; alpha[0, 1] = a1 / c[0]
    for t in range(1, T):
        a0 = (alpha[t - 1, 0] * A[0, 0] + alpha[t - 1, 1] * A[1, 0]) * B[t, 0]
        a1 = (alpha[t - 1, 0] * A[0, 1] + alpha[t - 1, 1] * A[1, 1]) * B[t, 1]
        c[t] = a0 + a1; alpha[t, 0] = a0 / c[t]; alpha[t, 1] = a1 / c[t]
    beta[T - 1, 0] = 1.0; beta[T - 1, 1] = 1.0
    for t in range(T - 2, -1, -1):
        b0 = B[t + 1, 0] * beta[t + 1, 0]; b1 = B[t + 1, 1] * beta[t + 1, 1]
        beta[t, 0] = (A[0, 0] * b0 + A[0, 1] * b1) / c[t + 1]
        beta[t, 1] = (A[1, 0] * b0 + A[1, 1] * b1) / c[t + 1]
    return alpha, beta, c


def hmm_fit(x, iters=300, tol=1e-7, init=None):
    """2-state Gaussian HMM by Baum-Welch (scaled). Returns dict(pi, A, mu, sd, ll)."""
    x = np.asarray(x, float)
    s = x.std()
    if init is None:
        mu = np.array([x.mean() + 0.1 * s, x.mean() - 0.3 * s]); sd = np.array([0.7 * s, 1.6 * s])
        A = np.array([[0.98, 0.02], [0.04, 0.96]]); pi = np.array([0.5, 0.5])
    else:
        pi, A, mu, sd = [np.array(v, float) for v in init]
    ll_old = -np.inf
    ll = -np.inf
    for _ in range(iters):
        B = _norm_pdf(x, mu, sd) + 1e-300
        alpha, beta, c = _fb(B, pi, A)
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        pi = gamma[0]
        A = xi.sum(0) / gamma[:-1].sum(0)[:, None]
        A /= A.sum(1, keepdims=True)
        w = gamma.sum(0)
        mu = (gamma * x[:, None]).sum(0) / w
        sd = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(0) / w)
        sd = np.maximum(sd, 1e-4 * s + 1e-12)
        ll = np.log(c).sum()
        if ll - ll_old < tol * abs(ll):
            break
        ll_old = ll
    return {"pi": pi, "A": A, "mu": mu, "sd": sd, "ll": ll}


def hmm_filter(x, p):
    """Filtered state probabilities P(s_t | x_1..x_t) for all t."""
    x = np.asarray(x, float)
    B = _norm_pdf(x, p["mu"], p["sd"]) + 1e-300
    alpha, beta, c = _fb(B, np.asarray(p["pi"], float), np.asarray(p["A"], float))
    return alpha


def hmm_walkforward(dates, x, refit_dates, min_train=750):
    """P(calm state) filtered, using parameters refit at each refit date on data strictly before it.
    The calm state = the one with the lower sd. Returns array aligned with x (nan before first refit)."""
    dates = np.asarray(dates)
    out = np.full(len(x), np.nan)
    prev = None
    for k, rd in enumerate(refit_dates):
        train = dates < rd
        if train.sum() < min_train:
            continue
        xt = x[train]
        best = None
        inits = [None] if prev is None else [None, (prev["pi"], prev["A"], prev["mu"], prev["sd"])]
        for ini in inits:
            p = hmm_fit(xt, init=ini)
            if best is None or p["ll"] > best["ll"]:
                best = p
        prev = best
        calm = int(np.argmin(best["sd"]))
        end = refit_dates[k + 1] if k + 1 < len(refit_dates) else np.datetime64("2100-01-01")
        use = (dates >= rd) & (dates < end)
        upto = dates < end
        filt = hmm_filter(x[upto], best)       # filter runs forward only; values at t use x up to t
        out[use] = filt[use[upto], calm]
    return out


# ---------------------------------------------------------------- Kalman local linear trend

@njit(cache=True)
def llt_filter(y, q_l, q_b, r):
    """Local linear trend Kalman filter. Returns filtered slope, its sd, and the log-likelihood."""
    n = len(y)
    a0 = y[0]; a1 = 0.0
    p00 = r + q_l; p01 = 0.0; p11 = q_b * 1e4 + 1e-8
    slope = np.empty(n); sslope = np.empty(n); ll = 0.0
    for t in range(n):
        if t > 0:
            a0 = a0 + a1
            n00 = p00 + 2 * p01 + p11 + q_l; n01 = p01 + p11; n11 = p11 + q_b
            p00, p01, p11 = n00, n01, n11
        v = y[t] - a0; F = p00 + r
        k0 = p00 / F; k1 = p01 / F
        a0 = a0 + k0 * v; a1 = a1 + k1 * v
        n00 = p00 - k0 * p00; n01 = p01 - k0 * p01; n11 = p11 - k1 * p01
        p00, p01, p11 = n00, n01, n11
        if t > 1:
            ll += -0.5 * (math.log(2 * math.pi * F) + v * v / F)
        slope[t] = a1; sslope[t] = math.sqrt(max(p11, 0.0))
    return slope, sslope, ll


def llt_fit(y):
    """MLE of log variances (q_l, q_b, r) by Nelder-Mead."""
    v = np.var(np.diff(y))
    def nll(th):
        q_l, q_b, r = np.exp(th)
        return -llt_filter(y, q_l, q_b, r)[2]
    x0 = np.log([0.8 * v, v * 1e-4, 0.2 * v])
    th, f = nelder_mead(nll, x0, step=1.0, iters=300)
    return np.exp(th)


# ---------------------------------------------------------------- GARCH / EWMA

@njit(cache=True)
def _garch(e, omega, alpha, beta, h0):
    n = len(e)
    h = np.empty(n + 1)
    h[0] = h0
    for t in range(n):
        h[t + 1] = omega + alpha * e[t] * e[t] + beta * h[t]
    return h


def garch_filter(e, omega, alpha, beta, s0=None):
    """Conditional variance h_t of e_t given e_1..e_{t-1}; also returns the one-step forecast h_{T+1}."""
    e = np.asarray(e, float)
    return _garch(e, float(omega), float(alpha), float(beta), float(np.var(e) if s0 is None else s0))


def garch_fit(e):
    """Gaussian QMLE of GARCH(1,1) with variance targeting off; params via logit/log transforms."""
    e = np.asarray(e, float)
    v = e.var()
    def unpack(th):
        omega = math.exp(th[0]); ab = 1 / (1 + math.exp(-th[1])); a_share = 1 / (1 + math.exp(-th[2]))
        alpha = ab * a_share; beta = ab * (1 - a_share)
        return omega, alpha, beta
    def nll(th):
        omega, alpha, beta = unpack(th)
        h = garch_filter(e, omega, alpha, beta)[:-1]
        h = np.maximum(h, 1e-12)
        return 0.5 * np.sum(np.log(h) + e ** 2 / h)
    x0 = np.array([math.log(v * 0.05), 2.9, -2.0])
    th, f = nelder_mead(nll, x0, step=0.7, iters=500)
    return unpack(th)


def ewma_var(e, lam=0.94, init=None):
    """h_{t+1} = lam h_t + (1-lam) e_t^2; returns array of length n+1 (h[t] = forecast for e_t)."""
    n = len(e)
    h = np.empty(n + 1)
    h[0] = np.nanvar(e[:60]) if init is None else init
    for t in range(n):
        x = e[t] if np.isfinite(e[t]) else 0.0
        h[t + 1] = lam * h[t] + (1 - lam) * x * x
    return h
