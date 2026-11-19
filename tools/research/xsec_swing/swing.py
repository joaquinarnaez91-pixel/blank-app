"""Swing-trade simulator for daily signals with whole shares, a slot limit and a cash account.

Mechanics (one call = one variant):
- Signals and exits are evaluated on day-t data. Fills: "close" mode trades at C[t] (a 15:58 proxy);
  "nextopen" mode trades at O[t+1] (robustness: no same-bar signal/fill).
- Each trade pays half the round-trip cost on entry and half on exit (cost = 10 bp round trip by default).
- Capital: start with `capital`; each entry gets equity/K (capped by cash); whole shares unless fractional.
  Exits are processed before entries on the same day, so sale proceeds can be reused the same day
  (each dollar still funds at most one purchase per day: there are no same-day round trips).
- Dividends are paid in cash on their ex-dates. A position whose price is missing is carried at its last price.
Returns the daily equity-curve return series and the list of trades.
"""
import math
import numpy as np, pandas as pd


def simulate(d, entry, priority, exit_fn, K=4, capital=1000.0, cost=0.0010, whole=True, mode="close",
             max_hold=5, t0=0, gate=None, min_price=0.0, return_frac=False):
    """Whole shares are counted in actual (unadjusted) shares: n = floor(alloc / actual price); the position is
    then carried in split-adjusted units (n * F) so later splits do not distort it."""
    """entry: bool T x N (signal at close t); priority: T x N (lower = better);
    exit_fn(t, j, entry_t) -> bool, evaluated at the close of t for an open position (after max_hold check)."""
    T, N = d.C.shape
    C, O, D, F = d.C, d.O, d.D, d.F
    cash = capital
    pos = {}          # j -> dict(sh, px, t_in, last)
    eq = np.full(T, np.nan)
    cashf = np.full(T, np.nan)        # cash / equity at the close, after the day's exits and entries
    trades = []
    pending_exit, pending_entry = [], []
    half = cost / 2
    for t in range(t0, T):
        # --- next-open fills decided yesterday --------------------------------------------------
        if mode == "nextopen":
            for j in pending_exit:
                p = pos.pop(j)
                px = O[t, j] if np.isfinite(O[t, j]) else p["last"]
                cash += p["sh"] * px * (1 - half)
                trades.append((p["t_in"], t, j, p["px"], px, p["sh"], p["div"]))
            pending_exit = []
            eq_open = cash + sum(p["sh"] * (O[t, j] if np.isfinite(O[t, j]) else p["last"]) for j, p in pos.items())
            for j in pending_entry:
                if len(pos) >= K or j in pos:
                    continue
                px = O[t, j]
                if not np.isfinite(px) or px <= 0:
                    continue
                alloc = min(eq_open / K, cash)
                sh = math.floor(alloc / (px * F[t, j] * (1 + half))) * F[t, j] if whole else alloc / (px * (1 + half))
                if sh <= 0:
                    continue
                cash -= sh * px * (1 + half)
                pos[j] = {"sh": sh, "px": px, "t_in": t, "last": px, "div": 0.0}
            pending_entry = []
        # --- dividends and marks ------------------------------------------------------------------
        for j, p in pos.items():
            if D[t, j] > 0 and p["t_in"] < t:
                cash += p["sh"] * D[t, j]
                p["div"] += D[t, j]
            if np.isfinite(C[t, j]):
                p["last"] = C[t, j]
        # --- exits at the close (or flagged for the next open) -----------------------------------
        for j in list(pos.keys()):
            p = pos[j]
            held = t - p["t_in"]
            if held <= 0:
                continue
            if held >= max_hold or exit_fn(t, j, p["t_in"]) or t == T - 1:
                if mode == "close" or t == T - 1:
                    px = p["last"]
                    cash += p["sh"] * px * (1 - half)
                    trades.append((p["t_in"], t, j, p["px"], px, p["sh"], p["div"]))
                    del pos[j]
                elif j not in pending_exit:
                    pending_exit.append(j)
        # --- entries --------------------------------------------------------------------------------
        if t < T - 1 and (gate is None or gate[t]):
            cand = np.flatnonzero(entry[t] & np.isfinite(C[t]) & (C[t] >= min_price))
            if len(cand):
                cand = cand[np.argsort(priority[t, cand], kind="stable")]
                equity = cash + sum(p["sh"] * p["last"] for p in pos.values())
                if mode == "close":
                    for j in cand:
                        if len(pos) >= K:
                            break
                        if j in pos:
                            continue
                        px = C[t, j]
                        alloc = min(equity / K, cash)
                        sh = math.floor(alloc / (px * F[t, j] * (1 + half))) * F[t, j] if whole else alloc / (px * (1 + half))
                        if sh <= 0:
                            continue
                        cash -= sh * px * (1 + half)
                        pos[j] = {"sh": sh, "px": px, "t_in": t, "last": px, "div": 0.0}
                else:
                    free = K - (len(pos) - len(pending_exit))
                    pending_entry = [j for j in cand if j not in pos][:max(0, free) + 3]
        eq[t] = cash + sum(p["sh"] * p["last"] for p in pos.values())
        cashf[t] = cash / eq[t] if mode == "close" else (cash - 0.0) / eq[t]
    e = pd.Series(eq, index=d.dates).ffill()
    r = e.pct_change().fillna(0.0)
    r = r[r.index >= d.dates[t0]]
    tr = pd.DataFrame(trades, columns=["t_in", "t_out", "j", "px_in", "px_out", "sh", "div"])
    if len(tr):
        tr["ret"] = (tr.px_out * (1 - half) + tr["div"]) / (tr.px_in * (1 + half)) - 1
        tr["pnl"] = tr.sh * (tr.px_out * (1 - half) + tr["div"] - tr.px_in * (1 + half))
        tr["days"] = tr.t_out - tr.t_in
        tr["sym"] = d.syms[tr.j.values]
        tr["date_in"] = d.dates[tr.t_in.values]
        tr["date_out"] = d.dates[tr.t_out.values]
    if return_frac:
        return r, tr, pd.Series(cashf, index=d.dates)
    return r, tr


def rsi(C, n=2):
    dlt = np.diff(C, axis=0, prepend=np.nan)
    up = pd.DataFrame(np.where(np.isfinite(dlt), np.maximum(dlt, 0), np.nan))
    dn = pd.DataFrame(np.where(np.isfinite(dlt), np.maximum(-dlt, 0), np.nan))
    au = up.ewm(alpha=1 / n, adjust=False, ignore_na=True).mean().values
    ad = dn.ewm(alpha=1 / n, adjust=False, ignore_na=True).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + au / ad)
    r = np.where(ad == 0, 100.0, r)
    return np.where(np.isfinite(C), r, np.nan)


def trade_stats(tr):
    if len(tr) == 0:
        return {}
    w = tr.ret
    return {"trades": len(tr), "mean_pct": 100 * w.mean(), "median_pct": 100 * w.median(), "win": float((w > 0).mean()),
            "avg_win_pct": 100 * w[w > 0].mean(), "avg_loss_pct": 100 * w[w <= 0].mean(), "days": float(tr.days.mean()),
            "worst_pct": 100 * w.min(), "best_pct": 100 * w.max(), "pnl_usd": float(tr.pnl.sum()),
            "worst_usd": float(tr.pnl.min())}
