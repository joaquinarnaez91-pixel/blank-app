"""Top-K long-only day portfolios for a $1,000 cash account, whole shares, with the sleeve's loss caps.

Inputs are (N stocks, T days) arrays. Each day: candidates = elig[:, t], ranked by score (descending), skipping
stocks whose entry price exceeds the slot size (capital / K) -- a whole share must fit. Shares = floor(slot / entry);
with a stop, shares are cut so the loss at the stop is <= $100 per trade and <= $200 for the day.
Returns two daily series: the whole-share dollar P&L / capital, and the fractional equal-weight version.
"""
import math
import numpy as np, pandas as pd


def topk(score, elig, entry, exitp, dates, K, stop=None, capital=1000.0, cost=0.0010,
         max_loss_trade=100.0, max_loss_day=200.0, collect=False):
    N, T = score.shape
    whole = np.zeros(T); frac = np.zeros(T); npos = np.zeros(T, int)
    trades = []
    slot = capital / K
    sc = np.where(elig, score, -np.inf)
    order = np.argsort(-sc, axis=0)
    for t in range(T):
        picked = []
        for n in order[:, t]:
            if not np.isfinite(sc[n, t]):
                break
            if entry[n, t] <= slot and np.isfinite(exitp[n, t]):
                picked.append(n)
                if len(picked) == K:
                    break
        if not picked:
            continue
        p = np.array(picked)
        e, x = entry[p, t], exitp[p, t]
        sh = np.floor(slot / e)
        if stop is not None:
            risk = np.maximum(e - stop[p, t], 0.01)
            sh = np.minimum(sh, np.floor(max_loss_trade / risk))
            tot = (sh * risk).sum()
            if tot > max_loss_day:
                sh = np.floor(sh * max_loss_day / tot)
        pnl = (sh * (x - e)).sum() - cost * (sh * e).sum()
        whole[t] = pnl / capital
        r = x / e - 1 - cost
        frac[t] = r.sum() / K
        npos[t] = int((sh > 0).sum())
        if collect:
            for j, n in enumerate(p):
                trades.append((dates[t], n, e[j], x[j], sh[j], r[j]))
    idx = pd.Index(dates)
    out = (pd.Series(whole, index=idx), pd.Series(frac, index=idx), pd.Series(npos, index=idx))
    return out + (trades,) if collect else out


def stop_exit(o, h, l, c, entry_bar, entry, stop, last_bar=6):
    """Exit price for a long entered at `entry` in bar `entry_bar` (array per stock-day, may vary), protective stop
    `stop`, else the close of `last_bar`. Stop-first: if a bar's low reaches the stop, the stop fills (at the bar open
    instead if the bar opens below it). The entry bar itself is checked too (pessimistic for buy-stop entries)."""
    N, T, _ = o.shape
    exitp = c[:, :, last_bar].astype(float).copy()
    done = np.zeros((N, T), bool)
    for k in range(1, last_bar + 1):
        active = (~done) & (entry_bar <= k)
        gap_thru = active & (entry_bar < k) & (o[:, :, k] <= stop)
        hit = active & ~gap_thru & (l[:, :, k] <= stop)
        exitp = np.where(gap_thru, o[:, :, k], np.where(hit, stop, exitp))
        done |= gap_thru | hit
    return exitp, done
