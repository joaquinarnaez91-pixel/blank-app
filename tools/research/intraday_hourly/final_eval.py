"""Hurdle check for the one candidate that survived: the cash-day stock gap-down reversal.

Rule (pre-specified before the 2017-2023 holdout was looked at in the regime split; see the report):
  On day t, only if SPY's close on t-1 was below its 200-day SMA (the overnight model is in cash for SPY's leg):
  among stocks with prior close >= $5 and 14-day mean volume >= 1M shares, take those whose 9:30 open is at least
  2% below the prior close, rank by gap size (largest first), buy the top 5 at the open with $200 each (whole
  shares; skip names priced above $200), sell everything at the close. 10 bp round trip (20/30 bp sensitivity).
DSR: trials = every variant this agent logged in the gaps_intraday family (registry), with the cross-trial Sharpe
variance taken as the larger of (a) the null sampling variance 252/n and (b) the empirical variance across the
regime cells, so the hurdle is not flattered by highly correlated variants.
Run (after the experiments are logged): python3 tools/research/intraday_hourly/final_eval.py
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from registry import load as reg_load
from stats import deflated_sharpe
from sim import topk
from common import evaluate, nw_t, moments, regime_below, cash_fraction, AUTHOR, OUT
from gaps_daily_holdout import build, SPLIT


def series():
    syms, cal, A = build()
    o, c, v = A["o"], A["c"], A["v"]
    N, T = c.shape
    dates = np.array(list(cal))
    pC = np.concatenate([np.full((N, 1), np.nan), c[:, :-1]], 1)
    adv = pd.DataFrame(v.T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    gap = o / pC - 1
    y = c / o - 1
    U = (pC >= 5) & (adv >= 1e6) & np.isfinite(gap) & np.isfinite(y) & (np.abs(gap) < 0.6) & (o > 0) & (np.abs(y) < 0.6)
    U[:, :20] = False
    el = U & (gap <= -0.02)
    sc = np.where(el, -gap, -np.inf)
    reg = regime_below(dates, "SPY")
    out = {}
    for cost in (0.0010, 0.0020, 0.0030):
        w, _, _ = topk(sc, el, o, c, dates, 5, cost=cost)
        out[f"SPY<200d, $1000, {cost*1e4:.0f}bp"] = pd.Series(np.where(reg, w.to_numpy(), 0.0), index=dates)[20:]
    cf = cash_fraction(dates)
    tot = np.zeros(T)
    for frac in (1 / 3, 2 / 3, 1.0):
        m = np.isclose(cf, frac)
        wf, _, _ = topk(sc, el, o, c, dates, 5, capital=1000 * frac, cost=0.0010)
        tot += np.where(m, wf.to_numpy() * frac, 0.0)
    out["capital freed by the overnight model (thirds), 10bp"] = pd.Series(tot, index=dates)[20:]
    return out, reg[20:]


def main():
    R = [r for r in reg_load() if r.get("author") == AUTHOR]
    fam = {}
    for r in R:
        fam[r["family"]] = fam.get(r["family"], 0) + r["variants"]
    n_gap = fam.get("gaps_intraday", 0)
    print("trials logged by this agent per family:", fam)
    cells = pd.read_csv(OUT / "gd_regime.csv")
    v_emp = float(np.var(cells[cells.cost_bp == 10].full_SR, ddof=1))
    S, reg = series()
    for name, r in S.items():
        n = len(r)
        e = evaluate(r, name, n_gap, max(252 / n, v_emp))
        sk, ku = moments(r)
        p_null, emax_null = deflated_sharpe(e["sharpe"], n, n_gap, 252 / n, sk, ku)
        p_emp, emax_emp = deflated_sharpe(e["sharpe"], n, n_gap, v_emp, sk, ku)
        pre = r[[d < SPLIT for d in r.index]]; post = r[[d >= SPLIT for d in r.index]]
        act = r[r != 0]
        h = len(act) // 2
        print(f"\n{name}: days {n}, active {len(act)}; mean {e['mean_bp']:+.2f} bp/day ({1e4*act.mean():+.1f} per active day); "
              f"t {e['t']:+.2f}, NW t {e['nw_t']:+.2f}; Sharpe {e['sharpe']:.2f}; skew {sk:+.2f} kurt {ku:.1f}")
        print(f"   DSR with {n_gap} trials: {p_null:.2f} (null SR variance {252/n:.3f}, E[max SR] {emax_null:.2f}); "
              f"{p_emp:.2f} (empirical variance {v_emp:.3f}, E[max] {emax_emp:.2f})")
        print(f"   halves (active days, chronological): {1e4*act.iloc[:h].mean():+.1f} / {1e4*act.iloc[h:].mean():+.1f} bp; "
              f"holdout 2017-2023 {1e4*pre.mean():+.2f} bp/day NW t {nw_t(pre):+.2f}; 2024-26 {1e4*post.mean():+.2f} bp/day "
              f"NW t {nw_t(post):+.2f}")
        eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
        trough = dd.idxmin(); peak = eq.loc[:trough].idxmax()
        yr = r.groupby([d.year for d in r.index]).apply(lambda x: round(100 * ((1 + x).prod() - 1), 1)).to_dict()
        print(f"   annualized on the full sleeve: {100*((1+r).prod()**(252/n)-1):+.1f}%/yr; max drawdown "
              f"{100*dd.min():.1f}% ({peak} -> {trough}); by year %: {yr}")


if __name__ == "__main__":
    main()
