"""Live monitor for the overnight model. Reads trader/overnight_ledger.csv:
   date,fund,shares,buy_price,sell_price,official_close,official_open,sleeve_value
(one row per leg per night; official prices from Yahoo daily bars for the buy and sell days).

Three checks, from fastest to slowest to give an answer:
 1. Execution tracking: live leg return (sell/buy) minus the official-price return (open/close). Its noise is
    ~15 bp per leg, so an execution problem shows within a couple of weeks. Alarm: mean < -5 bp with t < -2.
 2. Expected range: where cumulative P&L should be after N nights if the backtest is right (90% band).
 3. Bayesian edge estimate: skeptical prior on the mean nightly return (centered at half the backtest mean,
    sd = the backtest mean, so zero is plausible), normal-normal update with the live nights.
Backtest reference (feasible spec, half the sleeve per night, actual 3x fund prices, 10y): +7.0%/yr,
Sharpe 0.79 -> per night on the sleeve: mean 2.7 bp, sd 54 bp.
"""
import sys, math
from statistics import NormalDist
import pandas as pd
N01 = NormalDist()
MU_BT, SD_NIGHT = 0.00027, 0.0054


def main(path="trader/overnight_ledger.csv"):
    try:
        L = pd.read_csv(path)
    except FileNotFoundError:
        print("No live overnight trades yet."); return
    L["live"] = L.sell_price / L.buy_price - 1
    L["official"] = L.official_open / L.official_close - 1
    L["track_bp"] = 1e4 * (L.live - L.official)
    L["pnl"] = L.shares * (L.sell_price - L.buy_price)
    t = L.track_bp
    tstat = t.mean() / (t.std(ddof=1) / math.sqrt(len(t))) if len(t) > 2 and t.std(ddof=1) > 0 else float("nan")
    alarm = len(t) >= 10 and t.mean() < -5 and tstat < -2
    print(f"1. Execution tracking over {len(t)} legs: mean {t.mean():+.1f} bp (t {tstat:+.2f}){'  ALARM: fills are worse than the model' if alarm else ''}")
    nights = L.groupby("date").agg(pnl=("pnl", "sum"), value=("sleeve_value", "first"))
    n = len(nights)
    r = nights.pnl / nights.value
    cum = nights.pnl.sum()
    v = nights.value.iloc[-1]
    lo, hi = n * MU_BT * v - 1.645 * SD_NIGHT * v * math.sqrt(n), n * MU_BT * v + 1.645 * SD_NIGHT * v * math.sqrt(n)
    print(f"2. {n} nights, cumulative ${cum:+.2f}; the backtest's 90% range after {n} nights is ${lo:+.0f} to ${hi:+.0f}"
          f" -> {'inside: noise' if lo <= cum <= hi else 'OUTSIDE the range: investigate'}")
    m0, s0 = MU_BT / 2, MU_BT
    prec = 1 / s0 ** 2 + n / SD_NIGHT ** 2
    mpost = (m0 / s0 ** 2 + r.sum() / SD_NIGHT ** 2) / prec
    spost = math.sqrt(1 / prec)
    print(f"3. Edge estimate: {1e4*mpost:+.2f} bp per night (prior {1e4*m0:+.2f}), P(edge > 0) = {1 - N01.cdf(-mpost / spost):.0%}"
          f" ({n} live nights carry {n / SD_NIGHT**2 / prec:.0%} of the weight)")


if __name__ == "__main__":
    main(*sys.argv[1:])
