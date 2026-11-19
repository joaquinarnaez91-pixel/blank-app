"""$1,000 whole-share versions (3-5 slots, actual prices) of the canonical Connors rule and the weekly reversal,
plus the next-open-fill robustness check. python3 tools/research/xsec_swing/meanrev_1000.py"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, report
from swing import simulate, trade_stats
from meanrev import build_signals


def main():
    d = Data()
    V, aux = build_signals(d)
    t0 = 260
    for name in ("c_rsi10_x5_h5", "r_wk_u200_t5_sma"):
        entry, prio, ex, H = V[name]
        for K in (3, 5):
            r, tr = simulate(d, entry, prio, ex, K=K, capital=1000.0, cost=0.0010, whole=True, max_hold=H, t0=t0)
            ts = trade_stats(tr)
            report(f"{name} $1,000 {K} slots", r)
            print(f"      trades {ts['trades']}, mean {ts['mean_pct']:+.2f}%/trade, median {ts['median_pct']:+.2f}%, win {ts['win']:.0%}, "
                  f"avg win {ts['avg_win_pct']:+.2f}% avg loss {ts['avg_loss_pct']:+.2f}%, hold {ts['days']:.1f}d, worst {ts['worst_pct']:+.1f}%")
        r, tr = simulate(d, entry, prio, ex, K=10, capital=1e7, cost=0.0010, whole=False, max_hold=H, t0=t0, mode="nextopen")
        report(f"{name} 10 slots, next-open fills", r)


if __name__ == "__main__":
    main()
