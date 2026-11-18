"""Evaluate the stocks-in-play grid: top-K by relative volume, each entry x stop, portfolio = equal slots of 1/K.
Prints trade and portfolio statistics, a relative-volume control, and the deflated Sharpe of the best variant."""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
from stats import deflated_sharpe
E = pd.read_pickle("trader/data/sip_events.pkl")
C = E[E.eligible & E.bull].copy()
C["rank"] = C.groupby("date").rv.rank(ascending=False, method="first")
days = sorted(E.date.unique())
max_price = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
STOPS = ("orlow", "atr10", "atr25", "none")
rows = []
for K in (1, 3, 5, 10, 20):
    for e in ("E1", "E2"):
        for s in STOPS:
            sel = C[(C["rank"] <= K) & (C.open <= max_price)] if max_price < 1e9 else C[C["rank"] <= K]
            port = sel.groupby("date")[f"{e}_{s}_ret"].sum() / K
            port = port.reindex(days).fillna(0.0)
            n = len(port); mu, sd = port.mean(), port.std(ddof=1)
            ins = sel[sel[f"{e}_{s}_in"]]
            rows.append({"K": K, "entry": e, "stop": s, "trades": len(ins), "trade_mean_bp": 1e4 * ins[f"{e}_{s}_ret"].mean(),
                         "hit": (ins[f"{e}_{s}_ret"] > 0).mean(), "R": ins[f"{e}_{s}_R"].mean(),
                         "day_mean_bp": 1e4 * mu, "SR": mu / sd * math.sqrt(252) if sd > 0 else 0, "t": mu / sd * math.sqrt(n) if sd > 0 else 0,
                         "pos_days": (port > 0).mean()})
G = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print(G.round(3).to_string(index=False))
best = G.loc[G.SR.idxmax()]
p, emax = deflated_sharpe(best.SR, len(days), len(G), G.SR.var())
print(f"\nBest of {len(G)} variants: K={best.K} {best.entry} stop={best.stop} SR {best.SR:.2f} t {best.t:.2f}; "
      f"expected max SR of {len(G)} null trials {emax:.2f}; deflated-Sharpe probability {p:.2f}")
# Control: does relative volume matter? compare RV buckets for one simple rule (E1, no stop, 9:35 -> 15:55)
C["rv_bucket"] = pd.cut(C.rv, [1, 1.5, 2, 3, 5, 1e9], labels=["1-1.5", "1.5-2", "2-3", "3-5", "5+"])
print("\nRV control, E1 no stop (bp per trade, net):")
print(C.groupby("rv_bucket", observed=True).agg(n=("E1_none_ret", "size"), mean_bp=("E1_none_ret", lambda x: 1e4 * x.mean()),
                                                 hit=("E1_none_ret", lambda x: (x > 0).mean())).round(2).to_string())
print("\nRV control, E2 buy-stop with 10% ATR stop (bp per trade, net):")
X = C[C.E2_atr10_in]
print(X.groupby("rv_bucket", observed=True).agg(n=("E2_atr10_ret", "size"), mean_bp=("E2_atr10_ret", lambda x: 1e4 * x.mean()),
                                                 hit=("E2_atr10_ret", lambda x: (x > 0).mean()), R=("E2_atr10_R", "mean")).round(2).to_string())
