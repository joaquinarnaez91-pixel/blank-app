"""Whole-share check for a $1,000 sleeve: each night's target dollars per fund are rounded to whole shares at
TODAY's fund prices (a constant-price approximation: historical split-adjusted prices are not the prices traded
then), the sleeve is reset to $1,000 every night (P&L not compounded), and the leg's real fund return is scaled
by the rounded/target ratio. 2010-06 -> 2026-11, actual 3x fund overnight returns, 4 bp round trip.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, evalx  # noqa: E402
from exp_realfunds import FUND, START  # noqa: E402

SLEEVE = 1000.0


def run(legs, cap, rounding):
    P = core.panel(tuple(legs))
    L = core.spec_exposure(P, cap=cap)
    px = {s: core.daily(FUND[s])["c"].iloc[-1] for s in legs}
    fr = pd.DataFrame({s: core.index_frame(FUND[s])["r_on"].reindex(L.index) for s in legs}).fillna(0.0)
    target = L.fillna(0.0) * SLEEVE / len(legs) / 3.0
    if rounding == "none":
        dollars = target
    else:
        f = np.round if rounding == "nearest" else np.floor
        dollars = pd.DataFrame({s: f(target[s] / px[s]) * px[s] for s in legs})
        over = dollars.sum(axis=1) > SLEEVE                    # cash account: cannot exceed the sleeve
        dollars[over] = pd.DataFrame({s: np.floor(target[s] / px[s]) * px[s] for s in legs})[over]
    pnl = (dollars * fr).sum(axis=1) - core.COST_RT * dollars.sum(axis=1)
    r = (pnl / SLEEVE)[L.index >= START]
    d = dollars[L.index >= START]
    s = core.stats(r)
    zero_legs = float(((d == 0) & (target[L.index >= START] > 0)).sum().sum() / max(1, (target[L.index >= START] > 0).sum().sum()))
    return s, zero_legs, float(d.sum(axis=1).max()), px


def main():
    rows = []
    for legs, cap in ((("SPY", "QQQ", "IWM"), 3.0), (("SPY", "QQQ", "IWM"), 2.0), (("SPY", "QQQ", "IWM", "SMH"), 2.0),
                      (("SPY", "QQQ", "IWM", "SMH", "XLF"), 2.0)):
        for rnd in ("none", "nearest", "floor"):
            s, z, mx, px = run(legs, cap, rnd)
            nm = f"{'/'.join(legs)} cap {cap:g} {rnd}"
            rows.append({"name": nm, **s, "zero_leg_share": z, "max_dollars": mx})
            print(f"{nm:<40} SR {s['sharpe']:+.2f} CAGR {s['cagr']:+6.1%} DD {s['maxdd']:+6.1%} worst ${1000 * s['worst']:+6.0f} "
                  f"| legs wanted but 0 shares {z:.0%} | max invested ${mx:.0f}")
    print("fund prices used:", {FUND[k]: round(v, 2) for k, v in px.items()})
    json.dump(rows, open(evalx.STORE / "exp_wholeshares.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
