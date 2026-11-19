"""Recommended overnight spec (= the live spec with a per-leg exposure cap of 2.0 instead of 3.0, plus a pre-trade
stress guard) -- full statistics on every window, and a reference implementation of the 15:58 decision.

  python3 tools/research/overnight/final_spec.py            # stats
  python3 tools/research/overnight/final_spec.py --decide   # targets from the latest cached closes (reference)
Rule (per index i in SPY/QQQ/IWM, fund UPRO/TQQQ/TNA, sleeve S, decided ~15:58 ET, sold at the next 9:30 open):
  trend_i = price_i > mean(last 200 daily closes incl. today's price)
  sigma_i = sd (ddof=1) of the last 20 overnight returns (open + dividend on ex-date) / prior close - 1
  L_i     = min(CAP, 0.015 / sigma_i) if trend_i else 0          CAP = 2.0
  D_i     = (S/3) * L_i / 3   dollars in the 3x fund
  guard   : stress = sum_i 3 * D_i * G_i with G = worst overnight gap seen while above the 200-day average
            (SPY 4.0%, QQQ 7.0%, IWM 5.3% as of Nov 2026); if stress > $100 (per $1,000), scale every D_i by 100/stress
  shares  = round(D_i / fund price); if the total exceeds cash, floor instead.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, evalx, validate  # noqa: E402
from exp_sizing import stress_gaps, gross_cap  # noqa: E402
from exp_realfunds import FUND, real_returns  # noqa: E402

CAP = 2.0
G_NOW = {"SPY": 0.040, "QQQ": 0.070, "IWM": 0.053}


def recommended_L(P, cap=CAP, pit_guard=True):
    L = core.spec_exposure(P, cap=cap)
    if pit_guard:
        G = stress_gaps(P)
    else:
        G = pd.DataFrame({s: G_NOW[s] for s in P["c"].columns}, index=P["c"].index)
    return gross_cap(L, G, limit=0.10)


def main():
    P = core.panel(("SPY", "QQQ", "IWM"))
    r0, ff0 = core.sleeve_returns(P, core.spec_exposure(P))
    L1 = recommended_L(P)
    r1, ff1 = core.sleeve_returns(P, L1)
    out = {}
    print("Recommended (cap 2 + stress guard) vs current spec, proxy returns with realistic costs")
    for wn, a in (("2001-03..2026-11", "2001-03-15"), ("pre 2003-2007", "2003-01-01"), ("OOS 2008-2026", "2008-01-01"),
                  ("spec window 2017-09..", "2017-09-08")):
        x1 = evalx.window(r1, a); x0 = evalx.window(r0, a)
        if wn == "pre 2003-2007":
            x1, x0 = x1[x1.index < "2008-01-01"], x0[x0.index < "2008-01-01"]
        s1, s0 = core.stats(x1), core.stats(x0)
        h = core.halves(x1)
        d = validate.paired_sr_diff(x1.values, x0.reindex(x1.index).fillna(0).values, B=2000, b=20)
        out[wn] = {"rec": s1, "spec": s0, "halves": [h[0]["sharpe"], h[1]["sharpe"]], "dsr_diff": d}
        print(f"  {wn:<22} REC  " + core.fmt(s1)[39:] + f" halves {h[0]['sharpe']:.2f}/{h[1]['sharpe']:.2f}")
        print(f"  {'':<22} SPEC " + core.fmt(s0)[39:] + f"  dSR {d['d_sr']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}] p {d['p']:.3f}")
    ci = validate.boot_ci(evalx.window(r1, "2008-01-01").values, B=4000, b=20)
    print(f"  bootstrap OOS 2008-26: SR {ci['sr']:.2f} [{ci['sr_ci'][0]:.2f},{ci['sr_ci'][2]:.2f}], "
          f"CAGR {ci['cagr']:+.1%} [{ci['cagr_ci'][0]:+.1%},{ci['cagr_ci'][2]:+.1%}]")
    out["boot_oos"] = ci
    # real funds
    rr1, _ = real_returns(P, L1); rr0, _ = real_returns(P, core.spec_exposure(P))
    a1, a0 = core.stats(rr1[rr1.index >= "2010-06-01"]), core.stats(rr0[rr0.index >= "2010-06-01"])
    print("  real 3x funds 2010-06..: REC " + core.fmt(a1)[39:])
    print("                           SPEC" + core.fmt(a0)[39:])
    out["real"] = {"rec": a1, "spec": a0}
    # worst nights of the recommended rule
    w = r1[r1.index >= "2001-03-15"].nsmallest(5)
    print("  worst nights (rec):", [(d.date().isoformat(), f"{100 * v:.2f}%") for d, v in w.items()])
    # carry / costs share
    cy = core.carry(P)
    drag = (ff1.mul(cy, axis=0).sum(axis=1))[r1.index >= "2008-01-01"].mean() * 252
    cost = (core.COST_RT * ff1.sum(axis=1))[r1.index >= "2008-01-01"].mean() * 252
    print(f"  average annual drag OOS: financing carry {drag:.2%}, trading costs {cost:.2%} of the sleeve")
    out["drag"] = {"carry": drag, "cost": cost}
    json.dump(out, open(evalx.STORE / "final_spec.json", "w"), default=float, indent=1)


def decide(sleeve=1000.0):
    """Targets from the latest cached closes (stand-in for the 15:58 prices)."""
    P = core.panel(("SPY", "QQQ", "IWM"))
    c, r_on = P["c"], P["r_on"]
    sma = c.iloc[-200:].mean()
    trend = c.iloc[-1] > sma
    sig = r_on.iloc[-20:].std()
    L = (0.015 / sig).clip(upper=CAP) * trend
    D = sleeve / 3 * L / 3
    stress = sum(3 * D[s] * G_NOW[s] for s in D.index)
    k = min(1.0, 0.10 * sleeve / stress) if stress > 0 else 1.0
    D = D * k
    rows = []
    for s in D.index:
        px = core.daily(FUND[s])["c"].iloc[-1]
        n = int(round(D[s] / px))
        rows.append((s, FUND[s], bool(trend[s]), round(float(sig[s]) * 100, 3), round(float(L[s]), 2),
                     round(float(D[s]), 2), round(float(px), 2), n))
    if sum(r[-1] * r[-2] for r in rows) > sleeve:
        rows = [(*r[:-1], int(np.floor(r[5] / r[6]))) for r in rows]
    print(f"as of close {c.index[-1].date()}: stress ${stress:.0f} -> scale {k:.2f}")
    for r in rows:
        print("  %s->%s trend %s sigma20 %.3f%% L %.2f target $%.0f price $%.2f shares %d" % r)
    return rows


if __name__ == "__main__":
    if "--decide" in sys.argv:
        decide()
    else:
        main()
