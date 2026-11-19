"""(e) Asset set: add a semiconductor leg (SOXL, simulated as 3x SMH) and/or a financials leg (FAS, simulated as
3x XLF) to the SPY/QQQ/IWM thirds. Same rules per leg (200-day filter, 1.5% / sigma20 target, cap), equal
sleeve weights across legs, realistic costs (4 bp + carry). OOS 2008-2026, both halves, 2003-2007 check.
Caveat: SOXL's index (ICE/PHLX semiconductor) differs from SMH's; nightly tracking noise is large (tracking.py).
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, evalx  # noqa: E402


def main():
    P5 = core.panel(("SPY", "QQQ", "IWM", "SMH", "XLF"))
    P3 = core.panel(("SPY", "QQQ", "IWM"))
    r0, ff0 = core.sleeve_returns(P3, core.spec_exposure(P3))
    rows, variants = [], {}
    # descriptive: each index's own overnight series under the rule (one leg = whole sleeve)
    print("Single-leg (whole sleeve) under the spec rules, OOS 2008-2026:")
    for s in P5["c"].columns:
        Ps = {k: (v[[s]] if isinstance(v, pd.DataFrame) else v) for k, v in P5.items()}
        r, ff = core.sleeve_returns(Ps, core.spec_exposure(Ps))
        o = evalx.evaluate(f"{s} alone", r, r0, ff, boot=False)
        print("  " + evalx.show(o))
    corr = P5["r_on"][P5["r_on"].index >= evalx.OOS].corr().round(2)
    print("overnight return correlations OOS:\n", corr)
    for cap in (3.0, 2.0):
        for legs in (("SPY", "QQQ", "IWM", "SMH"), ("SPY", "QQQ", "IWM", "XLF"), ("SPY", "QQQ", "IWM", "SMH", "XLF")):
            Ps = {k: (v[list(legs)] if isinstance(v, pd.DataFrame) else v) for k, v in P5.items()}
            L = core.spec_exposure(Ps, cap=cap)
            r, ff = core.sleeve_returns(Ps, L)
            nm = f"{'+'.join(legs[3:])} legs, cap {cap:g}"
            variants[nm] = r
            o = evalx.evaluate(nm, r, r0, ff)
            pre = (r.index >= "2003-01-01") & (r.index < "2008-01-01")
            o["pre_sr"] = core.stats(r[pre])["sharpe"]
            rows.append(o)
    L3c2 = core.spec_exposure(P3, cap=2.0)
    r3c2, ff3c2 = core.sleeve_returns(P3, L3c2)
    o = evalx.evaluate("3 legs, cap 2 (reference)", r3c2, r0, ff3c2)
    o["pre_sr"] = core.stats(r3c2[(r3c2.index >= "2003-01-01") & (r3c2.index < "2008-01-01")])["sharpe"]
    base = evalx.evaluate("spec 3 legs cap 3 (benchmark)", r0, r0, ff0, boot=False)
    base["pre_sr"] = core.stats(r0[(r0.index >= "2003-01-01") & (r0.index < "2008-01-01")])["sharpe"]
    print("\nAsset-set variants, OOS 2008-01 -> 2026-11")
    for x in [base, o] + sorted(rows, key=lambda q: -q["sharpe"]):
        print(evalx.show(x) + f" | pre03-07 SR {x['pre_sr']:+.2f}")
    evalx.save_variants("assets", variants)
    json.dump({"rows": rows, "base": base, "ref_cap2": o}, open(evalx.STORE / "exp_assets.json", "w"),
              default=float, indent=1)


if __name__ == "__main__":
    main()
