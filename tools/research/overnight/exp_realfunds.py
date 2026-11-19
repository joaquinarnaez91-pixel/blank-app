"""Realism check: re-run the key variants with the ACTUAL 3x funds' overnight returns (open_t + div_t)/close_{t-1}
from Yahoo (UPRO, TQQQ, TNA, SOXL, FAS; all exist from 2010-03-11), instead of 3 x index - modelled carry.
Signals still come from the index ETFs (decided at the previous close). Costs: 4 bp round trip.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, evalx, signals as sg  # noqa: E402

FUND = {"SPY": "UPRO", "QQQ": "TQQQ", "IWM": "TNA", "SMH": "SOXL", "XLF": "FAS"}
START = "2010-06-01"


def real_returns(P, L, cost_rt=core.COST_RT):
    cols = list(L.columns)
    w = 1.0 / len(cols)
    fr = pd.DataFrame({s: core.index_frame(FUND[s])["r_on"].reindex(L.index) for s in cols})
    frac = L.fillna(0.0) * w / 3.0
    r = (frac * fr.fillna(0.0)).sum(axis=1) - cost_rt * frac.sum(axis=1)
    return r, frac


def main():
    P5 = core.panel(("SPY", "QQQ", "IWM", "SMH", "XLF"))
    rows = []
    out = {}
    for legs in (("SPY", "QQQ", "IWM"), ("SPY", "QQQ", "IWM", "SMH"), ("SPY", "QQQ", "IWM", "XLF"),
                 ("SPY", "QQQ", "IWM", "SMH", "XLF")):
        Ps = {k: (v[list(legs)] if isinstance(v, pd.DataFrame) else v) for k, v in P5.items()}
        for cap in (3.0, 2.0):
            for tname in ("sma200", "tsmom"):
                tr = None if tname == "sma200" else sg.tsmom(Ps, mode="majority")
                L = core.spec_exposure(Ps, cap=cap, trend=tr)
                rr, fr = real_returns(Ps, L)
                rp, fp = core.sleeve_returns(Ps, L)
                rr, rp = rr[rr.index >= START], rp[rp.index >= START]
                nm = f"{len(legs)} legs {'+'.join(legs[3:]) or 'base'} cap {cap:g} {tname}"
                out[nm] = rr
                a, b = core.stats(rr), core.stats(rp)
                h1, h2 = core.halves(rr)
                rows.append((nm, a, b, h1["sharpe"], h2["sharpe"]))
    base = out["3 legs base cap 3 sma200"]
    print(f"Actual 3x funds, {START} -> 2026-11 (proxy = 3x index - carry, same nights). SR halves in ().")
    for nm, a, b, s1, s2 in rows:
        print(f"{nm:<34} REAL mean {a['mean_bp']:+5.2f}bp t {a['t']:+5.2f} SR {a['sharpe']:+5.2f} ({s1:+.2f}/{s2:+.2f})"
              f" CAGR {a['cagr']:+6.1%} DD {a['maxdd']:+6.1%} worst {a['worst']:+6.2%} | PROXY SR {b['sharpe']:+5.2f}"
              f" CAGR {b['cagr']:+6.1%}")
    import validate
    print("\npaired bootstrap SR difference vs real 3-leg cap-3 spec:")
    for nm, r in out.items():
        if nm == "3 legs base cap 3 sma200":
            continue
        a, b = r.align(base, join="inner")
        d = validate.paired_sr_diff(a.values, b.values, B=2000, b=20)
        print(f"  {nm:<34} dSR {d['d_sr']:+.2f} CI [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}] p {d['p']:.3f}")
    # SOXL vs 3x SMH tracking by year (overnight, bp/night)
    so, sm = core.index_frame("SOXL"), core.index_frame("SMH")
    idx = so.index.intersection(sm.index)
    d = (so["r_on"].reindex(idx) - 3 * sm["r_on"].reindex(idx)).dropna()
    d = d[(d.index >= START) & (d.abs() < 0.05)]
    print("\nSOXL - 3xSMH overnight, bp/night by year:", (1e4 * d.groupby(d.index.year).mean()).round(1).to_dict())
    evalx.save_variants("realfunds", out)
    json.dump([(nm, a, b, s1, s2) for nm, a, b, s1, s2 in rows], open(evalx.STORE / "exp_realfunds.json", "w"),
              default=float, indent=1)


if __name__ == "__main__":
    main()
