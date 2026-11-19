"""(b) Regime / trend signals vs the 200-day filter, out of sample (2008-01 -> 2026-11).

Each signal either REPLACES the 200-day filter or is ANDed with it; sizing is the live spec's
(1.5% / sigma20 overnight target, cap 3, equal thirds SPY/QQQ/IWM via UPRO/TQQQ/TNA), realistic costs
(4 bp round trip + measured 3x-fund financing carry, dividend-corrected opens).
Fitted models (Kalman MLE, HMM) are refit every Jan 1 on prior data only; their signals are filtered (causal).
A meta-rule also picks, each Jan 1, the variant with the best Sharpe so far, to test whether choosing among
these signals beats the spec out of sample.
"""
import sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
sys.path.insert(0, "tools/research")
import core, signals as sg, evalx  # noqa: E402


def main():
    t0 = time.time()
    P = core.panel(("SPY", "QQQ", "IWM"))
    sma = sg.sma_trend(P, 200)
    L0 = core.spec_exposure(P)
    r0, ff0 = core.sleeve_returns(P, L0)

    print("computing signals ...", flush=True)
    S = {}
    S["tsmom_majority"] = sg.tsmom(P, mode="majority")
    S["tsmom_prop"] = sg.tsmom(P, mode="prop")
    S["tsmom_all"] = sg.tsmom(P, mode="all")
    S["kalman_mle"] = sg.kalman_slope(P, mode="mle")
    S["kalman_fixed_1e-4"] = sg.kalman_slope(P, mode="fixed", ratio=1e-4)
    S["kalman_fixed_1e-5"] = sg.kalman_slope(P, mode="fixed", ratio=1e-5)
    print("  kalman done %.0fs" % (time.time() - t0), flush=True)
    hmm = sg.hmm_calm(P)
    S["hmm_calm>0.5"] = (hmm > 0.5).astype(float).where(hmm.notna())
    S["hmm_prop"] = hmm
    print("  hmm done %.0fs" % (time.time() - t0), flush=True)
    S["vix_ts<1"] = sg.vix_term(P, 1.0)
    S["vix<ma50"] = sg.vix_ma(P, 50)
    S["vix<ma20"] = sg.vix_ma(P, 20)
    S["credit_hyg_ief>ma100"] = sg.credit_trend(P, 100)
    S["credit_hyg_ief>ma50"] = sg.credit_trend(P, 50)

    variants = {"spec_sma200": r0}
    rows = []
    # unfiltered reference (vol-targeted, always on)
    one = pd.DataFrame(1.0, index=P["c"].index, columns=P["c"].columns)
    r, ff = core.sleeve_returns(P, core.spec_exposure(P, trend=one))
    variants["no_filter"] = r
    rows.append(evalx.evaluate("no_filter (vol target only)", r, r0, ff))
    for k, sig in S.items():
        for mode in ("replace", "and_sma"):
            if mode == "replace":
                tr = sig
            else:
                tr = sig * sma
            r, ff = core.sleeve_returns(P, core.spec_exposure(P, trend=tr))
            nm = f"{k}|{mode}"
            variants[nm] = r
            rows.append(evalx.evaluate(nm, r, r0, ff))
    # majority vote of 5 diverse filters
    vote = (sma.fillna(0) + S["tsmom_majority"].fillna(0) + S["hmm_calm>0.5"].fillna(0)
            + S["vix_ts<1"].fillna(0) + S["credit_hyg_ief>ma100"].fillna(0))
    tr = (vote >= 3).astype(float)
    r, ff = core.sleeve_returns(P, core.spec_exposure(P, trend=tr))
    variants["vote5>=3"] = r
    rows.append(evalx.evaluate("vote5>=3", r, r0, ff))
    tr = vote / 5.0
    r, ff = core.sleeve_returns(P, core.spec_exposure(P, trend=tr))
    variants["vote5_prop"] = r
    rows.append(evalx.evaluate("vote5_prop", r, r0, ff))

    base = evalx.evaluate("spec_sma200 (benchmark)", r0, r0, ff0, boot=False)
    print("\nOOS 2008-01 -> 2026-11, realistic costs. (h1/h2) = Sharpe in the two halves; dSR vs spec, paired "
          "stationary bootstrap (b=20), p = P(dSR<=0)")
    print(evalx.show(base))
    for o in sorted(rows, key=lambda o: -o["sharpe"]):
        print(evalx.show(o))

    # meta: each Jan 1 pick the variant with the best Sharpe from 2001-06 (or its start) to date
    names = list(variants)
    R = pd.DataFrame(variants).loc["2001-06-01":]
    picks, meta = [], []
    for y in range(2008, 2027):
        hist = R[R.index < f"{y}-01-01"]
        ok = hist.notna().sum() >= 504
        srs = (hist.mean() / hist.std() * np.sqrt(252)).where(ok)
        best = srs.idxmax()
        picks.append((y, best, round(float(srs[best]), 2)))
        meta.append(R.loc[f"{y}-01-01":f"{y}-12-31", best])
    meta = pd.concat(meta)
    variants["meta_select_best_so_far"] = meta
    mo = evalx.evaluate("META pick best-so-far each Jan 1", meta, r0)
    print("\n" + evalx.show(mo))
    print("picks:", picks)
    evalx.save_variants("signals", variants)
    json.dump({"rows": rows, "base": base, "meta": mo, "picks": picks},
              open(evalx.STORE / "exp_signals.json", "w"), default=float, indent=1)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
