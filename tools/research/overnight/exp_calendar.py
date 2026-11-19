"""(c) Calendar / event structure of overnight returns, and whether skipping or sizing up such nights helps
out of sample.

Descriptive: equal-weight SPY/QQQ/IWM overnight total return on flagged vs other nights (Welch t), on all
nights and on nights the live spec is invested (trend on). The direction of each rule (skip vs size up) is
chosen on 2001-2007 (pre-sample) and judged on 2008-2026 (out of sample), with both halves.
"""
import sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core, evalx, calendar_flags as cf  # noqa: E402

FLAGS = ["mon", "tue", "wed", "thu", "fri", "weekend", "holiday_night", "pre_holiday", "tom", "opex_morning",
         "post_opex", "jobs", "fomc_day", "cpi"]


def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), float("nan")
    t = (a.mean() - b.mean()) / math.sqrt(a.var() / len(a) + b.var() / len(b))
    return 1e4 * a.mean(), 1e4 * b.mean(), t


def main():
    P = core.panel(("SPY", "QQQ", "IWM"))
    idx = P["c"].index
    F = cf.flags(idx)
    ew = P["r_on"].mean(axis=1)
    L0 = core.spec_exposure(P)
    on = (L0.fillna(0).sum(axis=1) > 0)
    r0, ff0 = core.sleeve_returns(P, L0)
    periods = {"pre 2001-07": ("2001-03-15", "2008-01-01"), "OOS 2008-26": ("2008-01-01", "2027-01-01"),
               "OOS h1 2008-17": ("2008-01-01", "2017-06-01"), "OOS h2 2017-26": ("2017-06-01", "2027-01-01")}
    desc = {}
    print("Equal-weight index overnight return, flagged vs other nights (bp, Welch t). [trend-on nights only]")
    for f in FLAGS:
        line = f"{f:<14}"
        desc[f] = {}
        for pn, (a, b) in periods.items():
            m = (idx >= a) & (idx < b) & on.values
            if f == "cpi":
                m = m & F["cpi_sample"].values
            fl = F[f].values
            x1, x0, t = welch(ew[m & fl], ew[m & ~fl])
            n1 = int((m & fl).sum())
            desc[f][pn] = (x1, x0, t, n1)
            line += f" | {pn}: {x1:+6.1f} vs {x0:+5.1f} t {t:+5.2f} n {n1:4d}"
        print(line)

    # sleeve-level rules
    rows, variants = [], {}
    pre_mask = (idx >= "2001-03-15") & (idx < "2008-01-01")
    for f in FLAGS:
        fl = F[f].reindex(L0.index).fillna(False)
        for act, mult in (("skip", 0.0), ("x1.5", 1.5)):
            L = L0.mul(np.where(fl, mult, 1.0), axis=0).clip(upper=3.0)
            r, ff = core.sleeve_returns(P, L)
            nm = f"{f}|{act}"
            variants[nm] = r
            o = evalx.evaluate(nm, r, r0, ff)
            pre = core.stats(r[pre_mask]); pre0 = core.stats(r0[pre_mask])
            o["pre_sr"] = pre["sharpe"]; o["pre_sr_bench"] = pre0["sharpe"]
            # pre-registered direction from 2001-2007 (CPI has no pre-sample: undetermined)
            o["chosen_on_pre"] = (pre["sharpe"] > pre0["sharpe"]) if f != "cpi" else None
            rows.append(o)
    base = evalx.evaluate("spec (benchmark)", r0, r0, ff0, boot=False)
    print("\nRules on the live spec. OOS 2008-01 -> 2026-11 (h1/h2), pre = Sharpe 2001-2007 (bench %.2f)"
          % core.stats(r0[pre_mask])["sharpe"])
    print(evalx.show(base))
    for o in sorted(rows, key=lambda o: -o["sharpe"]):
        tag = {True: "PRE-CHOSEN", False: "", None: "n/a"}[o["chosen_on_pre"]]
        print(evalx.show(o) + f" | pre SR {o['pre_sr']:+.2f} {tag}")
    evalx.save_variants("calendar", variants)
    json.dump({"desc": {f: {k: list(v) for k, v in d.items()} for f, d in desc.items()}, "rows": rows,
               "base": base}, open(evalx.STORE / "exp_calendar.json", "w"), default=float, indent=1)
    # event counts sanity
    print("\nflag counts (all nights 2000-2026):", {f: int(F[f].sum()) for f in FLAGS})


if __name__ == "__main__":
    main()
