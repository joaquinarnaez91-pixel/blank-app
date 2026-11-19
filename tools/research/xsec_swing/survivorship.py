"""How large could the survivorship bias be?

  python3 tools/research/xsec_swing/survivorship.py
1. Equal-weight point-in-time top-500 universe (daily close-to-close total return, overnight and intraday parts)
   built from (i) today's superset only (price >= $5 and >= 750k shares today) and (ii) every name listed today,
   compared with RSP (equal-weight S&P 500 ETF; it holds the names that were later delisted).
2. Per-year differences, so a bias concentrated in the early years shows up.
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, shift, summary


def ew(d, U, R, lag=True):
    M = shift(U, 1) if lag else U
    with np.errstate(all="ignore"):
        return pd.Series(np.nanmean(np.where(M, R, np.nan), axis=1), index=d.dates)


def main():
    d = Data()
    sup = {x["symbol"].upper().replace(".", "-") for x in json.load(open("trader/data/superset.json"))}
    t0 = 260
    sl = lambda s: s.iloc[t0 + 1:]
    out = {}
    for tag, names in (("superset", sup), ("all_listed", None)):
        U = d.universe(500, names=names)
        out[tag] = {"cc": sl(ew(d, U, d.RCC)), "on": sl(ew(d, U, d.RON)), "id": sl(ew(d, U, d.RID, lag=False))}
        held = np.where(U[t0:], 1, 0).sum(0) > 0
        print(f"{tag}: distinct names ever in the top 500: {held.sum()}")
    j = d.s("RSP")
    rsp = {"cc": sl(d.series(d.RCC[:, j])), "on": sl(d.series(d.RON[:, j])), "id": sl(d.series(d.RID[:, j]))}
    spy = d.s("SPY")
    print("\nAnnualized mean (252 x daily mean), %:")
    for part in ("cc", "on", "id"):
        line = f"  {part}: RSP {252*100*rsp[part].mean():+6.2f}"
        for tag in out:
            diff = out[tag][part] - rsp[part]
            t = diff.mean() / diff.std() * math.sqrt(len(diff))
            line += f" | {tag} {252*100*out[tag][part].mean():+6.2f} (minus RSP {252*100*diff.mean():+5.2f}, t {t:+.2f})"
        print(line)
    print("\nBy year, close-to-close excess over RSP (%/yr), superset vs all listed:")
    for y in sorted(set(rsp["cc"].index.year)):
        m = rsp["cc"].index.year == y
        a = 252 * 100 * (out["superset"]["cc"][m] - rsp["cc"][m]).mean()
        b = 252 * 100 * (out["all_listed"]["cc"][m] - rsp["cc"][m]).mean()
        print(f"  {y}: superset {a:+6.2f}  all listed {b:+6.2f}  difference {a-b:+5.2f}")
    diff = out["superset"]["cc"] - out["all_listed"]["cc"]
    print("\nSuperset minus all-listed universe: %+.2f%%/yr (t %.2f); overnight part %+.2f%%/yr, intraday %+.2f%%/yr" % (
        252 * 100 * diff.mean(), diff.mean() / diff.std() * math.sqrt(len(diff)),
        252 * 100 * (out["superset"]["on"] - out["all_listed"]["on"]).mean(),
        252 * 100 * (out["superset"]["id"] - out["all_listed"]["id"]).mean()))
    json.dump({tag: {p: 252 * float((out[tag][p] - rsp[p]).mean()) for p in ("cc", "on", "id")} for tag in out},
              open("trader/data/xsec_survivorship.json", "w"))


if __name__ == "__main__":
    main()
