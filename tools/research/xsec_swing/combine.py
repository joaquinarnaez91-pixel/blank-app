"""Diversification value and cash-account combination of a candidate with the live overnight index model.

  python3 tools/research/xsec_swing/combine.py <series.pkl> <column> [<series.pkl> <column>]
Each dollar is committed once per day: the index model buys at ~15:58 and sells at the 9:30 open; a candidate that
also trades close -> open uses the same window, so the two can only split the sleeve (static weights), or the
candidate takes the dollars on nights the index model holds cash (all three indices below their 200-day averages).
Weights are fixed in advance (not fitted): 50/50 static and "index model first, candidate when it is in cash".
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import report, summary, by_year


def main():
    a = sys.argv[1:]
    df = pd.read_pickle(a[0])
    idx = df["index_model"]
    x = df[a[1]]
    expo = df["index_expo"] if "index_expo" in df else (idx != 0).astype(float)
    both = pd.concat([idx, x, expo], axis=1, keys=["idx", "x", "e"]).dropna()
    idx, x = both.idx, both.x
    inv = both.e > 0                               # nights the index model holds a 3x fund
    print(f"candidate {a[1]}: corr with index model, all days {x.corr(idx):+.2f}; nights index model invested {x[inv].corr(idx[inv]):+.2f}")
    print(f"  candidate when index model in cash: mean {1e4*x[~inv].mean():+.1f}bp t {x[~inv].mean()/x[~inv].std()*math.sqrt((~inv).sum()):+.2f} "
          f"({(~inv).sum()} days); when invested: mean {1e4*x[inv].mean():+.1f}bp t {x[inv].mean()/x[inv].std()*math.sqrt(inv.sum()):+.2f}")
    report("index model alone", idx)
    report("candidate alone", x)
    report("50/50 static split", 0.5 * idx + 0.5 * x)
    report("index first, candidate when index in cash", idx.where(inv, x))
    report("50/50 when index invested, 100% candidate otherwise", (0.5 * idx + 0.5 * x).where(inv, x))


if __name__ == "__main__":
    main()
