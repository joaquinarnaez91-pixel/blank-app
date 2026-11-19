"""Does a real 3x fund's overnight return equal 3 x the index ETF's overnight return? And who bears the
financing carry: the overnight or the intraday holder?

For each pair (fund, ETF): d_on = fund_on - 3 * etf_on (both with dividends added back on ex-dates),
d_id = fund_id - 3 * etf_id * (1+etf_on)/(1+3 etf_on). Regress d_on on [cal_days/365, rf*cal_days/365].
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/overnight")
import core

PAIRS = [("UPRO", "SPY"), ("TQQQ", "QQQ"), ("TNA", "IWM"), ("SOXL", "SMH"), ("FAS", "XLF")]


def ols(y, X):
    X = np.column_stack(X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    s2 = e @ e / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov))


def main():
    rows = []
    for f, e in PAIRS:
        F, E = core.index_frame(f), core.index_frame(e)
        idx = F.index.intersection(E.index)
        F, E = F.reindex(idx), E.reindex(idx)
        rf = core.rf_series(idx).shift(1)
        df = pd.DataFrame({"f_on": F["r_on"], "e_on": E["r_on"], "f_id": F["r_id"], "e_id": E["r_id"],
                           "days": F["cal_days"], "rf": rf}).dropna()
        df = df.iloc[60:]  # skip launch weeks
        # drop absurd prints (bad opens) for the regression: |d| > 5%
        df["d_on"] = df.f_on - 3 * df.e_on
        df["d_id"] = df.f_id - 3 * df.e_id * (1 + df.e_on) / (1 + 3 * df.e_on)
        ok = (df.d_on.abs() < 0.05) & (df.d_id.abs() < 0.05)
        g = df[ok]
        b, se = ols(g.d_on.values, [g.days.values / 365, g.rf.values * g.days.values / 365])
        bi, sei = ols(g.d_id.values, [g.days.values / 365, g.rf.values * g.days.values / 365])
        print(f"\n{f} vs 3x{e}: nights {len(g)} (dropped {int((~ok).sum())} bad prints)")
        print(f"  overnight diff: mean {1e4*g.d_on.mean():+.2f}bp/night sd {1e4*g.d_on.std():.1f}bp | "
              f"fit d_on = -({-b[0]:.4f} + {-b[1]:.2f}*rf)*days/365  (se {se[0]:.4f}, {se[1]:.2f})")
        print(f"  intraday diff:  mean {1e4*g.d_id.mean():+.2f}bp/day   sd {1e4*g.d_id.std():.1f}bp | "
              f"fit d_id = ({bi[0]:+.4f} + {bi[1]:+.2f}*rf)*days/365  (se {sei[0]:.4f}, {sei[1]:.2f})")
        yr = g.groupby(g.index.year).agg(d_on=("d_on", "mean"), d_id=("d_id", "mean"), rf=("rf", "mean"))
        yr = (yr * [1e4, 1e4, 100]).round(2)
        print("  by year (bp/night overnight, bp/day intraday, rf %):")
        print("   " + " ".join(f"{y}:{r.d_on:+.1f}/{r.d_id:+.1f}/{r.rf:.1f}" for y, r in yr.iterrows()))
        wk = g.groupby(g.days.clip(upper=4)).d_on.mean() * 1e4
        print("  overnight diff by calendar days in the night (1=weekday, 3=weekend):", wk.round(2).to_dict())
        rows.append((f, e, len(g), 1e4 * g.d_on.mean(), b[0], b[1], 1e4 * g.d_id.mean(), bi[0], bi[1]))
    return rows


if __name__ == "__main__":
    main()
