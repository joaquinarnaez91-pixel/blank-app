"""Index intraday-momentum tests on hourly bars (Yahoo, ~2.9 years), traded long-only with 3x funds.
All returns are measured on the actual leveraged-ETF bars, net of COST per round trip."""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research")
from bars import frame
from stats import summary, by_year, fmt

COST = 0.0005  # 5 bp round trip: spread + slippage on liquid 3x ETFs (conservative)
CHK = [570, 630, 690, 750, 810, 870, 930]


def pivot(sym, field):
    df = frame(sym, "1h")
    full = df.groupby("date").tod.apply(lambda s: list(s) == CHK)
    df = df[df.date.isin(full[full].index)]
    return df.pivot_table(index="date", columns="tod", values=field)


def daily_close(sym):
    return pivot(sym, "c")[930]


results = {}


def record(name, r, index=None):
    if not isinstance(r, pd.Series):
        r = pd.Series(np.asarray(r, dtype=float), index=index)
    r = r.fillna(0.0)
    results[name] = r
    s = summary(r, name)
    print(fmt(s), "| by year", by_year(r))


def main():
    for und, bull, bear in [("SPY", "UPRO", "SPXU"), ("QQQ", "TQQQ", "SQQQ")]:
        O, H, L, C = (pivot(und, f) for f in "ohlc")
        bO, bH, bL, bC = (pivot(bull, f) for f in "ohlc")
        sO, sH, sL, sC = (pivot(bear, f) for f in "ohlc")
        days = O.index.intersection(bO.index).intersection(sO.index)
        O, H, L, C, bO, bH, bL, bC, sO, sH, sL, sC = (x.loc[days] for x in (O, H, L, C, bO, bH, bL, bC, sO, sH, sL, sC))
        prev = C[930].shift(1)
        print("\n==", und, "via", bull, "/", bear, ":", len(days) - 1, "days", days[1], "to", days[-1])

        # 0. Baselines: hold the bull fund 9:30 -> 16:00 every day; and the bear fund.
        record(f"{und} baseline: {bull} open->close every day", bC[930] / bO[570] - 1 - COST)

        # 1. Market intraday momentum (Gao-Han-Li-Zhou 2018, first hour version):
        #    sign of prev close -> 10:30 decides the 15:30 -> 16:00 trade.
        sig = C[570] / prev - 1
        last_bull = bC[930] / bO[930] - 1 - COST
        last_bear = sC[930] / sO[930] - 1 - COST
        record(f"{und} MIM: first hour sign -> last 30 min", np.where(sig > 0, last_bull, last_bear) * (sig.notna()), days)
        # predictive regression on the underlying itself
        y = (C[930] / O[930] - 1)
        ok = sig.notna()
        b = np.polyfit(sig[ok], y[ok], 1)
        resid = y[ok] - np.polyval(b, sig[ok])
        se = math.sqrt((resid ** 2).sum() / (ok.sum() - 2) / ((sig[ok] - sig[ok].mean()) ** 2).sum())
        r2 = 1 - (resid ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
        print(f"   regression last30 on first-hour: slope {b[0]:+.4f} t {b[0]/se:+.2f} R2 {r2:.3%}")
        # 1b. strong-signal only (|first hour| above its trailing 60-day median of |.|)
        thr = sig.abs().rolling(60, min_periods=20).median().shift(1)
        strong = sig.abs() > thr
        record(f"{und} MIM strong first hour only", np.where(strong, np.where(sig > 0, last_bull, last_bear), 0.0), days)
        # 1c. 14:30 bar momentum -> last 30 min
        sig2 = C[870] / O[870] - 1
        record(f"{und} 2:30-3:30 sign -> last 30 min", np.where(sig2 > 0, last_bull, last_bear), days)
        # 1d. day so far (prev close -> 15:30) sign -> last 30 min
        sig3 = C[870] / prev - 1
        record(f"{und} day-so-far sign -> last 30 min", np.where(sig3 > 0, last_bull, last_bear) * sig3.notna(), days)

        # 2. 60-minute opening-range breakout (direction of first hour candle), hold to close,
        #    stop at the fund's own first-hour low; entry 10:30 open.
        for stop_on in (True, False):
            rets = []
            for d in days:
                up = C.loc[d, 570] > O.loc[d, 570]
                fO, fL, fC, fLo = (bO, bL, bC, bL) if up else (sO, sL, sC, sL)
                entry = fO.loc[d, 630]
                stop = fLo.loc[d, 570]
                exitp = fC.loc[d, 930]
                if stop_on:
                    for k in CHK[1:]:
                        if fO.loc[d, k] <= stop:
                            exitp = fO.loc[d, k]; break
                        if fL.loc[d, k] <= stop:
                            exitp = stop; break
                rets.append(exitp / entry - 1 - COST)
            record(f"{und} 60-min ORB direction, hold to close{' +stop' if stop_on else ''}", pd.Series(rets, index=days))

        # 3. Noise-area momentum (Zarattini-Aziz-Barbon 2024), hourly checkpoints, long-only via 3x funds.
        mv = (C.div(O[570], axis=0) - 1).abs()           # |move from open| at each checkpoint close
        sigma = mv.rolling(14, min_periods=14).mean().shift(1)
        base_hi = pd.concat([O[570], prev], axis=1).max(axis=1)
        base_lo = pd.concat([O[570], prev], axis=1).min(axis=1)
        tp = (H + L + C) / 3
        vwap = (tp * pivot(und, "v").loc[days]).cumsum(axis=1) / pivot(und, "v").loc[days].cumsum(axis=1)
        rets = []
        for d in days:
            if sigma.loc[d].isna().any() or pd.isna(prev.loc[d]):
                rets.append(0.0); continue
            pos, entry, total = 0, None, 0.0
            for i, k in enumerate(CHK[:-1]):       # decide at bar close k, trade at next bar open
                ub = base_hi.loc[d] * (1 + sigma.loc[d, k]); lb = base_lo.loc[d] * (1 - sigma.loc[d, k])
                px, vw = C.loc[d, k], vwap.loc[d, k]
                nk = CHK[i + 1]
                if pos == 1 and px < max(ub, vw):
                    total += bO.loc[d, nk] / entry - 1 - COST; pos = 0
                elif pos == -1 and px > min(lb, vw):
                    total += sO.loc[d, nk] / entry - 1 - COST; pos = 0
                if pos == 0:
                    if px > ub:
                        pos, entry = 1, bO.loc[d, nk]
                    elif px < lb:
                        pos, entry = -1, sO.loc[d, nk]
            if pos == 1:
                total += bC.loc[d, 930] / entry - 1 - COST
            elif pos == -1:
                total += sC.loc[d, 930] / entry - 1 - COST
            rets.append(total)
        record(f"{und} noise-area momentum (hourly), VWAP trail", pd.Series(rets, index=days))

    pd.DataFrame(results).to_csv("trader/data/etf_momo_daily.csv")


if __name__ == "__main__":
    main()
