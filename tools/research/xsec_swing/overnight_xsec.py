"""(a) Overnight persistence (Aboody, Even-Tov, Lehavy & Trueman 2018) and (b) tug of war / overnight momentum
(Lou, Polk & Skouras 2019), long only, close -> next open, top-N of the point-in-time top-500 by dollar volume.

  python3 tools/research/xsec_swing/overnight_xsec.py
Costs: 10 bp per night (a full round trip each night: buy ~15:58, sell at the 9:30 open).
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, index_model, rolling_mean, shift, report, summary, by_year, halves, dsr, fmt

COST = 0.0010
OUT = {}


def topn_overnight(d, S, U, n, cost=COST, gate=None, asc=False):
    """Hold the n highest-S names of universe U from the close of t to the open of t+1 (equal weight).
    Returns net nightly series indexed by the morning of t+1, plus the picks' gross overnight and next-day intraday return."""
    T = len(d.dates)
    gross = np.full(T, np.nan); intra = np.full(T, np.nan); nn = np.zeros(T)
    picks = {}
    for t in range(T - 1):
        if gate is not None and not gate[t]:
            continue
        m = U[t] & np.isfinite(S[t]) & np.isfinite(d.C[t])
        idx = np.flatnonzero(m)
        if len(idx) < n:
            continue
        s = S[t, idx]
        sel = idx[np.argsort(s if asc else -s, kind="stable")[:n]]
        r = d.RON[t + 1, sel]
        ok = np.isfinite(r)
        if ok.sum() == 0:
            continue
        gross[t + 1] = r[ok].mean()
        intra[t + 1] = np.nanmean(d.RID[t + 1, sel])
        nn[t + 1] = ok.sum()
        picks[t] = sel
    g = pd.Series(gross, index=d.dates)
    net = (g - cost).fillna(0.0)
    return net, g, pd.Series(intra, index=d.dates), picks


def main():
    d = Data()
    U = d.universe(500)
    start = d.dates[260]          # first date with 12-month momentum and 200-day filters available
    sl = lambda s: s[s.index >= start]
    print("dates", d.dates[0].date(), d.dates[-1].date(), "stocks", (~d.is_etf).sum(), "start", start.date())
    print("universe size per day (median):", int(np.median(U.sum(1)[260:])))

    # --- benchmarks -------------------------------------------------------------------------------
    idx, legs = index_model(d)
    idx_raw, _ = index_model(d, use_funds=False)
    print("\n== Benchmarks (net) ==")
    report("Live index model (3x funds, 4bp)", sl(idx))
    report("Live index model (3x underlying, 4bp)", sl(idx_raw))
    spy_on = d.series(d.RON[:, d.s("SPY")]).fillna(0)
    report("SPY overnight 1x (gross)", sl(spy_on))
    ew_on = pd.Series(np.nanmean(np.where(shift(U, 1), d.RON, np.nan), axis=1), index=d.dates).fillna(0)
    ew_id = pd.Series(np.nanmean(np.where(U, d.RID, np.nan), axis=1), index=d.dates).fillna(0)
    report("Universe EW overnight (gross)", sl(ew_on))
    report("Universe EW overnight net 10bp", sl(ew_on - COST))
    report("Universe EW intraday (gross)", sl(ew_id))
    print("worst night: index model %.2f%%, SPY %.2f%%" % (100 * sl(idx).min(), 100 * sl(spy_on).min()))

    # --- (a) overnight persistence -----------------------------------------------------------------
    print("\n== (a) Overnight persistence: rank by past k-day mean overnight return, hold top N close->open ==")
    RON = d.RON
    res = {}
    for k in (5, 10, 15, 20):
        S = rolling_mean(RON, k, k)          # nights through the morning of t: known at the close of t
        for n in (5, 10, 20, 50):
            net, g, intra, _ = topn_overnight(d, S, U, n)
            name = f"a_on{k}_top{n}"
            res[name] = sl(net)
            s = summary(sl(net))
            print(f"  {name:<14} gross {1e4*sl(g).mean():+6.1f}bp  net {s['mean_bp']:+6.1f}bp  SR {s['sharpe']:+5.2f}  t {s['t']:+5.2f}"
                  f"  maxDD {s['maxdd']:+6.1%}  next-day intraday {1e4*sl(intra).mean():+6.1f}bp  worst {100*sl(net).min():+.1f}%")
    # decile spread (gross), to replicate the persistence itself
    print("\n  Persistence check (gross overnight next night by decile of past-20-night overnight return):")
    S = rolling_mean(RON, 20, 20)
    dec = np.full((len(d.dates), 10), np.nan)
    for t in range(260, len(d.dates) - 1):
        m = U[t] & np.isfinite(S[t]) & np.isfinite(RON[t + 1])
        idxs = np.flatnonzero(m)
        q = pd.Series(S[t, idxs]).rank(pct=True).values
        b = np.minimum((q * 10).astype(int), 9)
        for j in range(10):
            dec[t + 1, j] = RON[t + 1, idxs[b == j]].mean()
    D = pd.DataFrame(dec, index=d.dates).iloc[261:]
    print("   decile mean bp:", " ".join(f"{1e4*x:+.1f}" for x in D.mean()))
    spread = D[9] - D[0]
    print("   top-bottom spread %.1f bp/night, t %.2f" % (1e4 * spread.mean(), spread.mean() / spread.std() * math.sqrt(len(spread))))

    # --- (b) tug of war ------------------------------------------------------------------------------
    print("\n== (b) Tug of war: 12-1 month momentum, overnight vs intraday components ==")
    C = d.C
    mom = shift(C, 21) / shift(C, 252) - 1                       # 12-1 month price momentum at the close of t
    lon = np.log1p(np.where(np.isfinite(RON), RON, 0.0))
    on12 = pd.DataFrame(lon).rolling(252, min_periods=200).sum().values   # 12-month cumulative overnight log return
    decm = {"on": np.full((len(d.dates), 10), np.nan), "id": np.full((len(d.dates), 10), np.nan)}
    for t in range(260, len(d.dates) - 1):
        m = U[t] & np.isfinite(mom[t]) & np.isfinite(RON[t + 1]) & np.isfinite(d.RID[t + 1])
        idxs = np.flatnonzero(m)
        q = pd.Series(mom[t, idxs]).rank(pct=True).values
        b = np.minimum((q * 10).astype(int), 9)
        for j in range(10):
            decm["on"][t + 1, j] = RON[t + 1, idxs[b == j]].mean()
            decm["id"][t + 1, j] = d.RID[t + 1, idxs[b == j]].mean()
    for part in ("on", "id"):
        Dm = pd.DataFrame(decm[part], index=d.dates).iloc[261:]
        sp = Dm[9] - Dm[0]
        print(f"   {part}: decile mean bp " + " ".join(f"{1e4*x:+.1f}" for x in Dm.mean()) +
              f" | winners-losers {1e4*sp.mean():+.1f}bp t {sp.mean()/sp.std()*math.sqrt(len(sp)):+.2f}")
    for nm, S in (("mom12_1", mom), ("on12", on12)):
        for n in (5, 10, 20, 50):
            net, g, intra, _ = topn_overnight(d, S, U, n)
            name = f"b_{nm}_top{n}"
            res[name] = sl(net)
            s = summary(sl(net))
            print(f"  {name:<16} gross {1e4*sl(g).mean():+6.1f}bp  net {s['mean_bp']:+6.1f}bp  SR {s['sharpe']:+5.2f}  t {s['t']:+5.2f}"
                  f"  maxDD {s['maxdd']:+6.1%}  next-day intraday {1e4*sl(intra).mean():+6.1f}bp  worst {100*sl(net).min():+.1f}%")

    # --- trend-gated versions (trade only when SPY closes above its 200-day average, like the index model) ---
    print("\n== Gated by SPY > 200-day average (as the live model) ==")
    spy = C[:, d.s("SPY")]
    gate = spy > rolling_mean(spy[:, None], 200)[:, 0]
    for nm, S in (("a_on20", rolling_mean(RON, 20, 20)), ("a_on5", rolling_mean(RON, 5, 5)), ("b_mom12_1", mom)):
        for n in (10, 50):
            net, g, intra, _ = topn_overnight(d, S, U, n, gate=gate)
            name = f"{nm}_top{n}_gated"
            res[name] = sl(net)
            s = summary(sl(net))
            print(f"  {name:<22} net {s['mean_bp']:+6.1f}bp  SR {s['sharpe']:+5.2f}  t {s['t']:+5.2f}  maxDD {s['maxdd']:+6.1%}  ann {s['ann_ret']:+6.1%}")

    srs = {k: summary(v)["sharpe"] for k, v in res.items()}
    var = float(np.var(list(srs.values()), ddof=1))
    best = max(srs, key=srs.get)
    print(f"\n{len(res)} variants; Sharpe range {min(srs.values()):+.2f} .. {max(srs.values()):+.2f}; best {best}")
    report("BEST " + best, res[best], n_trials=len(res), sr_var=var)
    print("  corr(best, index model) = %.2f" % res[best].corr(sl(idx)))
    print("  best by year:", by_year(res[best]))
    pd.DataFrame(res).assign(index_model=sl(idx)).to_pickle("trader/data/xsec_overnight_series.pkl")
    json.dump({"srs": srs, "var": var, "n": len(res), "best": best}, open("trader/data/xsec_overnight_summary.json", "w"))


if __name__ == "__main__":
    main()
