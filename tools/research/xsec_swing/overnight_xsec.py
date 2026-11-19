"""(a) Overnight persistence (Aboody, Even-Tov, Lehavy & Trueman 2018) and (b) tug of war / overnight momentum
(Lou, Polk & Skouras 2019), long only, close -> next open, top-N of the point-in-time top-500 by dollar volume.

  python3 tools/research/xsec_swing/overnight_xsec.py [--superset-only]
Costs: 10 bp per night (a full round trip each night: buy ~15:58, sell at the 9:30 open).
"""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research/xsec_swing")
from common import Data, index_model, rolling_mean, shift, report, summary, by_year, halves, dsr, fmt

COST = 0.0010


def topn_overnight(d, S, U, n, cost=COST, gate=None, asc=False):
    """Hold the n highest-S names of universe U from the close of t to the open of t+1 (equal weight).
    Returns net nightly series indexed by the morning of t+1, the picks' gross overnight and next-day intraday return,
    and the per-stock overnight returns of the picks."""
    T = len(d.dates)
    gross = np.full(T, np.nan); intra = np.full(T, np.nan)
    singles = []
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
        singles.append(r[ok])
    g = pd.Series(gross, index=d.dates)
    net = (g - cost).fillna(0.0)
    return net, g, pd.Series(intra, index=d.dates), (np.concatenate(singles) if singles else np.array([]))


def topn_overnight_dollars(d, S, U, n, capital=1000.0, cost=COST, t0=260):
    """$1,000 version: whole shares at actual prices, each night slot = equity / n, choose the n best-ranked names
    whose actual price fits in a slot; buy at the close, sell at the next open. Returns daily series and nightly $ P&L."""
    T = len(d.dates)
    eq = capital
    r = np.zeros(T); pnl = np.zeros(T); worst_pos = []
    half = cost / 2
    for t in range(t0, T - 1):
        slot = eq / n
        m = U[t] & np.isfinite(S[t]) & np.isfinite(d.C[t]) & (d.CA[t] * (1 + half) <= slot)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        sel = idx[np.argsort(-S[t, idx], kind="stable")[:n]]
        p = 0.0
        for j in sel:
            sh = math.floor(slot / (d.CA[t, j] * (1 + half)))
            o = d.O[t + 1, j] if np.isfinite(d.O[t + 1, j]) else d.C[t, j]
            px_in, px_out = d.CA[t, j], (o + d.D[t + 1, j]) * d.F[t, j]          # actual prices (split on the morning handled by F[t])
            pj = sh * (px_out * (1 - half) - px_in * (1 + half))
            p += pj
            worst_pos.append(pj)
        pnl[t + 1] = p
        r[t + 1] = p / eq
        eq += p
    return pd.Series(r, index=d.dates), pd.Series(pnl, index=d.dates), np.array(worst_pos)


def main():
    superset_only = "--superset-only" in sys.argv
    d = Data()
    names = None
    if superset_only:
        names = {x["symbol"].upper().replace(".", "-") for x in json.load(open("trader/data/superset.json"))}
    U = d.universe(500, names=names)
    t0 = 260
    start = d.dates[t0]          # first date with 12-month momentum and 200-day filters available
    sl = lambda s: s[s.index >= start]
    print("dates", d.dates[0].date(), d.dates[-1].date(), "panel names", (~d.is_etf).sum(),
          "| universe:", "today's superset only" if superset_only else "all names listed today", "| start", start.date())
    print("universe size per day (median):", int(np.median(U.sum(1)[t0:])))

    # --- benchmarks -------------------------------------------------------------------------------
    idx, legs = index_model(d)
    idx_raw, _ = index_model(d, use_funds=False)
    print("\n== Benchmarks (net) ==")
    report("Live index model (3x funds, 4bp)", sl(idx))
    report("Live index model (3x underlying, 4bp)", sl(idx_raw))
    spy_on = d.series(d.RON[:, d.s("SPY")]).fillna(0)
    report("SPY overnight 1x (gross)", sl(spy_on))
    with np.errstate(all="ignore"):
        ew_on = pd.Series(np.nanmean(np.where(shift(U, 1), d.RON, np.nan), axis=1), index=d.dates).fillna(0)
        ew_id = pd.Series(np.nanmean(np.where(U, d.RID, np.nan), axis=1), index=d.dates).fillna(0)
    report("Universe EW overnight (gross)", sl(ew_on))
    report("Universe EW overnight net 10bp", sl(ew_on - COST))
    report("Universe EW intraday (gross)", sl(ew_id))
    print("worst night: index model %.2f%%, SPY %.2f%%" % (100 * sl(idx).min(), 100 * sl(spy_on).min()))

    RON, C = d.RON, d.C
    res, singles = {}, {}

    def run(name, S, n, gate=None, asc=False):
        net, g, intra, one = topn_overnight(d, S, U, n, gate=gate, asc=asc)
        res[name] = sl(net); singles[name] = one
        s = summary(sl(net))
        print(f"  {name:<24} gross {1e4*sl(g).mean():+6.1f}bp  net {s['mean_bp']:+6.1f}bp  SR {s['sharpe']:+5.2f}  t {s['t']:+5.2f}"
              f"  maxDD {s['maxdd']:+6.1%}  ann {s['ann_ret']:+6.1%}  next-day intraday {1e4*sl(intra).mean():+6.1f}bp  worst {100*sl(net).min():+.1f}%")

    # --- (a) overnight persistence -----------------------------------------------------------------
    print("\n== (a) Overnight persistence: rank by past k-day mean overnight return, hold top N close->open ==")
    for k in (5, 10, 15, 20):
        S = rolling_mean(RON, k, k)          # nights through the morning of t: known at the close of t
        for n in (5, 10, 20, 50):
            run(f"a_on{k}_top{n}", S, n)

    def deciles(S, R, t_from=t0):
        out = np.full((len(d.dates), 10), np.nan)
        for t in range(t_from, len(d.dates) - 1):
            m = U[t] & np.isfinite(S[t]) & np.isfinite(R[t + 1])
            ii = np.flatnonzero(m)
            q = pd.Series(S[t, ii]).rank(pct=True).values
            b = np.minimum((q * 10).astype(int), 9)
            for j in range(10):
                out[t + 1, j] = R[t + 1, ii[b == j]].mean()
        return pd.DataFrame(out, index=d.dates).iloc[t_from + 1:]

    print("\n  Persistence check (gross next-night overnight return by decile of past-20-night overnight return):")
    Dd = deciles(rolling_mean(RON, 20, 20), RON)
    sp = Dd[9] - Dd[0]
    print("   decile mean bp:", " ".join(f"{1e4*x:+.1f}" for x in Dd.mean()),
          "| top-bottom %.1f bp/night, t %.2f" % (1e4 * sp.mean(), sp.mean() / sp.std() * math.sqrt(len(sp))))

    # --- (b) tug of war ------------------------------------------------------------------------------
    print("\n== (b) Tug of war: 12-1 month momentum, overnight vs intraday components (gross, by decile) ==")
    mom = shift(C, 21) / shift(C, 252) - 1                       # 12-1 month price momentum at the close of t
    lon = np.log1p(np.where(np.isfinite(RON), RON, 0.0))
    on12 = pd.DataFrame(lon).rolling(252, min_periods=200).sum().values   # 12-month cumulative overnight log return
    for part, R in (("overnight", RON), ("intraday ", shift(d.RID, 0))):
        Dm = deciles(mom, RON if part == "overnight" else d.RID)
        sp = Dm[9] - Dm[0]
        print(f"   {part}: " + " ".join(f"{1e4*x:+.1f}" for x in Dm.mean()) +
              f" | winners-losers {1e4*sp.mean():+.1f}bp t {sp.mean()/sp.std()*math.sqrt(len(sp)):+.2f}"
              f" | halves t {sp.iloc[:len(sp)//2].mean()/sp.iloc[:len(sp)//2].std()*math.sqrt(len(sp)//2):+.2f}"
              f" / {sp.iloc[len(sp)//2:].mean()/sp.iloc[len(sp)//2:].std()*math.sqrt(len(sp)-len(sp)//2):+.2f}")
    print("\n  Long-only overnight momentum, top N:")
    for nm, S in (("mom12_1", mom), ("on12", on12)):
        for n in (5, 10, 20, 50):
            run(f"b_{nm}_top{n}", S, n)

    # --- gated by SPY > 200-day average, as the live model ---------------------------------------------
    print("\n== Gated by SPY > 200-day average (as the live model) ==")
    spy = C[:, d.s("SPY")]
    gate = spy > rolling_mean(spy[:, None], 200)[:, 0]
    for nm, S in (("a_on20", rolling_mean(RON, 20, 20)), ("a_on5", rolling_mean(RON, 5, 5)), ("b_mom12_1", mom)):
        for n in (10, 50):
            run(f"{nm}_top{n}_gated", S, n, gate=gate)

    # --- overnight reversal of today's intraday losers (liquidity provision at the close) ----------------
    print("\n== Overnight reversal: buy today's worst intraday (open->close) returns at the close, sell at the open ==")
    for n in (5, 20):
        run(f"x_idloser_top{n}", d.RID, n, asc=True)

    srs = {k: summary(v)["sharpe"] for k, v in res.items()}
    var = float(np.var(list(srs.values()), ddof=1))
    best = max(srs, key=srs.get)
    print(f"\n{len(res)} variants; Sharpe range {min(srs.values()):+.2f} .. {max(srs.values()):+.2f}; best {best}")
    report("BEST " + best, res[best], n_trials=len(res), sr_var=var)
    print("  corr(best, index model) = %.2f" % res[best].corr(sl(idx)))
    print("  best by year:", {y: v[:2] for y, v in by_year(res[best]).items()})
    one = singles[best]
    print("  single-stock nights in the picks: n %d, worst %.1f%%, 1st pct %.1f%%, share below -10%% %.2f%%, below -20%% %.3f%%"
          % (len(one), 100 * one.min(), 100 * np.percentile(one, 1), 100 * (one < -0.1).mean(), 100 * (one < -0.2).mean()))
    for c in (0.0, 0.0015, 0.0020):
        s = summary(res[best] + COST - c)
        print(f"  cost sensitivity {1e4*c:.0f}bp/night: mean {s['mean_bp']:+.1f}bp SR {s['sharpe']:+.2f} t {s['t']:+.2f} ann {s['ann_ret']:+.1%}")

    # walk-forward variant selection
    df = pd.DataFrame(res)
    wf, picks = [], {}
    for y in sorted(set(df.index.year)):
        if y < 2019:
            continue
        h = df[df.index.year < y]
        b = (h.mean() / h.std()).idxmax()
        picks[y] = b
        wf.append(df[df.index.year == y][b])
    wf = pd.concat(wf)
    print("\n== Walk-forward (yearly re-selection by expanding-window Sharpe), out of sample from 2019 ==")
    print("  picks:", picks)
    report("walk-forward OOS", wf, n_trials=len(res), sr_var=var)
    report("index model, same period", sl(idx)[sl(idx).index >= wf.index[0]])
    print("  corr(walk-forward, index model) = %.2f" % wf.corr(sl(idx)))

    # $1,000 whole-share version of the best variant
    if best.startswith("b_mom12_1_top"):
        n = int(best.split("top")[1].split("_")[0])
        print(f"\n== $1,000 whole-share version of {best} (actual prices, names that fit a slot) ==")
        for nn in sorted({3, 4, n}):
            r, pnl, pos = topn_overnight_dollars(d, mom, U, nn)
            r = sl(r)
            report(f"$1,000, {nn} names", r)
            print("    worst night $%.0f, worst single position $%.0f, nights with loss > $100: %d" %
                  (pnl.min(), pos.min(), int((pnl < -100).sum())))
    tag = "_superset" if superset_only else ""
    pd.DataFrame(res).assign(index_model=sl(idx), index_expo=sl(legs["_expo"]), walk_forward=wf).to_pickle(f"trader/data/xsec_overnight_series{tag}.pkl")
    json.dump({"srs": srs, "var": var, "n": len(res), "best": best, "wf_picks": {str(k): v for k, v in picks.items()}},
              open(f"trader/data/xsec_overnight_summary{tag}.json", "w"))


if __name__ == "__main__":
    main()
