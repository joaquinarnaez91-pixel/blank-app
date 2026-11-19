"""Execution and risk checks for the cash-day gap-down reversal (GD-930 on SPY < 200-day days).

1. Break-even cost: 10/20/30/50 bp round trip.
2. Risk: worst days, worst single positions, compliance with the $100/trade and $200/day loss caps,
   and a protective-stop variant (sell if the position falls 8% below entry, stop-first on the daily low).
3. Capital actually freed by the overnight model: capital_t = $1,000 x (legs in cash)/3, slots = capital_t / K.
4. Where the reversal accrues (hourly bars, 2024-26): 9:30 open -> 10:30 open -> close for the picked names, and
   the 10:30-entry version (a late fill).
5. First five minutes (5-minute bars, last ~60 sessions, all regimes): 9:30 open -> 9:35 open for gap-down picks.
6. Mechanism (Nagel 2012): the every-day rule's return by the prior day's VIX close, slope with NW t, VIX x regime.
7. Attribution: excess over the universe mean and alpha/beta vs SPY's same-day open->close (an ex-post check),
   with placebos (largest gap-UPs; random flat-open stocks).
8. ADR split (names from trader/universe.txt containing Depositary/ADS/ADR) and the profile of the picks.
Run: python3 tools/research/intraday_hourly/gd_checks.py [--log]
"""
import sys, math, re
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from sim import topk
from common import evaluate, fmt, nw_t, nw_slope, regime_below, cash_fraction, log_exp, OUT, CODE
from gaps_daily_holdout import build, SPLIT


def picks(score, elig, entry, K, slot):
    """Daily list of picked stock indices (same rule as sim.topk without sizing)."""
    N, T = score.shape
    sc = np.where(elig, score, -np.inf)
    order = np.argsort(-sc, axis=0)
    out = []
    for t in range(T):
        p = []
        for n in order[:, t]:
            if not np.isfinite(sc[n, t]):
                break
            if entry[n, t] <= slot:
                p.append(n)
                if len(p) == K:
                    break
        out.append(p)
    return out


def main():
    syms, cal, A = build()
    o, h, l, c, v = A["o"], A["h"], A["l"], A["c"], A["v"]
    N, T = c.shape
    dates = np.array(list(cal))
    pC = np.concatenate([np.full((N, 1), np.nan), c[:, :-1]], 1)
    adv = pd.DataFrame(v.T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    gap = o / pC - 1
    y = c / o - 1
    U = (pC >= 5) & (adv >= 1e6) & np.isfinite(gap) & np.isfinite(y) & (np.abs(gap) < 0.6) & (o > 0) & (np.abs(y) < 0.6)
    U[:, :20] = False
    el = U & (gap <= -0.02)
    sc = np.where(el, -gap, -np.inf)
    reg = regime_below(dates, "SPY")
    K = 5
    print("1. Break-even cost (base, gap<=-2%, K=5, SPY<200d days only; mean bp per regime day, NW t on full series):")
    be = {}
    for cost in (0.0010, 0.0020, 0.0030, 0.0050):
        w, _, _ = topk(sc, el, o, c, dates, K, cost=cost)
        r = pd.Series(np.where(reg, w.to_numpy(), 0.0), index=dates)[20:]
        be[round(cost * 1e4)] = (1e4 * r[reg[20:]].mean(), nw_t(r))
        print(f"   {cost*1e4:3.0f} bp: {1e4*r[reg[20:]].mean():+6.1f} bp/regime day, NW t {nw_t(r):+.2f}")
    # 2. risk
    w, fr, npos, trades = topk(sc, el, o, c, dates, K, cost=0.0010, collect=True)
    r = pd.Series(np.where(reg, w.to_numpy(), 0.0), index=dates)[20:]
    act = r[reg[20:]]
    print(f"\n2. Risk on {len(act)} regime days: worst days (% of $1,000): {np.round(100*np.sort(act.values)[:6], 2)}; "
          f"best: {np.round(100*np.sort(act.values)[-4:], 2)}")
    tr = pd.DataFrame(trades, columns=["date", "n", "entry", "exit", "sh", "ret"])
    tr = tr[tr.date.isin(set(dates[reg]))]
    tr["pnl"] = tr.sh * (tr.exit - tr.entry) - 0.001 * tr.sh * tr.entry
    print(f"   positions {len(tr)}; worst position P&L ${tr.pnl.min():.0f}; positions losing > $100: {(tr.pnl < -100).sum()}; "
          f"days losing > $200: {(act * 1000 < -200).sum()}; mean position ${(tr.sh*tr.entry).mean():.0f}")
    eq = (1 + act).cumprod()
    print(f"   regime-day equity: total x{eq.iloc[-1]:.2f}, max drawdown {100*(eq/eq.cummax()-1).min():.1f}%")
    # stop variant: sell at entry*(1-8%) if the day's low reaches it (stop-first)
    stop = o * (1 - 0.08)
    xs = np.where(l <= stop, stop, c)
    ws, _, _ = topk(sc, el, o, xs, dates, K, stop=stop, cost=0.0010)
    rs = pd.Series(np.where(reg, ws.to_numpy(), 0.0), index=dates)[20:]
    print(f"   with an 8% stop (stop-first): {1e4*rs[reg[20:]].mean():+.1f} bp/regime day, NW t {nw_t(rs):+.2f}, "
          f"worst day {100*rs.min():.2f}%")
    # 3. capital freed by the overnight model (thirds), any leg in cash
    cf = cash_fraction(dates)
    tot = np.zeros(T)
    for frac in (1 / 3, 2 / 3, 1.0):
        m = np.isclose(cf, frac)
        if m.sum() == 0:
            continue
        wf, _, _ = topk(sc, el, o, c, dates, K, capital=1000 * frac, cost=0.0010)
        tot += np.where(m, wf.to_numpy() * frac, 0.0)      # P&L relative to the whole $1,000 sleeve
    rt = pd.Series(tot, index=dates)[20:]
    e = evaluate(rt, "GD-930 K=5 on capital freed by the overnight model", 1, 0.25)
    print(f"\n3. {fmt(e)}")
    hold, post = rt[[d < SPLIT for d in rt.index]], rt[[d >= SPLIT for d in rt.index]]
    print(f"   holdout 2017-2023 {1e4*hold.mean():+.2f} bp/day (NW t {nw_t(hold):+.2f}); 2024-26 {1e4*post.mean():+.2f} "
          f"bp/day (NW t {nw_t(post):+.2f}); days with any leg in cash: {(cf[20:] > 0).sum()}")
    pd.DataFrame({"r_spy_regime": r, "r_capital_freed": rt}).to_csv(OUT / "gd930_daily.csv")
    # 4. hourly decomposition (2024-26) for the same picks
    from panel import load, features
    P = load(); F = features(P)
    ho, hc = P["o"].astype(float), P["c"].astype(float)
    hd = pd.to_datetime(pd.Series(P["dates"])).dt.date.to_numpy()
    U2 = F["full"] & (F["prevC"] >= 5) & (F["adv"] >= 1e6) & np.isfinite(F["gap"]) & (np.abs(F["gap"]) < 0.6)
    el2 = U2 & (F["gap"] <= -0.02)
    reg2 = regime_below(hd, "SPY")
    pk = picks(np.where(el2, -F["gap"], -np.inf), el2, ho[:, :, 0], K, 200.0)
    rows = []
    for t, p in enumerate(pk):
        if not p or not reg2[t]:
            continue
        p = np.array(p)
        rows.append({"date": hd[t], "first_hour": np.mean(ho[p, t, 1] / ho[p, t, 0] - 1),
                     "rest": np.mean(hc[p, t, 6] / ho[p, t, 1] - 1), "full": np.mean(hc[p, t, 6] / ho[p, t, 0] - 1)})
    Dd = pd.DataFrame(rows)
    if len(Dd):
        print(f"\n4. Hourly bars, {len(Dd)} SPY<200d days in 2024-26 ({P['o'].shape[0]} stocks loaded): picks' mean gross "
              f"9:30->10:30 {1e4*Dd.first_hour.mean():+.1f} bp, 10:30->close {1e4*Dd.rest.mean():+.1f} bp, "
              f"9:30->close {1e4*Dd.full.mean():+.1f} bp (t {Dd.full.mean()/Dd.full.std()*math.sqrt(len(Dd)):+.2f})")
    # 5. first five minutes, 5-minute bars (all regimes, ~60 sessions)
    rows = []
    import glob, os
    for f in glob.glob("trader/data/5m/*.npz"):
        s = os.path.basename(f)[:-4]
        if s in ("SPY", "QQQ", "IWM", "SMH", "^VIX") or s.endswith("QQQ"):
            continue
        df = frame(s, "5m")
        if df is None or len(df) < 500:
            continue
        g = df.groupby("date")
        first = g.o.first(); last = g.c.last(); o935 = df[df.tod == 575].groupby("date").o.first()
        vol = g.v.sum()
        x = pd.DataFrame({"o": first, "c": last, "o935": o935, "pc": last.shift(1), "adv": vol.rolling(14).mean().shift(1)})
        x["gap"] = x.o / x.pc - 1
        x = x[(x.gap <= -0.02) & (x.pc >= 5) & (x.adv >= 1e6) & (x.gap > -0.6)].dropna()
        for d, rr in x.iterrows():
            rows.append({"date": d, "sym": s, "gap": rr.gap, "r5": rr.o935 / rr.o - 1, "rest": rr.c / rr.o935 - 1,
                         "full": rr.c / rr.o - 1})
    X = pd.DataFrame(rows)
    if len(X):
        X["rk"] = X.groupby("date").gap.rank(method="first")
        top = X[X.rk <= 5].groupby("date")[["r5", "rest", "full"]].mean()
        print(f"5. 5-minute bars ({X.date.nunique()} sessions, all regimes): top-5 gap-downs, mean gross 9:30->9:35 "
              f"{1e4*top.r5.mean():+.1f} bp, 9:35->close {1e4*top.rest.mean():+.1f} bp, 9:30->close {1e4*top.full.mean():+.1f} bp")
    # 6. VIX mechanism
    wall, _, _ = topk(sc, el, o, c, dates, K, cost=0.0010)
    rall = pd.Series(wall.to_numpy(), index=dates)[20:]
    vix = frame("^VIX", "1d"); vix = vix[~vix.index.duplicated()].set_index("date").c
    df = pd.DataFrame({"r": rall, "vix": vix.shift(1).reindex(rall.index), "below": reg[20:]}).dropna()
    df["vb"] = pd.cut(df.vix, [0, 15, 20, 25, 30, 100])
    agg = lambda x: pd.Series({"days": len(x), "mean_bp": 1e4 * x.mean(), "t": x.mean() / x.std() * math.sqrt(len(x))})
    print("\n6. Every-day rule (K=5, 10 bp) by prior VIX close:")
    print(df.groupby("vb", observed=True).r.apply(agg).unstack().round(2).to_string())
    print(df.groupby(["below", "vb"], observed=True).r.apply(agg).unstack().round(2).to_string())
    g, tg = nw_slope(df.r.values, df.vix.values)
    print(f"   slope {1e4*g:+.2f} bp per VIX point, NW t {tg:+.2f}")
    # 7. attribution and placebos on regime days (gross)
    spy = frame("SPY", "1d"); spy = spy[~spy.index.duplicated()].set_index("date")
    spy_oc = (spy.c / spy.o - 1).reindex(dates).to_numpy()
    ew = np.nanmean(np.where(U, y, np.nan), 0)
    m = reg.copy(); m[:20] = False
    rng = np.random.default_rng(0)
    print("\n7. Attribution on SPY<200d days (gross, K=5):")
    attrib = {}
    for name, elx, scx in (("gap-down top5", el, -gap), ("gap-up top5 (placebo)", U & (gap >= 0.02), gap),
                           ("flat-open random5 (placebo)", U & (np.abs(gap) < 0.005), rng.random((N, T)))):
        wx, _, _ = topk(np.where(elx, scx, -np.inf), elx, o, c, dates, K, cost=0.0)
        x = wx.to_numpy()[m]; mk = spy_oc[m]; e_ = ew[m]
        Xr = np.column_stack([np.ones(m.sum()), mk]); b, *_ = np.linalg.lstsq(Xr, x, rcond=None); res = x - Xr @ b
        s2 = np.sum(res ** 2) / (len(x) - 2)
        sa = math.sqrt(s2 * (1 / len(x) + mk.mean() ** 2 / np.sum((mk - mk.mean()) ** 2)))
        attrib[name] = (1e4 * (x - e_).mean(), nw_t(pd.Series(x - e_)), b[1], 1e4 * b[0], b[0] / sa)
        print(f"   {name:28s} gross {1e4*x.mean():+6.1f} bp; minus universe {attrib[name][0]:+6.1f} bp (NW t {attrib[name][1]:+.2f}); "
              f"beta {b[1]:.2f}; alpha {1e4*b[0]:+6.1f} bp (t {b[0]/sa:+.2f})")
    # 8. ADR split and pick profile
    names = dict(ln.strip().split("|", 1) for ln in open("trader/universe.txt") if "|" in ln and not ln.startswith("#"))
    adr = np.array([bool(re.search(r"Depositary|\bADS\b|\bADR\b", names.get(s_, ""), re.I)) for s_ in syms])
    pre = np.array([d < SPLIT for d in dates])
    print(f"\n8. ADR split ({adr.sum()} ADR-flagged names):")
    for lab, msk in (("ex-ADR", ~adr), ("ADR only", adr)):
        elx = el & msk[:, None]
        wx, _, _ = topk(np.where(elx, -gap, -np.inf), elx, o, c, dates, K, cost=0.0010)
        rx = pd.Series(np.where(reg, wx.to_numpy(), 0.0), index=dates)[20:]
        print(f"   {lab:9s} {1e4*rx[reg[20:]].mean():+6.1f} bp/regime day, NW t {nw_t(rx):+.2f}, holdout NW t "
              f"{nw_t(rx[pre[20:]]):+.2f}, 2024-26 NW t {nw_t(rx[~pre[20:]]):+.2f}")
    di = {d: i for i, d in enumerate(dates)}
    tr["gap"] = [gap[n, di[d]] for d, n in zip(tr.date, tr.n)]
    dv = pd.DataFrame((v * c).T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    tr["dv"] = [dv[n, di[d]] for d, n in zip(tr.date, tr.n)]
    print(f"   picks: median price ${tr.entry.median():.0f}, median gap {100*tr.gap.median():.1f}%, median $ADV "
          f"${tr.dv.median()/1e6:.0f}M, share < $20M {100*(tr.dv < 2e7).mean():.0f}%")
    tr["gb"] = pd.cut(tr.gap, [-1, -0.15, -0.08, -0.04, -0.02])
    print(tr.groupby("gb", observed=True).ret.agg(n="size", mean_bp=lambda x: 1e4 * x.mean(), hit=lambda x: (x > 0).mean()).round(3).to_string())
    if "--log" in sys.argv:
        log_exp(family="gaps_intraday", name="Cash-day gap-down reversal: execution, risk, mechanism and attribution checks",
                hypothesis="The SPY<200d gap-down reversal survives realistic costs, late entry, stops, partial capital, "
                           "is explained by liquidity provision (VIX) rather than beta, and is not an ADR artifact",
                data="daily bars 2016-11..2026-11 (1,651 stocks); hourly bars 2024-26; 5-minute bars (last ~60 sessions)",
                method="costs 10/20/30/50 bp; 8% stop; capital freed by the overnight model; hourly 9:30/10:30 split; "
                       "VIX buckets; alpha vs SPY o->c with gap-up and random placebos; ADR split",
                variants=5,
                metrics={"nw_t_10bp": float(be[10][1]), "regime_bp_30bp": float(be[30][0]), "nw_t_30bp": float(be[30][1]),
                         "stop8_nw_t": float(nw_t(rs)), "capital_freed_t": float(e["t"]), "capital_freed_SR": float(e["sharpe"]),
                         "worst_day_pct": float(100 * act.min()), "vix_slope_t": float(tg),
                         "alpha_bp": float(attrib["gap-down top5"][3]), "alpha_t": float(attrib["gap-down top5"][4]),
                         "placebo_gapup_alpha_t": float(attrib["gap-up top5 (placebo)"][4])},
                verdict="promising", notes="see tools/research/reports/intraday_hourly.md",
                code=CODE + "gd_checks.py")


if __name__ == "__main__":
    main()
