"""Gap-down pilot: live vs backtest reconciliation and a backtest under live constraints (research only, Nov 24 2026).

The live pilot (Nov 19, 20, 23 2026) lost on all three days while the backtest says about +60 bp per cash day.
Part A  reconciles each live day: backtest top-5 (three code versions), live candidate list, live fills, and what
        each would have returned. Uses Yahoo 1-minute bars (cached to trader/data/1m/, only ~30 days are served)
        and Yahoo dividend events, to find the code-level reasons the lists differ.
Part B  re-runs the pilot backtest (intraday_hourly/gd_regime.py rule, daily bars, 9:30 open -> close, top 5 gaps
        <= -2%, prior close >= $5, 14d volume >= 1M, SPY < 200d on t-1, EW over picks, 10 bp) under live
        constraints one at a time and combined. Every variant is counted.
Part C  entry-time cost: 5-minute bars (last 60 sessions, full universe) give the daily-open -> 9:45 drift of the
        picks; hourly bars (Dec 2023 on, full universe) give the open -> 10:30 drift and the gap between the daily
        bar's open and the intraday feed's first print.
Run: python3 tools/research/paper/gapdown_reconcile.py [--log]      (about 1-2 minutes; fetches ~60 symbols once)
"""
import sys, json, math, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
import yfetch
from bars import frame, NY
from gaps_daily_holdout import build
from common import evaluate

COST, K, STOP, TP = 0.0010, 5, 0.08, 0.20
SPLIT = dt.date(2022, 1, 1)
LIVE_DAYS = [dt.date(2026, 11, 19), dt.date(2026, 11, 20), dt.date(2026, 11, 23)]
OUT = Path("trader/data/gapdown_reconcile"); OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


# ----------------------------------------------------------------------------------------------- panel
def panel():
    syms, cal, A = build()
    o, h, l, c, v = A["o"], A["h"], A["l"], A["c"], A["v"]
    N, T = c.shape
    dates = list(cal)
    di = {d: i for i, d in enumerate(dates)}
    D = np.zeros((N, T))                                   # cash dividend going ex on day t (Yahoo events cache)
    for n, s in enumerate(syms):
        p = Path("trader/data/1d_events") / (s.replace(".", "-") + ".json")
        if not p.exists():
            continue
        for ts, amt in (json.load(open(p)).get("div") or {}).items():
            d = dt.datetime.fromtimestamp(int(ts), NY).date()
            if d in di and amt:
                D[n, di[d]] += float(amt)
    pC = np.concatenate([np.full((N, 1), np.nan), c[:, :-1]], 1)
    adv = pd.DataFrame(v.T).rolling(14, min_periods=10).mean().shift(1).to_numpy().T
    s = frame("SPY", "1d"); s = s[~s.index.duplicated()].set_index("date").c
    cash = (s < s.rolling(200).mean()).shift(1).reindex(pd.Index(dates)).fillna(False).to_numpy(bool)
    return dict(syms=syms, dates=dates, o=o, h=h, l=l, c=c, pC=pC, adv=adv, D=D, cash=cash)


# ----------------------------------------------------------------------------------------------- Part B
VARIANTS = {   # name: dict of switches; base = the pilot backtest exactly (gd_regime.py)
    "B0 pilot backtest (no price cap)": {},
    "(i) open <= $95": {"cap": 1},
    "(ii) open >= $5": {"open5": 1},
    "(iii) gap > -30%": {"gap30": 1},
    "(iv) -8% stop on daily low": {"stop": 1},
    "(iv-b) -8% stop + 20% TP (stop first)": {"stop": 1, "tp": 1},
    "(vi) dividend-adjusted gap (as live)": {"divadj": 1},
    "(vii) no look-ahead |open->close| < 60% filter": {"noyfilt": 1},
    "(v) all live constraints combined": {"cap": 1, "open5": 1, "gap30": 1, "stop": 1, "tp": 1, "divadj": 1, "noyfilt": 1},
    "(v-a) combined, stop only (no TP)": {"cap": 1, "open5": 1, "gap30": 1, "stop": 1, "divadj": 1, "noyfilt": 1},
}


def basket(P, sw, start=20):
    o, h, l, c, pC, adv, D = P["o"], P["h"], P["l"], P["c"], P["pC"], P["adv"], P["D"]
    ref = pC - D if sw.get("divadj") else pC
    gap = o / ref - 1
    y = c / o - 1
    ok = np.isfinite(gap) & np.isfinite(y) & (np.abs(o / pC - 1) < 0.6) & (o > 0)
    if not sw.get("noyfilt"):
        ok &= np.abs(y) < 0.6                                  # look-ahead cleaning filter in the original backtest
    el = ok & (pC >= 5) & (adv >= 1e6) & (gap <= -0.02)
    if sw.get("cap"):
        el &= o <= 95
    if sw.get("open5"):
        el &= o >= 5
    if sw.get("gap30"):
        el &= gap > -0.30
    r = y.copy()
    if sw.get("stop") or sw.get("tp"):
        hit_s = (l / o - 1 <= -STOP) if sw.get("stop") else np.zeros_like(y, bool)
        hit_t = (h / o - 1 >= TP) if sw.get("tp") else np.zeros_like(y, bool)
        r = np.where(hit_s, -STOP, np.where(hit_t, TP, y))   # entry at the open, so the stop can't be gapped through
    rows, picks = [], {}
    for t in range(start, len(P["dates"])):
        if not P["cash"][t]:
            continue
        idx = np.where(el[:, t])[0]
        if len(idx) == 0:
            continue
        p = idx[np.argsort(gap[idx, t])][:K]
        rows.append((P["dates"][t], r[p, t].mean() - COST, len(p)))
        picks[P["dates"][t]] = [(P["syms"][i], gap[i, t], r[i, t]) for i in p]
    S = pd.DataFrame(rows, columns=["date", "ret", "n"]).set_index("date")
    return S, picks


def stats_row(name, S, P, n_trials, sr_var):
    r = S.ret
    full = pd.Series(0.0, index=P["dates"][20:]); full.loc[r.index] = r.values
    e = evaluate(full, name, n_trials, sr_var)
    h1, h2 = r[r.index < SPLIT], r[r.index >= SPLIT]
    return {"variant": name, "days": len(r), "mean_bp": 1e4 * r.mean(), "t": tstat(r), "hit": (r > 0).mean(),
            "median_bp": 1e4 * r.median(), "H1_bp": 1e4 * h1.mean(), "H1_t": tstat(h1), "H2_bp": 1e4 * h2.mean(),
            "H2_t": tstat(h2), "SR_full": e["sharpe"], "DSR": e["dsr_p"], "picks": S.n.mean()}


# ----------------------------------------------------------------------------------------------- Part A
def one_min(sym):
    p = Path("trader/data/1m") / (sym.replace(".", "-") + ".npz")
    m = frame(sym, "1m") if p.exists() else None
    if m is None or not set(LIVE_DAYS) <= set(m.date):
        (Path("trader/data/1m")).mkdir(parents=True, exist_ok=True)
        try:
            d = yfetch.fetch_bars(sym, "1m", "7d")
            if d is not None and len(d["t"]):
                yfetch.save(sym, "1m", d)
        except Exception as ex:
            print("1m fetch failed", sym, ex)
        m = frame(sym, "1m")
    return m


def divs(sym):
    p = OUT / "divs.json"
    cache = json.load(open(p)) if p.exists() else {}
    if sym not in cache:
        try:
            doc = yfetch.get("/v8/finance/chart/%s?range=1mo&interval=1d&events=div" % yfetch.ysym(sym))
            ev = (doc["chart"]["result"][0].get("events") or {}).get("dividends") or {}
            cache[sym] = {dt.datetime.fromtimestamp(int(k), NY).date().isoformat(): v["amount"] for k, v in ev.items()}
        except Exception:
            cache[sym] = {}
        json.dump(cache, open(p, "w"))
    return cache[sym]


def live_sim(m, day, entry_tod=585, exit_tod=945):
    """9:45 entry at the 1-minute bar open, -8% stop / +20% TP on 1-minute bars (stop first; a bar opening through the
    stop fills at its open), flat at the 15:45 bar open, net 10 bp. Returns (entry, exit, reason, ret)."""
    if m is None:
        return None
    b = m[(m.date == day) & (m.tod >= entry_tod) & (m.tod <= exit_tod)].dropna(subset=["o"])
    if b.empty:
        return None
    e = float(b.o.iloc[0]); s, tp = e * (1 - STOP), e * (1 + TP)
    for _, x in b.iloc[:-1].iterrows():
        if x.o <= s or x.l <= s:
            px = min(x.o, s); return e, px, "stop", px / e - 1 - COST
        if x.h >= tp:
            return e, tp, "tp", tp / e - 1 - COST
    px = float(b.o.iloc[-1])
    return e, px, "time", px / e - 1 - COST


def reconcile(P):
    syms, dates = P["syms"], P["dates"]
    si = {s: i for i, s in enumerate(syms)}
    out_rows, day_rows = [], []
    for day in LIVE_DAYS:
        t = dates.index(day)
        o, c, pC, adv, l = P["o"][:, t], P["c"][:, t], P["pC"][:, t], P["adv"][:, t], P["l"][:, t]
        gap = o / pC - 1
        base = (pC >= 5) & (adv >= 1e6) & (gap <= -0.02) & np.isfinite(c)
        order = lambda m: [syms[i] for i in np.where(m)[0][np.argsort(gap[m])]]
        versions = {
            "paper gapdown.py (no |gap| cap, no $95)": order(base)[:K],
            "pilot backtest gd_regime.py (|gap|<60%, no $95)": order(base & (np.abs(gap) < 0.6) & (np.abs(c / o - 1) < 0.6))[:K],
            "conditioning study (|gap|<60%, open<=$95)": order(base & (np.abs(gap) < 0.6) & (np.abs(c / o - 1) < 0.6) & (o <= 95))[:K],
        }
        pk = json.load(open(f"trader/gapdown_picks_{day}.json"))
        live = json.load(open(f"trader/pilot_{day}.json"))
        legs = live.get("legs") or live.get("entries")
        legs = [{"sym": k, **v} for k, v in legs.items()] if isinstance(legs, dict) else [{"sym": x.get("sym") or x.get("symbol"), **x} for x in legs]
        names = set(sum(versions.values(), [])) | {x["sym"] for x in pk["candidates"][:8]} | {x["sym"] for x in legs}
        live_pos = {x["sym"]: n for n, x in enumerate(pk["candidates"])}
        for s in sorted(names):
            i = si.get(s)
            m = one_min(s)
            dv = divs(s).get(day.isoformat(), 0.0)
            md = m[m.date == day] if m is not None else pd.DataFrame()
            first_tod = int(md.dropna(subset=["c"]).tod.iloc[0]) if len(md.dropna(subset=["c"])) else None
            f930 = md[md.tod == 570].c.dropna()
            ls = live_sim(m, day)
            cand = next((x for x in pk["candidates"] if x["sym"] == s), None)
            row = {"date": day, "sym": s,
                   "bt_rank": next((k + 1 for k, sy in enumerate(order(base)) if sy == s), None),
                   "live_rank": live_pos[s] + 1 if s in live_pos else None,
                   "pc_raw": pC[i] if i is not None else np.nan, "div_ex_today": dv,
                   "daily_open": o[i] if i is not None else np.nan,
                   "gap_daily_raw": gap[i] if i is not None else np.nan,
                   "gap_daily_divadj": o[i] / (pC[i] - dv) - 1 if i is not None else np.nan,
                   "first_1m_bar": f"{first_tod // 60}:{first_tod % 60:02d}" if first_tod else None,
                   "gap_1m_close_divadj": float(f930.iloc[0]) / (pC[i] - dv) - 1 if len(f930) and i is not None else np.nan,
                   "live_file_gap": cand["gap"] if cand else np.nan,
                   "bt_oc_net": c[i] / o[i] - 1 - COST if i is not None else np.nan,
                   "sim945_ret": ls[3] if ls else np.nan, "sim945_exit": ls[2] if ls else None,
                   "live_ret": np.nan}
            leg = next((x for x in legs if x["sym"] == s), None)
            if leg is not None:
                ex = live.get("exits")
                ex = ex.get(s) if isinstance(ex, dict) else next((e for e in ex if e.get("symbol") == s), None)
                qty = leg.get("shares") or leg.get("qty")
                px = ex.get("price") or ex.get("exit")
                row["live_ret"] = px / leg["fill"] - 1
                row["live_fill"], row["live_exit"] = leg["fill"], px
            out_rows.append(row)
        R = pd.DataFrame([r for r in out_rows if r["date"] == day]).set_index("sym")
        d = {"date": day, "live_pnl_bp": live["return_bp"], "live_names": ",".join(x["sym"] for x in legs)}
        for vn, lst in versions.items():
            d[vn] = ",".join(lst)
            d[vn + " oc_bp"] = 1e4 * np.nanmean([R.loc[s, "bt_oc_net"] for s in lst])
            d[vn + " sim945_bp"] = 1e4 * np.nanmean([R.loc[s, "sim945_ret"] for s in lst])
        d["live names, 1m sim bp"] = 1e4 * np.nanmean([R.loc[x["sym"], "sim945_ret"] for x in legs])
        d["live names, daily oc bp"] = 1e4 * np.nanmean([R.loc[x["sym"], "bt_oc_net"] for x in legs])
        d["SPY oc %"] = 100 * (frame("SPY", "1d").set_index("date").c.get(day) / frame("SPY", "1d").set_index("date").o.get(day) - 1)
        day_rows.append(d)
    return pd.DataFrame(out_rows), pd.DataFrame(day_rows)


# ----------------------------------------------------------------------------------------------- Part C
def entry_drift(P, picks_by_day, interval):
    """For each (cash) day with picks, compare the daily-bar open with intraday prices of the same picks."""
    rows = []
    cache = {}
    for day, plist in picks_by_day.items():
        for s, g, _ in plist:
            if s not in cache:
                cache[s] = frame(s, interval)
            m = cache[s]
            if m is None:
                continue
            md = m[m.date == day]
            if md.empty or 570 not in set(md.tod):
                continue
            t = P["dates"].index(day); i = P["syms"].index(s)
            o, c = P["o"][i, t], P["c"][i, t]
            px = {tod: float(md[md.tod == tod].o.iloc[0]) for tod in (570, 585, 630) if tod in set(md.tod)}
            r = {"date": day, "sym": s, "gap": g, "daily_open": o, "first_intraday_open": px[570],
                 "feed_diff": px[570] / o - 1, "oc_daily": c / o - 1}
            if 585 in px:
                r["drift_945"] = px[585] / o - 1; r["ret_945_close"] = c / px[585] - 1
                lo = md[md.tod >= 585].l.min()
                r["ret_945_stop"] = -STOP if lo <= px[585] * (1 - STOP) else c / px[585] - 1
            if 630 in px:
                r["drift_1030"] = px[630] / o - 1; r["ret_1030_close"] = c / px[630] - 1
            rows.append(r)
    X = pd.DataFrame(rows)
    if len(X):
        X = X[X.feed_diff.abs() < 0.10]                         # drop split-scaling mismatches between feeds (e.g. SUPN 1h)
    if X.empty:
        return X, X
    agg = X.groupby("date").mean(numeric_only=True)             # one observation per day (clustered by day)
    return X, agg


def live_replica_5m(P):
    """Live rule replayed on the 5-minute window (60 sessions, every day: none of them is a cash day).
    Live: gap = first 5-minute bar close / dividend-adjusted prior close - 1 (proxy for the first 1-minute close the
    picker uses), price $5..$95, top 5, buy the 9:45 bar open, -8% stop / +20% TP on 5-minute bars, out at the 15:45
    bar open. Backtest: daily open gap, open -> close. Mixed rows isolate selection from timing. Net 10 bp."""
    days = sorted(frame("SPY", "5m").date.unique())
    ti = [P["dates"].index(d) for d in days]
    N = len(P["syms"])
    F = {k: np.full((N, len(days)), np.nan) for k in ("c930", "o945", "o1545", "lo_after", "hi_after")}
    path = {}
    for n, s in enumerate(P["syms"]):
        m = frame(s, "5m")
        if m is None:
            continue
        for j, d in enumerate(days):
            g = m[m.date == d]
            if g.empty:
                continue
            tod = g.tod.to_numpy()
            if tod[0] == 570: F["c930"][n, j] = g.c.iloc[0]
            if 585 in tod and 945 in tod:
                F["o945"][n, j] = g.o[tod == 585].iloc[0]; F["o1545"][n, j] = g.o[tod == 945].iloc[0]
                path[(n, j)] = g[(tod >= 585) & (tod < 945)][["o", "h", "l"]].to_numpy()
    def run_live(e, xp, n, j):
        st, tp = e * (1 - STOP), e * (1 + TP)
        for o_, h_, l_ in path[(n, j)]:
            if o_ <= st or l_ <= st: return min(o_, st) / e - 1
            if h_ >= tp: return TP
        return xp / e - 1
    out = []
    for j, t in enumerate(ti):
        pC, D, adv, o, c = P["pC"][:, t], P["D"][:, t], P["adv"][:, t], P["o"][:, t], P["c"][:, t]
        base = (pC >= 5) & (adv >= 1e6) & np.isfinite(F["o945"][:, j])
        g_bt = o / pC - 1
        g_lv = F["c930"][:, j] / (pC - D) - 1
        bt = np.where(base & (g_bt <= -0.02) & (np.abs(g_bt) < 0.6))[0]; bt = bt[np.argsort(g_bt[bt])][:K]
        lv = np.where(base & (g_lv <= -0.02) & (F["c930"][:, j] >= 5) & (F["c930"][:, j] <= 95))[0]; lv = lv[np.argsort(g_lv[lv])][:K]
        r = {"date": P["dates"][t], "overlap": len(set(bt) & set(lv))}
        if len(bt):
            r["bt_oc"] = np.mean(c[bt] / o[bt] - 1) - COST
            r["bt_945_stop"] = np.mean([run_live(F["o945"][i, j], F["o1545"][i, j], i, j) for i in bt]) - COST
        if len(lv):
            r["live_945_stop"] = np.mean([run_live(F["o945"][i, j], F["o1545"][i, j], i, j) for i in lv]) - COST
            r["live_oc"] = np.mean(c[lv] / o[lv] - 1) - COST
        out.append(r)
    return pd.DataFrame(out).set_index("date")


def all_days_picks(P, sw, dates_subset):
    """Combined-rule picks on every day in dates_subset (not just cash days), for more power in Part C."""
    cash = P["cash"].copy()
    P2 = dict(P); P2["cash"] = np.array([d in dates_subset for d in P["dates"]])
    _, picks = basket(P2, sw)
    return picks


def main(do_log=False):
    P = panel()
    print(f"panel {len(P['syms'])} stocks x {len(P['dates'])} days; cash days {int(P['cash'][20:].sum())}")

    # ---- Part A
    A, Dd = reconcile(P)
    A.to_csv(OUT / "reconcile_names.csv", index=False); Dd.to_csv(OUT / "reconcile_days.csv", index=False)
    print("\n=== Part A: per-day reconciliation ===")
    for _, r in Dd.iterrows():
        print(f"\n{r.date}  SPY open->close {r['SPY oc %']:+.2f}%   live: {r.live_names}  live P&L {r.live_pnl_bp:+.0f} bp of cost")
        for k in r.index:
            if k.endswith(")") and not k.endswith("bp"):
                print(f"   {k:52s} {r[k]:40s} open->close {r[k+' oc_bp']:+7.0f} bp   9:45+stop sim {r[k+' sim945_bp']:+7.0f} bp")
        print(f"   {'live names':52s} {r.live_names:40s} open->close {r['live names, daily oc bp']:+7.0f} bp   9:45+stop sim {r['live names, 1m sim bp']:+7.0f} bp")
    cols = ["date", "sym", "bt_rank", "live_rank", "pc_raw", "div_ex_today", "daily_open", "gap_daily_raw", "gap_daily_divadj",
            "first_1m_bar", "gap_1m_close_divadj", "live_file_gap", "bt_oc_net", "sim945_ret", "sim945_exit", "live_ret"]
    print("\nper name (gaps and returns as fractions):")
    print(A[cols].sort_values(["date", "bt_rank"]).round(4).to_string(index=False))

    # ---- Part B
    sr_var = 0.0522                                          # cross-trial variance of annual SR in the gd_regime grid
    n_prior = 128 + 1 + 10                                   # gaps_intraday + stocks_gapdown + gapdown_conditioning
    n_trials = n_prior + len(VARIANTS)
    rows, series = [], {}
    for name, sw in VARIANTS.items():
        S, picks = basket(P, sw)
        series[name] = (S, picks)
        rows.append(stats_row(name, S, P, n_trials, sr_var))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "constrained.csv", index=False)
    print(f"\n=== Part B: cash days (SPY < 200d on t-1), EW top-5, net 10 bp; DSR with {n_trials} family trials ===")
    print(B.round(3).to_string(index=False))
    Sb, _ = series["B0 pilot backtest (no price cap)"]; Sv, pv = series["(v) all live constraints combined"]
    print("\nlive days in the combined variant:", {str(d): round(1e4 * Sv.ret.get(d, np.nan)) for d in LIVE_DAYS})
    print("pilot-backtest picks on live days:", {str(d): [p[0] for p in series['B0 pilot backtest (no price cap)'][1].get(d, [])] for d in LIVE_DAYS})
    print("combined-variant picks on live days:", {str(d): [p[0] for p in pv.get(d, [])] for d in LIVE_DAYS})
    # concentration: how much of the mean comes from the best days
    srt = np.sort(Sv.ret.to_numpy())
    print(f"combined: top-10 days share of total {srt[-10:].sum()/srt.sum():.2f}; trimmed (2% each side) mean "
          f"{1e4*srt[int(.02*len(srt)):int(.98*len(srt))].mean():+.1f} bp")

    # ---- Part C
    sw_v = VARIANTS["(v-a) combined, stop only (no TP)"]
    print("\n=== Part C: entry-time cost (picks of the combined rule; one observation per day) ===")
    days5 = set(frame("SPY", "5m").date.unique())
    out = {}
    for label, dset in (("5m, all days in window", days5), ("5m, cash days in window", {d for d in days5 if P["cash"][P["dates"].index(d)]})):
        pk = all_days_picks(P, sw_v, dset)
        X, agg = entry_drift(P, pk, "5m")
        out[label] = agg
        print(f"\n{label}: {len(agg)} days, {len(X)} stock-days")
        for k in ("feed_diff", "drift_945", "oc_daily", "ret_945_close", "ret_945_stop"):
            if k in agg:
                print(f"   {k:16s} mean {1e4*agg[k].mean():+7.1f} bp  t {tstat(agg[k]):+5.2f}")
    LR = live_replica_5m(P)
    LR.to_csv(OUT / "live_replica_5m.csv")
    print(f"\nlive-rule replica on the 5m window ({len(LR)} days, all non-cash days; mean overlap of the two top-5 lists "
          f"{LR.overlap.mean():.1f} names):")
    for k, lab in (("bt_oc", "backtest picks, daily open -> close"), ("bt_945_stop", "backtest picks, 9:45 entry + stop/TP, 15:45 out"),
                   ("live_oc", "live picks, daily open -> close"), ("live_945_stop", "live picks, 9:45 entry + stop/TP, 15:45 out")):
        x = LR[k].dropna()
        print(f"   {lab:48s} {1e4*x.mean():+7.1f} bp/day  t {tstat(x):+5.2f}  hit {(x>0).mean():.2f}")
    dd = (LR.bt_oc - LR.live_945_stop).dropna()
    print(f"   paired difference backtest - live replica: {1e4*dd.mean():+.1f} bp/day, t {tstat(dd):+.2f}")
    days1h = set(frame("SPY", "1h").date.unique())
    for label, dset in (("1h, all days Dec 2023-", days1h), ("1h, cash days Dec 2023-", {d for d in days1h if d in P["dates"] and P["cash"][P["dates"].index(d)]})):
        pk = all_days_picks(P, sw_v, dset)
        X, agg = entry_drift(P, pk, "1h")
        out[label] = agg
        print(f"\n{label}: {len(agg)} days, {len(X)} stock-days")
        for k in ("feed_diff", "drift_1030", "oc_daily", "ret_1030_close"):
            if k in agg:
                print(f"   {k:16s} mean {1e4*agg[k].mean():+7.1f} bp  t {tstat(agg[k]):+5.2f}")
        if label.startswith("1h, cash"):
            big = X[X.feed_diff.abs() > 0.01]
            print(f"   stock-days where daily open and first intraday print differ by > 1%: {len(big)} of {len(X)}; "
                  f"mean open->close on those (daily open) {1e4*big.oc_daily.mean():+.0f} bp vs from intraday first print "
                  f"{1e4*((1+big.oc_daily)/(1+big.feed_diff)-1).mean():+.0f} bp")
            # cash-day basket re-priced from the intraday feed's first print and from 10:30
            g = X.assign(oc_intra=(1 + X.oc_daily) / (1 + X.feed_diff) - 1).groupby("date").mean(numeric_only=True)
            for k in ("oc_daily", "oc_intra", "ret_1030_close"):
                print(f"   basket {k:15s} {1e4*(g[k]-COST).mean():+7.1f} bp/day net  t {tstat(g[k]-COST):+5.2f}  hit {(g[k]-COST>0).mean():.2f}")
    pd.concat({k: v for k, v in out.items()}).to_csv(OUT / "entry_drift.csv")

    if do_log:
        import registry
        b0 = B.iloc[0]; v = B[B.variant.str.startswith("(v) ")].iloc[0]
        registry.log(family="gapdown_reconcile", author="agent-gapdown-reconcile",
                     name="Gap-down pilot: live vs backtest reconciliation and backtest under live constraints",
                     hypothesis="The cash-day gap-down edge survives the constraints the live pilot actually trades under",
                     data=f"Yahoo daily {len(P['syms'])} stocks 2016-11..2026-11-23 ({int(len(Sb))} cash days traded); 1m bars Nov 17-23 2026; 5m last 60 sessions; 1h Dec 2023-",
                     method="pilot backtest rule; constraints one at a time and combined: open<=$95, open>=$5, gap>-30%, -8% stop on daily low, +20% TP, dividend-adjusted gap, no look-ahead |o->c|<60% filter; daily-series t, hit, halves 2018-21/2022-26, DSR with family trials; entry drift from 5m/1h",
                     variants=len(VARIANTS) + 4,   # + 5m live replica (2 lists x 2 entries counted as 2 new rules), 5m and 1h drift
                     metrics={"base_bp": b0.mean_bp, "base_t": b0.t, "combined_bp": v.mean_bp, "combined_t": v.t, "combined_hit": v.hit,
                              "combined_DSR": v.DSR, "combined_H1_bp": v.H1_bp, "combined_H2_bp": v.H2_bp,
                              **{f"bp_{r.variant.split(' ')[0]}": r.mean_bp for r in B.itertuples()},
                              **{f"t_{r.variant.split(' ')[0]}": r.t for r in B.itertuples()}},
                     verdict="rejected", notes="Lists differ because the live picker ranks by the CLOSE of the first 1-minute bar (not the open), "
                     "uses Yahoo's dividend-adjusted previous close (backtest uses the raw close: ex-dividend fake gaps SLDE/DHT Nov 23), "
                     "drops names without a 9:30 1-minute bar (DHT), and applies the $95 cap; backtest variants also differ on the |gap|<60% filter (CAPR). "
                     "Constrained backtest at the 9:30 open still t~3.1 (DSR 0.90), but the open is not tradeable (rule 8); a 9:45 entry costs ~10-30 bp "
                     "(1h cash days 2024-26: -10 bp at 10:30; 5m window: picks +29 bp from open to 9:45), leaving ~25-45 bp, t ~1.4-2.5, DSR < 0.9. "
                     "Fails the hurdles as tradeable. Report tools/research/reports/gapdown_reconcile.md",
                     code="tools/research/paper/gapdown_reconcile.py")
    return A, Dd, B


if __name__ == "__main__":
    main("--log" in sys.argv)
