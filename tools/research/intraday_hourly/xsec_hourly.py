"""(c) Cross-sectional prediction of the 10:30 open -> close return from features known at 10:30, ~710 sessions.

Universe each day (point in time): full hourly session, prior close >= $5, 14-day mean volume >= 1M shares,
all features finite.  Features: gap, r1h (first hour), log RV, prev_ret, ret5, dist20, atr_pct, log dollar volume;
market: SPY first-hour return (mkt1h) and SPY gap (mktgap).

Part 1, Fama-MacBeth (all days): daily OLS of y on cross-sectional ranks (in [-0.5, 0.5]); mean slope with plain and
Newey-West t. Interactions with the market's first hour: the daily slope series b_f(t) regressed on mkt1h(t)
(equivalent to a pooled f x mkt1h term with day fixed effects), NW t.
Part 2, walk-forward models (out of sample only): LightGBM on ranks + raw values + market features, target =
the day's cross-sectional rank of y; and a linear benchmark (rolling Fama-MacBeth composite). Expanding window,
first test day ~ day 270, refit every 21 sessions, 5-session embargo between the end of training and the test block
(labels end the same day, so no label overlap needs purging). Scored by daily rank IC (NW t) and long-only top-K
portfolios (whole shares, $1,000, 10 bp; 20 bp sensitivity), entry 10:30 open, exit at the close.
Run: python3 tools/research/intraday_hourly/xsec_hourly.py [--log]
"""
import sys, math, time
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from panel import load, features
from sim import topk
from bars import frame
from common import evaluate, fmt, passes, nw_t, nw_slope, log_exp, regime_below, OUT, CODE

COST = 0.0010
WARM, FIRST, STEP, EMB = 20, 270, 21, 5
FEAT = ["gap", "r1h", "lrv", "prev_ret", "ret5", "dist20", "atr_pct", "ldvol"]


def xrank(a, mask):
    x = np.where(mask, a, np.nan)
    return pd.DataFrame(x).rank(axis=0, pct=True).to_numpy() - 0.5


def main(do_log=False):
    t0 = time.time()
    P = load(); F = features(P)
    o, c = P["o"].astype(float), P["c"].astype(float)
    dates = pd.to_datetime(pd.Series(P["dates"])).dt.date.to_numpy()
    F["lrv"] = np.log(np.clip(F["rv"], 1e-3, None))
    N, T = F["y"].shape
    U = F["full"] & (F["prevC"] >= 5) & (F["adv"] >= 1e6) & np.isfinite(F["y"])
    for f in FEAT:
        U &= np.isfinite(F[f])
    U[:, :WARM] = False
    R = {f: xrank(F[f], U) for f in FEAT}
    y = F["y"]
    vx = frame("^VIX", "1d"); vx = vx[~vx.index.duplicated()].set_index("date").c
    vix_prev = vx.shift(1).reindex(pd.Index(dates)).to_numpy()          # prior-day VIX close, known before the open
    below = regime_below(dates, "SPY")
    print(f"universe stock-days {int(U.sum())}, days {T-WARM}, median stocks/day {int(np.median(U.sum(0)[WARM:]))}")

    # ---------------- Part 1: Fama-MacBeth ----------------
    days = [t for t in range(WARM, T) if U[:, t].sum() >= 50]
    B, Bu = [], {f: [] for f in FEAT}
    for t in days:
        m = U[:, t]
        X = np.column_stack([np.ones(m.sum())] + [R[f][m, t] for f in FEAT])
        b, *_ = np.linalg.lstsq(X, y[m, t], rcond=None)
        B.append(b)
        for f in FEAT:
            Xu = np.column_stack([np.ones(m.sum()), R[f][m, t]])
            bu, *_ = np.linalg.lstsq(Xu, y[m, t], rcond=None)
            Bu[f].append(bu[1])
    B = pd.DataFrame(B, columns=["const"] + FEAT, index=[dates[t] for t in days])
    mk = pd.Series([F["mkt1h"][0, t] for t in days], index=B.index)
    mkz = (mk - mk.mean()) / mk.std()
    vz = pd.Series([vix_prev[t] for t in days], index=B.index)
    vz = (vz - vz.mean()) / vz.std()
    fm_rows = []
    for f in FEAT:
        s = B[f]
        # interaction: b_f(t) = a + g * z(mkt1h_t), Newey-West t on g
        g1, ti_nw = nw_slope(s.values, mkz.values)
        g2, tv_nw = nw_slope(s.values, vz.values)
        su = pd.Series(Bu[f])
        fm_rows.append({"feature": f, "multi_bp": 1e4 * s.mean(), "t": s.mean() / s.std(ddof=1) * math.sqrt(len(s)),
                        "nw_t": nw_t(s), "pos_days": (s > 0).mean(), "uni_bp": 1e4 * su.mean(), "uni_nw_t": nw_t(su),
                        "x_mkt1h_bp_per_sd": 1e4 * g1, "x_mkt1h_nw_t": ti_nw, "x_vix_bp_per_sd": 1e4 * g2, "x_vix_nw_t": tv_nw,
                        "h1_bp": 1e4 * s.iloc[:len(s) // 2].mean(), "h2_bp": 1e4 * s.iloc[len(s) // 2:].mean()})
    FM = pd.DataFrame(fm_rows)
    pd.set_option("display.width", 220)
    print("\nFama-MacBeth, 10:30->close on daily ranks (slope = top-minus-bottom spread in bp; NW lags auto):")
    print(FM.round(2).to_string(index=False))
    FM.to_csv(OUT / "xsec_fm.csv", index=False)

    # ---------------- Part 2: walk-forward models ----------------
    import lightgbm as lgb
    nn, tt = np.nonzero(U)
    Xr = np.column_stack([R[f][nn, tt] for f in FEAT])
    Xraw = np.column_stack([F[f][nn, tt] for f in ("gap", "r1h", "lrv", "atr_pct")])
    Xm = np.column_stack([F["mkt1h"][nn, tt], F["mktgap"][nn, tt]])
    X = np.column_stack([Xr, Xraw, Xm]).astype(np.float32)
    X2 = np.column_stack([X, vix_prev[tt], below[tt].astype(float)]).astype(np.float32)
    yv = y[nn, tt]
    ytr_rank = xrank(y, U)[nn, tt]
    pred_g = np.full((N, T), np.nan); pred_l = np.full((N, T), np.nan); pred_g2 = np.full((N, T), np.nan)
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=2000, subsample=0.5,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, verbose=-1)
    for s in range(FIRST, T, STEP):
        e = min(s + STEP, T)
        tr = (tt < s - EMB)
        te = (tt >= s) & (tt < e)
        if te.sum() == 0:
            continue
        mdl = lgb.LGBMRegressor(**params).fit(X[tr], ytr_rank[tr])
        pred_g[nn[te], tt[te]] = mdl.predict(X[te])
        mdl2 = lgb.LGBMRegressor(**params).fit(X2[tr], ytr_rank[tr])
        pred_g2[nn[te], tt[te]] = mdl2.predict(X2[te])
        # linear benchmark: mean FM slopes over the training days only
        bt = B.loc[[dates[t] for t in days if t < s - EMB], FEAT].mean().values
        pred_l[nn[te], tt[te]] = Xr[te] @ bt
        print(f"  fit through day {s-EMB} ({dates[s-EMB]}), test {dates[s]}..{dates[e-1]}, train rows {int(tr.sum())} "
              f"{time.time()-t0:.0f}s", flush=True)
    imp = pd.Series(mdl.feature_importances_, index=FEAT + ["gap_raw", "r1h_raw", "lrv_raw", "atrp_raw", "mkt1h", "mktgap"])
    print("last model split importance:", imp.sort_values(ascending=False).to_dict())
    oos = list(range(FIRST, T))
    results, rows = {}, []
    for mname, pred in (("LightGBM", pred_g), ("LightGBM+VIX/regime", pred_g2), ("linear FM", pred_l)):
        ic = []
        for t in oos:
            m = U[:, t] & np.isfinite(pred[:, t])
            if m.sum() < 30:
                ic.append(np.nan); continue
            ic.append(pd.Series(pred[m, t]).rank().corr(pd.Series(y[m, t]).rank()))
        ic = pd.Series(ic, index=[dates[t] for t in oos]).dropna()
        # decile spread (gross, diagnostic) and top-decile long-only (fractional, net)
        top, spread = [], []
        for t in oos:
            m = U[:, t] & np.isfinite(pred[:, t])
            if m.sum() < 50:
                top.append(0.0); spread.append(0.0); continue
            q = pd.Series(pred[m, t]).rank(pct=True).to_numpy()
            yy = y[m, t]
            top.append(yy[q > 0.9].mean() - COST); spread.append(yy[q > 0.9].mean() - yy[q <= 0.1].mean())
        print(f"\n{mname}: OOS days {len(oos)} ({dates[FIRST]}..{dates[-1]}); rank IC mean {ic.mean():+.4f} "
              f"NW t {nw_t(ic):+.2f}; IC>0 on {(ic > 0).mean():.0%} of days; halves {ic.iloc[:len(ic)//2].mean():+.4f} / "
              f"{ic.iloc[len(ic)//2:].mean():+.4f}")
        sp = pd.Series(spread, index=[dates[t] for t in oos]); tp = pd.Series(top, index=sp.index)
        print(f"  decile spread (gross, not tradeable long-only) {1e4*sp.mean():+.2f}bp NW t {nw_t(sp):+.2f}; "
              f"top decile long, net 10bp {1e4*tp.mean():+.2f}bp NW t {nw_t(tp):+.2f}")
        rows.append({"name": f"{mname} rank IC", "ic": ic.mean(), "ic_nw_t": nw_t(ic), "spread_bp": 1e4 * sp.mean(),
                     "spread_nw_t": nw_t(sp), "topdec_net_bp": 1e4 * tp.mean(), "topdec_nw_t": nw_t(tp)})
        for K in (1, 3, 5, 10):
            sc = np.where(np.isfinite(pred), pred, -np.inf)
            w, fr, npos = topk(sc, U & np.isfinite(pred), o[:, :, 1], c[:, :, 6], dates, K, cost=COST)
            w20, _, _ = topk(sc, U & np.isfinite(pred), o[:, :, 1], c[:, :, 6], dates, K, cost=0.0020)
            results[(mname, K)] = (w.iloc[FIRST:], w20.iloc[FIRST:], fr.iloc[FIRST:])
    srs = [v[0].mean() / v[0].std(ddof=1) * math.sqrt(252) for v in results.values()]
    var = float(np.var(srs, ddof=1))
    n_tr = len(results)
    print(f"\nTop-K long-only portfolios (OOS only), {n_tr} variants, SR variance {var:.2f}")
    prow = []
    for (mname, K), (w, w20, fr) in results.items():
        s = evaluate(w, f"{mname} top-{K}, 10:30->close", n_tr, var)
        wb = w[below[FIRST:]]
        s.update({"model": mname, "K": K, "net20_bp": 1e4 * w20.mean(), "frac_bp": 1e4 * fr.mean(),
                  "below200_bp": 1e4 * wb.mean(), "below200_n": int(len(wb)), "pass": passes(s)})
        prow.append(s)
        print(fmt(s), f"| 20bp {1e4*w20.mean():+.2f} | frac {1e4*fr.mean():+.2f} | SPY<200d {1e4*wb.mean():+.1f}bp (n {len(wb)})",
              " PASS" if s["pass"] else "")
    PR = pd.DataFrame(prow); PR.to_csv(OUT / "xsec_portfolios.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "xsec_ic.csv", index=False)
    if do_log:
        best_fm = FM.loc[FM.nw_t.abs().idxmax()]
        log_exp(family="xsec_intraday", name="Fama-MacBeth: 10:30 features -> 10:30-close return (hourly, 710 days)",
                hypothesis="Features known at 10:30 (gap, first hour, RV, prior-day, 5-day, dist to 20d high, ATR%, $vol) "
                           "and their interaction with the market's first hour predict the rest-of-day cross-section",
                data=f"Yahoo 1h, {N} stocks, {int(U.sum())} stock-days, {dates[WARM]}..{dates[-1]}",
                method="daily OLS on ranks, NW t; interactions = slope series regressed on z(SPY first-hour return) "
                       "and on z(prior VIX close)",
                variants=3 * len(FEAT),
                metrics={**{f"{r.feature}_nw_t": float(r.nw_t) for r in FM.itertuples()},
                         **{f"{r.feature}_x_mkt_t": float(r.x_mkt1h_nw_t) for r in FM.itertuples()},
                         **{f"{r.feature}_x_vix_t": float(r.x_vix_nw_t) for r in FM.itertuples()}},
                verdict="rejected" if (FM.nw_t.abs() < 3).all() and (FM.x_mkt1h_nw_t.abs() < 3).all()
                and (FM.x_vix_nw_t.abs() < 3).all() else "promising",
                notes=f"largest |NW t|: {best_fm.feature} {best_fm.nw_t:+.2f} ({best_fm.multi_bp:+.1f} bp top-minus-bottom). "
                      "Slopes are long-short spreads; long-only deciles are reported in the walk-forward test.",
                code=CODE + "xsec_hourly.py")
        best = PR.loc[PR.sharpe.idxmax()]
        icg = rows[0]
        log_exp(family="xsec_intraday", name="Walk-forward LightGBM + linear FM composite, long-only top-K 10:30->close",
                hypothesis="A nonlinear model of 10:30 features ranks stocks well enough to earn > 10 bp per day long-only",
                data=f"OOS {dates[FIRST]}..{dates[-1]} ({T-FIRST} days), expanding-window training from {dates[WARM]}",
                method="LightGBM (fixed params, rank target; v2 adds prior VIX and the SPY 200-day regime) and a rolling "
                       "FM composite; refit every 21 days, 5-day "
                       "embargo; top-K in {1,3,5,10}, whole shares, $1000, 10 bp; rank IC with NW t",
                variants=n_tr,
                metrics={"gbm_ic": float(icg["ic"]), "gbm_ic_nw_t": float(icg["ic_nw_t"]),
                         "gbm_spread_bp": float(icg["spread_bp"]), "best_SR": float(best.sharpe),
                         "best_t": float(best.t), "best_mean_bp": float(best.mean_bp), "DSR_prob": float(best.dsr_p),
                         "n_pass": int(PR["pass"].sum())},
                verdict="promising" if PR["pass"].any() else "rejected",
                notes=f"best portfolio {best['name']}; halves {best.h1_bp:+.1f}/{best.h2_bp:+.1f} bp",
                code=CODE + "xsec_hourly.py")
    return FM, PR


if __name__ == "__main__":
    main("--log" in sys.argv)
