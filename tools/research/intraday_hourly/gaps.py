"""(a) Stock overnight gaps -> same-day continuation vs reversal, bucketed by gap size and first-hour relative
volume (news proxy), ~710 sessions of hourly bars.

Universe each day (point in time): full hourly session, prior close >= $5, 14-day mean volume >= 1M shares.
Diagnostic grid: for each gap bucket x RV bucket, the daily equal-weight mean of the 10:30 open -> close return
(RV is known only at 10:30), and per gap bucket the 9:30 open -> close return; both raw (what a long position earns
before costs) and in excess of the day's universe mean. t-stats on the daily cell series (Newey-West).
Tradeable long-only rules (whole shares, $1,000, top-K, 10 bp; 20 bp sensitivity):
  GD-930  gap <= -2%: buy at the 9:30 open the K largest gap-downs, sell at the close (reversal of gap-downs)
  GD-1030 gap <= -2% and RV < 2 (no news): buy at 10:30, K largest gap-downs
  GU-1030 gap >= +2% and RV >= 2 (news): buy at 10:30, K highest RV (continuation of news gaps)
  GU-cont gap >= +2% and RV >= 2 and first hour up: buy at 10:30, K highest RV
  GD-1030all gap <= -2%: buy at 10:30, K largest gap-downs (late-entry version of GD-930)
Run: python3 tools/research/intraday_hourly/gaps.py [--log]
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from panel import load, features
from sim import topk
from common import evaluate, fmt, passes, nw_t, log_exp, regime_below, OUT, CODE

COST = 0.0010
WARM = 20
GB = [(-1, -0.04, "<=-4%"), (-0.04, -0.02, "-4..-2%"), (-0.02, -0.005, "-2..-0.5%"), (-0.005, 0.005, "flat"),
      (0.005, 0.02, "0.5..2%"), (0.02, 0.04, "2..4%"), (0.04, 9, ">=4%")]
RB = [(0, 1, "rv<1"), (1, 2, "1-2"), (2, 4, "2-4"), (4, 1e9, "rv>=4")]


def main(do_log=False):
    P = load(); F = features(P)
    o, c = P["o"].astype(float), P["c"].astype(float)
    dates = pd.to_datetime(pd.Series(P["dates"])).dt.date.to_numpy()
    N, T = F["y"].shape
    U = F["full"] & (F["prevC"] >= 5) & (F["adv"] >= 1e6) & np.isfinite(F["gap"]) & np.isfinite(F["rv"])
    U[:, :WARM] = False
    y, y930, gap, rv = F["y"], F["y930"], F["gap"], F["rv"]
    ymean = np.nanmean(np.where(U, y, np.nan), axis=0)
    y9mean = np.nanmean(np.where(U, y930, np.nan), axis=0)
    print(f"universe stock-days {int(U.sum())}, days {T-WARM}")
    rows = []
    days = range(WARM, T)

    def cell(mask, ret, mean_all, label, entry):
        raw, exc, n = [], [], []
        for t in days:
            m = mask[:, t]
            if m.sum() >= 3:
                raw.append(np.mean(ret[m, t])); exc.append(np.mean(ret[m, t]) - mean_all[t]); n.append(m.sum())
        raw, exc = pd.Series(raw), pd.Series(exc)
        if len(raw) < 20:
            return None
        return {"cell": label, "entry": entry, "days": len(raw), "avg_n": float(np.mean(n)),
                "raw_bp": 1e4 * raw.mean(), "raw_net_bp": 1e4 * (raw.mean() - COST), "raw_nw_t": nw_t(raw),
                "net_nw_t": nw_t(raw - COST), "exc_bp": 1e4 * exc.mean(), "exc_nw_t": nw_t(exc),
                "h1_exc": 1e4 * exc.iloc[:len(exc) // 2].mean(), "h2_exc": 1e4 * exc.iloc[len(exc) // 2:].mean()}

    for lo, hi, gl in GB:
        gm = U & (gap > lo) & (gap <= hi)
        r = cell(gm, y930, y9mean, f"{gl} all RV", "9:30")
        if r: rows.append(r)
        for rlo, rhi, rl in RB:
            r = cell(gm & (rv >= rlo) & (rv < rhi), y, ymean, f"{gl} {rl}", "10:30")
            if r: rows.append(r)
    A = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print("\nGap x first-hour RV cells: daily equal-weight means (raw = long before costs; exc = minus universe mean)")
    print(A.round(2).to_string(index=False))
    A.to_csv(OUT / "gaps_cells.csv", index=False)
    n_cells = len(A)

    # tradeable long-only top-K rules
    below = regime_below(dates, "SPY")
    up1h = c[:, :, 0] > o[:, :, 0]
    rules = {
        "GD-930 gap<=-2%, buy 9:30 open": (U & (gap <= -0.02), -gap, o[:, :, 0]),
        "GD-1030 gap<=-2% & RV<2, buy 10:30": (U & (gap <= -0.02) & (rv < 2), -gap, o[:, :, 1]),
        "GD-1030all gap<=-2%, buy 10:30 (late entry)": (U & (gap <= -0.02), -gap, o[:, :, 1]),
        "GU-1030 gap>=2% & RV>=2, buy 10:30": (U & (gap >= 0.02) & (rv >= 2), rv, o[:, :, 1]),
        "GU-cont gap>=2% & RV>=2 & 1st hour up, buy 10:30": (U & (gap >= 0.02) & (rv >= 2) & up1h, rv, o[:, :, 1]),
    }
    res = {}
    for name, (el, score, entry) in rules.items():
        for K in (1, 3, 5, 10):
            sc = np.where(np.isfinite(score), score, -np.inf)
            w, fr, npos = topk(sc, el & np.isfinite(entry), entry, c[:, :, 6], dates, K, cost=COST)
            w20, _, _ = topk(sc, el & np.isfinite(entry), entry, c[:, :, 6], dates, K, cost=0.0020)
            res[(name, K)] = (w.iloc[WARM:], w20.iloc[WARM:], fr.iloc[WARM:], npos.iloc[WARM:])
    srs = [v[0].mean() / v[0].std(ddof=1) * math.sqrt(252) for v in res.values()]
    var = float(np.var(srs, ddof=1))
    n_tr = len(res) + n_cells
    print(f"\nTradeable long-only rules: {len(res)} variants (+{n_cells} diagnostic cells = {n_tr} trials), SR var {var:.2f}")
    prow = []
    for (name, K), (w, w20, fr, npos) in res.items():
        s = evaluate(w, f"{name} K={K}", n_tr, var)
        wb = w[below[WARM:]]
        s.update({"rule": name, "K": K, "net20_bp": 1e4 * w20.mean(), "frac_bp": 1e4 * fr.mean(),
                  "active_days": int((npos > 0).sum()), "below200_bp": 1e4 * wb.mean(),
                  "below200_t": wb.mean() / wb.std(ddof=1) * math.sqrt(len(wb)) if wb.std() > 0 else 0, "pass": passes(s)})
        prow.append(s)
        print(fmt(s), f"| 20bp {s['net20_bp']:+.2f} | frac {s['frac_bp']:+.2f} | SPY<200d {s['below200_bp']:+.1f} "
                      f"t {s['below200_t']:+.2f}", " PASS" if s["pass"] else "")
    PR = pd.DataFrame(prow); PR.to_csv(OUT / "gaps_rules.csv", index=False)
    if do_log:
        best = PR.loc[PR.sharpe.idxmax()]
        bc = A.loc[A.exc_nw_t.abs().idxmax()]
        log_exp(family="gaps_intraday", name="Stock gaps x first-hour RV -> same-day continuation/reversal (hourly, 710 days)",
                hypothesis="News gaps (high first-hour RV) continue intraday, no-news gaps reverse; long-only: buy gap-down "
                           "reversals or news gap-up continuations",
                data=f"Yahoo 1h, {N} stocks, {int(U.sum())} stock-days, {dates[WARM]}..{dates[-1]}",
                method=f"{n_cells} gap x RV cells (daily EW means, NW t, raw and market-excess) + {len(res)} top-K "
                       "long-only rules (whole shares, $1000, 10 bp)",
                variants=n_tr,
                metrics={"best_rule_SR": float(best.sharpe), "best_rule_t": float(best.t),
                         "best_rule_mean_bp": float(best.mean_bp), "DSR_prob": float(best.dsr_p),
                         "max_cell_exc_t": float(bc.exc_nw_t), "n_pass": int(PR["pass"].sum())},
                verdict="promising" if PR["pass"].any() else "rejected",
                notes=f"best rule {best['name']} (halves {best.h1_bp:+.1f}/{best.h2_bp:+.1f}); most extreme cell "
                      f"{bc.cell} @ {bc.entry}: excess {bc.exc_bp:+.1f} bp NW t {bc.exc_nw_t:+.2f}",
                code=CODE + "gaps.py")
    return A, PR


if __name__ == "__main__":
    main("--log" in sys.argv)
