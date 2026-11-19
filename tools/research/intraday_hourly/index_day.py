"""Index-level daytime ideas on SPY/QQQ/IWM, traded long-only through 3x funds (UPRO/SPXU, TQQQ/SQQQ, TNA/TZA),
on ~725 full sessions of hourly bars (Dec 2023 - Nov 2026). Costs: 5 bp per round trip on the fund.

Point in time: the 200-day regime for day t uses daily closes through t-1 (that is exactly when the live overnight
model holds cash for the night before day t). Gaps use the 9:30 open vs the prior official close. First-hour
relative volume = 9:30-bar volume / mean of the previous 14 sessions' 9:30-bar volume (known at 10:30).

Tests (every variant is counted; one registry record per experiment):
  A. Index gap -> same-day continuation vs reversal, bucketed by gap size (z = gap / trailing 20-day gap std) and
     first-hour RV, entry 9:30 (gap only) or 10:30 (gap x RV); traded with the fund in the continuation direction.
  D1. Regime day trades: below the 200-day, long bull fund (or bear fund) 9:30->close / 10:30->close.
  D2. Rebound days below the 200-day: gap down or prior day down -> bull fund 9:30->close / 10:30->close.
  D3. Gap fill after large gaps (|gap| > 0.5% / 1.0%): fade with the fund, exit at the fill (limit at the fund's
      prior close, which is where a 3x fund sits when its index is back at its prior close) or at the close.
  D4. Last half hour reversal: fade the day-so-far (prior close -> 15:30) from the 15:30 open to the close;
      also fade from the 14:30 open (90 min).
Run: python3 tools/research/intraday_hourly/index_day.py [--log]
"""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "tools/research"); sys.path.insert(0, "tools/research/intraday_hourly")
from bars import frame
from common import hourly_pivot, evaluate, fmt, passes, log_exp, OUT, CODE

COST = 0.0005
IDX = [("SPY", "UPRO", "SPXU"), ("QQQ", "TQQQ", "SQQQ"), ("IWM", "TNA", "TZA")]


def daily_from_hourly(sym):
    df = frame(sym, "1h")
    g = df.groupby("date")
    return pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "v": g.v.sum()})


def load_index(idx, bull, bear):
    d1 = frame(idx, "1d")
    d1 = d1[~d1.index.duplicated()].set_index("date")
    close = d1.c
    sma = close.rolling(200).mean()
    below_close = (close < sma)                         # known at day t's close
    di, I = hourly_pivot(idx)
    db, B = hourly_pivot(bull)
    ds, S = hourly_pivot(bear)
    dates = sorted(set(di) & set(db) & set(ds))
    ii = pd.Index(di).get_indexer(dates); ib = pd.Index(db).get_indexer(dates); is_ = pd.Index(ds).get_indexer(dates)
    I = {k: v[ii] for k, v in I.items()}; B = {k: v[ib] for k, v in B.items()}; S = {k: v[is_] for k, v in S.items()}
    dates = pd.Index(dates)
    # prior official close (daily file) and prior-day info, aligned to the hourly dates
    pos = close.index.get_indexer(dates)
    assert (pos > 1).all()
    prevC = close.to_numpy()[pos - 1]
    prev2C = close.to_numpy()[pos - 2]
    below = below_close.to_numpy()[pos - 1]             # regime known before day t opens
    dist = (close / sma - 1).to_numpy()[pos - 1]
    # the funds' prior close (from all hourly bars, early-close days included)
    fb = daily_from_hourly(bull).c; fs = daily_from_hourly(bear).c
    pb = fb.shift(1).reindex(dates).to_numpy(); ps = fs.shift(1).reindex(dates).to_numpy()
    gap = I["o"][:, 0] / prevC - 1
    # trailing 20-day std of overnight gaps (all daily-file days, through t-1)
    og = (d1.o / close.shift(1) - 1)
    gsd = og.rolling(20).std().shift(1).reindex(dates).to_numpy()
    v0 = pd.Series(I["v"][:, 0], index=dates)
    rv = (v0 / v0.rolling(14).mean().shift(1)).to_numpy()
    return dict(dates=dates, I=I, B=B, S=S, prevC=prevC, prev2C=prev2C, below=below, dist=dist, pb=pb, ps=ps,
                gap=gap, gz=gap / gsd, rv=rv)


def fund_ret(F, entry_bar, cost=COST):
    return F["c"][:, 6] / F["o"][:, entry_bar] - 1 - cost


def fill_ret(F, target, cost=COST):
    """Buy the fund at the 9:30 open; limit sell at `target` (fund's prior close); else sell at the close."""
    entry = F["o"][:, 0]
    hit = np.nanmax(F["h"], axis=1) >= target + 0.01
    out = np.where(hit, target, F["c"][:, 6]) / entry - 1 - cost
    return out, hit


def run():
    D = {idx: load_index(idx, b, s) for idx, b, s in IDX}
    common = sorted(set.intersection(*[set(D[i]["dates"]) for i in D]))
    for i in D:                                          # align all three on common dates
        k = D[i]["dates"].get_indexer(common)
        for key, val in list(D[i].items()):
            if key == "dates":
                continue
            if isinstance(val, dict):
                D[i][key] = {f: a[k] for f, a in val.items()}
            else:
                D[i][key] = val[k]
        D[i]["dates"] = pd.Index(common)
    dates = pd.Index(common)
    # drop the first 15 sessions (RV warm-up)
    start = 15
    print(f"sessions {len(dates) - start}: {dates[start]} .. {dates[-1]}")
    for i in D:
        b = D[i]["below"][start:]
        print(f"  {i}: below-200d days {int(b.sum())} ({b.mean():.0%})")
    allb = np.all([D[i]["below"] for i in D], axis=0)[start:]
    anyb = np.any([D[i]["below"] for i in D], axis=0)[start:]
    print(f"  all three below: {int(allb.sum())}, any below: {int(anyb.sum())}")

    results = {}

    def add(name, per_index, family_tag):
        """per_index: dict idx -> daily return array (0 = flat). Records each index and the 1/3-each portfolio."""
        for i, r in per_index.items():
            results[(family_tag, name, i)] = pd.Series(np.nan_to_num(r[start:]), index=dates[start:])
        port = np.mean([np.nan_to_num(per_index[i]) for i in per_index], axis=0)
        results[(family_tag, name, "EW3")] = pd.Series(port[start:], index=dates[start:])

    # ---------- A. gap continuation vs reversal (diagnostic grid on the index itself, then fund trades) ----------
    print("\nA. Index gap -> rest of day, signed in the gap direction (positive = continuation), gross bp of the INDEX")
    rows = []
    for i in D:
        x = D[i]; I = x["I"]
        r930 = I["c"][:, 6] / I["o"][:, 0] - 1
        r1030 = I["c"][:, 6] / I["o"][:, 1] - 1
        sgn = np.sign(x["gap"])
        az = np.abs(x["gz"])
        for zb, (lo, hi) in {"z<0.5": (0, .5), "0.5-1.5": (.5, 1.5), "z>1.5": (1.5, 99)}.items():
            for rvb, (rlo, rhi) in {"rv<1": (0, 1), "rv>=1": (1, 1e9), "all": (0, 1e9)}.items():
                m = (az >= lo) & (az < hi) & (x["rv"] >= rlo) & (x["rv"] < rhi)
                m[:start] = False
                for lab, r in (("9:30", r930), ("10:30", r1030)):
                    if rvb != "all" and lab == "9:30":
                        continue                     # RV is not known at 9:30
                    s = sgn[m] * r[m]
                    if len(s) < 10:
                        continue
                    rows.append({"idx": i, "z": zb, "rv": rvb, "entry": lab, "n": int(m.sum()),
                                 "cont_bp": 1e4 * s.mean(), "t": s.mean() / s.std(ddof=1) * math.sqrt(len(s))})
    A = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(A.round(2).to_string(index=False))
    nA = len(A)
    # tradeable version of the strongest-prior cells: large gaps (z>1.5) fade (reversal) and follow (continuation)
    for lab, eb in (("9:30", 0), ("10:30", 1)):
        for mode in ("follow", "fade"):
            per = {}
            for i in D:
                x = D[i]
                big = np.abs(x["gz"]) > 1.5
                up = x["gap"] > 0
                bull_r, bear_r = fund_ret(x["B"], eb), fund_ret(x["S"], eb)
                if mode == "follow":
                    r = np.where(big & up, bull_r, np.where(big & ~up, bear_r, 0.0))
                else:
                    r = np.where(big & up, bear_r, np.where(big & ~up, bull_r, 0.0))
                per[i] = r
            add(f"A: |gap z|>1.5 {mode} gap, fund {lab}->close", per, "A")

    # ---------- D1. regime day trades ----------
    for lab, eb in (("9:30", 0), ("10:30", 1)):
        for side in ("bull", "bear"):
            per = {i: np.where(D[i]["below"], fund_ret(D[i]["B" if side == "bull" else "S"], eb), 0.0) for i in D}
            add(f"D1: below 200d, {side} fund {lab}->close", per, "D1")
            per = {i: np.where(~D[i]["below"], fund_ret(D[i]["B" if side == "bull" else "S"], eb), 0.0) for i in D}
            add(f"D1: above 200d, {side} fund {lab}->close", per, "D1")
    # ---------- D2. rebound days below the 200-day ----------
    for lab, eb in (("9:30", 0), ("10:30", 1)):
        for cond in ("gap<0", "prev day down", "gap<-0.5%"):
            per = {}
            for i in D:
                x = D[i]
                c = {"gap<0": x["gap"] < 0, "prev day down": x["prevC"] < x["prev2C"], "gap<-0.5%": x["gap"] < -0.005}[cond]
                per[i] = np.where(x["below"] & c, fund_ret(x["B"], eb), 0.0)
            add(f"D2: below 200d & {cond}, bull {lab}->close", per, "D2")
    # ---------- D3. gap fill ----------
    for th in (0.005, 0.010):
        for exit_kind in ("fill", "close"):
            for regime in ("all", "below", "above"):
                per = {}
                for i in D:
                    x = D[i]
                    reg = {"all": np.ones(len(x["gap"]), bool), "below": x["below"], "above": ~x["below"]}[regime]
                    if exit_kind == "fill":
                        rb, _ = fill_ret(x["B"], x["pb"]); rs, _ = fill_ret(x["S"], x["ps"])
                    else:
                        rb, rs = fund_ret(x["B"], 0), fund_ret(x["S"], 0)
                    per[i] = np.where(reg & (x["gap"] < -th), rb, np.where(reg & (x["gap"] > th), rs, 0.0))
                add(f"D3: |gap|>{th:.1%} fade, exit {exit_kind}, {regime}", per, "D3")
            # one-sided: only fade gap DOWNS (buy the bull fund) -- the side a long-only trader prefers
            per = {}
            for i in D:
                x = D[i]
                rb, _ = fill_ret(x["B"], x["pb"]) if exit_kind == "fill" else (fund_ret(x["B"], 0), None)
                per[i] = np.where(x["gap"] < -th, rb, 0.0)
            add(f"D3: gap<-{th:.1%} buy bull, exit {exit_kind}, all", per, "D3")
    # ---------- D4. last half hour / 90-minute reversal ----------
    for lab, eb, sig_bar in (("15:30", 6, 5), ("14:30", 5, 4)):
        for regime in ("all", "below", "above"):
            per = {}
            for i in D:
                x = D[i]
                sofar = x["I"]["c"][:, sig_bar] / x["prevC"] - 1          # prior close -> start of the trade bar
                reg = {"all": np.ones(len(sofar), bool), "below": x["below"], "above": ~x["below"]}[regime]
                rb, rs = fund_ret(x["B"], eb), fund_ret(x["S"], eb)
                per[i] = np.where(reg & (sofar > 0), rs, np.where(reg & (sofar < 0), rb, 0.0))
            add(f"D4: fade day-so-far from {lab} open, {regime}", per, "D4")
    return results, nA, D, dates, start


def main():
    results, nA, D, dates, start = run()
    fams = {}
    for (fam, name, i), r in results.items():
        fams.setdefault(fam, []).append((name, i, r))
    out = {}
    print()
    for fam, lst in fams.items():
        # SR variance across all variants of this family (for the deflated Sharpe)
        srs = [r.mean() / r.std(ddof=1) * math.sqrt(252) for _, _, r in lst if r.std(ddof=1) > 0]
        var = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.25
        n_tr = len(lst) + (nA if fam == "A" else 0)
        print(f"== family {fam}: {len(lst)} series (index x rule){' + %d diagnostic cells' % nA if fam == 'A' else ''}, SR var {var:.2f}")
        for name, i, r in lst:
            s = evaluate(r, f"{name} [{i}]", n_tr, var)
            out[(fam, name, i)] = s
            flag = " PASS" if passes(s) else ""
            print(fmt(s) + flag)
    S = pd.DataFrame(out.values())
    S.to_csv(OUT / "index_day_summary.csv", index=False)
    pd.DataFrame({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}).to_csv(OUT / "index_day_daily.csv")
    return S, out, nA


def log_all(S, nA):
    """One registry record per experiment group (A, D1..D4), with the best series and the pre-specified EW3 rows."""
    S = S.copy()
    S["fam"] = S["name"].str.split(":").str[0]
    desc = {
        "A": ("Index gap -> same-day continuation vs reversal (z-buckets x first-hour RV), 3x funds",
              "Overnight index gaps continue (or reverse) intraday depending on size and first-hour volume",
              "36 diagnostic cells (gap z-bucket x RV bucket x entry) on the index + |z|>1.5 follow/fade via 3x funds, "
              "9:30 or 10:30 entry, 5 bp"),
        "D1": ("Index day trades by 200-day regime (bull/bear 3x fund, 9:30 or 10:30 -> close)",
               "Below the 200-day (overnight model in cash) the day session has a usable drift",
               "below/above 200d x bull/bear fund x 9:30/10:30 entry, 5 bp"),
        "D2": ("Rebound days below the 200-day (gap down / prior day down) via 3x bull funds",
               "In downtrends, down-gaps or down days are followed by intraday rebounds",
               "conditions {gap<0, prev day down, gap<-0.5%} x 9:30/10:30 entry, below 200d only, 5 bp"),
        "D3": ("Index gap fill after large overnight gaps via 3x funds (fill-or-close exits)",
               "Large index gaps partially fill intraday",
               "|gap| > {0.5%,1%} x exit {fill (limit at fund prior close +1c), close} x regime {all, below, above}"
               " + one-sided gap-down version, 5 bp"),
        "D4": ("Last-half-hour / 90-minute reversal of the day-so-far via 3x funds, by regime",
               "The day-so-far move reverses into the close (fade at 15:30 or 14:30)",
               "fade from 15:30 or 14:30 open x regime {all, below, above}, 5 bp"),
    }
    for fam, (name, hyp, meth) in desc.items():
        g = S[S.fam == fam]
        best = g.loc[g.sharpe.idxmax()]
        ew = g[g["name"].str.endswith("[EW3]")]
        bew = ew.loc[ew.sharpe.idxmax()]
        nvar = len(g) + (nA if fam == "A" else 0)
        log_exp(family="index_daytime_hourly", name=name, hypothesis=hyp,
                data="Yahoo 1h SPY/QQQ/IWM + UPRO/SPXU/TQQQ/SQQQ/TNA/TZA, 706 full sessions 2024-01-16..2026-11-17; "
                     "200-day SMA from daily closes through t-1",
                method=meth + "; each rule on 3 indices + 1/3-each basket; daily-series t, NW t, DSR within group",
                variants=int(nvar),
                metrics={"best_SR": float(best.sharpe), "best_t": float(best.t), "best_mean_bp": float(best.mean_bp),
                         "best_DSR": float(best.dsr_p), "best_EW3_SR": float(bew.sharpe), "best_EW3_t": float(bew.t),
                         "n_pass": int(sum(passes(r) for r in g.to_dict("records")))},
                verdict="rejected",
                notes=f"best series: {best['name']} (halves {best.h1_bp:+.1f}/{best.h2_bp:+.1f} bp); best basket: "
                      f"{bew['name']}. Nothing passes t>3 / DSR>0.9 / both halves.",
                code=CODE + "index_day.py")


if __name__ == "__main__":
    S, out, nA = main()
    if "--log" in sys.argv:
        log_all(S, nA)
