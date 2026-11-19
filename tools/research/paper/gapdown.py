"""Paper forward test: cash-day gap-down reversal (candidate from agent-intraday-hourly, Nov 19 2026).

Rule: on day t, if SPY closed below its 200-day SMA on t-1, take stocks in trader/data/superset.json with prior close
>= $5 and 14-day average volume >= 1M shares (both through t-1), gap = open_t / close_t-1 - 1 <= -2%; the 5 most
negative gaps are bought; all are sold at 15:55. Paper returns are net of 10 bp, for entries at the opening print
(the backtest's assumption) and at the 9:35, 9:45 and 10:00 bar opens (the owner's entry window starts at 9:45).

  python3 tools/research/paper/gapdown.py today          # after the close: fetch today's bars, append to the log
  python3 tools/research/paper/gapdown.py backfill       # recent cash days in the cached 5-minute window
Log: trader/paper/gapdown.csv (git-ignored).
"""
import sys, json, math, datetime, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, "tools/research")
import numpy as np, pandas as pd
import yfetch
from bars import frame, NY

COST, K = 0.0010, 5
ENTRIES = {"open": 570, "9:35": 575, "9:45": 585, "10:00": 600}
LOG = Path("trader/paper/gapdown.csv")


def spy_below_200d(day):
    d = frame("SPY", "1d").groupby("date").c.last()
    prior = d[d.index < day]
    return bool(prior.iloc[-1] < prior.iloc[-200:].mean()), float(prior.iloc[-1] / prior.iloc[-200:].mean() - 1)


def daily_row(sym, day):
    d = frame(sym, "1d")
    if d is None:
        return None
    g = d.groupby("date").agg(o=("o", "first"), c=("c", "last"), v=("v", "sum"))
    if day not in g.index:
        return None
    prior = g[g.index < day]
    if len(prior) < 14:
        return None
    return {"sym": sym, "prev_close": float(prior.c.iloc[-1]), "avgv14": float(prior.v.iloc[-14:].mean()),
            "open": float(g.loc[day, "o"]), "close": float(g.loc[day, "c"])}


def evaluate(day, rows, fetch_5m):
    U = pd.DataFrame([r for r in rows if r])
    U = U[(U.prev_close >= 5) & (U.avgv14 >= 1e6)].copy()
    U["gap"] = U.open / U.prev_close - 1
    picks = U[U.gap <= -0.02].sort_values("gap").head(K)
    out = []
    for _, p in picks.iterrows():
        if fetch_5m:
            yfetch.save(p.sym, "5m", yfetch.fetch_bars(p.sym, "5m", "5d"))
        m = frame(p.sym, "5m")
        m = m[m.date == day] if m is not None else None
        if m is None or m.empty or 955 not in set(m.tod):
            continue
        px = {k: float(m[m.tod == t].o.iloc[0]) for k, t in ENTRIES.items() if t in set(m.tod)}
        exitp = float(m[m.tod == 955].o.iloc[0])
        out.append({"sym": p.sym, "gap": round(p.gap, 4), "prev_close": p.prev_close, **{f"px_{k}": v for k, v in px.items()},
                    "exit_1555": exitp, **{f"ret_{k}": exitp / v - 1 - COST for k, v in px.items()}})
    return out, len(U)


def record(day, regime_gap, picks, n_univ):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"date": day.isoformat(), "spy_vs_200d": round(regime_gap, 4), "universe": n_univ, "n_picks": len(picks),
           "picks": json.dumps(picks)}
    for k in ENTRIES:
        vals = [p.get(f"ret_{k}") for p in picks if p.get(f"ret_{k}") is not None]
        row[f"port_{k}"] = float(np.mean(vals)) if vals else None   # equal slots; empty slots count as cash below
        row[f"sleeve_{k}"] = float(np.sum(vals) / K) if vals else 0.0
    df = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    df = df[df.date != row["date"]] if len(df) else df
    pd.concat([df, pd.DataFrame([row])]).sort_values("date").to_csv(LOG, index=False)
    print(f"{day}: SPY {100*regime_gap:+.2f}% vs 200-day; {len(picks)} picks of {n_univ} eligible: "
          + ", ".join(f"{p['sym']} {100*p['gap']:+.1f}%" for p in picks))
    print("   paper return per slot-weighted portfolio (net 10 bp): " +
          "  ".join(f"{k} {100*row[f'sleeve_{k}']:+.2f}%" for k in ENTRIES))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "today"
    syms = [x["symbol"] for x in json.load(open("trader/data/superset.json"))]
    if mode == "today":
        day = datetime.datetime.now(NY).date()
        yfetch.save("SPY", "1d", yfetch.fetch_bars("SPY", "1d", "1mo"))
        below, gap = spy_below_200d(day)
        if not below:
            print(f"{day}: SPY closed above its 200-day yesterday ({100*gap:+.2f}%): not a cash day, nothing to record."); return
        def upd(s):
            try:
                d = yfetch.fetch_bars(s, "1d", "1mo")
                if d is not None and len(d["t"]):
                    yfetch.save(s, "1d", d)
            except Exception:
                pass
            return daily_row(s, day)
        with cf.ThreadPoolExecutor(4) as ex:
            rows = list(ex.map(upd, syms))
        picks, n = evaluate(day, rows, fetch_5m=True)
        record(day, gap, picks, n)
    elif mode == "backfill":
        days5 = sorted(frame("SPY", "5m").date.unique())
        for day in days5:
            below, gap = spy_below_200d(day)
            if not below:
                continue
            rows = [daily_row(s, day) for s in syms]
            picks, n = evaluate(day, rows, fetch_5m=False)
            record(day, gap, picks, n)


if __name__ == "__main__":
    main()
