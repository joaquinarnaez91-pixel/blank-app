"""Fetch the daily data the overnight program needs: full daily history (Yahoo, period1/period2 so the
bars stay DAILY; range=max silently returns monthly bars) merged into trader/data/1d/, plus dividends.
At most 3 download threads.

  python3 tools/research/overnight/fetch.py              # default symbol list
  python3 tools/research/overnight/fetch.py SYM ...
Dividends (ex-date, cash amount) go to trader/data/div/<SYM>.csv (git-ignored cache).
"""
import sys, time, csv, datetime, concurrent.futures as cf
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools/research")
import yfetch

BARS = ["SPY", "QQQ", "IWM", "SMH", "XLF", "HYG", "IEF", "TLT", "LQD",
        "^VIX", "^VIX3M", "^VIX9D", "^IRX",
        "UPRO", "TQQQ", "TNA", "SOXL", "FAS"]
START = 631152000  # 1990-01-01


def fetch_daily(sym):
    doc = yfetch.get("/v8/finance/chart/%s?period1=%d&period2=%d&interval=1d&includePrePost=false&events=div"
                     % (yfetch.ysym(sym), START, int(time.time())))
    res = (doc or {}).get("chart", {}).get("result") or []
    if not res:
        return sym, 0, 0
    res = res[0]
    if res.get("meta", {}).get("dataGranularity") != "1d":
        raise RuntimeError("not daily: %s %s" % (sym, res.get("meta", {}).get("dataGranularity")))
    t = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    arr = {k: np.array([np.nan if x is None else x for x in (q.get(k) or [None] * len(t))], dtype=float)
           for k in ("open", "high", "low", "close", "volume")}
    t = np.array(t, dtype=np.int64)
    ok = ~np.isnan(arr["close"])
    d = {"t": t[ok], **{k: v[ok] for k, v in arr.items()},
         "instrumentType": res.get("meta", {}).get("instrumentType", "")}
    if len(d["t"]):
        yfetch.save(sym, "1d", d)
    divs = (res.get("events", {}) or {}).get("dividends") or {}
    rows = []
    for k, v in sorted(divs.items(), key=lambda kv: int(kv[0])):
        dt = datetime.datetime.fromtimestamp(int(v["date"]), datetime.timezone(datetime.timedelta(hours=-5))).date()
        rows.append((dt.isoformat(), float(v["amount"])))
    if rows:
        out = Path("trader/data/div"); out.mkdir(parents=True, exist_ok=True)
        with open(out / (yfetch.ysym(sym) + ".csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["exdate", "amount"]); w.writerows(rows)
    return sym, int(ok.sum()), len(rows)


if __name__ == "__main__":
    syms = sys.argv[1:] or BARS
    with cf.ThreadPoolExecutor(3) as ex:
        for s, n, nd in ex.map(fetch_daily, syms):
            print(f"{s:7s} daily bars {n:5d} dividends {nd}", flush=True)
