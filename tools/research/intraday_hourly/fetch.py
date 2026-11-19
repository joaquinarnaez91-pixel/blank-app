"""Fetch 1h/730d Yahoo bars for the liquid universe (trader/data/superset.json) with at most 3 threads.

Wraps tools/research/yfetch.py (same request, same merge-into-cache save) but
  - caps concurrency at 3 threads (Yahoo pacing rule),
  - catches failures per symbol instead of aborting the whole run,
  - skips symbols already cached with >= MIN_SESSIONS sessions (resumable),
  - writes a log of failures to trader/data/1h_fetch_failures.json.

  python3 tools/research/intraday_hourly/fetch.py            # whole universe
  python3 tools/research/intraday_hourly/fetch.py AAPL MSFT  # single symbols
"""
import sys, json, time, random, concurrent.futures as cf
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools/research")
import yfetch

INTERVAL, RANGE, WORKERS, MIN_SESSIONS = "1h", "730d", 3, 700
ROOT = Path("trader/data")


def cached_sessions(sym):
    p = ROOT / INTERVAL / (yfetch.ysym(sym) + ".npz")
    if not p.exists():
        return 0
    t = np.load(p)["t"]
    return len(np.unique((t - 5 * 3600) // 86400))      # rough count of distinct NY dates


def one(sym):
    if cached_sessions(sym) >= MIN_SESSIONS:
        return sym, "cached"
    for attempt in range(3):
        try:
            d = yfetch.fetch_bars(sym, INTERVAL, RANGE)
            if d is None or len(d["t"]) == 0:
                return sym, "empty"
            yfetch.save(sym, INTERVAL, d)
            time.sleep(0.15 + 0.1 * random.random())     # gentle pacing per thread
            return sym, "ok"
        except Exception as e:                           # yfetch already retried 5x (429s etc.)
            time.sleep(10 * (attempt + 1))
            err = str(e)
    return sym, "fail:" + err


def main(syms):
    (ROOT / INTERVAL).mkdir(parents=True, exist_ok=True)
    stat, fails, t0 = {}, [], time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for k, (s, st) in enumerate(ex.map(one, syms), 1):
            key = st.split(":")[0]
            stat[key] = stat.get(key, 0) + 1
            if key != "ok" and key != "cached":
                fails.append([s, st])
            if k % 100 == 0 or k == len(syms):
                print(f"{k}/{len(syms)} {stat} {time.time()-t0:.0f}s", flush=True)
    json.dump(fails, open(ROOT / "1h_fetch_failures.json", "w"))
    print("done", stat, "failures", fails[:30])


if __name__ == "__main__":
    a = sys.argv[1:]
    syms = a if a else [x["symbol"] for x in json.load(open(ROOT / "superset.json"))]
    main(syms)
