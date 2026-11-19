"""Fetch 10 years of daily bars (+ dividend and split events) for the liquid universe, at most 3 threads.

  python3 tools/research/xsec_swing/fetch.py            # superset + ETFs; skips symbols already complete
  python3 tools/research/xsec_swing/fetch.py --all      # also every name in trader/universe.txt (today's listings,
                                                        # incl. those now below $5 / thin: survivorship mitigation)
Bars are merged into trader/data/1d/<SYM>.npz through yfetch.save (the cache merges, never overwrites).
Dividends/splits go to trader/data/1d_events/<SYM>.json ({"div": {epoch: amount}, "split": {epoch: ratio}}).
"""
import sys, json, time, random, concurrent.futures as cf
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools/research")
import yfetch

ROOT = Path("trader/data")
EV = ROOT / "1d_events"
ETFS = ["SPY", "QQQ", "IWM", "RSP", "DIA", "MDY", "IJR", "UPRO", "TQQQ", "TNA", "SOXL", "FAS", "SPXU",
        "SQQQ", "SOXS", "TZA", "SMH", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
        "XLRE", "XLC"]


def fetch(sym, rng="10y"):
    doc = yfetch.get("/v8/finance/chart/%s?range=%s&interval=1d&includePrePost=false&events=div%%7Csplit"
                     % (yfetch.ysym(sym), rng))
    if not doc or not doc.get("chart", {}).get("result"):
        return None, None
    res = doc["chart"]["result"][0]
    t = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    if not t or not q:
        return None, None
    arr = {k: np.array([np.nan if x is None else x for x in (q.get(k) or [None] * len(t))], dtype=float)
           for k in ("open", "high", "low", "close", "volume")}
    t = np.array(t, dtype=np.int64)
    ok = ~np.isnan(arr["close"])
    bars = {"t": t[ok], **{k: v[ok] for k, v in arr.items()},
            "instrumentType": res.get("meta", {}).get("instrumentType", "")}
    ev = res.get("events", {}) or {}
    events = {"div": {k: v.get("amount") for k, v in (ev.get("dividends") or {}).items()},
              "split": {k: [v.get("numerator"), v.get("denominator")] for k, v in (ev.get("splits") or {}).items()},
              "firstTradeDate": res.get("meta", {}).get("firstTradeDate")}
    return bars, events


def done(sym):
    p = ROOT / "1d" / (yfetch.ysym(sym) + ".npz")
    e = EV / (yfetch.ysym(sym) + ".json")
    if not (p.exists() and e.exists()):
        return False
    z = np.load(p)
    return len(z["t"]) > 0


def one(sym):
    for k in range(3):
        try:
            bars, events = fetch(sym)
            break
        except Exception as e:   # yfetch.get already retried 5 times; back off and try again
            time.sleep(5 + 5 * k + random.random())
    else:
        return sym, "error"
    if bars is None or len(bars["t"]) == 0:
        return sym, "empty"
    yfetch.save(sym, "1d", bars)
    json.dump(events, open(EV / (yfetch.ysym(sym) + ".json"), "w"))
    time.sleep(0.2 + 0.2 * random.random())
    return sym, "ok"


def main(workers=3, everything=False):
    (ROOT / "1d").mkdir(parents=True, exist_ok=True)
    EV.mkdir(parents=True, exist_ok=True)
    syms = ETFS + [x["symbol"] for x in json.load(open(ROOT / "superset.json"))]
    if everything:
        syms += [l.split("|")[0].strip() for l in open("trader/universe.txt") if l.strip() and not l.startswith("#")]
    seen, todo = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            if not done(s):
                todo.append(s)
    print("to fetch", len(todo), "of", len(seen), flush=True)
    stats, n, t0 = {}, 0, time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        for s, st in ex.map(one, todo):
            n += 1
            stats[st] = stats.get(st, 0) + 1
            if st != "ok":
                print("  ", s, st, flush=True)
            if n % 100 == 0:
                print(n, "of", len(todo), stats, "%.0fs" % (time.time() - t0), flush=True)
    print("finished", stats, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main(everything="--all" in sys.argv)
