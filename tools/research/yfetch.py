"""Research data fetcher: Yahoo chart bars -> compact .npz files under trader/data/<interval>/.

  python3 tools/research/yfetch.py superset            # price/volume pass over trader/universe.txt -> trader/data/superset.json
  python3 tools/research/yfetch.py bars 5m 60d SYM...  # or --from trader/data/superset.json
Regular-session bars only (no pre/post). Timestamps are UTC epoch seconds of the bar start.
"""
import json, sys, time, gzip, random, urllib.request, urllib.error, concurrent.futures as cf
from pathlib import Path
import numpy as np

UA = "Mozilla/5.0 (X11; Linux x86_64) research-client"
HOSTS = ["https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com"]
ROOT = Path("trader/data")


def get(path, tries=5):
    last = None
    for k in range(tries):
        host = HOSTS[k % 2]
        try:
            req = urllib.request.Request(host + path, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            last = "HTTP %d" % e.code
            if e.code == 404:
                return None
            time.sleep(1.5 * (k + 1) + random.random())
        except Exception as e:  # network hiccup
            last = str(e)
            time.sleep(1.0 * (k + 1))
    raise RuntimeError(last)


def ysym(s):
    return s.upper().replace(".", "-")


def fetch_bars(sym, interval, rng):
    doc = get("/v8/finance/chart/%s?range=%s&interval=%s&includePrePost=false" % (ysym(sym), rng, interval))
    if not doc or not doc.get("chart", {}).get("result"):
        return None
    res = doc["chart"]["result"][0]
    t = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    if not t or not q:
        return None
    arr = {k: np.array([np.nan if x is None else x for x in (q.get(k) or [None] * len(t))], dtype=float)
           for k in ("open", "high", "low", "close", "volume")}
    t = np.array(t, dtype=np.int64)
    ok = ~np.isnan(arr["close"])
    return {"t": t[ok], **{k: v[ok] for k, v in arr.items()},
            "gmtoffset": res.get("meta", {}).get("gmtoffset", -18000),
            "instrumentType": res.get("meta", {}).get("instrumentType", "")}


def save(sym, interval, d):
    p = ROOT / interval / (ysym(sym) + ".npz")
    np.savez_compressed(p, t=d["t"], o=d["open"], h=d["high"], l=d["low"], c=d["close"], v=d["volume"],
                        itype=np.array(d["instrumentType"]))


def load(sym, interval):
    p = ROOT / interval / (ysym(sym) + ".npz")
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def cmd_superset(min_price=5.0, min_vol=750_000):
    syms = [l.split("|")[0].strip() for l in open("trader/universe.txt") if l.strip() and not l.startswith("#")]
    keep, n = [], 0
    for i in range(0, len(syms), 20):
        batch = syms[i:i + 20]
        try:
            doc = get("/v7/finance/spark?symbols=%s&range=5d&interval=1d" % ",".join(ysym(s) for s in batch))
        except Exception as e:
            print("batch fail", i, e, file=sys.stderr)
            continue
        for r in (doc or {}).get("spark", {}).get("result", []) or []:
            resp = (r.get("response") or [{}])[0]
            m = resp.get("meta", {})
            p, v = m.get("regularMarketPrice"), m.get("regularMarketVolume")
            if p and v and p >= min_price and v >= min_vol and m.get("instrumentType") == "EQUITY":
                keep.append({"symbol": r.get("symbol"), "price": p, "volume": v})
        n += len(batch)
        if (i // 20) % 25 == 0:
            print("spark", n, "of", len(syms), "kept", len(keep), flush=True)
        time.sleep(0.15)
    json.dump(keep, open(ROOT / "superset.json", "w"))
    print("superset", len(keep))


def cmd_bars(interval, rng, syms, workers=6):
    (ROOT / interval).mkdir(parents=True, exist_ok=True)
    done, fail = 0, []

    def one(s):
        d = fetch_bars(s, interval, rng)
        if d is None or len(d["t"]) == 0:
            return s, False
        save(s, interval, d)
        return s, True
    with cf.ThreadPoolExecutor(workers) as ex:
        for s, ok in ex.map(one, syms):
            done += 1
            if not ok:
                fail.append(s)
            if done % 100 == 0:
                print(interval, done, "of", len(syms), "failed", len(fail), flush=True)
    print("done", done, "failed", len(fail), fail[:20])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "superset":
        cmd_superset()
    elif a[0] == "bars":
        interval, rng, rest = a[1], a[2], a[3:]
        if rest and rest[0] == "--from":
            rest = [x["symbol"] for x in json.load(open(rest[1]))]
        cmd_bars(interval, rng, rest)
