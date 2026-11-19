"""Live picker for the cash-day gap-down pilot (pre-registered in tools/research/registry.jsonl, Nov 19 2026).

  python3 tools/research/paper/gapdown_live.py picks [--top 10] [--any-day]
At or after 9:31 ET: ranks the liquid universe by gap = first 1-minute price / prior close - 1 using Yahoo's batch
endpoint (20 symbols per request) and writes trader/gapdown_picks_<date>.json. Nothing is placed.
"""
import sys, json, argparse, datetime, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, "tools/research")
import numpy as np
import yfetch
from bars import frame, NY

HOLDINGS = {"NVDA", "TSM", "AMD", "CEG", "VST", "CCJ", "GLD", "SPCX", "IONQ", "SGOV", "VOO"}
MAX_PRICE = 95.0


def avgv14(sym, before):
    d = frame(sym, "1d")
    if d is None or len(d) < 15:
        return None
    g = d.groupby("date").v.sum()
    g = g[g.index < before]
    return float(g.iloc[-14:].mean()) if len(g) >= 14 else None


def spark(batch):
    try:
        doc = yfetch.get("/v7/finance/spark?symbols=%s&range=1d&interval=1m" % ",".join(yfetch.ysym(s) for s in batch))
    except Exception:
        return []
    out = []
    for r in (doc or {}).get("spark", {}).get("result", []) or []:
        resp = (r.get("response") or [{}])[0]
        t = resp.get("timestamp") or []
        c = ((resp.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        m = resp.get("meta", {})
        pairs = [(a, b) for a, b in zip(t, c) if b is not None]
        if not pairs:
            continue
        out.append({"sym": r.get("symbol"), "t0": pairs[0][0], "first": pairs[0][1], "last": pairs[-1][1],
                    "prev_close": m.get("chartPreviousClose")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--any-day", action="store_true", help="dry run on whatever session Yahoo returns")
    a = ap.parse_args()
    syms = [x["symbol"] for x in json.load(open("trader/data/superset.json"))]
    batches = [syms[i:i + 20] for i in range(0, len(syms), 20)]
    with cf.ThreadPoolExecutor(4) as ex:
        rows = [r for part in ex.map(spark, batches) for r in part]
    if not rows:
        raise SystemExit("no data from Yahoo")
    day = datetime.datetime.fromtimestamp(rows[0]["t0"], NY).date()
    today = datetime.datetime.now(NY).date()
    if day != today and not a.any_day:
        raise SystemExit(f"Yahoo's latest session is {day}, not today ({today}): run after 9:31 ET.")
    cands = []
    for r in rows:
        sym = r["sym"].replace("-", ".")
        if sym in HOLDINGS or not r["prev_close"] or r["prev_close"] < 5:
            continue
        t0 = datetime.datetime.fromtimestamp(r["t0"], NY)
        if t0.strftime("%H:%M") != "09:30":
            continue
        gap = r["first"] / r["prev_close"] - 1
        if gap > -0.02:
            continue
        v = avgv14(sym, day)
        if v is None or v < 1e6:
            continue
        cands.append({"sym": sym, "gap": round(gap, 4), "first": r["first"], "last": r["last"], "prev_close": r["prev_close"], "avgv14": int(v)})
    cands.sort(key=lambda x: x["gap"])
    out = {"date": day.isoformat(), "made_at": datetime.datetime.now(NY).isoformat(timespec="seconds"), "scanned": len(rows),
           "candidates": cands[:max(a.top, 15)]}
    Path("trader").mkdir(exist_ok=True)
    json.dump(out, open(f"trader/gapdown_picks_{day.isoformat()}.json", "w"), indent=1)
    print(f"{day}: scanned {len(rows)} of {len(syms)}; {len(cands)} gap-downs <= -2% pass the filters. Top {a.top}:")
    for c in cands[:a.top]:
        flag = "" if c["last"] <= MAX_PRICE else "  (above $95: skip)"
        print(f"  {c['sym']:<6} gap {100*c['gap']:+6.1f}%  prev ${c['prev_close']:.2f}  first ${c['first']:.2f}  now ${c['last']:.2f}  avg vol {c['avgv14']:,}{flag}")


if __name__ == "__main__":
    main()
