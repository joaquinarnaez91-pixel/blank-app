#!/usr/bin/env python3
"""Score the day's evaluated setups against what the price actually did.

Usage: python3 tools/score_setups.py SETUPS.json
SETUPS.json: [{"symbol":"GXO","time":"09:50","entry":56.47,"stop":55.42,"pred_1r":0.004,"pred_2r":0.032}, ...]
(time = New York HH:MM of the entry bar; pred_* = the Time Machine's average R, may be null).
Run after the close on the same day (Yahoo 5-minute bars, range 1d). Prints actual R for a
1R and a 2R target with the same stop, flat at 15:55, plus averages split by prediction and time.
"""
import json, statistics as st, sys, urllib.request, datetime, zoneinfo

TZ = zoneinfo.ZoneInfo("America/New_York")


def bars(sym):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=5m&range=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=20))["chart"]["result"][0]
    q = d["indicators"]["quote"][0]
    return [(datetime.datetime.fromtimestamp(t, TZ).strftime("%H:%M"), h, l, c)
            for t, h, l, c in zip(d["timestamp"], q["high"], q["low"], q["close"]) if h]


def actual_r(b, t0, entry, stop, k):
    target, last = entry + k * (entry - stop), entry
    for t, h, l, c in b:
        if t < t0 or t >= "15:55":
            continue
        if l <= stop:          # stop checked first: the conservative reading of a bar that hits both
            return -1.0
        if h >= target:
            return float(k)
        last = c
    return (last - entry) / (entry - stop)


def mean(xs):
    return round(st.mean(xs), 3) if xs else None


def main(path):
    rows = []
    for s in json.load(open(path)):
        try:
            b = bars(s["symbol"])
        except Exception as e:
            print(s["symbol"], "ERR", e)
            continue
        r1 = actual_r(b, s["time"], s["entry"], s["stop"], 1)
        r2 = actual_r(b, s["time"], s["entry"], s["stop"], 2)
        rows.append((s, r1, r2))
        print(f'{s["symbol"]:6} {s["time"]} pred1R {s.get("pred_1r")} actual {r1:+.2f} | pred2R {s.get("pred_2r")} actual {r2:+.2f}')
    if not rows:
        return
    print("setups", len(rows), "| mean actual 1R", mean([r[1] for r in rows]), "| 2R", mean([r[2] for r in rows]),
          "| stopped out", sum(1 for r in rows if r[1] <= -1), "of", len(rows))
    hi = [r[1] for r in rows if (r[0].get("pred_1r") or 0) > 0.02]
    lo = [r[1] for r in rows if (r[0].get("pred_1r") or 0) <= 0.02]
    print("predicted > +0.02R: n", len(hi), "mean actual", mean(hi), "| rest: n", len(lo), "mean actual", mean(lo))
    am = [r[1] for r in rows if r[0]["time"] < "10:30"]
    pm = [r[1] for r in rows if r[0]["time"] >= "10:30"]
    print("before 10:30: n", len(am), "mean", mean(am), "| after: n", len(pm), "mean", mean(pm))


if __name__ == "__main__":
    main(sys.argv[1])
