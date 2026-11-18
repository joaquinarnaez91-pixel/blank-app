"""Stocks-in-play opening-range breakout, long only, point-in-time (in the spirit of Zarattini, Barbon & Aziz 2024).

Each day, for every stock in the cached 5m universe, using ONLY prior days:
  ATR14 (regular-session daily bars), 14-day average daily volume, 14-day average first-5-minute volume.
  Eligible: open > $5, avg volume >= 1M shares, ATR14 >= $0.50, relative volume RV = first-5m volume / its 14-day avg >= 1.
  Long candidates: first 5-minute candle bullish (close > open). Ranked by RV; portfolio = top K.
Entries:  E1 = buy at the 9:35 open.   E2 = buy-stop one cent above the 5-minute range high (fills at max(level, bar open)).
Stops:    range low | 10% of ATR14 below entry | 25% of ATR14 | none.   Exit: stop, else the 15:55 price.
Intrabar order: bullish bar O-L-H-C, bearish bar O-H-L-C. Costs per round trip: COST (default 10 bp)."""
import sys, math, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "tools/research")
from bars import frame

COST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0010
STOPS = ("orlow", "atr10", "atr25", "none")
ENTRIES = ("E1", "E2")


def sim_day(o, h, l, c, tod, entry_kind, stop_kind, atr):
    """o,h,l,c,tod: numpy arrays for one session (5m bars). Returns (entered, ret, R) or (False,..)."""
    hi0, lo0 = h[0], l[0]
    n = len(o)
    if entry_kind == "E1":
        if n < 2:
            return False, 0.0, 0.0
        ei, entry = 1, o[1]
        start_check_same_bar = True
    else:
        level = round(hi0 + 0.01, 2)
        ei = None
        for j in range(1, n):
            if tod[j] > 925:
                break
            if h[j] >= level:
                ei, entry = j, max(level, o[j]); break
        if ei is None:
            return False, 0.0, 0.0
    if stop_kind == "orlow":
        stop = lo0
    elif stop_kind == "atr10":
        stop = entry - 0.10 * atr
    elif stop_kind == "atr25":
        stop = entry - 0.25 * atr
    else:
        stop = -1.0
    if stop >= entry:
        stop = entry - 0.01
    risk = entry - stop if stop > 0 else 0.10 * atr
    exitp = None
    for j in range(ei, n):
        if tod[j] >= 955:
            exitp = o[j]; break
        if j == ei:
            if entry_kind == "E1":
                if l[j] <= stop:
                    exitp = min(stop, o[j]); break
            else:
                # path after the breakout inside the trigger bar
                bullish = c[j] >= o[j]
                after_low = c[j] if bullish else l[j]
                if after_low <= stop:
                    exitp = stop; break
            continue
        if o[j] <= stop:
            exitp = o[j]; break
        if l[j] <= stop:
            exitp = stop; break
    if exitp is None:
        exitp = c[-1]
    ret = exitp / entry - 1 - COST
    return True, ret, (exitp - entry) / risk


def events_for(sym):
    df = frame(sym, "5m")
    if df is None or len(df) < 78 * 16:
        return []
    days = []
    for d, g in df.groupby("date"):
        if len(g) < 70 or g.tod.iloc[0] != 570:
            continue
        days.append((d, g))
    if len(days) < 16:
        return []
    dO = np.array([g.o.iloc[0] for _, g in days]); dH = np.array([g.h.max() for _, g in days])
    dL = np.array([g.l.min() for _, g in days]); dC = np.array([g.c.iloc[-1] for _, g in days])
    dV = np.array([g.v.sum() for _, g in days]); fV = np.array([g.v.iloc[0] for _, g in days])
    tr = np.maximum(dH[1:] - dL[1:], np.maximum(abs(dH[1:] - dC[:-1]), abs(dL[1:] - dC[:-1])))
    out = []
    for i in range(15, len(days)):
        d, g = days[i]
        atr = tr[i - 15:i - 1].mean()          # 14 true ranges ending yesterday
        avgv = dV[i - 14:i].mean()
        avgf = fV[i - 14:i].mean()
        if avgf <= 0:
            continue
        o, h, l, c, tod = (g[k].to_numpy() for k in ("o", "h", "l", "c", "tod"))
        rv = fV[i] / avgf
        i945 = int(np.searchsorted(tod, 585)); i1555 = int(np.searchsorted(tod, 955))
        r_rest = (o[i1555] / o[i945] - 1) if (i945 < len(o) and i1555 < len(o) and tod[i945] == 585 and tod[i1555] == 955) else np.nan
        r15 = c[i945 - 1] / o[0] - 1 if i945 >= 3 else np.nan
        hi20 = dH[max(0, i - 20):i].max()
        ev = {"sym": sym, "date": d, "rv": rv, "open": o[0], "atr": atr, "avgv": avgv, "r_rest": r_rest, "r15": r15,
              "prev_day": dC[i - 1] / dO[i - 1] - 1, "ret5": dC[i - 1] / dC[i - 6] - 1, "dist20": dC[i - 1] / hi20 - 1,
              "dvol": avgv * dC[i - 1], "atr_pct": atr / dC[i - 1],
              "gap": o[0] / dC[i - 1] - 1, "first_ret": c[0] / o[0] - 1, "or_atr": (h[0] - l[0]) / atr,
              "bull": c[0] > o[0], "eligible": bool(o[0] > 5 and avgv >= 1e6 and atr >= 0.5 and rv >= 1.0)}
        if ev["eligible"] and ev["bull"]:
            for e in ENTRIES:
                for s in STOPS:
                    ok, r, R = sim_day(o, h, l, c, tod, e, s, atr)
                    ev[f"{e}_{s}_in"] = ok; ev[f"{e}_{s}_ret"] = r if ok else 0.0; ev[f"{e}_{s}_R"] = R if ok else 0.0
            ev["eod_ret_from_935"] = o[-1] / o[1] - 1 if len(o) > 1 else 0.0   # 15:55 open vs 9:35 open
        out.append(ev)
    return out


if __name__ == "__main__":
    syms = [p.stem for p in Path("trader/data/5m").glob("*.npz")]
    etfs = {"SPY", "QQQ", "IWM", "SMH", "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "TNA", "TZA", "FAS", "FAZ", "^VIX"}
    syms = [s for s in syms if s not in etfs]
    rows = []
    for k, s in enumerate(syms):
        try:
            rows += events_for(s)
        except Exception as e:
            print("skip", s, e, file=sys.stderr)
        if k % 300 == 0:
            print("symbols", k, "events", len(rows), flush=True)
    E = pd.DataFrame(rows)
    E.to_pickle("trader/data/sip_events.pkl")
    print("events", len(E), "eligible bullish", int((E.eligible & E.bull).sum()), "days", E.date.nunique())
