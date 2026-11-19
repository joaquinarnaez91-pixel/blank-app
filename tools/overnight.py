"""Overnight index model (live). Decides at ~15:55 ET what to hold from the close to the next 9:30 open.

  python3 tools/overnight.py signal --capital 970 [--prices SPY=730.1,QQQ=659.5,IWM=270.1,UPRO=..,TQQQ=..,TNA=..]
      -> prints the decision and writes trader/overnight_plan.json (nothing is placed)

Spec (adopted Nov 19, 2026; research: tools/research/README.md, registry family "overnight"):
  legs: SPY->UPRO, QQQ->TQQQ, IWM->TNA, one third of the capital each.
  hold a leg only if the index's live price is above its 200-day simple moving average (today's price counts
  as today's close), exposure L = min(3, 0.015 / sigma20) where sigma20 is the sample std of the last 20 overnight
  returns (open_t / close_t-1 - 1, today's open included); dollars in the 3x fund = capital/3 * L/3, whole shares.
  stress check: every held index gaps down by max(5 x sigma20, 2.5%) at once; the loss must stay <= $100,
  otherwise all legs are scaled down. Sell everything at the next 9:30 open (market order queued after 16:00).
"""
import sys, json, math, argparse, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "research"))
import numpy as np
import yfetch
from bars import frame, NY

LEGS = [("SPY", "UPRO"), ("QQQ", "TQQQ"), ("IWM", "TNA")]
TARGET, CAP, MAX_STRESS_LOSS = 0.015, 3.0, 100.0


def live_price(sym):
    doc = yfetch.get("/v8/finance/chart/%s?range=1d&interval=1m" % sym)
    return float(doc["chart"]["result"][0]["meta"]["regularMarketPrice"])


def leg_signal(index, price, today):
    d = frame(index, "1d")
    d = d[~d.index.duplicated()]
    closes = d.c.copy()
    opens = d.o.copy()
    if d.date.iloc[-1] == today:          # today's partial bar: its open is known, its close is the live price
        closes.iloc[-1] = price
    else:
        raise SystemExit(f"no daily bar for {index} today ({today}); last is {d.date.iloc[-1]}. Is the market open?")
    sma = closes.iloc[-200:].mean()
    night = (opens / closes.shift(1) - 1).iloc[-20:]
    sigma = float(night.std(ddof=1))
    above = price > sma
    L = min(CAP, TARGET / sigma) if above else 0.0
    return {"index": index, "price": price, "sma200": round(float(sma), 2), "above": bool(above),
            "pct_vs_sma": round(100 * (price / sma - 1), 2), "sigma20": round(sigma, 5), "exposure": round(L, 3)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signal")
    s.add_argument("--capital", type=float, required=True, help="settled cash the sleeve may use tonight")
    s.add_argument("--prices", default="", help="SYM=price,... (Robinhood quotes; fund prices = ask)")
    s.add_argument("--no-fetch", action="store_true")
    s.add_argument("--test-last-day", action="store_true", help="dry run: treat the last cached day as today")
    a = ap.parse_args()
    today = datetime.datetime.now(NY).date()
    if a.test_last_day:
        today = frame("SPY", "1d").date.iloc[-1]
    px = dict((k, float(v)) for k, v in (p.split("=") for p in a.prices.split(",") if p))
    if not a.no_fetch:
        for idx, fund in LEGS:
            yfetch.save(idx, "1d", yfetch.fetch_bars(idx, "1d", "1mo"))
    legs, stress = [], 0.0
    for idx, fund in LEGS:
        p = px.get(idx) or live_price(idx)
        sig = leg_signal(idx, p, today)
        fp = px.get(fund) or live_price(fund)
        dollars = a.capital / 3 * sig["exposure"] / 3
        shares = int(dollars // fp)
        gap = max(5 * sig["sigma20"], 0.025)
        loss = shares * fp * 3 * gap
        stress += loss
        legs.append({**sig, "fund": fund, "fund_price": fp, "dollars_target": round(dollars, 2), "shares": shares,
                     "stress_gap": round(gap, 4), "stress_loss": round(loss, 2)})
    scale = 1.0
    if stress > MAX_STRESS_LOSS:
        scale = MAX_STRESS_LOSS / stress
        for leg in legs:
            leg["shares"] = int(leg["shares"] * scale)
            leg["stress_loss"] = round(leg["shares"] * leg["fund_price"] * 3 * leg["stress_gap"], 2)
    total = sum(l["shares"] * l["fund_price"] for l in legs)
    plan = {"date": today.isoformat(), "made_at": datetime.datetime.now(NY).isoformat(timespec="seconds"),
            "capital": a.capital, "legs": legs, "cost": round(total, 2),
            "stress_loss": round(sum(l["stress_loss"] for l in legs), 2), "scaled": scale < 1.0,
            "verdict": "PASS" if sum(l["stress_loss"] for l in legs) <= MAX_STRESS_LOSS and total <= a.capital + 0.01 else "BLOCK"}
    Path("trader").mkdir(exist_ok=True)
    json.dump(plan, open("trader/overnight_plan.json", "w"), indent=1)
    for l in legs:
        state = f"HOLD {l['shares']} {l['fund']} @ ~{l['fund_price']:.2f} (${l['shares']*l['fund_price']:.0f}, {l['exposure']:.2f}x)" if l["shares"] else "cash"
        print(f"{l['index']}: {l['price']:.2f} vs 200-day {l['sma200']:.2f} ({l['pct_vs_sma']:+.2f}%), "
              f"overnight vol {100*l['sigma20']:.2f}% -> {state}; stress loss ${l['stress_loss']:.0f}")
    print(f"Total ${plan['cost']:.2f} of ${a.capital:.2f}; stress loss ${plan['stress_loss']:.2f} (limit $100){' after scaling' if scale < 1 else ''} -> {plan['verdict']}")
    print("Wrote trader/overnight_plan.json. Nothing was placed.")


if __name__ == "__main__":
    main()
