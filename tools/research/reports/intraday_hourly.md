# Daytime strategies on ~730 sessions of hourly bars (agent-intraday-hourly, Nov 19 2026)

**Bottom line.** None of the four requested daytime ideas, (a) to (d), clears the hurdles on ~710 hourly sessions,
whether traded every day or only on the days the overnight model is in cash. One related rule came out of the gap
work and is worth a forward test:

> **Cash-day gap-down reversal.** On days when the overnight model is in cash (index below its 200-day average),
> buy the 5 liquid stocks with the largest overnight gap-downs at the open and sell them at the close.

Over 2017-2026 daily bars it earned **+58 bp per cash day** net of 10 bp costs.
- t-stat 3.3; the 2017-2023 holdout, never used for selection, gives t 3.1.
- Positive in all six bear episodes.
- Returns rise steadily with VIX, as the published liquidity-provision evidence predicts (Nagel 2012).
- It is not explained by market beta: alpha +53 bp per day, t 3.4.

It falls short of the strict deflated-Sharpe hurdle when every trial in the family is counted (about 0.8), and the
large-cap version earns about half as much. Today's-listing survivorship is therefore the main open risk. Verdict:
**promising, not adopted.** Exact rules are in section 6.

---

## 1. Data and method

| Item | Detail |
|---|---|
| Hourly bars | Yahoo 1h, Dec 21 2023 - Nov 18 2026 (730 sessions). Bars start 9:30, 10:30, ... 15:30 (the last bar is 30 min). Only full 7-bar sessions are traded; early-close days and sessions with missing bars are flat. |
| Stock universe | `trader/data/superset.json` (1,651 liquid US stocks/ADRs, today's listing), fetched here with `intraday_hourly/fetch.py` (3 threads, resumable). {{FETCH}} |
| Index / 3x funds | SPY, QQQ, IWM with UPRO/SPXU, TQQQ/SQQQ, TNA/TZA (hourly, cached). |
| Daily bars | SPY 1993+, QQQ 1999+, IWM 2000+ (index holdouts); the same 1,651 stocks Nov 2016+ (cached by another session), used for the 2017-2023 holdout of open-to-close stock rules. |
| Point in time | Daily inputs use days <= t-1 (14-day volume/ATR, 20-day high, 5-day return, 200-day SMA of the prior close). First-hour inputs (relative volume RV = 9:30-bar volume / its 14-day mean, first-hour return, SPY first hour) are used only for trades entered at the 10:30 open or later. |
| Costs | Stocks 10 bp round trip (20/30/50 bp sensitivities), 3x funds 5 bp. Whole shares in a $1,000 cash account: K slots of $1,000/K; names priced above the slot are skipped; loss at a stop capped at $100/trade and $200/day. |
| Statistics | Daily portfolio series (0 on flat days), plain and Newey-West t, annualized Sharpe, deflated-Sharpe probability (DSR) with the family's trial count, first/second-half means. Fitted models are scored only out of sample: walk-forward, expanding window, refit every 21 sessions, 5-session embargo. |
| Hurdles | t > 3 (or a strong published prior and t > 2), DSR > 0.9, same sign in both halves. |

**Survivorship.** Every stock universe here is today's listing (price >= $5 and volume >= 750k shares in Nov 2026).
Stocks that collapsed below $5 or were delisted in 2017-2026 are missing.
- **Gap-down (reversal) rules:** the bias flatters them. The worst gap-downs (bankruptcies, frauds, SPAC
  collapses) are exactly the names that would have kept falling and are absent.
- **Gap-up momentum rules:** the bias also flatters them, because future winners are included. It is smaller for
  same-day holds than for multi-day holds.
- **Index tests:** not affected.

---

## 2. (a) Overnight gaps: continuation vs reversal

### Stocks, hourly bars (2024-26)
{{GAPS_HOURLY}}

### Stocks, daily bars: gap buckets, 9:30 open -> close (`gaps_daily_holdout.py`)
Daily equal-weight means, raw (before costs). The holdout is 2017 to Dec 20 2023.

| Gap bucket | Holdout bp (NW t) | 2024-26 bp (NW t) |
|---|---|---|
| -8..-4% | +21.5 (2.2) | +10.8 (0.9) |
| -4..-2% | +7.7 (1.8) | +9.1 (1.4) |
| flat | -1.6 (-0.8) | +0.4 (0.1) |
| +2..4% | -6.6 (-1.5) | -9.4 (-1.6) |
| +4..8% | -21.3 (-2.5) | -22.5 (-2.1) |

Overnight gaps **reverse** intraday, monotonically and in both periods. The long side (buying gap-downs) clears 10 bp
only for the large gaps. Traded every day, the rule "buy the K largest gap-downs (<= -2%) at the open" earned:
- K=5: +10 bp/day (t 1.5) in the holdout and +2.9 bp/day (t 0.3) in 2024-26.
- At 20 bp costs: about zero.

**Rejected as an every-day rule.** The same rule restricted to cash days is section 5.

### Index via 3x funds, hourly (`index_day.py`, family A)
- 36 diagnostic cells: gap z-bucket (gap / trailing 20-day gap std) x first-hour RV x entry time.
- Continuation was negative or flat almost everywhere.
- Following large gaps (|z| > 1.5) lost money: IWM -12 bp/day, t -2.4.
- Fading them made money: three-index basket +7.0 bp/day, t 1.4, DSR 0.02; IWM alone +12.2 bp/day, t 2.1.
- RV (the news proxy) did not separate continuation from reversal.

**Rejected.** The IWM fade also fails the daily holdout (section 4).

---

## 3. (b) Stocks in play, hourly (60-minute opening range)
{{SIP}}

---

## 4. (d) Index-level daytime ideas via 3x funds (706 sessions, 5 bp)

Below-200-day days in the hourly sample: SPY 55, QQQ 64, IWM 105 (all three below on 43 days, any of them on 119).

| Idea (family) | Variants | Best basket (1/3 each) | Best single series | Verdict |
|---|---|---|---|---|
| Regime day trades, bull/bear fund 9:30 or 10:30 -> close (D1) | 32 | Below 200d, bull 9:30->close: +4.6 bp/day (+27 bp per active day), t 0.7. Above 200d, bull: -11 bp/day. | IWM below 200d, bull 10:30: t 1.2 | rejected |
| Rebound days below the 200-day: gap down / prior day down (D2) | 24 | Prior day down, bull 10:30->close: +6.8 bp/day, t 1.45, second half negative | IWM same: +10.4 bp/day, t 2.0, DSR 0.88 | rejected |
| Gap fill after large gaps, limit at the fund's prior close or exit at the close (D3) | 64 | \|gap\| > 1% fade, close: +7.3 bp/day, t 1.3 | IWM \|gap\| > 1% fade, above 200d: t 2.8, DSR 0.90 | rejected |
| Last half hour: fade the day so far from 15:30 (or 14:30) (D4) | 24 | 15:30, all days: +2.0 bp/day, t 0.8. Below 200d: +2.2 bp/day, t 1.8. The 14:30 version loses (t -1.5). | IWM below 200d: t 2.0 | rejected |
| Gap continuation vs reversal (A) | 16 + 36 cells | Fade \|z\| > 1.5: t 1.4 | IWM fade: t 2.1 | rejected |

**Holdouts on daily bars to Dec 20 2023** (`index_daily_holdout.py`, `gapfill_long.py`)

The fund return is synthetic: 3x the index from the prior close, daily reset, 5 bp.
- **IWM gap fade** (\|gap\| > 1%, exit at the close): t 2.3 in 2024-26 but **-0.6** in 2000-2023.
- **Gap-down fill** (buy the 3x bull fund at the open when the index gaps below -1%; sell at the prior close or
  the close):
  - Counting a bare touch of the prior close as a fill gives SPY t 3.5. Requiring a 5 bp trade-through gives
    t **1.4**; the three-index basket gives t 1.1.
  - The whole difference comes from 31 of 378 SPY days on which the high stopped within 5 bp of the prior close,
    the classic "filled at the day's high" illusion.
  - With realistic fills, no threshold (0.5-2%), exit, stop or regime variant reaches t 2 (64 daily + 32 hourly
    series).
- **Bull fund 9:30 -> close below the 200-day:** t 0.3 (basket, 1993-2023).

**Verdict for (d):** rejected, including on cash days. The index's own day session has no usable drift below the
200-day average.

---

## 5. (c) Cross-sectional prediction of the 10:30 -> close return
{{XSEC}}

---

## 6. Candidate: cash-day gap-down reversal (`gaps_daily_holdout.py`, `gd_regime.py`, `gd_checks.py`, `final_eval.py`)

**How it was found.** In the hourly sample, "buy the 5 largest gap-downs at the open" split by the 200-day regime
showed +87 bp per SPY-below-200d day (t 2.0, 55 days). The same split on the untouched 2017-2023 daily holdout
(4 K-variants, all positive) gave **K=5: +59 bp per cash day, t 3.1**. The 36-cell grid that followed is counted
as trials:
- universe {base, large caps}
- gap band {<= -2%, -10..-2%, <= -4%}
- K {3, 5, 10}
- regime {SPY < 200d, any index < 200d}

**Prior.** Returns to liquidity provision (short-term reversal) are much higher when markets are turbulent (Nagel
2012, *RFS*, "Evaporating liquidity"). Overnight returns partially reverse during the next trading session (Berkman,
Koch, Tuttle & Zhang 2012; Lou, Polk & Skouras 2019).

### Results

Pre-specified cell: base universe, gap <= -2%, K = 5, SPY < 200d, 10 bp; 2,492 days, 438 cash days.

| Measure | Value |
|---|---|
| Mean per cash day | +58.4 bp (median +20.6, 2%-trimmed +46.6, hit rate 53%, top-5 days = 31% of profit) |
| t / Newey-West t (full daily series) | 3.27 / 3.30; Sharpe 1.04 |
| Holdout 2017-2023 / 2024-26 | +59.2 bp per cash day (t 3.07) / +53.4 bp (t 1.1, 55 days) |
| By episode (bp per cash day) | 2018-19 +41 (80 d), 2020 +51 (62 d), 2022 +61 (212 d), 2023 +121 (27 d), 2025 +57 (42 d), 2026 +42 (13 d) |
| All 36 grid cells at 10 bp | positive on cash days |
| Large caps (price >= $10, $ADV >= $50M) | K=3 +47 bp (t 2.4); K=5 +30 bp (t 1.9). About half the base effect: survivorship and/or size. |
| Any index below 200d (822 days) | +43.7 bp per day, t 3.8 |
| Costs 10 / 20 / 30 / 50 bp | +58 / +50 / +41 / +24 bp per cash day (NW t 3.3 / 2.9 / 2.4 / 1.4) |
| Risk ($1,000, K=5) | worst day -11.9%, worst position -$53, no position worse than -$100, no day worse than -$200 |
| 8% protective stop (stop-first on the daily low) | +46 bp per cash day, t 3.0, worst day -7.8% |
| **Capital actually freed by the overnight model** ($1,000 x legs in cash / 3) | **+11.1 bp/day on the whole sleeve over all 2,492 days, t 3.6, Sharpe 1.15**; holdout t 3.2; 2024-26 +6.4 bp/day (t 1.8) |
| VIX mechanism (every-day rule, prior VIX close) | VIX <= 15: -14 bp; 15-20: 0; 20-25: +23; 25-30: +42; > 30: +79 bp/day. Slope +3.7 bp per VIX point (NW t 3.3). Below-200d days beat above-200d days within each VIX bucket. |
| Attribution on cash days (gross) | Picks beat the universe average by +54.5 bp (NW t 3.5). Beta to SPY open->close 1.26, alpha +53 bp (t 3.4). Placebos: largest gap-UPs -36 bp (t -2.6); random flat-open stocks +1 bp (t 0.3). |
| ADRs | Excluding the 83 ADRs: +57.5 bp (t 3.3, holdout t 3.0). ADRs only: +10 bp. Not an ADR artifact. |
| Picks | median price $23, median gap -6.2%, median $ADV $110M (8% below $20M). Return rises with gap size: -2..-4% +10 bp, -4..-8% +57 bp, -8..-15% +113 bp, beyond -15% +116 bp. |
| Timing (hourly, 2024-26 cash days) | {{GD_HOURLY}} |

### Hurdle check (`final_eval.py`)
{{FINAL_EVAL}}

### Exact rules for a forward test (not adopted; for the main session to decide)
1. **When.** Trade only on day t when at least one leg of the overnight model was in cash the night before, i.e. the
   index's close on t-1 was below its 200-day SMA (SPY->UPRO, QQQ->TQQQ, IWM->TNA, as in `tools/overnight.py`).
   Capital C = settled cash the overnight model did not commit (backtest: $1,000 x legs in cash / 3).
2. **Universe** (computed before the open from daily bars through t-1): stocks in `trader/data/superset.json` with
   prior close >= $5 and 14-day average daily volume >= 1,000,000 shares.
3. **Signal** at 9:30-9:31: gap = today's opening price / prior close - 1. Candidates have gap <= -2%. Rank by
   gap, most negative first. Skip any name priced above C/5. Take the first 5.
4. **Entry.** For each, buy floor((C/5) / price) shares with a market order (or a marketable limit at ask + 0.3%)
   in the first minutes after the open.
5. **Protection.** A day stop-market order 8% below each fill. This keeps the loss per $200 position near $16-20,
   well inside $100/trade and $200/day.
6. **Exit.** Sell everything with market orders at 15:55. If the overnight model re-enters that evening, its 15:58
   purchase is made after these sales. Check the cash-account good-faith rules for buying with same-day proceeds.
7. **Record** every fill against the 9:30 opening print and the close. The forward test mainly measures real
   costs; the backtest break-even is about 50 bp. Detecting the edge itself statistically would take about 160
   cash days (daily standard deviation about 3.7%).

---

## 7. Registry entries (author agent-intraday-hourly)
{{REGISTRY}}

## 8. Open questions for the next iteration
1. **Survivorship.** The cash-day gap-down result needs a point-in-time universe that includes delisted names
   (CRSP-style or a vendor list of 2017-2023 delistings). The large-cap version (about half the effect) is the
   conservative estimate until then.
2. **Entry precision.** Use 1-minute or 5-minute bars on bear-market days (Mar-Apr 2025, Mar-Apr 2026, not in the
   current 60-session 5-minute cache) to measure the cost of entering at 9:31-9:35 instead of the opening print.
3. **Capital outside the cash regime.** The overnight model also leaves cash unused when volatility targeting holds
   exposure below 3x. A smaller, VIX-scaled version of the rule (VIX > 20 alone: +39 bp/day, t 3.0) could use it.
   This needs a pre-registered test, not a new grid.
4. **Short-side mirror.** Gap-ups underperform on cash days (-36 bp, t -2.6), but long-only rules cannot use that
   (no shorting; inverse funds are index-level).
5. **Untouched hourly ideas.** Index intraday momentum with VIX-scaled thresholds and stocks-in-play with news
   tags (8-K or earnings) instead of RV.
