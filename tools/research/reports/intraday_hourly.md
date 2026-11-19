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

It falls short of the strict deflated-Sharpe hurdle when all 122 trials in the family are counted (0.87 for the
realistic version), and the large-cap version earns about half as much. Today's-listing survivorship is therefore the main open risk. Verdict:
**promising, not adopted.** Exact rules are in section 6.

---

## 1. Data and method

| Item | Detail |
|---|---|
| Hourly bars | Yahoo 1h, Dec 21 2023 - Nov 18 2026 (730 sessions). Bars start 9:30, 10:30, ... 15:30 (the last bar is 30 min). Only full 7-bar sessions are traded; early-close days and sessions with missing bars are flat. |
| Stock universe | `trader/data/superset.json` (1,651 liquid US stocks/ADRs, today's listing), fetched here with `intraday_hourly/fetch.py` (3 threads, resumable). Result: 1,617 of 1,651 fetched. 34 failed with HTTP 422; these were listed less than 730 days ago (e.g. GEV, RDDT, NBIS). The panel keeps the 1,610 stocks with at least 500 hourly bars. |
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
`gaps.py`: 1,610 stocks, 710 sessions; universe = prior close >= $5 and 14-day volume >= 1M shares.

- **35 cells** (gap bucket x first-hour RV x entry time). No cell's excess return over the universe reaches
  \|NW t\| 2.5.
  - At the 9:30 entry the daily-bar reversal shows again: gap-downs of -4..-2% earn +10.3 bp excess (t 2.0);
    gap-ups of +2..4% earn -9.8 bp (t -2.1) and gap-ups beyond +4% earn -19.1 bp (t -1.8).
  - **RV does not separate news from non-news gaps.** Gap-downs beyond -4% with RV >= 4 earn +19 bp (t 1.9);
    gap-ups beyond +4% with RV >= 4 earn +8.7 bp (t 0.9).
- **20 long-only top-K rules.**
  - Best: buy the 10 largest gap-downs (<= -2%) at the 9:30 open: +17.6 bp/day, t 2.0, Sharpe 1.21, DSR 0.45,
    halves +12 / +23 bp.
  - Every 10:30-entry rule is at or below +1.5 bp/day on all days. Gap-downs bought at 10:30 (K=5): -3.9 bp,
    t -0.5. News gap-ups (gap >= 2%, RV >= 2, K=5): -7.0 bp, t -1.2.
  - On SPY-below-200d days (55) the 10:30 gap-down rules were positive (+44 to +59 bp per day, t 1.4-2.0),
    consistent with section 6.

**Verdict:** rejected as every-day rules.

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

**Rejected as an every-day rule.** The same rule restricted to cash days is section 6.

### Index via 3x funds, hourly (`index_day.py`, family A)
- 36 diagnostic cells: gap z-bucket (gap / trailing 20-day gap std) x first-hour RV x entry time.
- Continuation was negative or flat almost everywhere.
- Following large gaps (|z| > 1.5) lost money: IWM -12 bp/day, t -2.4.
- Fading them made money: three-index basket +7.0 bp/day, t 1.4, DSR 0.02; IWM alone +12.2 bp/day, t 2.1.
- RV (the news proxy) did not separate continuation from reversal.

**Rejected.** The IWM fade also fails the daily holdout (section 4).

---

## 3. (b) Stocks in play, hourly (60-minute opening range)
`sip_hourly.py`: 1,610 stocks, 710 sessions, median 122 eligible names per day.

- **16 variants:** K {1, 3, 5, 10} x entry {10:30 open, buy-stop one cent above the first-hour high} x stop
  {first-hour low, none}.
- **Best:** K=1, 10:30 open, no stop: +5.6 bp/day, t 0.37, Sharpe 0.22, DSR 0.03 (56 family trials, including the
  40 of the 5-minute study); halves -6 / +17 bp.
- Every K >= 3 variant is negative. Buy-stop breakouts lose significantly (K=10: -11.8 bp/day, t -3.3).
- **RV control** (per trade, 10:30 open, no stop, net 10 bp): RV 1-1.5 -9.4 bp; 1.5-2 -6.6; 2-3 -5.5; 3-5 -1.5;
  5+ +0.7. Higher RV helps but never clears costs.
- **SPY-below-200d days:** positive but insignificant (K=1 +39 bp, t 0.7; K=5 +27 bp, t 0.9).

**Verdict:** rejected, now on 710 sessions instead of 45.

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
`xsec_hourly.py`: 1,610 stocks, 662,205 stock-days, median 950 names per day.

**Fama-MacBeth** (all 710 days; slope = top-minus-bottom spread in bp; 24 tests):
- **Main slopes:** no feature reaches \|NW t\| 2.
  - log RV +2.3 bp (t 1.9); ATR% +5.0 (t 1.1); gap +1.4 (t 0.5); first hour -1.7 (t -0.9).
  - prior-day return +1.4 (t 0.5); 5-day return -1.5 (t -0.6); distance to the 20-day high +0.3 (t 0.1);
    dollar volume -0.2 (t -0.1).
- **Interactions with SPY's first hour:** all \|t\| <= 1.5.
- **Interactions with the prior VIX close:** gap -2.7, ATR% +2.3, distance to the 20-day high -2.2. These point the
  same way as the reversal-in-turbulence mechanism of section 6, but none reaches 3.

**Walk-forward models** (out of sample only: 460 days, Jan 22 2025 - Nov 18 2026; expanding window, refit every 21
sessions, 5-session embargo):

| Model | Rank IC (NW t) | Halves | Decile spread, gross (not long-only) | Top decile long, net 10 bp |
|---|---|---|---|---|
| LightGBM (ranks + raw + SPY first hour/gap) | +0.018 (2.4) | +0.025 / +0.012 | +7.0 bp (t 1.3) | -4.1 bp (t -0.9) |
| LightGBM + VIX + 200-day regime | +0.024 (3.2) | +0.027 / +0.021 | +9.3 bp (t 1.6) | -3.3 bp (t -0.7) |
| Linear (rolling Fama-MacBeth composite) | -0.005 (-0.6) | - | -6.3 bp | -12.9 bp (t -2.5) |

**Long-only top-K** (whole shares, $1,000, 10 bp; 12 variants):
- Best: LightGBM top-1 +1.9 bp/day (t 0.2, DSR 0.08).
- Every LightGBM portfolio lies between -7.1 and +1.9 bp/day; every linear one loses (-14.7 to -32.4 bp).
- On SPY-below-200d days (n=55) the LightGBM portfolios were positive (+13 to +71 bp, not significant).

**Verdict:** rejected for trading.
- The VIX/regime model's ranking skill is statistically real (IC t 3.2), but it lives in the long-short spread
  (about 9 bp a day gross).
- The long side does not clear costs, and short sales are not allowed.

---

## 6. Candidate: cash-day gap-down reversal (`gaps_daily_holdout.py`, `gd_regime.py`, `gd_checks.py`, `final_eval.py`)

**How it was found.** In the hourly sample, "buy the 5 largest gap-downs at the open" split by the 200-day regime
showed +87 bp per SPY-below-200d day (t 2.0, 55 days) on the first ~550 stocks downloaded. On the full 1,610-stock
universe it is +46 bp (t 0.9). The same split on the untouched 2017-2023 daily holdout
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
| Timing (hourly, 2024-26 cash days) | Same picks chosen at the open; only the entry time changes (equal-weight top 5, net 10 bp; `entry_timing.py`).<br>**Hourly, SPY-below-200d days (n=55):** 9:30 open +52.8 bp/day (t 0.95); 10:30 open +44.9 (t 1.3), so about 85% survives. The picks moved only +7.6 bp from 9:30 to 10:30 and +55 bp from 10:30 to the close.<br>**Any leg in cash (n=120):** +43.0 vs +30.7 (about 71%). Other days (n=582): +1.6 vs -15.3.<br>**5-minute bars** (46 sessions, Sep 16 - Nov 18 2026): no SPY-below-200d day. On 18 days with some leg in cash, entries at 9:30 / 9:35 / 9:45 / 10:00 / 10:30 gave +37 / +53 / +20 / +31 / +54 bp/day, all t < 0.8: too few days to measure the decay. |

### Hurdle check (`final_eval.py`)
Strict version: DSR uses all 122 trials this agent logged in the `gaps_intraday` family. The null Sharpe
variance is 252/n; the empirical variance across the 36 regime cells is shown alongside.

| | SPY < 200d, full $1,000 | Capital freed by the overnight model (thirds) |
|---|---|---|
| Days / active days | 2,492 / 433 | 2,492 / 812 |
| Mean | +10.3 bp/day (+59 per cash day) | +11.1 bp/day (+34 per active day) |
| t / NW t | 3.27 / 3.30 | **3.60 / 3.70** |
| Sharpe (annualized, all days) | 1.04 | 1.15 |
| DSR, 122 trials, null variance 0.101 | 0.77 | **0.87** |
| DSR, empirical variance 0.052 | 0.94 | 0.97 |
| Halves (active days, chronological) | +75 / +43 bp | +35 / +33 bp |
| Holdout 2017-2023 / 2024-26 (NW t) | 3.07 / 1.11 | 3.24 / 1.77 |
| Whole-sleeve return / max drawdown | +25.7%/yr / -30.9% (Aug 16 - Oct 11 2022) | +28.5%/yr / -30.4% |
| By year | 2018 +9%, 2019 +25%, 2020 +28%, 2022 +211%, 2023 +37%, 2025 +23%, 2026 +5% | 2017 -1%, 2018 +3%, 2019 +24%, 2020 +36%, 2021 +12%, 2022 +191%, 2023 +40%, 2025 +35%, 2026 +13% |

Costs at 20 / 30 bp (SPY version): t 2.79 / 2.31, DSR 0.58 / 0.38 (null) and 0.84 / 0.68 (empirical).

**Hurdles:**
- **t:** passes. t > 3 at 10 bp, with a strong published prior on top.
- **Halves:** passes; same sign in both, and holdout-confirmed.
- **DSR > 0.9:** fails under the strict family count with the null variance (0.77-0.87). It passes with the
  empirical variance, or when counting only the 4 variants carried into the untouched holdout (about 0.99).
- **Concentration:** 2022 (212 cash days) supplies most of the compounded gain.

**Verdict: promising, not adopted.**

### Exact rules for a forward test (not adopted; for the main session to decide)
1. **When.** Trade only on day t when at least one leg of the overnight model was in cash the night before, i.e. the
   index's close on t-1 was below its 200-day SMA (SPY->UPRO, QQQ->TQQQ, IWM->TNA, as in `tools/overnight.py`).
   Capital C = settled cash the overnight model did not commit (backtest: $1,000 x legs in cash / 3).
2. **Universe** (computed before the open from daily bars through t-1): stocks in `trader/data/superset.json` with
   prior close >= $5 and 14-day average daily volume >= 1,000,000 shares.
3. **Signal** after the open: gap = today's official opening price / prior close - 1. Candidates have gap <= -2%. Rank by
   gap, most negative first. Skip any name priced above C/5. Take the first 5.
4. **Entry at 9:45**, the earliest the sleeve's rules allow (`no_entries_before`). For each name, buy
   floor((C/5) / price) shares with a marketable limit at ask + 0.3%. The picks are still chosen by the gap at the
   open.
   - Evidence for the delay: on 2024-26 cash days a 10:30 entry kept about 85% (SPY in cash, n=55) and about 71%
     (any leg in cash, n=120) of the opening-print return.
   - A 9:45 entry should sit between those two points. Plan on roughly +40 to +50 bp per cash day before
     survivorship, against +58 bp at the opening print.
   - The 5-minute data has too few cash days (18) to measure 9:45 directly.
5. **Protection.** A day stop-market order 8% below each fill. This keeps the loss per $200 position near $16-20,
   well inside $100/trade and $200/day.
6. **Exit.** Sell everything with market orders at 15:55. If the overnight model re-enters that evening, its 15:58
   purchase is made after these sales. Check the cash-account good-faith rules for buying with same-day proceeds.
7. **Record** every fill against the 9:30 opening print and the close. The forward test mainly measures real
   costs; the backtest break-even is about 50 bp. Detecting the edge itself statistically would take about 160
   cash days (daily standard deviation about 3.7%).

---

## 7. Registry entries (author agent-intraday-hourly)
| Family | Experiment | Variants | Verdict |
|---|---|---|---|
| index_daytime_hourly | Index gap -> same-day continuation vs reversal (z-buckets x first-hour RV), 3x funds | 52 | rejected |
| index_daytime_hourly | Index day trades by 200-day regime (bull/bear 3x fund, 9:30 or 10:30 -> close) | 32 | rejected |
| index_daytime_hourly | Rebound days below the 200-day (gap down / prior day down) via 3x bull funds | 24 | rejected |
| index_daytime_hourly | Index gap fill after large overnight gaps via 3x funds (fill-or-close exits) | 64 | rejected |
| index_daytime_hourly | Last-half-hour / 90-minute reversal of the day-so-far via 3x funds, by regime | 24 | rejected |
| index_daytime_hourly | Holdout (2017-2023 daily bars) of the best index day rules | 24 | rejected |
| index_daytime_hourly | Index gap-down fill (buy 3x bull at open, sell at prior close or close), long history | 64 | rejected |
| gaps_intraday | Holdout 2017-2023 (daily bars): buy the largest stock gap-downs at the open, sell at the close | 26 | rejected |
| gaps_intraday | Gap-down reversal only when SPY < 200-day (overnight model in cash), daily bars 2017-2026 | 36 | promising |
| stocks_in_play | Hourly stocks-in-play (60-min range), long only, top-K by RV, whole shares | 16 | rejected |
| gaps_intraday | Stock gaps x first-hour RV -> same-day continuation/reversal (hourly, 710 days) | 55 | rejected |
| gaps_intraday | Cash-day gap-down reversal: execution, risk, mechanism and attribution checks | 5 | promising |
| gaps_intraday | Cash-day gap-down reversal: entry-time sensitivity (hourly 9:30 vs 10:30; 5-min 9:30-10:30) | 0 | promising |
| xsec_intraday | Fama-MacBeth: 10:30 features -> 10:30-close return (hourly, 710 days) | 24 | rejected |
| xsec_intraday | Walk-forward LightGBM + linear FM composite, long-only top-K 10:30->close | 12 | rejected |

Note: the record named "Holdout (2017-2023 daily bars)..." actually used daily bars from 1993 (SPY), 1999 (QQQ) and 2000 (IWM) to Dec 20 2023.

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
