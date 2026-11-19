# Overnight model: validation and improvement program (agent-overnight, Nov 19 2026)

## Verdict in one paragraph

The overnight premium behind the live model is real and robust (out-of-sample Sharpe 0.73, 95% CI 0.32 to 1.13,
2008-2026; the spec's own 96-variant parameter family beats cash with SPA p < 0.001; probability of backtest
overfitting 0.12). **No regime signal, calendar rule, volatility model or Kelly rule beats the 200-day / sigma20
spec after counting the 81 variants tried** (Hansen SPA p = 0.27). Two findings change the live model anyway:
(1) the spec as written **breaches the $100 per-trade limit**: in the longer history (2001-2026) its worst night is
2016-06-24 (Brexit), -10.8% of the sleeve (-$108; -$110 with real funds and whole shares), because calm markets
push every leg to the 3x cap; (2) the backtest ignored the **financing carry of 3x funds, which the overnight holder
pays** (about 2.4 x the T-bill rate a year on the fund dollars: at today's 4.1%, ~2.7 bp on a normal night and ~8 bp
on Thursday nights, which carry the weekend). **Recommendation: keep the spec, but cap each leg at L = 2.0 (not 3.0) and add a pre-trade stress guard.**
That raises the out-of-sample Sharpe slightly (0.78 vs 0.73, both halves higher, p = 0.07), cuts the worst night to
-7.8% and max drawdown to -24%, and costs about 2 points of CAGR (10.4% vs 12.3% a year, 2008-2026). Adding a
SOXL leg (SMH) is the only promising return improvement; it is not adopted yet (details in (e)).

## Data and method

- Yahoo daily bars, full history: SPY (1993-), QQQ (1999-), IWM (2000-05), SMH, XLF, HYG, IEF, ^VIX, ^VIX3M,
  ^IRX (T-bill), the real 3x funds (UPRO, TQQQ, TNA, SOXL, FAS; 2008/2010-), and cash dividends.
  `overnight/fetch.py` (period1/period2 requests, 3 threads).
- **Data incident (fixed):** `range=max` with `interval=1d` silently returns MONTHLY bars from Yahoo, and
  `yfetch.save` merged ~400 monthly bars per file into `trader/data/1d/` (SPY, QQQ, IWM, SMH, XLF, ^VIX, ^VIX9D and
  the 3x funds) for about 5 minutes (06:00-06:05 UTC). `overnight/repair_cache.py` removed them; every file was
  re-verified at the end (no monthly bars, no duplicate dates). Any script that read these files in that window
  should be re-run. Suggest a guard in `yfetch.py` (refuse `range=max` for 1d, or check `meta.dataGranularity`).
- **Night t** = close of day t-1 to open of day t. Every signal for night t uses data up to the close of day t-1
  (stand-in for the 15:58 price) and nothing later. Fitted models are refit each Jan 1 on earlier data only (HMM
  filtered, never smoothed). Windows: **out of sample 2008-01 to 2026-11 (4,751 nights)**, halves split at mid-2017;
  pre-sample check 2003-2007; the spec's own window Sept 2017-Nov 2026 (2,312 nights).
- **Realistic returns** (`core.py`): index overnight total return r = (open + dividend on ex-date) / prior close - 1;
  3x fund overnight return = 3r - carry; 4 bp round trip on fund dollars; sleeve return = sum over legs of
  (weight x L / 3) x fund return. All Sharpe ratios, t-stats and bootstraps are on the daily sleeve series.
- Statistics (`validate.py`, numpy only): Politis-Romano stationary bootstrap; White Reality Check and Hansen SPA
  (lower/consistent/upper, vol-matched to the spec so it compares Sharpe); CSCV probability of backtest overfitting
  (Bailey, Borwein, Lopez de Prado & Zhu 2017, S = 16, 12,870 splits); deflated Sharpe with 210 trials
  (33 earlier + 81 here + 96 grid); paired bootstrap of the Sharpe difference vs the spec (b = 20).

## Corrections to the cost model (tracking.py)

Actual 3x fund overnight return minus 3 x the ETF's (2009/10-2026, per-night regression on the T-bill rate):

| Fund vs 3x ETF | slope on T-bill (per year) | intercept | mean, bp/night |
|---|---|---|---|
| UPRO vs SPY | -2.22 (se 0.34) | ~0 | -1.1 |
| TQQQ vs QQQ | -2.40 (se 0.28) | ~0 | -1.1 |
| TNA vs IWM | -2.63 (se 0.38) | ~0 | -1.4 |

- The overnight holder pays the fund's leverage financing (theory: 2 x T-bill + spread). It is charged per trading
  night, with the weekend's accrual on the Thursday -> Friday night (Friday mornings show ~2.5x the drag at high
  rates). Model used: carry = 2.4 x T-bill x (days to the next trading date) / 365. At 4.1% that is ~2.7 bp on a
  normal night and ~8 bp on Thursday nights per dollar in the fund, ~4.8% a year of the sleeve at the recommended
  average exposure.
- The 3x fund does **not** drop on the ETF's ex-dividend night (UPRO vs raw SPY: +128 bp on those nights), so
  backtests on raw ETF opens understate by 3 x the dividend four times a year.
- Real-fund check (`exp_realfunds.py`, actual UPRO/TQQQ/TNA/SOXL/FAS opens and closes, 2010-2026): the spec's real
  Sharpe 0.80 vs 0.77 for the proxy on the same nights. The proxy is slightly conservative; conclusions unchanged.

Current spec re-measured (realistic costs): Sept 2017-Nov 2026 SR 0.89, CAGR +16.3%, maxDD -28.5%, worst night -8.4%
(main session's convention, raw opens and no carry, reproduces SR 0.99 here vs the reported 0.92); 2008-2026 SR 0.73,
CAGR +12.3%, maxDD -28.8%; 2001-2026 SR 0.75, CAGR +12.1%, **worst night -10.8% (2016-06-24)**.

## (a) Validation of the current spec (exp_validate.py)

| Window | Sharpe [95% CI], b = 20 | CAGR [95% CI] | P(SR <= 0) |
|---|---|---|---|
| Spec window 2017-09..2026-11 | 0.89 [0.26, 1.49] | +16.3% [+3.1%, +30.6%] | 0.003 |
| Out of sample 2008-2026 | 0.73 [0.32, 1.13] | +12.3% [+4.3%, +21.1%] | < 0.001 |
| 2001-2026 | 0.75 [0.39, 1.11] | +12.1% [+5.4%, +19.1%] | < 0.001 |

CIs barely move with block length 5 or 60.
- **CSCV on the spec's own family** (SMA 100/150/200/250 x sigma 10/20/40/60 x target 1.0/1.5/2.0% x cap 2/3, 96
  variants, 2001-2026): PBO 0.12, the in-sample best keeps OOS Sharpe 0.81 (from 0.88). The live parameters rank
  38/96 (SR 0.75). Grid averages: cap 2 0.74 > cap 3 0.71; SMA 250 0.82 > 200 0.76 > 150 0.70 > 100 0.61; target
  2.0% 0.75 > 1.5% 0.73 > 1.0% 0.70; sigma window flat. Not used to re-tune (in sample).
- **SPA / Reality Check, H0 "no variant beats the spec"** (81 program variants, vol-matched): best = SOXL leg cap 2,
  T = 2.52, SPA p (consistent) 0.27, White RC p 0.64: **not rejected**. Sub-families: signals p 0.84, calendar 0.50,
  sizing 0.45, asset set 0.026 (RC 0.011). Spec family vs cash: p < 0.001.
- **CSCV across all 82 program variants** (2008-2026): PBO 0.40; picking the in-sample best yields OOS Sharpe 0.73,
  the same as the spec. Selecting among these variants does not pay.
- **Deflated Sharpe** (210 trials): spec 0.984; cap-2 0.991; SOXL leg cap 2 0.998; TSMOM 0.993 (all pass > 0.9;
  DSR tests SR > 0, not the improvement).

## (b) Regime signals vs the 200-day filter (exp_signals.py), out of sample 2008-2026

Same sizing as the spec; "replace" swaps the 200-day filter; "and" requires both. dSR = Sharpe minus spec's, paired
bootstrap; p = P(dSR <= 0). Spec: SR 0.73 (0.53 / 0.91), CAGR +12.3%, worst -10.8%.

| Variant | SR (h1 / h2) | CAGR | maxDD | worst | dSR [95% CI] | p |
|---|---|---|---|---|---|---|
| TSMOM 21/63/126/252 majority, replace | 0.80 (0.54/1.05) | +12.8% | -29.5% | -8.4% | +0.07 [-0.07, +0.22] | 0.15 |
| TSMOM proportional, replace | 0.79 (0.57/0.98) | +10.5% | -22.0% | -7.4% | +0.06 [-0.11, +0.23] | 0.23 |
| TSMOM majority AND 200d | 0.77 (0.51/1.01) | +12.1% | -28.3% | -8.4% | +0.04 | 0.25 |
| Kalman local-linear-trend slope, MLE walk-forward, AND 200d | 0.71 | +11.7% | -28.8% | -10.8% | -0.01 | 0.71 |
| Kalman MLE, replace (slope never turns negative) | 0.54 | +9.9% | -46.8% | -18.0% | -0.19 | 0.95 |
| HMM 2-state P(calm) > 0.5, AND 200d | 0.64 | +9.7% | -29.4% | -10.8% | -0.09 | 0.85 |
| HMM, replace | 0.63 | +10.3% | -41.7% | -10.8% | -0.09 | 0.81 |
| VIX/VIX3M < 1, AND 200d | 0.63 (0.29/0.96) | +10.0% | -33.2% | -10.8% | -0.09 | 0.90 |
| VIX < 50-day average, AND 200d | 0.54 | +6.2% | -28.9% | -7.4% | -0.18 | 0.86 |
| HYG/IEF > 50-day average, AND 200d | 0.65 | +8.7% | -33.9% | -10.8% | -0.08 | 0.71 |
| 5-signal majority vote | 0.69 | +11.6% | -35.2% | -10.8% | -0.04 | 0.70 |
| No filter (vol target only) | 0.54 | +10.4% | -46.8% | -18.0% | -0.18 | 0.94 |
| Meta: each Jan 1 pick the best-so-far | 0.62 | +10.7% | -43.0% | -18.0% | -0.11 | 0.87 |

30 variants in all (the rest are in `trader/data/overnight/exp_signals.json`). **Verdict: keep the 200-day
filter.** TSMOM is the only one ahead, not significantly, and it is worse in 2003-2007 (0.77 vs 0.98): promising
only. Real funds 2010-2026 agree (TSMOM 0.88 vs 0.80, p 0.17).

## (c) Calendar and event nights (exp_calendar.py)

Equal-weight SPY/QQQ/IWM overnight return on trend-on nights, flagged vs other nights (bp; Welch t):

| Night (morning of) | 2001-2007 | t | 2008-2026 | t | h1 / h2 t |
|---|---|---|---|---|---|
| FOMC statement day | +19.1 vs +5.0 | **+3.12** | +15.6 vs +4.6 | **+2.23** | +1.47 / +1.69 |
| Jobs report (8:30) | +14.4 vs +5.0 | +1.29 | +8.8 vs +4.7 | +0.69 | +0.32 / +0.64 |
| CPI (8:30; 2018-2025 only) | n/a | | +12.6 vs +6.2 | +0.71 | |
| Monthly opex Friday | +18.6 vs +4.8 | +1.75 | +11.6 vs +4.6 | +1.24 | +1.40 / +0.14 |
| Night after opex | -3.0 vs +5.9 | -1.41 | +0.5 vs +5.2 | -0.88 | -0.54 / -0.70 |
| Weekend (Fri -> Mon) | +6.0 vs +5.4 | +0.18 | +4.7 vs +5.0 | -0.08 | |
| Turn of month | +8.1 vs +4.8 | +0.96 | +5.2 vs +4.9 | +0.15 | |
| Pre-holiday / holiday night | | -0.54 / +0.21 | | +0.50 / -0.38 | |

Rules (skip or 1.5x the flagged nights, 28 variants, direction chosen on 2001-2007): best pre-chosen rule = skip
post-opex nights, SR 0.78 vs 0.73 (p 0.19); FOMC x1.5 adds +0.01 (L is usually at its cap); skipping FOMC nights
costs -0.08 (p 0.98). **Verdict: no calendar rule adopted. Do not skip event nights**: FOMC mornings carry the
pre-FOMC drift (Lucca & Moench 2015), and jobs/CPI mornings are above average too, despite the higher variance.
Event dates are approximate: the BLS and Federal Reserve sites were unreachable from this session, so the jobs report
follows the BLS rule (third Friday after the week containing the 12th, with January/July-4/shutdown adjustments),
FOMC dates 2000-2026 and CPI dates 2018-2025 were typed from memory (`calendar_flags.py`).

## (d) Sizing (exp_sizing.py), 200-day filter, out of sample 2008-2026

Forecast accuracy for the next night's sigma (QLIKE, Diebold-Mariano t vs 20-night sd; negative = better):
EWMA 0.94 -4.0/-2.7/-3.5 (SPY/QQQ/IWM), GARCH(1,1) MLE walk-forward -5.3/-3.5/-4.1, VIX-scaled -4.8/-2.5/-3.5.
Better forecasts, **no better strategy**:

| Variant | SR (h1 / h2) | CAGR | maxDD | worst | stress* | dSR, p | SR 2003-07 |
|---|---|---|---|---|---|---|---|
| **Spec (cap 3)** | 0.73 (0.53/0.91) | +12.3% | -28.8% | -10.8% | -16.3% | | 0.98 |
| GARCH(1,1) sigma | 0.72 | +11.8% | -28.5% | -9.3% | -16.3% | -0.00, 0.54 | 0.78 |
| EWMA 0.94 sigma | 0.72 | +11.9% | -28.8% | -10.7% | -16.3% | -0.01, 0.72 | 1.00 |
| VIX-scaled sigma | 0.73 | +11.5% | -30.6% | -9.3% | -16.0% | +0.00, 0.51 | 0.52 |
| Inverse variance (Moreira-Muir) | 0.69 | +11.5% | -30.8% | -11.2% | -16.3% | -0.04, 0.91 | 0.97 |
| Half Kelly, mean shrunk N(0, 3bp) | 0.78 | +15.7% | -34.7% | **-12.3%** | -16.3% | +0.06, 0.06 | 0.99 |
| Quarter Kelly | 0.70 | +12.4% | -30.4% | -11.2% | -16.3% | -0.02, 0.85 | 0.93 |
| Target 1.0% / 1.25% (cap 3) | 0.67 / 0.70 | +8.6% / +10.6% | | -7.9% / -9.9% | | -0.06 / -0.03 | |
| Cap 2.5 | 0.75 (0.56/0.92) | +11.7% | -27.0% | -9.3% | -13.5% | +0.02, 0.08 | 0.99 |
| **Cap 2.0** | **0.78 (0.60/0.94)** | +10.6% | -24.3% | **-7.8%** | -10.8% | +0.05, 0.06 | 0.99 |
| Cap 1.75 | 0.79 (0.61/0.96) | +9.7% | -22.6% | -7.5% | -9.5% | +0.07, 0.05 | 0.99 |
| Cap 2.0 + stress guard (recommended) | 0.78 (0.60/0.94) | +10.4% | -24.3% | -7.8% | -10.0% | +0.05, 0.07 | 0.99 |

\* stress = sleeve loss if every invested leg took its point-in-time worst overnight gap seen while above its
200-day average (now SPY -4.0% on 2024-08-05, QQQ -7.0% on 2020-03-09, IWM -5.3% on 2024-08-05).

Lowering the cap removes exposure only in calm markets, where realized volatility understates jump risk (Brexit,
Feb 24 2020 and Aug 5 2024 all hit legs at or near the cap); lowering the target instead cuts exposure everywhere and
lowers Sharpe. **Verdict: adopt cap 2.0 + stress guard as a risk-limit fix** (the $100 limit binds); not a
statistically significant return improvement (sizing SPA p 0.45). Keep sigma20 (GARCH/EWMA add complexity for
nothing). Kelly: rejected (half Kelly breaks the $100 limit; quarter Kelly no gain).

## (e) Asset set (exp_assets.py, exp_realfunds.py, exp_wholeshares.py)

Overnight correlations 2008-2026: SMH 0.81/0.88/0.76 and XLF 0.87/0.73/0.85 with SPY/QQQ/IWM. Equal sleeve weights.

| Legs (cap) | SR proxy 2008-26 (h1/h2) | dSR, p | SR 2003-07 | SR real funds 2010-26 | real dSR, p | worst |
|---|---|---|---|---|---|---|
| SPY/QQQ/IWM (3) = spec | 0.73 (0.53/0.91) | | 0.98 | 0.80 | | -10.8% |
| SPY/QQQ/IWM (2) | 0.78 (0.60/0.94) | +0.05, 0.06 | 0.99 | 0.85 | +0.05, 0.07 | -7.8% |
| + SMH (2) | 0.90 (0.58/1.17) | +0.17, 0.01 | 0.92 | 0.96 | +0.16, 0.009 | -7.8% |
| + SMH + XLF (2) | 0.89 (0.60/1.15) | +0.16, 0.00 | 1.12 | 0.96 | +0.16, 0.005 | -8.4% |
| + XLF (2) | 0.78 (0.62/0.94) | +0.06, 0.16 | 1.22 | 0.87 | +0.06, 0.12 | -8.2% |
| + SMH (3) | 0.83 (0.53/1.11) | +0.11, 0.02 | 0.87 | 0.91 | +0.11, 0.02 | -10.6% |

Whole shares on a constant $1,000 sleeve at today's prices (UPRO $125, TQQQ $52, TNA $51, SOXL $49, FAS $161), real
funds 2010-2026, nearest rounding: spec SR 0.82 / worst -$110; 3 legs cap 2 SR 0.86 / -$80; +SOXL cap 2 SR 0.99 /
-$69; 5 legs: FAS and UPRO are coarser than the $133 per-leg target (floor rounding leaves 21% of wanted legs empty).
**Verdict: promising, not adopted.** The SOXL leg wins in 2010-2026 on proxy and real data (within-question SPA
p 0.026), but not in 2003-2009 (2003-07 SR 0.92 vs 0.99 for 3 legs at cap 2; 2008-17 half 0.58 vs 0.60), and the
family-wide SPA is p 0.27: the gain may be the decade's semiconductor boom. SOXL's index differs from SMH's (overnight
tracking gap -30 bp/night in 2024) and SOXL moved +/-20% overnight several times in 2026. XLF adds nothing on its own;
the 5-leg book is not tradable in whole shares at $1,000.

## Recommended live spec (exact, executable)

Unchanged from the live spec except **CAP = 2.0** and the **stress guard**. Reference code: `final_spec.py --decide`.

1. Legs: SPY -> UPRO, QQQ -> TQQQ, IWM -> TNA; leg budget = sleeve / 3.
2. At ~15:58 ET, for each index i: price p_i = current trade price; SMA_i = mean of the last 199 daily closes and p_i.
   Trend_i = 1 if p_i > SMA_i, else 0.
3. sigma_i = sample standard deviation (n-1) of the last 20 overnight returns (open_d + dividend if d is the ETF's
   ex-date) / close_{d-1} - 1, the latest being this morning's.
4. L_i = Trend_i x min(2.0, 0.015 / sigma_i). Dollars D_i = (sleeve / 3) x L_i / 3 (at most 2/9 of the sleeve per fund).
5. Stress guard: stress = 3 x (0.040 D_SPY + 0.070 D_QQQ + 0.053 D_IWM). If stress > $100 (scale with the sleeve),
   multiply every D_i by 100 / stress. (Binds only when all three legs are near the cap: max total 1.84x the sleeve.)
   Update a G_i if a larger overnight gap occurs while that index is above its 200-day average.
6. Shares_i = round(D_i / fund price); if the total cost exceeds settled cash, round down instead.
7. Buy at ~15:58 (marketable limit), sell everything at the next 9:30 open. Hold through FOMC/jobs/CPI nights.
8. Backtest, realistic costs: OOS 2008-2026 SR 0.78 [0.36, 1.18], CAGR +10.4% [+4.3%, +16.9%], maxDD -24.3%, worst
   night -7.8% (2024-08-05; Brexit -7.5%), t = 3.37; 2001-2026 SR 0.79, CAGR +9.8%; real funds 2010-2026 SR 0.85,
   CAGR +11.7%. Average annual drag of the sleeve: trading costs ~4.8%, financing carry 1.6% (2008-2026 rates; ~4.8%
   at today's 4.1% T-bill). Expect live results below these.
9. As of the Nov 18 close, SPY, QQQ and IWM are all below their 200-day averages: both specs say cash tonight.

## Open questions for the next iteration

1. Shadow-track a 4th SOXL leg (sleeve / 4 per leg, cap 2) for 3-6 months before adopting; re-test with SOXX
   (closer to SOXL's index) and with a semiconductor-earnings-night flag (NVDA/AMD/AVGO report after the close).
2. Longer trend windows (SMA 250 led the in-sample grid) and TSMOM need a proper nested walk-forward test.
3. Event calendars: fetch the official BLS/Fed schedules (needs bls.gov and federalreserve.gov allowed) and re-run (c);
   add earnings-season and election nights; macro-announcement nights look favourable, not risky, on average.
4. Carry: at 4% rates the 3x financing costs ~10% a year per fund dollar held overnight. Compare against holding
   1x ETFs on low-conviction nights (no carry) and against shifting exposure to the cheapest-carry fund.
5. Intraday 15:58 vs 16:00 close and open-auction slippage for TNA (thinner) with live fills (the execution
   family's fill study reports -1.5 to +0.6 bp; keep measuring).
6. The previous summary (SR 0.92) could not be reproduced exactly (0.99 with the same conventions here); the main
   session should keep one reference implementation (`core.py`).

## Files

`tools/research/overnight/`: `fetch.py` (data), `repair_cache.py` (cache fix), `core.py` (returns, carry, spec,
stats), `models.py` (Nelder-Mead, HMM Baum-Welch/filter, Kalman LLT, GARCH, EWMA; numba optional), `signals.py`,
`calendar_flags.py`, `validate.py` (bootstrap, SPA/RC, CSCV, DSR), `evalx.py`, experiments `tracking.py`,
`exp_signals.py`, `exp_calendar.py`, `exp_sizing.py`, `exp_assets.py`, `exp_realfunds.py`, `exp_wholeshares.py`,
`exp_validate.py`, `final_spec.py`, `log_results.py` (registry; already run once). Result files and every variant's
nightly series: `trader/data/overnight/` (git-ignored). Run from the repository root, in that order.
