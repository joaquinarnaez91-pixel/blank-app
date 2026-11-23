# Gap-down basket: does SPY's state at the open tell us when to skip? (Nov 23 2026)

**Bottom line: no.** None of the 10 pre-specified buckets differs from the baseline by more than noise. The
largest difference has t 1.1 against a multiple-testing hurdle of about 3 (STANDARDS.md rule 5; Bonferroni for
10 tests also needs about 2.8). The trend that does show runs the "wrong" way for a skip rule: days when SPY
itself gaps down more than 1% were the *best* days. **Verdict: rejected. No skip rule is justified, so keep the
pilot rules unchanged.** What does drive the basket is SPY's direction *after* the open. That is not known at
entry, which is exactly why the Nov 19-20 losses could not have been filtered out.

## Setup
- **Data and rule:** the pilot's own backtest data and rule (`intraday_hourly/gd_regime.py`): Yahoo daily bars
  for `trader/data/superset.json`, 433 cash days from 2018-04-03 to 2026-11-20, including both pilot days.
  - Cash day: SPY closed below its 200-day SMA on t-1.
  - Filters: prior close >= $5, 14-day volume >= 1M, gap <= -2%, open <= $95.
  - Trade: the 5 most negative gaps, equal weight, open->close, net of 10 bp.
- **Entry approximation:** daily bars have no 9:45 price, so entry is at the 9:30 open. The pilot's 9:45 entry was
  estimated earlier at about 45 bp vs about 58 bp here.
- **Sensitivity (`oco`):** the stop -8% / take-profit +20% applied to the daily high and low, with the stop
  assumed to hit first.
- **Point in time:** all three conditioning variables are known at 9:30 or earlier. Nothing after the open is
  used as a filter.
- **t-stats:** computed on the daily basket series (one observation per day).
- **Halves:** H1 = 2018-2021, H2 = 2022-2026.

| Variable | Bucket | Days | Mean bp (net) | t | Hit | H1 / H2 bp | Diff vs rest bp (t) | With stop/TP bp |
|---|---|---|---|---|---|---|---|---|
| Baseline | all cash days | 433 | +62 | 3.19 | 53% | +61 / +62 | | +48 |
| SPY open gap | < -1% | 60 | +142 | 1.74 | 55% | +111 / +156 | +93 (1.1) | +120 |
| | -1..0% | 160 | +45 | 1.58 | 49% | -3 / +63 | -27 (-0.7) | +25 |
| | 0..+1% | 151 | +38 | 1.42 | 54% | +74 / +22 | -37 (-1.0) | +33 |
| | > +1% | 62 | +89 | 1.72 | 60% | +114 / +70 | +31 (0.6) | +75 |
| SPY prior day | down > 1% | 118 | +73 | 1.54 | 50% | +24 / +97 | +15 (0.3) | +51 |
| | down 0..1% | 118 | +50 | 1.50 | 55% | +79 / +41 | -16 (-0.4) | +42 |
| | up | 197 | +63 | 2.50 | 54% | +75 / +56 | +1 (0.0) | +50 |
| SPY vs 200-day | 0-3% below | 170 | +47 | 1.63 | 53% | +43 / +49 | -25 (-0.6) | +44 |
| | 3-7% below | 116 | +82 | 2.38 | 53% | +83 / +82 | +27 (0.7) | +68 |
| | > 7% below | 147 | +64 | 1.67 | 54% | +63 / +64 | +3 (0.1) | +37 |
| *Look-ahead, descriptive only* | SPY falls open->close | 200 | -111 | -4.72 | 33% | -90 / -120 | -321 (-9.1) | -104 |
| | SPY rises open->close | 233 | +211 | 7.93 | 70% | +186 / +222 | | +179 |

**Continuous slopes** (basket bp per 1% of the variable, OLS t):
- SPY gap: -16 bp (t -1.1)
- prior-day return: -6 bp (t -0.6)
- distance to the 200-day: -4 bp (t -0.9)
- *SPY open->close (look-ahead): +145 bp (t 12.5, correlation 0.52)*

## Reading it
- **Every bucket is positive, and no difference is significant.** No bucket mean reaches t 3.
  - Skipping any bucket gives up expected return; it does not avoid losses.
  - The "SPY gap -1..0%" bucket is flat in H1 (-3 bp) but +63 bp in H2. The halves disagree, so it is not a usable
    signal.
- **The basket is a high-beta bet on an intraday rebound.**
  - On the 46% of cash days when SPY falls from the open, the basket loses about 1.1% on average.
  - On the days SPY rises, it gains about 2.1%.
  - Nothing known at the open forecasts which kind of day it will be (all slopes |t| < 1.1).
- **The two pilot losses fit the base rates.**
  - Nov 19: SPY open->close -2.9%. Nov 20: -3.6%. Both are SPY-falls days.
  - With a 53% hit rate, two losing days in a row happen about 22% of the time.
  - Two days say nothing about the edge. The pre-registered 20-day review and the $100 stop remain the right checks.
- **All results are in-sample.**
  - The buckets were chosen before looking, and nothing was fitted. Even so, the whole 2018-2026 sample was used.
  - The survivorship flattering the baseline (see `reports/intraday_hourly.md`) applies to every bucket equally.
- **Not tested:** SPY's 9:30-9:45 move, which would be known at a 9:45 entry. The daily bars cannot measure it,
  and 5-minute bars cover only 60 sessions (about 10 cash days), which is too few. It is a candidate for a
  pre-registered forward check in the paper log, not a rule.

Code: `tools/research/paper/gapdown_conditioning.py` (about 15 s). Output: `trader/data/gapdown_conditioning.csv`.
