# Bear-regime overnight edge (agent-bear-overnight, Nov 25 2026)

## Verdict

**Rejected. No long-only close-to-open hold below the 200-day passes the hurdles.** The best of 44 variants
(3x bull funds after a close below the lower Bollinger band, three-index sleeve) earns +12.8 bp of the $450 sleeve
per active night (+$0.57), t = 1.10 over 2008-2026, deflated-Sharpe probability 0.26. With the actual funds
(2010 on) that drops to +2.9 bp, t = 0.24. A walk-forward selector that picks the best rule each year from earlier
data loses money out of sample (-1.9 bp per active night, t = -0.45). Inverse 3x funds lose money below the 200-day
in every pre-declared form, which confirms the Nov 20 `bear_inverse` result. Cash stays the default in downtrends.

## Question and constraints

The live overnight model (`tools/overnight.py`) is in cash while SPY, QQQ and IWM are below their 200-day SMA.
Is there a long-only close-to-next-open edge in that regime? Instruments: the 3x bull and inverse funds (UPRO,
TQQQ, TNA / SPXU, SQQQ, TZA). No shorts, options or margin. Capital is about $450 (half the sleeve, cash
account). The worst stress-tested night must fit a $100 loss.

## Pre-declared hypotheses (11 rules, fixed parameters, nothing fitted)

Every rule trades only the legs whose index closed below its 200-day SMA on day t-1. It buys at ~15:58 on day t-1
(the close is the proxy) and sells at the 9:30 open on day t.

| | Rule | Trade |
|---|---|---|
| H1a | below the 200-day only (baseline) | 3x bull |
| H1b / H1c | 2+ / 3+ consecutive down closes | 3x bull |
| H1d | close below the lower Bollinger band (20 days, 2 sd) | 3x bull |
| H1e | RSI(2) < 10 | 3x bull |
| H2a | below the 200-day only | 3x inverse |
| H2b | after an up close (fade the bounce) | 3x inverse |
| H2c | after 2+ down closes (momentum) | 3x inverse |
| H3a / H3b | VIX in the top / bottom third of its below-200d history (expanding, point in time) | 3x bull |
| H3c | VIX term structure inverted (VIX > VIX3M) | 3x bull |

Each rule was run as the equal-weight three-index sleeve and on each index alone: **44 variants**, plus one
walk-forward selector, so **45** are logged. The deflated Sharpe uses 74 trials: these 44 plus the 30 from
`bear_inverse` (Nov 20).

## Method

- Data: Yahoo daily SPY/QQQ/IWM 2000-06 to 2026-11-24, ^VIX, ^VIX3M, ^IRX, and the real 3x funds (UPRO/TQQQ/TNA
  from 2010; SPXU/SQQQ/TZA from 2016). Loaded through `tools/research/overnight/core.py`.
- **Out of sample: 2008-01 onward** (4,700+ nights), the same window as the overnight program. Results for
  2000-2007 are shown only as a check. That period is also burn-in for the expanding VIX thresholds and the
  walk-forward selector.
- Every signal for night t uses data up to the close of day t-1.
- Sizing is the live spec: L = min(2, 0.015 / sigma20), and the 3x fund holds capital/3 x L/3 per leg.
- Point-in-time stress guard: all held legs gap against the position at once by max(the worst adverse gap seen so far
  while below the 200-day, 5 x sigma20). The 3x loss must stay at or under $100, otherwise every leg scales down.
- Bull-fund returns: 3 r_on - financing carry (2.4 x T-bill). Inverse-fund returns: -3 r_on (dividend owed) less
  the 0.95%/yr fee. Costs are 4 bp round trip on fund dollars.
- t-stats are computed on the full nightly sleeve series, with flat nights counted as 0. Halves are the two halves
  of the OOS window.

## Results, out of sample 2008-2026 (three-index sleeve, net)

| Rule | active nights | bp / active night | $ / active night | t | DSR | halves (bp/night) | worst night |
|---|---|---|---|---|---|---|---|
| H1a below200 | 1,672 | +0.0 | +0.00 | 0.01 | 0.04 | -0.33 / +0.35 | -$54 |
| H1b 2+ down | 533 | +2.8 | +0.13 | 0.48 | 0.10 | -0.51 / +1.15 | -$54 |
| H1c 3+ down | 290 | +4.1 | +0.18 | 0.50 | 0.11 | -0.28 / +0.78 | -$54 |
| **H1d lower BB** | 208 | **+12.8** | **+0.57** | **1.10** | 0.26 | +0.19 / +0.93 | -$54 |
| H1e RSI(2)<10 | 335 | +8.7 | +0.39 | 1.02 | 0.24 | -0.15 / +1.38 | -$54 |
| H2a inverse | 1,672 | -3.6 | -0.16 | -1.17 | 0.00 | -0.88 / -1.66 | -$41 |
| H2b inverse after up | 893 | +0.8 | +0.03 | 0.22 | 0.06 | -0.26 / +0.54 | -$28 |
| H2c inverse after 2 down | 533 | -5.9 | -0.27 | -1.00 | 0.00 | +0.19 / -1.52 | -$41 |
| H3a VIX top third | 530 | -0.9 | -0.04 | -0.13 | 0.03 | -0.18 / -0.01 | -$38 |
| H3b VIX bottom third | 597 | -0.3 | -0.01 | -0.12 | 0.03 | -0.02 / -0.05 | -$16 |
| H3c VIX > VIX3M | 381 | -3.0 | -0.14 | -0.33 | 0.02 | -0.83 / +0.34 | -$54 |

- **Single-index variants:** the best is H1e on SPY (RSI(2) < 10): +20.6 bp per active night, t = 1.33,
  DSR 0.34. No single-index variant reaches t = 2.
- **Unlevered signal strength (index r_on on signal nights, gross):** H1e +17.1 bp, t = 2.29; H1d +13.9 bp,
  t = 1.29; inverse H2a -2.7 bp, t = -0.95. The RSI(2) washout points the right way, and it agrees with the
  published short-term reversal evidence. Even before costs, carry and sizing it falls short of t = 3, and it is the
  best of 11. Vol-targeted sizing weakens it further, because the largest rebounds come on the highest-vol nights,
  where L is smallest.
- **Real funds and whole shares (H1d, 2010-2026):** +2.9 bp per active night, t = 0.24 (proxy on the same nights:
  +9.3 bp, t = 0.67). Whole shares at $450: +$0.18 per active night, worst night -$43.
- **Walk-forward selector:** it chose H3b for most years, then H1e from 2023. Out of sample it made -1.9 bp per
  active night, t = -0.45.
- **Pre-sample 2000-2007:** results are mixed. H1d is negative there (-3.5 bp). H1c and H3c are positive on few
  nights. Nothing replicates consistently across both periods.
- **Risk:** every variant's worst night fits the $100 limit. The worst is -$72 (QQQ alone), and the stress guard
  peaks at $94. Risk is not the binding constraint. There is no edge to size.

## Hurdles (STANDARDS.md)

| Hurdle | Best variant (H1d, 3-index) | Pass? |
|---|---|---|
| t > 3 out of sample (or t > 2 with a prior) | 1.10 (best of all 44: 1.33) | no |
| deflated-Sharpe probability > 0.9 | 0.26 | no |
| same sign in both halves | +0.19 / +0.93 | yes |
| costs included, worst night <= $100 | -$54 | yes |

Variants passing every hurdle: **0 of 44**.

## Implications

- Keep the sleeve in cash overnight while the indexes are below their 200-day. Holding inverse funds overnight
  makes it worse: every inverse variant is near or below zero.
- The washout signal (RSI(2) < 10, lower Bollinger band) is the only direction with a sign consistent with the
  literature. At about $0.40-0.60 a night on $450 and t near 1, it is not worth a live pilot under the pilot-tier
  rule, which needs t > 3 on the holdout.
- Files: `tools/research/bear_overnight.py`, `tools/research/bear/bear_overnight.csv` (all 88 rows: OOS and
  pre-sample), registry family `bear_overnight`.
