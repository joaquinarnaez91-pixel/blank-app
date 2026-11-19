# Cross-sectional overnight and swing strategies on daily data

Author: agent-xsec-swing, Nov 19 2026. Code: `tools/research/xsec_swing/`. Registry: families `xsec_overnight`,
`xsec_swing`, `xsec_data`, plus one record each in `overnight` and `gaps_intraday`.

## Bottom line

- **Nothing new to trade.** No cross-sectional stock strategy beats the live index overnight model or usefully
  adds to it after realistic costs and a survivorship correction. Keep the live model as it is.
- **The one robust effect:** momentum winners earn their extra return overnight (Lou, Polk & Skouras replicated:
  winners minus losers +10.0 bp a night, t 3.30, about zero intraday). Long only, it does not survive costs. The
  best top-N version made +9.3 bp a night net (Sharpe 0.68, t 2.04, deflated-Sharpe probability 0.55, max drawdown
  -68%), against Sharpe 0.89 for the index model. At 15 bp a night the Sharpe falls to 0.31.
- **Survivorship bias is large for concentrated picks.** Adding the names that are still listed but now trade
  below $5 or thinly cut the Sharpe of overnight momentum (top 5) from 0.93 to 0.58, the Connors RSI(2) rule from
  0.60 to 0.37 and the weekly loser reversal from 0.73 to 0.43. Names that were delisted are still missing, so the
  true figures are probably lower still.
- **Connors RSI(2) has no edge after its 2008 publication.** Sharpe 0.37 (t 1.10). Moving the fills from the
  signal close to the next open drops it to 0.09: most of the backtest edge was the bounce of the close used both
  as the signal and as the fill.
- **Cash-day gap-down reversal** (agent-intraday-hourly's rule): adding fallen names leaves it unchanged
  (+79 bp a cash day, t 3.5). Names with long histories earn about half that (+48 bp, t 2.4). Recent listings
  supply the rest (+101 bp), and that is where the bias from delisted names is largest. After Dec 21, 2023 the
  broad set earned about 0 (t 0.04, few cash days). Plan on roughly +30 to +50 bp a cash day, not 58 to 100.

## Data and method

- **Data.** Yahoo daily bars from Nov 21, 2016 to Nov 18, 2026 for 5,958 symbols: the 1,651-name superset, plus
  every name in `trader/universe.txt` (today's listings, including the ones now below $5 or thin), plus 28 ETFs.
  Dividends and splits come from Yahoo's chart events. Fetched with 3 threads (`fetch.py`). The panel
  (`panel.py`, `trader/data/xsec_panel.npz`) keeps the 3,195 names whose 60-day mean dollar volume ever reached
  $20M. The point-in-time top-500 cut-off never fell below $79M, so this filter loses nothing.
- **Point-in-time universe.** Each day t: common stocks (no ETFs) ranked by mean dollar volume over t-60..t-1,
  top 500, with an actual (split-unadjusted) close at t-1 of at least $5. Yahoo prices are split-adjusted, so a
  plain $5 filter would quietly admit stocks that later did reverse splits. Whole-share sizing also uses actual
  prices. The median universe holds 500 names, and 1,124 distinct names pass through it.
- **Returns.**
  - Overnight: (open of t+1 + dividend) / close of t - 1.
  - Intraday: close / open - 1.
  - Decisions use the close of t as a 15:58 proxy. Exits at the open are 9:30 market orders.
- **Costs.** 10 bp per round trip. An overnight strategy pays this every night; a swing strategy pays it once per
  trade. 3x funds pay 4 bp a night.
- **Statistics.**
  - Daily portfolio series, with zeros on flat days. t = mean / sd x sqrt(n); Sharpe is annualized with sqrt(252).
  - Halves split at May 25, 2022.
  - Deflated-Sharpe probability uses the family's trial count and the variance of the Sharpes across its variants.
  - Walk-forward: each year, pick the variant with the best Sharpe on all data before that year (expanding
    window). Out-of-sample results run from 2019.
  - Tests cover Dec 4, 2017 to Nov 18, 2026 (2,252 days) after a 260-day warm-up.
- **Index-model benchmark.** I rebuilt the model from actual fund prices: equal thirds of UPRO/TQQQ/TNA, the
  200-day filter, exposure min(3, 1.5% / 20-night volatility), 4 bp a night. It gives Sharpe 0.89, +16.3% a
  year, max drawdown -29.1%, worst night -8.45%, close to the README's 0.92, +17%, -27% and -8.4%. It predates
  the cap-2.0 / stress-guard revision and the financing-carry corrections made by other agents.

## Survivorship estimate (`survivorship.py`)

| Measure (equal weight, point-in-time top 500, Dec 2017 - Nov 2026) | %/yr | t |
|---|---|---|
| Superset only minus all names listed today | +0.84 | 1.53 |
| ... overnight part / intraday part | -0.15 / +0.85 | |
| All listed names minus RSP (equal-weight S&P 500; holds the dead names) | +1.58 | 0.65 |

The universe-level bias is about 1% a year. It sits entirely in intraday returns and is concentrated in 2021
(+4.6%) and 2025 (+2.3%), when speculative names collapsed. It is much larger for concentrated strategies:

| Strategy | Sharpe, superset only | Sharpe, all listed names |
|---|---|---|
| Overnight momentum, top 5 | 0.93 (+13.7 bp a night) | 0.58 (+8.7 bp) |
| Connors RSI(2), 10 slots | 0.60 | 0.37 |
| Weekly loser reversal, top 200, above the 200-day | 0.73 | 0.43 |

Truly delisted names are still missing: bankruptcies (for example NKLA, RIDE, BBBY, SIVB, FRC) and acquisitions.
For concentrated speculative picks, assume a bias at least as large again.

## (a) Overnight persistence (Aboody, Even-Tov, Lehavy & Trueman 2018): `overnight_xsec.py`, 20 variants

- **Deciles.** Next-night overnight return by decile of the past-20-night mean overnight return, gross, in bp:
  +4.0, +4.8, +4.1, +4.2, +4.1, +5.1, +6.1, +5.6, +6.7, +9.0. The top-minus-bottom spread is 5.0 bp a night, t 2.09.
- **Long-only top N** (N = 5/10/20/50, lookbacks 5/10/15/20 nights), net of 10 bp:
  - Best: 20-night lookback, top 5. +6.0 bp a night, Sharpe 0.36, t 1.07, max drawdown -82%.
  - Every top-50 version lost money. Gating by SPY above its 200-day average did not help (4 variants, all near 0).
- **Verdict: rejected.** Persistence is real but small in liquid names; costs exceed it.

## (b) Tug of war (Lou, Polk & Skouras 2019): `overnight_xsec.py`, `momentum_hold.py`, `letf_rotation.py`

12-1 month momentum deciles (gross, bp per night or per day):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Winners minus losers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Overnight | +2.4 | +3.2 | +3.6 | +3.7 | +4.2 | +4.6 | +5.2 | +6.7 | +7.3 | +12.4 | +10.0, t 3.30 (halves 1.54 / 3.03) |
| Intraday | +0.3 | +1.9 | +2.1 | +0.9 | +0.9 | +1.1 | -1.3 | -0.9 | -1.6 | -0.7 | -1.0, t -0.24 |

**Long only, close to next open, 10 bp a night** (10 variants: 12-1 momentum or 12-month cumulative overnight
return, times N = 5/10/20/50, plus 2 versions gated by SPY above its 200-day):

| Variant | Gross bp | Net bp | Sharpe | t | Max DD | Halves Sharpe |
|---|---|---|---|---|---|---|
| 12-month overnight return, top 10 (best) | 19.3 | +9.3 | 0.68 | 2.04 | -68% | 0.79 / 0.58 |
| 12-1 momentum, top 5 | 18.7 | +8.7 | 0.58 | 1.74 | -61% | 0.36 / 0.80 |
| 12-1 momentum, top 50 | 12.5 | +2.5 | 0.27 | 0.80 | -48% | |
| Live index model | | +6.7 | 0.89 | 2.65 | -29% | 1.04 / 0.74 |

- **Best variant.** Deflated-Sharpe probability 0.55 (32 trials in the family). Cost sensitivity: Sharpe 1.42
  with no costs, 0.31 at 15 bp, -0.05 at 20 bp.
- **Worst single-stock night** among its picks: -40%. 0.12% of the picked stock-nights fell more than 20%.
- **Walk-forward across all 32 overnight variants:** Sharpe 0.35 (t 0.98), against 0.86 for the index model over
  the same period.
- **Against the index model.** Correlation 0.57. On nights the index model holds cash, the candidate made
  +1.4 bp (t 0.13), so it adds nothing in downtrends. A 50/50 split gives Sharpe 0.84 and max drawdown -49%,
  both worse than the index model alone.
- **Holding the winners through the day** (top 5/10, sell when the rank drops below 2N; 2 variants): Sharpe
  0.73 / 0.75 with max drawdowns of -59% / -67%. Holding intraday adds risk but no return, as the tug-of-war paper
  predicts.
- **3x-fund rotation** (7 variants: UPRO/TQQQ/TNA/SOXL/FAS with the live rules, top K by momentum). The best
  momentum rotation reached Sharpe 0.90, against 0.89 for the live thirds. Equal fifths of all five gave 0.94, a
  difference of +0.0 bp a night (t 0.06). Walk-forward out-of-sample: 0.81, against 0.86 for the live model.
  Momentum adds nothing at the index level.
- **Verdict: rejected.** The long-only momentum portfolio cannot beat the index overnight model, either alone or
  combined with it.

## (c) Short-term mean reversion (swing): `meanrev.py`, `meanrev_1000.py`, `regime.py`

**Signal portfolios** (10 slots of 10%, fractional, close fills, 10 bp, all names listed today; 10 variants):

| Variant | Sharpe | t | Trades | Mean / trade | Win | Max DD | Superset-only Sharpe |
|---|---|---|---|---|---|---|---|
| Connors: RSI(2) < 10, close > 200d; exit close > 5d SMA or 5 days (published rule) | 0.37 | 1.10 | 6,014 | +0.13% | 63% | -47% | 0.60 |
| Same, RSI(2) < 5 | 0.15 | 0.46 | 5,001 | +0.06% | 62% | -48% | 0.40 |
| Same, exit by 10 days (best) | 0.48 | 1.44 | 5,039 | +0.20% | 66% | -55% | 0.66 |
| RSI(2) < 10, time exit 1 / 3 / 5 days | -0.44 / 0.30 / 0.29 | | | | | | |
| Connors rule, top 200 | 0.37 | 1.09 | 5,138 | +0.14% | 63% | -42% | 0.49 |
| Weekly losers (bottom 10 by 5-day return), top 200, hold 5 days | 0.40 | 1.18 | 4,510 | +0.37% | 51% | -70% | 0.66 |
| Same, above the 200-day | 0.43 | 1.28 | 3,284 | +0.38% | 52% | -62% | 0.73 |
| Weekly losers, top 500 | 0.28 | 0.85 | 4,510 | +0.30% | 49% | -79% | 0.48 |

- **Best variant:** deflated-Sharpe probability 0.58. Walk-forward out-of-sample Sharpe 0.17 (t 0.47).
- **$1,000, whole shares, actual prices.**
  - Connors with 3 slots: Sharpe 0.42, max drawdown -59%.
  - Connors with 5 slots: Sharpe 0.38. 3,137 trades: mean +0.16%, median +0.91%, win 63%, average win +2.88%,
    average loss -4.45%, worst -39.6%.
  - Weekly reversal with 5 slots: Sharpe 0.50, max drawdown -70%, worst trade -66.7%.
  - A -40% to -67% trade on a $200-$333 slot breaks the $100 per-trade limit.
- **Next-open fills** (signal at the close, buy and sell at the next 9:30 open): Connors falls to Sharpe 0.09,
  the weekly reversal to 0.30.
- **Only while the index model holds cash** (all three indices below their 200-day; prior: Nagel 2012; 10 variants;
  353 such closes):
  - Best in sample: weekly reversal above the 200-day. Gated t 1.59, deflated-Sharpe probability 0.77. Combined
    with the index model (leftover cash goes into the index model each night): Sharpe 1.03 against 0.89.
  - Walk-forward: combined 0.87 against 0.86 for the index model alone. The increment is +3.0 bp a day, t 1.08.
  - The $1,000 version's worst trade was -$153.
- **Verdict: rejected.** No variant passes. The correlation with the index model is low (0.30), but there is no
  return to diversify with.

## (d) Post-earnings drift, proxied: `pead_proxy.py`, 4 variants

- **No earnings dates.** SEC EDGAR is not on this session's network allowlist, and Yahoo's earnings-date
  endpoints need a session crumb, which I did not pursue.
- **Proxy event:** volume at least 3x its 60-day median, with both the opening gap and the day's return at least
  +3% (or +6%). Buy at the close and hold 20 or 60 days.
- **Results:** best Sharpe 0.55 (t 1.63), max drawdown -72%. In the event study, the 5- to 60-day excess return
  over the universe was -0.9% to +1.6%, with month-clustered |t| below 1.
- **Verdict: rejected** for the proxy. The real test needs earnings dates.

## Survivorship check of the cash-day gap-down reversal: `gapdown_check.py`, 6 subsets

Rule: on days when SPY closed below its 200-day average the prior day, buy the 5 largest opening gap-downs
(<= -2%) at the open and sell at the close. Equal weight, 10 bp.

| Candidate set | Active days | Net bp / day | t | Before Dec 21, 2023 | After |
|---|---|---|---|---|---|
| Superset, no point-in-time filter | 438 | +144.4 | 6.01 | +166 (t 6.5) | -3 (t -0.04) |
| Superset, point-in-time price >= $5 and 60-day volume >= 750k | 434 | +75.2 | 3.50 | +84 (t 3.6) | +18 (t 0.3) |
| All listed names, same filter | 438 | +78.8 | 3.51 | +90 (t 3.7) | +2 (t 0.04) |
| Long histories only (bars since Jan 2017) | 434 | +48.3 | 2.36 | +50 (t 2.3) | +36 (t 0.65) |
| Recent listings only | 395 | +100.6 | 4.14 | +101 (t 3.9) | +96 (t 1.5) |
| Point-in-time top 500 by dollar volume | 428 | +29.2 | 1.54 | +19 (t 0.9) | +99 (t 2.3) |

Reading:
- Without a point-in-time price filter, the backtest is grossly inflated.
- Today's price and volume filter is not what drives the result.
- Half the edge comes from post-2017 listings, where failed listings (now delisted) are invisible.
- **Verdict: promising, but discount it.** Keep the paper forward test, and track the long-history and top-500
  subsets separately.

## Verdicts

| Idea | Variants | Best result | Verdict |
|---|---|---|---|
| (a) Overnight persistence | 20 | +6.0 bp a night net, Sharpe 0.36, t 1.07 | rejected |
| (b) Momentum winners earn overnight (decomposition) | 1 | Winners minus losers +10.0 bp a night, t 3.30 | effect confirmed (not tradeable long only) |
| (b) Long-only overnight momentum | 10 | Sharpe 0.68, t 2.04, deflated 0.55, walk-forward 0.35 | rejected |
| (b) Overnight reversal of intraday losers | 2 | -3.3 bp a night | rejected |
| (b) 3x-fund momentum rotation | 7 | +0.0 bp a night versus the live model, t 0.06 | rejected |
| (b) Momentum held through the day | 2 | Sharpe 0.75, max drawdown -67% | rejected |
| (c) Connors / weekly reversal | 10 + 6 | Sharpe 0.48, t 1.44; next-open fills 0.09 | rejected |
| (c) Reversal only on cash days | 10 | Walk-forward increment +3.0 bp a day, t 1.08 | rejected |
| (d) Post-earnings drift proxy | 4 | Sharpe 0.55, t 1.63 | rejected |
| Gap-down reversal, survivorship check | 6 | Long histories +48 bp a cash day, t 2.36 | promising, discounted |

## Exact rules

None recommended for live trading. To reproduce the strongest candidate (overnight momentum, rejected): at
~15:58, rank the point-in-time top 500 by the sum of log overnight returns over the last 252 nights; buy the top
10 with equal dollars; sell at the next 9:30 open. It needs costs of 10 bp a night or less, and it still trails
the index model.

## Open questions

1. **Earnings dates.** Add data.sec.gov to the allowlist (8-K item 2.02) to run real earnings drift tests and to
   exclude earnings nights from overnight stock holds; earnings nights are the main tail risk.
2. **Delisted names.** Use a source with delisted names (for example CRSP-style data) to bound the remaining bias.
3. **Execution costs.** Measure real fill costs for single stocks at 15:58 and at the 9:30 open from forward paper
   fills. Overnight momentum lives or dies between 10 and 15 bp a night.
4. **More cash-regime history.** Reversal on cash days needs years before 2017, or VIX-based states.
5. **Gap-down rule.** Forward-test the long-history and top-500 subsets alongside the base rule.
