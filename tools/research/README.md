# Research log

**Where things stand (Nov 19, 2026).** Live: the overnight index model (`tools/overnight.py`), half the sleeve
per night because a cash account cannot spend unsettled sale proceeds. Daytime: paper forward test only.
Every experiment, including failures, is in `registry.jsonl` (`python3 tools/research/registry.py` prints it
with trial counts); rules for a valid test are in `STANDARDS.md`; agent reports are in `reports/`.
Live monitoring: `monitor.py` (execution tracking, expected range, Bayesian edge estimate).

Tools for testing day-trading and overnight rules on free Yahoo data before any rule is traded.
Data is cached under `trader/data/` (git-ignored). Run every script from the repository root.

| Script | What it tests |
|---|---|
| `yfetch.py` | Downloads bars to `trader/data/<interval>/` (`superset` builds a liquid-stock list from `trader/universe.txt`) |
| `bars.py`, `stats.py` | Shared loaders; Sharpe, day-clustered t-stats, deflated Sharpe ratio (Bailey & Lopez de Prado 2014) |
| `etf_momo.py` | Index intraday momentum on ~2.9 years of hourly bars, traded long-only with 3x funds |
| `letf_orb.py` | The sleeve's own 3x-fund opening-range rules over the last ~60 sessions |
| `sip_orb.py`, `sip_eval.py` | Stocks-in-play opening-range breakout (long only), 1,651 stocks, point-in-time filters, 40 variants |
| `xsec.py` | Fama-MacBeth cross-sectional test of what predicts the 9:45 to 15:55 return |

## Findings, Nov 18 2026

- **Index intraday momentum (720 days, hourly):** following the first hour or the day-so-far into the last
  half hour lost money (day-so-far: t = -4.9 on SPY, -4.0 on QQQ, net of 5 bp). The late-day move tended to
  reverse, the opposite of Gao, Han, Li & Zhou (2018). Fading it earns about +4 bp a day on 3x funds at a
  2 bp cost (t = 1.5): too small to matter, and one of many variants tried.
- **3x-fund opening-range breakouts (60 sessions):** no edge without the market filter (+0.01R). With the
  filter, +0.20R a trade, but a day-clustered t of 0.0. Not distinguishable from zero.
- **Stocks in play (45 sessions, 72,614 stock-days):** all 40 long-only variants were at or below zero after
  10 bp costs. Best Sharpe 0.57, deflated-Sharpe probability 0.01. Higher relative volume did not help.
- **Cross-section (68,868 stock-days):** no feature known at 9:45 passed a multiple-testing bar (best t = 2.07).
  Every long-only decile was negative after costs.
- **Overnight vs intraday (10 years, daily):** nearly all of the return came overnight. SPY +3.8 bp a night
  (t = 2.6) vs +1.6 bp a day; QQQ +5.2 (t = 3.0) vs +2.7; IWM +6.2 (t = 3.4) vs -2.2.
- **3x overnight hold with a 200-day trend filter (QQQ, 10 years, 4 bp costs):** Sharpe 0.95, +29% a year,
  max drawdown -50%. Sizing to a 1.5% overnight volatility target, capped at 3x: Sharpe 0.91, +19% a year,
  max drawdown -26%, worst night -9.4%. Stable across both halves of the sample.

Caveats: about 60 days of intraday history is short. Expect live results below backtests, because of
selection across variants and decay after publication. Next step: multi-year minute data.
- **Three-index version (SPY/QQQ/IWM thirds via UPRO/TQQQ/TNA, same rules, 2,312 nights):** +17.0% a year,
  Sharpe 0.92, max drawdown -27%, worst night -8.4%, worst year -12%, Sharpe 0.99 / 0.86 in the two halves.
  More stable than any single index.
- **Downtrends:** below the 200-day average, the next night averaged about zero (t between -0.1 and +1.2), so
  holding inverse funds overnight has no support. Cash is the default in downtrends.

## Findings, Nov 19 2026 (main session)

- **Cash-account settlement halves the overnight model.** Robinhood's buying power excludes unsettled sale
  proceeds, so money sold at the 9:30 open cannot buy again at 15:58 the same day. Feasible spec: half the
  sleeve per night. With actual 3x fund prices (fees and financing included), 10 years: full capital every
  night +13.6%/yr, Sharpe 0.79, max drawdown -30% (needs limited margin, no borrowing); half capital
  +7.0%/yr, Sharpe 0.79, max drawdown -16%, worst night -$42; holding around the clock while above the
  200-day average: Sharpe 0.57 (rejected).
- **Execution is not a hidden cost.** Buying at ~15:58 instead of the official close: -1.5 to +0.6 bp on
  average; selling at the 9:30 first print vs the official open: -0.5 bp. Noise about 15 bp per leg per night.
- **Live evidence accrues slowly.** At about 2.7 bp mean and 54 bp standard deviation per night on the sleeve,
  15 live nights move the edge estimate about 4% of the way from the prior. The live monitor therefore watches
  execution tracking first, and judges P&L against the backtest's expected range.
