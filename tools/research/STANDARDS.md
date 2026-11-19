# Research standards (every experiment, every agent)

The sleeve: $1,000 in a Robinhood **cash** account (proceeds settle next day, so each dollar can be committed
once per day), whole shares, long only. Allowed: liquid US stocks, unleveraged ETFs, 3x bull funds
(TQQQ, SOXL, UPRO, TNA, FAS) and 3x inverse funds (SQQQ, SOXS, SPXU, TZA). Not allowed: options, margin,
short sales, crypto. Loss limits: $100 per trade (for an overnight hold: the plausible worst night), $200
per day, halt at $500 cumulative realized loss. Since Nov 19, 2026 overnight holds are allowed.

## Rules for a valid test
1. **Point in time.** Every input must be known at the decision time (use yesterday's values for anything
   computed from today's close unless the decision is made at the close itself). No future universe
   membership beyond what is unavoidable, and say when it is.
2. **Costs.** Charge a realistic round trip: 3x funds 4 bp (overnight) to 5 bp (intraday); liquid stocks
   10 bp; thinner names 20 bp. Report gross and net when the edge is small.
3. **Honest sample size.** Same-day trades move together: compute t-stats on the DAILY portfolio series
   (or Fama-MacBeth), never on pooled trades as if independent.
4. **Count every variant.** Log every variant tried in the registry (`registry.py`). The deflated Sharpe
   ratio (`stats.deflated_sharpe`) uses the family's total trial count, not just the ones you liked.
5. **Hurdles.** A new pattern needs t > 3 (Harvey, Liu & Zhu 2016) OR a strong prior from published
   evidence plus t > 2, AND a deflated-Sharpe probability > 0.9, AND the same sign in both halves of the sample.
6. **Fitted parameters** are judged only out of sample: walk-forward (expanding window) or purged,
   embargoed cross-validation (Lopez de Prado 2018). Report the out-of-sample numbers only.
7. **Record failures too.** A rejected idea logged now saves a repeat later.
8. **Executable.** The rule must be tradeable with this account: whole shares, Robinhood order types
   (market, limit, stop, OCO exits), no fills you could not get (no exact opening/closing auction prices
   for intraday decisions; buy at 15:58 and sell at the 9:30 open are acceptable proxies with costs).

## Data on disk (Yahoo, cached by `yfetch.py`, loaded with `bars.frame(sym, interval)`)
- `trader/data/5m/`  60 sessions of 5-minute bars for 1,651 liquid stocks + index/3x funds.
- `trader/data/1h/`  ~730 sessions (Dec 2023 on) of hourly bars for index/3x funds (bars start 9:30, 10:30 ... 15:30).
- `trader/data/1d/`  daily bars (SPY, QQQ, IWM 10y; others as fetched).
- Fetch more with `python3 tools/research/yfetch.py bars <interval> <range> SYM...`; pace requests (<= 4 threads).
