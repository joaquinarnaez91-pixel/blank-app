# Gap-down pilot: live vs backtest reconciliation (Nov 24 2026)

**Bottom line.** The live picker is not trading the backtested rule. It ranks by a different price, on a different
previous close, and drops names the backtest keeps. On the three pilot days the backtest's own picks would also have
lost: -273, -412 and +4 bp open->close, and -504, -488 and -43 bp with the live 9:45 entry and stops. So the live
losses are mostly bad days, not a broken pipeline.

Under every live constraint that daily data can express, the backtest still shows +54 bp/day (t 3.10, DSR 0.90).
That number assumes buying at the 9:30 opening print, which STANDARDS rule 8 rules out. The 9:45 entry costs an
estimated 10-30 bp, which leaves about 25-45 bp/day, t about 1.4-2.5 and DSR below 0.9. **Verdict: as it can
actually be traded, the strategy does not clear the hurdles (rejected; the pilot stays a measurement, not an
adoption).**

Code: `tools/research/paper/gapdown_reconcile.py` (about 2 min). Outputs: `trader/data/gapdown_reconcile/*.csv`.
1-minute bars for the pilot days are cached in `trader/data/1m/`, because Yahoo serves them for only about 30 days.

## 1. Reconciliation of the three live days

There are three "backtests" in the repo, and they disagree with each other:

| Code | Rule |
|---|---|
| `paper/gapdown.py` | No \|gap\| filter, no price cap |
| `intraday_hourly/gd_regime.py` (the pilot backtest) | \|gap\| < 60%, a look-ahead \|open->close\| < 60% filter, no cap |
| `paper/gapdown_conditioning.py` (the "+62 bp, 433 days" figure) | Same as gd_regime, plus open <= $95 |

The Nov 23 list quoted in the brief (CAPR, MNRO, SLDE, DHT, UNH) is the `gapdown.py` version. gd_regime drops CAPR,
because its -77% gap is beyond the 60% filter.

All returns below are net of 10 bp, as an equal-weight mean of the names. "9:45 sim" means:
- buy at the 9:45 one-minute open;
- -8% stop and +20% take-profit, checked on one-minute bars;
- out at 15:45.

The simulation reproduces the actual live fills closely: SLS -8.10% simulated vs -8.08% actual, TU -2.92% vs
-2.81%, MNSO -1.86% vs -1.90%.

| Day (SPY open->close) | Version | Names | Open->close | 9:45 sim | Live actual |
|---|---|---|---|---|---|
| Nov 19 (-2.9%) | gapdown.py / gd_regime | SLS, MNSO, WSM, WMG, HP | -273 bp | -504 bp | |
| | conditioning ($95 cap) | SLS, MNSO, WMG, HP, UPBD | -126 | -363 | |
| | live (all filled) | SLS, MNSO, CPB, WMG, RMIX | -363 | -344 | **-352 bp** |
| Nov 20 (-3.3%) | gapdown.py / gd_regime | SKM, INTU, ERAS, UMC, ASX | -412 | -488 | |
| | conditioning | SKM, ERAS, UMC, ASX, AXTI | -454 | -430 | |
| | live (WOLF unfilled) | SKM, AEHR, AXTI, TTMI | -936 | -487 | **-507 bp** |
| Nov 23 (-0.3%) | gapdown.py | CAPR, MNRO, SLDE, DHT, UNH | +282 | -201 | |
| | gd_regime | MNRO, SLDE, DHT, UNH, CRWD | +4 | -43 | |
| | conditioning | MNRO, SLDE, DHT, CVS, VOD | +24 | -85 | |
| | live (MNRO, VOD unfilled) | CAPR, TU | +353 | -551 | **-551 bp** |

### Why the lists differ: code-level causes, in order of impact

1. **The live gap uses the close of the first 1-minute bar, not the open.**
   - `gapdown_live.py` sets `first = pairs[0][1]`, which is the spark endpoint's *close* of the 9:30-9:31 bar. The
     backtest uses the daily bar's official open.
   - The live file's gaps match the first-minute close to the cent (for example SLS -12.92%, AEHR -7.70%).
   - In one minute the ranking reshuffles:
     - Nov 20: AEHR, TTMI and WOLF were ranks 23, 48 and 20 by the open, but 2, 5 and 4 live, because they kept
       falling in the first minute. INTU, ERAS and UMC (ranks 2-4) bounced and dropped out.
     - Nov 19: RMIX (open gap -1.1%) and CLRO (+0.7%) were not even gap-downs at the open.
   - In effect the live rule selects names that are still falling after the open, which is short-term momentum, the
     opposite of the reversal being bet on.
   - In the 5-minute replica (section 3) the live and backtest top-5 lists share only 2.5 names on average.
2. **The live picker uses a dividend-adjusted previous close.** The backtest uses the raw close.
   - Yahoo's `chartPreviousClose` is dividend-adjusted. SLDE went ex $1.57 and DHT ex $1.45 on Nov 23. Their
     "gaps" of -6.1% and -5.2% are dividend drops: adjusted, they are -0.2% and +1.6%.
   - The backtest bought both of them. Here the live picker is right and the backtest is wrong.
   - Fixing it historically (variant vi) slightly *raises* the backtest, from +62 to +65 bp.
3. **The live picker drops names with no 9:30 one-minute bar** (`t0 != "09:30"`).
   - DHT's first bar on Nov 23 was at 9:31.
   - In a replay of yesterday's scan, 18 names opened late and 10 were missing from the spark response.
4. **The $95 cap and unfilled limit orders.**
   - The cap removed WSM, INTU, UNH and CRWD.
   - Limit orders went unfilled for WOLF, MNRO and VOD (3 of 14 attempted).
   - The holdings exclusion did not bind on these days.
5. **CAPR.** The live picker keeps the -74% collapse, and so does `gapdown.py`. gd_regime drops it.
   - Open->close from the $1.75 open was +9.6%.
   - Live bought at $1.905 at 9:45, was stopped at $1.75 at 10:03, and the low was $1.70.

## 2. Backtest under live constraints

Setup:
- Sample: 436 cash days (SPY below its 200-day SMA at t-1), 2018-04 to 2026-11-23.
- Trade: equal-weight top 5, net 10 bp.
- Stats: t on the daily series (one observation per day).
- Halves: H1 = 2018-21, H2 = 2022-26.
- DSR: 149 family trials and the SR variance from the gd_regime grid.

| Variant | Days | Mean bp/day | t | Hit | H1 bp (t) | H2 bp (t) | DSR |
|---|---|---|---|---|---|---|---|
| B0 pilot backtest (no cap) | 436 | +61.6 | 3.10 | 52.5% | +55 (1.6) | +65 (2.7) | 0.90 |
| (i) open <= $95 | 434 | +62.3 | 3.21 | 53.2% | +61 (1.8) | +63 (2.7) | 0.92 |
| (ii) open >= $5 | 436 | +59.9 | 2.93 | 52.5% | +45 (1.2) | +67 (2.8) | 0.85 |
| (iii) gap > -30% | 436 | +55.0 | 2.83 | 52.1% | +52 (1.5) | +57 (2.4) | 0.83 |
| (iv) -8% stop on daily low | 436 | +47.7 | 2.65 | 47.7% | +57 (1.9) | +43 (1.9) | 0.78 |
| (iv-b) stop + 20% TP (stop first) | 436 | +45.9 | 2.62 | 47.5% | +59 (1.9) | +40 (1.8) | 0.77 |
| (vi) dividend-adjusted gap | 435 | +64.6 | 3.23 | 53.3% | +56 (1.6) | +69 (2.8) | 0.92 |
| (vii) drop look-ahead \|o->c\| < 60% filter | 436 | +61.6 | 3.10 | 52.5% | identical: the filter never binds | | 0.90 |
| **(v) all combined (i-iv-b, vi, vii)** | 433 | **+53.8** | **3.10** | **50.6%** | +68 (2.4) | +47 (2.2) | **0.90** |
| (v-a) combined, stop but no TP | 433 | +54.6 | 3.10 | 50.6% | +64 (2.3) | +50 (2.3) | 0.90 |

What the table shows:
- **The $95 cap does not hurt.** (i) is exactly the conditioning study's +62 bp.
- **The collapse filter and the stop cost the most.** Gaps beyond -30% are worth about 7 bp/day. The -8% stop costs
  about 14 bp/day, and it is the stop that pushes the hit rate below 50%.
- **In combination the filters partly offset each other.** Once the collapses are removed, the stop costs less.
- **The combined result is concentrated.** Median +4 bp. The top 10 days give 55% of the total. The 2%-trimmed
  mean is +41 bp.
- **The pilot days in the combined variant** were -126, -467 and -26 bp. The dividend events in the cache predate
  Nov 23, so SLDE and DHT are still in the Nov 23 list.

### What daily data cannot test, and the drift estimates

Daily bars cannot test three things:
- the 9:45 entry;
- ranking on the first-minute price;
- fill failures.

Intraday data on disk:

| Source | Sample | Daily open -> entry drift of the picks | Open->close | Entry->close |
|---|---|---|---|---|
| 5m, every day (no cash day in the window, Aug 26-Nov 18) | 60 days | +29 bp to 9:45 (t 1.0) | +27 bp | -1 bp |
| 1h, every day, Dec 2023- | 727 days | +22 bp to 10:30 (t 2.2) | +30 bp | +7 bp |
| 1h, cash days, Dec 2023- | 55 days | +11 bp to 10:30 (t 0.3) | basket +62 bp net (t 1.1) | basket +52 bp net from 10:30 (t 1.3) |

How to read the drift estimates:
- **The picks rise between the open and the entry.** That is the reversal starting before a 9:45 buyer can get
  in. On all days it takes most of the open->close return.
- **On cash days the drift is smaller and noisy.** It is about 10 bp by 10:30.
- **Earlier work agrees.** The 5-minute entry-timing study found -17 bp from 9:30 to 9:45, and the pilot
  pre-registration assumed -13 bp.
- **Central estimate of the cost of a 9:45 entry: 10-30 bp.** That takes the combined +54 bp down to about
  +25-45 bp. Scaled the same way, t falls to about 1.4-2.5.
- **Feed check.** On the 1h data the daily open equals the intraday feed's first print (median difference 0), so
  the opening price itself is not a data artifact. One-minute bars can differ: INTU on Nov 20 had a daily open of
  $284.06 against a 1m open of $293.76.

**Live-rule replica on the 5m window** (60 non-cash days; first 5-minute close as the ranking price; 9:45 entry with
stop and take-profit; out at 15:45):

| Picks and entry | bp/day |
|---|---|
| Backtest picks, open->close | +11 (t 0.2) |
| Backtest picks, 9:45 with stops | -24 (t -0.7) |
| Live picks, 9:45 with stops | +3 (t 0.1) |

These are not cash days, so this window says nothing about the edge itself.

## 3. Verdict against STANDARDS.md

**Variants counted:** 10 backtest variants, plus 4 entry-time diagnostics, so 14 are logged under family
`gapdown_reconcile`. Together with the earlier family trials, the DSR uses 149 trials.

| Hurdle | 9:30 open (not tradeable, rule 8) | As traded (9:45 entry, estimated) |
|---|---|---|
| t > 3, or a prior plus t > 2 | 3.10 passes; Nagel's prior also applies | about 1.4-2.5: fails t > 3; passes prior + t > 2 only at the optimistic end |
| DSR > 0.9 | 0.899: fails narrowly | well below 0.9: fails |
| Same sign in both halves | yes (+68 / +47) | probably, but unmeasurable before 2024 |

The edge estimate is also inflated by three things not yet counted:
- survivorship: in the agent-xsec-swing check, names with long histories give +48 bp at t 2.4;
- opening spreads wider than 10 bp: at 30 bp, t falls to 2.4;
- unfilled limit orders: 3 of 14.

**Rejected as a tradeable rule.** The three losing days are themselves unremarkable:
- With a hit rate near 50%, three losses in a row happen about 12% of the time.
- The backtest's own picks also lost on Nov 19-20, when SPY fell about 3% from the open.

**Recommendations, if the pilot continues as a measurement:**
1. Make the live picker rank on the official open, meaning the 9:30 bar's **open**, not its close, so that it
   trades the backtested list.
2. Keep the dividend-adjusted previous close, and fix the backtest to match.
3. Add the gap > -30% filter.
4. Treat the expected edge as about 25-45 bp, not 62 bp.
5. Keep the pre-registered stop criteria, which are a cumulative loss over $100 and a review at 20 days.
