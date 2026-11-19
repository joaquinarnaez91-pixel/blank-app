"""One registry record per experiment of the cross-sectional overnight / swing program (agent-xsec-swing).
Run once: python3 tools/research/xsec_swing/log_all.py"""
import sys
sys.path.insert(0, "tools/research")
from registry import log

A = "agent-xsec-swing"
DATA = "Yahoo 1d 2016-11-21..2026-11-18, 3,195 names listed today (superset + names now < $5 or thin), PIT top-500 by 60-day $ volume, actual price >= $5; tests 2017-12-04..2026-11-18 (2,252 days)"
X = "tools/research/xsec_swing/"

log("xsec_overnight", "Overnight persistence (Aboody et al. 2018): top-N by past 5/10/15/20-night mean overnight return, close->open",
    "Stocks with high recent overnight returns keep earning them overnight", DATA,
    "N in {5,10,20,50} x lookback {5,10,15,20}; plus 4 SPY>200d-gated; 10 bp per night; daily-series t; DSR over 32 family trials",
    20, {"decile_spread_bp": 5.0, "decile_spread_t": 2.09, "top_decile_gross_bp": 9.0, "best_net_bp": 6.0, "best_SR": 0.36,
         "best_t": 1.07, "best_maxDD": -0.815}, "rejected",
    "Persistence exists but is small in liquid names (5 bp/night top-bottom); every long-only top-N is ~0 or negative after 10 bp.",
    X + "overnight_xsec.py", A)
log("xsec_overnight", "Tug of war (Lou, Polk & Skouras 2019): 12-1 momentum deciles, overnight vs intraday returns",
    "Momentum profits accrue overnight", DATA, "decile sorts each close, next-night and next-day returns, gross", 1,
    {"WML_overnight_bp": 10.0, "WML_overnight_t": 3.30, "WML_halves_t": "1.54/3.03", "WML_intraday_bp": -1.0,
     "WML_intraday_t": -0.24, "top_decile_overnight_bp": 12.4, "universe_overnight_bp": 5.4}, "info",
    "Replicates: winners beat losers overnight only. A long-short spread; long-only after costs is the next record.",
    X + "overnight_xsec.py", A)
log("xsec_overnight", "Long-only overnight momentum: top-N by 12-1 momentum or 12-month overnight return, close->open",
    "Hold momentum winners only overnight; beat the index overnight model?", DATA,
    "N in {5,10,20,50} x signal {12-1, 12m overnight}; plus 2 SPY>200d-gated; 10 bp per night; yearly walk-forward over all 32 xsec_overnight variants",
    10, {"best": "on12_top10", "net_bp": 9.3, "SR": 0.68, "t": 2.04, "DSR": 0.55, "maxDD": -0.676, "halves_SR": "0.79/0.58",
         "SR_cost15": 0.31, "SR_cost20": -0.05, "mom12_1_top5_SR": 0.58, "mom12_1_top5_SR_superset_only": 0.93,
         "WF_OOS_SR": 0.35, "index_model_SR": 0.89, "corr_index": 0.57, "net_bp_when_index_in_cash": 1.4, "combo_5050_SR": 0.84},
    "rejected",
    "Gross 16-20 bp/night but 10 bp per night eats it; survivorship (adding today's sub-$5 names) cut top-5 SR 0.93 -> 0.58. Never beats or improves the index model.",
    X + "overnight_xsec.py, combine.py", A)
log("xsec_overnight", "Overnight reversal: buy today's worst open->close losers at the close, sell at the next open",
    "Liquidity-provision bounce overnight", DATA, "N in {5,20}; 10 bp per night", 2,
    {"top5_net_bp": -3.3, "top5_SR": -0.20, "top20_net_bp": -6.2, "top20_SR": -0.59}, "rejected", "", X + "overnight_xsec.py", A)
log("overnight", "3x-fund overnight momentum rotation: UPRO/TQQQ/TNA/SOXL/FAS by 12-1 or 12-month overnight momentum",
    "A cheap long-only overnight momentum portfolio: rotate the live model's 3x exposure into the strongest index",
    "Yahoo 1d fund and underlying (SPY/QQQ/IWM/SMH/XLF) bars 2016-2026",
    "live rules per leg (200-day trend, 1.5% vol target, cap 3, no stress guard); equal fifths; top-K in {1,2,3} x 2 signals; 4 bp per night",
    7, {"live_thirds_SR": 0.89, "fifths_SR": 0.94, "fifths_minus_live_t": 0.06, "best_rotation_SR": 0.90,
        "WF_OOS_SR": 0.81, "live_same_period_SR": 0.86}, "rejected",
    "Momentum rotation adds nothing; adding SOXL/FAS legs is +0.05 SR, not significant. Reconstruction predates the cap-2.0/stress-guard revision.",
    X + "letf_rotation.py", A)
log("xsec_swing", "Connors RSI(2) and 1-week loser reversal swing trades (top-200/500 liquid, 10 slots)",
    "Short-term mean reversion in liquid large caps", DATA,
    "RSI2<10 or <5 & close>200d SMA, exit close>5d SMA or 5/10 days, or time exits 1/3/5 days; weekly losers (bottom 10 by 5-day return) hold 5 days, +-200d filter; close fills; 10 bp round trip; yearly walk-forward",
    10, {"canonical_SR": 0.37, "canonical_t": 1.10, "canonical_trades": 6014, "canonical_mean_pct": 0.13, "canonical_win": 0.63,
         "best": "rsi10_x5_h10", "best_SR": 0.48, "best_t": 1.44, "best_DSR": 0.58, "WF_OOS_SR": 0.17,
         "canonical_SR_superset_only": 0.60, "weekly_rev_SR": 0.43, "weekly_rev_SR_superset_only": 0.73, "corr_index": 0.30},
    "rejected", "Post-publication (2017-26) Connors edge is gone once fallen names are included; survivorship inflated Sharpe by 0.2-0.3.",
    X + "meanrev.py, swing.py", A)
log("xsec_swing", "Momentum hold: top-5/10 by 12-1 momentum, sell when rank > 2N (close to close)",
    "Holding winners through the day (no nightly round trip) vs overnight only", DATA, "10 bp round trip; fractional and $1,000 whole shares",
    2, {"top5_SR": 0.73, "top5_maxDD": -0.586, "top10_SR": 0.75, "top10_maxDD": -0.669, "top5_usd1000_SR": 0.67}, "rejected",
    "Drawdowns far beyond the $500 halt.", X + "momentum_hold.py", A)
log("xsec_swing", "PEAD proxy: buy high-volume gap-ups (volume >= 3x median, gap and day >= 3% or 6%), hold 20/60 days",
    "Post-announcement drift after earnings-like events", DATA,
    "events at the close, 10 slots, 10 bp; event study vs universe with month-clustered t. Earnings dates unavailable (SEC not on allowlist; Yahoo needs a session crumb)",
    4, {"best_SR": 0.55, "best_t": 1.63, "best_maxDD": -0.718, "event_excess_20d_pct": -0.41, "event_excess_60d_pct": 0.75,
        "event_t_max": 0.93}, "rejected", "No drift after high-volume gap-ups in liquid names, 2017-26.", X + "pead_proxy.py", A)
log("xsec_swing", "Reversal swing only while the index model holds cash (Nagel 2012), combined with the index model",
    "Liquidity provision pays more in turbulent markets; use idle sleeve dollars", DATA,
    "the 10 meanrev variants, new entries only when SPY, QQQ and IWM all close below their 200-day; leftover cash in the index model each night; yearly walk-forward",
    10, {"best": "weekly_rev_top200_above200d", "gated_t": 1.59, "gated_DSR": 0.77, "combined_SR_insample": 1.03,
         "index_alone_SR": 0.89, "WF_combined_OOS_SR": 0.87, "WF_index_same_SR": 0.86, "WF_increment_bp": 3.0,
         "WF_increment_t": 1.08, "usd1000_worst_trade_usd": -153}, "rejected",
    "Only 353 cash-regime closes; increment not significant out of sample; a single trade can break the $100 limit.",
    X + "regime.py", A)
log("xsec_data", "Survivorship-bias estimate for the 10-year daily panel",
    "Today's price/volume filter inflates backtests", DATA + "; RSP as a survivorship-free benchmark",
    "EW PIT top-500 (superset only vs all names listed today) vs RSP; strategy reruns with and without today's sub-$5/thin names",
    0, {"superset_minus_listed_pct_yr": 0.84, "superset_minus_listed_t": 1.53, "of_which_overnight": -0.15,
        "listed_minus_RSP_pct_yr": 1.58, "listed_minus_RSP_t": 0.65, "connors_SR_drop": 0.23, "weekly_rev_SR_drop": 0.30,
        "on_mom_top5_SR_drop": 0.35}, "info",
    "Universe-level bias ~1%/yr (intraday, concentrated 2021); much larger for concentrated speculative picks. Delisted names still missing.",
    X + "survivorship.py", A)
log("gaps_intraday", "Survivorship sensitivity of the cash-day gap-down reversal (5 largest gaps <= -2%, open->close, SPY < 200d)",
    "Does the rule survive adding names now below $5 and restricting to long histories?", DATA,
    "equal weight, 10 bp; candidate sets: superset raw, superset PIT, all listed PIT, long-history PIT, recent-listing PIT, top-500 PIT (robustness subsets)",
    6, {"superset_raw_bp": 144.4, "superset_pit_bp": 75.2, "superset_pit_t": 3.50, "listed_pit_bp": 78.8, "listed_pit_t": 3.51,
        "long_hist_bp": 48.3, "long_hist_t": 2.36, "recent_bp": 100.6, "recent_t": 4.14, "top500_bp": 29.2, "top500_t": 1.54,
        "listed_post_2023_12_21_bp": 2.4, "listed_post_t": 0.04}, "promising",
    "Adding fallen-but-listed names leaves it unchanged; long-history names earn about half (+48 bp, t 2.4); recent listings drive the rest, where unobservable delisting bias is largest. Without a PIT price filter the backtest is badly inflated. Post-2023 evidence ~0 for the broad set.",
    X + "gapdown_check.py", A)
print("logged")
