"""Log this program's experiments to the research registry (run ONCE after the experiment scripts)."""
import sys, json
sys.path.insert(0, "tools/research")
from registry import log  # noqa: E402

A = "agent-overnight"
D = "Yahoo 1d full history (SPY 1993-, QQQ 1999-, IWM 2000-, SMH/XLF 1999/2000-, 3x funds 2008/2010-), ^IRX, ^VIX, ^VIX3M, HYG, IEF; dividends"
C = "tools/research/overnight/"
S = json.load(open("trader/data/overnight/exp_signals.json"))
K = json.load(open("trader/data/overnight/exp_calendar.json"))
Z = json.load(open("trader/data/overnight/exp_sizing.json"))
E = json.load(open("trader/data/overnight/exp_assets.json"))
V = json.load(open("trader/data/overnight/exp_validate.json"))
F = json.load(open("trader/data/overnight/final_spec.json"))


def row(rows, name):
    return next(r for r in rows if r["name"] == name)


log("overnight", "3x fund overnight tracking: dividend add-back and financing carry (UPRO/TQQQ/TNA/SOXL/FAS vs 3x ETF)",
    "A 3x fund's overnight return = 3 x the ETF's overnight return; the leverage financing is charged somewhere",
    "Yahoo 1d 2008/2010-2026 for the funds and ETFs, ^IRX", "fund r_on - 3 x ETF r_on (both dividend-corrected), "
    "regressed on per-night T-bill rate; split by weekday", 0,
    {"carry_slope_UPRO": 2.22, "carry_slope_TQQQ": 2.40, "carry_slope_TNA": 2.63, "intercept_bp_yr": 0.0,
     "exdiv_night_gap_bp_if_raw_UPRO": 128.5, "fri_morning_drag_x": 2.5},
    "adopted", "Overnight holder bears ~2.4 x T-bill per year on the fund dollars (charged per trading night, weekend "
    "accrual on Thursday->Friday nights); the fund does NOT drop on the ETF's ex-dividend night, so backtests from raw "
    "ETF opens understate by 3 x the dividend on those nights. Cost model now in core.py (CARRY_A=2.4). Measurement, "
    "not a strategy variant.", C + "tracking.py", A)

log("overnight", "Current spec re-measured: dividend-corrected opens + carry; history extended to 2001-2026",
    "The live spec's 10-year numbers hold with realistic costs and on earlier data",
    D, "spec as live (200d SMA, 1.5%/sigma20, cap 3, thirds); 4bp + carry", 3,
    {"replicated_SR_main_convention_2017_26": 0.99, "realistic_SR_2017_26": 0.89, "realistic_CAGR_2017_26": 0.163,
     "SR_2008_26": 0.73, "SR_2001_26": 0.75, "CAGR_2001_26": 0.121, "worst_night_2001_26": -0.1082},
    "adopted", "Measurement. Reported 0.92 not exactly reproduced (0.99 with raw opens/no carry; implementation detail). "
    "Worst night in the longer history: 2016-06-24 (Brexit) -10.8% of the sleeve at L=3 on all legs = a $108 loss, "
    "breaching the $100 per-trade limit; with real funds and whole shares -$110.", C + "core.py", A)

b = V["boot"]
log("overnight", "(a) Validation of the live spec: stationary bootstrap CIs, CSCV PBO, White RC / Hansen SPA, deflated Sharpe",
    "The live spec's edge is real and its parameter choice is not overfit", D,
    "Politis-Romano bootstrap (b=5/20/60, 4000 draws); CSCV S=16 (12,870 splits) on a 96-variant spec grid "
    "(SMA 100-250 x sigma 10-60 x target 1-2% x cap 2/3); SPA vs spec over all program variants; DSR with 210 trials",
    96, {"SR_OOS_2008_26": b["OOS 2008-2026|b20"]["sr"], "SR_CI_lo": b["OOS 2008-2026|b20"]["sr_ci"][0],
         "SR_CI_hi": b["OOS 2008-2026|b20"]["sr_ci"][2], "CAGR_CI_lo": b["OOS 2008-2026|b20"]["cagr_ci"][0],
         "PBO_spec_grid": V["cscv_spec_family"]["pbo"], "PBO_program": V["cscv_my_family"]["pbo"],
         "SPA_p_any_beats_spec": V["spa_vs_spec"]["p_consistent"], "RC_p": V["spa_vs_spec"]["p_rc"],
         "SPA_p_family_vs_cash": V["spa_family_vs_cash"]["p_consistent"], "DSR_spec": V["dsr"]["spec (OOS 2008-26)"]["p"]},
    "adopted", "Spec is real (CI excludes 0, DSR 0.98, family beats cash p<0.001, PBO 0.12) but no variant beats it "
    "after the family-wide SPA (p 0.27). Spec ranks 38/96 in its own grid; grid averages favour cap 2 over 3 and longer "
    "SMA (250 > 200), the latter not tested out of sample.", C + "exp_validate.py", A)

ts = row(S["rows"], "tsmom_majority|replace")
log("overnight", "(b) Regime signals vs 200-day filter: TSMOM 21/63/126/252, Kalman LLT slope, 2-state HMM, VIX/VIX3M, VIX vs MA, HYG/IEF",
    "A better regime filter raises out-of-sample Sharpe", D,
    "replace or AND the 200d filter; HMM (Baum-Welch, filtered, Jan-1 refits) and Kalman MLE walk-forward; OOS 2008-2026; "
    "meta-rule picks best-so-far each Jan 1", 30,
    {"best": "tsmom_majority", "best_SR": ts["sharpe"], "spec_SR": ts["bench_sr"], "best_dSR": ts["d_sr"],
     "best_p": ts["p_dsr"], "meta_SR": S["meta"]["sharpe"], "SPA_p_signals": V["spa_signals"]["p_consistent"]},
    "rejected", "TSMOM majority vote: SR 0.80 vs 0.73 OOS (p 0.15), both halves >=, but WORSE pre-sample 2003-07 "
    "(0.77 vs 0.98) -> promising only. HMM, VIX term structure, VIX<MA, credit and Kalman all below the 200-day filter. "
    "Choosing among them each year (meta) gave SR 0.62.", C + "exp_signals.py", A)

fo = K["desc"]["fomc_day"]
log("overnight", "(c) Calendar/event nights: weekday, weekend, holiday, pre-holiday, turn of month, opex, jobs, CPI, FOMC",
    "Overnight returns differ on calendar/event nights; skipping or sizing up helps", D + "; event dates approximate "
    "(BLS jobs rule; FOMC 2000-26 and CPI 2018-25 from memory; official sites unreachable)",
    "Welch t flagged vs other trend-on nights; skip / x1.5 rules, direction chosen on 2001-07, judged 2008-26", 28,
    {"fomc_pre_bp": fo["pre 2001-07"][0], "fomc_pre_t": fo["pre 2001-07"][2], "fomc_oos_bp": fo["OOS 2008-26"][0],
     "fomc_oos_t": fo["OOS 2008-26"][2], "best_rule_dSR": max(r["d_sr"] for r in K["rows"]),
     "SPA_p_calendar": V["spa_calendar"]["p_consistent"]},
    "rejected", "No rule adopted (best pre-chosen: skip post-opex nights dSR +0.05 p 0.19). FOMC-morning nights earn "
    "~+15-19bp vs ~+5bp (pre-FOMC drift, both halves positive) and jobs/CPI nights are also above average: do NOT skip "
    "event nights. Sizing them up adds ~nothing because L is usually near its cap.", C + "exp_calendar.py", A)

c2 = row(Z["rows"], "spec cap 2.0 per leg")
log("overnight", "(d) Sizing: EWMA/GARCH(1,1)/VIX vol forecasts, inverse variance, Bayesian fractional Kelly, per-leg caps, stress guard",
    "Better vol forecasts or Kelly sizing raise Sharpe within the $100 worst-night limit", D,
    "walk-forward GARCH MLE and VIX scaling (Jan-1 refits); QLIKE + Diebold-Mariano; Kelly mu = expanding posterior "
    "N(0, tau^2); caps 1.5-2.5; point-in-time stress guard", 18,
    {"garch_qlike_DM_t_SPY": Z["qlike"]["SPY|garch11"][1], "garch_SR": row(Z["rows"], "vol=garch11")["sharpe"],
     "kelly_half_SR": row(Z["rows"], "kelly frac 0.5 tau 3bp")["sharpe"],
     "kelly_half_worst": row(Z["rows"], "kelly frac 0.5 tau 3bp")["worst"], "cap2_SR": c2["sharpe"],
     "cap2_dSR_p": c2["p_dsr"], "cap2_worst": c2["worst"], "SPA_p_sizing": V["spa_sizing"]["p_consistent"]},
    "adopted", "ADOPT per-leg cap 2.0 + stress guard as a RISK-LIMIT fix: OOS SR 0.78 vs 0.73 (p 0.07; both halves "
    "higher; equal pre-sample), worst night -7.8% vs -10.8%, maxDD -24% vs -29%, CAGR 10.4% vs 12.3%. Not a significant "
    "alpha gain (sizing SPA p 0.45). GARCH/EWMA/VIX forecasts are better forecasts (DM t -2.5 to -5.3) but give no Sharpe "
    "gain; half-Kelly breaches the $100 limit (-12.3%); inverse variance worse.", C + "exp_sizing.py, final_spec.py", A)

s4 = row(E["rows"], "SMH legs, cap 2")
s5 = row(E["rows"], "SMH+XLF legs, cap 2")
log("overnight", "(e) Asset set: add SOXL (3x SMH) and/or FAS (3x XLF) legs, cap 3 and cap 2",
    "More legs diversify the overnight premium and raise Sharpe", D, "equal sleeve weights; same rules per leg; "
    "OOS 2008-26; real-fund re-run 2010-26; whole shares at today's prices", 6,
    {"SMH4_cap2_SR": s4["sharpe"], "SMH4_cap2_dSR_p": s4["p_dsr"], "SMH4_pre_SR": s4["pre_sr"],
     "SMH5_cap2_SR": s5["sharpe"], "SMH5_pre_SR": s5["pre_sr"], "SPA_p_assets": V["spa_assets"]["p_consistent"],
     "RC_p_assets": V["spa_assets"]["p_rc"]},
    "promising", "Adding SMH/SOXL: OOS SR 0.90 vs 0.73 (cap 2), real funds 2010-26 0.96 vs 0.80, within-question SPA "
    "p 0.03; but no gain 2003-2009 and the family-wide SPA is p 0.27; gains come from 2010-26 (semis' boom). XLF alone "
    "adds nothing. 5 legs fail whole shares at $1,000 (FAS $161 > per-leg target). Shadow-track SOXL as a 4th leg.",
    C + "exp_assets.py", A)

log("overnight", "Real-fund re-run of key variants (actual UPRO/TQQQ/TNA/SOXL/FAS overnight returns, 2010-2026)",
    "The 3x-index proxy with modelled carry matches real fund results", "Yahoo 1d 3x funds 2010-2026",
    "same signals; actual fund open/close with dividends; 4bp", 7,
    {"spec_real_SR": F["real"]["spec"]["sharpe"], "rec_real_SR": F["real"]["rec"]["sharpe"],
     "spec_real_worst": F["real"]["spec"]["worst"], "rec_real_worst": F["real"]["rec"]["worst"]},
    "adopted", "Proxy is slightly conservative (spec real SR 0.80 vs proxy 0.77 on the same nights). Conclusions unchanged.",
    C + "exp_realfunds.py", A)

log("overnight", "Whole-share feasibility on a $1,000 sleeve (nearest vs floor rounding at today's fund prices)",
    "Whole-share rounding does not destroy the edge", "real funds 2010-2026", "constant $1,000 sleeve", 8,
    {"spec_nearest_worst_usd": -110, "cap2_nearest_SR": 0.86, "cap2_nearest_worst_usd": -80, "smh4_nearest_SR": 0.99,
     "five_legs_floor_zero_leg_share": 0.21},
    "adopted", "3 legs and 4 legs (with SOXL) are fine with nearest rounding; 5 legs are not (FAS/UPRO too coarse).",
    C + "exp_wholeshares.py", A)
print("logged")
