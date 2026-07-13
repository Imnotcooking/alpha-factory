# Matched Long-Short Momentum/Reversal Router

## Purpose

The first volatility-router test used a long-only breakout and a composite mean-reversion sleeve. That
was not an apples-to-apples replication of *Momentum, Market Volatility, and Reversal*. This experiment
constructs matched cross-sectional long-short portfolios so that the paper's mechanism receives a fairer
test in intraday Chinese futures.

## Frozen primary specification

- Source: one-minute adjusted-main CN futures panel, aggregated once to a cached 15-minute panel.
- Compact cache: 1,065,746 contract-bars; the 15,986,190-row minute parquet is not reopened on normal runs.
- Universe: top 40 contracts by lagged 20-day median notional turnover.
- Decision grid: synchronized, non-overlapping wall-clock decisions.
- Momentum formation: prior eight traded 15-minute bars, skipping the latest bar.
- Reversal formation: negative of the latest 15-minute return.
- Score: formation return divided by lagged 64-bar volatility, then cross-sectionally demeaned.
- Portfolio: long top quintile and short bottom quintile, +0.5/-0.5 target weights and 1.0 gross.
- Roll control: the complete momentum formation window must have zero roll flags.
- Execution: next-bar returns, fixed weights through the hold, no execution across a three-hour session gap.
- Cost: 0.5 tick per side plus the InstrumentMaster open/close fee schedule.
- Regime: trailing 252-day futures-market volatility quartiles; Q1-Q3 momentum and Q4 reversal on the
  next observed day.
- Development: first routable day through 2025-09-30.
- Holdout: 2025-10-09 through 2026-07-09.

Decision targets were verified to have maximum absolute net exposure below `2.3e-8` and gross exposure
between `1.0` and `1.00000005`. Missing exchange-specific marks do not trigger portfolio resizing; each
contract is marked at its next available bar while entry weights remain frozen.

## Implementation corrections

Three zero/invalid results were rejected before economic interpretation:

1. A 30-minute session gap split the day at lunch and made the momentum lookback impossible. The valid
   boundary is three hours: exchange breaks remain connected, while night/day blocks remain separate.
2. Per-contract modulo decisions were asynchronous and never produced a 20-contract cross-section.
   Decisions now use synchronized wall-clock buckets.
3. Restricting momentum formation to one session made the 120-minute hold infeasible. Momentum now uses
   the last eight traded bars across session breaks, while execution still stops at session boundaries.

These are geometry corrections, not return-based parameter choices.

## Holdout results

### Primary 60-minute hold

| Portfolio | Gross annual | Net annual | Sharpe | Average daily turnover | Daily cost |
|---|---:|---:|---:|---:|---:|
| Momentum | -1.08% | -25.37% | -3.38 | 5.17x | 9.65 bps |
| Reversal | 8.76% | -19.65% | -2.85 | 6.13x | 11.27 bps |
| Paper router | 1.19% | -24.08% | -3.42 | 5.45x | 10.03 bps |
| Static 50/50 | 3.84% | -22.51% | -4.36 | 5.65x | 10.46 bps |

The hourly cross-sectional rebalance is not executable after realistic futures costs.

### Slower 120-minute hold

| Portfolio | Gross annual | Net annual | Sharpe | Max drawdown | Daily turnover | Daily cost |
|---|---:|---:|---:|---:|---:|---:|
| Momentum | -1.86% | -9.59% | -1.55 | -12.50% | 1.69x | 3.07 bps |
| Reversal | 7.25% | -0.31% | -0.06 | -4.10% | 1.69x | 3.00 bps |
| Paper router | 4.16% | -3.54% | -0.68 | -5.94% | 1.69x | 3.06 bps |
| Static 50/50 | 2.70% | -4.95% | -1.31 | -7.39% | 1.69x | 3.04 bps |

Slowing the clock preserves much more of the reversal edge, but it does not create a momentum premium.
A 180-minute hold was not treated as a valid robustness result: the Chinese day block cannot support the
formation history, a broad 20-contract cross-section, and a complete 180-minute execution window.

## Quartile mechanism at 120 minutes

Holdout values are average net basis points per day.

| Quartile | Days | Momentum | Reversal | Reversal minus momentum | Spread t-stat |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | -3.88 | -1.72 | 2.16 | 0.49 |
| Q2 | 37 | 1.43 | 4.00 | 2.57 | 0.44 |
| Q3 | 51 | -1.72 | -0.53 | 1.19 | 0.24 |
| Q4 | 72 | -7.91 | -0.71 | 7.20 | 0.79 |

The mechanism moves in the paper's direction: the reversal advantage is largest in Q4. But the required
sign switch is absent because reversal also beats momentum in Q1-Q3. None of the spreads is significant.
The literal router therefore gives up reversal returns on three quarters of the state space.

## Regime instability

At 120 minutes, development gross annual returns were `+1.34%` for momentum and `-3.33%` for reversal.
In holdout they changed to `-1.86%` and `+7.25%`, respectively. The volatility quartile did not identify
this broad strategy-level rotation. This suggests that the important state was not simply volatility
magnitude, or that the sample is too short to estimate it reliably.

## Decision

The first matched strategy family does not support a publishable claim that the paper router beats both
component strategies. The 120-minute result is still informative:

- matching the portfolios makes Q4 look more reversal-friendly;
- costs are a first-order constraint for intraday cross-sectional futures portfolios;
- the proposed 120-minute momentum horizon is too short to produce continuation;
- routing cannot rescue a component with negative gross expected return.

The next paper-valid test should keep the classifier frozen and replace only the slow sleeve's formation
horizon. Pre-declared candidates are one, three, and five trading sessions, rebalanced once per day or
session, with the same top/bottom quintiles and matched volatility. This tests the paper's essential
slow-momentum versus fast-reversal separation without searching volatility thresholds. A router should
be evaluated only after both sleeves show positive gross edge in at least one stable conditional region.

## Artifacts

- Script: `scripts/research/analyze_cn_futures_matched_momentum_reversal_router.py`
- Cached bars: `runtime/artifacts/research/matched_momentum_reversal_router/cn_futures_adjusted_main_15m_bars.parquet`
- Primary daily results: `runtime/artifacts/research/matched_momentum_reversal_router/cn_futures_matched_momentum_reversal_daily.csv`
- 120-minute daily results: `runtime/artifacts/research/matched_momentum_reversal_router/cn_futures_matched_momentum_reversal_daily_120m_hold.csv`
- Primary summary: `runtime/artifacts/research/matched_momentum_reversal_router/cn_futures_matched_momentum_reversal_summary.json`
- 120-minute summary: `runtime/artifacts/research/matched_momentum_reversal_router/cn_futures_matched_momentum_reversal_summary_120m_hold.json`
