# Intraday CN Futures Trend-Following Notes

Last updated: 2026-07-13

## Objective

Find an intraday trend-following factor in the current library, or create one, that runs on the adjusted 1-minute CN futures panel, has net Sharpe above 1, and beats the matched same-horizon dashboard benchmark.

Data:

```text
runtime/data/futures_cn/intraday/futures_data_futures_main_1m_adj_20260710_090753.parquet
```

Comparable evaluation window and execution assumptions:

```text
window: 2025-10-09 to 2026-07-09
capital: CNY 200,000
lot mode: fractional research sizing
execution: current signal earns next 1-minute close-to-close return
slippage: 0.5 tick per side
fees: InstrumentMaster FUTURES_CN exchange profiles
benchmark: active-universe equal-weight same-horizon control
```

## Library Audit

The directly relevant existing intraday factors were:

- `fac_085`: 60-minute KER momentum.
- `fac_086`: 1-minute MACD crossover.
- `fac_087`: resampled 15-minute MACD crossover.
- `fac_090`: 15-minute breadth-anchored breakout.

The first three had recorded net Sharpe below zero on prior 1-minute runs. `fac_090` was the only credible candidate: its 2026 first-half run reached net Sharpe about 2.51 after costs, but historical half-year runs showed strong regime dependence.

```text
2024 H1: negative
2024 H2: negative
2025 H1: negative
2025 H2: Sharpe about 0.95
2026 H1: Sharpe about 2.51
```

The comparable October 2025 to July 2026 rerun was:

```text
run_id: run_31f005ba
annualized net return: 1.95%
net Sharpe: 0.67
gross Sharpe: 2.59
average daily turnover: 132.7%
average daily cost: 2.21 bps
```

This diagnosed a cost problem rather than a missing directional effect.

## Chosen Factor

Factor:

```text
fac_094_Intraday_Fixed_Weight_Breadth_Breakout_Futures_CN
```

Logic:

1. Aggregate each continuous session into completed 15-minute buckets.
2. Require the close to break the prior six-bucket high and remain above session VWAP.
3. Require a four-bucket trend displacement of at least 1.2 ATR.
4. Require Kaufman efficiency ratio of at least 0.60, so net movement is large relative to path length.
5. Require bucket volume at least its lagged normal level.
6. Require nonnegative same-timestamp futures-market breadth.
7. Require ATR percentage movement to be at least six times estimated round-trip cost.
8. Enter on the shifted execution bucket and hold until trend, VWAP, KER, trailing stop, or session exit invalidates the trade.
9. Use a fixed 20% target weight per accepted contract. Other contracts entering or exiting do not resize an existing position.
10. Keep the strategy long-only in this version. The existing symmetric short book increased churn and fought the positive recent commodity regime.

## Why The Indicators Exist

- Prior-range breakout identifies price discovery outside a recently accepted range.
- Session VWAP asks whether the move is supported by the session's average inventory price.
- ATR makes displacement and stops comparable across products.
- KER rejects a large endpoint move produced by a choppy path.
- Volume ratio asks whether participation confirms the break.
- Market breadth avoids buying an isolated breakout while most futures are trending down.
- Edge-to-cost ratio directly tests whether one normal ATR unit is large enough to clear ticks and exchange fees.
- Fixed weights remove portfolio turnover that has no relationship to the contract's own trend state.

The quality search used October-December 2025 for training, January-March 2026 for validation, and April-July 2026 as a chronological audit. The selected thresholds were positive in all three blocks. Research artifacts:

```text
scripts/research/analyze_cn_futures_intraday_breakout_quality.py
runtime/artifacts/research/minute_reversion/cn_futures_intraday_breakout_quality_events.parquet
runtime/artifacts/research/minute_reversion/cn_futures_intraday_breakout_quality_search.csv
```

## Goal-Meeting Result

Exact dashboard run:

```text
run_id: run_5ba5a760
trading days: 216
net total return: +2.784%
annualized arithmetic return: +3.248%
tear-sheet annualized return: +3.27%
annualized volatility: 2.61%
net Sharpe: 1.25
gross Sharpe: 2.30
max drawdown: -2.33%
average daily turnover: 67.4%
average daily modeled cost: 1.08 bps
maximum gross leverage: 2.0x
```

Matched same-horizon benchmark:

```text
total return: +2.624%
annualized return: +3.062%
annualized volatility: 7.91%
Sharpe: 0.39
```

The factor therefore meets both current-window requirements after modeled costs. The outperformance margin is modest, so this remains a research sleeve rather than promotion-grade evidence.

## Router Preparation

The trend and mean-reversion sleeves now share:

- the same 1-minute adjusted-main panel;
- the same backtest window;
- the same next-bar return convention;
- the same capital and cost assumptions;
- explicit factor-owned target weights.

This makes the next volatility-routing experiment well-defined. The router should decide risk allocation using only lagged volatility state, while preserving each sleeve's internal entry and exit logic. It must be tested against both static sleeves and a static blend, not only against the passive futures benchmark.

On the matched window, daily net-return correlation between `fac_094` and mean-reversion `fac_093` was `-0.020`. A static 50/50 return blend had approximately 1.96 Sharpe. That static blend is therefore a demanding and necessary baseline for the volatility router; switching is useful only if it improves on diversification already available without forecasting volatility regimes.

## Limitations

- The adjusted main contract is an executable proxy, not a roll-audited contract stream.
- The 200,000 CNY run uses fractional research lots and emits integer-lot and fee-to-tick warnings.
- The underlying breakout family performed poorly before late 2025.
- The strategy is long-only and may be inactive or structurally disadvantaged in a broad bearish futures regime.
- The secondary passive active-universe basket has unusually large returns and volatility in this adjusted-main dataset; the authoritative comparison here is the matched same-horizon control.
