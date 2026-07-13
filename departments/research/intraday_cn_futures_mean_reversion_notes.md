# Intraday CN Futures Mean-Reversion Notes

Last updated: 2026-07-10

## Research Question

Can we build a cost-aware intraday mean-reversion factor for Chinese futures that beats the relevant benchmark, while keeping the "why" legible enough for the research dashboard?

Current candidate: `fac_044_Relative_Velocity_Fade`.

## Internet Inspiration

The useful public research points away from generic oversold/overbought indicators and toward market microstructure:

1. Order-flow imbalance matters more than raw volume.
   Cont, Kukanov, and Stoikov show that short-horizon price changes are strongly linked to order-flow imbalance at the best bid/ask, with depth changing the impact slope.
   Source: https://arxiv.org/abs/1011.6402

2. For Chinese CSI 300 index futures, order-flow imbalance can be modeled as a shock with mean-reverting memory, and forecast horizon matters.
   Source: https://arxiv.org/abs/2505.17388

3. For Chinese stock-index futures calendar spreads, the liquid front contract leads other maturities, but large lead-lag spread deviations create negative feedback on the leader. That is a contrarian, dislocation-fade idea rather than a vanilla momentum idea.
   Source: https://arxiv.org/abs/2501.03171

4. Chinese high-frequency technical rules can look significant before costs and fail after costs. This is the warning label on simple MA/KDJ/Bollinger recipes.
   Source: https://arxiv.org/abs/1710.07470

5. Intraday spreads in Chinese markets have strong time-of-day structure, especially early-session relaxation. Any intraday signal should avoid pretending all ticks are equally cheap or equally informative.
   Source: https://arxiv.org/abs/0710.2402

6. Chinese price-limit research is mixed for direct reversal. Limit-hit studies find cooling-off behavior but also next-day continuation after limit hits, especially down-limit events. So the factor should not blindly fade limit pressure.
   Source: https://arxiv.org/abs/1503.03548

## Chosen Factor

`fac_044_Relative_Velocity_Fade` fades extreme short-window tick velocity.

Default rule:

```text
fast_move_ticks = mid_price_now - mid_price_6_ticks_ago, measured in inferred ticks
threshold = rolling 99th percentile of abs(fast_move_ticks) over 5,000 ticks
event = abs(fast_move_ticks) >= threshold and abs(fast_move_ticks) >= 2 ticks
quality = abs(fast_move_ticks) / threshold >= 1.10
direction = fade the fast move
hold = 85 ticks
cooldown = 12,000 ticks after accepted events
execution = shift signal one tick before trading
```

Why this factor:

- It is a microstructure dislocation factor, not a generic oscillator. The trigger is an unusually fast local price displacement relative to that contract's own recent tick distribution.
- It uses a rolling percentile rather than a fixed price move. AU, AG, soda ash, PTA, and fuel oil do not share the same tick economics or volatility, so the threshold must adapt.
- The 1.10 threshold-ratio filter makes the event exceed the percentile line by a margin. That is a cost-control choice: near-threshold events are common and easy to overtrade.
- The 85-tick hold is the observed snap-back window for AU in prior tests. Shorter holds left money on the table; much longer holds diluted the reversal into noise.
- The 12,000-tick cooldown treats clustered bursts as one research decision. This is the main defense against transaction costs.
- The factor uses bid/ask mid-price and inferred tick size rather than last-price returns alone, because the edge is about a fast local displacement in the executable book.

## Latest Local Evidence

Run on 2026-07-10:

```text
run_id: run_4a35ea3b
data: runtime/data/futures_cn/tick/8contract_raw_20260608_20260612_tick_all_data_OOS.parquet
selected: au2608
rows: 291,373 ticks
raw events: 1,173
quality events: 620
throttled events: 27
active ticks: 2,295
capital profile: small_personal_futures_cn, 200,000 CNY
execution: C++ futures engine, fractional lots, explicit exchange fees, 0.5 tick slippage per side
benchmark: active futures contract buy-and-hold, with Nanhua as secondary when available
```

Result:

```text
trading days: 5
annualized return: +0.40%
annualized vol: 0.08%
Sharpe: 4.79
max drawdown: approximately 0.00%
average daily turnover: 54.0%
average daily cost: 0.18 bps
holdout IC: +0.0406
holdout hit rate: 73.66%
```

Daily net returns were small but positive on average. The active-contract benchmark was negative over most of the same dates, so this run beat the benchmark. The sample is only five trading days, so this is encouraging evidence, not promotion-grade proof.

Prior AU calibration on the wider 2026-05-18 to 2026-06-12 file found the same default parameter family was the best current run:

```text
run_id: run_5e81de1d
trading days: 23
annualized return: +0.0476%
Sharpe: +0.4464
holdout IC: +0.0073
holdout hit rate: 56.90%
events: 5,145 raw -> 2,836 quality -> 117 throttled
```

## What To Learn From This

The factor seems to work only when it is sparse. The signal is not "price moved up, sell it." It is closer to:

```text
The local book just moved unusually far, unusually fast, relative to its own recent tick distribution. If follow-through does not immediately dominate, the price often partially mean-reverts. But costs are high enough that we only want the cleanest bursts.
```

The key indicators each serve a reason:

- `rtv_fast_move_ticks`: measures the shock.
- `rtv_threshold_ticks`: asks whether the shock is unusual for this instrument and session.
- `rtv_threshold_ratio`: demands excess over the threshold to pay for costs.
- `hold_ticks`: encodes the empirical decay horizon.
- `cooldown_ticks`: collapses burst clusters so we do not pay fees repeatedly for the same event.
- `forward_return`: tests the actual holding horizon, not a generic next-bar label.

## Next Tests Before Promotion

1. Run the same settings on a later AU tick window not used in the June calibration.
2. Sweep hold values around 82 to 88 ticks and confirm the result is not a single-point accident.
3. Add session diagnostics: day session versus night session versus open/close windows.
4. Re-run with harsher slippage, especially 1.0 tick per side.
5. Test other products with product-specific cost filters. Prior broad sweeps showed the default AU settings do not transfer cleanly to higher-fee or different-microstructure products.

## 1-Minute Main-Contract Follow-Up

After review, the minute-level research target is better represented by the adjusted 1-minute main-contract panel:

```text
runtime/data/futures_cn/intraday/futures_data_futures_main_1m_adj_20260710_090753.parquet
```

This file covers 85 main-contract symbols from 2024-01-02 to 2026-07-09. The current backtest route classifies it as an executable proxy rather than fully executable contract data because main-contract rolling and roll liquidity still need audit.

Two minute-level candidates were tested on 2026-04-01 to 2026-07-09:

### fac_089 Keltner Exhaustion Reentry

Logic:

```text
1. Aggregate 1-minute bars into 15-minute decision buckets by contract/session.
2. Build a Keltner channel: EMA(close, 20) +/- 2.0 * EMA(true_range, 14).
3. Do not fade the first band break.
4. Enter only after price re-enters the channel from an outside-band excursion.
5. Require the prior bucket to show elevated range and elevated volume.
6. Map the active state back to the 1-minute execution grid.
```

Recent-slice result:

```text
run_id: run_ecde98a5
trading days: 78
annualized return: -9.23%
Sharpe: -2.87
max drawdown: -3.42%
average daily turnover: 181.5%
average daily cost: 3.83 bps
holdout IC: +0.0008
holdout hit rate: 49.06%
```

Read: the idea is too weak or too busy at default settings. Cost drag dominates.

### fac_090 1-Minute Relative Velocity Fade

This adapts the tick-level idea to minute bars.

Logic:

```text
fast_move = close_t - close_(t-5min)
fast_move_atr = fast_move / EWMA(true_range, 30)
threshold = rolling 99.9th percentile of abs(fast_move_atr), shifted by one bar
event = abs(fast_move_atr) >= 3.5
        and abs(fast_move_atr) / threshold >= 1.5
        and volume / rolling_median(volume, 480) >= 1.5
direction = fade the fast move
hold = 5 minutes
cooldown = 720 minutes per contract/session
target gross = 10%
max weight per contract = 3%
```

Ultra-sparse result:

```text
run_id: run_6134d792
trading days: 78
annualized return: -0.95%
Sharpe: -3.61
max drawdown: -0.34%
average daily turnover: 8.6%
average daily cost: 0.20 bps
holdout IC: +0.0012
holdout hit rate: 57.14%
trades: 117
trade win rate: 46.15%
```

Read: throttling fixed the turnover problem, but the event payoff is still negative on average. The minute-level fade has a plausible hit-rate signature but does not yet beat the benchmark or costs. This should stay in research, not promotion.

Next minute-level research step: calibrate the minute fade with a walk-forward objective that directly penalizes daily turnover and daily cost, then split by product family. Broad cross-asset minute mean reversion is probably too heterogeneous for one global threshold.

## Extended 1-Minute Research Trail: Mechanism Before Parameters

The broad minute-velocity result motivated a falsifiable sequence of event studies. The selection period was split into 2024 training and January to September 2025 validation. October 2025 onward was treated as a later regime audit. Earlier baseline work had already touched parts of the full sample, so these labels are research splits rather than a claim of pristine, never-observed data.

### 1. Outright Shock Plus Confirmation

Script:

```text
scripts/research/analyze_cn_futures_minute_reversion.py
```

Definition:

```text
shock at t-1 = five-minute contract return / lagged one-minute volatility
confirmation at t = first one-minute return against the shock
trade return = fade direction from t over 1, 3, 5, 10, 15, or 30 minutes
cost = contract exchange fees + 1.25 round-trip ticks
```

Development evidence from 2024-01-01 through 2026-03-31:

```text
confirmed events: 233,326
gross reversal at 5 minutes: -0.65 bps/event
estimated net reversal at 5 minutes: -5.33 bps/event
gross reversal at 30 minutes: -1.66 bps/event
```

Why it failed: the first opposing minute was usually a pause inside a continuing move, not proof that the information or inventory shock had finished.

### 2. Sector-Residual Shock

The next test removed the contemporaneous sector median return before fading. This asks whether a contract moved without confirmation from related products.

```text
events: 258,452
gross reversal at 3 to 10 minutes: about +0.7 bps/event
median estimated round-trip cost: about 4.4 bps/event
```

Why it was rejected: the relative-value effect existed, but its gross size was too small for the product mix. A statistically visible effect is not automatically a tradable factor.

### 3. Session-Opening Gap Reversal

Scripts:

```text
scripts/research/analyze_cn_futures_session_open_reversal.py
scripts/research/search_cn_futures_session_open_reversal.py
```

The literature and the data both suggested that a larger dislocation might exist at the session open. The test measured each day, night, and afternoon session gap against the contract's trailing 75th-percentile gap, entered at a fixed minute, and exited at the session close.

A frozen development rule emerged:

```text
family: oilseeds and vegetable oils
open: day session
decision: minute 5
gap size: 1.0 <= abs(gap_z) < 3.0
exit: session close
2024 training net mean: +8.64 bps/event
2025 Jan-Sep validation net mean: +12.79 bps/event
positive selection quarters: 6 of 7
```

It did not survive later data:

```text
2025 Q4: -6.32 bps/event
2026 Q1: +5.54 bps/event
2026 Q2: -2.09 bps/event
```

Why it was rejected: the mechanism was conditional and unstable. Chinese commodity opening imbalances can continue when they contain real overnight information; an opening gap is not automatically stale inventory.

### 4. Economic Pair Reversion

Scripts:

```text
scripts/research/analyze_cn_futures_pair_reversion.py
scripts/research/search_cn_futures_pair_reversion.py
scripts/research/evaluate_cn_futures_trailing_edge_gate.py
```

Eleven pairs were declared before selection: protein meals, vegetable oils, steel, copper, equity-index curves, government-bond curves, precious metals, and crude/bitumen. Each pair used inverse-volatility leg weights, a five-minute spread shock, lagged spread volatility, and two-leg transaction costs.

The 10-year/30-year bond pair was the only immediate-entry specification positive in both early windows, but its edge was tiny:

```text
rule: abs(shock_z) >= 2.5, hold 60 minutes
2024 train: +0.44 bps/event
2025 Jan-Sep validation: +0.24 bps/event
2025 Q4 to 2026 Q1 audit: -0.62 bps/event
```

It was rejected after the audit.

Gold/silver extreme events showed a large recent convergence effect, but the first event study over-counted overlapping bars inside the same dislocation cluster. The first actual non-overlapping dashboard factor exposed this error:

```text
run_b8cff725: -0.28% annualized, Sharpe -0.35
```

This led to two execution corrections:

1. Mark an extreme at `abs(z) >= 4`, but do not enter immediately.
2. Enter only after the same-sign spread retraces inside `abs(z) = 3`.
3. Freeze the inverse-volatility leg weights at entry.
4. Permit no overlapping entry and reset at each session boundary.

### 5. Failed-Auction Rejection

Script:

```text
scripts/research/analyze_cn_futures_failed_auction.py
```

This test required a one-minute bar to break the prior 20-minute high or low and close back inside the old range. Even this visually strong rejection pattern had negative gross payoff in aggregate:

```text
selection events: 264,390
gross mean at 5 minutes: -0.12 bps/event
estimated net mean at 5 minutes: -4.41 bps/event
```

Why it was rejected: a wick is evidence that one auction failed, but it does not imply the broader directional order flow has ended.

## Current Factor: fac_091 Precious-Metals Relative-Value Fade

Factor:

```text
departments/research/factors/daily_signals/
fac_091_Intraday_Precious_Metals_Relative_Value_Fade_Futures_CN.py
```

Final current-regime logic:

```text
pair = SHFE gold main contract / SHFE silver main contract
fast leg returns = five-minute close-to-close log returns
leg volatility = lagged 240-minute standard deviation
leg weights = inverse-volatility risk balance
spread shock = w_au * return_au - w_ag * return_ag
shock_z = spread shock / (lagged spread sigma * sqrt(5))
regime gate = lagged 2,400-minute au/ag return correlation >= 0.75
mark extreme = abs(shock_z) >= 4.0
enter = same-sign retracement to abs(shock_z) <= 3.0 within 30 minutes
position = short rich leg, long cheap leg
weights = frozen at entry
hold = at most 60 one-minute bars and never across a detected session gap
cooldown = 60 bars
execution = next bar
default gross = 3.5x, chosen to volatility-match the dashboard control
```

Why each indicator exists:

- Five-minute residual: long enough to exceed one-bar noise, short enough to represent a local liquidity mismatch.
- Inverse-volatility weights: compare risk contributions, not raw contract returns or prices with different units.
- Lagged spread sigma: asks whether the relative move is unusual for this pair using only information available before entry.
- `abs(z) >= 4`: ordinary pair noise was not large enough to clear two-leg costs.
- Reentry at `abs(z) <= 3`: avoids fading the first print while the spread is still widening.
- Correlation at least `0.75`: a relative-value trade needs a currently coherent pair; otherwise the divergence can be a genuine repricing of gold versus silver.
- Frozen weights: prevents minute-by-minute hedge-ratio drift from manufacturing turnover.
- Sixty-minute TTL: convergence in the surviving events occurred over tens of minutes, not the next bar.
- 3.5x gross: the unscaled factor ran at only 0.45% annual volatility versus 8.20% for the control. The chosen gross produced 7.85% annual volatility and remained below the benchmark risk in the winning run.

Current-regime dashboard result:

```text
run_id: run_41f956af
window: 2025-10-09 to 2026-07-09
trading days: 216
net total return: +10.30%
annualized return: +12.12%
annualized volatility: 7.85%
Sharpe: 1.50
max drawdown: -3.57%
average daily turnover: 230.1%
average daily modeled cost: 1.14 bps
executed-return p-value: 0.0842
```

Matched same-horizon dashboard control:

```text
total return: +7.57%
annualized return: +8.89%
annualized volatility: 8.20%
Sharpe: 1.08
```

Runner holdout beginning 2026-03-21:

```text
factor total return: +6.10%
same-horizon control: -0.67%
passive active-universe basket: -2.58%
```

This reaches the current-regime benchmark objective after modeled costs and at slightly lower volatility.

## Important Negative Stress Result

The same frozen 3.5x specification was run on 2024-01-02 through 2025-09-30:

```text
run_id: run_8b17b633
annualized return: -2.43%
Sharpe: -1.43
max drawdown: -4.81%
```

All losses occurred during two active correlation episodes in 2024 Q2 and Q3; the gate then kept the strategy inactive through September 2025. A one-month correlation estimator reduced current performance and did not replace the 2,400-minute gate.

Conclusion: `fac_091` is an explainable current-regime research factor that beat the benchmark in the recent dashboard window. It is not yet a timeless or promotion-grade factor. Production promotion requires a genuinely new forward sample, executable-contract roll validation, integer-lot tests, and a leverage/margin review. The older stress failure must remain visible rather than being optimized away.

## Whole-Universe 1-Minute Indicator Research

The next objective was stricter: start from the 85-symbol adjusted 1-minute main-contract panel, remove very low-liquidity contracts, use familiar indicators without treating them as magic, and require both benchmark outperformance and net Sharpe above 1.

### Indicator roles

The tested interpretation was:

- Bollinger z-score measures a price displacement relative to recent 5-minute variation.
- RSI and KDJ describe whether directional pressure is still fully aligned. In the event study, extreme confirmation often meant continuation, so they became vetoes rather than entry votes.
- Volume distinguishes a quiet liquidity overshoot from a move accompanied by unusually strong information flow.
- ATR normalizes risk across contracts with different price and volatility units.
- Kaufman efficiency ratio and Bollinger-center slope distinguish a noisy range from an efficient trend.
- Lagged 20-day median notional rank removes the low-liquidity tail before any signal is accepted.

The broad prototype was `fac_092_Intraday_Liquid_Universe_Indicator_Reentry_Futures_CN`. It aggregated the 1-minute panel into completed 5-minute decision bars, entered on a mild Bollinger reentry, and mapped the state back to next-minute execution.

### What failed and why

Several exact runs exposed implementation and research mistakes:

```text
run_c3ebdc48: Sharpe -0.88; cross-contract resizing created excessive turnover
run_2b8dc502: Sharpe -2.18; event timing and session-end events did not match the study
run_189cfb3d: Sharpe +0.60; corrected 120-minute session horizon
run_fd1786de: Sharpe +0.72; inverse-ATR sizing and broader confirmation rule
run_064fbcbe: Sharpe +0.80; removed an unsupported post-exit cooldown
```

The most important audit found that the Polars event-study join reordered rows around midnight before forward shifts. Some alleged 180-minute exit timestamps pointed backward into the prior evening. Adding an explicit `sort(["symbol", "datetime"])` after the join removed the false edge. The corrected train/validation search produced no standalone Bollinger specification near the target; its best minimum selection Sharpe was about `0.24`.

That result changed the conclusion: Bollinger reentry is a modest diversifying sleeve in the recent regime, not a standalone whole-universe alpha with Sharpe above 1.

Other whole-universe hypotheses were rejected:

- Opening-hour cross-sectional fades continued rather than reverted, both before and after lunch.
- Night-return to day-return reversal appeared in 2024 but reversed sign after 2025, even with lagged edge gates.
- The overnight-plus-first-half-hour move did not profitably reverse during the final half hour in this panel.

These tests are reproducible in:

```text
scripts/research/analyze_cn_futures_indicator_reentry.py
scripts/research/search_cn_futures_indicator_portfolio.py
scripts/research/diagnose_cn_futures_indicator_factor_alignment.py
scripts/research/analyze_cn_futures_opening_cross_sectional_reversal.py
scripts/research/analyze_cn_futures_night_day_reversal.py
```

The timing tests were motivated by published evidence that Chinese crude-oil night returns can negatively predict day returns and that broader Chinese commodity intraday effects vary by liquidity and market-maker behavior:

- https://doi.org/10.1016/j.econmod.2021.01.005
- https://doi.org/10.1016/j.pacfin.2024.102534

### Current goal-reaching factor: fac_093

`fac_093_Intraday_Multi_Sleeve_Mean_Reversion_Futures_CN` combines:

1. The broad indicator sleeve: top-20% lagged-notional universe, 1.5 to 1.75 Bollinger reentry, volume ratio at most 1.2, no more than two continuation flags, inverse-ATR event sizing, and a 120-minute hold.
2. The precious-metals relative-value sleeve from `fac_091`: an extreme gold/silver five-minute spread shock, high lagged correlation, confirmed same-sign retracement, and a 60-minute hold.

Why combine them: their recent daily net-return correlation was only `0.088`. The broad sleeve alone had net Sharpe `0.80`; the pair sleeve had net Sharpe `1.50`. The composite uses a 10x broad-signal multiplier and 0.5x pair multiplier to make their recent volatility contributions comparable, while remaining below the 4x gross research cap. This is portfolio diversification between two mean-reversion mechanisms, not a claim that every indicator independently predicts returns.

Exact current-window dashboard result:

```text
run_id: run_ee854132
window: 2025-10-09 to 2026-07-09
trading days: 216
net total return: +7.60%
annualized net return: +8.86% (tear-sheet compounded display: +9.09%)
annualized volatility: 5.64%
net Sharpe: 1.57 (tear-sheet display: 1.61)
max drawdown: -2.72%
average daily turnover: 272.1%
average daily modeled cost: 1.60 bps
maximum realized gross leverage: 2.75x
```

Matched same-horizon benchmark:

```text
total return: +6.33%
annualized return: +7.39%
annualized volatility: 12.22%
Sharpe: 0.60
```

This reaches the requested current-window benchmark and Sharpe target after modeled costs. It remains private research because the winning relative-value sleeve failed the older 2024 to September 2025 stress window, the main-contract file is an executable proxy rather than a roll-audited contract stream, and integer-lot feasibility is not established for the 200,000 CNY capital profile.
