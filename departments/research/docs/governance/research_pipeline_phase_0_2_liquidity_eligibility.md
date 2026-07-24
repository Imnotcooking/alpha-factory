# Phase 0.2: Liquidity Eligibility

## Position In The Pipeline

Phase 0.2 runs after the dataset has been materialized and fingerprinted, but
before factor ranking, sleeve construction, routing, or optimization.

1. Phase 0.1: freeze the exact dataset fingerprint.
2. Phase 0.1B: materialize API data into immutable local research snapshots.
3. **Phase 0.2: decide which observations are liquid enough for the frozen
   capital and execution assumptions.**
4. Phase 0.3: construct or reduce the universe among products that passed Phase
   0.2.
5. Later phases: factor IC, strategy construction, routing, costs, robustness,
   and promotion.

Liquidity eligibility is a hard tradability gate. It is not an alpha factor,
router, risk overlay, or rule for choosing the "best" products.

## Core Rule

For equities and futures, daily traded notional is:

```text
equity traded notional = price x volume
futures traded notional = price x volume x contract multiplier
```

The engine uses the median over the previous 20 completed sessions and requires
at least 15 valid observations. Session `t` therefore does not use session
`t`'s final volume. This also prevents minute-level research from seeing the
rest of the current trading day.

The required daily traded notional is:

```text
max(
    taxonomy static floor,
    capital x maximum product weight / maximum daily volume participation
)
```

The default maximum product weight is 5% and the default daily participation
limit is 1%. With CNY 10 million of capital:

```text
CNY 10m x 5% / 1% = CNY 50m
```

Thus a product must have at least CNY 50 million of trailing median daily
traded notional before the strategy assumes that a full 5% target can be
implemented without exceeding 1% of normal daily trading activity.

For futures, the engine also requires sufficient open-interest notional:

```text
open-interest notional = price x open interest x contract multiplier
required OI notional = capital x maximum product weight / maximum OI share
```

The default maximum share is 2%. At CNY 10 million capital, the default OI
notional floor is CNY 25 million.

These are capacity assumptions, not claims that 1% participation or 2% of open
interest is universally optimal. They are explicit parameters that must be
stress-tested later.

## Asset-Aware Inputs

| Asset family | Primary inputs | Default eligibility logic |
|---|---|---|
| US, China, HK equities | price, share volume | Prior-session trailing median traded notional; multiplier is 1. |
| China futures | price, contract volume, open interest | Traded and OI notional use the product multiplier from `InstrumentMaster`. |
| US futures | price, contract volume, open interest, supplied multiplier | Same futures rule; the dataset must provide a valid multiplier until a US futures registry is added. |
| US and China options | volume, open interest, bid, ask, mark/close | Nonzero volume and OI, executable mark, and quoted spread no wider than 25%; settlement proxies remain explicitly labelled and configurable. |
| FX and crypto | price, volume | Capacity-based traded notional with multiplier 1; venue-specific refinements remain pending. |

Current equity taxonomy floors remain CNY 50 million for China equities, USD 1
million for US equities, and HKD 5 million for HK equities. The engine uses the
larger of that static floor and the capital-based requirement.

## Intraday Handling

Minute bars are aggregated into completed trading sessions first. Per-bar
volume is summed by default; datasets reporting cumulative session volume must
override the aggregation mode to `last`. Every bar in the next session receives
the same eligibility state calculated from prior sessions.

The session key is selected in this order:

1. `trading_day`
2. `economic_day`
3. `session_date`
4. normalized `date` or `datetime`

Chinese futures night-session datasets should therefore supply `trading_day`
or `economic_day`; calendar-date normalization is only the fallback.

## Row Preservation And Reason Codes

The engine never deletes a row. It adds `liquidity_eligible` and an explicit
`liquidity_reason_code`, then forces execution targets to zero only after the
alpha measurements have been formed on eligible observations.

Main reason codes include:

| Code | Meaning |
|---|---|
| `eligible` | All frozen policy tests passed. |
| `warmup_insufficient_history` | Fewer than 15 prior valid sessions exist. |
| `missing_required_market_data` | Price or volume is unavailable. |
| `missing_contract_multiplier` | Futures notional cannot be calculated safely. |
| `below_traded_notional_floor` | Trailing median traded notional is below the required capacity floor. |
| `missing_open_interest_history` | Futures OI history is incomplete. |
| `below_open_interest_notional_floor` | Futures OI capacity is too small. |
| `below_option_volume_floor` | Option contract volume is below policy. |
| `below_option_open_interest_floor` | Option open interest is below policy. |
| `option_spread_too_wide` | Quoted bid-ask spread exceeds policy. |

Keeping the row is essential. If a held product becomes ineligible, the
backtest must still observe the row and generate a flat target so the position
can be exited. Silently removing the row can make a stale position disappear
without a trade or cost.

## Interaction With Factors And Strategies

The shared factor-portfolio composer masks ineligible raw factor observations
before cross-sectional normalization. This prevents an untradable product from
changing the ranks or z-scores of tradable products.

After factor blending, sleeve routing, and risk overlays, the final liquidity
gate forces these execution columns to zero when present:

```text
routed_target_weight
final_target_weight
target_weight
signal
```

The ungated values are retained in `pre_liquidity_*` audit columns. Policy
parameters, the policy SHA-256 fingerprint, eligibility rate, reason counts,
and forced-flat counts are stored in DataFrame attributes and the run assumption
manifest.

Strategy recipes may override policy parameters explicitly:

```yaml
execution:
  max_weight_per_asset: 0.05

liquidity:
  lookback_sessions: 20
  min_observations: 15
  decision_lag_sessions: 1
  max_daily_volume_participation: 0.01
  max_open_interest_share: 0.02
```

Changing any assumption changes the policy fingerprint, so results using
different capacity rules are not treated as the same statistical trial.

## What Passing The Gate Does Not Prove

- It does not prove that the order can be filled at the modeled slippage.
- It does not replace contract-level roll and main-contract validation.
- Liquidity measured on a continuous or index proxy remains proxy evidence.
- It does not model order-book depth, limit queues, intraday volume curves, or
  market impact beyond the participation constraint.
- It does not justify removing more products to improve Sharpe. That is Phase
  0.3 and must be tested without selecting the universe on final performance.

The policy therefore establishes a reproducible minimum bar for research. It
does not certify production capacity by itself.
