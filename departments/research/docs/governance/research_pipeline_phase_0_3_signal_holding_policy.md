# Phase 0.3: Signal Frequency And Holding Policy

## Position In The Pipeline

This phase runs after dataset identity and liquidity eligibility are frozen, but
before factor IC, strategy optimization, or router comparison.

1. Phase 0.1: dataset fingerprint.
2. Phase 0.1B: immutable API materialization.
3. Phase 0.2: asset-aware liquidity eligibility.
4. **Phase 0.3: signal decision frequency and holding policy.**
5. Later phases: universe construction, costs, factor evaluation, strategy
   construction, routing, optimization, and promotion.

## The Main Distinction

Signal frequency is a rule established before the backtest. Trade count is an
outcome observed after the backtest.

For example, a daily-close trend factor may have approximately 250 decision
opportunities in one year. If its state changes only 12 times, it may produce
12 entries or reversals. That does not make it a "12-times-per-year signal."
Its signal frequency remains daily close; 12 is the realized event count.

Likewise, a declared holding rule is not the same as the average holding period
measured afterward. A Chandelier-exit trend factor may hold one trade for three
sessions and another for 40 sessions. Its frozen holding mode is
`factor_managed`; the three- and 40-session outcomes are realized diagnostics.

## Five Separate Clocks

| Clock | Question |
|---|---|
| Data frequency | How often does the dataset contain a new observation? |
| Signal decision frequency | On which observations may the strategy accept a new signal? |
| Execution lag | When can an accepted signal first trade? |
| Holding policy | What happens to the target after entry and before the next accepted decision? |
| IC return horizon | Which future return is used to test whether the signal predicts anything? |

These clocks may differ. A strategy can consume one-minute data, accept a
signal only after each completed 15-minute bar, trade on the next one-minute
bar, maintain the position until an exit event, and evaluate the signal against
a separately declared return horizon.

## Supported Decision Schedules

The shared temporal engine supports:

| Schedule | Meaning |
|---|---|
| `every_bar` | Every completed input bar is a decision opportunity. |
| `session_close` | Only the final observation of each product session is accepted. |
| `fixed_interval` | Accept every N bars, N sessions, or the close of each N-minute bucket. |
| `event_driven` | Accept rows marked by an event column, or explicit target changes when no event column is supplied. |

For daily data, `session_close` normally means one accepted decision per product
per trading day. For intraday data, it means the final bar associated with the
declared `trading_day`, `economic_day`, or calendar session.

## Supported Holding Modes

### `until_next_decision`

Accept the candidate target on a scheduled decision row and carry it forward
until the next scheduled decision. A zero candidate exits unless
`zero_signal_action: hold` is explicitly declared.

This is appropriate for ordinary daily cross-sectional scores and periodically
rebalanced portfolios.

### `fixed_period`

Accept the candidate target and expire it after a declared maximum number of
bars, minutes, or sessions if no earlier scheduled decision replaces it.

This is appropriate for hypotheses such as "hold a reversal trade for five
sessions." The numeric holding period must be declared explicitly.

### `factor_managed`

The factor already emits a persistent position state and owns its exit logic.
The shared layer does not manufacture an additional exit.

This is appropriate for EMA state, Donchian breakout, Chandelier stop, and
similar trend factors. Their exit parameters remain part of the frozen factor
parameter fingerprint. Realized holding duration is expected to vary.

### `session_flat`

Carry an intraday target only within its current session and reset before the
next session. This is appropriate for strategies that must not hold overnight.

## Current Daily Research Defaults

For an ordinary daily factor using the risk-desk allocator:

```text
data frequency       = daily
decision frequency   = session close
execution lag        = next open, from the factor contract
holding mode         = until next decision
zero signal action   = exit
IC horizon           = separately declared return_horizon
```

For a direct stateful trend factor:

```text
decision frequency   = session close
execution lag        = next open
holding mode         = factor managed
exit rule             = owned by the frozen factor definition and parameters
realized holding time = measured after execution, not selected in advance
```

These are defaults, not universal claims about the best frequency. A strategy
recipe may override them, and the override creates a different policy
fingerprint and therefore a different research trial.

## Causal Enforcement

The engine preserves the original candidate in `pre_temporal_*` and adds:

| Field | Meaning |
|---|---|
| `signal_decision_row` | Whether this row was allowed to accept a new signal. |
| `signal_decision_reason` | Scheduled decision, between decisions, or liquidity blocked. |
| `signal_holding_age` | Bars, minutes, or sessions since the most recent accepted decision. |
| `signal_target_changed_by_temporal_policy` | Whether scheduling or expiry changed the raw candidate. |

The temporally effective target is synchronized across the equivalent strategy
execution columns. The liquidity gate then runs last.

If liquidity fails while a target is being carried, the target is flattened and
the carry state is reset. It cannot silently reappear before another valid
scheduled decision.

## Realized Diagnostics

The policy summary records:

- scheduled decision rows and decision rate;
- actual target changes;
- entries, exits, and reversals;
- active target rows;
- number, mean, median, and maximum length of realized active runs.

These diagnostics answer how the frozen policy behaved. They do not redefine
the policy after seeing the results.

## Strategy Declaration

```yaml
return_horizon: close_signal_next_open_to_next_open

temporal:
  signal_frequency: session_close
  decision_interval: 1
  decision_unit: sessions
  holding_mode: until_next_decision
  holding_unit: sessions
  zero_signal_action: exit
```

A fixed five-session hypothesis would instead declare:

```yaml
temporal:
  signal_frequency: session_close
  decision_interval: 5
  decision_unit: sessions
  holding_mode: fixed_period
  holding_period: 5
  holding_unit: sessions
  zero_signal_action: exit
```

Signal frequency, holding policy, execution lag, and IC horizon are written to
the run assumption manifest. The temporal-policy SHA-256 fingerprint is also
included in the statistical trial signature, preventing two different timing
rules from being treated as the same experiment.

## What This Phase Does Not Choose

- It does not optimize the best frequency or holding period.
- It does not add stop-loss or take-profit rules.
- It does not assume that lower turnover means better alpha.
- It does not infer holding rules from profitable historical trades.
- It does not make a factor-managed exit rule comparable to a fixed-period exit
  unless both are tested as separately frozen hypotheses.

Those comparisons belong to later robustness or optimization phases, using
walk-forward data and multiple-testing controls.
