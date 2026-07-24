# Research Component Taxonomy

The research library is assembled from independently testable components. A
file belongs to the factor registry only when it produces a causal expected-
return direction or ranking. Everything that answers a different question is
stored and tested separately.

## Factor Families

| Family | Economic question | Typical examples |
|---|---|---|
| Trend / momentum | Will the current direction persist? | Time-series momentum, breakouts, moving-average and order-flow continuation |
| Mean reversion | Will a recent displacement reverse? | Residual shock, range exhaustion, velocity fade |
| Value / carry | Is the contract cheap or expensive relative to a causal anchor? | Curve carry, basis, roll yield, inventory-adjusted value |
| Relative value | Is one product mispriced against peers or a hedge basket? | Sector residuals, spreads, cointegration |
| Flow / positioning | Does participant positioning predict future returns? | Open-interest or volume-flow alpha with an explicit forward-return hypothesis |
| Volatility alpha | Does volatility itself predict a tradable return direction or premium? | Volatility risk premium or dispersion trades, not descriptive volatility states |

Subfamilies retain the economic mechanism, such as `ker_momentum`,
`sector_residual_shock_reversal`, or `tick_rule_order_flow_imbalance`.

## Non-Factor Components

| Component | Question answered | Registry |
|---|---|---|
| Market state | What environment are we in? | `departments/research/routers/states/` |
| Router | Which frozen sleeve receives capital? | `departments/research/routers/` |
| Position policy | How much exposure should each position receive? | `departments/research/position_policies/` |
| Diagnostic | What market mechanism or data property are we measuring? | `departments/research/diagnostics/` |
| Risk overlay | Should completed strategy exposure be reduced? | `departments/research/strategy_overlays/` |
| Strategy | Which components are assembled together? | `departments/research/strategies/` |
| Execution | How are targets translated into trades and costs? | Shared backtesting and execution engine |

Volatility scaling, leverage, contract sizing, concentration limits, liquidity
caps, stop policies, and drawdown throttles are not separate alpha factors.
Market volatility, breadth, correlation, shock age, and liquidity conditions
are states unless a frozen test demonstrates that the measure directly predicts
returns without relying on a downstream routing decision.

## Duplicate-Removal Rule

Two files enter the same duplicate contest only when market, frequency,
evaluation geometry, execution mode, lag, return assumption, and portfolio
layer agree. Within that cohort:

1. Compare raw alpha-score correlation.
2. Compare standardized target correlation and active-position overlap.
3. Compare IC level, sign, stability, and product coverage.
4. Compare net Sharpe, turnover, costs, drawdown, and holdout stability under a
   common execution and risk specification.
5. Retain the clearest pure-alpha implementation when economic and empirical
   evidence are otherwise similar.
6. Archive redundant files with their evidence and read-only aliases; never
   silently delete research history or reuse their IDs.

A numeric ID collision is only an identity defect. It is resolved before this
process and does not itself justify deleting either formula.
