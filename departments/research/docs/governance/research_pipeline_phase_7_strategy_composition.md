# Phase 7: Strategy Construction

## Research question

Can frozen research components be assembled into an executable strategy without
changing their hypotheses, obscuring where positions came from, or
double-counting transaction costs?

## Position-producing core

The core type tells the strategy engine how the first target-position stream is
created. It is an execution geometry, not a factor family or market regime.

| Core type | Inputs | Position construction | Intended use |
| --- | --- | --- | --- |
| `direct_factor` | One legacy factor with embedded targets | Uses the factor-owned target after strategy-level limits | Compatibility for existing time-series rules; new pure factors should not use this path |
| `factor_sleeve` | One pure factor and one `slv_*` sleeve | The factor emits a score and the sleeve selects, signs, and weights assets | Standard single-factor cross-sectional strategy |
| `factor_blend` | Two or more pure factors | Compatible scores are normalized and combined before portfolio construction | Multi-factor strategies with one common execution geometry |
| `statistical_arbitrage` | One spread or basket component | Preserves declared leg ratios before strategy-level limits | Pairs, baskets, and other relative-value structures |
| `ml_predictive` | One registered fitted experiment and its stored out-of-sample predictions | Converts causal OOS predictions into direct targets or passes them through a sleeve | Supervised ML strategies with reproducible model evidence |
| `routed_components` | At least two complete position streams and one `rtr_*` router | Builds each branch independently, then allocates capital according to a causal router state | Conditional allocation across sleeves or heterogeneous strategies |

### Factor and sleeve boundary

The target architecture is strict:

- A **factor** emits a causal predictive score or event.
- A **sleeve** converts a score into asset selection, direction, holding, and
  weights.
- A **router** allocates capital among already constructed position streams.
- A **risk overlay** may reduce a completed target but may not create alpha.

The active factor registry still contains legacy recipes that embed targets or
holding rules. They remain executable so historical work is reproducible, but
the Factor Library labels them explicitly as `Embedded target`, `Signal +
legacy target`, or `Embedded holding rule`. New factor recipes must use the
`Pure signal` boundary. Migration of legacy target logic into reusable sleeves
or position policies is a separate controlled cleanup, not an automatic rewrite.

## Risk and allocation

The engine applies controls after the core has produced positions:

1. Apply a router when the core has multiple branches.
2. Apply declared `ovl_*` risk overlays in order.
3. Apply the per-contract notional concentration cap.
4. Apply the market-appropriate portfolio budget.
5. Convert final weights into executable lots.
6. Calculate fees and slippage once from final trades.

For Chinese futures, the primary portfolio budget is margin utilization rather
than a user-selected gross-leverage multiplier:

\[
\text{Margin utilization}_t
=\sum_i |w_{i,t}|\,m_i \leq 30\%,
\]

where \(w_{i,t}\) is contract \(i\)'s notional portfolio weight and \(m_i\) is
its configured margin rate in the Instrument Master. The default 30% limit
therefore implies a minimum 70% cash reserve. The cap is applied across the
whole portfolio after routing, overlays, temporal rules, and liquidity gates.
It is a ceiling, not a requirement to consume all 30%.

The configured margin schedule is currently a static versioned research input;
it does not reconstruct every historical exchange or broker margin change.
That limitation must be recorded when interpreting older backtests.

For markets whose portfolio budget is naturally expressed as gross exposure,
the allocator retains `max_gross_leverage`. The two controls are not treated as
interchangeable labels.

## Reproducible configuration

A strategy draft contains stable component references plus strategy-level
capital, allocation limits, and execution assumptions. A Chinese-futures
example is:

```yaml
strategy:
  strategy_id: str_cn_futures_research
  name: CN futures research strategy
  market_vertical: FUTURES_CN
  core:
    type: factor_sleeve
    branches:
      - branch_id: core
        factor_ids: [fac_example]
        sleeve_id: slv_001_Cross_Sectional_Quintile_Long_Short
        execution_mode: risk_desk
  risk_overlays: []
  allocator:
    max_gross_leverage: null
    max_contract_weight: 0.10
    max_margin_utilization: 0.30
  execution:
    capital: 10000000
    capital_currency: CNY
    transaction_cost_profile: cn_futures_broker_v1
    slippage_ticks_per_side: 0.5
```

Parameters that define factor signals, sleeve selection rules, router
thresholds, or overlay behavior belong to their component artifacts. A frozen
strategy references those artifacts rather than silently overriding them.

## Evidence artifacts

A completed run must record the strategy fingerprint, component IDs and
fingerprints, transformation diagnostics, margin utilization, final positions,
gross and net returns, turnover, fees, slippage, and the exact transaction-cost
profile. Hypothetical component-level costs are not consumed by the final
strategy; costs are calculated once after final positions are known.
