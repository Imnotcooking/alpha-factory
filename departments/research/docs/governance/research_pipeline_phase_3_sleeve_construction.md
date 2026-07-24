# Research Pipeline Phase 3: Sleeve Construction

## Position in the pipeline

Phase 2 asks whether a pure factor predicts a causally available forward
return. Phase 3 asks a different question:

> What happens when the frozen factor score is converted into one explicit,
> feasible set of target positions?

A sleeve is the first portfolio layer. It owns ranking, selection, weighting,
holding, rebalancing, exposure limits, missing-signal treatment, and execution
delay. It does not change the factor formula, choose a market regime, apply a
router, add leverage, or optimise parameters.

## Frozen default sleeve

The first Phase 3 batch uses one deliberately simple default for every eligible
daily Chinese-futures cross-sectional factor:

| Setting | Frozen value |
|---|---|
| Geometry | Daily cross-sectional |
| Expression | Market-neutral long-short |
| Selection | Long top 20%; short bottom 20% of non-neutral scores |
| Within-leg weighting | Equal weight |
| Winsorisation | 1st and 99th cross-sectional percentiles |
| Rebalance | Every decision date |
| Holding period | One session |
| Target exposure | 100% gross; 0% net |
| Contract cap | 5% absolute target weight |
| Sector cap | Disabled for this batch |
| Missing score | Neutral target |
| Exact zero score | Neutral target; excluded from ranking |
| Minimum active cross-section | 10 products and at least two distinct score levels |
| Additional execution delay | Zero; the upstream panel is already causally aligned |
| Optimisation | Prohibited |

The sector cap is declared but disabled because the current instrument master
maps 53 of 88 products into one generic `Macro` category. Enforcing a cap on
that taxonomy would create artificial concentration control rather than a
credible sector constraint.

The contract cap is not followed by leverage rescaling. If too few selected
products can absorb the long and short budgets within the 5% limit, realized
gross remains below 100%. This makes cap binding visible instead of hiding it
with an implicit second sizing rule.

## Timing and cost translation

The frozen return horizon is:

```text
score observed after close t
entry at actual main-contract open t+1
exit at actual main-contract close t+1
```

The position is session-flat. Therefore every active contract incurs both an
opening and a same-day closing transaction, even when the next decision has the
same target. Daily turnover is:

```text
(entry notional + exit notional) / CNY 10,000,000 capital
```

Ideal weights are truncated toward zero to whole contracts. Fees use each
product's recorded fixed or notional-rate opening and same-day-close fields.
Slippage is 0.5 minimum price movement on entry and 0.5 on exit. Gross and net
contributions are:

```text
gross contribution = executed weight * next-open-to-close return
net contribution   = gross contribution - exchange fees - slippage
```

## First completed batch

The five Phase 1B daily Chinese-futures mean-reversion factors were evaluated
sequentially on the same 2,300 decision dates and 88-product frozen cohort. No
factor or sleeve parameter was changed after viewing Phase 2 or holdout results.

| Factor | Full gross mean | Full cost | Full net mean | Full net Sharpe | Holdout gross mean | Holdout net Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| `fac_001_ST_Reversal` | 0.94% | 11.36% | -10.43% | -1.57 | -1.97% | -1.94 |
| `fac_011_Regime_Filtered_Rev` | 2.00% | 10.97% | -8.97% | -1.54 | 0.26% | -1.76 |
| `fac_012_Capitulation_Fade` | 3.17% | 11.14% | -7.97% | -1.31 | 5.60% | -0.73 |
| `fac_014_Range_Amplified_CLV` | 2.41% | 10.16% | -7.75% | -1.52 | 6.03% | -0.57 |
| `fac_042_Bollinger_Binary` | 0.52% | 0.42% | 0.10% | 0.09 | 0.25% | -0.22 |

All figures are arithmetic annualized daily means except Sharpe. Full-sample
annualized turnover for the four continuous factors ranges from approximately
383x to 403x. Their ideal targets average about 97.8% gross; whole-contract
execution realizes approximately 76% to 80% gross. No active executed position
has a missing forward return.

## Interpretation

None of the five factors passes the fixed Phase 3 validation test. `fac_001` and
`fac_011` are weak at the gross-return level as well, so trading less cannot by
itself establish a strong strategy.

`fac_012` and `fac_014` contain a more useful research clue. Their holdout gross
annualized means are 5.60% and 6.03%, with gross Sharpes of 0.91 and 1.21, but
the session-flat implementation costs approximately 10.06% and 8.86% per year
in holdout. They are examples of positive gross predictive economics that do
not survive this trading implementation.

`fac_042` demonstrates why score semantics belong in the sleeve contract. It is
a sparse binary event factor: 89.4% of its valid scores are exactly zero. Those
zeros must remain flat rather than be forced into quantile positions through an
arbitrary ticker tie-break. After this rule is enforced, its annualized turnover
is 16.8x rather than hundreds of times. The full sample is slightly positive
after costs, but the holdout net mean is -0.27% with Sharpe -0.22. The result is
therefore a weak, unstable event hypothesis, not a validated sleeve.

This result does not authorize optimisation. Phase 4 subsequently applies the
predeclared standalone-economics gates and confirms that none of these five
sleeves is eligible for router research. Plausible later hypotheses include a
slower rebalance schedule, an event-triggered expression, or a longer
economically matched holding horizon. Each would be a new frozen sleeve
specification, not a retroactive improvement to this baseline.

## Reproducible artifacts

The implementation lives in:

```text
src/oqp/research/sleeves/contracts.py
src/oqp/research/sleeves/engine.py
src/oqp/research/sleeves/evidence.py
departments/research/strategies/sleeves/slv_001_Cross_Sectional_Quintile_Long_Short.py
scripts/research/build_default_sleeve_evidence.py
```

Each evidence bundle is written to:

```text
runtime/artifacts/research/sleeve_construction/
  <factor_id>/<market_vertical>/<sleeve_id>/
```

Factor Drilldown reads the saved daily path and summaries. It does not rerun the
factor or scan the raw market dataset.
