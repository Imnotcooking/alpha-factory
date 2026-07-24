# Research Pipeline Phase 5: Conditional Behaviour

## Position in the pipeline

Phase 5 describes how a frozen sleeve behaves under transparent, observable
conditions. It does not use the disputed GMM/HMM states and does not estimate a
router. Its question is:

> Which observable market conditions are associated with materially different
> sleeve economics, and which differences are stable enough to justify a new,
> prospectively frozen router hypothesis?

All Phase 4 eligibility decisions remain unchanged. A failed standalone sleeve
does not become acceptable because one retrospectively selected bucket looks
better.

## Causal timing

The common condition source is the same frozen roll-clean Chinese-futures panel
used by the Phase 1B factor cohort. Every condition for decision date \(t\) uses
data known after close \(t\). The tested sleeve return begins at open \(t+1\).
Future returns, volume, OI, and future percentile thresholds are not used.

Contract returns are close-to-close returns on the roll-clean price path.
Trailing percentile calculations compare the current observation only with
preceding observations. The source includes 2016 as warm-up, while evaluated
sleeve returns begin in 2017.

## Observable conditions

| Condition | Scope | Frozen definition |
|---|---|---|
| Market trailing-volatility percentile | Market/date | 60-session volatility of the equal-weight product return, ranked against the preceding 756 market observations |
| Contract volatility percentile | Contract/date | Each product's 60-session volatility, ranked against only its own preceding 756 observations |
| High-volatility fraction | Market/date | Fraction of products whose own volatility percentile is at least 75% |
| Cross-sectional dispersion | Market/date | Standard deviation of current product returns, ranked against the preceding 756 market observations |
| Directional coherence | Market/date | Absolute cross-sectional mean of product return signs; 0 is balanced and 1 is unanimous |
| Volume participation | Market/date | Median own-history product volume percentile using the preceding 252 observations |
| OI participation breadth | Market/date | Fraction of products with positive same-contract OI change; contract-switch observations are excluded |
| Shock age | Market/date | Sessions since median absolute product return exceeded the prior rolling 99th-percentile threshold |

Market and contract volatility, dispersion, and volume participation use fixed
quartile boundaries on causal percentile values. High-volatility breadth,
directional coherence, and OI breadth use predeclared economically readable
boundaries. Shock age uses `0`, `1-2`, `3-5`, `6-20`, and more than `20`
sessions. None of these boundaries was selected using sleeve performance.

Market conditions assign the entire sleeve return on a date to one bucket. The
contract-volatility condition instead assigns each contract's gross return,
cost, turnover, and net contribution to its own bucket. The bucket contributions
are required to sum back to the original sleeve P&L within numerical tolerance.

## Bucket evidence

For validation, holdout, and the full sample, each bucket reports:

- Annualised gross and net mean return
- Net Sharpe and annualised volatility
- Turnover, exchange fees, slippage, and total cost
- Date count, active date count, position count, and active position count
- Active-date net hit rate and mean executed gross exposure
- 95% confidence interval for annualised net mean

The confidence interval uses a five-lag Newey-West standard error of the daily
mean. It is descriptive. It is not adjusted for the eight conditions, 33
buckets, five factors, or other prior research choices. A bucket whose interval
excludes zero is therefore not automatically a validated discovery.

## Completed findings

All five Phase 1B mean-reversion sleeves were evaluated, generating eight
conditions and 33 populated buckets per sample split.

### Volatility alone remains unstable

The ordering of market-volatility quartile returns does not survive reliably
from validation to holdout. Across the five sleeves, the median Spearman
correlation between validation and holdout quartile net returns is `-0.40`; only
two of five correlations are positive.

For the four continuous sleeves, the highest-volatility quartile generally makes
validation losses less negative, but it does not create robust positive net
economics and the ordering often changes in holdout. For the sparse
`fac_042_Bollinger_Binary` sleeve, validation net means range only from roughly
`0.0%` to `1.2%` across market-volatility buckets and every 95% interval crosses
zero. Its highest-volatility holdout bucket is negative.

This reinforces the earlier lesson: volatility magnitude describes stress but
does not identify shock direction, stage, participation, or liquidity mechanism.

### Shock age is the clearest prospective hypothesis, not a result

Shock-age bucket ordering is positive from validation to holdout for all five
sleeves, with a median rank correlation of `0.60`. Shock-day returns are the
highest bucket in holdout for each sleeve. However, the evidence contains only
21 validation shock days and 6 holdout shock days. Annualising six observations
produces visually large returns and Sharpes that are not reliable estimates of
deployable performance.

Most validation shock-day confidence intervals cross zero. The apparent pattern
is therefore recorded as a prospective hypothesis:

> Immediate reversal may be stronger on the broad shock day than during the
> subsequent digestion period.

It must be frozen and tested on genuinely new shock events. Phase 5 does not
authorize a shock-age route.

### Dispersion and volume deserve continued observation

Cross-sectional dispersion and volume participation each show a median
validation-to-holdout bucket rank correlation of `0.80`, positive for four of
five sleeves. This is more stable than volatility quartiles, but much of the
ordering distinguishes degrees of negative performance rather than identifying
a profitable state. Extreme participation buckets also have small and uneven
samples: volume Q1/Q4 contain 65/103 validation dates and only 22/48 holdout
dates.

These variables may help explain whether a move offers cross-sectional
selection opportunities and whether participation supports execution. They are
not yet evidence for switching strategies.

### The sparse candidate remains unconfirmed

For `fac_042_Bollinger_Binary`, none of the 33 validation buckets and none of the
33 holdout buckets has a 95% HAC interval that excludes zero. Conditional
analysis therefore does not repair its Phase 4 evidence shortage or negative
holdout result.

## Decision

Phase 5 produces market-anatomy evidence, not a router candidate. The current
actionable conclusions are:

1. Do not route on market-volatility quartiles alone.
2. Record shock age as a prospective hypothesis requiring new shock events.
3. Keep dispersion and volume participation as explanatory dimensions, with no
   threshold or route frozen yet.
4. Do not route any of the five current sleeves until a new standalone sleeve
   specification passes Phase 4.

## Reproducible artifacts

The implementation lives in:

```text
src/oqp/research/sleeves/conditional.py
scripts/research/build_conditional_behaviour.py
tests/research/test_conditional_behaviour.py
apps/research_dashboard/views/conditional_behaviour_panel.py
```

The common observable panel and factor-specific evidence are written to:

```text
runtime/artifacts/research/conditional_behaviour/common/
runtime/artifacts/research/conditional_behaviour/
  <factor_id>/<market_vertical>/<sleeve_id>/
```

Factor Drilldown reads these saved artifacts. It does not rerun the market panel,
factor, sleeve, or conditional analysis.

Phase 6 is specified separately in
`research_pipeline_phase_6_router_hypothesis.md`. The present Phase 5 findings do
not meet its admission requirements because no current sleeve passed Phase 4 and
the observed shock-age pattern does not predate an untouched holdout.
