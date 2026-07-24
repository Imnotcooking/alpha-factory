# Research Pipeline Phase 0.6: Primary Success Criterion

## Position in the pipeline

This is the final Phase 0 research-design item. It comes after freezing:

1. Dataset identity.
2. Universe and liquidity eligibility.
3. Signal frequency and holding policy.
4. Transaction costs and execution assumptions.
5. Tunable parameters and the permitted optimization method.

It comes before factor evaluation, factor promotion, sleeve construction, router
testing, and final strategy selection.

## What the criterion means

A primary success criterion is the answer to this question **written before the
experiment is run**:

> What exact result would count as evidence that this research object achieved
> its intended job?

It contains three different layers.

### 1. One primary metric

This is the quantity the experiment is principally trying to improve. Examples:

- Cross-sectional factor: validation mean Rank IC.
- Time-series factor: validation mean Pearson IC.
- Executable daily strategy: validation net Sharpe ratio.
- Router: validation net mean return.
- Predictive model: validation mean Rank IC relative to a simple model.

Only one metric is primary. Sharpe, return, drawdown, IC, hit rate, and turnover
cannot all become interchangeable definitions of success after results are seen.

### 2. One frozen comparator

A positive value is not automatically useful. The result must be compared with
the relevant alternative that was fixed in advance:

- A factor versus zero or its source-default parameterization.
- An optimized factor versus the frozen unoptimized factor.
- A strategy versus its declared benchmark.
- A model versus a simple linear or tree baseline.
- A router versus the best standalone sleeve **and** the frozen static blend.

For a maximizing metric, improvement is:

```text
candidate primary metric - comparator primary metric
```

For a minimizing metric, the subtraction is reversed. The experiment passes the
comparator test only when this improvement reaches the predeclared minimum.

### 3. Hard gates

Gates prevent one attractive headline from hiding a disqualifying weakness. They
are conditions, not extra terms mixed into an opaque score. Examples include:

- Minimum observation or fold count.
- ICIR and sign consistency.
- Positive performance after frozen transaction costs.
- Cost break-even margin.
- Router increment HAC evidence.
- Liquidity, turnover, leverage, drawdown, or capacity limits.

A candidate that wins the primary metric but fails a hard gate does not pass.

## Pass, fail, and incomplete

The evaluator returns three possible decisions:

- `pass`: the primary floor, comparator improvement, and every gate pass.
- `fail`: all required metrics exist, but at least one declared condition fails.
- `incomplete`: at least one required metric is missing or non-finite.

`incomplete` is deliberately different from `fail`. Missing evidence is never
silently converted to zero. It means the experiment cannot yet answer its own
predeclared question.

## Initial profiles

The versioned registry is `config/research/success_criteria.yaml`.

| Profile | Intended claim | Primary comparison |
|---|---|---|
| `factor_cross_sectional_predictive_v1` | Factor ranks next returns across assets | Validation mean Rank IC versus frozen baseline |
| `factor_time_series_predictive_v1` | Factor predicts returns through time | Validation mean Pearson IC versus frozen baseline |
| `strategy_daily_internal_net_value_v1` | Daily strategy has executable economic value | Validation net Sharpe versus frozen benchmark |
| `sleeve_daily_standalone_net_value_v1` | Frozen sleeve has standalone executable value | Validation net Sharpe versus cash before routing |
| `router_incremental_net_value_v1` | Router adds value beyond simpler alternatives | Validation net mean return versus best standalone/static alternative |
| `model_cross_sectional_prediction_v1` | Model improves predictive ranking | Validation mean Rank IC versus simple-model baseline |

These are versioned desk policies, not universal laws of finance. Changing a
threshold changes the profile version or creates a new profile. It must not
silently alter the meaning of an existing experiment.

## Router example

Suppose a volatility router reports:

```text
router validation net mean return       = 1.20% per month
best sleeve/static alternative          = 1.00% per month
increment                               = 0.20% per month
HAC t-statistic of monthly increment    = 0.80
```

The router beats the comparator numerically, but the current router profile also
requires `validation_increment_hac_t >= 1.645`. Its decision is therefore
`fail`, not pass. The result remains economically interesting, but it does not
support the stronger claim that routing has demonstrated incremental value.

If the HAC statistic had not been calculated, the result would be `incomplete`.

## Strategy example

A strategy can have positive net Sharpe and still fail if the benchmark has a
higher net Sharpe. Conversely, beating a negative benchmark with a still-negative
Sharpe also fails the absolute floor. The internal daily-strategy profile requires:

- Positive validation net Sharpe.
- Net Sharpe at least as high as the frozen benchmark.
- Positive annualized return after costs.
- Gross edge at least equal to estimated costs.
- At least 252 validation trading days.

Risk-appetite limits such as maximum acceptable drawdown should be added through
a separately versioned desk profile once those limits are formally approved;
they are not invented in this initial registry.

## How it is used

Factor-portfolio YAML can declare:

```yaml
success_criterion:
  profile_id: strategy_daily_internal_net_value_v1
```

The runner resolves that profile, validates its research-object type, and attaches
the full definition and SHA-256 fingerprint to the research frame. The assumption
manifest records the frozen definition and any later evaluation result. The
dashboard shows whether the criterion was absent, declared, incomplete, failed,
or passed.

For standalone factor and model experiments, use
`attach_success_criterion_attrs()` before evaluation and
`attach_success_criterion_result_attrs()` after the required validation metrics
have been computed.

## What this does not do

- It does not guarantee future profitability.
- It does not replace uncertainty estimates or multiple-testing controls.
- It does not allow the final holdout to be used during model or parameter search.
- It does not make a high IC sufficient for production.
- It does not let a router compare itself only with the weaker sleeve.

The criterion prevents goalpost movement. It does not remove estimation risk.

## Next pipeline step

With Phase 0 frozen, Phase 1 is **single-factor evaluation**. Each factor is first
tested at its source defaults under its declared geometry. IC, Rank IC, ICIR,
fold stability, coverage, turnover, costs, and limitations are recorded before
factors are combined into sleeves or exposed to routers.
