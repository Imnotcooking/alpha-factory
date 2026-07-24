# Research Pipeline Phase 1: Factor Library

## Position in the pipeline

Phase 0 freezes the experiment: dataset identity, eligible universe, timing,
costs, parameter permissions, optimization method, and success criterion.
Phase 1 then asks whether each factor is a clear, reproducible predictive Lego
piece before any sleeve, router, sizing rule, or strategy is constructed.

Phase 1 has two gates. A factor must pass the definition gate before it enters
the empirical comparison gate.

## Gate 1: complete factor definition

Every selectable `fac_*` module declares:

1. Stable ID equal to its filename stem.
2. Factor family and subfamily.
3. Economic hypothesis.
4. Required input columns.
5. Native market, data frequency, and signal frequency.
6. Declarative defaults for every `compute()` parameter.
7. Signal orientation.
8. Evaluation geometry.
9. Expected economic holding horizon.
10. Known limitations.

The expected holding horizon is not inferred from the number of trades. It is
the period over which the proposed mechanism should predict returns. The actual
test holding policy remains a separately frozen Phase 0 assumption.

Signal orientation has three allowed values:

- `higher_is_bullish`: a larger score predicts a larger forward return.
- `higher_is_bearish`: a larger score predicts a smaller forward return.
- `unsigned_event`: magnitude identifies an event but direction comes from a
  separately declared transformation.

The factor parameter schema records source defaults even when a parameter is
not tunable. A genuinely parameterless factor declares `FACTOR_PARAMETERS = {}`.
Search bounds authorize an optimizer to propose values; they do not change the
factor's defaults.

## Pure-factor boundary

A Phase 1 factor produces an alpha score or predictive signal. It does not own:

- router logic;
- leverage or target gross exposure;
- Kelly, HRP, or volatility sizing;
- per-asset weight caps;
- portfolio risk budgets;
- stop-loss or take-profit policy;
- execution scheduling beyond the causal signal timestamp.

Those components remain separate so the same factor can be evaluated and reused
without silently changing its economic meaning. Existing `direct_target` and
embedded-hold files remain visible, but they must be split before Phase 1
promotion.

## Gate 2: common-data evidence

Complete factors are compared only inside a compatible cohort. The cohort must
agree on native market, data frequency, evaluation geometry, signal timing,
return assumption, and prediction target. Comparing a daily cross-sectional
rank with a tick-level time-series event would not measure duplication.

At source-default parameters, each comparable factor records at least:

- Pearson IC: linear association between score and forward return;
- Rank IC: Spearman association between score rank and forward-return rank;
- ICIR: mean period IC divided by its period-to-period standard deviation;
- positive-IC period share;
- chronological fold results;
- valid observation and asset coverage;
- turnover and cost diagnostics where a standardized portfolio translation is
  used;
- limitations and failure regimes.

For period IC values `IC_t`, the reported ICIR is:

```text
ICIR = mean(IC_t) / standard_deviation(IC_t)
```

The report must state whether it is raw or annualized. Rank IC is normally the
primary predictive metric for cross-sectional factors. Pearson IC remains useful
for magnitude-sensitive time-series predictions.

## Near-duplicate grouping

Correlation is a grouping test, not a quality test. Two factors form a review
edge only when every frozen condition passes:

- at least 10,000 common score observations;
- absolute score Spearman correlation at least 0.80;
- absolute standardized-target Pearson correlation at least 0.90;
- active-position overlap at least 0.60.

Absolute correlation catches both copies and sign inversions. Exact thresholds
are versioned policy and must be frozen before looking at a new cohort.

Connected correlation edges form a near-duplicate cluster. A factor outside a
cluster is unique, not necessarily good. A unique factor can still have negative
IC, unstable performance, poor coverage, or no economic value.

## Representative selection

Within each cluster, selection uses five declared dimensions:

1. Predictiveness.
2. IC stability.
3. Positive-period consistency.
4. Coverage.
5. Simplicity.

The implementation uses epsilon-Pareto dominance rather than one weighted
backtest score. Small differences inside frozen tolerances are treated as
research noise. One factor becomes the proposed representative only when it is
no worse outside those tolerances on every dimension and materially better on
at least one.

If multiple factors make genuine trade-offs, the result is `manual_review`.
The system does not change metric weights until one happens to win. This is how
we avoid retaining ten near-identical formulas while also avoiding deletion
because one formula had a slightly better historical Sharpe.

Even a dominated factor is only an archive candidate. Dependency review must
confirm that no active strategy, report, or reproducibility artifact still
requires its canonical ID. Stable aliases are preserved after archival.

## Current baseline

The metadata-only audit on 2026-07-22 found:

- 52 stable, normalized factors across 22 deduplication cohorts;
- 52 missing an explicit signal orientation;
- 52 missing a structured expected holding horizon;
- 52 missing explicit known limitations;
- 48 missing a complete declarative parameter schema;
- 25 still using a non-pure portfolio layer;
- one explicit allocation parameter still embedded in a factor schema.

Therefore, zero factors are currently marked Phase-1-ready under the stricter
contract. This does not mean all factors are economically invalid. It means the
new promotion standard is stricter than the old normalization standard and we
have not yet filled the missing declarations or separated every mixed component.

The generated migration queue is:

```text
runtime/artifacts/research/factor_registry_normalization/phase_1_factor_manifest.csv
```

## Execution order

1. Migrate one contract-compatible family at a time.
2. Review its economic hypotheses, horizons, orientation, limitations, and
   parameter defaults in human-readable form.
3. Extract embedded sizing or holding rules into the appropriate Lego registry.
4. Freeze one common dataset and run source-default factor evaluation.
5. Form high-correlation review clusters.
6. Apply the Pareto representative rule and dependency review.
7. Archive only evidence-backed duplicates while preserving stable aliases.
8. Optimize parameters only after the default factor has demonstrated credible
   predictive behavior and the final holdout remains untouched.

The first migration batch should be daily Chinese-futures factors because that
market has the strongest current data fingerprint, liquidity policy, and
instrument-specific cost model.
