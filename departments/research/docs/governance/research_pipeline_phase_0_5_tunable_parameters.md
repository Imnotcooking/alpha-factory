# Research Pipeline Phase 0.5: Tunable Parameters

## Position in the pipeline

Phase 0.5 comes after the research run has frozen:

1. Dataset identity and fingerprint.
2. Universe and liquidity eligibility.
3. Signal decision frequency and holding policy.
4. Transaction-cost profile and execution assumptions.

It comes before factor comparison, parameter optimization, sleeve construction,
routing, leverage, and production review. Optimization is optional. A factor can
be evaluated at its source defaults without being optimized.

## The rule

Each factor declares its parameters as data in `FACTOR_PARAMETERS`. A factor does
not import Optuna and does not contain `trial.suggest_*` calls.

```python
FACTOR_PARAMETERS = {
    "lookback": {
        "default": 20,
        "type": "int",
        "low": 5,
        "high": 120,
        "step": 5,
        "tunable": True,
    },
    "epsilon": {
        "default": 1e-8,
        "tunable": False,
    },
}
```

The shared optimizer adapter reads this declaration and proposes only parameters
with `tunable: True`. Fixed parameters are still recorded so that the complete
factor calculation is reproducible.

Schema validation requires:

- Every `compute()` input after the data frame to be declared.
- Every declared parameter to exist in `compute()`.
- Schema defaults and source-code defaults to match exactly.
- Numerical search bounds to contain the default.
- Valid positive steps and valid log-search domains.
- Stable schema and selected-value fingerprints on the run.

## Three different parameter states

These states must not be confused:

1. **Source default**: the conservative value written in the factor file. It is
   changed only by a reviewed code edit.
2. **Optimization proposal**: a value sampled during a specific development
   search. It belongs only to that trial and dataset fingerprint.
3. **Frozen candidate configuration**: a separately recorded parameter set that
   passed the agreed validation process. Freezing a candidate does not rewrite
   the source default.

Therefore an optimizer can return `lookback=35` while the factor continues to
declare `lookback=20`. The run records the override and both fingerprints.
Environment variables must not silently replace factor defaults; parameter
changes are passed explicitly through the runner.

## Optimization protocol

### 1. Freeze the experiment

Record the dataset fingerprint, eligible universe, temporal policy, cost profile,
factor schema fingerprint, objective, search space, sampler seed, and trial budget.
Changing any of these creates a new experiment rather than continuing the old one.

### 2. Search on development data only

The optimizer may use training and development-validation folds. It must not see
the final holdout. The objective must be declared before the search, rather than
selected after inspecting whichever metric looks best.

### 3. Check predictive evidence

For every proposed parameter set, retain at least:

- **IC**: linear association between the signal and the subsequent return.
- **Rank IC**: association between their ranks, reducing dependence on extreme
  numerical observations.
- **ICIR**: mean period IC divided by its period-to-period variability. It asks
  whether IC is persistent rather than produced by a small number of periods.
- Walk-forward fold results, including the distribution and sign consistency of
  the metrics rather than only their pooled average.

No universal IC or ICIR threshold proves that a factor works. The candidate must
improve on its frozen default or benchmark with uncertainty, fold stability, and
economic magnitude considered together.

### 4. Inspect the parameter surface

An optimum at the lower or upper search boundary is a warning. It may mean the
search range was poorly chosen or that the optimizer is chasing a monotonic sample
artifact.

Nearby values must also be evaluated. A broad plateau of similar results is more
credible because small estimation errors do not destroy the finding. A single
sharp peak surrounded by weak values is an overfitting warning, not a promotion
candidate.

The shared diagnostics report:

- Exact and near-boundary parameters.
- Number and share of near-best trials.
- Whether locally neighboring values support the best trial.
- Whether neighborhood sampling was insufficient to make that judgment.

These are diagnostic flags. They do not silently accept or reject a factor.

### 5. Recompute turnover and costs

Parameter changes can improve gross IC while making the strategy uneconomic by
increasing signal changes, turnover, market impact, or commissions. Report gross
and net results under the same frozen transaction-cost profile.

### 6. Open the final holdout once

After selecting one frozen candidate without using the holdout, evaluate it once
on the untouched period. A failed holdout is recorded; the holdout must not be
reused as another tuning set.

### 7. Recheck the economic explanation

The selected values must still match the factor's proposed mechanism and intended
horizon. A statistically attractive setting that changes the economic meaning of
the factor requires a new hypothesis and a new experiment identity.

## Required evidence record

An optimization result is incomplete until it records:

- Validation IC, Rank IC, and ICIR.
- Walk-forward fold results.
- Boundary diagnostics.
- Neighborhood and plateau diagnostics.
- Turnover and transaction costs.
- Final untouched-holdout result.
- Economic-interpretation review.

The code exposes this list as `OPTIMIZATION_EVIDENCE_REQUIREMENTS`; downstream
promotion tooling should treat missing entries as incomplete evidence rather than
fill them with defaults.

## Current migration status

The four factors that previously embedded Optuna trial calls now use declarative
schemas:

- `fac_043_Bearish_Breakdown`
- `fac_044_Relative_Velocity_Fade`
- `fac_045_Dual_Kalman_StatArb`
- `fac_096_Tick_Imbalance`

Other factors remain valid at their source defaults and can adopt a schema when a
real optimization experiment is proposed. Router parameters now use the same
declarative contract through `ROUTER_PARAMETERS`. Tick-pulse heuristic calibration
and tick XGBoost hyperparameter search have also migrated to the shared study
runner while retaining their distinct research purposes and data splits.
