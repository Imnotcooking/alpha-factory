# Phase 8: Optional Optimisation

## Purpose

Phase 8 asks whether one already-defined research component can be improved
within a small, frozen parameter space without losing attribution. It is
disabled by default and is not required for a factor or strategy to advance.

One study may tune exactly one layer:

| Mutable layer | Typical parameters |
| --- | --- |
| Factor | Lookback, decay, threshold |
| Sleeve | Quantile, rebalance interval, winsor limit |
| Router | Volatility window, percentile threshold, switch buffer |
| Allocator | Volatility target, position cap, leverage |
| Overlay | Exposure scale, activation threshold |

The mutable component is identified by one stable ID and one declarative
parameter-schema fingerprint. Every other component is recorded by its frozen
fingerprint. A study cannot tune a factor, sleeve, router, allocator, and overlay
together.

## Frozen protocol

Before any trial, the YAML must freeze:

- dataset, universe, liquidity, timing, cost, and holdout fingerprints;
- mutable layer and component ID;
- complete parameter search-space fingerprint;
- purpose-appropriate Optuna sampler, seed, and maximum trial budget;
- chronological fold count and sizes;
- purge and embargo periods;
- objective definitions, directions, and shrinkage priors;
- a versioned Phase 9 objective-profile ID and fingerprint;
- hard constraints;
- the lexicographic priority used to select one point from the Pareto frontier.

Changing any item creates a new study ID. The optimizer proposes parameter
values but never rewrites the component's default schema.

## Inner folds

Only dates strictly before the final holdout enter optimisation. Phase 8 uses
expanding chronological training windows and fixed validation blocks. For each
fold, the final training observations covered by an overlapping forward label
are purged. A separate embargo gap then remains before validation.

The fold evaluator receives only that fold's training and validation frames. It
never receives final-holdout rows.

## Objectives and Pareto frontier

Phase 9 supplies separate objective profiles for factor, sleeve, router, and
allocator studies. A Phase 8 study is rejected when its mutable layer,
objectives, constraints, priority order, or frozen upstream components do not
match that profile. Overlay optimisation remains unavailable until its own
economic objective profile is approved.

Each trial is evaluated on every inner fold. Several objectives may be declared,
for example:

- maximise validation Rank IC;
- maximise net Sharpe;
- minimise turnover;
- minimise maximum drawdown magnitude.

A candidate is Pareto-efficient when no other feasible trial is at least as good
on every objective and strictly better on one. Phase 8 does not hide these
trade-offs inside one fitted weighted score. The protocol freezes a lexicographic
priority before search to choose one candidate from the Pareto set. Trial number
is only the final deterministic tie-breaker.

## Bayesian shrinkage

For objective values observed across folds, Phase 8 uses a normal-normal update.
If the fold mean is \(\bar{x}\), its estimated variance is \(s^2/n\), and the
frozen prior is \(N(\mu_0,\tau^2)\), the posterior mean is

\[
\mu_{post}=
\frac{\mu_0/\tau^2+\bar{x}/\sigma_{\bar{x}}^2}
{1/\tau^2+1/\sigma_{\bar{x}}^2}.
\]

A frozen noise floor prevents zero fold dispersion from implying false
certainty. Noisy candidates are pulled more strongly toward the prior; stable
candidates retain more of their observed mean. Priors are objective-specific and
must be declared before trials begin.

## Trial ledger

Optuna operational state is stored in SQLite. Summarised research artifacts
store every trial, including failed trials, their sampled parameters, failure
type and message, fold metrics, constraints, and timestamps. An ordinary failed
trial does not terminate the remaining frozen budget.

## Candidate freeze and final holdout

After search:

1. retain the complete feasible Pareto set;
2. apply the predeclared selection priority;
3. write an immutable `frozen_candidate.json`;
4. evaluate that candidate once on the fingerprinted final holdout.

The first holdout access writes `holdout_access.json` before evaluation begins.
If evaluation fails, the holdout is still considered consumed and the failure is
recorded. A second attempt is rejected. Any revised implementation or parameter
choice requires new future data, not another pass over the same holdout.

## Current state

Phase 8 infrastructure is implemented, but no optimisation study is currently
declared or enabled. The readiness state is therefore `disabled`, not failed.
This preserves the fixed, unoptimised research baselines developed in earlier
phases.
