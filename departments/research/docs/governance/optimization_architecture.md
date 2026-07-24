# Optimization Architecture

## Why this layer exists

Optimization is not one algorithm and it is not a machine that certifies alpha.
It is a controlled way to search a declared decision space after the dataset,
universe, timing, costs, objective, constraints, and final holdout have been
frozen. The architecture therefore separates six different jobs:

1. Factor calculation parameters.
2. Router decision parameters.
3. Model hyperparameters.
4. Model-weight training.
5. Portfolio allocation.
6. Universe selection.

The optional research-pipeline protocol for single-layer TPE studies, purged
inner folds, shrinkage, candidate freezing, and one-time holdout access is in
`research_pipeline_phase_8_optional_optimisation.md`.

Results from one job cannot silently be used as evidence for another. For
example, Adam trains neural-network weights; it does not optimize a discontinuous
EMA rule. CVXPY solves a constrained allocation problem; it does not discover a
factor lookback.

## Shared experiment contract

Every black-box search uses an `OptimizationStudySpec` containing:

- A stable study ID and purpose.
- A declarative component parameter schema and its fingerprint.
- One or more predeclared objectives.
- Hard feasibility constraints.
- Frozen dataset, universe, liquidity, timing, cost, and holdout fingerprints.
- A sampler, seed, trial budget, and timeout.
- A locked final holdout.

The study ID is immutable. Reusing it with different frozen inputs or a different
parameter schema raises an error. Optuna's operational state lives in
`runtime/state/optimization/optuna.sqlite3`; summarized evidence lives in the
research registry; immutable trial and result JSON files live under
`runtime/artifacts/research/optimization`.

All research searches run with `n_jobs=1`. This is deliberate for deterministic,
low-memory execution on the current workstation. A larger machine may use a
separate reviewed execution profile later; parallelism must not change the study
definition silently.

## Search families

| Family | Adapter | Use it when | Main warning |
|---|---|---|---|
| Exhaustive | `grid`, `bruteforce` | One to three small discrete dimensions and cheap evaluations | Combination count grows exponentially |
| Baseline stochastic | `random`, `qmc` | First coverage test and mandatory benchmark for smarter search | Does not learn from completed trials |
| Sequential surrogate | `tpe`, `gp` | Expensive black-box evaluations where past trials should guide the next one | Can exploit sample noise and apparent sharp peaks |
| Population/evolutionary | `cmaes`, `nsga2` | Rugged continuous spaces or constrained multi-objective Pareto search | Requires enough trials to form a useful population |
| Global continuous | SciPy differential evolution and dual annealing | Numeric spaces with one callable objective outside the Optuna study loop | Categorical logic needs a different representation |
| Gradient training | PyTorch Adam, AdamW, SGD, RMSprop, LBFGS | Differentiable model-weight training | Not valid for discontinuous backtest rules |
| Convex allocation | CVXPY | Portfolio weights with convex risk, turnover, cost, gross, net, and position constraints | Expected returns and covariance can still be badly estimated |

The SciPy adapter is intentionally separate because it directly solves one
continuous objective. The PyTorch factory records how model weights were trained.
The CVXPY allocator checks disciplined-convex-program validity and solver status;
it never falls back silently to a heuristic portfolio.

The governed purpose-to-method map lives in
`config/research/optimization_methods.yaml`. It distinguishes black-box
parameter search, differentiable model-weight training, direct convex
allocation, and experimental universe selection. The Research Review
Optimisation workspace reads this registry so the dashboard cannot present one
algorithm as interchangeable across fundamentally different jobs.

## Selection protocol

### 1. Establish a random baseline

TPE, GP, CMA-ES, or another adaptive method must be compared with random search
using the same parameter schema and trial budget. If the adaptive method cannot
beat random coverage, its added complexity has not earned a role.

### 2. Optimize development evidence only

The evaluator receives an `OptimizationEvaluationContext` with the final holdout
locked. It may use chronological training and validation folds. It must return
declared metrics and fold evidence; the runner extracts only the predeclared
objective and constraints.

### 3. Treat feasibility separately from preference

Minimum observations, liquidity, turnover ceilings, leverage limits, or minimum
fold coverage are constraints, not bonuses hidden inside a Sharpe-like score.
Infeasible candidates stay in the audit trail but cannot become the selected
candidate.

### 4. Inspect the surface

The result flags boundary optima and checks whether nearby sampled values form a
broad near-best region. A narrow isolated peak is an overfitting warning. It is
not rescued merely because its headline objective is high.

### 5. Select once, then open the holdout

One candidate is frozen from development evidence. The final holdout is evaluated
once. A failure remains a failure and starts a new hypothesis; the holdout does
not become another optimization fold.

## Purpose-specific rules

### Factors and routers

Factors declare `FACTOR_PARAMETERS`; routers declare `ROUTER_PARAMETERS`.
Schemas contain source defaults, type, bounds or choices, and `tunable` status.
The optimizer proposes overrides without rewriting source defaults. Factor
promotion still requires IC, Rank IC, ICIR, walk-forward stability, turnover,
costs, and an economic explanation. Router evidence must compare the routed
strategy with every standalone sleeve and a static blend under the same costs.

### Machine-learning models

Model hyperparameters use the black-box study runner. Neural-network weight
updates use `TrainingOptimizerSpec` and `build_torch_optimizer`. These records are
different because a TPE trial may choose a learning rate while Adam uses that
learning rate to train millions of weights inside the trial.

### Portfolio allocation

`ConvexPortfolioAllocator` solves weights directly from expected alpha,
covariance, previous weights, costs, eligibility, and explicit exposure limits.
The older `PortfolioOptimizer` name remains as a compatibility alias for
`HeuristicPortfolioSizer`; it is not a mathematical optimizer and should be
reported as a sizing heuristic.

## Current migrated components

- Declarative factor schemas, including the former factor-local Optuna cases.
- Declarative router schemas for the reproducible router library.
- Tick-pulse relative-velocity heuristic calibration with fixed calendar holdout.
- Tick XGBoost hyperparameter calibration with fixed chronological validation and
  final test blocks.
- VAE and VQ-VAE model-weight optimizer declarations.
- Convex portfolio allocation.

Genetic programming that invents entry-rule syntax and particle swarm search are
not currently implemented. They would add another research degree of freedom
without a validated use case. Discrete rule parameters can first be tested with
grid, random, TPE, or NSGA-II; a dedicated representation should be introduced
only when a frozen experiment requires it.

## Where this sits in Phase 0

This module implements the search machinery needed by Phase 0.5, **Tunable
Parameters**. It does not complete Phase 0 itself. Work resumes afterward with
the primary success criterion, then factor evaluation and strategy construction.
