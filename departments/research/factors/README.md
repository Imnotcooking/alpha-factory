# Private Factor Registry

This directory is the single flat registry for private factor recipes.
Filesystem subfolders do not classify the library. Each `fac_*.py` recipe
declares its market, frequency, category, prediction contract, and information
timing in `FACTOR_METADATA` and `FACTOR_CONTRACT`; the dashboard organizes
those fields for researchers. Position lifecycle belongs to a sleeve, never
to the factor.

`catalog.yaml` records migration provenance for legacy components. New routers,
router states, position policies, and diagnostics belong in their dedicated
registries under `departments/research/`; final compositions belong in
`departments/research/strategies/`. Project folders own runners and reports,
never duplicate active factor logic.

Use `factor_template_private.py` as the starting point for a new live factor:

1. Copy it into this directory.
2. Rename the copy to a real `fac_*.py` identifier.
3. Fill out `FACTOR_ID`, `FACTOR_METADATA`, and `FACTOR_CONTRACT`.
4. Declare market eligibility with `supported_markets`, for example
   `["FUTURES_CN"]`, `["EQUITY_US"]`, or `["OPTIONS_US"]`.
5. Expose a deterministic `compute(data: pd.DataFrame) -> pd.DataFrame`
   function. The backtest runner calls `compute`, and the returned frame must
   contain `date`, `ticker`, and the contract's `alpha_signal_col`.
   For option factors, prefer `OPTIONS_DAILY_DIRECTIONAL` from
   `oqp.research.factor_presets`; return an underlying-level direction score
   in `factor_score`, and the options CLI will convert sign changes into
   option entry/exit events.
6. Keep allocator choices out of the factor file. Routers, Kelly/HRP sizing,
   leverage, volatility scaling, concentration caps, and contract sizing are
   separate Lego components selected by the strategy or evaluation run.
7. Keep generated data, model files, and backtest output in `runtime/`.
8. Add a synthetic test under `tests/research/private_factors/` before
   promoting reusable logic. This is a separate ignored/private pytest lane.

## Phase 1 Definition

Every selectable factor must additionally declare:

- `ECONOMIC_HYPOTHESIS`: why the signal should predict the target return;
- `FACTOR_PARAMETERS`: every `compute()` default, including fixed parameters;
- `SIGNAL_ORIENTATION`: `higher_is_bullish`, `higher_is_bearish`, or
  `unsigned_event`;
- `EXPECTED_HOLDING_HORIZON`: minimum, maximum, unit, and economic rationale;
- `KNOWN_LIMITATIONS`: at least one falsifiable data, market, capacity, or
  regime limitation.

`portfolio_layer` must be `alpha_score` or `predictive_signal`. Parameters that
control gross exposure, leverage, portfolio volatility, risk budgets, Kelly
sizing, or per-asset weights belong outside the factor. The Phase 1 audit never
guesses missing declarations: an incomplete factor remains visible but is not
eligible for empirical promotion or duplicate retirement.

The normalized manifest now includes a content-purity result, extracted
component lineage, implementation fingerprint, and predictive-evidence
currency check. `direct_target`, `alpha_score_with_legacy_direct_target`, and
`event_signal_with_embedded_hold` are hard boundary violations. They may
remain only in read-only archives, not in the active factor registry.

Do not add new `cnf_*` files. No `cnf_*` implementation remains in the active
factor registry. Historical prototypes are archived and non-factor components
have stable owner-specific IDs. A file receives a new `fac_*` ID only after it
satisfies the factor contract and survives the empirical deduplication gate.
An identity-only rekey may be used earlier to resolve a numeric collision; that
preserves the component for testing and does not constitute promotion.

The completed intraday cleanup retains `fac_087` and `fac_094` as research
candidates. Files prefixed with `_`, including
`_intraday_breadth_breakout_base.py`, are shared implementation helpers and are
never selectable factor IDs.

Do not move live recipes directly into `src/oqp`. Promote only reusable,
sanitized, testable infrastructure there.

## Component Vocabulary

- **Signal:** a causal score or event; it does not allocate capital.
- **State:** information describing the market environment at decision time.
- **Sleeve:** a signal plus explicit selection, direction, holding, and
  weighting rules.
- **Router:** a causal allocation rule across already-defined sleeves.
- **Position policy:** leverage, volatility scaling, risk budgets, liquidity
  limits, concentration caps, and contract sizing applied after alpha.
- **Risk overlay:** a causal exposure brake on a completed strategy target.
- **Strategy:** the reproducible recipe that assembles factors, sleeves,
  routers, position policies, overlays, and execution.

The canonical factor-family taxonomy and the boundaries between these Lego
pieces are defined in
`docs/governance/factor_component_taxonomy.md`.

The lifecycle is `hypothesis -> diagnostic -> exploratory ->
frozen_validation -> internal_candidate`. Failed components become `rejected`
or `diagnostic_only`; they are recorded rather than silently rediscovered.

## Normalized Metadata

Factors enter empirical correlation and IC deduplication only after declaring
metadata schema version 1. The schema records a stable ID, factor family,
native market, data and signal frequency, portfolio layer, cost-model role,
legacy aliases, and a `deduplication_cohort`. Files are compared only within a
cohort whose factor contracts also agree on evaluation geometry, execution
mode, lag, and return assumption.

Canonical IDs equal filename stems. Historical IDs are retained in
`stable_ids.yaml` as read-only aliases and are never reassigned to another
factor.

### Normalization Baseline

The content audit completed on 2026-07-24 establishes this starting point for
empirical deduplication:

- 264 active factors reviewed implementation-by-implementation, with 264 pure
  predictive signals and zero remaining content-boundary violations;
- 46 sleeves, 7 routers, 10 router states, 11 position policies, 5 risk
  overlays, and 2 diagnostics in dedicated registries;
- 44 hybrid factors individually refactored while preserving their factor IDs,
  plus one shared temporal-policy extraction across all 191 GTJA wrappers;
- zero registry-boundary violations, component-registry metadata gaps,
  stable-ID mismatches, or duplicate numeric factor IDs. Phase 1 declaration
  readiness is tracked separately and may still block empirical promotion.

Run the strict audits after any factor or component change:

```bash
PYTHONPATH=src:. python scripts/research/audit_factor_purity.py
PYTHONPATH=src:. python scripts/research/audit_component_registry.py
PYTHONPATH=src:. python scripts/research/build_factor_cohort_manifest.py
```

Changing a factor or sleeve implementation changes its definition
fingerprint. Older predictive, sleeve, standalone, and routing evidence must
then be treated as stale and rebuilt; historical protocol hashes are never
rewritten to disguise that invalidation.

Architecture archival does not by itself imply economic failure. Legacy
`fac_029` was removed mechanically because its score was a positive scalar
multiple of `fac_028`. The subsequent common-data daily audit archived
`fac_008` as the empirically inferior exact inverse of `fac_009`, and `fac_013`
as a highly correlated, economically inferior version of `fac_014`. The
retained representatives are research factors, not production approvals.
Further correlation and IC decisions require common-data evidence within a
contract-compatible cohort.

## ML factor contract

An ML factor owns the transformation from model predictions to an alpha signal;
it does not own model training, validation splits, or artifact storage. Declare:

```python
MODEL_TYPE = "xgboost"  # or "lightgbm"
TARGET_COLUMN = "target_4d_rank"
REQUIRES_OOS_PREDICTIONS = True
```

Historical factor evaluation must use `data["ml_prediction"]`, which contains
only purged out-of-sample predictions. Rows from training periods remain null
and must not be traded. Do not reload the final model and predict the full
historical matrix; that leaks fitted information into the backtest.

For future/live inference, resolve the model selected by the runner instead of
hardcoding a legacy folder:

```python
from oqp.research.ml import resolve_model_artifact_path

model_path = resolve_model_artifact_path(
    data,
    factor_id=FACTOR_ID,
    model_type=MODEL_TYPE,
)
```

The shared trainer records the validation policy, feature list, data fingerprint,
metrics, artifact path, and taxonomy lane in `ml_experiments` and
`model_artifacts`. A reproducible training and backtest command is:

```bash
python scripts/research/run_ml_backtest.py \
  --asset FUTURES_CN \
  --factor fac_054_XGBoost_Alpha \
  --model xgboost \
  --target-column target_4d_rank \
  --validation-mode walk_forward \
  --retrain
```

Feature matrices must carry `date`, `ticker`, `asset_class`, numeric `f_*`
features, and the selected `target_*` column. The runner filters the matrix to
the requested taxonomy lane before either training or factor evaluation.
