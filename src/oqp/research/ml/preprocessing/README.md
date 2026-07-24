# Shared ordered-matrix preprocessing

This package owns the small preprocessing layer that sits between a research
dataset and a fitted model. It is independent of Paper 01, pandas,
scikit-learn, joblib, dashboards, and execution code.

```python
from oqp.contracts.regime_state import OrderedFeatureSchema
from oqp.research.ml.preprocessing import (
    MissingValuePolicy,
    PreprocessingSpec,
    fit_matrix_preprocessor,
)

schema = OrderedFeatureSchema(
    schema_id="daily-risk-v1",
    feature_names=("gk_gap", "amihud", "ker_20d"),
)
spec = PreprocessingSpec(
    missing_value_policy=MissingValuePolicy.MEDIAN,
    winsor_quantiles=(0.01, 0.99),
    standardize=True,
)
fitted = fit_matrix_preprocessor(
    training_matrix,
    feature_schema=schema,
    spec=spec,
    artifact_id="m3-validation-2024-preprocessor",
    training_run_id="validation-2024",
)
model_matrix = fitted.transform(new_matrix, feature_schema=schema)
```

## Leakage boundary

Only `fit_matrix_preprocessor` may learn values. It learns imputation values,
winsor bounds, means, and population standard deviations from the supplied
training matrix. `transform` never updates those values. A validation or
holdout matrix therefore cannot influence its own representation.

Winsor quantiles use the fixed `linear` method. Imputation occurs before
winsor bounds are estimated; transform repeats the same impute, clip, then
standardize order. Features whose fitted standard deviation is at or below
`scale_floor` use a scale of one and are recorded in `low_variance_mask`.

## Identity and persistence

`artifact_sha256` authenticates the ordered schema, behavioral specification,
training-input digest, learned parameters, row count, and training-run lineage.
Use `lineage_sha256` when binding the preprocessor to a fitted model.

`dump_preprocessor_json` writes inert canonical JSON atomically. Loading
requires the expected artifact ID and SHA-256 from an independent registry.
Pickle is deliberately unsupported. Stored arrays and transformed outputs are
backed by immutable bytes, so callers cannot silently rewrite fitted state.
