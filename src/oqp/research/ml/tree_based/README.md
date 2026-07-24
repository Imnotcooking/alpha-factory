# Tree-based models

This is the algorithm-family layer. It contains the exact supervised trainer
implementations:

- `LightGBMRegressorTrainer`
- `XGBoostRegressorTrainer`

Both implement the common regression and chronological-validation contract
from `oqp.research.ml.regression`. Historical names `LGBMModel` and
`XGBoostTrainingEngine` remain identity aliases.
