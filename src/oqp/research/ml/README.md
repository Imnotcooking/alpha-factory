# OQP research ML library

`oqp.research.ml` is the canonical umbrella for reusable machine-learning
components. The folders answer different questions instead of mixing every
model into one trainer module.

```text
oqp.research.ml
├── regimes/       unsupervised sequential state models
├── latent/        self-supervised representation models
├── state_space/   online adaptive state-space estimators
├── tree_based/    algorithm family: LightGBM and XGBoost
├── regression/    supervised regression task and validation contract
├── preprocessing/ fitted, leakage-safe numeric transforms
├── features/      feature discovery and governance
├── evaluation/    out-of-sample model and feature evaluation
├── core/          framework-neutral training utilities
└── catalog.py     implementation inventory used by Model Library
```

## Two classifications that must not be confused

`tree_based` describes **how a model works**. `regression` describes **what
prediction task it solves**. LightGBM and XGBoost therefore live in
`tree_based` while implementing the common contract from `regression`.

An HMM is also machine learning, but it is not a supervised regressor. It
infers hidden sequential states without a target column, so it belongs under
`regimes` and does not inherit the target-dependent regression base class.
VQ-VAE learns a discrete representation by reconstructing its inputs, so it
belongs under `latent`. Dual Kalman regression updates a probabilistic linear
relationship sequentially, so it belongs under `state_space`; its dashboard
scan and artifact recipes do not.

The Model Library records these dimensions separately: learning paradigm,
research task, input geometry, output contract, and implementation path.

## Canonical imports

```python
from oqp.research.ml.regimes import GaussianHMM, GMMHMM, StudentTHMM
from oqp.research.ml.latent.vqvae import VQVAEConfig, VQVAETrainer
from oqp.research.ml.state_space import (
    DualKalmanRegression,
    DualKalmanRegressionConfig,
)
from oqp.research.ml.tree_based import (
    LightGBMRegressorTrainer,
    XGBoostRegressorTrainer,
)
from oqp.research.ml.regression import ValidationConfig
from oqp.research.ml.preprocessing import PreprocessingSpec
```

The three HMM names are real family-specific estimators. They validate only
the parameters their family understands, then delegate fitting to one shared,
deterministic diagonal-EM implementation. This preserves a single numerical
source of truth without presenting three different models as one generic
catalog entry.

The immutable `ml.latent.vqvae` core and the historical joblib-oriented
`ml.latent.encoders.vqvae` adapter intentionally have separate configuration
classes. New reusable work should prefer the immutable core; the adapter
remains available for the existing temporal dashboard experiment.

## Namespace policy

The former `oqp.research.regimes`, `oqp.research.state_space`, and
`oqp.research.latent` packages were removed after their implementations and
live consumers moved to their owning layers. Use `oqp.research.ml.regimes`,
`oqp.research.ml.state_space`, and `oqp.research.ml.latent` directly. The
remaining compatibility aliases are intentionally narrower:

| Historical path | Canonical path |
|---|---|
| `oqp.research.preprocessing` | `oqp.research.ml.preprocessing` |
| `oqp.research.ml.lgbm_model` | `oqp.research.ml.tree_based.lightgbm` |
| `oqp.research.ml.xgboost_model` | `oqp.research.ml.tree_based.xgboost` |
| other former flat supervised modules | `tree_based` or `regression` |

Only the paths listed in the table remain compatibility aliases. New code must
use canonical imports.

Experiment orchestration is kept in `departments/research/workflows/`, while
dashboard-specific discovery, artifact loading, caching, and preview logic is
kept in `apps/research_dashboard/`. Neither layer is part of the reusable ML
package.

## Evidence and artifact boundaries

An inventory entry means an implementation exists; it does not mean that a
model has been fitted, validated, or promoted. Fitted artifacts belong in the
model registry and executed runs in their governed experiment ledger.

The reusable library does not own Paper 01's frozen stage engine. That engine
lives beside its paper under
`notebooks/Phase_7_Research_Projects/07_01_daily_latent_regimes_cn_futures/engine/`;
it remains a numerical-evidence consumer, not a source for new shared code.

All package initializers are lazy so importing one focused family does not
initialize pandas, scikit-learn, PyTorch, XGBoost, LightGBM, or hmmlearn from an
unrelated family.
