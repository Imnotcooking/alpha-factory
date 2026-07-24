# Research ML model-library architecture

## Decision

All reusable machine-learning implementations are organized beneath
`oqp.research.ml`. A package name identifies one architectural dimension; the
Model Library records the remaining dimensions explicitly rather than trying
to encode every concept in one inheritance tree.

| Package | Architectural meaning | Current public implementations |
|---|---|---|
| `ml.regimes` | Unsupervised sequential inference | `GaussianHMM`, `GMMHMM`, `StudentTHMM` |
| `ml.latent` | Self-supervised representations | `VQVAETrainer` |
| `ml.state_space` | Online adaptive estimation | `DualKalmanRegression` |
| `ml.tree_based` | Tree-based algorithm family | `LightGBMRegressorTrainer`, `XGBoostRegressorTrainer` |
| `ml.regression` | Supervised regression task contract | validation policies, base contract, run ledger |
| `ml.preprocessing` | Fitted numerical input transforms | immutable matrix preprocessor |
| `ml.features` | Feature discovery and governance | feature-matrix governance |
| `ml.evaluation` | Out-of-sample diagnostics | purged MDA |
| `ml.core` | Shared training utilities | declarative optimizer construction |

## Algorithm family versus task

“Tree-based” and “regression” are not competing labels. Tree-based describes
the algorithm; regression describes the supervised task. A future linear or
neural regressor can implement the same regression contract from a different
algorithm-family package.

HMM, VQ-VAE, and state-space estimators do not inherit the regression base.
They do not use the supervised forward-target contract and have different
input/output geometries.

## Exact public model identities

The Model Library points to exact estimator classes:

```text
oqp.research.ml.regimes.gaussian_hmm:GaussianHMM
oqp.research.ml.regimes.gmm_hmm:GMMHMM
oqp.research.ml.regimes.student_t_hmm:StudentTHMM
oqp.research.ml.latent.vqvae.model:VQVAETrainer
oqp.research.ml.state_space.dual_kalman_regression:DualKalmanRegression
oqp.research.ml.tree_based.lightgbm:LightGBMRegressorTrainer
oqp.research.ml.tree_based.xgboost:XGBoostRegressorTrainer
```

The three named HMM estimators are family-specific configuration facades over
one shared deterministic diagonal-EM backend. This avoids both misleading
catalog entries and duplicated forward-backward/EM implementations.

Older pandas/hmmlearn `MarketHMM` wrappers are isolated under
`ml.regimes.legacy`; they are not examples for new model development.

The older temporal VQ-VAE encoder remains under `ml.latent.encoders.vqvae`
for joblib compatibility. It is distinct from the immutable reusable
`ml.latent.vqvae` core; documentation and imports must state which one is
being used.

## Layer boundary

- Reusable estimators and deterministic model mathematics live under
  `src/oqp/research/ml/`.
- Research experiment and artifact workflows live under
  `departments/research/workflows/`.
- Dashboard file discovery, persistence adapters, caching, rendering, and
  preview-only logic live under `apps/research_dashboard/`.

An estimator may be reused by both workflows and apps. The reusable package
must not import either outer layer.

## Dependency rule

Every package initializer is lazy. Importing one focused family must not load
an unrelated optional stack. Architecture tests specifically guard against a
Gaussian-HMM import initializing pandas, scikit-learn, joblib, hmmlearn, or
PyTorch.

## Namespace rule

The retired `oqp.research.regimes`, `oqp.research.state_space`, and
`oqp.research.latent` namespaces are not compatibility facades. Reusable models
use `oqp.research.ml`; experiment workflows use
`departments.research.workflows`; dashboard services use
`apps.research_dashboard`. The unrelated preprocessing compatibility alias
remains until its own artifact migration is complete.

Frozen Paper 01 modules live under the project-local
`engine.daily_regimes` namespace. They are evidence consumers and
numerical-parity references, not reusable model-library modules.
