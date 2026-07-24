# Regime models

This package owns reusable sequential latent-state estimators and causal
filtering. Use the exact public family name:

```python
from oqp.research.ml.regimes import GaussianHMM, GMMHMM, StudentTHMM

model = GaussianHMM(n_states=2).fit(
    observation_batch,
    model_id="daily-cn-futures-gaussian-hmm-v1",
    preprocessing_artifact_sha256=preprocessor.lineage_sha256,
)
```

`GaussianHMM`, `GMMHMM`, and `StudentTHMM` own family-valid configuration.
They delegate forward-backward, deterministic restarts, EM, acceptance gates,
hashing, and immutable fitted artifacts to the shared diagonal-HMM engine.
`DeterministicDiagonalHMMTrainer` remains available as the advanced low-level
API, not as the single public identity for all HMM families.

Historical pandas/hmmlearn wrappers (`MarketHMM`, `MarketGMMHMM`, and macro
training helpers) are isolated under `regimes.legacy`. They use canonical
module identities; retired `oqp.research.regimes` imports are not supported.

`alignment.py` provides deterministic state-label canonicalization across
refits. It summarizes shared `FittedDiagonalHMM` emissions using an
authenticated feature schema and an explicit training-data SHA-256, then uses
minimum-cost assignment with a deterministic lexicographic tie policy. This is
only a diagnostic and array-reordering layer: it never changes a fit,
likelihood, target, or selection score.

The numerical core does not construct futures rolls, fit preprocessing on
holdout rows, choose folds, name anonymous states, or make portfolio decisions.
Those responsibilities belong to dataset, study, semantics, and strategy
adapters.
