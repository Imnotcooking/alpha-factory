# Regime Model Architecture

## Decision

Regime estimators are shared research infrastructure. A paper, dashboard,
router, or operational service may consume them, but none of those consumers
owns the model implementation.

| Layer | Location | Responsibility |
|---|---|---|
| Cross-layer contracts | `src/oqp/contracts/regime_state.py` | Model identity, ordered feature schema, causal inference payloads, and probability semantics. |
| Shared preprocessing | `src/oqp/research/ml/preprocessing/` | Training-only imputation, winsorization, standardization, immutable parameters, and portable artifacts. |
| Shared HMM core | `src/oqp/research/ml/regimes/` | Deterministic Gaussian, GMM, and fixed-degree Student-t HMM training, immutable parameters, causal filtering, checkpoints, and portable artifacts. |
| Shared VQ-VAE core | `src/oqp/research/ml/latent/vqvae/` | Prepared-matrix training, code assignment, reconstruction, codebook access, and portable artifacts. |
| Paper 01 evidence engine | `notebooks/Phase_7_Research_Projects/07_01_daily_latent_regimes_cn_futures/engine/` | Frozen workflow, project-local adapter, and reproducibility evidence. It is a parity reference, not the destination for new generic code. |
| Research workflows | Purpose-named packages under `departments/research/workflows/` | Dataset boundaries, controlled comparisons, diagnostics, reporting, and artifact recipes for one research question. |
| Operations | Apps and runtime services | Load independently approved artifacts, maintain one causal session per entity sequence, and consume `RegimeInference`. |

## Lifecycle

```text
point-in-time adapter
    -> ordered feature schema + explicit entity sequence
    -> training-only fitted preprocessing artifact
    -> offline trainer bound to that preprocessing digest
    -> immutable fitted artifact
    -> independently registered IDs and hashes
    -> causal filter session
    -> RegimeInference
    -> router/dashboard/risk consumer
```

State numbers remain anonymous. Economic names such as “quiet” or “stress”
belong to a separately versioned semantics policy; they are not learned facts.

## Safety invariants

- Feature order is supplied and hash-checked at every public inference call.
- A filter state cannot cross a model, parameter, schema, entity, or declared
  sequence boundary.
- Observation time advances strictly within a live filter session.
- Failed operational inference is transactional and does not consume the row.
- Filtered and one-step probabilities are distinct; smoothed probabilities are
  excluded from operational payloads.
- Predictive density remains in log space.
- Model and checkpoint loading requires identities and digests held outside the
  artifact being loaded.
- Preprocessing parameters are learned on training rows only; their digest is
  authenticated by each fitted study model.
- Canonical artifacts use JSON/NumPy, never pickle or joblib.
- VQ-VAE bundle directories are immutable and versioned; they are never
  replaced in place.
- Fitted VQ-VAE artifacts retain immutable tensors, not a public live neural
  graph; inference materializes the canonical graph privately.
- Focused model imports do not initialize pandas, sklearn, joblib, hmmlearn, or
  unrelated research packages.

## Migration rule

Do not rewrite the immutable Stage 4–12 release snapshots or their recorded
paths. The live evidence engine was moved as one governed package in July 2026;
historical manifest strings and source archives remain unchanged. New work is
additive, and parity tests compare shared code with the project-local frozen
implementation without making the shared package import the paper package.

## Current capability boundary

The shared HMM layer now owns deterministic multi-restart EM training, fitted
parameters, and production-safe causal inference for diagonal Gaussian,
two-level Gaussian-mixture, and fixed-degree Student-t emissions. The shared
VQ-VAE layer owns training and inference for finite prepared matrices. The
shared preprocessing layer owns versioned numeric transforms, while dataset
and window construction remain study-specific responsibilities.

Paper 01 has an explicit one-way adapter. The
`departments/research/workflows/hmm_complexity` workflow consumes these shared
components and registers controlled iid-versus-Markov and emission-family
comparisons. This is a tested computational foundation, not a claim that the
new empirical study has already been run on the final futures panel.

## Current handoff status

1. **Complete:** the Paper 01 prospective adapter authenticates the exact Stage
   4 M2 feature artifact, Stage 5 fold plan, and fold-local target files. It
   reconstructs product-local sequence boundaries and blocks the inspected
   2025-01-01--2026-07-10 holdout.
2. **Complete:** external QLIKE is a separately named training-only target
   probe, and paired inference uses a synchronized non-circular moving
   date-block bootstrap that resamples the full product cross section by date.
3. **Data-gated:** execute the registered two-state ladder on genuinely fresh
   post-2026-07-10 evaluation data. The current price-bar export contains
   2026-07-13, but the required turnover companion ends on 2026-07-10, so no
   complete fresh M2 row exists; historical folds can only be labelled
   exploratory.
4. **Pending after primary execution:** run \(K=3\) as a separately declared
   sensitivity analysis rather than silently tuning or rescuing \(K=2\).
5. **Pending after inference passes:** freeze new empirical artifacts and
   manuscript tables, then consider independent operational promotion.
