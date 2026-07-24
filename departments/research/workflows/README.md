# Research workflows

This folder owns reproducible experiment, scan, and artifact-producing
recipes. A workflow composes reusable code from `src/oqp` but does not become
part of the installed ML library.

The boundary is deliberate:

- reusable estimator or deterministic transform → `src/oqp/research/`
- experiment orchestration and artifact recipe → this folder
- dashboard file discovery, caching, controls, rendering, or preview →
  `apps/research_dashboard/`
- generated outputs → `runtime/`

Current workflows:

- `hmm_complexity/` — controlled iid/Markov/emission-family study with a
  separately named external QLIKE and date-block-inference adapter
- `latent/temporal_vqvae.py` — temporal VQ-VAE experiment recipe
- `state_space/dual_kalman_features.py` — Dual Kalman feature artifact recipe
- `statistical_arbitrage/` — relationship estimation and opportunity scan

Workflows may import reusable source packages. Reusable source packages must
not import workflows or application code.
