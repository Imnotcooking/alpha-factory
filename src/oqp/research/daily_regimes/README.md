# Daily Latent-Regime Research Package

This package implements the computational architecture for Paper 1. Completed
stages are verified on deterministic synthetic fixtures; these fixtures,
scaffold outputs, and interface checks are never empirical evidence.

## Architectural layers

| Layer | Modules | Current responsibility |
|---|---|---|
| Foundation | `config`, `contracts`, `capabilities`, `seeding`, `runtime_paths`, `artifacts` | Strict research contract, deterministic identities, paths, and provenance |
| Stage 2 plumbing | `synthetic`, `pipeline`, `smoke` | Offline clean fixture, guarded orchestration, and a two-attempt reproducibility gate |
| Stage 3 | `continuous_series` | Point-in-time contract selection and roll-safe product series |
| Stage 4 | `features`, `preprocessing`, `stage4_fixtures` | Causal H7/M2/M3 construction, training-only scaling/winsorization, HPCA3 |
| Stage 5 | `folds`, `targets` | Expanding folds first; then fold-fitted forward targets, purge and embargo |
| Stage 6 | `baselines` | Observable, iid, and simple discrete benchmarks |
| Stage 7 | `hmm`, `filtering`, `state_alignment` | HMM families, genuine forward filtering, label alignment |
| Stage 8 | `diagnostics`, `evaluation` | Failure-aware validation and common-target scoring |
| Stage 9 | `vqvae` | Product-safe windowed discrete representation |
| Stage 10 | `risk_throttle` | Lagged, exposure-matched defensive scaling |
| Stage 13 | `reporting` | Evidence-gated manuscript tables and figures |

Later-stage modules currently define immutable configurations, results,
protocols, and validation semantics. They do not contain placeholder algorithms
under production model names.

## Evidence boundary

- Synthetic fixture truth is stored separately from model inputs.
- Synthetic runs cannot emit scientific-evidence or manuscript artifacts.
- Training-estimated target thresholds require a fold-local fit, training-row
  hash, and fitted-parameter hash; the pipeline places folds before targets.
- Smoothed probabilities are representable for historical description but are
  rejected by prospective evaluation and risk-decision contracts.
- VQ inputs reject targets, forward returns, HMM probabilities, and strategy
  labels; every window and code batch is confined to one declared fold.
- Reporting contracts reject synthetic, unfrozen, and placeholder artifacts as
  paper evidence.
- The capability registry is the machine-readable source of implementation and
  paper-eligibility status.

## Offline smoke gate

From the repository root:

```bash
python scripts/research/run_daily_regime_smoke.py
```

The command reads the dedicated `config/smoke.yaml`, performs exactly two
independent synthetic attempts, and compares the hashes of every computational
payload. It now exercises the Stage 2 foundation, Stage 3 point-in-time roll
construction, and Stage 4 feature/preprocessing implementations. Manifests and
their legitimate timestamps/path metadata are excluded from the exact-match
digest. Existing deterministic run IDs are immutable unless the caller
explicitly supplies `--overwrite`.

Stages 2--4 are complete on synthetic fixtures. The next implementation step is
Stage 5: expanding walk-forward folds and external targets. Real-data feature
generation remains separate until the production source and contract metadata
are frozen.
