# Alpha Lab Integration Plan

Last reviewed: 2026-07-02

This plan keeps `alpha_research_lab` productive while gradually promoting the
public-safe, reusable research infrastructure into `src/oqp`. The goal is not a
large folder move. The goal is to make every promoted layer importable,
testable, reproducible, and clear about what remains private.

## Integration Principles

- Promote contracts, schemas, evaluation rules, and synthetic-testable
  utilities before moving strategy logic.
- Keep live factor recipes, raw data, execution logs, model checkpoints, and
  trial artifacts private by default.
- Keep Streamlit pages as consumers of `src/oqp`, not owners of research logic.
- Preserve compatibility shims in `alpha_research_lab` during migration so
  active notebooks, scripts, and dashboards keep running.
- Each promoted module needs a focused test in `tests/` before old imports are
  removed.

## Folder Classification

| Alpha path | Target | Status | Notes |
| --- | --- | --- | --- |
| `factor_contract.py` | `src/oqp/research/contracts.py` | promoted | Public-safe factor interface and return-assumption contract. |
| `factors/contracts.py` | `src/oqp/research/factor_presets.py` | promoted | Reusable contract presets, no private edge. |
| `data_engine/dataset_policy.py` | `src/oqp/research/datasets.py` | promoted | Dataset tradability labels for promotion gates. |
| `evaluation/alpha_metrics.py` | `src/oqp/research/evaluation.py` | promoted | IC and geometry evaluation layer. |
| `evaluation/split_policy.py` | `src/oqp/research/splits.py` | promoted | Chronological split, purge, and embargo policy. |
| `evaluation/multiple_testing.py` | `src/oqp/research/multiple_testing.py` | promoted | Trial hash and p-value adjustment helpers. |
| `evaluation/statistical_tests.py` | `src/oqp/research/statistical_tests.py` | promoted | Alpha evidence tests and Sharpe p-value helper. |
| `evaluation/research_trial_ledger.py` | `src/oqp/research/trials.py` | promoted | Research trial history, vertical metadata, and multiple-testing refresh. |
| `data_engine/instrument_master.py` | `src/oqp/data/instruments.py` | promoted | CN futures physical metadata, fee profile, tick size, and US equity placeholder registry. |
| `config.py` market taxonomy | `src/oqp/contracts/market_vertical.py` | promoted | Market vertical enum, aliases, and public asset taxonomy metadata. |
| `ui_v2/asset_taxonomy.py` | `src/oqp/data/asset_taxonomy.py` | promoted | Shared taxonomy UI/dataframe helpers and lane metadata. |
| `evaluator.py` execution desk | `src/oqp/research/backtesting/` | partially promoted | Capital policy, trade policy, portfolio optimizer, and execution modes are promoted; legacy evaluator remains a private alpha workflow until its factor/backtest handoff is sanitized. |
| `cpp_engine/` | `src/oqp/native/` | bridge temporarily | Source promoted; legacy extension remains fallback while scripts migrate. |
| `benchmark_engine/` | `src/oqp/research/backtesting/benchmarks.py` | promoted | Dynamic equal-weight, absolute, risk-free, named-index, and sector-neutral benchmark generators for research backtests; alpha wrapper remains. |
| `tick_pulse/` | `src/oqp/research/tick_pulse/` | promoted | Native bridge, C++ RTV bridge, feature building, backend selection, asset ranking, heuristic calibration, tick XGBoost, gatekeeper, and tick-ML cache helpers are promoted; saved parquet/cache/model artifacts remain private. |
| `ml_engine/artifact_store.py` | `src/oqp/research/artifacts.py` | promoted | Model reproducibility layer with stable paths and hashes. |
| `ml_engine/model_registry.py` | `src/oqp/research/model_registry.py` | promoted | SQLite registry for artifact, data, feature, metric, and split provenance. |
| `ml_engine/feature_governance.py` | `src/oqp/research/ml/feature_governance.py` | promoted | Feature detection, missingness, IC stability, turnover proxy, correlation clusters, PCA baseline, and keeper shortlist. |
| `ml_engine/oos_mda.py` | `src/oqp/research/ml/oos_mda.py` | promoted | Purged walk-forward mean-decrease-accuracy audits for out-of-sample feature importance. |
| `ui_v2/config.py` text/theme | `src/oqp/ui/research_dashboard_config.py` | promoted | Shared bilingual research UI text and theme helpers; alpha wrapper keeps local DB/log paths. |
| `ml_engine/supervised_model_base.py` and model helpers | `src/oqp/research/ml/` | promoted | Walk-forward supervised base, LightGBM wrapper, XGBoost training engine, and model factory now use promoted artifact/registry helpers. |
| `regime_engine/hmm_regime.py` and `train_macro_hmm.py` | `src/oqp/intelligence/regime_engine/` | promoted | HMM/GMM-HMM models and macro-HMM training utilities are import-safe; alpha paths remain wrappers while `.pkl` artifacts stay private. |
| `risk_engine/risk_factor_breadth.py` | `src/oqp/risk/factor_breadth.py` | promoted | PCA covariance breadth engine now uses promoted instrument metadata; alpha path is a wrapper. |
| `state_space_engine/` | `src/oqp/research/state_space/` | promoted | Opportunity scoring, spread models, opportunity scanning, DKF filters, diagnostics, and relationship lab helpers are promoted. |
| `latent_engine/` | `src/oqp/research/latent/` | promoted | Generic VAE/VQ-VAE encoders, temporal panel windows, STORM-style comparison helpers, and codebook diagnostics are promoted; trained latent artifacts stay private. |
| `factors/fac_*.py` | none by default | keep private | Publish only sanitized retired examples through allowlist. |
| `data_cache/`, `execution_logs/`, `metadata/` | none | keep private | Runtime or research artifacts, not source integration targets. |
| `ui_v2/` | `apps/research_dashboard/` | late | Migrate one page at a time after backend extraction. |
| `archive/`, `backtest_1-main/`, `manager_research_demo/` | `departments/archive/` if needed | archive/reference | Do not mix into active OQP core. |

## Immediate Sequence

1. Completed: promote the research contract and evaluation hygiene modules.
2. Completed: leave compatibility wrappers in the old alpha-lab paths.
3. Completed: add OQP-level tests that import `oqp.research` directly.
4. Completed: run the old alpha-lab tests for the wrapped modules to confirm
   no script imports were broken.
5. Completed: promote the trial ledger and model artifact registry.
6. Completed: promote instrument metadata and market taxonomy.
7. Completed: migrate risk-factor breadth onto the promoted instrument
   metadata.
8. Completed: migrate tick-pulse asset ranking and state-space opportunity
   scanning onto the promoted instrument/risk helpers.
9. Completed: migrate tick-pulse feature plumbing.
10. Completed: migrate state-space DKF/filter modules.
11. Completed: migrate benchmark engine.
12. Completed: migrate ML feature governance/OOS MDA.
13. Completed: migrate supervised ML base, LightGBM/XGBoost engines, and model
    factory.
14. Completed: migrate execution capital/trade policy, optimizer, and execution
    mode bridge.
15. Completed: migrate diagnostics, regime HMM training, and latent feature
    research utilities.
16. Completed: migrate tick-pulse C++ bridge, heuristic calibration, tick
    XGBoost, gatekeeper, and tick-ML cache utilities.
17. Next: migrate or retire remaining UI pages one-by-one after each page's
    backend logic has a promoted OQP owner.
18. Manual heavy migration map: use
    `departments/research/manual_heavy_migration_map.md` when moving
    live factors, bulky data files, model artifacts, and remaining UI pieces by
    hand.

## Completion Criteria For This Slice

- New code can import factor contracts, dataset policy, split policy, metrics,
  p-value adjustment, statistical evidence, trial records, and model artifact
  provenance from `oqp.research`.
- New code can import market vertical taxonomy and instrument metadata from
  `oqp.contracts` and `oqp.data`.
- New code can import PCA risk-factor breadth analysis from `oqp.risk`.
- New code can import tick-pulse asset ranking and state-space opportunity
  scanning from `oqp.research`.
- New code can import tick-pulse feature builders and backend selection from
  `oqp.research.tick_pulse`.
- New code can import state-space DKF filters, diagnostics, relationship
  helpers, and opportunity scanning from `oqp.research.state_space`.
- New code can import benchmark generators from `oqp.research.backtesting` or
  `oqp.research`.
- New code can import ML feature governance and OOS MDA audits from
  `oqp.research.ml` or `oqp.research`.
- New code can import supervised ML base/model factory helpers from
  `oqp.research.ml` or `oqp.research`.
- New code can import execution capital/trade policy and execution mode bridge
  helpers from `oqp.research.backtesting` or `oqp.research`.
- New code can import regime HMM/GMM-HMM and macro-HMM training utilities from
  `oqp.intelligence.regime_engine`.
- New code can import latent VAE/VQ-VAE, temporal panel, and codebook
  diagnostics from `oqp.research.latent` or `oqp.research`.
- New code can import tick-pulse calibration, tick XGBoost, gatekeeper, C++
  RTV bridge, and tick-ML cache utilities from `oqp.research.tick_pulse`.
- Old alpha-lab imports still resolve.
- No live factor implementation, cached market data, execution logs, or model
  artifact is required by the promoted tests.
