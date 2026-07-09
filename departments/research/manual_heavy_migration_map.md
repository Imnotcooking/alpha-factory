# Manual Heavy Migration Map

Last reviewed: 2026-07-02

Move status: heavy data/artifact files and live `fac_*.py` recipes were moved
on 2026-07-02. The remaining items in this document are review/refactor moves,
not direct file drops.

This is the hand-move checklist for the remaining heavy `alpha_research_lab`
surface. It intentionally ignores:

- `alpha_research_lab/manager_research_demo/`
- `alpha_research_lab/backtest_1-main/`

The rule of thumb is simple:

- Reusable, testable source code belongs in `src/oqp`.
- Private live factor recipes belong in the research strategy registry first,
  not directly in shared source.
- Raw data, trained models, result files, and databases belong under ignored
  `runtime/` paths.
- Streamlit pages belong in `apps/research_dashboard`; shared UI text/config
  belongs in `src/oqp/ui`.

## Safe Direct Moves

These can be moved by hand with little architectural risk.

| Alpha path | Move to | Notes |
| --- | --- | --- |
| `alpha_research_lab/data_cache/*raw*tick*.parquet` | `runtime/data/futures_cn/tick/` | Raw vendor/local CN futures tick files. Keep private and ignored. |
| `alpha_research_lab/data_cache/*_1m_*.parquet` | `runtime/data/futures_cn/intraday/` | CN futures intraday market data cache. |
| `alpha_research_lab/data_cache/*_1d_*.parquet`, `*.csv` | `runtime/data/futures_cn/daily/` | Daily CN futures market data. |
| `alpha_research_lab/data_cache/*universe.parquet` | `runtime/data/universes/` | Local research universes. Promote only schema/code, not files. |
| `alpha_research_lab/ML_Feature_Matrix.parquet` | `runtime/data/feature_store/ML_Feature_Matrix.parquet` | Generated feature-store output. |
| `alpha_research_lab/ML_Stacked_Matrix.parquet` | `runtime/data/feature_store/ML_Stacked_Matrix.parquet` | Generated ML matrix. |
| `alpha_research_lab/GMM_Rolling_Probabilities.parquet` | `runtime/data/regime/GMM_Rolling_Probabilities.parquet` | Generated regime probability output. |
| `alpha_research_lab/Macro_Index_V2.parquet` | `runtime/data/regime/Macro_Index_V2.parquet` | Generated macro regime input/output. |
| `alpha_research_lab/Macro_Regimes.parquet` | `runtime/data/regime/Macro_Regimes.parquet` | Generated macro regime output. |
| `alpha_research_lab/research_memory.db` | `runtime/db/research/research_memory.db` | Local research trial database. |
| `alpha_research_lab/optimization_memory.db` | `runtime/db/research/optimization_memory.db` | Local optimization database. |
| `alpha_research_lab/execution_logs/returns/` | `runtime/artifacts/research/returns/` | Backtest output. |
| `alpha_research_lab/execution_logs/trades/` | `runtime/artifacts/research/trades/` | Backtest output. |
| `alpha_research_lab/execution_logs/tca/` | `runtime/artifacts/research/tca/` | Execution-cost output. |
| `alpha_research_lab/execution_logs/tick_pulse_cache/` | `runtime/artifacts/research/tick_pulse_cache/` | Dashboard/research cache. |
| `alpha_research_lab/execution_logs/feature_importance/` | `runtime/artifacts/research/feature_importance/` | Model diagnostics output. |
| `alpha_research_lab/execution_logs/shap_regime_dna.csv` | `runtime/artifacts/research/diagnostics/shap_regime_dna.csv` | Generated diagnostic. |
| `alpha_research_lab/ml_engine/*.pkl` | `runtime/artifacts/research/models/` | Trained model artifacts. |
| `alpha_research_lab/ml_engine/*model*.json` | `runtime/artifacts/research/models/` | Trained model artifacts. |
| `alpha_research_lab/regime_engine/*.pkl` | `runtime/artifacts/research/regime_models/` | Trained regime artifacts. |
| `alpha_research_lab/metadata/*.csv` | `runtime/data/metadata/` | Private metadata unless sanitized. |
| `alpha_research_lab/data_engine/metadata/*.csv` | `runtime/data/metadata/` | Private metadata unless sanitized. |

## Factor Recipe Moves

Do not move live `fac_*.py` files directly into `src/oqp` as public source.
They are live research edge. The clean first landing zone is:

```text
departments/research/factors/
```

Recommended subfolders:

| Factor group | Files | Move to |
| --- | --- | --- |
| Daily technical / classic alpha | `fac_001` to `fac_004`, `fac_008` to `fac_010`, `fac_012` to `fac_024`, `fac_026`, `fac_028` to `fac_033`, `fac_039` to `fac_042`, `fac_050` to `fac_052` | `factors/daily_signals/` |
| ML / router / ensemble recipes | `fac_005`, `fac_006`, `fac_034` to `fac_038`, `fac_053` to `fac_057`, `fac_AI_001` | `factors/ml_routers/` |
| Tick / microstructure recipes | `fac_025`, `fac_043_Bearish_Breakdown.py`, `fac_043_Tick_Imbalance.py`, `fac_044_Relative_Velocity_Fade.py` | `factors/tick_pulse/` |
| State-space / stat-arb recipes | `fac_007`, `fac_045_Dual_Kalman_StatArb.py` | `factors/state_space/` |
| Macro / carry / fundamental variants | `fac_027`, `fac_029`, and any future macro/carry files | `factors/macro_carry/` |

After the files land there, refactor imports from alpha shims to promoted OQP
modules:

| Old import style | Preferred import |
| --- | --- |
| `from factors.contracts import ...` | `from oqp.research.factor_presets import ...` |
| `from factor_contract import ...` | `from oqp.research.contracts import ...` |
| `from ml_engine...` | `from oqp.research.ml...` or `from oqp.research.tick_pulse...` |
| `from state_space_engine import ...` | `from oqp.research.state_space import ...` |
| `from regime_engine...` | `from oqp.intelligence.regime_engine...` |
| `from tick_pulse...` | `from oqp.research.tick_pulse...` |
| `from execution...` | `from oqp.research.backtesting...` |

Only promote a factor into `src/oqp/research/factors/` after it is sanitized,
has synthetic tests, and no longer depends on private data/model artifacts.

## Review Before Moving

These are important, but should not be copied blindly because they still mix
source logic, local paths, runtime outputs, or legacy imports.

| Alpha path | Likely target | What to do first |
| --- | --- | --- |
| `alpha_research_lab/feature_engineering.py` | `src/oqp/research/features/feature_matrix.py` | Split reusable feature builders from local file IO; route generated matrices to `runtime/data/feature_store/`. |
| `alpha_research_lab/train_rolling_gmm.py` | `src/oqp/intelligence/regime_engine/asset_training.py` | Replace alpha imports with `oqp.intelligence.regime_engine`; add config object and output path argument. |
| `alpha_research_lab/evaluator.py` | `src/oqp/research/backtesting/` | Keep only reusable evaluation/backtesting classes; move CLI/runtime assumptions to scripts. |
| `alpha_research_lab/run_backtest.py` | `scripts/research/run_backtest.py` | Convert paths to `DATA_ROOT`/`ARTIFACT_ROOT`; import from `oqp.research`. |
| `alpha_research_lab/run_ml_backtest.py` | `scripts/research/run_ml_backtest.py` | Same as above, plus model artifact paths under `runtime/artifacts`. |
| `alpha_research_lab/oracle_evaluator.py` | `src/oqp/research/evaluation/` or `departments/research/` docs | Review for private assumptions before promotion. |
| `alpha_research_lab/data_engine/data_feed.py` | `src/oqp/data/vendors/` or `src/oqp/data/adapters/` | Strip local file paths/vendor assumptions; keep secrets out. |
| `alpha_research_lab/data_engine/tick_data_adapter.py` | `src/oqp/data/adapters/` | Promote schema normalization only; data files stay in `runtime/data`. |
| `alpha_research_lab/data_engine/builders/build_datasets.py` | `scripts/data_platform/build_alpha_datasets.py` | Treat as a local build script, not package code. |
| `alpha_research_lab/experiments/` | `runtime/artifacts/research/experiment_logs/` for outputs, `departments/research/` docs for narrative notes | Separate notes/config from generated files. |
| `alpha_research_lab/archive/` | none by default | Keep ignored or delete locally; promote only specific reusable ideas into active departments. |

## UI Moves

Most alpha Streamlit pages already have an app-side destination:

| Alpha page | App destination |
| --- | --- |
| `ui_v2/pages/09_Data_Artifact_Health.py` | `apps/research_dashboard/pages/01_Data_Health.py` |
| `ui_v2/pages/02_Pulse_Scan.py` | `apps/research_dashboard/pages/02_Pulse_Scan.py` |
| `ui_v2/pages/03_Tick_Event_Study.py` | `apps/research_dashboard/pages/03_Tick_Event_Study.py` |
| `ui_v2/pages/04_Arbitrage_Lab.py` | `apps/research_dashboard/pages/04_Arbitrage_Lab.py` |
| `ui_v2/pages/06_Regime_Characterisation.py` | `apps/research_dashboard/pages/05_Regime_Analysis.py` |
| `ui_v2/pages/08_Risk_Factor_Breadth_Lab.py` | `apps/research_dashboard/pages/06_Risk_Breadth.py` |
| `ui_v2/pages/07_Feature_Review.py` | `apps/research_dashboard/pages/07_Feature_Review.py` |
| `ui_v2/pages/01_Factor_Promotion_Pipeline.py` | `apps/research_dashboard/pages/08_Factor_Review.py` |
| `ui_v2/pages/05_Strategy_Comparison.py` | `apps/research_dashboard/pages/09_Strategy_Comparison.py` |

The app destination order follows the analyst workflow: trust gate, discovery,
relationship scan, regime/risk context, feature governance, promotion, and
strategy review.

Shared UI modules should be split as follows:

| Alpha UI path | Target |
| --- | --- |
| `ui_v2/config.py` | Already represented by `src/oqp/ui/research_dashboard_config.py`; move only missing text keys. |
| `ui_v2/tick_pulse_lab/text.py` | `src/oqp/ui/research_dashboard_config.py` if shared; otherwise app-local component text. |
| `ui_v2/tick_pulse_lab/*.py` | `apps/research_dashboard/components/tick_pulse_lab/` if UI-only; `src/oqp/research/tick_pulse/` only for reusable backend logic. |
| `ui_v2/arbitrage_lab/*.py` | `apps/research_dashboard/components/arbitrage_lab/` for UI; backend logic should use `src/oqp/research/state_space/`. |
| `ui_v2/views/*.py` | `apps/research_dashboard/components/views/` unless the file is pure backend logic. |

## Do Not Move

These should be deleted later or left ignored, not migrated:

- `__pycache__/`
- `.DS_Store`
- `*.cpython-*.so` build outputs, such as `cpp_engine/quant_core.cpython-313-darwin.so`
- duplicate Finder/version files with names like `config 2.py`,
  `instrument_master 2.py`, `codebook_diagnostics 2.py`
- generated `.parquet`, `.csv`, `.pkl`, `.json`, and `.db` files into `src/`
- live `fac_*.py` directly into public source

## After Moving

After each hand-move batch:

1. Update any hardcoded paths to use `DATA_ROOT=runtime/data` and
   `ARTIFACT_ROOT=runtime/artifacts`.
2. Replace alpha imports with `oqp.*` imports.
3. Run the closest focused tests.
4. Keep alpha wrappers temporarily until the research dashboard and scripts no
   longer import old paths.
5. Commit source and docs separately from private runtime/data moves.
