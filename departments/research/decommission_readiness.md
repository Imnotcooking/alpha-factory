# Alpha Research Lab Decommission Readiness

This document records the completed decommission of the former
`alpha_research_lab/` folder. The migration goal was not only cleaner folders:
it was to make the upgraded architecture reproducible without depending on
legacy import paths, local data caches, or dashboard wrapper scripts.

## Current Status

The former lab folder has been deleted. Its responsibilities now live here:

| Former lab responsibility | Current architecture home |
| --- | --- |
| Market data caches | `runtime/data/alpha_lab/market_data/` |
| Feature matrices | `runtime/data/alpha_lab/feature_store/` |
| Regime probability outputs | `runtime/data/alpha_lab/regime/` |
| Research metadata | `runtime/data/alpha_lab/metadata/` |
| Research and optimization databases | `runtime/db/research/alpha_lab/` |
| Backtest, model, diagnostic, and experiment artifacts | `runtime/artifacts/research/alpha_lab/` |
| Operational logs | `runtime/logs/` |
| Private factor recipes | `departments/research/factors/` |
| Public retired examples | `departments/research/retired_factors/` |
| Research dashboard app code | `apps/research_dashboard/` |
| Backtest runners | `scripts/research/` |
| Backtest evaluator | `src/oqp/research/backtesting/evaluator.py` |
| Factor discovery | `src/oqp/research/factors.py` |

The active research modules under `src/oqp/research/`, `src/oqp/intelligence/`,
`src/oqp/data/`, and `src/oqp/native/` now carry the reusable architecture.

## What This Pass Completed

These deletion blockers have been removed:

1. `apps/research_dashboard/` now owns the Streamlit app and page source
   directly.
2. Root tests no longer import `alpha_research_lab.*`.
3. Private factor recipes import promoted `oqp.*` modules and presets.
4. Backtest CLIs were promoted to `scripts/research/`.
5. `AlphaEvaluator` was promoted to `src/oqp/research/backtesting/evaluator.py`.
6. Public examples were moved to
   `departments/research/retired_factors/`.
7. `manager_research_demo/` and `backtest_1-main/` were moved to repo root.

## Deleted Legacy Contents

The duplicate legacy tree was removed after the active app, scripts, tests, and
private factor registry stopped depending on it. Removed contents included the
old dashboard copy, runner/evaluator copies, compatibility wrappers, scratch
files, local caches, old tests, archive snippets, and compiled native byproducts.

## Deletion Gates

The decommission passed these gates:

| Gate | Required result |
| --- | --- |
| Dependency search | Active code under `apps/research_dashboard`, `src/oqp`, `scripts/research`, root `tests`, and private factors no longer depends on the deleted folder. |
| Dashboard ownership | Done: `apps/research_dashboard/` imports real app code directly, with no `apps._compat.run_legacy_streamlit_script` call into the old lab UI. |
| Backtest ownership | Done: backtest CLIs live under `scripts/research/` and reusable evaluator logic lives under `src/oqp/research/backtesting/`. |
| Factor namespace | Done for private registry recipes: private factor recipes import promoted `oqp.*` modules and presets. |
| Test ownership | Done for root tests. Legacy lab tests were deleted with the old folder. |
| Runtime paths | No source code writes to root `data/`, root `logs/`, or deleted lab runtime paths. |
| Native engine | C++ acceleration is owned by `src/oqp/native/`; no production or research path requires the deleted lab C++ build. |
| Reproducibility | Focused research tests and dashboard import checks pass with only `PYTHONPATH=src:.`. |
| Public hygiene | `python scripts/check_public_commit_hygiene.py` passes for the intended staged set. |

## Post-Delete Rule

Do not recreate the old lab folder. New work should land directly in the
upgraded architecture below.

## Safe Working Rule From Now On

Use these homes by default:

| New work | Destination |
| --- | --- |
| Reusable Python research code | `src/oqp/research/` |
| Regime intelligence | `src/oqp/intelligence/` |
| Data adapters and schemas | `src/oqp/data/` |
| Native/C++ acceleration | `src/oqp/native/` |
| Streamlit research app entrypoints | `apps/research_dashboard/` |
| Private factors and research notes | `departments/research/factors/` |
| Generated data and local artifacts | `runtime/data/`, `runtime/db/`, `runtime/artifacts/`, `runtime/logs/` |
