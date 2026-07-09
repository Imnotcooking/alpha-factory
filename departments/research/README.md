# Research Department Map

Last reviewed: 2026-07-03

This folder is the research department layer of the repo. It is not the main
source-code home and it is not runtime storage. Its job is to hold research
policy, private factor material, public/private boundary docs, and
human-readable research notes.

## Where The Former Alpha Lab Went

The old `alpha_research_lab/` folder was deliberately split by responsibility:

| Responsibility | Current home | Commit posture |
| --- | --- | --- |
| Reusable research engines, backtesting, ML, diagnostics, and factor contracts | `src/oqp/research/` | Commit source and tests. |
| Regime/intelligence engines used by ops/research | `src/oqp/intelligence/` | Commit source and tests. |
| Native/C++ acceleration | `src/oqp/native/` | Commit source/build config, not build products. |
| Research Streamlit UI | `apps/research_dashboard/` | Commit app code and UI helpers. |
| Research dashboard shared text/config | `src/oqp/ui/` | Commit shared bilingual catalogs and theme helpers. |
| Backtest runner scripts | `scripts/research/` | Commit lightweight runners; keep outputs in runtime. |
| Private live factor recipes | `departments/research/factors/` | Private by default; do not publish live edge. |
| Public sanitized examples | `departments/research/retired_factors/` | Commit only reviewed examples. |
| CN futures market data | `runtime/data/futures_cn/{daily,intraday,tick}/` | Local/private vendor/static market data. |
| CN equity market data | `runtime/data/equity_cn/{daily,intraday,tick}/` | Planned local QMT/static market data. |
| CN option market data | `runtime/data/options_cn/{daily,tick}/` | Planned local commodity option chain data. |
| US API caches | `runtime/data/{equity_us,options_us}/api_cache/` | Optional local cache only; vendors remain API-owned. |
| Feature matrices, regime outputs, research metadata | `runtime/data/` | Local/private generated research state. |
| Research databases | `runtime/db/research/` | Local/private runtime state. |
| Backtest returns, trades, TCA, models, diagnostics | `runtime/artifacts/research/` | Local/private runtime artifacts. |
| Dashboard/job logs | `runtime/logs/` | Local/private operational logs. |

## Active Subfolders

`factors/`

- Private live factor recipes grouped by strategy family.
- This is the correct holding area for live edge before any strategy is
  sanitized, tested, and promoted.
- These files should not be moved directly into `src/oqp`.
- `README.md` and `factor_template_private.py` are the only GitHub-safe files
  in this folder; live `fac_*.py` recipes remain ignored/private.

`retired_factors/`

- Public, sanitized, educational factor examples.
- These are safe examples of the factor contract shape, not live research edge.
- New files here should be reviewed against `public_allowlist.md`.
- Use `factor_template_retired_public.py` as the GitHub-safe starting point.

Policy docs now live directly under this folder:

- `public_private_boundary.md`
- `public_allowlist.md`
- `integration_plan.md`
- `manual_heavy_migration_map.md`
- `decommission_readiness.md`

## Working Rules

1. Put reusable, testable research code in `src/oqp/research/`.
2. Put private factor recipes in
   `departments/research/factors/`.
3. Put generated data, databases, models, and backtest outputs under
   `runtime/`. Market data is market-first: for example CN futures files
   belong under `runtime/data/futures_cn/{daily,intraday,tick}/`; generated
   feature/regime research state stays under `runtime/data/`.
4. Put Streamlit page code under `apps/research_dashboard/`.
5. Put shared dashboard text/config under `src/oqp/ui/`.
6. Do not recreate root `data/`, root `logs/`, or the deleted
   `alpha_research_lab/` folder.

## Cleanup Status

Current cleanup:

- Root `logs/` was removed; use `runtime/logs/`.
- `departments/research/**/__pycache__/` generated folders were removed.
- The unused `experiment_artifacts/` and `legacy_offline_lab/` scaffolds were
  removed.
