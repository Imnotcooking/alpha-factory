# Research Department Map

Last reviewed: 2026-07-20

This folder is the research department layer of the repo. It is not runtime
storage. Its job is to hold research policy, private factor material,
experiment workflows, public/private boundary docs, and human-readable
research notes.

## Start Here

```bash
oqp init --profile demo
oqp dashboard research
python -m pytest -q tests -k research
```

New factors begin with the public factor template and package-owned backtesting
contracts; generated evidence belongs in `runtime/artifacts/research/`.

## Where The Former Alpha Lab Went

The old `alpha_research_lab/` folder was deliberately split by responsibility:

| Responsibility | Current home | Commit posture |
| --- | --- | --- |
| Reusable research engines, backtesting, ML, diagnostics, and factor contracts | `src/oqp/research/` | Commit source and tests. |
| Reusable ML model families, including regimes and latent models | `src/oqp/research/ml/` | Commit source, compatibility tests, and synthetic parity tests. |
| Research experiment and artifact workflows | `departments/research/workflows/` | Commit reproducible orchestration; keep generated outputs in runtime. |
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

`docs/`

- Research governance, migration records, and project notes.
- Start with `docs/README.md`; notes are grouped by purpose instead of being
  left loose in the department root.

`factors/`

- Flat private registry for live factor recipes and research components.
- Market, frequency, family, and lifecycle are declared in file metadata and
  `catalog.yaml`, then organized by the dashboard.
- This is the correct holding area for live edge before any strategy is
  sanitized, tested, and promoted.
- These files should not be moved directly into `src/oqp`.
- `README.md` and `factor_template_private.py` are the only GitHub-safe files
  in this folder; live `fac_*.py` recipes remain ignored/private.

`retired_factors/`

- Public, sanitized, educational factor examples.
- These are safe examples of the factor contract shape, not live research edge.
- New files here should be reviewed against
  `docs/governance/public_allowlist.md`.
- Use `factor_template_retired_public.py` as the GitHub-safe starting point.

`routers/`

- Private causal allocation rules across frozen strategy sleeves.
- Router recipes use stable `rtr_*` IDs; reusable validation and composition
  logic remains in `src/oqp/research/strategy_routing/`.

`strategies/`

- Reproducible factor-portfolio recipes that reference stable factor IDs.
- Equal/static factor weights, normalization, missing-data handling, and
  execution constraints are declared in YAML rather than hidden in a page.
- Reusable composition logic remains in `src/oqp/research/factor_portfolios/`;
  generated runs remain in `runtime/artifacts/research/`.

`workflows/`

- Reproducible recipes that compose reusable engines into an experiment,
  scan, or artifact-producing run.
- Workflow code may import `src/oqp` components. Reusable `src/oqp` modules
  must not import the department layer.
- Dashboard-only loading, caching, controls, and rendering stay under
  `apps/research_dashboard/`.

Documentation is grouped under `docs/`:

- `docs/governance/` contains the public/private boundary and allowlist.
- `docs/governance/ml_model_library_architecture.md` explains the reusable ML
  package taxonomy and compatibility rules.
- `docs/migration/` contains completed architecture and decommission records.
- `docs/intraday_cn_futures/` contains the related research-note series.

## Working Rules

1. Put reusable estimators, transforms, and deterministic engines in
   `src/oqp/research/`.
2. Put experiment orchestration and artifact workflows in
   `departments/research/workflows/`.
3. Put private factor recipes in
   `departments/research/factors/`.
4. Put generated data, databases, models, and backtest outputs under
   `runtime/`. Market data is market-first: for example CN futures files
   belong under `runtime/data/futures_cn/{daily,intraday,tick}/`; generated
   feature/regime research state stays under `runtime/data/`.
5. Put dashboard application services and Streamlit page code under
   `apps/research_dashboard/`.
6. Put shared dashboard text/config under `src/oqp/ui/`.
7. Do not recreate root `data/`, root `logs/`, or the deleted
   `alpha_research_lab/` folder.

## Cleanup Status

Current cleanup:

- Root `logs/` was removed; use `runtime/logs/`.
- `departments/research/**/__pycache__/` generated folders were removed.
- The unused `experiment_artifacts/` and `legacy_offline_lab/` scaffolds were
  removed.
