# Data Storage Map

Last reviewed: 2026-07-02

This repo intentionally has several paths with `data` in the name. They should
not all mean the same thing.

## Source Packages

`src/oqp/data/`

- Role: commit-ready source code for data contracts, adapters, vendor clients,
  instrument metadata, and taxonomy helpers.
- Git policy: commit source and tests.
- Examples: `oqp.data.instruments`, `oqp.data.asset_taxonomy`,
  `oqp.data.vendors`.

`departments/data_platform/`

- Role: architecture, process, and operating docs for data quality, market
  data, instrument master, vendors, and feature-store work.
- Git policy: commit docs and lightweight specs, not vendor extracts.

## Runtime Storage

`runtime/data/`

- Role: preferred local runtime data root. This is the path exposed through
  `DATA_ROOT=runtime/data` in `.env.example`.
- Git policy: ignored. Treat contents as generated, vendor-sourced, or
  environment-specific runtime state.
- Recommended use: new market-data caches, local feature-store outputs, and
  reproducible generated datasets that can be rebuilt from code/config.

`runtime/db/paper_trading/`

- Role: paper-trading SQLite ledger storage.
- Git policy: ignored through the parent `runtime/` rule.
- Default path: `runtime/db/paper_trading/paper_trading.db`.
- Override: `PAPER_TRADING_DB_PATH`.

`runtime/logs/`

- Role: operational logs and health/status JSON for dashboards, scheduled jobs,
  heartbeat checks, and server sync evidence.
- Git policy: ignored through the parent `runtime/` rule.
- Examples: `portfolio_snapshot_health.json`, `paper_trading_health.json`,
  dashboard `.log` files.

`runtime/artifacts/`

- Role: generated research/trading artifacts that may be audited or replayed,
  but should not be committed.
- Git policy: ignored through the parent `runtime/` rule.
- Examples: alpha backtest return series, trade ledgers, feature-importance
  outputs, model audit copies, strategy candidates, and trade proposals.

`data/`

- Role: retired legacy runtime state path.
- Git policy: ignored through `.gitignore`.
- Architectural note: do not recreate this folder. Paper-trading storage moved
  to `runtime/db/paper_trading/`.

`logs/`

- Role: retired legacy operational log path.
- Architectural note: do not recreate this folder. Use `runtime/logs/` for
  operational logs and `runtime/artifacts/` for research/backtest outputs.

## Research-Local Storage

`alpha_research_lab/data_cache/`

- Role: alpha-lab research data cache for local experiments and dashboard
  source files.
- Git policy: ignored/private. Do not promote cached market data.

`alpha_research_lab/metadata/` and `alpha_research_lab/data_engine/metadata/`

- Role: local research metadata and data-engine side files.
- Git policy: ignored/private unless a specific sanitized schema or template is
  intentionally promoted.

`notebooks/**/data/`

- Role: notebook-local scratch or teaching data.
- Git policy: not a production source of truth. Promote reusable logic into
  `src/oqp` and keep bulky data outside Git.

## Migration Guidance

- New reusable code belongs under `src/oqp/data`, `src/oqp/research`, or another
  domain package, not under root `data/`.
- New runtime artifacts should prefer `runtime/data/` or `runtime/artifacts/`.
- Root `data/` and root `logs/` should stay deleted. Do not add new runtime or
  source assets there.
