# Middle Office Retirement Audit

Date: 2026-07-08

## Result

The standalone Middle Office app has been fully retired from active runtime
architecture. The previous legacy archive has been removed after the useful
logic was extracted into shared modules.

Active code now uses:

- `apps/ops_dashboard/`
- `src/oqp/accounts/`
- `src/oqp/portfolio/`
- `src/oqp/investing/`
- `src/oqp/risk/`
- `src/oqp/options/`
- runtime paths under `runtime/`

## Runtime-Only Policy

The codebase no longer resolves or reads the old `Middle_Office/` root, the
deleted archive folder, or old Middle Office JSON/database files as fallbacks.

Runtime data must be placed in:

- `runtime/imports/broker_exports/`
- `runtime/db/portfolio/`
- `runtime/db/accounts/`
- `runtime/state/portfolio/`
- `runtime/state/investing/`
- `runtime/exports/portfolio_snapshots/`

## Active References

No active Streamlit page imports or executes the retired Middle Office app or
legacy pages. Remaining generic terms such as `middle_office/` in department
docs refer to this active operational department, not to the old app archive.

Other `legacy_path` fields in research/model code are unrelated metadata for
native backends and model registry provenance. They are not Middle Office
fallbacks.

## Removed During Retirement

- Old Streamlit Middle Office app and pages.
- Old `Portfolio/etl_engine.py` compatibility wrapper.
- Old Middle Office Streamlit config/theme files.
- Old local fallback paths for FMP/Gemini key JSON.
- Old fallback path for Middle Office watchlists.
- Old fallback path for broker CSV imports.
- Old fallback path for manual NAV inputs and `ibkr_metrics.json`.
- Old exported portfolio aliases from the retired app.
- Duplicate `tests/test_middle_office_etl.py`, now covered by canonical
  portfolio ingestion tests.

## Still Worth Porting Later

- A redesigned macro/regime oracle if the old idea is still useful.
- More specialized options structures, but only through `src/oqp/options/`.
- Any reusable styling idea, but only through active UI helpers.

## Private Data

The deleted archive contained local/private ignored files such as secrets, old
broker exports, old SQLite databases, and manual input JSON. They must stay out
of public Git history. The active runtime equivalents remain ignored.
