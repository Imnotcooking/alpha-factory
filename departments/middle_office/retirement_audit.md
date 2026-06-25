# Middle Office Retirement Audit

Date: 2026-06-25

## Result

The root `Middle_Office/` folder has been archived to:

`departments/archive/legacy_middle_office/`

The active money dashboard now uses native pages and shared modules:

- `apps/money_dashboard/app.py`
- `apps/money_dashboard/pages/01_Stock_Valuation.py`
- `apps/money_dashboard/pages/02_Risk_Management.py`
- `apps/money_dashboard/pages/03_Options_Desk.py`
- `src/oqp/portfolio/`
- `src/oqp/investing/`
- `src/oqp/risk/`
- `src/oqp/options/`

## Active References

No active Streamlit page imports or executes the archived Middle Office app or
legacy pages.

Remaining `middle_office` references are intentionally retained for one of four
reasons:

- Compatibility names in the IBKR adapter that still produce the legacy row
  shape consumed by portfolio ingestion.
- Legacy fallback paths for old watchlists, FMP/Gemini key JSON files, manual
  inputs, raw broker CSV exports, and old ledger state.
- Backward-compatible exported constants and aliases in `src/oqp/portfolio/`.
- Documentation describing the migration.

## Archive-Aware Fallback

`src/oqp/config/paths.py` now resolves the legacy Middle Office root in this
order:

1. `OQP_LEGACY_MIDDLE_OFFICE_ROOT`
2. `Middle_Office/`, if it exists
3. `departments/archive/legacy_middle_office/`

This keeps migration fallbacks working locally after the archive move and keeps
server deployments safe if the old root folder still exists there.

## Removed Shadow Files

The following stale duplicate files were removed because they could appear as
extra Streamlit pages or stale modules:

- `apps/money_dashboard/app 2.py`
- `apps/money_dashboard/pages/01_Stock_Valuation 2.py`
- `apps/money_dashboard/pages/02_Risk_Management 2.py`
- `apps/paper_trading_dashboard/app 2.py`
- `src/oqp/brokers/ibkr 2.py`
- `src/oqp/risk/__init__ 2.py`
- `tests/test_middle_office_etl 2.py`
- `Middle_Office/Portfolio/etl_engine 2.py`
- `Middle_Office/db_setup 2.py`

## Phase 2A Ported

- The legacy options strategy router from `pages/3_Options_Desk.py` has been
  extracted into `src/oqp/options/analytics.py` as `score_option_strategies`.
- The native money dashboard Options Desk now shows a Strategy Fit tab before
  individual contract scans.
- The router ranks strategy families only. It does not create orders, write
  trade proposals, or bypass the paper-trading safety layer.

## Phase 2B Ported

- Multi-leg option payoff simulation now lives in `src/oqp/options/analytics.py`
  via `OptionLeg` and `simulate_multi_leg_options`.
- The native Options Desk now scans calendars, iron condors, call butterflies,
  call/put ratio spreads, call/put backspreads, and bull/bear vertical spreads.
- These scanners return candidate analytics only: debit/credit, max profit,
  max loss where defined, probability of profit, expected value, VaR 95, and
  structure text. They do not create orders.

## Phase 2C Ported

- Options scanner candidates can now become paper-trading proposal drafts via
  `src/oqp/execution/options_bridge.py`.
- The native Options Desk writes proposal artifacts into the same
  `runtime/artifacts/trade_proposals/` queue used by research exports.
- This is still non-executing. Current paper policy blocks option asset classes
  unless explicitly enabled, and every draft must pass paper guardrails before
  anything could be routed.
- Option paper guardrails are config-driven through `PAPER_OPTIONS_ENABLED`,
  allowed underlyings, allowed strategies, max contracts, max premium, max
  defined risk, and max spread width.

## Still Worth Porting Later

- The ML macro oracle from `pages/2_Risk_Management.py`.
- Deep-value LEAPS fundamental filters from `pages/3_Options_Desk.py`, if they
  still fit the new FMP data contract.
- Optional later expansion of option execution policy by strategy family, such
  as different limits for condors, calendars, and debit spreads.
- Any reusable visual styling from `utils/theme.py`, if desired.

## Private Data

The archived folder still contains local/private ignored files such as secrets,
old broker exports, old SQLite databases, and manual input JSON. These remain
excluded by `.gitignore` and should not be published.
