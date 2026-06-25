# Middle Office Department

## Purpose

Middle Office is the operational truth layer for real holdings. It answers:

- What do the brokers say we own?
- What does our unified portfolio ledger say we own?
- Are cash, PnL, exposures, and risk views internally consistent?
- What breaks, exceptions, or manual inputs need attention?

This department should stay read-only with respect to broker accounts. It can ingest,
reconcile, report, and raise control warnings, but it should not place trades.

## Archived Legacy Source

The old side project now lives in `departments/archive/legacy_middle_office/`.
Treat it as a legacy source system. Its useful operational parts have been
extracted into shared modules and native dashboards; remaining files are
reference material, migration fallbacks, or private ignored local data.

Do not restore it to the repo root. New code should use runtime paths and shared
modules under `src/oqp/`.

## Legacy Inventory

| Legacy path | Current role | Target home |
| --- | --- | --- |
| Legacy path under `departments/archive/legacy_middle_office/` | Current role | Target home |
| --- | --- | --- |
| `app.py` | Legacy Streamlit command center for portfolio overview, macro pulse, allocation, risk metrics, and reporting | Main command center replaced by `apps/money_dashboard/app.py` |
| `Portfolio/etl_engine.py` | Compatibility wrapper for broker ingestion | Canonical implementation is now `src/oqp/portfolio/ingestion_job.py` plus `scripts/update_live_portfolio_snapshot.py` |
| `db_setup.py` | SQLite schema bootstrap for `historical_nav` and `live_positions` | Shared storage migrations under `src/oqp/storage/` |
| `pages/2_Risk_Management.py` | Legacy macro oracle, hedging, FMP features, beta hedge calculator | Operational risk and hedge sizing now live in `apps/money_dashboard/pages/02_Risk_Management.py` and `src/oqp/risk/`; the ML macro oracle remains a later extraction |
| `pages/3_Options_Desk.py` | Legacy options scanner, volatility models, strategy simulation | Native page now lives in `apps/money_dashboard/pages/03_Options_Desk.py`; shared analytics live in `src/oqp/options/` |
| `pages/5_Stock_Valuation.py` | Legacy FMP-backed DCF, peer valuation, watchlist, AI analyst | Native page now lives in `apps/money_dashboard/pages/01_Stock_Valuation.py`; shared investing logic lives in `src/oqp/investing/` |
| `GARCH_model.py` | GARCH volatility helper | Optional future model under `src/oqp/risk/` or `src/oqp/data/features/` |
| `HAR_model.py` | HAR volatility helper | Optional future model under `src/oqp/risk/` or `src/oqp/data/features/` |
| `Historical_Distribution.py` | Historical return distribution helper | Mostly replaced by `src/oqp/options/analytics.py`; keep as reference |
| `utils/theme.py` | Legacy Streamlit CSS | Fold into app-specific UI helpers only if still needed |

## Consumes

- IBKR account snapshots through TWS or IB Gateway.
- Broker CSV exports from Futubull and Trading212.
- Market prices from Yahoo Finance in the legacy app.
- Fundamentals and alternative data from FMP.
- Options Greeks from Polygon.io, now treated as Massive. In new code, use
  `massive` as the canonical vendor name and keep `polygon` as a legacy alias
  for old environment variables, class names, and artifacts.
- Manual cash and real-asset inputs.

## Produces

- Unified portfolio position snapshots.
- Broker-level cash and NAV summaries.
- Reconciliation warnings.
- Allocation and exposure reports.
- Risk and hedging diagnostics.
- Historical NAV and live position tables.

## Does Not Own

- Research factor promotion.
- Paper trading candidate approval.
- Order placement or live execution.
- API secrets or account passwords.
- Raw private broker exports in public GitHub history.

## Data Contracts To Preserve

### `live_positions`

Current SQLite table produced by `scripts/update_live_portfolio_snapshot.py`:

| Column | Meaning |
| --- | --- |
| `date` | Snapshot date |
| `broker` | Source broker |
| `ticker` | Broker or normalized symbol |
| `asset_type` | Equity, Option, or other instrument type |
| `shares` | Position quantity |
| `avg_cost` | Average entry cost |
| `current_price` | Latest broker or market price |
| `unrealized_pnl` | Broker or computed unrealized PnL |
| `currency` | Trading currency |
| `delta` | Per-unit or approximate position delta |
| `gamma` | Per-unit or approximate position gamma |

### `historical_nav`

Current SQLite table intended for portfolio equity curve tracking:

| Column | Meaning |
| --- | --- |
| `date` | NAV date |
| `total_net_worth` | Total portfolio NAV in USD |
| `total_cash` | Total cash/reserves |
| `portfolio_beta` | Estimated beta to benchmark |
| `daily_pnl` | Daily profit/loss |

## First Dissection Step

Started with `departments/archive/legacy_middle_office/Portfolio/etl_engine.py`.

Reason: it is the source of truth for the command center. Once ingestion is clean,
the dashboards become replaceable views over a stable portfolio snapshot contract.

Safe extraction order:

1. Define shared portfolio snapshot models in `src/oqp/portfolio/`. Done:
   `src/oqp/portfolio/snapshots.py`.
2. Move broker CSV parsers into small pure functions with tests. Done:
   `src/oqp/portfolio/broker_imports.py`.
3. Extract the SQLite `live_positions` / `historical_nav` schema and read/write
   helpers. Done: `src/oqp/portfolio/ledger.py`.
4. Wire daily NAV history, equity curve, and max-drawdown views. Done:
   the legacy command center writes `historical_nav`; the Ops dashboard reads it.
5. Extract portfolio valuation/NAV math from the legacy Streamlit page. Done:
   `src/oqp/portfolio/valuation.py`.
6. Add a scheduled server-side NAV updater. Done:
   `src/oqp/portfolio/nav_job.py` and `scripts/update_portfolio_nav.py`.
7. Replace direct `ib_insync` logic with the existing `src/oqp/brokers/ibkr.py`
   read-only adapter. Done: unified ingestion uses
   `fetch_ibkr_readonly_portfolio_snapshot(...)`.
8. Extract the live ingestion job itself. Done:
   `src/oqp/portfolio/ingestion_job.py` and
   `scripts/update_live_portfolio_snapshot.py` now write runtime outputs under
   `runtime/db/portfolio/` and `runtime/state/portfolio/`.
9. Keep the old ETL path as a wrapper until server cron, docs, and dashboards
   are fully migrated.
10. Replace the Money dashboard entrypoint with a native page over the runtime
   ledger. Done: `apps/money_dashboard/app.py`.
11. Extract the Stock Valuation page. Done:
   `apps/money_dashboard/pages/01_Stock_Valuation.py` now renders natively,
   `src/oqp/investing/stock_valuation.py` owns FMP/Yahoo/DCF helpers, and
   `src/oqp/investing/watchlist.py` owns runtime watchlist storage.
12. Extract the operational Risk Management page. Done:
   `apps/money_dashboard/pages/02_Risk_Management.py` now renders natively over
   the runtime ledger, and `src/oqp/risk/portfolio.py` owns concentration,
   exposure, drawdown, VaR/CVaR, beta-hedge, Black-Scholes, and manual hedge
   diagnostics. The legacy ML macro oracle remains in
   `departments/archive/legacy_middle_office/pages/2_Risk_Management.py` until
   it is redesigned as a separate research/risk model workflow.
13. Extract the Options Desk page. Done:
   `apps/money_dashboard/pages/03_Options_Desk.py` now renders natively,
   and `src/oqp/options/analytics.py` owns Black-Scholes pricing, implied
   volatility, volatility snapshots, historical holding-period odds, Monte
   Carlo payoff simulation, and first-pass long option / cash-secured put scans.
   The old 2,151-line scanner remains as reference for gradually porting
   specialized structures like calendars, condors, butterflies, and ratio
   spreads.
14. Only after that, retire or archive the legacy Streamlit pages.

## Private Data Guardrail

Before this repo goes public, review and exclude:

- `departments/archive/legacy_middle_office/.streamlit/secrets.toml`
- `departments/archive/legacy_middle_office/fmp_config.json`
- any generated `api_keys.json`
- `departments/archive/legacy_middle_office/Portfolio/raw_data/`
- `departments/archive/legacy_middle_office/Portfolio/clean_data/`
- broker CSV exports, account snapshots, and local SQLite databases
