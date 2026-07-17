# Middle Office Department

## Purpose

Middle Office is now the account truth and reconciliation layer inside the
active Alpha Factory architecture. It answers:

- What do the brokers say we own?
- What does the unified portfolio ledger say we own?
- Are cash, PnL, exposures, and risk views internally consistent?
- What breaks, exceptions, or manual inputs need attention?

This department is read-only with respect to broker accounts. It can ingest,
reconcile, report, and raise control warnings, but it should not place trades.

## Start Here

Use the broker-free Ops Dashboard to understand canonical account and portfolio
views before configuring an adapter:

```bash
oqp init --profile demo
oqp dashboard ops
python -m pytest -q tests -k "account or portfolio or ops"
```

## Active Homes

The old standalone Middle Office application has been retired. Its useful logic
has been extracted into package-owned services and the Ops dashboard:

| Responsibility | Active home |
| --- | --- |
| Live broker/account ingestion | `src/oqp/portfolio/ingestion_job.py` |
| Portfolio ledger schema and reads/writes | `src/oqp/portfolio/ledger.py` |
| Account snapshot ledger | `src/oqp/accounts/` |
| NAV valuation job | `src/oqp/portfolio/nav_job.py` |
| Live portfolio reporting | `src/oqp/portfolio/live_reporting.py` |
| Stock valuation and watchlists | `src/oqp/investing/` |
| Options analytics | `src/oqp/options/` |
| Operational dashboards | `apps/ops_dashboard/` |
| Server entrypoints | `scripts/update_live_portfolio_snapshot.py`, `scripts/update_portfolio_nav.py` |

New code should consume these modules directly. Do not add fallbacks to the old
`Middle_Office/` root or the deleted legacy archive.

## Runtime Data Boundary

Runtime state lives outside source-controlled code:

| Data | Runtime path |
| --- | --- |
| Broker CSV imports | `runtime/imports/broker_exports/` |
| Portfolio ledger | `runtime/db/portfolio/portfolio_ledger.db` |
| Account ledger | `runtime/db/accounts/account_ledger.db` |
| Portfolio state JSON | `runtime/state/portfolio/` |
| Investing watchlist | `runtime/state/investing/stock_watchlist.json` |
| Snapshot backups | `runtime/exports/portfolio_snapshots/` |

These paths are private runtime state and should remain ignored by Git.

## Data Contracts To Preserve

### `live_positions`

Current SQLite table produced by `scripts/update_live_portfolio_snapshot.py`:

| Column | Meaning |
| --- | --- |
| `date` | Snapshot date |
| `broker` | Source broker |
| `ticker` | Broker or normalized symbol |
| `asset_type` | Equity, option, cash, future, or other instrument type |
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
| `total_net_worth` | Total portfolio NAV in reporting currency |
| `total_cash` | Total cash/reserves |
| `portfolio_beta` | Estimated beta to benchmark |
| `daily_pnl` | Daily profit/loss |

## Guardrails

- API keys come from `.env` or process environment only.
- Manual inputs come from `runtime/state/portfolio/manual_inputs.json`.
- Watchlists come from `runtime/state/investing/stock_watchlist.json`.
- Broker exports come from `runtime/imports/broker_exports/`.
- Public commits must never include `.env`, Streamlit secrets, broker exports,
  SQLite databases, runtime JSON, CSV/parquet data, or account screenshots.
