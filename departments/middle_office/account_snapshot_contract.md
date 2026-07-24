# Unified Account Snapshot Contract

The account layer is the shared language between live monitoring, paper
trading, broker imports, dashboards, and reconciliation controls.

## Why It Exists

Before this contract, live and paper state were useful but separate:

- live IBKR and broker imports wrote `live_positions` and `historical_nav`
- paper IBKR wrote `paper_account_snapshots`, `paper_positions`, and `paper_nav`
- dashboards needed to know which ledger belonged to which account type

The unified account layer normalizes those sources into a common account ledger:

```text
runtime/db/accounts/account_ledger.db
```

## Canonical Objects

The canonical dataclasses live in `src/oqp/accounts/models.py`:

- `AccountSnapshot`
- `PositionSnapshot`
- `CashSnapshot`
- `NavSnapshot`
- `TradeEvent`

These are intentionally small and broker-neutral. IBKR live, IBKR paper,
future broker imports, and later execution events can all map into this shape.

## Storage Tables

The unified SQLite schema lives in `src/oqp/accounts/ledger.py`:

- `account_snapshots`
- `account_positions`
- `account_cash`
- `account_nav`
- `account_trade_events`

All five tables are active. Paper proposal, routing, submission, and review
flows already write `account_trade_events`; Ops and journal views consume them.
Broker fill matching remains a separate reconciliation stage.

## Current Writers

The live snapshot job writes both:

```text
runtime/db/portfolio/portfolio_ledger.db
runtime/db/accounts/account_ledger.db
```

The paper snapshot job writes both:

```text
runtime/db/paper_trading/paper_trading.db
runtime/db/accounts/account_ledger.db
```

QMT live or paper snapshots write directly to the account ledger through
`scripts/trading/update_qmt_account_snapshot.py`. Approved manual external holdings are
synchronized by `scripts/data/refresh_manual_external_holdings.py`, and
`scripts/ops/update_unified_live_account_snapshot.py` creates the consolidated
`unified_live` view.

The legacy portfolio and paper ledgers remain sidecars during migration. The
account ledger and fresh `unified_live` profile are already authoritative for
the current account reporting helpers and Ops health controls.

## Current And Next Consumers

The next good consumers are:

- Ops Dashboard: shows live/paper account state, freshness, and trade events
- Ops Dashboard: display approved strategy runtime posture
  from account state, risk, regime, and allocation context
- Reconciliation: the package-owned in-memory comparison foundation lives in
  `src/oqp/accounts/reconciliation.py`; persistence and break workflow remain
  the next controlled integration

## Source Inventory

`departments/middle_office/account_sources.yaml` is the committed source
inventory. `src/oqp/accounts/source_catalog.py` validates it without opening a
broker connection or reading private account data.
