# Unified Account Snapshot Contract

The account layer is the shared language between live monitoring, paper
trading, broker imports, dashboards, and future reconciliation.

## Why It Exists

Before this contract, live and paper state were useful but separate:

- live IBKR and broker imports wrote `live_positions` and `historical_nav`
- paper IBKR wrote `paper_account_snapshots`, `paper_positions`, and `paper_nav`
- dashboards needed to know which ledger belonged to which account type

The unified account layer keeps those existing ledgers intact while also
writing a common account ledger:

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

The first four are active now. `account_trade_events` is reserved for paper
fills, future live trade monitoring, and broker reconciliation.

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

This is a sidecar migration. Existing dashboards keep working from their
current ledgers, while new dashboards can start reading the unified account
ledger.

## Next Consumers

The next good consumers are:

- Ops Dashboard: show live/paper account freshness from one table
- Ops Dashboard Intelligence page: manage approved strategy runtime posture
  from account state, risk, regime, and allocation context
- Reconciliation: compare broker-specific ledgers against unified snapshots
