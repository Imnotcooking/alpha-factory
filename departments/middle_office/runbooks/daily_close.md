# Daily Account Close Runbook

## Preconditions

- Live broker profiles are read-only.
- `ALLOW_LIVE_TRADING=false` unless a separately approved execution process is active.
- Server time, market calendar, account profile, and reporting currency are confirmed.

## Sequence

1. Capture broker snapshots and retain their source timestamps and health state.
2. Import approved broker files without modifying the source exports.
3. Validate and synchronize manual external holdings.
4. Materialize canonical account snapshots in `account_ledger.db`.
5. Materialize `unified_live` only after its component sources are available.
6. Run reconciliation with the approved profile and operational cut.
7. Review position, cash, NAV, valuation, and event breaks.
8. Run account freshness and portfolio health checks.
9. Record unresolved critical breaks, owner, and next action before sign-off.
10. Preserve logs and evidence under private `runtime/` paths.

## Close Status

- **Complete:** required sources are fresh and no critical breaks remain open.
- **Complete with exception:** approved exception exists with owner and expiry.
- **Incomplete:** required source is absent/stale or a critical break is unresolved.

Never mark a close complete solely because the dashboard rendered successfully.
