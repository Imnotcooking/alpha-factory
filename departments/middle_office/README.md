# Middle Office Department Map

Last reviewed: 2026-07-17

Middle Office is the broker-neutral account truth, valuation-control, and
reconciliation layer in Alpha Factory. It observes account state, normalizes evidence,
reports breaks, and records their resolution. It never places trades.

## Ownership Boundary

| Responsibility | Canonical home | Commit posture |
| --- | --- | --- |
| Broker adapters and connection profiles | `src/oqp/brokers/` | Commit source and tests; never credentials. |
| Canonical account models, ledger, and reconciliation logic | `src/oqp/accounts/` | Commit source and tests. |
| Legacy portfolio ingestion and valuation during migration | `src/oqp/portfolio/` | Commit source and tests; do not add new account contracts here. |
| Account policies, controls, and operating runbooks | `departments/middle_office/` | Commit docs and lightweight specifications. |
| Operational presentation | `apps/ops_dashboard/` | Display package-owned results; do not implement controls in Streamlit. |
| Private account state and evidence | `runtime/` | Never commit databases, exports, account JSON, or screenshots. |

Middle Office may read from broker adapters and account state. Order creation,
approval, routing, and submission remain under Trading. Market and instrument
data ownership remains under Data Platform.

## Start Here

Use the broker-free Ops Dashboard to understand canonical account and portfolio
views before configuring an adapter:

```bash
oqp init --profile demo
oqp dashboard ops
python -m pytest -q tests -k "account or portfolio or ops"
```

## Department Files

- `account_snapshot_contract.md`: normalized account object and ledger contract.
- `account_sources.yaml`: validated inventory of account sources, writers, and
  freshness expectations.
- `source_of_truth.md`: precedence between brokers, manual holdings, unified
  reporting, and migration ledgers.
- `reconciliation_policy.md`: comparison keys, tolerances, break severity, and
  resolution lifecycle.
- `valuation_policy.md`: pricing, FX, multiplier, NAV, and PnL controls.
- `manual_adjustments.md`: governance for externally maintained holdings and
  approved manual values.
- `runbooks/daily_close.md`: account snapshot and control sequence.
- `runbooks/break_resolution.md`: evidence-preserving break investigation.
- `retirement_audit.md`: historical record of the retired standalone app.

## Current Implementation Map

| Capability | Implementation |
| --- | --- |
| Canonical account objects | `src/oqp/accounts/models.py` |
| Unified account ledger | `src/oqp/accounts/ledger.py` |
| Account source catalog | `src/oqp/accounts/source_catalog.py` |
| Read-only snapshot reconciliation | `src/oqp/accounts/reconciliation.py` |
| Manual external positions | `src/oqp/accounts/manual_external.py` |
| Unified live materialization | `src/oqp/accounts/unified_snapshot.py` |
| Broker snapshot conversion | `src/oqp/accounts/converters.py` |
| Broker profiles and adapters | `src/oqp/brokers/` |
| Account health checks | `src/oqp/ops/portfolio_health.py`, `src/oqp/ops/paper_health.py` |
| Operational visibility | `apps/ops_dashboard/` |

## Runtime Boundary

| Data | Runtime path |
| --- | --- |
| Canonical account ledger | `runtime/db/accounts/account_ledger.db` |
| Legacy portfolio ledger | `runtime/db/portfolio/portfolio_ledger.db` |
| Paper trading ledger | `runtime/db/paper_trading/paper_trading.db` |
| Broker CSV imports | `runtime/imports/broker_exports/` |
| Portfolio and manual state | `runtime/state/portfolio/` |
| Snapshot backups | `runtime/exports/portfolio_snapshots/` |
| Health and operational logs | `runtime/logs/` |

These paths are private runtime state. A path listed in the source catalog is a
contract, not evidence that a source is configured, available, or fresh.
`freshness_max_age_hours: 0` denotes an event-driven or migration source with no
automatic age SLA; it does not mean the source is always fresh.

## Change Procedure

1. Register or amend the source in `account_sources.yaml`.
2. Normalize inbound state into `AccountSnapshot` and related canonical objects.
3. Keep comparison logic in `src/oqp/accounts/`, not in a dashboard page.
4. Define tolerances and ownership before enabling automated break alerts.
5. Add focused tests for matching, missing data, and tolerance behavior.
6. Preserve raw evidence and redact account identifiers in public output.

## Guardrails

- Live broker profiles remain read-only and `ALLOW_LIVE_TRADING=false` remains
  the default.
- Reconciliation never repairs source data by overwriting it.
- Manual state is not broker evidence and must remain visibly attributable.
- A fresh file is not necessarily a freshly priced position.
- API keys and account identifiers come from private runtime configuration only.
- Do not restore fallbacks to the retired `Middle_Office/` root.
