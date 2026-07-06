# Trading Department Map

Last reviewed: 2026-07-03

This folder is the trading department layer of the repo. It is for process
docs, execution examples, approval boundaries, and runbooks. Active reusable
execution code lives under `src/oqp/`, while local broker/account state lives
under `runtime/`.

## What Belongs Here

| Folder | Role | Active code home |
| --- | --- | --- |
| `paper_trading/` | Paper-trading process docs, safety switches, submitter notes, and next-phase checklist. | `src/oqp/paper_trading/`, `scripts/run_paper_*`, Ops Dashboard Paper page. |
| `order_management/` | Public-safe example payloads for research signals and draft trade proposals. | `src/oqp/contracts/`, `src/oqp/execution/`, `src/oqp/paper_trading/order_router.py`. |

Broker gateway operating notes currently belong in
`departments/platform/deployment/` because they are deployment/runbook material.
Live-trading policy should be added here only when the live lane has a real
review, kill-switch, and signoff document to preserve.

## Current Trading Flow

```text
research signal
  -> strategy candidate / proposal artifact
  -> paper execution safety review
  -> dry-run ticket
  -> human/configured approval
  -> guarded paper broker submitter
  -> account snapshots and reconciliation
```

Current source/runtime homes:

- Trade proposal contracts: `src/oqp/contracts/`
- Execution guardrails: `src/oqp/execution/`
- Paper ledger/reviews/tickets/submitter: `src/oqp/paper_trading/`
- Broker adapters and profiles: `src/oqp/brokers/`
- Ops dashboard execution and paper pages: `apps/ops_dashboard/`
- Proposal artifacts: `runtime/artifacts/trade_proposals/`
- Paper trading database: `runtime/db/paper_trading/paper_trading.db`
- Account ledger: `runtime/db/accounts/account_ledger.db`
- Operational logs: `runtime/logs/`

## Safety Rules

1. `ALLOW_LIVE_TRADING=false` is the default posture.
2. Paper broker submission requires explicit `ALLOW_PAPER_ORDER_SUBMIT=true`.
3. Do not store broker credentials, account ledgers, or runtime logs here.
4. Keep example payloads small, synthetic, and public-safe.
5. Promote reusable execution logic into `src/oqp`, not `departments/trading`.
