# QMT Connector Bridge

Status: skeleton installed, real MiniQMT wiring pending broker registration.

## Objective

Run OQP automation on Ubuntu while keeping MiniQMT and `xtquant` on a Windows
host. The Mac remains the cockpit for dashboards, approval, and intervention.

```text
Research signals
  -> TradeProposal artifacts
  -> paper safety review
  -> approved order tickets
  -> Ubuntu QMT adapter
  -> Windows QMT connector
  -> MiniQMT / xtquant
  -> broker
```

Inbound account truth flows the other way:

```text
MiniQMT / xtquant
  -> Windows QMT connector
  -> Ubuntu snapshot job
  -> runtime/db/accounts/account_ledger.db
  -> Ops Dashboard
```

## Host Roles

Ubuntu server:

- research jobs, backtests, proposal generation
- safety review and order-ticket approval state
- account ledgers and Ops/Research dashboards
- IBKR Gateway workloads
- QMT connector client and scheduler jobs

Windows QMT host:

- MiniQMT logged in and healthy
- `xtquant` installed in the connector runtime
- local QMT connector service
- local QMT request/callback logs

Mac:

- browser cockpit for Streamlit dashboards
- SSH/RDP intervention
- development only, not production execution

## Current Repo Skeleton

- `src/oqp/brokers/qmt.py`: connector-backed `BrokerAdapter`
- `src/oqp/brokers/registry.py`: QMT profiles
- `src/oqp/qmt_connector/stub_server.py`: stdlib fake connector for pre-QMT development
- `scripts/trading/run_qmt_connector_stub.py`: local connector stub runner
- `scripts/trading/update_qmt_account_snapshot.py`: pulls QMT state into account ledger
- `src/oqp/accounts/converters.py`: generic broker snapshot to account snapshot
- `src/oqp/ops/status.py`: QMT connector status and safety checks
- `.env.example`: QMT runtime and safety variables

## Environment Variables

```bash
QMT_CONNECTOR_ENABLED=false
QMT_CONNECTOR_URL=http://127.0.0.1:58668
QMT_SUBMIT_CONNECTOR_URL=http://127.0.0.1:58669
QMT_API_TOKEN=
QMT_REQUEST_SIGNING_SECRET=
QMT_AUDIT_LOG_PATH=runtime/logs/qmt_connector_audit.jsonl
QMT_REQUIRE_PRIVATE_CONNECTOR=true
QMT_ACCOUNT_ID=
QMT_PAPER_ACCOUNT_ID=
QMT_LIVE_ACCOUNT_ID=
QMT_ACCOUNT_TYPE=STOCK
QMT_SESSION_ID=880001
QMT_TIMEOUT_SECONDS=5
QMT_LIVE_MONITOR_ENABLED=false
ALLOW_QMT_PAPER_ORDER_SUBMIT=false
ALLOW_QMT_LIVE_TRADING=false
```

QMT paper submit requires both:

```bash
ALLOW_PAPER_ORDER_SUBMIT=true
ALLOW_QMT_PAPER_ORDER_SUBMIT=true
```

QMT live submit requires both:

```bash
ALLOW_LIVE_TRADING=true
ALLOW_QMT_LIVE_TRADING=true
```

Both live flags must remain false until a separate live-trading runbook is
written and tested.

## Local Stub Bring-Up

Before broker registration, run the local connector stub from the repo:

```bash
PYTHONPATH=src:. python scripts/trading/run_qmt_connector_stub.py \
  --host 127.0.0.1 \
  --port 58668 \
  --mode readonly \
  --api-token dev-qmt-token \
  --account-id PAPER123
```

Point OQP at it:

```bash
QMT_CONNECTOR_ENABLED=true
QMT_CONNECTOR_URL=http://127.0.0.1:58668
QMT_API_TOKEN=dev-qmt-token
QMT_PAPER_ACCOUNT_ID=PAPER123
```

Then test the read-only ledger path:

```bash
PYTHONPATH=src:. python scripts/trading/update_qmt_account_snapshot.py --profile qmt_paper_readonly
```

For a controlled paper-submit demo only, restart the stub with:

```bash
PYTHONPATH=src:. python scripts/trading/run_qmt_connector_stub.py \
  --host 127.0.0.1 \
  --port 58669 \
  --mode paper_submit \
  --api-token dev-qmt-token \
  --signing-secret dev-qmt-signing-secret \
  --account-id PAPER123 \
  --account-type STOCK \
  --allowed-symbol 600000.SH \
  --allowed-asset-class equity \
  --allowed-account-type STOCK \
  --max-quantity 1000 \
  --max-notional 20000 \
  --min-limit-price 1 \
  --max-limit-price 20 \
  --audit-log-path runtime/logs/windows_qmt_connector_audit.jsonl
```

Then point OQP submit traffic at the isolated submit port:

```bash
QMT_SUBMIT_CONNECTOR_URL=http://127.0.0.1:58669
QMT_REQUEST_SIGNING_SECRET=dev-qmt-signing-secret
QMT_AUDIT_LOG_PATH=runtime/logs/qmt_connector_audit.jsonl
```

The OQP side still requires `ALLOW_PAPER_ORDER_SUBMIT=true`,
`ALLOW_QMT_PAPER_ORDER_SUBMIT=true`, `QMT_API_TOKEN`, a non-empty
`QMT_REQUEST_SIGNING_SECRET`, and a submit URL that differs from
`QMT_CONNECTOR_URL` before it will post to the connector.

## Security Controls

- `paper_submit` connector mode refuses to start without `--api-token`,
  `--signing-secret`, and an explicit connector risk policy.
- The connector refuses public bind addresses by default. Use localhost,
  private LAN IPs, WireGuard IPs, or Tailscale `100.64.0.0/10` addresses.
- Submit requests must pass symbol, asset-class, account-type, quantity,
  notional, and limit-price-band checks on the connector side.
- `client_order_id` is required. Replays with the same `client_order_id` and
  identical order terms return the original order instead of creating another
  order; conflicting terms return an error.
- HMAC request signing uses `X-OQP-Timestamp`, `X-OQP-Nonce`,
  `X-OQP-Signature-Version`, and `X-OQP-Signature`. Nonces cannot be reused.
- Every connector `/submit_order` and `/cancel_order` attempt is appended to
  the Windows-side JSONL audit log when `--audit-log-path` is set.
- Every OQP `/submit_order` and `/cancel_order` call is mirrored to
  `QMT_AUDIT_LOG_PATH`.
- Read-only and submit traffic should run on separate connector URLs:
  `QMT_CONNECTOR_URL` for read-only state and `QMT_SUBMIT_CONNECTOR_URL` for
  paper/live order actions.

## Connector API Contract

The Windows connector should expose JSON endpoints over a private network only
(Tailscale, WireGuard, or localhost tunnel).

### `GET /health`

Response:

```json
{
  "status": "ok",
  "connected": true,
  "mini_qmt_connected": true,
  "mode": "readonly",
  "account_id": "ACCOUNT_ID",
  "session_id": 880001,
  "message": "MiniQMT connected"
}
```

### `GET /account`

Query:

```text
account_id=...&account_type=STOCK
```

Response:

```json
{
  "account": {
    "account_id": "ACCOUNT_ID",
    "account_type": "STOCK",
    "currency": "CNY",
    "cash": 1000000.0,
    "frozen_cash": 0.0,
    "market_value": 250000.0,
    "total_asset": 1250000.0
  }
}
```

### `GET /positions`

Response:

```json
{
  "positions": [
    {
      "symbol": "600000.SH",
      "asset_class": "equity",
      "quantity": 1000,
      "avg_price": 10.2,
      "market_price": 10.5,
      "market_value": 10500.0,
      "unrealized_pnl": 300.0,
      "currency": "CNY",
      "multiplier": 1
    }
  ]
}
```

Futures shorts should include `direction: "short"` or QMT's short direction
constant so OQP can store signed quantities.

### `GET /orders`

Response:

```json
{
  "orders": [
    {
      "order_id": "123",
      "order_sysid": "ABC",
      "symbol": "600000.SH",
      "asset_class": "equity",
      "side": "buy",
      "quantity": 1000,
      "order_type": "limit",
      "price": 10.5,
      "order_status": 50,
      "status_msg": "reported"
    }
  ]
}
```

### `GET /trades`

Response:

```json
{
  "trades": [
    {
      "trade_id": "fill-123",
      "order_id": "123",
      "symbol": "600000.SH",
      "asset_class": "equity",
      "side": "buy",
      "traded_volume": 1000,
      "traded_price": 10.5,
      "currency": "CNY"
    }
  ]
}
```

### `POST /submit_order`

Initially keep this disabled in the Windows connector until the OQP side
passes paper safety review and explicit approval.

Request:

```json
{
  "account_id": "ACCOUNT_ID",
  "account_type": "STOCK",
  "symbol": "600000.SH",
  "asset_class": "equity",
  "side": "buy",
  "quantity": 1000,
  "order_type": "limit",
  "limit_price": 10.5,
  "price_type": "FIX_PRICE",
  "strategy_id": "demo_strategy",
  "client_order_id": "paper-dryrun-demo-1"
}
```

Response:

```json
{
  "order": {
    "order_id": "123",
    "order_sysid": "ABC",
    "order_status": 50,
    "message": "submitted"
  }
}
```

### `POST /cancel_order`

Request:

```json
{
  "account_id": "ACCOUNT_ID",
  "account_type": "STOCK",
  "broker_order_id": "123"
}
```

Response:

```json
{
  "cancelled": true,
  "message": "cancel request accepted"
}
```

## First Real-World Bring-Up

1. Register QMT with the broker and install MiniQMT on the Windows host.
2. Install a Python version supported by `xtquant`.
3. Start MiniQMT and verify manual login.
4. Build the Windows connector against this API contract.
5. Set `QMT_CONNECTOR_URL` on Ubuntu to the private Windows read-only address.
6. Set `QMT_CONNECTOR_ENABLED=true`.
7. Run:

```bash
PYTHONPATH=src:. python scripts/trading/update_qmt_account_snapshot.py --profile qmt_paper_readonly
```

8. Open Ops Dashboard and confirm QMT heartbeat/account rows.
9. Only after read-only snapshots reconcile, start a separate submit connector
   on a separate private port with token auth, HMAC signing, audit logging, and
   one allowlisted symbol.

## Safety Defaults

- No QMT submit path is armed by default.
- Live QMT submit has a separate QMT-specific flag.
- Market orders are blocked by the OQP QMT adapter skeleton.
- The Windows connector refuses public binds by default.
- Submit mode requires token auth, HMAC signing, risk limits, and idempotent
  `client_order_id`.
- All submit requests must be logged on both Ubuntu and Windows.
