# Portfolio NAV Update Scheduler

## Purpose

`scripts/run_portfolio_snapshot_job.sh` is the non-UI job for daily
real-portfolio tracking. It checks the read-only IBKR connection, refreshes the
latest `live_positions` snapshot, fetches market history, values the portfolio
with the shared valuation engine, and writes one row into `historical_nav`.

This is the job to run on the Ubuntu server while IB Gateway is already running
and logged in.

## Command

From the repository root:

```bash
./scripts/run_portfolio_snapshot_job.sh
```

Useful options:

```bash
./scripts/run_portfolio_snapshot_job.sh --dry-run
```

Use `--dry-run` to run the readiness check and broker ingestion, then calculate NAV
without writing `historical_nav`.

## Data Flow

1. `scripts/update_live_portfolio_snapshot.py` writes latest `live_positions`
   into `runtime/db/portfolio/portfolio_ledger.db`.
2. It writes IBKR cash/NAV/margin metrics into
   `runtime/state/portfolio/ibkr_metrics.json`.
3. `scripts/update_portfolio_nav.py` loads non-secret manual inputs from
   `runtime/state/portfolio/manual_inputs.json`, falling back to the old Middle
   Office defaults only during migration.
4. It fetches Yahoo close history for positions, FX pairs, and macro tickers.
5. It calls `oqp.portfolio.value_portfolio_snapshot(...)`.
6. It upserts the daily `historical_nav` row.

If IBKR has cash but no open positions, the broker ETL still writes
`ibkr_metrics.json`. The NAV updater can then write a cash/manual-only NAV row
from those metrics, even when `live_positions` has no IBKR rows.

## Suggested Ubuntu Schedule

Preferred reproducible deployment uses the tracked systemd units:

```bash
sudo cp departments/platform/deployment/systemd/oqp-portfolio-snapshot.service /etc/systemd/system/
sudo cp departments/platform/deployment/systemd/oqp-portfolio-snapshot.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now oqp-portfolio-snapshot.timer
```

Cron remains a simple fallback:

Run the combined broker-ingestion and NAV update job:

```cron
30 21 * * 1-5 cd /home/ubuntu/oqp_new && ./scripts/run_portfolio_snapshot_job.sh >> runtime/logs/portfolio_snapshot_job.log 2>&1
```

For live IBKR data, the server still needs TWS or IB Gateway running and logged
in separately. This scheduler only reads the SQLite snapshot and does not place
orders.

The ordering matters:

1. `scripts/check_ibkr_server_readiness.py` confirms the read-only live profile
   can connect.
2. `scripts/update_live_portfolio_snapshot.py` talks to brokers and CSV exports.
3. It writes `live_positions` and runtime `ibkr_metrics.json` for account
   cash/NAV/margin metrics.
4. `scripts/update_portfolio_nav.py` values the latest snapshot and writes
   `historical_nav`.
5. `scripts/check_portfolio_snapshot_health.py` verifies that the SQLite ledger
   has a fresh positive NAV row and writes `runtime/logs/portfolio_snapshot_health.json`.
6. The Ops/Money dashboards read SQLite and JSON outputs; they should not be the
   only place where NAV history gets updated on the server.

## Health Check

Run the health check directly with:

```bash
PYTHONPATH=src:. python scripts/check_portfolio_snapshot_health.py
```

The combined job runs this automatically after real NAV writes. The unified live
account ledger is authoritative once it has a fresh `unified_live` row. Legacy
portfolio-ledger freshness is still printed for migration visibility, but it no
longer fails the job when the unified live account snapshot is current.

To post failure payloads to Discord, create a private server-only env file:

```bash
cat > ~/.oqp_portfolio_health_env <<'EOF'
export OQP_LIVE_DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'
EOF
chmod 600 ~/.oqp_portfolio_health_env
```

The combined runner sources this file automatically. If
`OQP_LIVE_DISCORD_WEBHOOK_URL` is blank, the health checker falls back to
`OQP_PORTFOLIO_DISCORD_WEBHOOK_URL`, then `OQP_DISCORD_WEBHOOK_URL`, then
`OQP_HEALTH_WEBHOOK_URL`. Keep the env file out of git.

## IBKR Server Notes

The broker ETL uses the shared read-only IBKR adapter. On the server, configure:

```bash
IBKR_LIVE_MONITOR_ENABLED=true
IBKR_HOST=127.0.0.1
IBKR_LIVE_PORT=4001
IBKR_LIVE_CLIENT_ID=201
```

Keep `ALLOW_LIVE_TRADING=false`; this path only reads account state.
