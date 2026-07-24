# IBKR Ubuntu Readiness

For full server rebuild instructions, see
`departments/platform/deployment/SERVER_RUNBOOK.md`.

## What To Check

Use the redacted readiness script before running the daily broker ETL:

```bash
PYTHONPATH=src:. python scripts/ops/check_ibkr_server_readiness.py --profile live
```

Once IB Gateway or TWS is logged in on the server, run the deeper adapter check:

```bash
PYTHONPATH=src:. python scripts/ops/check_ibkr_server_readiness.py --profile live --adapter-check
```

The script checks:

- live read-only gate is enabled when using the live profile
- `ALLOW_LIVE_TRADING=false`
- IBKR host/port/client id values
- required Python packages
- local ledger path
- raw TCP socket reachability
- optional read-only adapter account snapshot with redacted account id

It does not print API/vendor secrets or broker login credentials.

## Expected Live Server Values

For IB Gateway live:

```bash
IBKR_LIVE_MONITOR_ENABLED=true
IBKR_HOST=127.0.0.1
IBKR_LIVE_PORT=4001
IBKR_LIVE_CLIENT_ID=201
ALLOW_LIVE_TRADING=false
```

For TWS live, use:

```bash
IBKR_LIVE_PORT=7496
```

The current reproducible Ubuntu template exposes paper Gateway on local host
port `7497`, mapped to the container's internal paper API port. Keep
`IBKR_PAPER_PORT=7497` unless the Compose file is changed too.

## Docker Compose Template

The tracked template is:

```bash
departments/platform/deployment/docker-compose.ibkr.yml
```

If the server has Docker but not Docker Compose, use the tracked fallback:

```bash
departments/platform/deployment/ibkr_gateway_docker_run.sh check
departments/platform/deployment/ibkr_gateway_docker_run.sh start
departments/platform/deployment/ibkr_gateway_docker_run.sh status
```

It binds live, paper, and VNC ports to `127.0.0.1` only:

```text
live API      127.0.0.1:4001
live VNC      127.0.0.1:5901
paper API     127.0.0.1:7497
paper VNC     127.0.0.1:5902
```

Filled broker login credentials belong in `/home/ubuntu/.oqp_server_env`, not
in this repository.

## Daily Order

Run these after IB Gateway/TWS is already logged in:

```bash
PYTHONPATH=src:. python scripts/ops/check_ibkr_server_readiness.py --profile live --adapter-check
PYTHONPATH=src:. python scripts/ops/update_live_portfolio_snapshot.py
PYTHONPATH=src:. python scripts/ops/update_portfolio_nav.py
```

The first command checks the pipe. The ingestion script writes positions and
IBKR account metrics. The NAV updater writes the historical equity curve.
