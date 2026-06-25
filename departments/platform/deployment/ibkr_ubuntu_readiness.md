# IBKR Ubuntu Readiness

## What To Check

Use the redacted readiness script before running the daily broker ETL:

```bash
PYTHONPATH=src:. python scripts/check_ibkr_server_readiness.py --profile live
```

Once IB Gateway or TWS is logged in on the server, run the deeper adapter check:

```bash
PYTHONPATH=src:. python scripts/check_ibkr_server_readiness.py --profile live --adapter-check
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

Paper defaults are usually `4002` for IB Gateway paper and `7497` for TWS paper.

## Daily Order

Run these after IB Gateway/TWS is already logged in:

```bash
PYTHONPATH=src:. python scripts/check_ibkr_server_readiness.py --profile live --adapter-check
PYTHONPATH=src:. python scripts/update_live_portfolio_snapshot.py
PYTHONPATH=src:. python scripts/update_portfolio_nav.py
```

The first command checks the pipe. The ingestion script writes positions and
IBKR account metrics. The NAV updater writes the historical equity curve.
