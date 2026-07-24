# Paper Trading Monitor

## Purpose

`scripts/trading/run_paper_snapshot_job.sh` is the read-only paper account monitoring
job. It checks the IBKR paper profile, writes a separate paper trading SQLite
ledger, runs a freshness check, and posts a Discord daily report.

It does not place orders.

## Data

The paper ledger defaults to:

```text
runtime/db/paper_trading/paper_trading.db
```

Tables:

- `paper_account_snapshots`
- `paper_positions`
- `paper_nav`
- `paper_orders`
- `paper_fills`
- `paper_execution_reviews`

The order/fill tables are created now so the later paper execution phase can
write into the same ledger without changing dashboards or alerts.
`paper_execution_reviews` records proposal safety reviews before any broker
submission code exists.

## Command

From the repository root:

```bash
./scripts/trading/run_paper_snapshot_job.sh
```

Individual steps:

```bash
PYTHONPATH=src:. python scripts/ops/check_ibkr_server_readiness.py --profile paper --adapter-check
PYTHONPATH=src:. python scripts/ops/update_paper_trading_snapshot.py
PYTHONPATH=src:. python scripts/ops/check_paper_trading_health.py --notify-always
```

## Proposal Review

Paper proposal review is separate from the daily account monitor:

```bash
PYTHONPATH=src:. python scripts/trading/review_paper_trade_proposal.py runtime/artifacts/trade_proposals --notify
```

This evaluates proposal artifacts against the paper safety policy and writes
`paper_execution_reviews`. It does not place orders.

## Paper Strategy Runner

`scripts/trading/run_paper_strategy_runner.py` is the automated proposal scanner for
strategies that are already approved as `paper_running` in
`paper_strategy_registry`.

Pipeline:

1. Load JSON proposal artifacts from `runtime/artifacts/trade_proposals`.
2. Skip proposals whose strategy is missing, paused, retired, or kill-switched.
3. Run the paper execution safety review for eligible proposals.
4. Write `paper_execution_reviews` and unified `paper_review` account events.
5. Create `dry_run` paper order tickets when safety passes.

It does not place broker orders.

Manual run:

```bash
./scripts/trading/run_paper_strategy_runner_job.sh
```

Direct Python run:

```bash
PYTHONPATH=src:. python scripts/trading/run_paper_strategy_runner.py runtime/artifacts/trade_proposals --notify-on-action
```

`--notify-on-action` stays quiet when no proposals are reviewed, which keeps
the timer from spamming Discord.

## Discord

The paper health checker reads `OQP_PAPER_DISCORD_WEBHOOK_URL` first, then falls
back to `OQP_DISCORD_WEBHOOK_URL`. A separate webhook is optional but useful if
paper reports should go to a different Discord channel from real-portfolio
health alerts.

Recommended server-only env file:

```bash
cat > ~/.oqp_paper_trading_env <<'EOF'
export OQP_PAPER_DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'
EOF
chmod 600 ~/.oqp_paper_trading_env
```

The runner sources both `~/.oqp_portfolio_health_env` and
`~/.oqp_paper_trading_env`.

## Suggested Cron

Preferred reproducible deployment uses the tracked systemd units:

```bash
sudo cp departments/platform/deployment/systemd/oqp-paper-snapshot.service /etc/systemd/system/
sudo cp departments/platform/deployment/systemd/oqp-paper-snapshot.timer /etc/systemd/system/
sudo cp departments/platform/deployment/systemd/oqp-paper-strategy-runner.service /etc/systemd/system/
sudo cp departments/platform/deployment/systemd/oqp-paper-strategy-runner.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now oqp-paper-snapshot.timer
sudo systemctl enable --now oqp-paper-strategy-runner.timer
```

Cron remains a simple fallback.

Run after the real portfolio snapshot, or at another time that fits the paper
trial cadence:

```cron
45 21 * * 1-5 cd /home/ubuntu/oqp_new && ./scripts/trading/run_paper_snapshot_job.sh >> runtime/logs/paper_snapshot_job.log 2>&1
*/15 13-22 * * 1-5 cd /home/ubuntu/oqp_new && ./scripts/trading/run_paper_strategy_runner_job.sh >> runtime/logs/paper_strategy_runner.log 2>&1
```

## Safety Boundary

This lane remains non-executing. `ALLOW_PAPER_TRADING=false` blocks proposal
reviews by default, the paper strategy runner writes dry-run tickets only, and
there is still no broker order submission path in the paper runner.
