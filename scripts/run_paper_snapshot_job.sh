#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

if [[ -f "$HOME/.oqp_portfolio_health_env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.oqp_portfolio_health_env"
fi

if [[ -f "$HOME/.oqp_paper_trading_env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.oqp_paper_trading_env"
fi

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONPATH="src:.:${PYTHONPATH:-}"

echo "[$(date -Is)] paper snapshot job started"
python scripts/check_ibkr_server_readiness.py --profile paper --adapter-check
python scripts/update_paper_trading_snapshot.py
python scripts/check_paper_trading_health.py \
  --max-age-hours "${OQP_PAPER_HEALTH_MAX_AGE_HOURS:-36}" \
  --status-path logs/paper_trading_health.json \
  --notify-always
echo "[$(date -Is)] paper snapshot job completed"
