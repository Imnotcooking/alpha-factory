#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p runtime/logs

if [[ -f "$HOME/.oqp_server_env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.oqp_server_env"
fi

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

PROPOSAL_PATH="${OQP_PAPER_STRATEGY_PROPOSAL_PATH:-runtime/artifacts/trade_proposals}"
MAX_FILES="${OQP_PAPER_STRATEGY_MAX_FILES:-50}"
INCLUDE_REVIEWED="${OQP_PAPER_STRATEGY_INCLUDE_REVIEWED:-false}"

ARGS=(
  "$PROPOSAL_PATH"
  --max-files "$MAX_FILES"
  --notify-on-action
)

if [[ "${INCLUDE_REVIEWED,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  ARGS+=(--include-reviewed)
fi

echo "[$(date -Is)] paper strategy runner started"
python scripts/trading/run_paper_strategy_runner.py "${ARGS[@]}"
echo "[$(date -Is)] paper strategy runner completed"
