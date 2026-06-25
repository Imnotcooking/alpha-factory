#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

if [[ -f "$HOME/.oqp_portfolio_health_env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.oqp_portfolio_health_env"
fi

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONPATH="src:.:${PYTHONPATH:-}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

echo "[$(date -Is)] portfolio snapshot job started"
python scripts/check_ibkr_server_readiness.py --profile live --adapter-check
python scripts/update_live_portfolio_snapshot.py

if [[ "$DRY_RUN" == "1" ]]; then
  python scripts/update_portfolio_nav.py --dry-run
else
  python scripts/update_portfolio_nav.py
  python scripts/check_portfolio_snapshot_health.py \
    --max-age-hours "${OQP_PORTFOLIO_HEALTH_MAX_AGE_HOURS:-36}" \
    --status-path logs/portfolio_snapshot_health.json
fi

echo "[$(date -Is)] portfolio snapshot job completed"
