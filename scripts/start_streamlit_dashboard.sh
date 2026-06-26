#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
APP_PATH="${1:?app path is required}"
PORT="${2:?port is required}"
DASHBOARD_NAME="${3:-dashboard}"

cd "$REPO_ROOT"
mkdir -p logs

export HOME="${HOME:-/Users/oscarchen}"
export PATH="/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] starting ${DASHBOARD_NAME} on 127.0.0.1:${PORT}"
exec /opt/anaconda3/bin/python -m streamlit run "$APP_PATH" \
  --server.port "$PORT" \
  --server.address 127.0.0.1 \
  --server.headless true
