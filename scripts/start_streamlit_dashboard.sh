#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
APP_PATH="${1:?app path is required}"
PORT="${2:?port is required}"
DASHBOARD_NAME="${3:-dashboard}"

cd "$REPO_ROOT"
mkdir -p runtime/logs

export HOME="${HOME:-/Users/oscarchen}"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

PYTHON_BIN="${OQP_PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] starting ${DASHBOARD_NAME} on 127.0.0.1:${PORT}"
echo "using python: ${PYTHON_BIN}"
exec "$PYTHON_BIN" -m streamlit run "$APP_PATH" \
  --server.port "$PORT" \
  --server.address 127.0.0.1 \
  --server.headless true
