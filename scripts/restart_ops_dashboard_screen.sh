#!/usr/bin/env bash
set -Eeuo pipefail

SESSION="oqp-ops-dashboard"
PORT="8529"
APP_PATH="apps/ops_dashboard/Homepage.py"
REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

screen -S "$SESSION" -X quit >/dev/null 2>&1 || true

if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids:-}" ]]; then
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
      if [[ "$command" == *"streamlit run apps/ops_dashboard/"* ]]; then
        kill "$pid" >/dev/null 2>&1 || true
      else
        echo "Port ${PORT} is used by a non-OQP process:" >&2
        echo "  pid=${pid} ${command}" >&2
        exit 1
      fi
    done <<< "$pids"
  fi
fi

mkdir -p "$REPO_ROOT/runtime/logs"
LAUNCH_CMD="cd '$REPO_ROOT' && ./scripts/start_streamlit_dashboard.sh '$APP_PATH' '$PORT' 'ops dashboard' >> '$REPO_ROOT/runtime/logs/ops_dashboard.log' 2>&1"

if command -v screen >/dev/null 2>&1; then
  screen -dmS "$SESSION" /bin/bash -lc "$LAUNCH_CMD" || true
  sleep 1
  if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Ops dashboard started at http://127.0.0.1:${PORT}"
    exit 0
  fi
  echo "screen launch did not stay up; falling back to nohup." >&2
fi

nohup /bin/bash -lc "$LAUNCH_CMD" >/dev/null 2>&1 &

echo "Ops dashboard started at http://127.0.0.1:${PORT}"
