#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p runtime/logs

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required to run local dashboards detached from the terminal." >&2
  exit 1
fi

start_dashboard() {
  local session="$1"
  local app_path="$2"
  local port="$3"
  local name="$4"
  local log_path="$5"

  if { screen -list || true; } | grep -q "[.]${session}[[:space:]]"; then
    echo "${session} already running"
    return
  fi

  screen -dmS "$session" /bin/bash -lc \
    "cd '$REPO_ROOT' && ./scripts/start_streamlit_dashboard.sh '$app_path' '$port' '$name' >> '$log_path' 2>&1"
  echo "started ${session} on http://127.0.0.1:${port}"
}

start_dashboard "oqp-research-dashboard" "apps/research_dashboard/Homepage.py" "8524" "research dashboard" "runtime/logs/research_dashboard.log"
start_dashboard "oqp-ops-dashboard" "apps/ops_dashboard/Homepage.py" "8529" "ops dashboard" "runtime/logs/ops_dashboard.log"
