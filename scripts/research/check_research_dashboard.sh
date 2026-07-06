#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${OQP_PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/oqp_pycache}"

echo "research dashboard check"
echo "repo: $REPO_ROOT"
echo "python: $PYTHON_BIN"

"$PYTHON_BIN" -m pytest \
  tests/test_research_dashboard_preflight.py \
  tests/test_research_dashboard_real_data_smoke.py \
  tests/test_research_dashboard_pages_apptest.py \
  tests/test_research_reproducibility.py \
  tests/test_research_ml_governance.py \
  tests/test_research_latent.py \
  tests/test_risk_factor_breadth.py \
  tests/test_tick_pulse_features.py \
  tests/test_tick_pulse_asset_ranker.py \
  tests/test_tick_pulse_ml_migration.py \
  -q
