#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_USER="${OQP_SERVER_USER:-ubuntu}"
REMOTE_HOST="${OQP_SERVER_HOST:-18.190.212.21}"
REMOTE_REPO="${OQP_SERVER_REPO:-/home/ubuntu/oqp_new}"
SSH_KEY="${OQP_SERVER_SSH_KEY:-$HOME/.ssh/oqp_aws_ed25519}"

cd "$REPO_ROOT"
mkdir -p runtime/logs runtime/db/accounts runtime/db/portfolio runtime/state/portfolio runtime/state/server_sync

SSH_CMD=(
  ssh
  -i "$SSH_KEY"
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=8
)

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
RUNTIME_SOURCES=(
  "${REMOTE}:${REMOTE_REPO}/./runtime/db/accounts/account_ledger.db"
  "${REMOTE}:${REMOTE_REPO}/./runtime/db/portfolio/portfolio_ledger.db"
  "${REMOTE}:${REMOTE_REPO}/./runtime/state/portfolio/banked_profits.json"
  "${REMOTE}:${REMOTE_REPO}/./runtime/state/portfolio/ibkr_metrics.json"
)

rsync -azR --partial -e "${SSH_CMD[*]}" "${RUNTIME_SOURCES[@]}" "$REPO_ROOT/"

HEALTH_FILES=(
  "portfolio_snapshot_health.json"
  "paper_trading_health.json"
  "ibkr_adapter_heartbeat_health.json"
)

for file in "${HEALTH_FILES[@]}"; do
  if "${SSH_CMD[@]}" "$REMOTE" "test -f '$REMOTE_REPO/runtime/logs/$file'"; then
    rsync -az --partial -e "${SSH_CMD[*]}" \
      "${REMOTE}:${REMOTE_REPO}/runtime/logs/${file}" \
      "$REPO_ROOT/runtime/logs/${file}"
  elif "${SSH_CMD[@]}" "$REMOTE" "test -f '$REMOTE_REPO/logs/$file'"; then
    rsync -az --partial -e "${SSH_CMD[*]}" \
      "${REMOTE}:${REMOTE_REPO}/logs/${file}" \
      "$REPO_ROOT/runtime/logs/${file}"
  else
    echo "Warning: ${file} not found under runtime/logs or logs on ${REMOTE}" >&2
  fi
done

READINESS_DIR="$(mktemp -d)"
trap 'rm -rf "$READINESS_DIR"' EXIT

for profile in live paper; do
  set +e
  output="$("${SSH_CMD[@]}" "$REMOTE" \
    "cd '$REMOTE_REPO' && source .venv/bin/activate && PYTHONPATH=src:. python scripts/check_ibkr_server_readiness.py --profile '$profile' --adapter-check --json" \
    2>&1)"
  status=$?
  set -e
  printf '%s\n' "$status" > "$READINESS_DIR/${profile}.status"
  printf '%s\n' "$output" > "$READINESS_DIR/${profile}.out"
done

python - "$READINESS_DIR" "$REPO_ROOT/runtime/logs/server_ibkr_readiness_health.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_checks(text: str) -> list[dict[str, str]]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return [{"name": "readiness output", "status": "fail", "detail": text.strip() or "missing output"}]
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return [{"name": "readiness output", "status": "fail", "detail": f"Invalid JSON: {exc}"}]
    return parsed if isinstance(parsed, list) else [{"name": "readiness output", "status": "fail", "detail": "Unexpected payload shape"}]


tmp = Path(sys.argv[1])
out_path = Path(sys.argv[2])
profiles = {}
overall = "pass"
for profile in ("live", "paper"):
    rc = int((tmp / f"{profile}.status").read_text(encoding="utf-8").strip() or "1")
    text = (tmp / f"{profile}.out").read_text(encoding="utf-8")
    checks = parse_checks(text)
    failed = rc != 0 or any(str(check.get("status")) == "fail" for check in checks)
    warned = any(str(check.get("status")) == "warn" for check in checks)
    status = "fail" if failed else "warn" if warned else "pass"
    if status == "fail":
        overall = "fail"
    elif status == "warn" and overall != "fail":
        overall = "warn"
    profiles[profile] = {"status": status, "return_code": rc, "checks": checks}

payload = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "status": overall,
    "profiles": profiles,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

SYNCED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
cat > runtime/state/server_sync/status.json <<JSON
{
  "status": "pass",
  "synced_at": "$SYNCED_AT",
  "remote": "$REMOTE",
  "remote_repo": "$REMOTE_REPO"
}
JSON

echo "Synced server runtime evidence at ${SYNCED_AT}"
