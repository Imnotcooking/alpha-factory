#!/usr/bin/env bash
set -Eeuo pipefail

LABEL="com.oqp.ops-dashboard"
PORT="8529"
APP_PATH="apps/ops_dashboard/Homepage.py"
REPO_ROOT="${OQP_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/${LABEL}.plist"

if [[ "$REPO_ROOT" == "$HOME/Documents/"* && -z "${OQP_ALLOW_DOCUMENTS_LAUNCHD:-}" ]]; then
  cat >&2 <<EOF
Refusing to install ${LABEL} from a repo under ~/Documents.

macOS privacy controls often block launchd background jobs from reading files in
Documents, which makes the dashboard crash with "Operation not permitted".

Use the screen helper instead:
  ./scripts/restart_ops_dashboard_screen.sh

Or move the repo to an unprotected project folder such as:
  ~/Developer/oxford_quant_pipeline

Set OQP_ALLOW_DOCUMENTS_LAUNCHD=1 only if you have explicitly granted the
needed macOS privacy permissions and want to force installation.
EOF
  exit 2
fi

xml_escape() {
  printf '%s' "$1" \
    | sed \
      -e 's/&/\&amp;/g' \
      -e 's/</\&lt;/g' \
      -e 's/>/\&gt;/g' \
      -e 's/"/\&quot;/g' \
      -e "s/'/\&apos;/g"
}

shell_quote() {
  printf '%q' "$1"
}

stop_screen_session() {
  screen -S oqp-ops-dashboard -X quit >/dev/null 2>&1 || true
}

stop_existing_streamlit_listener() {
  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi

  local pids
  pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return
  fi

  local pid command
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$command" == *"streamlit run ${APP_PATH}"* ]] || [[ "$command" == *"streamlit run apps/ops_dashboard/app.py"* ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    else
      echo "Port ${PORT} is used by a non-OQP process:" >&2
      echo "  pid=${pid} ${command}" >&2
      exit 1
    fi
  done <<< "$pids"
}

mkdir -p "$PLIST_DIR" "$REPO_ROOT/runtime/logs"

stop_screen_session
launchctl bootout "gui/$UID" "$PLIST_PATH" >/dev/null 2>&1 || true
stop_existing_streamlit_listener

COMMAND="cd $(shell_quote "$REPO_ROOT") && exec ./scripts/start_streamlit_dashboard.sh $(shell_quote "$APP_PATH") ${PORT} 'ops dashboard'"
COMMAND_XML="$(xml_escape "$COMMAND")"
REPO_ROOT_XML="$(xml_escape "$REPO_ROOT")"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>${COMMAND_XML}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT_XML}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>StandardOutPath</key>
  <string>${REPO_ROOT_XML}/runtime/logs/ops_dashboard.launchd.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${REPO_ROOT_XML}/runtime/logs/ops_dashboard.launchd.stderr.log</string>
</dict>
</plist>
PLIST

chmod 644 "$PLIST_PATH"
launchctl bootstrap "gui/$UID" "$PLIST_PATH"
launchctl kickstart -k "gui/$UID/$LABEL"

echo "Installed ${LABEL}"
echo "Dashboard URL: http://127.0.0.1:${PORT}"
launchctl print "gui/$UID/$LABEL" | sed -n '1,80p'
