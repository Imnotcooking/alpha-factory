#!/usr/bin/env bash
set -Eeuo pipefail

LABEL="com.oqp.ops-dashboard"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"

launchctl bootout "gui/$UID" "$PLIST_PATH" >/dev/null 2>&1 || true
rm -f "$PLIST_PATH"

echo "Removed ${LABEL}"
