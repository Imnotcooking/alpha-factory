#!/usr/bin/env bash
set -Eeuo pipefail

for session in \
  oqp-research-dashboard \
  oqp-paper-dashboard \
  oqp-ops-dashboard \
  oqp-money-dashboard
do
  if screen -list | grep -q "[.]${session}[[:space:]]"; then
    screen -S "$session" -X quit
    echo "stopped ${session}"
  else
    echo "${session} not running"
  fi
done
