#!/usr/bin/env bash
set -Eeuo pipefail

COMMAND="${1:-status}"
SERVER_ENV="${OQP_SERVER_ENV:-$HOME/.oqp_server_env}"

if [[ -f "$SERVER_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$SERVER_ENV"
fi

IMAGE="${IB_GATEWAY_IMAGE:-ghcr.io/gnzsnz/ib-gateway:latest}"
EXISTING_SESSION_ACTION="${IBKR_EXISTING_SESSION_ACTION:-primary}"
PAPER_READ_ONLY_API="${IBKR_PAPER_READ_ONLY_API:-yes}"
DOCKER=(docker)

required_for_start=(
  IBKR_LIVE_USER
  IBKR_LIVE_PASSWORD
  IBKR_PAPER_USER
  IBKR_PAPER_PASSWORD
  IBKR_VNC_PASSWORD
)

usage() {
  cat <<'EOF'
Usage:
  ibkr_gateway_docker_run.sh status
  ibkr_gateway_docker_run.sh check
  ibkr_gateway_docker_run.sh start
  ibkr_gateway_docker_run.sh recreate
  ibkr_gateway_docker_run.sh stop

Commands:
  status    Show current IBKR containers.
  check     Validate required server env values without starting containers.
  start     Create missing containers, or start existing stopped containers.
  recreate  Remove and recreate both containers from the current env file.
  stop      Stop both containers without deleting them.

This script is the Docker CLI fallback for servers without `docker compose`.
It expects filled secrets in ~/.oqp_server_env or OQP_SERVER_ENV.
EOF
}

require_env() {
  local missing=()
  local name
  for name in "${required_for_start[@]}"; do
    if [[ -z "${!name:-}" || "${!name:-}" == REPLACE_ME* ]]; then
      missing+=("$name")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    printf 'Missing required env values in %s:\n' "$SERVER_ENV" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi
}

container_exists() {
  "${DOCKER[@]}" inspect "$1" >/dev/null 2>&1
}

select_docker() {
  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
    return 0
  fi
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
    return 0
  fi
  printf 'Cannot access Docker. Run as a user in the docker group or allow passwordless sudo docker.\n' >&2
  return 1
}

run_live() {
  "${DOCKER[@]}" run -d \
    --name ib-gateway-live \
    --restart unless-stopped \
    -p "127.0.0.1:${IBKR_LIVE_API_PORT:-4001}:${IBKR_LIVE_CONTAINER_API_PORT:-4001}" \
    -p "127.0.0.1:${IBKR_LIVE_VNC_PORT:-5901}:5900" \
    -e TWS_USERID="$IBKR_LIVE_USER" \
    -e TWS_PASSWORD="$IBKR_LIVE_PASSWORD" \
    -e TRADING_MODE=live \
    -e GATEWAY_OR_TWS=gateway \
    -e READ_ONLY_API=yes \
    -e TWS_ACCEPT_INCOMING=accept \
    -e EXISTING_SESSION_DETECTED_ACTION="$EXISTING_SESSION_ACTION" \
    -e VNC_SERVER_PASSWORD="$IBKR_VNC_PASSWORD" \
    "$IMAGE"
}

run_paper() {
  "${DOCKER[@]}" run -d \
    --name ib-gateway-paper \
    --restart unless-stopped \
    -p "127.0.0.1:${IBKR_PAPER_API_PORT:-7497}:${IBKR_PAPER_CONTAINER_API_PORT:-4004}" \
    -p "127.0.0.1:${IBKR_PAPER_VNC_PORT:-5902}:5900" \
    -e TWS_USERID="$IBKR_PAPER_USER" \
    -e TWS_PASSWORD="$IBKR_PAPER_PASSWORD" \
    -e TRADING_MODE=paper \
    -e GATEWAY_OR_TWS=gateway \
    -e READ_ONLY_API="$PAPER_READ_ONLY_API" \
    -e TWS_ACCEPT_INCOMING=accept \
    -e EXISTING_SESSION_DETECTED_ACTION="$EXISTING_SESSION_ACTION" \
    -e VNC_SERVER_PASSWORD="$IBKR_VNC_PASSWORD" \
    "$IMAGE"
}

start_one() {
  local name="$1"
  local runner="$2"
  if container_exists "$name"; then
    "${DOCKER[@]}" start "$name" >/dev/null
    printf '%s exists; started or already running.\n' "$name"
  else
    "$runner" >/dev/null
    printf '%s created.\n' "$name"
  fi
}

case "$COMMAND" in
  status)
    select_docker
    "${DOCKER[@]}" ps --filter 'name=ib-gateway' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  check)
    select_docker
    require_env
    printf 'IBKR Docker env looks complete. image=%s live_port=%s paper_port=%s\n' \
      "$IMAGE" "${IBKR_LIVE_API_PORT:-4001}" "${IBKR_PAPER_API_PORT:-7497}"
    ;;
  start)
    select_docker
    require_env
    start_one ib-gateway-live run_live
    start_one ib-gateway-paper run_paper
    ;;
  recreate)
    select_docker
    require_env
    "${DOCKER[@]}" rm -f ib-gateway-live ib-gateway-paper >/dev/null 2>&1 || true
    run_live >/dev/null
    run_paper >/dev/null
    printf 'ib-gateway-live and ib-gateway-paper recreated.\n'
    ;;
  stop)
    select_docker
    "${DOCKER[@]}" stop ib-gateway-live ib-gateway-paper >/dev/null 2>&1 || true
    printf 'IBKR containers stopped if they existed.\n'
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
