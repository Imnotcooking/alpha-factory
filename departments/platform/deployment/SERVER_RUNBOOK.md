# Alpha Factory Server Runbook

This runbook makes the Ubuntu deployment reproducible. It captures the server
shape that currently works:

- live IBKR Gateway for read-only account monitoring
- paper IBKR Gateway for paper monitoring and later paper execution tests
- Streamlit dashboards bound to localhost
- scheduled live and paper snapshot jobs
- Discord health alerts

The tracked files are templates. Filled secrets stay on the server and out of
Git.

## File Map

```text
departments/platform/deployment/
  docker-compose.ibkr.yml       IBKR live/paper gateway containers
  ibkr_gateway_docker_run.sh    Docker CLI fallback when Compose is unavailable
  create_server_env_from_existing_state.py
                                One-time migration helper for existing servers
  server.env.example            Server-only env template, no real secrets
  systemd/                      Dashboard services and snapshot timers
  SERVER_RUNBOOK.md             This runbook
```

Existing job docs:

- `departments/platform/schedulers/portfolio_nav_update.md`
- `departments/platform/schedulers/paper_trading_monitor.md`
- `departments/platform/deployment/ibkr_ubuntu_readiness.md`

## Server Assumptions

Default paths and users match the current server:

```text
Linux user: ubuntu
Repo path:  /home/ubuntu/oqp_new
Python env: /home/ubuntu/oqp_new/.venv
Env file:   /home/ubuntu/.oqp_server_env
```

If those change, update the systemd unit files before installing them.

## 1. Prepare The Repo

```bash
cd /home/ubuntu
git clone https://github.com/Imnotcooking/alpha-factory.git oqp_new
cd /home/ubuntu/oqp_new
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p logs runtime data/paper_trading
```

For an existing server:

```bash
cd /home/ubuntu/oqp_new
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Create The Server Env File

```bash
cp departments/platform/deployment/server.env.example /home/ubuntu/.oqp_server_env
chmod 600 /home/ubuntu/.oqp_server_env
nano /home/ubuntu/.oqp_server_env
```

For an already-running server, this helper can bootstrap the env file from the
current Docker containers and private webhook env files without printing secret
values:

```bash
python3 departments/platform/deployment/create_server_env_from_existing_state.py
```

If the helper cannot recover the VNC password, it writes
`IBKR_VNC_PASSWORD="REPLACE_ME_BEFORE_CONTAINER_RECREATE"`. This is acceptable
for dashboard and snapshot timer operation. The Docker CLI helper treats the
VNC password as optional; if it is blank, use a temporary `x11vnc` session over
SSH only when MFA is required.

Fill in:

- `IBKR_LIVE_USER`
- `IBKR_LIVE_PASSWORD`
- `IBKR_PAPER_USER`
- `IBKR_PAPER_PASSWORD`
- `IBKR_VNC_PASSWORD` if you want the container to own VNC authentication
- vendor API keys if dashboards/jobs need them
- Discord webhook URLs if alerts should post

Keep these defaults unless there is a clear reason to change them:

```bash
IBKR_HOST=127.0.0.1
IBKR_LIVE_PORT=4001
IBKR_PAPER_PORT=7497
IBKR_LIVE_MONITOR_ENABLED=true
ALLOW_LIVE_TRADING=false
ALLOW_PAPER_TRADING=false
```

## 3. Start IBKR Gateways

Preferred path when Docker Compose is available:

```bash
cd /home/ubuntu/oqp_new
set -a
source /home/ubuntu/.oqp_server_env
set +a

docker compose \
  --env-file /home/ubuntu/.oqp_server_env \
  -f departments/platform/deployment/docker-compose.ibkr.yml \
  up -d
```

Fallback path when the server has Docker but not Docker Compose:

```bash
cd /home/ubuntu/oqp_new
chmod +x departments/platform/deployment/ibkr_gateway_docker_run.sh
departments/platform/deployment/ibkr_gateway_docker_run.sh check
departments/platform/deployment/ibkr_gateway_docker_run.sh start
```

For a fresh rebuild where old containers should be replaced deliberately:

```bash
departments/platform/deployment/ibkr_gateway_docker_run.sh recreate
```

Expected bindings:

```text
ib-gateway-live   127.0.0.1:4001 -> container API
ib-gateway-live   127.0.0.1:5901 -> VNC
ib-gateway-paper  127.0.0.1:7497 -> container API
ib-gateway-paper  127.0.0.1:5902 -> VNC
```

Check:

```bash
departments/platform/deployment/ibkr_gateway_docker_run.sh status
sudo ss -lntp | egrep ':4001|:7497|:5901|:5902'
```

## 4. MFA And VNC Access

VNC should only be reached through an SSH tunnel from your laptop.

Live gateway:

```bash
ssh -i ~/.ssh/oqp_aws_ed25519 -N -L 5901:127.0.0.1:5901 ubuntu@SERVER_IP
```

Open:

```text
vnc://127.0.0.1:5901
```

Paper gateway:

```bash
ssh -i ~/.ssh/oqp_aws_ed25519 -N -L 5902:127.0.0.1:5902 ubuntu@SERVER_IP
```

Open:

```text
vnc://127.0.0.1:5902
```

Use the `IBKR_VNC_PASSWORD` value from `/home/ubuntu/.oqp_server_env` if the
container is configured with one. If not, start a temporary `x11vnc` session
inside the container and keep access behind SSH localhost forwarding only.

## 5. Verify IBKR Readiness

After IB Gateway is logged in and MFA is complete:

```bash
cd /home/ubuntu/oqp_new
source .venv/bin/activate
set -a
source /home/ubuntu/.oqp_server_env
set +a

PYTHONPATH=src:. python scripts/check_ibkr_server_readiness.py --profile live --adapter-check
PYTHONPATH=src:. python scripts/check_ibkr_server_readiness.py --profile paper --adapter-check
```

Expected result:

- live profile connects read-only to the live account
- paper profile connects read-only to the paper account
- `ALLOW_LIVE_TRADING=false`
- socket host is local

## 6. Rotate The Live IBKR API Username

Use a dedicated IBKR username for the Ubuntu live Gateway. Do not use that
username casually in Client Portal, phone apps, or desktop TWS sessions.

In IBKR Client Portal, create or enable the dedicated API user first. Then
update only the live credentials on the server:

```bash
cd /home/ubuntu/oqp_new
cp /home/ubuntu/.oqp_server_env "/home/ubuntu/.oqp_server_env.backup.$(date -u +%Y%m%dT%H%M%SZ)"
chmod 600 /home/ubuntu/.oqp_server_env
nano /home/ubuntu/.oqp_server_env
```

Change only:

```bash
IBKR_LIVE_USER=
IBKR_LIVE_PASSWORD=
```

Then recreate only the live Gateway:

```bash
departments/platform/deployment/ibkr_gateway_docker_run.sh recreate-live
```

Complete live MFA over tunneled VNC, then verify:

```bash
source .venv/bin/activate
set -a
source /home/ubuntu/.oqp_server_env
set +a

PYTHONPATH=src:. python scripts/check_ibkr_server_readiness.py --profile live --adapter-check
PYTHONPATH=src:. python scripts/check_ibkr_adapter_heartbeat.py --profile live
```

The paper Gateway does not need to be restarted for a live username rotation.

## 7. Install Dashboard Services

Copy systemd service templates:

```bash
cd /home/ubuntu/oqp_new
sudo cp departments/platform/deployment/systemd/oqp-*.service /etc/systemd/system/
sudo cp departments/platform/deployment/systemd/oqp-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

Enable dashboards:

```bash
sudo systemctl enable --now oqp-money-dashboard.service
sudo systemctl enable --now oqp-paper-dashboard.service
sudo systemctl enable --now oqp-ops-dashboard.service
```

Optional research dashboard:

```bash
sudo systemctl enable --now oqp-research-dashboard.service
```

Dashboard ports:

```text
money dashboard     127.0.0.1:8531
paper dashboard     127.0.0.1:8527
ops dashboard       127.0.0.1:8529
research dashboard  127.0.0.1:8524
```

Prefer SSH tunnels or a reverse proxy with authentication. Do not expose these
ports directly to the internet.

## 8. Install Snapshot Timers

Enable scheduled jobs:

```bash
sudo systemctl enable --now oqp-portfolio-snapshot.timer
sudo systemctl enable --now oqp-paper-snapshot.timer
sudo systemctl enable --now oqp-paper-strategy-runner.timer
sudo systemctl enable --now oqp-ibkr-heartbeat.timer
```

Timer schedule:

```text
oqp-portfolio-snapshot.timer  Mon-Fri 21:30 server time
oqp-paper-snapshot.timer      Mon-Fri 21:45 server time
oqp-paper-strategy-runner.timer
                               Mon-Fri every 15 minutes, 13:00-22:45 server time
oqp-ibkr-heartbeat.timer       Every 15 minutes after boot
```

Check the server timezone:

```bash
timedatectl
systemctl list-timers 'oqp-*'
```

Manual one-shot runs:

```bash
sudo systemctl start oqp-portfolio-snapshot.service
sudo systemctl start oqp-paper-snapshot.service
sudo systemctl start oqp-paper-strategy-runner.service
sudo systemctl start oqp-ibkr-heartbeat.service
```

Logs:

```bash
tail -100 /home/ubuntu/oqp_new/logs/portfolio_snapshot_job.log
tail -100 /home/ubuntu/oqp_new/logs/paper_strategy_runner.log
tail -100 /home/ubuntu/oqp_new/logs/paper_snapshot_job.log
tail -100 /home/ubuntu/oqp_new/logs/ibkr_adapter_heartbeat.log
journalctl -u oqp-portfolio-snapshot.service -n 100 --no-pager
journalctl -u oqp-paper-snapshot.service -n 100 --no-pager
journalctl -u oqp-ibkr-heartbeat.service -n 100 --no-pager
```

## 9. Health Checks

Run directly:

```bash
cd /home/ubuntu/oqp_new
source .venv/bin/activate
set -a
source /home/ubuntu/.oqp_server_env
set +a

PYTHONPATH=src:. python scripts/check_portfolio_snapshot_health.py --notify-always
PYTHONPATH=src:. python scripts/check_paper_trading_health.py --notify-always
PYTHONPATH=src:. python scripts/check_ibkr_adapter_heartbeat.py --notify-always
```

Expected status files:

```text
logs/portfolio_snapshot_health.json
logs/paper_trading_health.json
logs/ibkr_adapter_heartbeat_health.json
```

## 10. Firewall Posture

IBKR API and VNC ports should bind to `127.0.0.1` only. Even then, keep the
host firewall restrictive:

```bash
sudo ufw allow OpenSSH
sudo ufw deny 4001/tcp
sudo ufw deny 7497/tcp
sudo ufw deny 5901/tcp
sudo ufw deny 5902/tcp
sudo ufw enable
sudo ufw status verbose
```

If dashboards are exposed later, use a reverse proxy with authentication and
TLS instead of opening raw Streamlit ports.

## 11. Rebuild Checklist

Use this when replacing the server:

1. Provision Ubuntu and SSH access.
2. Install Docker, Docker Compose, Python, Git, and build tools.
3. Clone repo into `/home/ubuntu/oqp_new`.
4. Create `.venv` and install `requirements.txt`.
5. Create `/home/ubuntu/.oqp_server_env`.
6. Start IBKR containers with `docker compose`.
7. Complete IBKR MFA through tunneled VNC.
8. Run live and paper IBKR readiness checks.
9. Install and enable dashboard services.
10. Install and enable snapshot timers.
11. Run manual portfolio and paper snapshot jobs once.
12. Confirm Discord health alerts arrive.
13. Confirm raw IBKR/VNC/dashboard ports are not publicly reachable.

## Safety Notes

- Live IBKR remains read-only.
- `ALLOW_LIVE_TRADING=false` must remain the default.
- `ALLOW_PAPER_TRADING=false` keeps proposal review blocked until execution
  safety work is explicitly armed.
- `/home/ubuntu/.oqp_server_env` is the sensitive server file. Do not copy it
  into the repo.
- Generated ledgers, logs, runtime state, and raw broker data stay ignored by
  Git.
