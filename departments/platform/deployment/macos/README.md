# macOS Ops Dashboard Agent

The local Ops Dashboard can run as a macOS `launchd` user agent instead of a
manual `screen` session.

This keeps the bookmark stable at:

```text
http://127.0.0.1:8529
```

Install or refresh the agent:

```bash
./scripts/platform/install_macos_ops_dashboard_agent.sh
```

Remove the agent:

```bash
./scripts/platform/uninstall_macos_ops_dashboard_agent.sh
```

Logs:

```text
runtime/logs/ops_dashboard.launchd.stdout.log
runtime/logs/ops_dashboard.launchd.stderr.log
runtime/logs/ops_dashboard.log
```

The agent starts `apps/ops_dashboard/Homepage.py` and restarts it if the
Streamlit process exits unexpectedly.

Note: macOS may block `launchd` from reading repos under `~/Documents` because
Documents is protected by privacy controls. If the installer refuses to run,
either keep using the screen-based helper:

```bash
./scripts/platform/restart_ops_dashboard_screen.sh
```

or move the repo to an unprotected project directory such as:

```text
~/Developer/oxford_quant_pipeline
```
