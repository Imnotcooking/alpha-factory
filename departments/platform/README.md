# Platform Department Map

Platform owns deployment, process supervision, schedulers, runtime health,
configuration delivery, and repository-operating standards. Platform does not
own research logic, account truth, or trading decisions.

## Start Here

For local use, prefer the installed CLI:

```bash
oqp doctor
oqp dashboard research
oqp dashboard ops
```

For a server deployment, begin with the [server runbook](deployment/SERVER_RUNBOOK.md).

## Directory Map

| Directory | Purpose |
| --- | --- |
| `deployment/` | Server, macOS, Docker, systemd, environment, and readiness documentation |
| `schedulers/` | Job cadence, ownership, inputs, outputs, and failure policy |

Executable helpers belong in `scripts/platform/`; reusable health and
configuration logic belongs in `src/oqp/ops/` and `src/oqp/config/`.

## Operating Rules

- Deployment files may reference environment-variable names, never secret
  values.
- Services must use explicit working directories, runtime profiles, and health
  checks.
- Dashboard availability is not evidence that broker connectivity or order
  submission is safe.
- Scheduled jobs must document idempotency, output paths, freshness limits, and
  recovery procedures.
- Runtime logs belong under `runtime/logs/`, not at the repository root.

## Verification

```bash
oqp doctor
oqp test smoke
```

Before publishing or deploying, also review
[repository commit readiness](deployment/repo_commit_readiness.md).
