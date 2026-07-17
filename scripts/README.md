# Command entrypoints

`scripts/` contains operator-facing entrypoints. Reusable business logic belongs
in `src/oqp/`; a script should normally parse arguments, load configuration, call
a package-owned command, and return an exit code.

## Start Here

Public onboarding and dashboard launch commands use the installed `oqp` CLI,
not a script path:

```bash
oqp init --profile demo
oqp doctor
oqp dashboard research
oqp dashboard ops
```

Use the scripts below for domain-specific jobs, experiments, deployment, and
scheduled operations after the basic runtime is understood.

## Responsibility Map

| Area | Owner and purpose |
| --- | --- |
| Data | Manual imports and data-refresh commands |
| Ops | Account, portfolio, broker-readiness, and health-snapshot jobs |
| Platform | Dashboard launchers, deployment helpers, runtime sync, and repository hygiene |
| Research | Reproducible research and backtest entrypoints |
| Trading | Paper execution and QMT connector commands |

Domain subdirectories are the target layout. Older root-level scripts may
remain during migration; treat their current path as compatibility, not as a
reason to add another root-level entrypoint.

The daily-regime files directly under `research/` are intentionally frozen in
place because their paths are members of the release manifest. Exploratory CN
futures studies live under `research/experiments/` and are indexed there.

## Rules

- Parse arguments and environment configuration at the script boundary.
- Call reusable package APIs instead of copying calculations into scripts.
- Write generated output to `runtime/`, never beside the script.
- Make scheduled jobs idempotent where practical and document their health
  evidence and recovery path.
- Keep credentials in environment or approved secret storage.

When moving an entrypoint, update scheduler units, runbooks, and shell callers in
the same change. Do not leave a compatibility wrapper unless an external caller
cannot be migrated atomically.

## Verification

Use `--help` before running an unfamiliar entrypoint, then run the focused test
lane owned by its package:

```bash
python scripts/<domain>/<command>.py --help
oqp test smoke
```

Application launch commands are documented in [Applications](../apps/README.md).
