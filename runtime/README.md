# Runtime Storage

`runtime/` contains generated local state. Everything here is private and
ignored by Git except this guide. Source code must never depend on a particular
developer's runtime contents.

## Start Here

Create an isolated, deterministic runtime without broker credentials:

```bash
oqp init --profile demo
oqp doctor
```

## Directory Map

| Directory | Contents |
| --- | --- |
| `data/` | Raw, cached, normalized, and derived datasets by asset class and timeframe |
| `artifacts/` | Backtests, trades, models, diagnostics, reports, and manifests |
| `db/` | Research, account, portfolio, and paper-trading ledgers |
| `state/` | Current process, portfolio, broker, and scheduler state |
| `logs/` | Dashboard, job, execution, and health logs |
| `demo/` | Deterministic broker-free fixtures isolated from normal runtime state |

## Rules

- Never commit market data, broker exports, account state, databases, logs,
  model binaries, or generated research evidence.
- Use shared runtime-path helpers and environment overrides instead of
  hard-coded absolute paths.
- Generated artifacts should record their inputs, parameters, timestamps,
  code revision, and checksums where practical.
- Demo initialization may rebuild `runtime/demo/`; it must not overwrite the
  normal runtime ledgers.
- Treat cleanup as an explicit operating action. Do not recursively delete
  runtime state unless its ownership and recovery path are understood.

Storage ownership and data-retention policy begin in the
[Data Platform guide](../departments/data_platform/README.md).
