# Applications

`apps/` contains Alpha Factory's Streamlit user interfaces. Applications compose and
present package-owned capabilities; reusable calculations, contracts, storage,
and safety rules belong in `src/oqp/`.

## Start Here

Initialize the broker-free profile once, then launch either dashboard:

```bash
oqp init --profile demo
oqp dashboard research
oqp dashboard ops
```

The Research Dashboard uses port `8524`; the Ops Dashboard uses port `8529`.

## Application Map

| Application | Audience | Guide |
| --- | --- | --- |
| `research_dashboard/` | Factor discovery, diagnostics, backtests, and promotion | [Research Dashboard](research_dashboard/README.md) |
| `ops_dashboard/` | Accounts, portfolios, paper trading, options, and operating health | [Ops Dashboard](ops_dashboard/README.md) |

## Rules

- Streamlit pages may load data, call services, and render results.
- Domain logic must live in the owning `src/oqp/` package.
- Shared UI components and English/Chinese text belong in `src/oqp/ui/`.
- Dashboard code must respect the selected runtime profile and must not embed
  credentials or broker-specific secrets.
- Live submission must remain independently gated from read-only monitoring.

## Verification

```bash
oqp test smoke
python -m pytest -q tests
```

See the [public onboarding guide](../docs/START_HERE.md) for installation and
runtime-profile details.
