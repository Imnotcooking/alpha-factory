# Start Here

This guide explains how to approach Oxford Quant Pipeline without already
knowing its folder structure, brokers, or research history.

## 1. Pick An Audience Path

### Viewer

You want to inspect the architecture and dashboards without supplying data or
credentials.

```bash
oqp init --profile demo
oqp doctor
oqp dashboard research
```

Then launch `oqp dashboard ops` in another terminal.

### Researcher

Start with the Research Dashboard, `departments/research/README.md`, and the
public factor template. Reusable research logic belongs in `src/oqp/research/`;
repeatable runs belong in `scripts/research/`; local outputs belong in
`runtime/artifacts/research/`.

### Operator

Start with the Ops Dashboard and `departments/platform/deployment/`. Account
snapshots use `src/oqp/accounts/`; portfolio valuation uses
`src/oqp/portfolio/`; paper execution uses `src/oqp/paper_trading/`.

### Contributor

Read `ARCHITECTURE.md`, `scripts/README.md`, and `tests/README.md`. Add reusable
logic to the owning `src/oqp/` domain and a focused test to the matching test
directory.

## 2. Understand The Four Layers

| Layer | Purpose | Should contain |
| --- | --- | --- |
| `apps/` | User interfaces | Streamlit composition and display logic |
| `src/oqp/` | Product code | Reusable contracts, calculations, storage, services |
| `scripts/` | Entry points | Argument parsing and orchestration |
| `departments/` | Governance | Ownership, policy, contracts, and runbooks |

`runtime/` is generated local state. It is ignored by Git and should never be
treated as source code.

## 3. Why `src/oqp` Exists

`src/` is the Python packaging layout. It prevents tests and scripts from
silently importing files merely because the current directory happens to be the
repository root. `oqp` is the stable package namespace, so imports remain
consistent in local development, CI, servers, and installed environments:

```python
from oqp.accounts import AccountSnapshot
from oqp.options import OptionBacktestEngine
from oqp.research.backtesting import BacktestExecutionEngine
```

Flattening those modules directly into `src/` would remove the package boundary
and create naming collisions with other Python libraries.

## 4. Demo Isolation

The demo profile uses these path overrides:

```text
runtime/demo/data/
runtime/demo/artifacts/research/
runtime/demo/db/research/research_memory.db
runtime/demo/db/accounts/account_ledger.db
```

The normal runtime remains under `runtime/data`, `runtime/artifacts`, and
`runtime/db`. Dashboard commands read the selected profile marker and inject the
matching paths into the child Streamlit process. This is why a demo tour cannot
overwrite a real account or research ledger.

## 5. No Matching Broker Required

The core system is broker-neutral. Dashboards consume canonical account,
position, cash, NAV, trade, and order contracts. IBKR and QMT are implementations
of those contracts.

For another broker:

1. implement `BrokerAdapter` under `src/oqp/brokers/`;
2. normalize its data into `AccountSnapshot` and `TradeEvent` records;
3. register a named profile;
4. keep submission disabled until read-only and paper tests pass;
5. add contract tests without embedding credentials.

Manual CSV/JSON snapshots are also valid for read-only portfolio monitoring.

## 6. Verification Ladder

Use the smallest relevant lane first:

```bash
oqp doctor
oqp test smoke
python -m pytest -q tests -k account
python -m pytest -q tests -k option
python -m pytest -q tests -k research
```

The complete suite is intentionally broad. Begin with the smallest directory or
test module owned by the domain you changed before running the complete suite.

## 7. Safety Before Connectivity

Never begin by enabling execution. First establish:

1. deterministic data and account normalization;
2. read-only monitoring;
3. paper review;
4. separately gated paper submission;
5. reconciliation and health checks;
6. only then, an explicitly approved live path.

`oqp doctor` reports the live gate, but it does not connect to a broker or place
orders.
