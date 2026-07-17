# Ops Dashboard

The Ops Dashboard is the operating cockpit for canonical account state,
portfolio monitoring, paper trading, discretionary analysis, options, and
system health. It presents evidence from package-owned services and ledgers.

## Start Here

For a safe broker-free tour:

```bash
oqp init --profile demo
oqp dashboard ops
```

Open [http://127.0.0.1:8529](http://127.0.0.1:8529).

## Page Map

| Page | Responsibility |
| --- | --- |
| `Homepage.py` | Operating status, pipeline gates, and account summaries |
| `01_Live_Portfolio.py` | Read-only live and manually reconciled holdings |
| `02_Paper_Trading.py` | Paper proposals, review, and guarded submission state |
| `03_Discretionary_Workbench.py` | Valuation, evidence, and options analysis |
| `04_Risk_Control_Room.py` | Transitional risk and allocation workspace |
| `05_Execution_Strategy_Monitor.py` | Transitional proposal and execution monitor |
| `06_Journal_Reports.py` | Decision journal and operating reports |

## Operating Modes

| Mode | External connectivity | Submission |
| --- | --- | --- |
| Demo | None; deterministic synthetic ledgers | Disabled |
| Manual/read-only | Imported snapshots or read-only adapters | Disabled |
| Paper review | Paper evidence and proposal review | Separately gated |
| Live monitoring | Read-only broker/account monitoring | Disabled by default |

The presence of a broker adapter never implies permission to submit an order.

## Ownership Rules

- Account truth belongs in `src/oqp/accounts/`.
- Portfolio valuation belongs in `src/oqp/portfolio/`.
- Option analytics belong in `src/oqp/options/`.
- Risk measurements and limits belong in `src/oqp/risk/`.
- Order gates and paper execution belong in `src/oqp/execution/` and
  `src/oqp/paper_trading/`.
- Streamlit pages must not implement a second ledger or bypass package safety
  controls.

## Verification

```bash
oqp doctor
oqp test smoke
python -m pytest -q tests -k "ops or account or portfolio"
```

Server and broker setup begins in the
[Platform Department guide](../../departments/platform/README.md), not inside
the dashboard pages.
