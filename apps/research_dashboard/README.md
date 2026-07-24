# Research Dashboard

The Research Dashboard is the analyst-facing workspace for data readiness,
signal discovery, market structure, feature governance, research review, and
strategy construction. It reads research evidence; reusable research engines
remain in `src/oqp/research/`.

## Start Here

```bash
oqp init --profile demo
oqp dashboard research
```

Open [http://127.0.0.1:8524](http://127.0.0.1:8524). For operational launch,
runtime roots, and troubleshooting, read the [runbook](RUNBOOK.md).

## Page Map

| Order | Page | Primary question |
| --- | --- | --- |
| Home | Research ledger | What has been tested, and what evidence exists? |
| 01 | Data Health | Is the selected dataset usable and sufficiently fresh? |
| 02 | Discovery Lab | Which patterns, defined events, or cross-asset relationships deserve formal validation? |
| 05 | Regime Analysis | Which market state is active and how stable is it? |
| 06 | Market Breadth Lab | Is participation, concentration, volatility, or risk broad? |
| 07 | ML Hub | Are feature data, model experiments, artifacts, and promotion evidence reproducible and useful out of sample? |
| 08 | Research Review | Which factors, sleeves, routers, and constructed strategies survive evidence, execution, and governance checks? |

## Code Map

| Location | Responsibility |
| --- | --- |
| `Homepage.py` | Run ledger and cross-view composition |
| `pages/` | Streamlit page entrypoints |
| `views/` | Reusable homepage views |
| `discovery/` | Embedded discovery implementations behind the consolidated workflow shell |
| `arbitrage_lab/`, `tick_pulse_lab/`, `quartile_router_lab/`, `factor_portfolio_lab/` | Page-local presentation helpers used by those implementations |
| `config.py`, `ui_state.py` | Runtime-aware configuration and UI state |

Calculations should move to `src/oqp/` once they are used by more than one page
or become part of a reproducible research contract.

## Data And Runtime

- Market and derived datasets: `runtime/data/`
- Research databases: `runtime/db/research/`
- Backtest and model evidence: `runtime/artifacts/research/`
- Demo equivalents: `runtime/demo/`

Pages must use shared runtime-path helpers rather than hard-coded local paths.

## Verification

```bash
oqp test smoke
python -m pytest -q tests -k research_dashboard
```

The [page roadmap](PAGE_ROADMAP.md) records product intent; it is not a second
source of executable behavior.
