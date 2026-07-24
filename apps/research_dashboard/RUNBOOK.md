# Research Dashboard Runbook

This runbook describes how the migrated research dashboard is expected to run inside the upgraded architecture.

## Launch

Use the shared dashboard launcher from the repository root:

```bash
./scripts/platform/start_streamlit_dashboard.sh apps/research_dashboard/Homepage.py 8524 "research dashboard"
```

The dashboard should be available at:

```text
http://127.0.0.1:8524
```

Runtime logs belong under:

```text
runtime/logs/research_dashboard.log
```

The dashboard imports reusable research logic from `src/oqp/`. App-local modules under `apps/research_dashboard/` should mostly contain Streamlit layout, compatibility wrappers, and page-specific presentation helpers.

## Runtime Roots

The dashboard uses these architecture roots:

| Purpose | Path |
|---|---|
| Research runtime data | `runtime/data/` |
| CN futures daily market data | `runtime/data/futures_cn/daily/` |
| Research artifacts and result files | `runtime/artifacts/research/` |
| Research database | `runtime/db/research/research_memory.db` |
| Shared Python package logic | `src/oqp/` |
| Streamlit app shell | `apps/research_dashboard/` |

For public GitHub use, private runtime files may be absent. Pages should show clear empty states or skip local-only smoke checks rather than crashing.

## Idea Evidence Ledgers

The analyst workflow is backed by two related research ledgers:

| Ledger | Purpose | API |
|---|---|---|
| `research_trials` | Statistical trial history and multiple-testing adjustment | `oqp.research.record_research_trial` |
| `research_evidence_tickets` | Lab-local idea evidence that connects discovery hypotheses, context, metrics, and artifacts | `oqp.research.record_evidence_ticket` |

Evidence tickets belong to Pattern Scan, Event Study, and Relationship Scan in
Discovery Lab. They remain separate from Research Review until an idea has been
implemented as a developed factor with recorded run evidence. Each ticket can
link to `factor_id`, `run_id`, `trial_signature`, metrics, artifacts, and
context JSON without mixing exploratory workflow state into the statistical
trial ledger.

## Page Map

| Page | Role | Primary Dependencies | Smoke Coverage |
|---|---|---|---|
| `01_Data_Health` | Check data/artifact readiness before research | runtime data, artifact roots, DB schema, native extension status | system health snapshot smoke |
| `02_Discovery_Lab` | Scan multi-timeframe patterns, test intraday/tick events, and investigate pair/spread relationships | daily/intraday/tick data, research cache, `src/oqp/research/tick_pulse/`, `src/oqp/research/ml/state_space/`, and `departments/research/workflows/statistical_arbitrage/` | preflight, evidence-ticket, and real-data relationship tests |
| `05_Regime_Analysis` | Interpret GMM regimes and latent/VQ cross-checks | feature matrix, GMM probabilities, latent artifacts | real-data regime/latent smoke |
| `06_Market_Breadth_Lab` | Estimate market covariance breadth across vectorizable asset classes | `runtime/data/*/daily/`, `src/oqp/risk/factor_breadth.py` | real-data breadth smoke |
| `07_ML_Hub` | Govern feature data, model experiments, registered artifacts, MDA, and latent diagnostics | feature matrices, research DB, `src/oqp/research/ml/` | real-data governance and page AppTest smoke |
| `08_Research_Review` | Review factor, sleeve, and router components; construct strategies; compare completed runs | `research_memory.db`, component registries, returns/trades and diagnostics artifacts | board, library, construction, and run-comparison smoke tests |

## Verification

One-command local dashboard check:

```bash
./scripts/research/check_research_dashboard.sh
```

Fast preflight:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/oqp_pycache PYTHONPATH=src:. \
  .venv/bin/python -m pytest tests/research/dashboard/test_research_dashboard_preflight.py -q
```

Local real-data smoke:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/oqp_pycache PYTHONPATH=src:. \
  .venv/bin/python -m pytest tests/research/dashboard/test_research_dashboard_real_data_smoke.py -q
```

Focused research dashboard suite:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/oqp_pycache PYTHONPATH=src:. \
  .venv/bin/python -m pytest \
  tests/research/dashboard/test_research_dashboard_preflight.py \
  tests/research/dashboard/test_research_dashboard_real_data_smoke.py \
  tests/research/test_research_ml_governance.py \
  tests/research/test_research_latent.py \
  tests/risk/test_risk_factor_breadth.py \
  tests/research/test_tick_pulse_features.py \
  tests/research/test_tick_pulse_asset_ranker.py \
  tests/research/test_tick_pulse_ml_migration.py \
  -q
```

## Debug Checklist

1. Confirm the server is alive:

```bash
curl -sS http://127.0.0.1:8524/_stcore/health
```

2. Check the latest dashboard log:

```bash
tail -n 160 runtime/logs/research_dashboard.log
```

3. If a page import fails, check whether the failing dependency belongs in `src/oqp/` and whether the app-local wrapper re-exports the legacy UI helper names.

4. If a page data load fails, verify the page's runtime root first. Avoid copying private data into app folders; place CN futures market data under `runtime/data/futures_cn/{daily,intraday,tick}/`, generated research matrices under `runtime/data/`, and outputs under `runtime/artifacts/research/`.

5. If a page works locally but would fail on a public clone, add a public-safe empty state or an optional smoke test skip.

## Design Rules

- Reusable research engines live in `src/oqp/`.
- Streamlit pages should orchestrate controls, presentation, and page-specific explanations.
- App-local compatibility wrappers are acceptable during migration, but they should delegate to `src/oqp/`.
- Runtime data and generated artifacts stay under `runtime/`, not inside `apps/` or `src/`.
- Tests should protect both public importability and local real-data usability.
