# Research Dashboard Runbook

This runbook describes how the migrated research dashboard is expected to run inside the upgraded architecture.

## Launch

Use the shared dashboard launcher from the repository root:

```bash
./scripts/start_streamlit_dashboard.sh apps/research_dashboard/Homepage.py 8524 "research dashboard"
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

## Evidence Tickets

The analyst workflow is backed by two related research ledgers:

| Ledger | Purpose | API |
|---|---|---|
| `research_trials` | Statistical trial history and multiple-testing adjustment | `oqp.research.record_research_trial` |
| `research_evidence_tickets` | Analyst-facing evidence tickets that connect discovery, validation, context, promotion, and review | `oqp.research.record_evidence_ticket` |

Evidence tickets should be saved from discovery and validation pages before a
candidate reaches the factor review board. Each ticket can link to `factor_id`,
`run_id`, `trial_signature`, metrics, artifacts, and context JSON without
mixing UI workflow state into the statistical trial ledger.

## Page Map

| Page | Role | Primary Dependencies | Smoke Coverage |
|---|---|---|---|
| `01_Data_Health` | Check data/artifact readiness before research | runtime data, artifact roots, DB schema, native extension status | system health snapshot smoke |
| `02_Pattern_Lab` | Discover daily, intraday, and tick-level patterns before forming hypotheses | `runtime/data/futures_cn/daily/`, `runtime/data/futures_cn/intraday/`, `runtime/data/futures_cn/tick/`, `src/oqp/research/tick_pulse/` | preflight wrapper/import tests |
| `03_Intraday_Event_Study` | Test intraday/tick-pulse directional hypotheses and saved seeds | tick data, research cache, `src/oqp/research/tick_pulse/` | preflight wrapper/import tests |
| `04_Arbitrage_Lab` | Scan pair/spread dislocations and relationship stability | `runtime/data/futures_cn/daily/`, `src/oqp/research/state_space/` | real-data relationship scan |
| `05_Regime_Analysis` | Interpret GMM regimes and latent/VQ cross-checks | feature matrix, GMM probabilities, latent artifacts | real-data regime/latent smoke |
| `06_Market_Breadth_Lab` | Estimate market covariance breadth across vectorizable asset classes | `runtime/data/*/daily/`, `src/oqp/risk/factor_breadth.py` | real-data breadth smoke |
| `07_Feature_Review` | Rank feature quality, redundancy, MDA, latent diagnostics | feature matrices, `src/oqp/research/ml/` | real-data governance smoke |
| `08_Factor_Review` | Review factor evidence and candidate status | `research_memory.db`, factor files, diagnostics artifacts | `test_research_dashboard_real_data_smoke.py` board load |
| `09_Strategy_Comparison` | Compare completed backtest runs | research DB, returns/trades artifacts | real-data run ledger load |

## Verification

One-command local dashboard check:

```bash
./scripts/research/check_research_dashboard.sh
```

Fast preflight:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/oqp_pycache PYTHONPATH=src:. \
  .venv/bin/python -m pytest tests/test_research_dashboard_preflight.py -q
```

Local real-data smoke:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/oqp_pycache PYTHONPATH=src:. \
  .venv/bin/python -m pytest tests/test_research_dashboard_real_data_smoke.py -q
```

Focused research dashboard suite:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/oqp_pycache PYTHONPATH=src:. \
  .venv/bin/python -m pytest \
  tests/test_research_dashboard_preflight.py \
  tests/test_research_dashboard_real_data_smoke.py \
  tests/test_research_ml_governance.py \
  tests/test_research_latent.py \
  tests/test_risk_factor_breadth.py \
  tests/test_tick_pulse_features.py \
  tests/test_tick_pulse_asset_ranker.py \
  tests/test_tick_pulse_ml_migration.py \
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
