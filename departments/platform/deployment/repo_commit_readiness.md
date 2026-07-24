# Repository Commit Readiness

Last reviewed: 2026-07-24

This note records the safe Git boundary for the OQP restructuring work. The
goal is to keep the main architecture commit clean while avoiding accidental
publication of private alpha research edge, runtime state, or local scratch
projects.

## Current Safety Status

- Secrets and runtime files are ignored through `.gitignore`:
  `.env`, `.env.*`, Streamlit secrets, API key JSON files, broker/default
  config JSON files, SQLite databases, logs, runtime artifacts, raw portfolio
  data, CSV/parquet files, model checkpoints, and native build outputs.
- Docker build context is also shielded through `.dockerignore`, including
  local envs, secrets, runtime state, broker exports, model checkpoints, and
  `watchlist.json`.
- The old Middle Office app/archive has been retired from active runtime code.
  Portfolio/account jobs now read only from `.env`/process env and `runtime/`
  paths; do not restore fallback reads from `Middle_Office/` or an archive copy.

## Commit Boundary

Architecture commits should include the active platform surface:

- `.gitignore`
- `.dockerignore`
- `.env.example`
- `apps/`
- `src/oqp/`
- `scripts/`
- `tests/`
- `departments/`
- `notebooks/` when the notebooks are educational/public and contain no
  private alpha edge or generated data outputs

The former `alpha_research_lab/` folder is now decommissioned. Committing its
deletion is allowed once active code imports from `apps/research_dashboard/`,
`src/oqp/research/`, `scripts/research/`, and
`departments/research/` instead.

The restructuring commit should include old root app deletions only when we are
ready to fully retire them from the root:

- `Dockerfile`
- `app.py`
- `database_engine.py`
- `master_daily_cron.py`
- `portfolio_optimizer.py`
- root `models/`
- root `cpp_engine/`

Already retired during cleanup:

- root `offline_quant_lab/`
- root `strategy_agents/`
- legacy alpha-factory, dashboard, and Middle Office archive directories

## Do Not Mix Into Public Commits

- Alpha factor implementations and research edge artifacts are private by
  default. Do not publish live `fac_*.py` recipes, factor metadata, execution
  logs, cached research data, promotion/trial/candidate artifacts, trained
  local model JSON/PKL files, or archive workbench scripts.
- Retired factors can be committed later, but only as a deliberate public
  allowlist. Treat them like educational examples: freeze the code, remove
  proprietary parameters or dataset-specific tricks, strip performance sweep
  logs, mark the factor as retired/deprecated, and move or copy it into a
  public examples or retired-factors path before staging.
- The fuller policy lives in
  `departments/research/docs/governance/public_private_boundary.md`.
- `backtest_1-main/` and the former nested `manager_research_demo/` are not
  public source trees. The manager repository is produced separately through
  `scripts/platform/export_manager_repository.py`, which includes private
  research components but excludes middle-office and runtime state.
- Existing churn in `Photos/`, old notebooks, compiled objects, and old model
  artifacts should be reviewed separately before any final Git commit.
- Ignored local folders under archived legacy projects, especially old
  `venv/` and `.venv/`, do not need to be committed and can be deleted from
  disk when local cleanup is desired.

## Immediate Next Action

Before committing, stage only the intended OQP restructuring paths and inspect
the staged diff. Avoid `git add -A`; use explicit path staging instead.

Run the hygiene checker before committing:

```bash
python scripts/platform/check_public_commit_hygiene.py
git diff --cached --stat
git diff --cached --name-only
```

For a broader audit of everything currently dirty:

```bash
python scripts/platform/check_public_commit_hygiene.py --all
```

The `--all` audit may fail while another workstream is actively editing private
alpha-lab code. That is useful signal, not a reason to hide those changes.
Public commits should remain explicit-path staged.
