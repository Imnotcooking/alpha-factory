# Repository Commit Readiness

Last reviewed: 2026-06-25

This note records the safe Git boundary for the OQP restructuring work. The
goal is to keep the main architecture commit clean while avoiding accidental
changes to alpha research work supervised elsewhere.

## Current Safety Status

- Secrets and runtime files are ignored through `.gitignore`:
  `.env`, `.env.*`, Streamlit secrets, API key JSON files, broker/default
  config JSON files, SQLite databases, logs, runtime artifacts, raw portfolio
  data, CSV/parquet files, model checkpoints, and native build outputs.
- Docker build context is also shielded through `.dockerignore`, including
  local envs, secrets, runtime state, broker exports, model checkpoints, and
  `watchlist.json`.
- The archived Middle Office folder remains only as a compatibility fallback
  for old local portfolio files. Its local secrets, watchlists, databases, and
  raw portfolio data are ignored.
- No active OQP app should import the retired root `Middle_Office/` path.
  Compatibility lookups now prefer `departments/archive/legacy_middle_office/`
  when legacy local files are needed.

## Commit Boundary

The restructuring commit should include the active platform surface:

- `.gitignore`
- `.dockerignore`
- `.env.example`
- `apps/`
- `src/oqp/`
- `scripts/`
- `tests/`
- `departments/`

The restructuring commit should include the old root app deletions only when
we are ready to fully retire them from the root:

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
- `departments/archive/legacy_alpha_factory/`
- `departments/archive/legacy_dashboards/`

## Do Not Mix Into This Commit

- `alpha_research_lab/` changes should remain owned by the separate alpha lab
  workstream.
- Alpha factor implementations and research edge artifacts are private by
  default. Do not publish `alpha_research_lab/factors/fac_*.py`, factor
  metadata, execution logs, cached research data, promotion/trial/candidate
  artifacts, trained local model JSON/PKL files, or archive workbench scripts.
- Retired factors can be committed later, but only as a deliberate public
  allowlist. Treat them like educational examples: freeze the code, remove
  proprietary parameters or dataset-specific tricks, strip performance sweep
  logs, mark the factor as retired/deprecated, and move or copy it into a
  public examples or retired-factors path before staging.
- The fuller policy lives in
  `departments/research/public_private_boundary.md`.
- Existing staged churn in `Photos/`, `backtest_1/`, old notebooks, compiled
  objects, and old model artifacts should be reviewed separately before any
  final Git commit.
- Ignored local folders under archived legacy projects, especially old
  `venv/` and `.venv/`, do not need to be committed and can be deleted from
  disk when local cleanup is desired.

## Immediate Next Action

Before committing, stage only the OQP restructuring paths and inspect the
staged diff. If the alpha lab is still dirty, avoid `git add -A`; use explicit
path staging instead.

Run the hygiene checker before committing:

```bash
python scripts/check_public_commit_hygiene.py
git diff --cached --stat
git diff --cached --name-only
```

For a broader audit of everything currently dirty:

```bash
python scripts/check_public_commit_hygiene.py --all
```

The `--all` audit may fail while another workstream is actively editing private
alpha-lab code. That is useful signal, not a reason to hide those changes.
Public commits should remain explicit-path staged.
