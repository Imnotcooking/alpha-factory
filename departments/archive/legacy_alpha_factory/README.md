# Legacy Alpha Factory Archive

This folder contains the old root-level Alpha Factory prototype that was used
before the unified OQP structure.

Archived entrypoints include:

- `Dockerfile`
- `app.py`
- `master_daily_cron.py`
- `database_engine.py`
- `portfolio_optimizer.py`
- `strategy_agents/`
- `offline_quant_lab/`
- `models/`
- local runtime databases and parquet feature matrices

The active server workflow now lives in:

- `scripts/run_portfolio_snapshot_job.sh`
- `scripts/run_paper_snapshot_job.sh`
- `src/oqp/portfolio/`
- `src/oqp/paper_trading/`
- `apps/`
- `departments/`

Do not run `master_daily_cron.py` as production automation. It used direct
paper-order placement and has been replaced by read-only live monitoring,
read-only paper monitoring, separate SQLite ledgers, and Discord health alerts.
