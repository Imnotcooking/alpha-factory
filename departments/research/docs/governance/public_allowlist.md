# Alpha Research Public Allowlist

Last reviewed: 2026-07-24

Use this allowlist when preparing a public GitHub commit. It is intentionally
conservative: if a file is not clearly listed here, review it before staging.

## Public Framework Candidates

These paths are suitable public candidates when their contents remain free of
credentials, account data, real vendor exports, and proprietary results:

- `apps/research_dashboard/`
- `departments/research/factors/README.md`
- `departments/research/factors/factor_template_private.py`
- component README/template files under `departments/research/routers/`,
  `position_policies/`, `strategy_overlays/`, and `strategies/`
- `departments/research/retired_factors/`
- `src/oqp/research/`
- `src/oqp/data/`
- `src/oqp/research/ml/`
- `src/oqp/native/`
- `notebooks/Phase_7_Research_Projects/07_06_daily_volatility_router_cn_futures_replication_public/`

The `07_06` project is allowlisted only while its boundary tests pass. It may contain
rounded aggregate replication evidence, manuscript sources and generated public figures;
it may not contain private runtime paths, contract-level data, monthly return series,
execution parameters or private factor imports.

## Public Test Candidates

Prefer tests that use synthetic data and synthetic/demo factors:

- `tests/architecture/test_alpha_public_examples.py`
- `tests/research/test_research_*.py`
- `tests/research/dashboard/`
- `tests/research/test_tick_pulse_ml_migration.py`
- `tests/research/test_regime_engine_migration.py`
- organized domain tests under `tests/` that use synthetic fixtures or public examples

Do not stage tests that import live private factor recipes unless they are
rewritten to use synthetic fixtures or public examples.

## Private Unless Sanitized

Keep these out of the public commit:

- `departments/research/factors/`
- `runtime/data/`
- `runtime/data/futures_cn/`
- `runtime/db/research/`
- `runtime/artifacts/research/`
- candidate, trial, promotion, sweep, and backtest result artifacts
- live routers, position policies, sleeves, overlays, strategy recipes, and
  optimization-study definitions under `departments/research/`
- research archives, experiment scripts, and internal factor-performance notes
- generated `reports/` and `output/` artifacts

## Retired Factor Promotion

To publish a retired factor, copy it into
`departments/research/retired_factors/`, strip live parameters and comments,
mark it as retired or educational, and add a synthetic test. Never stage it
directly from `departments/research/factors/`.
