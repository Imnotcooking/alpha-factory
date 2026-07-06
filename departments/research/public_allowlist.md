# Alpha Research Public Allowlist

Last reviewed: 2026-07-02

Use this allowlist when preparing a public GitHub commit. It is intentionally
conservative: if a file is not clearly listed here, review it before staging.

## Public Framework Candidates

These paths are suitable public candidates when their contents remain free of
credentials, account data, real vendor exports, and proprietary results:

- `apps/research_dashboard/`
- `departments/research/factors/README.md`
- `departments/research/factors/factor_template_private.py`
- `departments/research/retired_factors/`
- `src/oqp/research/`
- `src/oqp/data/`
- `src/oqp/intelligence/regime_engine/`
- `src/oqp/native/`

## Public Test Candidates

Prefer tests that use synthetic data and synthetic/demo factors:

- `tests/test_alpha_public_examples.py`
- `tests/test_research_*.py`
- `tests/test_tick_pulse_ml_migration.py`
- `tests/test_regime_engine_migration.py`
- promoted root tests under `tests/` that use synthetic fixtures or public examples

Do not stage tests that import live private factor recipes unless they are
rewritten to use synthetic fixtures or public examples.

## Private Unless Sanitized

Keep these out of the public commit:

- `departments/research/factors/`
- `runtime/data/alpha_lab/`
- `runtime/db/research/alpha_lab/`
- `runtime/artifacts/research/alpha_lab/`
- candidate, trial, promotion, sweep, and backtest result artifacts

## Retired Factor Promotion

To publish a retired factor, copy it into
`departments/research/retired_factors/`, strip live parameters and comments,
mark it as retired or educational, and add a synthetic test. Never stage it
directly from `departments/research/factors/`.
