# Alpha Research Public Allowlist

Last reviewed: 2026-06-25

Use this allowlist when preparing a public GitHub commit. It is intentionally
conservative: if a file is not clearly listed here, review it before staging.

## Public Framework Candidates

These paths are suitable public candidates when their contents remain free of
credentials, account data, real vendor exports, and proprietary results:

- `alpha_research_lab/README.md`
- `alpha_research_lab/requirements.txt`
- `alpha_research_lab/config.py`
- `alpha_research_lab/factor_contract.py`
- `alpha_research_lab/factors/__init__.py`
- `alpha_research_lab/factors/contracts.py`
- `alpha_research_lab/factors/factor_metadata_template.py`
- `alpha_research_lab/evaluation/`
- `alpha_research_lab/execution/`
- `alpha_research_lab/data_engine/dataset_policy.py`
- `alpha_research_lab/data_engine/instrument_master.py`
- `alpha_research_lab/public_examples/`

## Public Test Candidates

Prefer tests that use synthetic data and synthetic/demo factors:

- `tests/test_alpha_public_examples.py`
- `alpha_research_lab/tests/test_factor_contract.py`
- `alpha_research_lab/tests/test_factor_contract_coverage.py`
- `alpha_research_lab/tests/test_evaluation_split_policy.py`
- `alpha_research_lab/tests/test_execution_*.py`

Do not stage tests that import live private `alpha_research_lab/factors/fac_*.py`
modules unless they are rewritten to use synthetic fixtures or public examples.

## Private Unless Sanitized

Keep these out of the public commit:

- `alpha_research_lab/factors/fac_*.py`
- `alpha_research_lab/data_cache/`
- `alpha_research_lab/execution_logs/`
- `alpha_research_lab/ml_engine/*model*.json`
- `alpha_research_lab/regime_engine/*.pkl`
- `alpha_research_lab/archive/`
- `departments/archive/legacy_alpha_factory/**/factor_library/fac_*.py`
- `departments/archive/legacy_alpha_factory/**/models/`
- `departments/archive/legacy_alpha_factory/strategy_agents/agent_*.py`
- candidate, trial, promotion, sweep, and backtest result artifacts

## Retired Factor Promotion

To publish a retired factor, copy it into `alpha_research_lab/public_examples/`
or another explicit public examples path, strip live parameters and comments,
mark it as retired or educational, and add a synthetic test. Never stage it
directly from the live `alpha_research_lab/factors/` directory.
