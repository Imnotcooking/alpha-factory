# Test organization

Tests mirror the package and operating domains they protect:

## Start Here

Run the public onboarding lane first:

```bash
oqp test smoke
```

Then use `-k` or an explicit test module to run the smallest lane covering the
code you changed, for example:

```bash
python -m pytest -q tests -k data
python -m pytest -q tests -k research_dashboard
```

## Domain Map

| Domain | Coverage |
| --- | --- |
| Accounts | Account ledger, reconciliation, and source catalog |
| Architecture | Public/private boundaries, package imports, and translations |
| Data | Catalogs, taxonomy, missingness, market data, and return horizons |
| Execution | Candidate and trade-proposal contracts |
| Investing | Research-assisted investing, journal, and evidence workflows |
| Ops | Broker readiness and operational health |
| Options | Pricing, analytics, book, proposal, and backtesting behavior |
| Onboarding | Public CLI, demo fixtures, runtime isolation, and doctor checks |
| Portfolio | Ingestion, valuation, NAV, ledger, and reporting |
| Research | Backtesting, ML, state-space, tick, regime, and dashboard behavior |
| Risk | Breadth, limits, volatility, and portfolio risk |
| Trading | Paper trading and QMT connector safety |

Tests may be grouped into matching domain directories or retain a legacy flat
filename while the test-layout migration is incomplete.

The `test_daily_regime_*.py` files intentionally remain at the root. Their exact
paths are included in the frozen daily-regime release and recovery manifests.
Moving them would change the reproducibility seal.

## Rules

- Mirror the owning `src/oqp/` domain when choosing a test directory.
- Prefer deterministic synthetic fixtures over vendor or broker connectivity.
- Keep external-service checks opt-in and safe when credentials are absent.
- Give regressions a focused test close to the contract they protect.
- Do not make the broad suite depend on private data, live accounts, or local
  absolute paths.

Private factor implementation tests live in
`tests/research/private_factors/`. They use synthetic data, remain a separate
local pytest lane, and are excluded from the public repository in `.gitignore`.

Package ownership is documented in [Source Layout](../src/README.md).
