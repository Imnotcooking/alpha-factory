# Risk Department Map

Last reviewed: 2026-07-17

Risk owns independent measurement, risk appetite, limit definitions, scenario
standards, model-risk review, and breach escalation across Alpha Factory. It consumes
reconciled account truth and approved market data. It does not create alpha,
change broker records, or submit orders.

## Start Here

Review breadth and model diagnostics in the Research Dashboard, then verify the
risk contracts independently:

```bash
oqp dashboard research
python -m pytest -q tests -k "risk or option"
```

## Ownership Boundary

| Responsibility | Canonical home | Risk relationship |
| --- | --- | --- |
| Account, cash, NAV, and reconciliation truth | `src/oqp/accounts/`, `departments/middle_office/` | Risk consumes only attributed, timestamped account evidence. |
| Market, reference, and freshness contracts | `src/oqp/data/`, `departments/data_platform/` | Risk records data and imputation provenance. |
| Exposure, breadth, volatility, limits, and scenarios | `src/oqp/risk/` | Risk owns reusable calculations and control decisions. |
| Option contracts, marks, Greeks, payoff, and margin models | `src/oqp/options/` | Risk defines review standards and limits; Options owns calculations. |
| Allocation algorithms such as HRP, Kelly, and vol targeting | `src/oqp/research/backtesting/` | Risk constrains outputs but does not own the allocator. |
| Order checks and enforcement | `src/oqp/execution/`, `src/oqp/paper_trading/` | Trading enforces approved hard blocks produced by Risk. |
| Operational presentation | `apps/ops_dashboard/` | Dashboards display package-owned results; they do not define limits. |

## Department Files

- `options_risk/README.md`: current listed-option risk boundary and model gaps.
- `options_risk/phase2_readaptation.md`: recorded next-stage option-risk work.

Risk appetite, limit catalogs, scenario policy, and breach runbooks should be
added here with their implementation owner and approval status made explicit.

## Current Implementation Map

| Capability | Implementation | Status |
| --- | --- | --- |
| Portfolio exposure and legacy historical NAV risk | `src/oqp/risk/portfolio.py` | Migration: legacy frame schemas. |
| PCA covariance breadth and component stability | `src/oqp/risk/factor_breadth.py` | Active research diagnostic. |
| Realized volatility and imputation comparison | `src/oqp/risk/realized_volatility.py` | Active risk-view diagnostic. |
| Live factor proxies and PCA crowding | `src/oqp/risk/live_factor_lab.py` | Active diagnostic. |
| Live position, sector, currency, and concentration views | `src/oqp/portfolio/live_reporting.py` | Active but misplaced; migrate to Risk. |
| Option book and strategy risk | `src/oqp/options/` | Active with documented model gaps. |
| Portfolio scenario repricing | Not implemented | Planned. |
| Persistent risk snapshots and breaches | Not implemented | Planned. |

## Migration Debt

- `src/oqp/risk/portfolio.py` reads the old `live_positions` and
  `historical_nav` schemas rather than canonical account snapshots.
- Live Ops exposure and concentration calculations currently live in
  `src/oqp/portfolio/live_reporting.py`.
- `src/oqp/risk/portfolio.py` contains a second Black-Scholes implementation;
  the option model spine belongs in `src/oqp/options/`.
- Historical NAV VaR/CVaR does not reprice options or distinguish cash flows
  from investment returns.
- Execution settings contain partial safety checks, but there is no central,
  approved portfolio limit service or post-trade exposure calculation yet.

These are explicit migration targets, not endorsed permanent boundaries.

## Runtime Evidence

Risk inputs remain in the account, data, and market runtime lanes. Generated
research diagnostics belong under `runtime/artifacts/`; operational logs belong
under `runtime/logs/`. There is no canonical `risk_ledger.db` yet. Do not create
an ad hoc database from a dashboard page.

## Change Procedure

1. Define the metric, owner, interpretation, and intended control posture.
2. Keep it observational until data, calculation, and interpretation are tested.
3. Add package-owned calculation and tests under `src/oqp/risk/`.
4. Approve warning and hard thresholds in a reviewed risk policy before enforcement.
5. Wire hard controls into Execution without duplicating the calculation there.
6. Persist risk snapshots and breach actions only after the ledger contract is
   reviewed.

## Guardrails

- Missing or stale inputs cannot produce a passing hard control.
- Brownian Bridge output is a risk-only synthetic view, never market truth.
- Option margin helpers are conservative research approximations, not broker
  margin guarantees.
- A model estimate and a broker-reported value must remain separately labelled.
- `ALLOW_LIVE_TRADING=false` is a safety posture, not a substitute for risk
  appetite or limits.
