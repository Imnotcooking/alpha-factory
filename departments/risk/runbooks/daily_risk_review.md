# Daily Risk Review Runbook

## Preconditions

- Latest required account sources are fresh.
- Critical Middle Office reconciliation breaks are reviewed.
- Market marks, FX, multipliers, and option contracts have provenance.
- Limit catalog and calculation versions are recorded.

## Sequence

1. Select the reconciled operational account cut and reporting currency.
2. Review missing, stale, synthetic, and excluded risk inputs.
3. Review gross/net exposure, leverage, cash, sector, currency, and concentration.
4. Review daily PnL, drawdown, historical tail metrics, and their caveats.
5. Review option packages, delta, gamma, theta, vega, DTE, liquidity, and model gaps.
6. Evaluate every active control in `limit_catalog.yaml`.
7. Run approved deterministic scenarios when the scenario engine is available.
8. Record warnings, hard breaches, owner, and required action.
9. Confirm Trading received any approved block state.
10. Sign off complete, complete-with-exception, or incomplete.

## Status

- **Complete:** required inputs are fresh and no hard breach is open.
- **Complete with exception:** approved waiver exists with owner and expiry.
- **Incomplete:** required data is unavailable or a hard breach remains unresolved.

An observational metric cannot by itself make the review complete or incomplete.
It provides evidence until an approved threshold promotes it.
