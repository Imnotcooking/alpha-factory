# Data Quality Policy

Last reviewed: 2026-07-17

## Core Principle

Raw data remains faithful to the source. Corrections and imputations are
derived views with explicit quality flags; they never overwrite the raw input.

## Missing-Data Views

| View | Permitted treatment | Intended use |
| --- | --- | --- |
| Raw | Preserve observed values and gaps. | Audit, lineage, and source comparison. |
| Accounting | Forward-fill within an explicit stale-bar limit and retain freshness flags. | NAV, exposure, and margin valuation. |
| Alpha | Use fresh observations only; stale rows cannot update features or create orders. | Signals, factors, and backtests. |
| Risk | Use raw/forward-filled data or an explicitly selected Brownian Bridge reconstruction. | Covariance, realized volatility, stress tests, and Monte Carlo research. |

Brownian Bridge output is synthetic risk evidence. It is not executable price
history and must never enter alpha or order-generation logic.

## Dataset Gates

Before a dataset becomes canonical, check:

- required schema and stable data types;
- parseable, monotonic timestamps within each instrument;
- duplicate instrument/timestamp keys;
- valid prices, volumes, strikes, expiries, and contract identifiers;
- exchange calendar and session consistency;
- corporate-action or futures-roll treatment where applicable;
- point-in-time universe membership and survivorship policy;
- row count, asset count, date range, and freshness;
- source and output hashes for generated datasets.

## Severity

- `FAIL`: unreadable required data, missing required columns, duplicate primary
  keys, invalid time ordering, or lineage mismatch. Do not promote downstream
  evidence.
- `WARN`: optional lane absent, stale observations, partial metadata, or a
  recoverable quality limitation. Research may continue only with disclosure.
- `OK`: checks passed for the declared use. This does not certify profitability
  or execution suitability.

Failed output must not replace a canonical dataset. Keep diagnostics under
`runtime/artifacts/` and rerun the controlled backfill procedure.
