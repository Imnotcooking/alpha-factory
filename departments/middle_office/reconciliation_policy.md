# Account Reconciliation Policy

Last reviewed: 2026-07-17

Reconciliation compares authoritative source evidence with a normalized or
derived account view. It reports differences; it never edits either side.

## Comparison Order

1. Confirm environment, account identity, timestamps, and base currency.
2. Normalize instrument identifiers through the instrument master where
   available.
3. Compare position presence, quantity, multiplier, and market value.
4. Compare cash by currency and total account cash.
5. Compare NAV and account-level totals.
6. Compare internal order/trade events with broker orders and fills when that
   evidence is available.

## Position Identity

The current canonical key is normalized symbol, asset class, and currency.
Contract multipliers are compared explicitly. Option symbols must preserve the
contract identity; expiry, strike, right, and multiplier must not be collapsed
to an underlying symbol.

## Tolerance Contract

Every automated run must receive an explicit tolerance profile. For a reference
value `r`, absolute tolerance `a`, and relative tolerance `p`, the allowed
difference is:

```text
max(a, abs(r) * p)
```

A field matches when `abs(observed - reference)` is no greater than that limit.
The package default is strict zero tolerance; it is a software default, not an
approved production policy. Production tolerances must be approved per currency
and asset class before alerting is enabled.

## Break Classes

| Class | Examples | Default severity |
| --- | --- | --- |
| Identity | Wrong environment or base currency | Critical |
| Missing position | Present in source but absent from ledger | Critical |
| Unexpected position | Present only in observed view | Critical unless an aggregate policy explicitly permits additions |
| Quantity or multiplier | Share, contract, or option multiplier mismatch | Critical |
| Valuation | Market-value mismatch or missing mark | Warning |
| Cash | Currency balance or total-cash mismatch | Critical |
| NAV | Net-liquidation mismatch | Critical |
| Event | Missing, duplicate, or mismatched order/fill event | Critical |

## Lifecycle

Persistent dashboard workflow is a later integration stage. Its states must be:

```text
open -> acknowledged -> resolved
                     -> waived
```

Each action requires timestamp, owner, reason, evidence reference, and the
original immutable break. A waiver expires or carries an explicit permanent
control decision; it is not deletion.

## Aggregate Accounts

When reconciling a broker snapshot to `unified_live`, approved manual positions
may be allowed as additional observed rows. Shared broker positions, broker cash,
and broker NAV still require comparison. The reconciliation run must state that
additional positions were permitted; this must never be inferred silently.

## Current Scope

`src/oqp/accounts/reconciliation.py` implements read-only snapshot comparison in
memory. Persistence, trade-event matching, ownership workflow, and Ops dashboard
controls remain gated follow-up work.
