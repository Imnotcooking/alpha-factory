# Manual Adjustment Controls

Last reviewed: 2026-07-17

Manual records cover externally held assets and approved corrections that are
not represented by connected broker feeds. They must remain exceptional,
attributable, and reversible.

## Current Source

The active external-holdings source is:

```text
runtime/state/portfolio/manual_external_holdings.json
```

`src/oqp/accounts/manual_external.py` validates it, retains a previous valid
backup, synchronizes active rows into the account ledger, and labels resulting
positions as manual. `unified_live` may include these positions in consolidated
reporting.

## Required Evidence

Each row should identify:

- stable external position id and normalized symbol;
- asset class, quantity, native currency, and multiplier;
- market price, FX rate, source, and as-of timestamp;
- custodian or external account label without public account identifiers;
- owner, reason, and review note;
- active/inactive status rather than destructive deletion.

## Control Rules

- Manual rows never overwrite broker rows.
- Manual cash, buying power, fills, and margin are prohibited unless a separate
  approved contract is introduced.
- Missing or stale marks remain visible; editing the file does not refresh the
  market timestamp.
- The previous valid file and change evidence must be retained.
- Public logs and screenshots must redact account and custodian identifiers.

## Approval Gap

The current JSON editor and backup provide validation and rollback, but not a
full maker-checker workflow. Before multiple operators or live execution use
the consolidated account, add request, approval, effective-time, and revocation
records to the account control ledger. Until then, manual edits require human
review outside the application and must not trigger automated orders.
