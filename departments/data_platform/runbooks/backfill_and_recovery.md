# Backfill And Recovery Runbook

Last reviewed: 2026-07-17

## Procedure

1. Identify the dataset entry and canonical path in `source_catalog.yaml`.
2. Record provider, request scope, calendar, adjustment mode, and current
   source/output hashes.
3. Reproduce the failure in a temporary run location under `runtime/`.
4. Fetch or rebuild only the required interval using an idempotent command.
5. Normalize symbols through the Instrument Master and preserve source fields.
6. Run schema, duplicate, timestamp, range, freshness, and lineage checks.
7. Compare overlap with the previous canonical data and explain revisions.
8. Write the dataset and manifest atomically only after validation passes.
9. Rerun dependent derived datasets rather than editing them in place.
10. Refresh Data Health and retain recovery evidence under
    `runtime/artifacts/`.

## Safety Rules

- Never overwrite the only copy of a raw vendor extract.
- Never treat a provider fallback as equivalent without recording the source.
- Never fill missing alpha observations merely to make a matrix rectangular.
- Never publish partial option chains without expiry, strike, right, and
  underlying identifiers.
- Never copy runtime data, credentials, or recovery logs into Git.
