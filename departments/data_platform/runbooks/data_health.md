# Data Health Runbook

Last reviewed: 2026-07-17

The Research Dashboard Data Health page is a read-only operational view. It
must not mutate, repair, download, or backfill datasets while rendering.

## Triage Order

1. Resolve `FAIL` rows for required lanes before trusting affected pages.
2. Check provider configuration separately from local file coverage.
3. Verify latest update, data date range, asset count, and row count.
4. Review freshness flags before using an accounting-filled matrix for alpha.
5. Confirm generated research artifacts still reference existing source files
   and manifests.

## Interpretation

- API readiness checks credential and adapter configuration without a live
  network call.
- Runtime folder coverage is driven by `source_catalog.yaml`.
- A missing optional lane is a warning, not a global dashboard failure.
- A present file can still be unusable because of stale dates, bad schema, or
  missing lineage.
- Brownian Bridge statistics are risk-only diagnostics.

## Recovery

Do not repair data inside Streamlit. Identify the canonical catalog lane,
follow `backfill_and_recovery.md`, validate the rebuilt output, then refresh the
health snapshot.
