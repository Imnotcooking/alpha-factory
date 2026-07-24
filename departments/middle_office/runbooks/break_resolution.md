# Reconciliation Break Resolution Runbook

## Triage

1. Freeze the original source and observed snapshot identifiers.
2. Confirm account profile, environment, operational cut, and timezone.
3. Check symbol normalization, currency, and contract multiplier.
4. Check whether the difference is timing, a genuine source mismatch, stale
   pricing, a missing cash flow, or an expected manual addition.
5. Re-run only after recording why new evidence supersedes the prior evidence.

## Resolution Actions

| Cause | Action |
| --- | --- |
| Late broker snapshot | Wait for a new immutable snapshot; do not edit the old one. |
| Symbol mapping error | Correct the package-owned normalization rule and add a regression test. |
| Stale or missing mark | Refresh approved market data and retain the stale-mark evidence. |
| Manual position error | Submit a reviewed manual change and preserve the previous file. |
| Missing cash flow | Record the deposit, withdrawal, fee, or transfer as explicit evidence. |
| Wrong multiplier or FX | Correct the valuation contract and re-run reconciliation. |
| Genuine unexplained break | Escalate and keep the close incomplete. |

## Evidence

Resolution must retain the break key, original values, tolerance, owner,
timestamp, reason, and evidence location. A resolved or waived break remains in
the audit history. Never delete or overwrite broker evidence to make a control
pass.
