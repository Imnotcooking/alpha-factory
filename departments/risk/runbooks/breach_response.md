# Risk Breach Response Runbook

## Immediate Actions

1. Preserve the account snapshot, market-data cut, metric version, and limit version.
2. Confirm whether the breach is genuine, stale-data driven, or calculation failure.
3. Block new risk only when the catalog control is approved for hard enforcement.
4. Escalate to the named owner; do not change the threshold to make the breach pass.
5. Record mitigation, waiver, or resolution evidence.

## Lifecycle

```text
open -> acknowledged -> mitigated -> resolved
                    \-> waived (owner, reason, expiry required)
```

## Triage

| Cause | Response |
| --- | --- |
| Genuine exposure or loss breach | Stop risk-increasing actions and prepare a controlled reduction plan. |
| Missing/stale account or market input | Treat the hard control as unavailable; refresh evidence before reassessment. |
| Symbol, multiplier, FX, or contract error | Correct the package-owned contract and add a regression test. |
| Model coverage failure | Exclude the unsupported result visibly and use an approved conservative fallback if one exists. |
| False alert from unapproved threshold | Return the control to Observe mode and complete approval properly. |

## Evidence

Retain control id, observed value, warning/hard threshold, severity, source
snapshot, calculation version, owner, timestamps, action, and closure reason.
Resolution never deletes the original breach.
