# Risk Appetite And Control Authority

Last reviewed: 2026-07-17

This document defines how risk constraints become authoritative. It deliberately
contains no invented portfolio thresholds. Numerical limits require explicit
approval before any control can warn or block.

## Objectives

Risk appetite exists to prevent four different failures:

1. Insolvency or margin failure from leverage, concentration, or nonlinear loss.
2. Unintended factor, sector, currency, volatility, or liquidity concentration.
3. Trading from stale, unreconciled, or model-invalid inputs.
4. A research allocation becoming a live risk decision without independent
   review.

## Hierarchy

Controls are evaluated from broadest to narrowest:

```text
portfolio -> account -> strategy -> asset class -> instrument -> order
```

A lower-level limit cannot override a stricter higher-level limit. Account and
portfolio controls use consolidated reporting currency, while native-currency
evidence remains available for investigation.

## Control Modes

| Mode | Meaning | Runtime effect |
| --- | --- | --- |
| Observe | Metric is calculated and displayed without a threshold. | No warning or block. |
| Warn | Approved warning threshold exists. | Alert and require review; no automatic order block. |
| Block | Approved hard threshold exists. | Execution must reject the action or prevent new risk. |

`limit_catalog.yaml` begins in Observe mode. A control moves to Warn or Block
only after its source, units, denominator, missing-data behavior, owner, and
threshold are approved and tested.

## Control Inputs

Hard controls require:

- a fresh canonical account snapshot;
- no unresolved critical account reconciliation break affecting the metric;
- approved marks, FX rates, multipliers, and option contract identity;
- deterministic calculation version and timestamp;
- explicit treatment of missing or stale inputs.

Unavailable data is not equivalent to zero risk. A hard control with unavailable
inputs returns unavailable/block according to its enforcement policy; it cannot
silently pass.

## Authority

- **Middle Office** certifies account and reconciliation evidence.
- **Data Platform** certifies data contracts and freshness evidence.
- **Risk** defines metrics, scenarios, thresholds, and breach severity.
- **Trading/Execution** enforces approved hard blocks.
- **Research** proposes allocations but cannot waive limits.
- **Human owner** approves temporary waivers with reason and expiry.

## Approval Record

Every Warn or Block promotion must record:

- control id and version;
- scope and reporting currency;
- warning and/or hard threshold;
- calculation source and data dependencies;
- approving owner and approval timestamp;
- effective date, review date, and rollback plan.

Until this record exists, the control remains observational regardless of what
a dashboard label or environment variable suggests.
