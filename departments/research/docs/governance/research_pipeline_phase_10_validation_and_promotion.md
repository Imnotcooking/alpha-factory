# Phase 10: Validation and Promotion

## Purpose

Phase 10 decides whether a frozen research object may advance to the next
operational stage. The lifecycle is:

```text
Discovery -> chronological validation -> frozen holdout
-> paper trading -> production review
```

Passing Phase 10 does not authorize live trading. A research router can become
eligible for paper trading; satisfactory paper evidence can make it eligible
for human production review. Production approval remains outside the research
engine.

## What Phase 10 consumes

The router review consumes the frozen Phase 6 evidence bundle and, when Phase 8
was used, the frozen optimisation-candidate fingerprint. It does not search for
a better router. Parameter perturbations are evaluated on chronological
validation only; the selected router's holdout remains a single frozen test.

The review configuration freezes:

- router ID and router-configuration fingerprint;
- complete source-evidence fingerprint;
- promotion-policy ID;
- perturbation-plan fingerprint;
- optional frozen optimisation-candidate fingerprint;
- review date.

The evidence fingerprint covers the Phase 6 manifest, summary, decisions,
executed positions, daily comparisons, routing diagnostics, monthly paired
tests, and subperiod results.

## Promotion gates

The versioned policy is `config/research/promotion_policy.yaml`.

### Incremental economics

The comparator is selected once from the attainable alternatives in the
chronological validation sample. It can be sleeve A, sleeve B, the static blend,
or the exposure-scaled blend. The oracle is never eligible.

The same frozen comparator is used in holdout. Both validation and holdout mean
monthly net increments must be positive. Phase 6's predeclared validation
criterion, including its paired HAC test, must also pass. Full-sample Sharpe is
recorded nowhere in the promotion decision.

### Month concentration

For month (m), the net increment is

\[
d_m = r^{net}_{router,m} - r^{net}_{baseline,m}.
\]

For positive months, the contribution share is

\[
s_m = \frac{\max(d_m,0)}{\sum_j \max(d_j,0)}.
\]

The largest validation-month share may not exceed 35%. Validation mean
increment must also remain positive after removing the best month. These two
checks prevent one isolated event from carrying the promotion result.

### Product concentration

Executed net contributions are compared product by product between the router
and frozen baseline. The largest positive product share may not exceed 35%, and
total validation increment must remain positive after removing the best
product. Contribution is additive return attribution, not a second backtest.

### Validation-period sign

At least two chronological validation subperiods are required. Version 1
requires the router-minus-baseline annualised mean to remain positive in every
observed validation subperiod. Holdout is not included in this parameter
selection check; it has its own frozen increment gate.

### Routing events

At least 12 validation switches and at least 10 selections of each sleeve are
required. A router that almost always selects one sleeve has not generated
enough routing evidence, even when its headline return is high.

### Parameter perturbations

At least four reasonable, predeclared neighbouring parameter configurations are
required. At least 75% must retain a positive validation net increment, and all
results must be reproducible. The perturbation plan is a local stability check,
not another optimisation search. It must not evaluate alternative parameters on
the frozen holdout.

### Switching value and cost

Validation gross selection benefit and incremental cost are calculated against
the frozen baseline:

\[
B = \operatorname{AnnMean}(r^{gross}_{router}-r^{gross}_{baseline}),
\]

\[
C = \operatorname{AnnMean}(cost_{router}-cost_{baseline}).
\]

Gross benefit must be positive and (B/C \ge 1) when incremental cost is
positive. If the router has no incremental cost, positive gross benefit passes
this ratio gate. The separate net-increment gates ensure that the after-cost
result is also positive.

## Decision meanings

| Decision | Meaning |
| --- | --- |
| `eligible_for_paper_trading` | Frozen validation and holdout evidence passed; production is not authorized |
| `eligible_for_production_review` | Frozen research and paper-trading gates passed; human review is required |
| `hold_for_more_evidence` | Sample size, events, or perturbations are insufficient |
| `blocked_governance` | Fingerprint, causal lineage, or reproducibility failed |
| `failed_research_result` | Evidence was sufficient, but an economic or robustness gate failed |

The order matters. Insufficient observations do not become an economic failure,
and a reproducibility defect does not become a claim about the market. Genuine
negative economic results remain in the permanent promotion ledger.

## Paper trading

Production-review eligibility requires at least 60 paper observations, at least
four switches, positive net increment over the frozen comparator, two
reproducible paper snapshots, and realised costs no more than 1.5 times the
modeled cost. These are minimum review gates, not guarantees of live success.

## Artifacts

Each review writes:

- `summary.json` and `manifest.json`;
- `gate_results.csv`;
- `month_concentration.csv` and `product_concentration.csv`;
- `validation_periods.csv`;
- `perturbations.csv`.

Run the global audit with:

```bash
PYTHONPATH=src:. python scripts/research/audit_validation_promotion.py
```

It writes the readiness summary, frozen policy table, and permanent promotion
ledger under `runtime/artifacts/research/validation_promotion/`. These are shown
in the **Promotion** tab of Research Review.
