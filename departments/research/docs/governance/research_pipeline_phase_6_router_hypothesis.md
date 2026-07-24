# Phase 6: Router Hypothesis

## Purpose

Phase 6 asks whether one causal observable can select between two independently
credible sleeves. It does not ask whether the observable predicts the market.
It predicts which sleeve should have the larger next-period net return.

For sleeves A and B, the frozen target is:

```text
d(t+1) = net_return_A(t+1) - net_return_B(t+1)
```

A positive canonical router score predicts that A will outperform B. A negative
score predicts that B will outperform A. A condition such as high volatility
that is expected to favour B is therefore sign-adjusted before its routing IC is
calculated.

## Admission requirements

An empirical router test is blocked unless all of the following are true:

1. Sleeve A passed the frozen Phase 4 standalone validation and holdout checks.
2. Sleeve B passed the same checks independently.
3. Both sleeves use the same dataset, market, return horizon, capital, and cost
   profile.
4. The router states an economic claim and mechanism before testing.
5. The score formula, orientation, threshold, source fingerprint, and freeze date
   are recorded before the untouched holdout begins.
6. Router optimisation is disabled.

The hypothesis freeze date matters. A pattern discovered after examining a
historical holdout cannot use that same period as untouched confirmation. It may
be recorded prospectively and evaluated only on later observations.

## Frozen hypothesis contract

Every hypothesis records:

- Stable hypothesis and router IDs
- Sleeve A and sleeve B factor/sleeve IDs
- Economic claim and mechanism
- Observable score name and source fingerprint
- Whether a higher raw score favours A or B
- Frozen threshold
- Hypothesis freeze date
- Decision date and next-execution timing
- Static-blend weights and exposure-scaling rule
- HAC lag and confidence level

The engine converts the raw score into:

```text
predicted_relative_advantage_score =
    raw_score - threshold                 if a higher score favours A
    threshold - raw_score                 if a higher score favours B
```

The router selects A when this canonical score is non-negative and B otherwise.
The decision is made after close `t`; executable return begins at the next actual
open.

## Routing evidence

For validation, holdout, and the full sample, the engine reports:

- Pearson routing IC between the canonical score and `net A - net B`
- Spearman routing Rank IC
- Monthly Pearson ICIR and monthly Rank ICIR
- Better-sleeve selection hit rate, excluding exact return ties
- Mean selected-minus-unselected advantage on correct and wrong dates
- Valid dates and months
- Switch count and switch rate
- Router turnover and costs relative to the static blend

ICIR is the mean monthly routing IC divided by its sample standard deviation. It
is not annualised. A high routing IC does not by itself establish value because
the selected sleeve may trade too often, cost too much, or remain inferior to a
simple baseline.

## Required comparators

Every admissible test reconstructs and re-executes these alternatives:

1. Sleeve A at full allocation
2. Sleeve B at full allocation
3. Frozen static blend, initially 50/50
4. Causal exposure-scaled static blend
5. Router
6. Hindsight oracle, shown only as an unattainable upper bound

The exposure baseline applies a frozen annual volatility target to the static
blend using only its previous 60 observations. Scale is capped between zero and
one, so it can de-risk but not add leverage. Its purpose is to ask whether the
router adds directional selection value beyond a simple risk throttle.

All alternatives are built from final combined target positions and translated
again into whole contracts. Instrument fees and 0.5-tick-per-side slippage are
then recomputed. This allows opposing sleeve positions to net before costs and
prevents double charging.

The oracle observes both realised sleeve returns before choosing. It is excluded
from the best-alternative selection, HAC gates, success criterion, and promotion
status.

## Paired monthly HAC test

Daily net returns are compounded inside each calendar month. For each attainable
comparator `j`, the monthly paired increment is:

```text
increment_m(j) = router_return_m - comparator_return_m(j)
```

The report shows the mean increment, annualised increment, positive-month share,
Newey-West standard error, 95% interval, and HAC t-statistic. The frozen success
criterion compares the router with the strongest attainable validation
alternative, not with the oracle.

The current gate requires:

- Positive validation net mean beyond the best alternative
- Validation monthly increment HAC `t >= 1.645`
- Positive validation net Sharpe
- At least 24 validation routing months

Holdout incremental performance is a separate confirmation. Passing Phase 6 is
not production approval; it permits strategy-level review.

## Current library status

The readiness audit on the five Phase 1B Chinese-futures reversal sleeves finds:

```text
Phase 4 sleeves tested: 5
Phase 4 router-eligible sleeves: 0
Eligible sleeve pairs: 0
Frozen hypotheses predating untouched data: 0
Phase 6 empirical status: blocked
```

Therefore no router result has been manufactured from the Phase 5 charts. Shock
age remains a prospective idea, but its current historical observations have
already been inspected and cannot serve as a new untouched holdout.

## Reproducible implementation

```text
src/oqp/research/strategy_routing/evidence.py
scripts/research/audit_router_hypothesis_readiness.py
tests/research/test_router_hypothesis_evidence.py
apps/research_dashboard/views/router_hypothesis_panel.py
```

Current readiness artifacts are written to:

```text
runtime/artifacts/research/router_hypotheses/readiness.json
runtime/artifacts/research/router_hypotheses/sleeves.csv
```

Future admissible evidence bundles belong under:

```text
runtime/artifacts/research/router_hypotheses/evidence/<hypothesis_id>/
```
