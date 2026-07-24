# Phase 9: Optimisation Objectives

## Purpose

Phase 9 defines what improvement means for each research layer. There is no
universal optimisation score because factor prediction, sleeve economics,
router selection, and portfolio allocation answer different questions.

An objective ranks feasible candidates. A hard constraint is a gate that a
candidate must pass before it can be selected. Phase 8 performs the search;
Phase 9 supplies the versioned objective profile that governs that search.

The registry is `config/research/optimization_objectives.yaml`. Every Phase 8
study freezes both its profile ID and profile fingerprint. Editing a profile
therefore cannot silently change the interpretation of an existing study.

## Common selection rule

All profiles use the same research procedure, but not the same metrics:

1. evaluate candidates on purged, embargoed chronological validation folds;
2. apply objective-specific Bayesian shrinkage to noisy fold means;
3. reject candidates that fail hard constraints;
4. retain the feasible Pareto frontier;
5. select one candidate using a priority order frozen before search;
6. evaluate the frozen candidate once on the untouched holdout.

The priority order is lexicographic, not a fitted weighted score. For example,
the factor profile first compares Rank IC, then Rank ICIR if Rank IC is tied,
then stability, coverage, Pearson IC, and Pearson ICIR. This preserves the
meaning of each metric instead of hiding trade-offs inside one number.

## Factor objective profile

Economic question: does the parameter choice improve predictive strength,
coverage, and stability before portfolio construction?

| Role | Metrics |
| --- | --- |
| Objectives | Mean Pearson IC, mean Rank IC, Pearson ICIR, Rank ICIR, joint signal/return coverage, stability floor |
| Hard gate | Shrunk joint coverage must be at least 50% |
| Frozen upstream component | None; the factor is the first predictive layer |

The stability floor is the weakest predeclared subperiod result returned by the
factor evaluator. A candidate with one excellent period and one poor period is
therefore less attractive than a broad plateau of useful parameters. IC remains
pre-cost predictive evidence; it does not establish trading profitability.

## Sleeve objective profile

Economic question: does one fixed sleeve construction improve executable net
economics while preserving acceptable upstream factor evidence?

| Role | Metrics |
| --- | --- |
| Objectives | Net Sharpe, annualised turnover, maximum drawdown loss, minimum subperiod net Sharpe |
| Hard gates | Upstream factor mean Rank IC greater than zero; upstream factor Rank ICIR greater than zero |
| Frozen upstream component | At least one `fac_*` component |

Maximum drawdown loss is the positive magnitude of drawdown. It is minimised,
so a 15% loss is better than a 30% loss. The factor gate prevents sleeve
optimisation from rescuing a factor that has lost its predictive interpretation.

## Router objective profile

Economic question: does routing improve net performance over the strongest
standalone sleeve using causal relative-advantage evidence?

| Role | Metrics |
| --- | --- |
| Objectives | Net increment over the best standalone sleeve, routing IC, paired HAC t-statistic, switching-cost return, minimum subperiod net increment |
| Hard gates | Shrunk net increment greater than zero; shrunk paired HAC t-statistic at least 1.645 |
| Frozen upstream component | At least two `slv_*` components |

The comparator is the stronger standalone sleeve, not the weaker sleeve and not
zero. Routing IC measures whether the router score predicts the next-period
return difference between the two sleeves. The paired HAC statistic tests the
same-date router increment while allowing for heteroskedasticity and serial
dependence. The 1.645 threshold is a predeclared one-sided 5% evidence gate; it
does not make the result immune to data mining, so subperiod stability and the
single final holdout remain necessary.

Switching-cost return is the return lost specifically because routing changes
allocations. It is minimised separately from the strategy's ordinary execution
costs.

## Allocator objective profile

Economic question: does risk allocation improve the frozen alpha portfolio's
net risk-adjusted return without worsening concentration or tail loss?

| Role | Metrics |
| --- | --- |
| Objectives | Net Sharpe, maximum drawdown loss, concentration HHI, expected shortfall 95% loss |
| Hard gate | None in version 1; Pareto dominance exposes every trade-off |
| Frozen upstream component | At least one `fac_*`, `slv_*`, or `rtr_*` component |

Concentration HHI is the sum of squared portfolio weights. Expected shortfall
95% loss is the positive mean loss inside the worst 5% of observations. Both are
minimised. Upstream alpha must remain frozen so the allocator cannot receive
credit for changing the signal, sleeve, or router.

## Deliberate exclusion

There is no active overlay objective profile. Overlay optimisation remains
blocked until its economic question, primary comparator, tail-risk measure, and
failure gates are specified. The generic optimisation engine can technically
sample overlay parameters, but Phase 9 will reject any Phase 8 overlay study
because no approved profile exists.

## Audit artifacts

Run:

```bash
PYTHONPATH=src:. python scripts/research/audit_optimization_objectives.py
```

The audit writes `readiness.json`, `profiles.csv`, `objectives.csv`,
`constraints.csv`, and `upstream_requirements.csv` under
`runtime/artifacts/research/optimization_objectives/`. The Research Review
dashboard renders those files in the **Optimisation Objectives** tab.
