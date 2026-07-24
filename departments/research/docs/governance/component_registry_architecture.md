# Research Component Registry Architecture

The research library separates economic signals, routing decisions, and final
strategy recipes so each layer can be tested without silently changing another.

## Ownership Boundary

| Layer | Registry | Stable ID | Responsibility |
|---|---|---|---|
| Factor | `departments/research/factors/` | `fac_*` | Produce a causal alpha score. |
| Sleeve | `departments/research/strategies/sleeves/` | `slv_*` | Convert one frozen factor score into target positions. |
| Router | `departments/research/routers/` | `rtr_*` | Allocate among frozen sleeves from causal states. |
| Risk overlay | `departments/research/strategy_overlays/` | `ovl_*` | Scale or gate a finished strategy target without creating alpha direction. |
| Strategy | `departments/research/strategies/` | `str_*` | Declare factors, sleeves, routers, overlays, execution, and risk constraints. |
| Shared engine | `src/oqp/research/` | Python package | Validate contracts and execute recipes without private alpha definitions. |

A factor must not import a router. The strategy runner constructs every sleeve
independently, then loads the declared router and applies allocations to the
finished sleeve targets. Turnover and costs are measured once, from the final
routed target positions.

A strategy risk overlay is applied only after the factor portfolio or routed
target exists. It may react causally to the strategy's realized path, but it
may not masquerade as a factor, select sleeves, flip position direction, or
increase gross unless its contract explicitly permits that behavior.

## Cleanup Gates

1. **Static audit:** classify every file, detect naming and metadata gaps, and
   record dependencies without importing factor modules.
2. **Mechanical migration:** replace `cnf_*` names only after the component is
   classified. Duplicate wrappers are retired rather than assigned new factor
   IDs.
3. **Empirical deduplication:** compute comparable factor outputs on one frozen
   dataset, cluster high absolute correlations, and retain candidates using
   predeclared IC, net Sharpe, turnover, stability, and data-quality gates.
4. **Strategy validation:** rerun affected strategy recipes and confirm that
   sleeve targets, router states, final turnover, and costs reconcile.

The static audit is generated with:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/research/audit_component_registry.py
```

Outputs are written to
`runtime/artifacts/research/component_registry_audit/`. The report recommends
dispositions; it does not promote or delete factors using filename similarity.

## Current Migration Decisions

- `cnf_monthly_ema_trend.py` and `cnf_monthly_macd_crossover.py` are legacy
  sleeve adapters around existing factor families. Their dependencies move to
  `fac_064` and `fac_065`; the wrappers are retired afterward.
- Volatility, breadth, positioning, shock-stage, and product-state modules are
  router inputs, not investable factors. They move with router support logic.
- Methodology audit and frozen paper-replication helpers remain reproducibility
  assets for their owner experiments; they are not active factor candidates.
- Intraday opening-gap reversal and breadth-event trend remain candidate alpha
  sleeves until the empirical deduplication gate compares them with the current
  `fac_*` library.
- The former `fac_095` prototype was removed from active factor discovery. Its
  state estimator and allocation rule now live in
  `rtr_004_Intraday_Volatility_Quartile`; the original combined implementation
  is retained privately under a non-factor archive name.
- The former `fac_079` and `fac_080` hybrids were removed from active factor
  discovery. `fac_068` remains their canonical KER alpha; the drawdown/churn
  and drawdown-only rules now live under stable `ovl_001` and `ovl_002` IDs and
  are declared by strategy recipes.

## Gate To Empirical Deduplication

The first static audit found 50 incomplete factor/router metadata contracts, 22
declared IDs that do not match their filenames, and three duplicated numeric
factor IDs. Correlation and IC pruning must wait until each comparison family
has a common market, frequency, forward-return horizon, execution lag, and cost
basis. Raw score correlation across incompatible geometries is not a valid
deletion rule.
