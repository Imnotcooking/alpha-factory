# Research Strategy Recipes

This folder stores lightweight, reproducible strategy definitions. A strategy
recipe references factor IDs and portfolio policies; reusable composition and
backtesting logic stays in `src/oqp/research/factor_portfolios/`.

Exploratory factor-portfolio recipes may still declare factor and construction
parameters while those components are being studied. A final Phase 7 strategy
is stricter: it references frozen `slv_*`, `rtr_*`, and `ovl_*` components and
cannot change their parameters.

## Factor Portfolio Boundary

A factor emits a score. A factor portfolio aligns and normalizes several
scores, applies declared factor weights, then hands one `composite_score` to
the shared execution and backtesting engine.

A routed strategy first builds each sleeve independently, then applies a
router from `departments/research/routers/` to the frozen sleeve target
positions. Routers allocate capital; they do not rewrite factor signals.

A strategy risk overlay from
`departments/research/strategy_overlays/` may then reduce the finished target
using information available at the declared decision time. An overlay does not
create alpha direction and does not choose among sleeves. This keeps three
questions separate: what to own, which sleeve receives capital, and how much
risk the completed strategy should carry.

Do not place Python engines, market data, fitted weights, backtest outputs, or
private vendor extracts here. Generated evidence belongs under
`runtime/artifacts/research/`.

Recipes whose component factors fail the common empirical gate move to
`departments/research/archive/strategy_prototypes/`; they are not offered as
active assemblies even when their YAML remains reproducible.

## Run An Example

```bash
PYTHONPATH=src:. .venv/bin/python scripts/research/run_factor_portfolio.py \
  --config departments/research/strategies/examples/cn_futures_value_momentum.yaml \
  --data-file runtime/data/futures_cn/daily/YOUR_LONG_FORMAT_FILE.parquet \
  --build-only
```

Remove `--build-only` after contract and coverage checks pass to write a full
research run through the existing evaluator.

A routed recipe declares named sleeves and a stable router ID:

```yaml
sleeves:
  - sleeve_id: slow_trend
    factors:
      - factor_id: fac_064_Classic_TS_EMA_Trend
  - sleeve_id: macd_trend
    factors:
      - factor_id: fac_065_MACD_Crossover_Futures_CN
router:
  router_id: rtr_001_Volatility_Quartile
  state_file: runtime/artifacts/research/router_states.parquet
  parameters:
    assignments:
      Q1: slow_trend
      Q2: slow_trend
      Q3: slow_trend
      Q4: macd_trend
```

The state file must contain the decision date, effective date, and state column
declared by the router contract. A monthly router normally maps information
available in month `t` to allocations effective in month `t+1`.

## Phase 7 Composition Boundary

A final composition lives under `compositions/` only after its sleeves have
passed Phase 4 and its router has passed Phase 6 validation and holdout
confirmation. Its shape is:

```yaml
strategy:
  strategy_id: str_cn_futures_router
  name: CN futures routed strategy
  market_vertical: FUTURES_CN
  core:
    type: routed_components
    branches:
      - branch_id: trend
        factor_ids: [fac_trend]
        sleeve_id: slv_trend
        execution_mode: risk_desk
      - branch_id: reversal
        factor_ids: [fac_reversal]
        sleeve_id: slv_reversal
        execution_mode: risk_desk
    router_id: rtr_volatility_threshold
  risk_overlays:
    - ovl_drawdown_brake
  allocator:
    max_gross_leverage: null
    max_contract_weight: 0.10
    max_margin_utilization: 0.30
  execution:
    capital: 10000000
    capital_currency: CNY
    transaction_cost_profile: cn_futures_broker_v1
    slippage_ticks_per_side: 0.5
```

The engine applies components in one fixed order: core targets, optional router
allocations, risk overlays, allocator limits, final execution, then transaction
costs. Component-level hypothetical costs are ignored. Fees and slippage are
calculated once from the final executed positions after branches have netted.
For Chinese futures, `max_margin_utilization: 0.30` caps configured margin use
at 30% of capital and implies a minimum 70% cash reserve.

## Reproducibility Rules

1. Use stable factor IDs rather than importing recipe files by path.
2. Declare the market vertical, normalization, missing-data policy, weighting
   method, and execution constraints.
3. Start with equal or static weights. Adaptive weights must be estimated only
   from information available before each rebalance date.
4. Blend factors only when their market, frequency, return horizon, evaluation
   geometry, execution lag, and return assumption are compatible.
5. Treat factors with different universes or execution horizons as separate
   sleeves and allocate capital between them instead of merging raw scores.
6. During exploratory factor-portfolio work, pin non-default parameters in that
   research recipe. Before Phase 7, freeze the resulting sleeve under a stable
   `slv_*` ID; the final composition cannot override it.
7. Apply path-dependent drawdown or turnover controls through a stable `ovl_*`
   recipe after final target construction, so their incremental effect can be
   tested against the unchanged base strategy.
