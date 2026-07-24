# Private Router Registry

This folder stores reproducible private routing recipes. A router observes a
causal state and allocates capital across already-defined strategy sleeves. It
may not alter the signal or holdings logic inside a sleeve.

Router files use stable `rtr_NNN_*` IDs and declare `ROUTER_METADATA` plus
`ROUTER_CONTRACT` and `ROUTER_PARAMETERS`. The parameter schema is validated on
load and may expose only the router's own causal threshold or switching rules.
Generic validation and routed-position construction live in
`src/oqp/research/strategy_routing/`; final strategy recipes reference routers
from `departments/research/strategies/`.

Costs are calculated from final routed positions. Sleeve-level hypothetical
costs may be reported diagnostically but must not be added again to routed
turnover.

## Phase 6 evidence contract

A router recipe is not eligible for empirical testing until it states a dated
economic claim about the next-period **relative net return** of two independently
validated sleeves. Its score must target `net_return_A - net_return_B`, not the
market return. The threshold, score orientation, observable-source fingerprint,
and hypothesis freeze date are fixed before the untouched holdout begins.

Every test compares the router with sleeve A, sleeve B, a static blend, and a
causal exposure-scaled blend. Final combined targets are re-executed before fees
and slippage are calculated. A hindsight oracle may be displayed only as an
unattainable upper bound and is excluded from selection, gates, and promotion.
