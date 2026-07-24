"""Reusable statistical-arbitrage mathematics."""

from oqp.research.statistical_arbitrage.spreads import (
    SPREAD_CONTRACT_VALUE,
    SPREAD_LINEAR_PRICE,
    SPREAD_PRICE_RATIO,
    SPREAD_RETURN_RESIDUAL,
    SpreadModelConfig,
    build_price_matrix,
    construct_pair_spread,
    contract_multipliers_for_pair,
    estimate_half_life,
    estimate_ols_beta,
    log_return_matrix,
    rolling_zscore,
)

__all__ = [
    "SPREAD_CONTRACT_VALUE",
    "SPREAD_LINEAR_PRICE",
    "SPREAD_PRICE_RATIO",
    "SPREAD_RETURN_RESIDUAL",
    "SpreadModelConfig",
    "build_price_matrix",
    "construct_pair_spread",
    "contract_multipliers_for_pair",
    "estimate_half_life",
    "estimate_ols_beta",
    "log_return_matrix",
    "rolling_zscore",
]
