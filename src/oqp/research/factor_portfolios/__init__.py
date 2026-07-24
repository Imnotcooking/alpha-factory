"""Factor-level portfolio construction for research strategies."""

from oqp.research.factor_portfolios.composer import (
    CompositionResult,
    FactorPortfolioComposer,
)
from oqp.research.factor_portfolios.contracts import (
    FactorPortfolioConfig,
    FactorSpec,
    RouterSpec,
    SleeveSpec,
    StrategyRiskOverlaySpec,
    load_factor_portfolio_config,
)
from oqp.research.factor_portfolios.diagnostics import (
    contribution_summary,
    factor_correlation,
    factor_coverage,
    leave_one_out_summary,
)
from oqp.research.factor_portfolios.data import (
    FactorPortfolioDataBundle,
    load_factor_portfolio_data,
    load_router_state_data,
)
from oqp.research.factor_portfolios.inventory import (
    compatible_factor_inventory,
    compatible_router_inventory,
    compatible_strategy_risk_overlay_inventory,
    factor_inventory,
    router_inventory,
    strategy_risk_overlay_inventory,
)
from oqp.research.factor_portfolios.runner import (
    FactorPortfolioBuildResult,
    FactorPortfolioRunner,
)

__all__ = [
    "CompositionResult",
    "FactorPortfolioComposer",
    "FactorPortfolioConfig",
    "FactorPortfolioBuildResult",
    "FactorPortfolioDataBundle",
    "FactorPortfolioRunner",
    "FactorSpec",
    "RouterSpec",
    "SleeveSpec",
    "StrategyRiskOverlaySpec",
    "contribution_summary",
    "factor_correlation",
    "factor_coverage",
    "factor_inventory",
    "compatible_factor_inventory",
    "compatible_router_inventory",
    "compatible_strategy_risk_overlay_inventory",
    "load_factor_portfolio_data",
    "load_factor_portfolio_config",
    "load_router_state_data",
    "leave_one_out_summary",
    "router_inventory",
    "strategy_risk_overlay_inventory",
]
