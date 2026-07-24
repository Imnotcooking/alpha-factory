"""Reusable contracts and execution for strategy-level risk overlays."""

from oqp.research.strategy_risk_overlays.contracts import (
    StrategyRiskOverlayContract,
    resolve_strategy_risk_overlay_contract,
)
from oqp.research.strategy_risk_overlays.engine import (
    StrategyRiskOverlayResult,
    apply_strategy_risk_overlay,
)
from oqp.research.strategy_risk_overlays.registry import (
    PRIVATE_STRATEGY_RISK_OVERLAY_ROOT,
    iter_strategy_risk_overlay_files,
    load_strategy_risk_overlay_module,
    resolve_strategy_risk_overlay_path,
)

__all__ = [
    "PRIVATE_STRATEGY_RISK_OVERLAY_ROOT",
    "StrategyRiskOverlayContract",
    "StrategyRiskOverlayResult",
    "apply_strategy_risk_overlay",
    "iter_strategy_risk_overlay_files",
    "load_strategy_risk_overlay_module",
    "resolve_strategy_risk_overlay_contract",
    "resolve_strategy_risk_overlay_path",
]
