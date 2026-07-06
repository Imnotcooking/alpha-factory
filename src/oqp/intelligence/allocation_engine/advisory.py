"""Allocation advisory engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from oqp.intelligence.allocation_engine.base import BaseAllocationEngine
from oqp.intelligence.allocation_engine.constraints import apply_weight_constraints
from oqp.intelligence.allocation_engine.hrp import hrp_weights
from oqp.intelligence.allocation_engine.kelly import kelly_weights
from oqp.intelligence.allocation_engine.vol_target import (
    portfolio_volatility,
    scale_to_vol_target,
)
from oqp.intelligence.base import EngineStatus
from oqp.intelligence.context import EngineContext


@dataclass(frozen=True, slots=True)
class AllocationAdvisoryConfig:
    kelly_fraction: float = 0.25
    max_abs_weight: float = 0.20
    max_gross: float = 1.0
    target_volatility: float = 0.10
    max_leverage: float = 1.0


class AllocationAdvisoryEngine(BaseAllocationEngine):
    """Blend HRP, fractional Kelly, and volatility targeting when inputs exist."""

    engine_id = "allocation_advisory"
    engine_name = "Allocation Advisory"
    version = "0.1.0"

    def __init__(self, config: AllocationAdvisoryConfig | None = None) -> None:
        self.config = config or AllocationAdvisoryConfig()

    def run(self, context: EngineContext):
        returns = context.metadata.get("allocation_returns")
        expected_returns = context.metadata.get("expected_returns")
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            requirements = pd.DataFrame(
                [
                    {
                        "Input": "allocation_returns",
                        "Status": "missing",
                        "Detail": "Wide asset return matrix is needed for HRP and vol targeting.",
                    },
                    {
                        "Input": "expected_returns",
                        "Status": "optional",
                        "Detail": "Expected returns enable Kelly sizing; historical mean is fallback.",
                    },
                ]
            )
            return self.result(
                status=EngineStatus.SKIPPED,
                summary="Allocation advisory is waiting for research returns/signals.",
                frames={"requirements": requirements},
                metrics={"assets": 0},
            )

        clean_returns = returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        expected = _expected_returns(expected_returns, clean_returns)
        cov = clean_returns.cov()
        hrp = hrp_weights(clean_returns)
        kelly = kelly_weights(
            expected,
            cov,
            kelly_fraction=self.config.kelly_fraction,
            max_abs_weight=self.config.max_abs_weight,
            max_gross=self.config.max_gross,
        )
        blended = (hrp.reindex(expected.index).fillna(0.0) + kelly.reindex(expected.index).fillna(0.0)) / 2.0
        constrained = apply_weight_constraints(
            blended,
            max_abs_weight=self.config.max_abs_weight,
            max_gross=self.config.max_gross,
        )
        target = scale_to_vol_target(
            constrained,
            cov,
            target_volatility=self.config.target_volatility,
            max_leverage=self.config.max_leverage,
        )
        allocation = pd.DataFrame(
            {
                "Asset": expected.index,
                "Expected Return": expected.values,
                "HRP Weight": hrp.reindex(expected.index).fillna(0.0).values,
                "Kelly Weight": kelly.reindex(expected.index).fillna(0.0).values,
                "Constrained Weight": constrained.reindex(expected.index).fillna(0.0).values,
                "Vol Target Weight": target.reindex(expected.index).fillna(0.0).values,
            }
        )
        realized_vol = portfolio_volatility(target, cov)
        return self.result(
            status=EngineStatus.PASS,
            summary="Allocation advisory produced HRP/Kelly/vol-target weights.",
            frames={"allocation": allocation},
            metrics={
                "assets": int(len(allocation)),
                "target_gross": float(target.abs().sum()),
                "estimated_volatility": realized_vol,
            },
            signals={"target_weights": target.to_dict()},
            metadata={"config": asdict(self.config)},
        )


def _expected_returns(value: Any, returns: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").reindex(returns.columns).fillna(0.0)
    if isinstance(value, dict):
        return pd.Series(value, index=returns.columns, dtype=float).fillna(0.0)
    return returns.mean().reindex(returns.columns).fillna(0.0)
