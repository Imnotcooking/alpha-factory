"""Research execution trade-threshold policy."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


DEFAULT_MIN_TRADE_WEIGHT_DELTA = 1e-8


@dataclass(frozen=True)
class ExecutionTradePolicy:
    min_trade_weight_delta: float
    source: str

    def to_attrs(self) -> dict[str, float | str]:
        return {
            "min_trade_weight_delta": float(self.min_trade_weight_delta),
            "min_trade_weight_delta_source": self.source,
        }


def resolve_execution_trade_policy(
    *,
    factor_module: ModuleType | None = None,
    min_trade_weight_delta: float | None = None,
) -> ExecutionTradePolicy:
    if min_trade_weight_delta is not None:
        return ExecutionTradePolicy(
            min_trade_weight_delta=_nonnegative_float(min_trade_weight_delta),
            source="cli_min_trade_weight_delta",
        )

    factor_contract = _factor_contract(factor_module)
    contract_value = factor_contract.get("min_trade_weight_delta")
    if contract_value not in (None, ""):
        return ExecutionTradePolicy(
            min_trade_weight_delta=_nonnegative_float(contract_value),
            source="factor_contract_min_trade_weight_delta",
        )

    return ExecutionTradePolicy(
        min_trade_weight_delta=DEFAULT_MIN_TRADE_WEIGHT_DELTA,
        source="execution_default_dust_tolerance",
    )


def attach_trade_policy_attrs(df, policy: ExecutionTradePolicy):
    for key, value in policy.to_attrs().items():
        df.attrs[key] = value
    return df


def _factor_contract(factor_module: ModuleType | None) -> dict[str, Any]:
    if factor_module is None:
        return {}
    raw = getattr(factor_module, "FACTOR_CONTRACT", {}) or {}
    return raw if isinstance(raw, dict) else {}


def _nonnegative_float(value: Any) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError("min_trade_weight_delta must be non-negative.")
    return parsed


__all__ = [
    "DEFAULT_MIN_TRADE_WEIGHT_DELTA",
    "ExecutionTradePolicy",
    "attach_trade_policy_attrs",
    "resolve_execution_trade_policy",
]
