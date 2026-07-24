"""Contracts for reproducible causal strategy routers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RouterContract:
    """Declared input, output, and timing behavior for one router recipe."""

    router_id: str
    state_col: str = "state"
    decision_date_col: str = "decision_date"
    effective_date_col: str = "effective_date"
    sleeve_col: str = "sleeve_id"
    allocation_col: str = "allocation"
    decision_lag_periods: int = 1
    allow_partial_allocation: bool = False
    allow_negative_allocation: bool = False
    supported_markets: tuple[str, ...] = ("*",)
    contract_source: str = "declared"

    def __post_init__(self) -> None:
        router_id = str(self.router_id).strip()
        if not router_id:
            raise ValueError("router_id cannot be empty")
        if int(self.decision_lag_periods) < 0:
            raise ValueError("decision_lag_periods cannot be negative")
        markets = tuple(str(value).strip() for value in self.supported_markets if str(value).strip())
        if not markets:
            raise ValueError("supported_markets cannot be empty")
        object.__setattr__(self, "router_id", router_id)
        object.__setattr__(self, "decision_lag_periods", int(self.decision_lag_periods))
        object.__setattr__(self, "supported_markets", markets)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        router_id: str | None = None,
        contract_source: str = "declared",
    ) -> "RouterContract":
        return cls(
            router_id=str(router_id or payload.get("router_id") or ""),
            state_col=str(payload.get("state_col", "state")),
            decision_date_col=str(payload.get("decision_date_col", "decision_date")),
            effective_date_col=str(payload.get("effective_date_col", "effective_date")),
            sleeve_col=str(payload.get("sleeve_col", "sleeve_id")),
            allocation_col=str(payload.get("allocation_col", "allocation")),
            decision_lag_periods=int(payload.get("decision_lag_periods", 1)),
            allow_partial_allocation=bool(payload.get("allow_partial_allocation", False)),
            allow_negative_allocation=bool(payload.get("allow_negative_allocation", False)),
            supported_markets=tuple(payload.get("supported_markets") or ("*",)),
            contract_source=contract_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_router_contract(module: ModuleType, *, router_id: str | None = None) -> RouterContract:
    """Resolve and validate a router module's declared contract."""

    declared_id = str(router_id or getattr(module, "ROUTER_ID", "")).strip()
    module_name = str(getattr(module, "__name__", declared_id or type(module).__name__))
    payload = getattr(module, "ROUTER_CONTRACT", None)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{declared_id or module_name} must declare ROUTER_CONTRACT")
    return RouterContract.from_mapping(
        payload,
        router_id=declared_id,
        contract_source=f"{module_name}.ROUTER_CONTRACT",
    )


__all__ = ["RouterContract", "resolve_router_contract"]
