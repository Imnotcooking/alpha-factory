"""Contracts for causal strategy-level risk overlays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StrategyRiskOverlayContract:
    """Declared input, output, and timing behavior for one risk overlay."""

    overlay_id: str
    date_col: str = "date"
    ticker_col: str = "ticker"
    price_col: str = "close"
    source_weight_col: str = "final_target_weight"
    output_weight_col: str = "final_target_weight"
    decision_time: str = "daily_close"
    effective_time: str = "next_open"
    scope: str = "portfolio_scalar"
    allow_sign_flip: bool = False
    allow_gross_increase: bool = False
    supported_markets: tuple[str, ...] = ("*",)
    contract_source: str = "declared"

    def __post_init__(self) -> None:
        overlay_id = str(self.overlay_id).strip()
        if not overlay_id:
            raise ValueError("overlay_id cannot be empty")
        markets = tuple(
            str(value).strip()
            for value in self.supported_markets
            if str(value).strip()
        )
        if not markets:
            raise ValueError("supported_markets cannot be empty")
        if self.scope not in {"portfolio_scalar", "position_level"}:
            raise ValueError("scope must be portfolio_scalar or position_level")
        object.__setattr__(self, "overlay_id", overlay_id)
        object.__setattr__(self, "supported_markets", markets)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        overlay_id: str | None = None,
        contract_source: str = "declared",
    ) -> "StrategyRiskOverlayContract":
        return cls(
            overlay_id=str(overlay_id or payload.get("overlay_id") or ""),
            date_col=str(payload.get("date_col", "date")),
            ticker_col=str(payload.get("ticker_col", "ticker")),
            price_col=str(payload.get("price_col", "close")),
            source_weight_col=str(
                payload.get("source_weight_col", "final_target_weight")
            ),
            output_weight_col=str(
                payload.get("output_weight_col", "final_target_weight")
            ),
            decision_time=str(payload.get("decision_time", "daily_close")),
            effective_time=str(payload.get("effective_time", "next_open")),
            scope=str(payload.get("scope", "portfolio_scalar")),
            allow_sign_flip=bool(payload.get("allow_sign_flip", False)),
            allow_gross_increase=bool(
                payload.get("allow_gross_increase", False)
            ),
            supported_markets=tuple(payload.get("supported_markets") or ("*",)),
            contract_source=contract_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_strategy_risk_overlay_contract(
    module: ModuleType,
    *,
    overlay_id: str | None = None,
) -> StrategyRiskOverlayContract:
    """Resolve and validate a private overlay module's declared contract."""

    declared_id = str(
        overlay_id or getattr(module, "OVERLAY_ID", "")
    ).strip()
    module_name = str(
        getattr(module, "__name__", declared_id or type(module).__name__)
    )
    payload = getattr(module, "OVERLAY_CONTRACT", None)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{declared_id or module_name} must declare OVERLAY_CONTRACT"
        )
    return StrategyRiskOverlayContract.from_mapping(
        payload,
        overlay_id=declared_id,
        contract_source=f"{module_name}.OVERLAY_CONTRACT",
    )


__all__ = [
    "StrategyRiskOverlayContract",
    "resolve_strategy_risk_overlay_contract",
]
