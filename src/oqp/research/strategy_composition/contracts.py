"""Immutable contracts for Phase 7 strategy composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


STRATEGY_COMPOSITION_SCHEMA_VERSION = 1


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StrategyAllocatorConfig:
    max_gross_leverage: float | None = 1.0
    max_contract_weight: float = 0.10
    max_margin_utilization: float | None = None

    def __post_init__(self) -> None:
        max_gross = self.max_gross_leverage
        if max_gross is not None and float(max_gross) <= 0.0:
            raise ValueError("max_gross_leverage must be positive or null")
        max_contract = float(self.max_contract_weight)
        if not 0.0 < max_contract <= 1.0:
            raise ValueError("max_contract_weight must be between zero and one")
        if max_gross is not None and max_contract > float(max_gross):
            raise ValueError(
                "max_contract_weight cannot exceed the configured gross leverage"
            )
        max_margin = self.max_margin_utilization
        if max_margin is not None and not 0.0 < float(max_margin) <= 1.0:
            raise ValueError("max_margin_utilization must be between zero and one")
        object.__setattr__(
            self,
            "max_gross_leverage",
            None if max_gross is None else float(max_gross),
        )
        object.__setattr__(
            self, "max_contract_weight", max_contract
        )
        object.__setattr__(
            self,
            "max_margin_utilization",
            None if max_margin is None else float(max_margin),
        )


@dataclass(frozen=True, slots=True)
class StrategyExecutionConfig:
    capital: float = 10_000_000.0
    capital_currency: str = "CNY"
    transaction_cost_profile: str = "cn_futures_broker_v1"
    slippage_ticks_per_side: float = 0.5

    def __post_init__(self) -> None:
        if float(self.capital) <= 0.0:
            raise ValueError("capital must be positive")
        if float(self.slippage_ticks_per_side) < 0.0:
            raise ValueError("slippage_ticks_per_side cannot be negative")
        for field in ("capital_currency", "transaction_cost_profile"):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} cannot be empty")
            object.__setattr__(self, field, value)
        object.__setattr__(self, "capital", float(self.capital))
        object.__setattr__(
            self,
            "slippage_ticks_per_side",
            float(self.slippage_ticks_per_side),
        )


@dataclass(frozen=True, slots=True)
class StrategyCompositionConfig:
    strategy_id: str
    name: str
    market_vertical: str
    sleeves: tuple[str, ...]
    router: str | None
    risk_overlays: tuple[str, ...] = ()
    allocator: StrategyAllocatorConfig = StrategyAllocatorConfig()
    execution: StrategyExecutionConfig = StrategyExecutionConfig()
    components_immutable: bool = True
    optimization_permitted: bool = False
    schema_version: int = STRATEGY_COMPOSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in ("strategy_id", "name", "market_vertical"):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} cannot be empty")
            object.__setattr__(self, field, value)
        sleeves = tuple(str(value).strip() for value in self.sleeves)
        overlays = tuple(str(value).strip() for value in self.risk_overlays)
        router = str(self.router).strip() if self.router is not None else None
        if not sleeves:
            raise ValueError("a strategy requires at least one sleeve")
        if any(not value.startswith("slv_") for value in sleeves):
            raise ValueError("Phase 7 sleeve references must use stable slv_* IDs")
        if len(set(sleeves)) != len(sleeves):
            raise ValueError("Phase 7 sleeve references must be unique")
        if any(not value.startswith("ovl_") for value in overlays):
            raise ValueError("risk overlay references must use stable ovl_* IDs")
        if len(set(overlays)) != len(overlays):
            raise ValueError("risk overlay references must be unique")
        if len(sleeves) > 1 and not router:
            raise ValueError("multiple sleeves require one frozen router")
        if router and not router.startswith("rtr_"):
            raise ValueError("router references must use a stable rtr_* ID")
        if router and len(sleeves) < 2:
            raise ValueError("a router requires at least two sleeves")
        if not bool(self.components_immutable):
            raise ValueError("Phase 7 components must be immutable")
        if bool(self.optimization_permitted):
            raise ValueError("Phase 7 composition cannot optimize components")
        object.__setattr__(self, "sleeves", sleeves)
        object.__setattr__(self, "risk_overlays", overlays)
        object.__setattr__(self, "router", router)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrategyCompositionConfig":
        raw = payload.get("strategy", payload)
        if not isinstance(raw, Mapping):
            raise ValueError("strategy configuration must be a mapping")
        allowed = {
            "strategy_id",
            "name",
            "market_vertical",
            "sleeves",
            "router",
            "risk_overlays",
            "allocator",
            "execution",
            "components_immutable",
            "optimization_permitted",
            "schema_version",
        }
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise ValueError(
                "Phase 7 recipes may only reference frozen components; "
                f"unsupported fields: {unknown}"
            )
        sleeves = raw.get("sleeves") or ()
        overlays = raw.get("risk_overlays") or ()
        if not isinstance(sleeves, (list, tuple)) or any(
            not isinstance(value, str) for value in sleeves
        ):
            raise ValueError("sleeves must be a list of stable slv_* references")
        if not isinstance(overlays, (list, tuple)) or any(
            not isinstance(value, str) for value in overlays
        ):
            raise ValueError("risk_overlays must be a list of stable ovl_* references")
        allocator = raw.get("allocator") or {}
        execution = raw.get("execution") or {}
        if not isinstance(allocator, Mapping) or not isinstance(execution, Mapping):
            raise ValueError("allocator and execution must be mappings")
        return cls(
            strategy_id=str(raw.get("strategy_id") or ""),
            name=str(raw.get("name") or raw.get("strategy_id") or ""),
            market_vertical=str(raw.get("market_vertical") or ""),
            sleeves=tuple(sleeves),
            router=raw.get("router"),
            risk_overlays=tuple(overlays),
            allocator=StrategyAllocatorConfig(**dict(allocator)),
            execution=StrategyExecutionConfig(**dict(execution)),
            components_immutable=bool(raw.get("components_immutable", True)),
            optimization_permitted=bool(raw.get("optimization_permitted", False)),
            schema_version=int(
                raw.get("schema_version", STRATEGY_COMPOSITION_SCHEMA_VERSION)
            ),
        )


def load_strategy_composition_config(
    path: str | Path,
) -> StrategyCompositionConfig:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("strategy composition YAML must contain a mapping")
    return StrategyCompositionConfig.from_mapping(payload)


__all__ = [
    "STRATEGY_COMPOSITION_SCHEMA_VERSION",
    "StrategyAllocatorConfig",
    "StrategyCompositionConfig",
    "StrategyExecutionConfig",
    "load_strategy_composition_config",
]
