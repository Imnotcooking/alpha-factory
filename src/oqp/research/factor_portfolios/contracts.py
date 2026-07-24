"""Configuration contracts for reproducible multi-factor strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical


VALID_NORMALIZATIONS = {
    "cross_sectional_zscore",
    "cross_sectional_rank",
    "raw",
}
VALID_WEIGHTING_METHODS = {"equal", "static"}
VALID_MISSING_POLICIES = {"renormalize_available", "zero", "complete_case"}


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """One factor's role inside a composite signal."""

    factor_id: str
    weight: float = 1.0
    orientation: int = 1
    signal_col: str | None = None
    normalization: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factor_id = str(self.factor_id).strip()
        if not factor_id:
            raise ValueError("factor_id cannot be empty")
        if float(self.weight) < 0:
            raise ValueError(f"{factor_id} weight must be non-negative")
        if int(self.orientation) not in {-1, 1}:
            raise ValueError(f"{factor_id} orientation must be 1 or -1")
        if self.normalization not in (None, *VALID_NORMALIZATIONS):
            raise ValueError(
                f"{factor_id} normalization must be one of {sorted(VALID_NORMALIZATIONS)}"
            )
        if not isinstance(self.parameters, Mapping):
            raise ValueError(f"{factor_id} parameters must be a mapping")
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "orientation", int(self.orientation))
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FactorSpec":
        return cls(
            factor_id=str(payload.get("factor_id") or payload.get("id") or ""),
            weight=float(payload.get("weight", 1.0)),
            orientation=int(payload.get("orientation", 1)),
            signal_col=_optional_text(payload.get("signal_col")),
            normalization=_optional_text(payload.get("normalization")),
            parameters=payload.get("parameters") or {},
        )


@dataclass(frozen=True, slots=True)
class SleeveSpec:
    """One independently constructed factor sleeve available to a router."""

    sleeve_id: str
    factors: tuple[FactorSpec, ...]
    weighting_method: str = "equal"
    normalization: str = "cross_sectional_zscore"
    missing_policy: str = "renormalize_available"
    min_available_factors: int = 1
    winsor_limit: float | None = 3.0
    description: str = ""

    def __post_init__(self) -> None:
        sleeve_id = str(self.sleeve_id).strip()
        if not sleeve_id:
            raise ValueError("sleeve_id cannot be empty")
        factors = tuple(self.factors)
        if not factors:
            raise ValueError(f"{sleeve_id} must contain at least one factor")
        factor_ids = [spec.factor_id for spec in factors]
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError(f"{sleeve_id} factor IDs must be unique")
        if self.weighting_method not in VALID_WEIGHTING_METHODS:
            raise ValueError(
                f"{sleeve_id} weighting_method must be one of "
                f"{sorted(VALID_WEIGHTING_METHODS)}"
            )
        if self.normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"{sleeve_id} normalization must be one of "
                f"{sorted(VALID_NORMALIZATIONS)}"
            )
        if self.missing_policy not in VALID_MISSING_POLICIES:
            raise ValueError(
                f"{sleeve_id} missing_policy must be one of "
                f"{sorted(VALID_MISSING_POLICIES)}"
            )
        if not 1 <= int(self.min_available_factors) <= len(factors):
            raise ValueError(
                f"{sleeve_id} min_available_factors must be between 1 and factor count"
            )
        if self.winsor_limit is not None and float(self.winsor_limit) <= 0:
            raise ValueError(f"{sleeve_id} winsor_limit must be positive or null")
        if self.weighting_method == "static" and sum(
            spec.weight for spec in factors
        ) <= 0:
            raise ValueError(f"{sleeve_id} static weights must contain positive exposure")
        object.__setattr__(self, "sleeve_id", sleeve_id)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "min_available_factors", int(self.min_available_factors))
        object.__setattr__(
            self,
            "winsor_limit",
            None if self.winsor_limit is None else float(self.winsor_limit),
        )

    @property
    def normalized_weights(self) -> dict[str, float]:
        if self.weighting_method == "equal":
            weight = 1.0 / len(self.factors)
            return {spec.factor_id: weight for spec in self.factors}
        gross = sum(abs(spec.weight) for spec in self.factors)
        return {spec.factor_id: spec.weight / gross for spec in self.factors}

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> "SleeveSpec":
        defaults = defaults or {}
        raw_factors = payload.get("factors") or ()
        if not isinstance(raw_factors, (list, tuple)):
            raise ValueError("sleeve factors must be a list")
        blend = payload.get("blend") or {}
        if not isinstance(blend, Mapping):
            raise ValueError("sleeve blend must be a mapping")
        return cls(
            sleeve_id=str(payload.get("sleeve_id") or payload.get("id") or ""),
            factors=tuple(FactorSpec.from_mapping(item) for item in raw_factors),
            weighting_method=str(
                blend.get("weighting_method", defaults.get("weighting_method", "equal"))
            ),
            normalization=str(
                blend.get("normalization", defaults.get("normalization", "cross_sectional_zscore"))
            ),
            missing_policy=str(
                blend.get("missing_policy", defaults.get("missing_policy", "renormalize_available"))
            ),
            min_available_factors=int(
                blend.get("min_available_factors", defaults.get("min_available_factors", 1))
            ),
            winsor_limit=_optional_float(
                blend.get("winsor_limit", defaults.get("winsor_limit", 3.0))
            ),
            description=str(payload.get("description") or ""),
        )


@dataclass(frozen=True, slots=True)
class RouterSpec:
    """Reference to one router recipe and its frozen state-map parameters."""

    router_id: str
    state_file: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        router_id = str(self.router_id).strip()
        if not router_id:
            raise ValueError("router_id cannot be empty")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("router parameters must be a mapping")
        object.__setattr__(self, "router_id", router_id)
        object.__setattr__(self, "state_file", _optional_text(self.state_file))
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RouterSpec":
        return cls(
            router_id=str(payload.get("router_id") or payload.get("id") or ""),
            state_file=_optional_text(payload.get("state_file")),
            parameters=payload.get("parameters") or {},
        )


@dataclass(frozen=True, slots=True)
class StrategyRiskOverlaySpec:
    """Reference to one causal strategy-level exposure overlay."""

    overlay_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlay_id = str(self.overlay_id).strip()
        if not overlay_id:
            raise ValueError("overlay_id cannot be empty")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("risk overlay parameters must be a mapping")
        object.__setattr__(self, "overlay_id", overlay_id)
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "StrategyRiskOverlaySpec":
        return cls(
            overlay_id=str(payload.get("overlay_id") or payload.get("id") or ""),
            parameters=payload.get("parameters") or {},
        )


@dataclass(frozen=True, slots=True)
class FactorPortfolioConfig:
    """Auditable recipe for turning several factors into one strategy signal."""

    strategy_id: str
    name: str
    market_vertical: str
    factors: tuple[FactorSpec, ...] = field(default_factory=tuple)
    sleeves: tuple[SleeveSpec, ...] = field(default_factory=tuple)
    router: RouterSpec | None = None
    risk_overlays: tuple[StrategyRiskOverlaySpec, ...] = field(
        default_factory=tuple
    )
    weighting_method: str = "equal"
    normalization: str = "cross_sectional_zscore"
    missing_policy: str = "renormalize_available"
    min_available_factors: int = 1
    winsor_limit: float | None = 3.0
    execution_mode: str = "risk_desk"
    max_gross_leverage: float = 1.0
    max_weight_per_asset: float | None = 0.05
    max_margin_utilization: float | None = None
    neutralize: bool = True
    sizing_modules: tuple[str, ...] = field(default_factory=tuple)
    kelly_fraction: float = 0.5
    liquidity_policy: Mapping[str, Any] = field(default_factory=dict)
    temporal_policy: Mapping[str, Any] = field(default_factory=dict)
    return_horizon: str = "auto"
    success_criterion_profile: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        strategy_id = str(self.strategy_id).strip()
        if not strategy_id:
            raise ValueError("strategy_id cannot be empty")
        name = str(self.name).strip() or strategy_id
        market_vertical = normalize_market_vertical(self.market_vertical)
        if market_vertical not in ASSET_TAXONOMY:
            raise ValueError(
                f"market_vertical must be one of {sorted(ASSET_TAXONOMY)}"
            )
        factors = tuple(self.factors)
        sleeves = tuple(self.sleeves)
        risk_overlays = tuple(self.risk_overlays)
        if not isinstance(self.liquidity_policy, Mapping):
            raise ValueError("liquidity_policy must be a mapping")
        liquidity_policy = dict(self.liquidity_policy)
        if not isinstance(self.temporal_policy, Mapping):
            raise ValueError("temporal_policy must be a mapping")
        temporal_policy = dict(self.temporal_policy)
        success_criterion_profile = _optional_text(
            self.success_criterion_profile
        )
        if bool(factors) == bool(sleeves):
            raise ValueError("configure either factors or routed sleeves, but not both")
        factor_ids = [spec.factor_id for spec in factors]
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor_id values must be unique within a portfolio")
        sleeve_ids = [spec.sleeve_id for spec in sleeves]
        if len(set(sleeve_ids)) != len(sleeve_ids):
            raise ValueError("sleeve_id values must be unique within a strategy")
        if sleeves and len(sleeves) < 2:
            raise ValueError("a routed strategy requires at least two sleeves")
        if sleeves and self.router is None:
            raise ValueError("routed sleeves require a router")
        if factors and self.router is not None:
            raise ValueError("a router requires named sleeves rather than top-level factors")
        overlay_ids = [spec.overlay_id for spec in risk_overlays]
        if len(set(overlay_ids)) != len(overlay_ids):
            raise ValueError("risk overlay IDs must be unique within a strategy")
        if self.weighting_method not in VALID_WEIGHTING_METHODS:
            raise ValueError(
                f"weighting_method must be one of {sorted(VALID_WEIGHTING_METHODS)}"
            )
        if self.normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {sorted(VALID_NORMALIZATIONS)}"
            )
        if self.missing_policy not in VALID_MISSING_POLICIES:
            raise ValueError(
                f"missing_policy must be one of {sorted(VALID_MISSING_POLICIES)}"
            )
        if factors and not 1 <= int(self.min_available_factors) <= len(factors):
            raise ValueError("min_available_factors must be between 1 and factor count")
        if self.winsor_limit is not None and float(self.winsor_limit) <= 0:
            raise ValueError("winsor_limit must be positive or null")
        if float(self.max_gross_leverage) <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if self.max_weight_per_asset is not None and float(self.max_weight_per_asset) <= 0:
            raise ValueError("max_weight_per_asset must be positive or null")
        if self.max_margin_utilization is not None and not 0.0 < float(
            self.max_margin_utilization
        ) <= 1.0:
            raise ValueError("max_margin_utilization must be between zero and one")
        if factors and self.weighting_method == "static" and sum(spec.weight for spec in factors) <= 0:
            raise ValueError("static factor weights must contain positive exposure")

        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "market_vertical", market_vertical)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "sleeves", sleeves)
        object.__setattr__(self, "risk_overlays", risk_overlays)
        object.__setattr__(self, "min_available_factors", int(self.min_available_factors))
        object.__setattr__(
            self,
            "winsor_limit",
            None if self.winsor_limit is None else float(self.winsor_limit),
        )
        object.__setattr__(self, "max_gross_leverage", float(self.max_gross_leverage))
        object.__setattr__(
            self,
            "max_weight_per_asset",
            None
            if self.max_weight_per_asset is None
            else float(self.max_weight_per_asset),
        )
        object.__setattr__(
            self,
            "max_margin_utilization",
            None
            if self.max_margin_utilization is None
            else float(self.max_margin_utilization),
        )
        object.__setattr__(self, "sizing_modules", tuple(self.sizing_modules))
        object.__setattr__(self, "kelly_fraction", float(self.kelly_fraction))
        object.__setattr__(self, "liquidity_policy", liquidity_policy)
        object.__setattr__(self, "temporal_policy", temporal_policy)
        object.__setattr__(
            self,
            "success_criterion_profile",
            success_criterion_profile,
        )

    @property
    def normalized_weights(self) -> dict[str, float]:
        if not self.factors:
            return {}
        if self.weighting_method == "equal":
            equal_weight = 1.0 / len(self.factors)
            return {spec.factor_id: equal_weight for spec in self.factors}
        gross = sum(abs(spec.weight) for spec in self.factors)
        return {spec.factor_id: spec.weight / gross for spec in self.factors}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normalized_weights"] = self.normalized_weights
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FactorPortfolioConfig":
        raw_factors = payload.get("factors") or ()
        if not isinstance(raw_factors, (list, tuple)):
            raise ValueError("factors must be a list")
        raw_sleeves = payload.get("sleeves") or ()
        if not isinstance(raw_sleeves, (list, tuple)):
            raise ValueError("sleeves must be a list")
        execution = payload.get("execution") or {}
        if not isinstance(execution, Mapping):
            raise ValueError("execution must be a mapping")
        blend = payload.get("blend") or {}
        if not isinstance(blend, Mapping):
            raise ValueError("blend must be a mapping")
        defaults = {
            "weighting_method": blend.get("weighting_method", payload.get("weighting_method", "equal")),
            "normalization": blend.get("normalization", payload.get("normalization", "cross_sectional_zscore")),
            "missing_policy": blend.get("missing_policy", payload.get("missing_policy", "renormalize_available")),
            "min_available_factors": blend.get("min_available_factors", payload.get("min_available_factors", 1)),
            "winsor_limit": blend.get("winsor_limit", payload.get("winsor_limit", 3.0)),
        }
        raw_router = payload.get("router")
        if raw_router is not None and not isinstance(raw_router, Mapping):
            raise ValueError("router must be a mapping")
        raw_overlays = payload.get("risk_overlays") or ()
        if not isinstance(raw_overlays, (list, tuple)):
            raise ValueError("risk_overlays must be a list")
        liquidity = payload.get("liquidity") or payload.get("liquidity_policy") or {}
        if not isinstance(liquidity, Mapping):
            raise ValueError("liquidity must be a mapping")
        temporal = payload.get("temporal") or payload.get("temporal_policy") or {}
        if not isinstance(temporal, Mapping):
            raise ValueError("temporal must be a mapping")
        raw_criterion = payload.get("success_criterion") or payload.get(
            "success_criterion_profile"
        )
        if isinstance(raw_criterion, Mapping):
            success_criterion_profile = _optional_text(
                raw_criterion.get("profile_id")
            )
        else:
            success_criterion_profile = _optional_text(raw_criterion)
        return cls(
            strategy_id=str(payload.get("strategy_id") or ""),
            name=str(payload.get("name") or payload.get("strategy_id") or ""),
            market_vertical=str(payload.get("market_vertical") or ""),
            factors=tuple(FactorSpec.from_mapping(item) for item in raw_factors),
            sleeves=tuple(
                SleeveSpec.from_mapping(item, defaults=defaults) for item in raw_sleeves
            ),
            router=RouterSpec.from_mapping(raw_router) if raw_router else None,
            risk_overlays=tuple(
                StrategyRiskOverlaySpec.from_mapping(item)
                for item in raw_overlays
            ),
            weighting_method=str(
                blend.get("weighting_method", payload.get("weighting_method", "equal"))
            ),
            normalization=str(
                blend.get("normalization", payload.get("normalization", "cross_sectional_zscore"))
            ),
            missing_policy=str(
                blend.get("missing_policy", payload.get("missing_policy", "renormalize_available"))
            ),
            min_available_factors=int(
                blend.get("min_available_factors", payload.get("min_available_factors", 1))
            ),
            winsor_limit=_optional_float(
                blend.get("winsor_limit", payload.get("winsor_limit", 3.0))
            ),
            execution_mode=str(execution.get("mode", payload.get("execution_mode", "risk_desk"))),
            max_gross_leverage=float(
                execution.get("max_gross_leverage", payload.get("max_gross_leverage", 1.0))
            ),
            max_weight_per_asset=_optional_float(
                execution.get("max_weight_per_asset", payload.get("max_weight_per_asset", 0.05))
            ),
            max_margin_utilization=_optional_float(
                execution.get(
                    "max_margin_utilization",
                    payload.get("max_margin_utilization"),
                )
            ),
            neutralize=bool(execution.get("neutralize", payload.get("neutralize", True))),
            sizing_modules=tuple(
                execution.get("sizing_modules", payload.get("sizing_modules", ())) or ()
            ),
            kelly_fraction=float(
                execution.get("kelly_fraction", payload.get("kelly_fraction", 0.5))
            ),
            liquidity_policy=dict(liquidity),
            temporal_policy=dict(temporal),
            return_horizon=str(payload.get("return_horizon", "auto")),
            success_criterion_profile=success_criterion_profile,
            description=str(payload.get("description") or ""),
        )


def load_factor_portfolio_config(path: str | Path) -> FactorPortfolioConfig:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"factor portfolio config not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("factor portfolio config must contain a mapping")
    return FactorPortfolioConfig.from_mapping(payload)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "FactorPortfolioConfig",
    "FactorSpec",
    "RouterSpec",
    "SleeveSpec",
    "StrategyRiskOverlaySpec",
    "VALID_MISSING_POLICIES",
    "VALID_NORMALIZATIONS",
    "VALID_WEIGHTING_METHODS",
    "load_factor_portfolio_config",
]
