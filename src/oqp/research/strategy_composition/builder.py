"""Typed drafts for assembling heterogeneous research strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from oqp.research.contracts import validate_factor_market_compatibility
from oqp.research.factors import load_factor_module
from oqp.research.sleeves import (
    ExtractedSleeveConfig,
    PersistentSleeveConfig,
    SleeveConstructionConfig,
    load_sleeve_module,
    supports_extracted_sleeve_execution,
)
from oqp.research.strategy_composition.contracts import (
    StrategyAllocatorConfig,
    StrategyExecutionConfig,
)


STRATEGY_BUILDER_SCHEMA_VERSION = 1


class StrategyCoreType(str, Enum):
    DIRECT_FACTOR = "direct_factor"
    FACTOR_SLEEVE = "factor_sleeve"
    FACTOR_BLEND = "factor_blend"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    ML_PREDICTIVE = "ml_predictive"
    ROUTED_COMPONENTS = "routed_components"


@dataclass(frozen=True, slots=True)
class StrategyBranchConfig:
    """One position stream that can stand alone or enter a router."""

    branch_id: str
    factor_ids: tuple[str, ...]
    sleeve_id: str | None = None
    execution_mode: str = "risk_desk"

    def __post_init__(self) -> None:
        branch_id = str(self.branch_id).strip()
        factor_ids = tuple(str(value).strip() for value in self.factor_ids)
        sleeve_id = str(self.sleeve_id).strip() if self.sleeve_id else None
        execution_mode = str(self.execution_mode).strip().lower()
        if not branch_id:
            raise ValueError("branch_id cannot be empty")
        if not factor_ids or any(not value.startswith("fac_") for value in factor_ids):
            raise ValueError("each branch requires stable fac_* references")
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor references must be unique within a branch")
        if sleeve_id and not sleeve_id.startswith("slv_"):
            raise ValueError("sleeve references must use stable slv_* IDs")
        if execution_mode not in {"direct", "risk_desk", "statarb"}:
            raise ValueError("execution_mode must be direct, risk_desk, or statarb")
        object.__setattr__(self, "branch_id", branch_id)
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "sleeve_id", sleeve_id)
        object.__setattr__(self, "execution_mode", execution_mode)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrategyBranchConfig":
        raw_factors = payload.get("factor_ids") or ()
        if not isinstance(raw_factors, (list, tuple)):
            raise ValueError("branch factor_ids must be a list")
        return cls(
            branch_id=str(payload.get("branch_id") or ""),
            factor_ids=tuple(str(value) for value in raw_factors),
            sleeve_id=_optional_text(payload.get("sleeve_id")),
            execution_mode=str(payload.get("execution_mode") or "risk_desk"),
        )


@dataclass(frozen=True, slots=True)
class StrategyCoreConfig:
    """The position-producing core of a strategy draft."""

    core_type: StrategyCoreType
    branches: tuple[StrategyBranchConfig, ...]
    router_id: str | None = None
    router_state_file: str | None = None
    ml_experiment_id: str | None = None
    ml_predictions_path: str | None = None

    def __post_init__(self) -> None:
        core_type = StrategyCoreType(self.core_type)
        branches = tuple(self.branches)
        router_id = _optional_text(self.router_id)
        router_state_file = _optional_text(self.router_state_file)
        ml_experiment_id = _optional_text(self.ml_experiment_id)
        ml_predictions_path = _optional_text(self.ml_predictions_path)
        if not branches:
            raise ValueError("a strategy requires at least one position-producing branch")
        if len({branch.branch_id for branch in branches}) != len(branches):
            raise ValueError("strategy branch IDs must be unique")
        if core_type == StrategyCoreType.ROUTED_COMPONENTS:
            if len(branches) < 2:
                raise ValueError("a routed strategy requires at least two branches")
            if not router_id or not router_id.startswith("rtr_"):
                raise ValueError("a routed strategy requires one stable rtr_* reference")
        elif len(branches) != 1:
            raise ValueError(f"{core_type.value} requires exactly one branch")
        elif router_id is not None or router_state_file is not None:
            raise ValueError("routers are only valid for routed_components")

        branch = branches[0] if len(branches) == 1 else None
        if core_type == StrategyCoreType.DIRECT_FACTOR:
            if len(branch.factor_ids) != 1 or branch.sleeve_id is not None:
                raise ValueError("direct_factor requires one factor and no sleeve")
            if branch.execution_mode != "direct":
                raise ValueError("direct_factor requires direct execution")
        elif core_type == StrategyCoreType.FACTOR_SLEEVE:
            if len(branch.factor_ids) != 1 or branch.sleeve_id is None:
                raise ValueError("factor_sleeve requires one factor and one sleeve")
        elif core_type == StrategyCoreType.FACTOR_BLEND:
            if len(branch.factor_ids) < 2 or branch.sleeve_id is not None:
                raise ValueError("factor_blend requires at least two factors and no sleeve")
        elif core_type == StrategyCoreType.STATISTICAL_ARBITRAGE:
            if len(branch.factor_ids) != 1 or branch.sleeve_id is not None:
                raise ValueError("statistical_arbitrage requires one direct stat-arb component")
            if branch.execution_mode != "statarb":
                raise ValueError("statistical_arbitrage requires statarb execution")
        elif core_type == StrategyCoreType.ML_PREDICTIVE:
            if len(branch.factor_ids) != 1:
                raise ValueError("ml_predictive requires one factor adapter")
            if not ml_experiment_id or not ml_predictions_path:
                raise ValueError(
                    "ml_predictive requires a registered experiment and OOS predictions"
                )

        object.__setattr__(self, "core_type", core_type)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "router_id", router_id)
        object.__setattr__(self, "router_state_file", router_state_file)
        object.__setattr__(self, "ml_experiment_id", ml_experiment_id)
        object.__setattr__(self, "ml_predictions_path", ml_predictions_path)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrategyCoreConfig":
        raw_branches = payload.get("branches") or ()
        if not isinstance(raw_branches, (list, tuple)):
            raise ValueError("core branches must be a list")
        return cls(
            core_type=StrategyCoreType(str(payload.get("type") or "")),
            branches=tuple(
                StrategyBranchConfig.from_mapping(item) for item in raw_branches
            ),
            router_id=_optional_text(payload.get("router_id")),
            router_state_file=_optional_text(payload.get("router_state_file")),
            ml_experiment_id=_optional_text(payload.get("ml_experiment_id")),
            ml_predictions_path=_optional_text(payload.get("ml_predictions_path")),
        )


@dataclass(frozen=True, slots=True)
class StrategyBuilderConfig:
    """A reproducible strategy draft independent of any one core geometry."""

    strategy_id: str
    name: str
    market_vertical: str
    core: StrategyCoreConfig
    risk_overlays: tuple[str, ...] = ()
    allocator: StrategyAllocatorConfig = StrategyAllocatorConfig()
    execution: StrategyExecutionConfig = StrategyExecutionConfig()
    research_mode: str = "exploratory"
    schema_version: int = STRATEGY_BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("strategy_id", "name", "market_vertical"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, value)
        if not self.strategy_id.startswith("str_"):
            raise ValueError("strategy_id must use a stable str_* ID")
        overlays = tuple(str(value).strip() for value in self.risk_overlays)
        if any(not value.startswith("ovl_") for value in overlays):
            raise ValueError("risk overlay references must use stable ovl_* IDs")
        if len(set(overlays)) != len(overlays):
            raise ValueError("risk overlay references must be unique")
        mode = str(self.research_mode).strip().lower()
        if mode not in {"exploratory", "validation", "frozen_holdout"}:
            raise ValueError("unknown research_mode")
        object.__setattr__(self, "risk_overlays", overlays)
        object.__setattr__(self, "research_mode", mode)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["core"]["type"] = self.core.core_type.value
        payload["core"].pop("core_type", None)
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StrategyExecutionSupport:
    runnable: bool
    backend: str
    reason: str


def strategy_execution_support(config: StrategyBuilderConfig) -> StrategyExecutionSupport:
    """State what the current shared runner can execute without approximation."""

    core = config.core
    if core.core_type == StrategyCoreType.FACTOR_SLEEVE:
        return _factor_sleeve_execution_support(config)
    if core.core_type == StrategyCoreType.ML_PREDICTIVE:
        branch = core.branches[0]
        if branch.sleeve_id:
            return StrategyExecutionSupport(
                True,
                "registered OOS predictions + reusable sleeve engine",
                "Registered OOS predictions feed the factor adapter, frozen sleeve, and shared evaluator.",
            )
        return StrategyExecutionSupport(
            True,
            "factor_portfolio + registered OOS predictions",
            "The registered OOS predictions feed the factor adapter and shared evaluator.",
        )
    if core.core_type == StrategyCoreType.ROUTED_COMPONENTS:
        if not core.router_state_file:
            return StrategyExecutionSupport(
                False,
                "factor_portfolio router",
                "A causal router-state file is required before this strategy can run.",
            )
        if any(branch.sleeve_id for branch in core.branches):
            return StrategyExecutionSupport(
                False,
                "router + reusable sleeve engine",
                "Routed reusable-sleeve branches are not connected to the generic evaluator yet.",
            )
        if any(branch.execution_mode != "risk_desk" for branch in core.branches):
            return StrategyExecutionSupport(
                False,
                "heterogeneous branch router",
                "Routing direct-target or stat-arb branches requires the completed-stream router adapter.",
            )
        return StrategyExecutionSupport(
            True,
            "factor_portfolio router",
            "The shared runner builds each factor branch, then applies the selected causal router.",
        )
    return StrategyExecutionSupport(
        True,
        "factor_portfolio",
        "This core is supported by the shared factor-portfolio evaluator.",
    )


def _factor_sleeve_execution_support(
    config: StrategyBuilderConfig,
) -> StrategyExecutionSupport:
    """Resolve factor/sleeve capability from registered, data-free contracts."""

    backend = "factor + reusable sleeve engine"
    branch = config.core.branches[0]
    factor_id = branch.factor_ids[0]
    sleeve_id = branch.sleeve_id
    if sleeve_id is None:
        return StrategyExecutionSupport(
            False,
            backend,
            "A factor-sleeve core requires one registered sleeve.",
        )

    try:
        factor_module = load_factor_module(factor_id)
        validate_factor_market_compatibility(
            factor_module,
            config.market_vertical,
            factor_id=factor_id,
        )
    except (ImportError, ModuleNotFoundError, TypeError, ValueError) as exc:
        return StrategyExecutionSupport(
            False,
            backend,
            f"{factor_id} cannot run on {config.market_vertical}: {exc}",
        )

    factor_contract = getattr(factor_module, "FACTOR_CONTRACT", None)
    if not isinstance(factor_contract, Mapping):
        return StrategyExecutionSupport(
            False,
            backend,
            f"{factor_id} must declare a data-independent FACTOR_CONTRACT.",
        )
    factor_geometry = str(
        factor_contract.get("evaluation_geometry") or ""
    ).strip().lower()
    factor_return = str(
        factor_contract.get("return_assumption") or ""
    ).strip().lower()
    factor_metadata = getattr(factor_module, "FACTOR_METADATA", {}) or {}
    orientation = str(
        getattr(factor_module, "SIGNAL_ORIENTATION", None)
        or (
            factor_metadata.get("signal_orientation")
            if isinstance(factor_metadata, Mapping)
            else None
        )
        or ""
    ).strip().lower()
    if orientation not in {"higher_is_bullish", "higher_is_bearish"}:
        return StrategyExecutionSupport(
            False,
            backend,
            f"{factor_id} must declare a valid SIGNAL_ORIENTATION.",
        )

    try:
        sleeve_module = load_sleeve_module(sleeve_id)
        sleeve_config = sleeve_module.build_config(
            factor_id,
            market_vertical=config.market_vertical,
            signal_orientation=orientation,
        )
    except (ImportError, ModuleNotFoundError, TypeError, ValueError) as exc:
        return StrategyExecutionSupport(
            False,
            backend,
            f"{sleeve_id} cannot build an execution config for {factor_id}: {exc}",
        )

    if isinstance(sleeve_config, SleeveConstructionConfig):
        if sleeve_config.construction_geometry != factor_geometry:
            return StrategyExecutionSupport(
                False,
                backend,
                "Factor and sleeve evaluation geometries differ: "
                f"{factor_geometry or 'missing'} != "
                f"{sleeve_config.construction_geometry}.",
            )
        if sleeve_config.return_assumption != factor_return:
            return StrategyExecutionSupport(
                False,
                backend,
                "Factor and sleeve return assumptions differ: "
                f"{factor_return or 'missing'} != "
                f"{sleeve_config.return_assumption}.",
            )
        return StrategyExecutionSupport(
            True,
            backend,
            "The registered factor contract and standard sleeve config have "
            "compatible geometry and return timing.",
        )

    if isinstance(sleeve_config, PersistentSleeveConfig):
        if factor_geometry != "cross_sectional":
            return StrategyExecutionSupport(
                False,
                backend,
                f"{sleeve_id} requires a cross-sectional factor; "
                f"{factor_id} declares {factor_geometry or 'missing'}.",
            )
        return StrategyExecutionSupport(
            True,
            backend,
            "The cross-sectional factor feeds a registered persistent sleeve.",
        )

    if isinstance(sleeve_config, ExtractedSleeveConfig):
        if factor_id not in sleeve_config.source_factor_ids:
            return StrategyExecutionSupport(
                False,
                backend,
                f"{sleeve_id} is not registered for factor {factor_id}.",
            )
        if not supports_extracted_sleeve_execution(sleeve_config):
            support = (
                "declares execution_supported=False"
                if not sleeve_config.execution_supported
                else "has no registered rule-family execution adapter"
            )
            return StrategyExecutionSupport(
                False,
                backend,
                f"{sleeve_id} {support}.",
            )
        construction_geometry = str(
            (sleeve_config.parameters or {}).get("construction_geometry") or ""
        ).strip().lower()
        required_geometry = {
            "time_series_stateful": "time_series",
            "cross_sectional": "cross_sectional",
        }.get(construction_geometry)
        if required_geometry != factor_geometry:
            return StrategyExecutionSupport(
                False,
                backend,
                "Factor and extracted sleeve evaluation geometries differ: "
                f"{factor_geometry or 'missing'} != "
                f"{construction_geometry or 'missing'}.",
            )
        return StrategyExecutionSupport(
            True,
            backend,
            f"The factor feeds the registered {sleeve_config.rule_family} "
            "execution adapter; return timing is inherited from its factor "
            "contract.",
        )

    return StrategyExecutionSupport(
        False,
        backend,
        f"{sleeve_id} returned unsupported config type "
        f"{type(sleeve_config).__name__}.",
    )


def load_strategy_builder_config(path: str | Path) -> StrategyBuilderConfig:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("strategy builder YAML must contain a mapping")
    raw = payload.get("strategy", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("strategy must be a mapping")
    raw_core = raw.get("core")
    if not isinstance(raw_core, Mapping):
        raise ValueError("strategy.core must be a mapping")
    allocator = raw.get("allocator") or {}
    execution = raw.get("execution") or {}
    if not isinstance(allocator, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("allocator and execution must be mappings")
    return StrategyBuilderConfig(
        strategy_id=str(raw.get("strategy_id") or ""),
        name=str(raw.get("name") or raw.get("strategy_id") or ""),
        market_vertical=str(raw.get("market_vertical") or ""),
        core=StrategyCoreConfig.from_mapping(raw_core),
        risk_overlays=tuple(raw.get("risk_overlays") or ()),
        allocator=StrategyAllocatorConfig(**dict(allocator)),
        execution=StrategyExecutionConfig(**dict(execution)),
        research_mode=str(raw.get("research_mode") or "exploratory"),
        schema_version=int(raw.get("schema_version", STRATEGY_BUILDER_SCHEMA_VERSION)),
    )


def write_strategy_builder_config(
    config: StrategyBuilderConfig,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"strategy": config.to_dict()}
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return destination


def build_strategy_backtest_command(
    *,
    config_path: str | Path,
    data_file: str | Path,
    build_only: bool = False,
) -> tuple[str, ...]:
    command = [
        ".venv/bin/python",
        "scripts/research/run_strategy_backtest.py",
        "--config",
        Path(config_path).as_posix(),
        "--data-file",
        Path(data_file).as_posix(),
    ]
    if build_only:
        command.append("--build-only")
    return tuple(command)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = [
    "STRATEGY_BUILDER_SCHEMA_VERSION",
    "StrategyBranchConfig",
    "StrategyBuilderConfig",
    "StrategyCoreConfig",
    "StrategyCoreType",
    "StrategyExecutionSupport",
    "build_strategy_backtest_command",
    "load_strategy_builder_config",
    "strategy_execution_support",
    "write_strategy_builder_config",
]
