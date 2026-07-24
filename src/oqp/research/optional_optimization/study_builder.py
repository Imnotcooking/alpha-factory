"""Build immutable, disabled Phase 8 study definitions from live registries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import importlib.util
import json
import math
from pathlib import Path
import re
from types import ModuleType
from typing import Any, Mapping

import pandas as pd
import yaml

from oqp.optimization import (
    ComponentParameterSchema,
    FrozenResearchInputs,
    OptimizationMethodRegistry,
    SearchBudget,
    resolve_component_parameter_schema,
    stable_optimization_hash,
)
from oqp.research.dataset_fingerprints import (
    DEFAULT_DATASET_MANIFEST_ROOT,
    DatasetManifest,
    load_dataset_manifest,
)
from oqp.research.factors import load_factor_module
from oqp.research.optimization_objectives import (
    OptimizationObjectiveRegistry,
    validate_phase8_objective_profile,
)
from oqp.research.optional_optimization.contracts import (
    Phase8ExperimentSpec,
    Phase8FoldConfig,
)
from oqp.research.strategy_routing import load_router_module


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STUDY_CONFIG_ROOT = (
    REPO_ROOT / "departments" / "research" / "optimization_studies"
)
DEFAULT_PHASE8_ARTIFACT_ROOT = (
    REPO_ROOT
    / "runtime"
    / "artifacts"
    / "research"
    / "optional_optimization"
)
FACTOR_MANIFEST = (
    REPO_ROOT
    / "runtime"
    / "artifacts"
    / "research"
    / "factor_registry_normalization"
    / "cohort_manifest.csv"
)


@dataclass(frozen=True, slots=True)
class OptimizationComponentOption:
    component_id: str
    component_type: str
    label: str
    source_path: str
    market_vertical: str = ""
    data_frequency: str = ""
    research_status: str = ""
    schema_declared: bool = False


@dataclass(frozen=True, slots=True)
class OptimizationDatasetOption:
    manifest_path: str
    dataset_id: str
    dataset_version: str
    market_vertical: str
    data_frequency: str
    row_count: int | None
    instrument_count: int | None
    date_start: str | None
    date_end: str | None
    aggregate_sha256: str

    @property
    def label(self) -> str:
        period = " to ".join(
            value[:10] for value in (self.date_start, self.date_end) if value
        )
        instruments = (
            f"{self.instrument_count} instruments"
            if self.instrument_count is not None
            else "instrument count unknown"
        )
        return (
            f"{self.dataset_id} | {self.market_vertical} {self.data_frequency} | "
            f"{period or 'dates unknown'} | {instruments}"
        )


@dataclass(frozen=True, slots=True)
class FrozenComponentOption:
    component_id: str
    mutable_layer: str
    study_id: str
    candidate_fingerprint: str
    frozen_at: str
    holdout_start: str
    dataset_fingerprint: str
    holdout_fingerprint: str
    source_path: str

    @property
    def label(self) -> str:
        return (
            f"{self.component_id} | {self.mutable_layer} | "
            f"{self.study_id} | frozen {self.frozen_at[:10]}"
        )


def load_component_options(
    purpose_id: str,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> tuple[OptimizationComponentOption, ...]:
    root = Path(repo_root).expanduser().resolve()
    purpose = OptimizationMethodRegistry.load().resolve_purpose(purpose_id)
    if purpose.layer == "factor":
        return _factor_options(root)
    if purpose.layer == "sleeve":
        return _file_options(
            root / "departments/research/strategies/sleeves",
            "slv_*.py",
            "sleeve",
            "SLEEVE_PARAMETERS",
        )
    if purpose.layer == "router":
        return _file_options(
            root / "departments/research/routers",
            "rtr_*.py",
            "router",
            "ROUTER_PARAMETERS",
        )
    if purpose.layer == "overlay":
        return _file_options(
            root / "departments/research/strategy_overlays",
            "ovl_*.py",
            "risk_overlay",
            "OVERLAY_PARAMETERS",
        )
    return ()


def _factor_options(root: Path) -> tuple[OptimizationComponentOption, ...]:
    manifest_path = root / FACTOR_MANIFEST.relative_to(REPO_ROOT)
    if not manifest_path.exists():
        return ()
    manifest = pd.read_csv(manifest_path)
    eligible = manifest.loc[
        manifest["phase_1_ready"].astype(bool)
        & pd.to_numeric(manifest["parameter_count"], errors="coerce").fillna(0).gt(0)
    ].copy()
    rows = []
    for record in eligible.to_dict(orient="records"):
        source = str(record.get("source") or "")
        rows.append(
            OptimizationComponentOption(
                component_id=str(record["factor_id"]),
                component_type="factor",
                label=str(record["factor_id"]),
                source_path=source,
                market_vertical=str(record.get("native_market") or ""),
                data_frequency=str(record.get("data_frequency") or ""),
                research_status="phase_1_ready",
                schema_declared=True,
            )
        )
    return tuple(sorted(rows, key=lambda item: item.component_id))


def _file_options(
    root: Path,
    pattern: str,
    component_type: str,
    declaration_name: str,
) -> tuple[OptimizationComponentOption, ...]:
    rows = []
    for path in sorted(root.glob(pattern)):
        text = path.read_text(encoding="utf-8")
        rows.append(
            OptimizationComponentOption(
                component_id=path.stem,
                component_type=component_type,
                label=path.stem,
                source_path=str(path.relative_to(REPO_ROOT)),
                schema_declared=bool(
                    re.search(
                        rf"(?m)^{re.escape(declaration_name)}\s*=",
                        text,
                    )
                ),
            )
        )
    return tuple(rows)


def load_frozen_component_options(
    *,
    artifact_root: str | Path = DEFAULT_PHASE8_ARTIFACT_ROOT,
    accepted_prefixes: tuple[str, ...] = (),
) -> tuple[FrozenComponentOption, ...]:
    root = Path(artifact_root).expanduser().resolve()
    rows: list[FrozenComponentOption] = []
    for path in sorted(root.glob("*/frozen_candidate.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol = json.loads(
                (path.parent / "protocol.json").read_text(encoding="utf-8")
            )
            component_id = str(payload["component_id"]).strip()
            mutable_layer = str(payload["mutable_layer"]).strip()
            study_id = str(payload["study_id"]).strip()
            frozen_at = str(payload["frozen_at"]).strip()
            holdout_start = str(payload["holdout_start"]).strip()
            dataset_fingerprint = str(
                protocol["frozen_inputs"]["dataset_fingerprint"]
            ).strip()
            holdout_fingerprint = str(payload["holdout_fingerprint"]).strip()
        except Exception:
            continue
        if accepted_prefixes and not component_id.startswith(accepted_prefixes):
            continue
        rows.append(
            FrozenComponentOption(
                component_id=component_id,
                mutable_layer=mutable_layer,
                study_id=study_id,
                candidate_fingerprint=stable_optimization_hash(payload),
                frozen_at=frozen_at,
                holdout_start=holdout_start,
                dataset_fingerprint=dataset_fingerprint,
                holdout_fingerprint=holdout_fingerprint,
                source_path=str(path),
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.component_id,
                item.frozen_at,
                item.study_id,
            ),
        )
    )


def load_dataset_options(
    *,
    manifest_root: str | Path = DEFAULT_DATASET_MANIFEST_ROOT,
    market_vertical: str = "",
    data_frequency: str = "",
) -> tuple[OptimizationDatasetOption, ...]:
    root = Path(manifest_root).expanduser().resolve()
    rows: list[OptimizationDatasetOption] = []
    for path in sorted(root.glob("*/*.json")):
        try:
            manifest = load_dataset_manifest(path)
        except Exception:
            continue
        if market_vertical and manifest.market_vertical != market_vertical:
            continue
        if data_frequency and manifest.data_frequency != data_frequency:
            continue
        rows.append(_dataset_option(path, manifest))
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.instrument_count or 0,
                item.row_count or 0,
                item.date_end or "",
            ),
            reverse=True,
        )
    )


def _dataset_option(path: Path, manifest: DatasetManifest) -> OptimizationDatasetOption:
    return OptimizationDatasetOption(
        manifest_path=str(path),
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        market_vertical=manifest.market_vertical,
        data_frequency=manifest.data_frequency,
        row_count=manifest.row_count,
        instrument_count=manifest.instrument_count,
        date_start=manifest.date_start,
        date_end=manifest.date_end,
        aggregate_sha256=manifest.aggregate_sha256,
    )


def resolve_selected_component_schema(
    purpose_id: str,
    component_id: str,
) -> ComponentParameterSchema:
    purpose = OptimizationMethodRegistry.load().resolve_purpose(purpose_id)
    if purpose.layer == "factor":
        module = load_factor_module(component_id, include_public_examples=False)
        return resolve_component_parameter_schema(
            module,
            component_type="factor",
        )
    if purpose.layer == "router":
        module = load_router_module(component_id)
        return resolve_component_parameter_schema(
            module,
            component_type="router",
        )
    declaration = {
        "sleeve": "SLEEVE_PARAMETERS",
        "overlay": "OVERLAY_PARAMETERS",
        "allocator": "ALLOCATOR_PARAMETERS",
    }.get(purpose.layer)
    if declaration is None:
        raise ValueError(
            f"{purpose.label} does not use the generic Phase 8 component schema"
        )
    option = next(
        (
            value
            for value in load_component_options(purpose_id)
            if value.component_id == component_id
        ),
        None,
    )
    if option is None:
        raise KeyError(f"unknown {purpose.layer} component: {component_id}")
    module = _load_module(
        REPO_ROOT / option.source_path,
        f"oqp_optimization_component_{component_id}",
    )
    return resolve_component_parameter_schema(
        module,
        component_type=(
            "risk_overlay" if purpose.layer == "overlay" else purpose.layer
        ),
        declaration_name=declaration,
        component_id=component_id,
    )


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load component module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_schema_rows(
    schema: ComponentParameterSchema,
) -> list[dict[str, Any]]:
    return [
        {
            "parameter": parameter.name,
            "default": parameter.default,
            "type": parameter.parameter_type,
            "tunable": parameter.tunable,
            "low": parameter.low,
            "high": parameter.high,
            "step": parameter.step,
            "choices": list(parameter.choices),
            "log_scale": parameter.log,
            "description": parameter.description,
        }
        for parameter in schema.parameters
    ]


def build_factor_study_definition(
    *,
    study_id: str,
    purpose_id: str,
    component_id: str,
    dataset_manifest_path: str | Path,
    holdout_start: str,
    frozen_on: str,
    max_trials: int,
    seed: int = 42,
    initial_capital: float = 10_000_000.0,
    capital_currency: str = "CNY",
    max_position_weight: float = 0.05,
    fold_config: Phase8FoldConfig | None = None,
    cost_profile_id: str | None = None,
) -> tuple[Phase8ExperimentSpec, ComponentParameterSchema, dict[str, Any]]:
    return build_component_study_definition(
        study_id=study_id,
        purpose_id=purpose_id,
        component_id=component_id,
        dataset_manifest_path=dataset_manifest_path,
        holdout_start=holdout_start,
        frozen_on=frozen_on,
        max_trials=max_trials,
        seed=seed,
        initial_capital=initial_capital,
        capital_currency=capital_currency,
        max_position_weight=max_position_weight,
        fold_config=fold_config,
        cost_profile_id=cost_profile_id,
        frozen_component_fingerprints={},
        frozen_component_dataset_fingerprints={},
        frozen_component_holdout_fingerprints={},
    )


def build_component_study_definition(
    *,
    study_id: str,
    purpose_id: str,
    component_id: str,
    dataset_manifest_path: str | Path,
    holdout_start: str,
    frozen_on: str,
    max_trials: int,
    seed: int = 42,
    initial_capital: float = 10_000_000.0,
    capital_currency: str = "CNY",
    max_position_weight: float = 0.05,
    fold_config: Phase8FoldConfig | None = None,
    cost_profile_id: str | None = None,
    frozen_component_fingerprints: Mapping[str, str] | None = None,
    frozen_component_dataset_fingerprints: Mapping[str, str] | None = None,
    frozen_component_holdout_fingerprints: Mapping[str, str] | None = None,
) -> tuple[Phase8ExperimentSpec, ComponentParameterSchema, dict[str, Any]]:
    # Load the backtesting package first because its evaluator establishes the
    # existing liquidity-module import order.
    import oqp.research.backtesting  # noqa: F401
    from oqp.execution import TransactionCostRegistry
    from oqp.research.liquidity_eligibility import resolve_liquidity_policy
    from oqp.research.temporal_policy import resolve_signal_holding_policy

    methods = OptimizationMethodRegistry.load()
    purpose = methods.resolve_purpose(purpose_id)
    if purpose.status != "governed":
        raise ValueError(
            f"{purpose.label} is {purpose.status}, not a governed Phase 8 purpose"
        )
    if purpose.layer not in {"factor", "sleeve", "router"}:
        raise ValueError(
            f"The generic frozen-definition builder does not support "
            f"{purpose.layer!r} studies."
        )
    if not purpose.objective_profile_id:
        raise ValueError(f"{purpose.label} has no active objective profile")

    schema = resolve_selected_component_schema(purpose_id, component_id)
    if not schema.tunable_names:
        raise ValueError(f"{component_id} declares no tunable parameters")

    manifest_path = Path(dataset_manifest_path).expanduser().resolve()
    manifest = load_dataset_manifest(manifest_path)
    component = _load_selected_component_module(
        purpose_id,
        component_id,
    )
    metadata, contract = _component_metadata_contract(
        component,
        purpose.layer,
    )
    native_market, frequency = _component_market_frequency(
        metadata,
        purpose.layer,
    )
    if native_market and native_market != manifest.market_vertical:
        raise ValueError(
            f"{component_id} is native to {native_market}, not "
            f"{manifest.market_vertical}"
        )
    if frequency and frequency != manifest.data_frequency:
        raise ValueError(
            f"{component_id} expects {frequency} data, not "
            f"{manifest.data_frequency}"
        )

    frozen_date = pd.Timestamp(frozen_on).normalize()
    holdout_date = pd.Timestamp(holdout_start).normalize()
    if frozen_date >= holdout_date:
        raise ValueError("holdout_start must be after the actual protocol freeze date")
    if manifest.date_end and pd.Timestamp(manifest.date_end).normalize() >= holdout_date:
        raise ValueError(
            "the registered development dataset reaches into the proposed holdout"
        )

    liquidity = resolve_liquidity_policy(
        manifest.market_vertical,
        initial_capital=initial_capital,
        capital_currency=capital_currency,
        max_position_weight=max_position_weight,
    )
    if purpose.layer == "factor":
        temporal_frame = pd.DataFrame()
        temporal_frame.attrs.update(
            {
                "data_frequency": manifest.data_frequency,
                "factor_metadata": dict(metadata),
                "factor_contract": dict(contract),
                "signal_frequency": metadata.get("signal_frequency"),
            }
        )
        temporal = resolve_signal_holding_policy(temporal_frame)
        temporal_payload = temporal.to_dict()
        temporal_fingerprint = temporal.fingerprint
    else:
        temporal_payload = {
            "policy_id": f"phase8_{purpose.layer}_temporal_envelope_v1",
            "data_frequency": manifest.data_frequency,
            "component_id": component_id,
            "component_contract": dict(contract),
            "mutable_parameters": list(schema.tunable_names),
            "boundary": (
                "The component may vary only its declared parameters; execution "
                "delay and data frequency remain frozen."
            ),
        }
        temporal_fingerprint = stable_optimization_hash(temporal_payload)
    cost_registry = TransactionCostRegistry.load()
    cost_profile = cost_registry.resolve(
        manifest.market_vertical,
        profile_id=cost_profile_id,
    )
    cost_profile.assert_ready("research_net")

    universe_payload = {
        "policy_id": "dataset_manifest_universe_v1",
        "dataset_fingerprint": manifest.aggregate_sha256,
        "dataset_id": manifest.dataset_id,
        "instrument_count": manifest.instrument_count,
        "row_scope": "manifest_profile_non_missing_close",
        "daily_session_collision_policy": (
            "highest_positive_reported_volume_then_earliest_timestamp"
            if manifest.data_frequency == "daily"
            else "not_applicable"
        ),
    }
    holdout_payload = {
        "definition": "post_freeze_forward_holdout_v1",
        "development_dataset_fingerprint": manifest.aggregate_sha256,
        "market_vertical": manifest.market_vertical,
        "data_frequency": manifest.data_frequency,
        "holdout_start": holdout_date.date().isoformat(),
    }
    frozen_inputs = FrozenResearchInputs(
        dataset_fingerprint=manifest.aggregate_sha256,
        universe_fingerprint=stable_optimization_hash(universe_payload),
        liquidity_policy_fingerprint=liquidity.fingerprint,
        temporal_policy_fingerprint=temporal_fingerprint,
        transaction_cost_profile_fingerprint=cost_profile.fingerprint,
        holdout_fingerprint=stable_optimization_hash(holdout_payload),
    )
    objective_profile = OptimizationObjectiveRegistry.load().resolve(
        purpose.objective_profile_id
    )
    upstream_fingerprints = _validate_frozen_upstream_components(
        objective_profile,
        frozen_component_fingerprints,
    )
    _validate_frozen_upstream_context(
        upstream_fingerprints,
        frozen_component_dataset_fingerprints,
        frozen_component_holdout_fingerprints,
        dataset_fingerprint=manifest.aggregate_sha256,
        holdout_fingerprint=frozen_inputs.holdout_fingerprint,
    )
    sampler_id, grid_combinations = _resolve_study_sampler(
        purpose,
        schema,
        max_trials=max_trials,
    )
    spec = Phase8ExperimentSpec(
        study_id=study_id,
        layer=purpose.layer,
        component_id=component_id,
        parameter_schema_fingerprint=schema.fingerprint,
        objective_profile_id=objective_profile.profile_id,
        objective_profile_fingerprint=objective_profile.fingerprint,
        objectives=tuple(
            objective.to_phase8_objective()
            for objective in objective_profile.objectives
        ),
        selection_priority=objective_profile.selection_priority,
        frozen_inputs=frozen_inputs,
        budget=SearchBudget(max_trials=max_trials, n_jobs=1),
        fold_config=fold_config or Phase8FoldConfig(),
        holdout_start=holdout_date.date().isoformat(),
        frozen_on=frozen_date.date().isoformat(),
        constraints=objective_profile.constraints,
        frozen_component_fingerprints=upstream_fingerprints,
        sampler_id=sampler_id,
        seed=seed,
        enabled=False,
    )
    validate_phase8_objective_profile(spec, objective_profile)
    metadata_payload = {
        "purpose_id": purpose_id,
        "component_source": str(
            getattr(component, "__file__", "") or ""
        ),
        "dataset_manifest_path": str(manifest_path),
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "market_vertical": manifest.market_vertical,
        "data_frequency": manifest.data_frequency,
        "liquidity_policy": liquidity.to_dict(),
        "temporal_policy": temporal_payload,
        "transaction_cost_profile_id": cost_profile.profile_id,
        "universe_policy": universe_payload,
        "holdout_definition": holdout_payload,
        "sampler_selection": {
            "registry_primary_method": purpose.primary_method,
            "selected_method": sampler_id,
            "finite_grid_combinations": grid_combinations,
            "trial_budget": int(max_trials),
            "reason": (
                "Exhaustive grid selected because the complete discrete search "
                "space fits within the frozen trial budget."
                if sampler_id == "grid" and grid_combinations is not None
                else "Registry method selected after checking finite-grid feasibility."
            ),
        },
        "frozen_component_fingerprints": upstream_fingerprints,
        "frozen_component_dataset_fingerprints": dict(
            frozen_component_dataset_fingerprints or {}
        ),
        "frozen_component_holdout_fingerprints": dict(
            frozen_component_holdout_fingerprints or {}
        ),
    }
    return spec, schema, metadata_payload


def _resolve_study_sampler(
    purpose,
    schema: ComponentParameterSchema,
    *,
    max_trials: int,
) -> tuple[str, int | None]:
    combinations = _finite_grid_combinations(schema)
    grid_allowed = "grid" in purpose.method_ids
    if (
        grid_allowed
        and combinations is not None
        and len(schema.tunable_names) <= 3
        and combinations <= int(max_trials)
    ):
        return "grid", combinations
    if purpose.primary_method != "grid":
        return purpose.primary_method, combinations
    fallback = next(
        (
            method_id
            for method_id in purpose.method_ids
            if method_id not in {"grid", purpose.benchmark_method}
        ),
        None,
    )
    if fallback is None:
        raise ValueError(
            "The frozen trial budget cannot cover the full grid and no adaptive "
            "alternative is registered."
        )
    return fallback, combinations


def _finite_grid_combinations(
    schema: ComponentParameterSchema,
) -> int | None:
    combinations = 1
    for parameter in schema.parameters:
        if not parameter.tunable:
            continue
        if parameter.parameter_type == "bool":
            count = 2
        elif parameter.parameter_type == "categorical":
            count = len(parameter.choices)
        elif parameter.parameter_type in {"int", "float"}:
            if (
                parameter.low is None
                or parameter.high is None
                or parameter.step is None
                or parameter.log
            ):
                return None
            span = (float(parameter.high) - float(parameter.low)) / float(
                parameter.step
            )
            rounded = round(span)
            if not math.isclose(span, rounded, rel_tol=0.0, abs_tol=1e-9):
                return None
            count = int(rounded) + 1
        else:
            return None
        combinations *= count
    return combinations


def _load_selected_component_module(
    purpose_id: str,
    component_id: str,
) -> ModuleType:
    purpose = OptimizationMethodRegistry.load().resolve_purpose(purpose_id)
    if purpose.layer == "factor":
        return load_factor_module(component_id, include_public_examples=False)
    if purpose.layer == "router":
        return load_router_module(component_id)
    option = next(
        (
            value
            for value in load_component_options(purpose_id)
            if value.component_id == component_id
        ),
        None,
    )
    if option is None:
        raise KeyError(f"unknown {purpose.layer} component: {component_id}")
    return _load_module(
        REPO_ROOT / option.source_path,
        f"oqp_optimization_component_{component_id}",
    )


def _component_metadata_contract(
    component: ModuleType,
    layer: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    names = {
        "factor": ("FACTOR_METADATA", "FACTOR_CONTRACT"),
        "sleeve": ("SLEEVE_METADATA", "SLEEVE_CONTRACT"),
        "router": ("ROUTER_METADATA", "ROUTER_CONTRACT"),
        "overlay": ("OVERLAY_METADATA", "OVERLAY_CONTRACT"),
    }
    metadata_name, contract_name = names[layer]
    metadata = getattr(component, metadata_name, {}) or {}
    contract = getattr(component, contract_name, {}) or {}
    if not isinstance(metadata, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{layer} metadata and contract must be mappings")
    return metadata, contract


def _component_market_frequency(
    metadata: Mapping[str, Any],
    layer: str,
) -> tuple[str, str]:
    if layer == "factor":
        return (
            str(metadata.get("native_market") or "").strip(),
            str(metadata.get("data_frequency") or "").strip().lower(),
        )
    market = str(
        metadata.get("market_scope")
        or metadata.get("native_market")
        or ""
    ).strip()
    if market.lower() in {"agnostic", "any", "all"}:
        market = ""
    if not market:
        supported = tuple(metadata.get("supported_markets") or ())
        if len(supported) == 1:
            market = str(supported[0]).strip()
    frequency = str(
        metadata.get("frequency_scope")
        or metadata.get("frequency")
        or ""
    ).strip().lower()
    if frequency in {"agnostic", "any", "all"}:
        frequency = ""
    return market, frequency


def _validate_frozen_upstream_components(
    objective_profile,
    supplied: Mapping[str, str] | None,
) -> dict[str, str]:
    frozen = {
        str(component_id).strip(): str(fingerprint).strip()
        for component_id, fingerprint in dict(supplied or {}).items()
    }
    if any(not key or not value for key, value in frozen.items()):
        raise ValueError("frozen upstream component IDs and fingerprints are required")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in frozen.values()):
        raise ValueError("frozen upstream fingerprints must be SHA-256 values")
    requirements = tuple(objective_profile.upstream_requirements)
    if not requirements:
        if frozen:
            raise ValueError(
                f"{objective_profile.layer} optimisation does not accept upstream "
                "component fingerprints"
            )
        return {}
    accepted_any: set[str] = set()
    for requirement in requirements:
        matches = {
            component_id
            for component_id in frozen
            if component_id.startswith(requirement.accepted_prefixes)
        }
        if len(matches) < requirement.minimum_count:
            prefixes = ", ".join(requirement.accepted_prefixes)
            raise ValueError(
                f"{requirement.name} requires at least "
                f"{requirement.minimum_count} frozen component(s) with prefix "
                f"{prefixes}; received {len(matches)}"
            )
        accepted_any.update(matches)
    unexpected = sorted(set(frozen).difference(accepted_any))
    if unexpected:
        raise ValueError(
            "unexpected frozen upstream component(s): " + ", ".join(unexpected)
        )
    return dict(sorted(frozen.items()))


def _validate_frozen_upstream_context(
    frozen_components: Mapping[str, str],
    dataset_fingerprints: Mapping[str, str] | None,
    holdout_fingerprints: Mapping[str, str] | None,
    *,
    dataset_fingerprint: str,
    holdout_fingerprint: str,
) -> None:
    datasets = {
        str(key).strip(): str(value).strip()
        for key, value in dict(dataset_fingerprints or {}).items()
    }
    holdouts = {
        str(key).strip(): str(value).strip()
        for key, value in dict(holdout_fingerprints or {}).items()
    }
    expected_keys = set(frozen_components)
    if set(datasets) != expected_keys or set(holdouts) != expected_keys:
        if expected_keys:
            raise ValueError(
                "each frozen upstream component requires matching dataset and "
                "holdout fingerprints"
            )
        if datasets or holdouts:
            raise ValueError(
                "factor optimisation does not accept upstream context fingerprints"
            )
        return
    if any(value != dataset_fingerprint for value in datasets.values()):
        raise ValueError(
            "all frozen upstream components must use the selected development dataset"
        )
    if any(value != holdout_fingerprint for value in holdouts.values()):
        raise ValueError(
            "all frozen upstream components must preserve the same final holdout"
        )


def study_definition_payload(
    spec: Phase8ExperimentSpec,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "generation_metadata": dict(metadata),
        "optimization": spec.to_dict(),
    }


def write_frozen_study_definition(
    spec: Phase8ExperimentSpec,
    metadata: Mapping[str, Any],
    *,
    config_root: str | Path = DEFAULT_STUDY_CONFIG_ROOT,
) -> Path:
    root = Path(config_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", spec.study_id).strip("_")
    path = root / f"phase8_{safe_id}.yaml"
    payload = study_definition_payload(spec, metadata)
    text = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        existing_spec = Phase8ExperimentSpec.from_mapping(existing)
        if existing_spec.fingerprint != spec.fingerprint:
            raise FileExistsError(
                f"{path.name} already exists with a different frozen definition"
            )
        return path
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def default_study_id(component_id: str, frozen_on: str | date) -> str:
    stamp = pd.Timestamp(frozen_on).strftime("%Y%m%d")
    token = re.sub(r"[^a-z0-9]+", "_", component_id.lower()).strip("_")
    return f"phase8_{token}_{stamp}"


__all__ = [
    "DEFAULT_PHASE8_ARTIFACT_ROOT",
    "DEFAULT_STUDY_CONFIG_ROOT",
    "FrozenComponentOption",
    "OptimizationComponentOption",
    "OptimizationDatasetOption",
    "build_component_study_definition",
    "build_factor_study_definition",
    "default_study_id",
    "load_component_options",
    "load_dataset_options",
    "load_frozen_component_options",
    "parameter_schema_rows",
    "resolve_selected_component_schema",
    "study_definition_payload",
    "write_frozen_study_definition",
]
