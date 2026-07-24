"""Deterministic planning and execution for daily single-factor screens.

This module owns component matching, typed strategy configuration, manifests,
and batch isolation. Factor computation, sleeve construction, execution
simulation, and evaluation remain delegated to the canonical shared engines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from oqp.config import REPO_ROOT
from oqp.research.factors import (
    PRIVATE_FACTOR_ROOT,
    canonical_factor_id,
    load_factor_module,
)
from oqp.research.sleeves import (
    ExtractedSleeveConfig,
    PersistentSleeveConfig,
    SleeveConstructionConfig,
    iter_sleeve_files,
    load_sleeve_module,
    supports_extracted_sleeve_execution,
)
from oqp.research.strategy_composition import (
    StrategyAllocatorConfig,
    StrategyBranchConfig,
    StrategyBuilderConfig,
    StrategyCoreConfig,
    StrategyCoreType,
    StrategyExecutionConfig,
    write_strategy_builder_config,
)
from oqp.research_runtime import alpha_research_runtime_paths


DAILY_FACTOR_SCREEN_SCHEMA_VERSION = 1
DEFAULT_BATCH_ID = "daily_single_factor_screen_v1"
DEFAULT_MARKET_VERTICAL = "FUTURES_CN"
CANONICAL_FUTURES_DAILY_DATASET_ID = "FUTURES_DAILY_CORE"
CANONICAL_FUTURES_DAILY_DATA_PATH = (
    Path("runtime") / "data" / "feature_store" / "ML_Feature_Matrix.parquet"
)
CANONICAL_FUTURES_DAILY_PRODUCT_COUNT = 75

QUINTILE_SLEEVE_ID = "slv_001_Cross_Sectional_Quintile_Long_Short"
ZSCORE_SLEEVE_ID = "slv_004_Cross_Sectional_ZScore_Long_Short"
LONG_ONLY_EVENT_SLEEVE_ID = "slv_011_Sparse_Positive_Event_Long_Only_3D"
SIGNED_EVENT_SLEEVE_ID = "slv_012_Signed_Score_Inverse_Vol_5D_Staggered"
TIME_SERIES_OVERNIGHT_SLEEVE_ID = "slv_047_Daily_Time_Series_Sign_Overnight"
TIME_SERIES_SESSION_FLAT_SLEEVE_ID = (
    "slv_048_Daily_Time_Series_Sign_Session_Flat"
)

VALID_DIRECTIONAL_ORIENTATIONS = {"higher_is_bullish", "higher_is_bearish"}
INACTIVE_STATUS_MARKERS = ("archive", "retired", "disabled", "unavailable")
PURE_PORTFOLIO_LAYERS = {"alpha_score", "predictive_signal"}
SUPPORTED_CROSS_SECTIONAL_LAGS = {"already_lagged", "next_open"}
STANDARD_SLEEVE_RETURN_ASSUMPTION = "close_signal_next_open_to_close"

SPARSE_EVENT_MARKERS = (
    "absorption",
    "binary",
    "breakout",
    "capitulation",
    "event",
    "exhaustion",
    "liquidation",
    "reentry",
    "shock",
    "threshold",
    "touch",
    "turn",
)
CONTINUOUS_MARKERS = (
    "continuous",
    "standardized",
    "z_score",
    "zscore",
)
RANK_MARKERS = (
    "percentile",
    "quantile",
    "rank",
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        return ()
    return tuple(
        str(item).strip()
        for item in values
        if str(item).strip()
    )


def _slug(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or fallback


def _portable_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _component_short_name(component_id: str, declared_name: str = "") -> str:
    name = str(declared_name or "").strip()
    if name and name != component_id:
        return name
    stem = re.sub(r"^(?:fac|slv)_\d+_", "", component_id)
    return stem.replace("_", " ").strip() or component_id


@dataclass(frozen=True, slots=True)
class BaselineRiskAssumptions:
    """Frozen baseline required for comparable daily screens."""

    capital: float = 10_000_000.0
    capital_currency: str = "CNY"
    max_margin_utilization: float = 0.30
    minimum_cash_reserve: float = 0.70
    max_contract_weight: float = 0.10
    max_gross_leverage: float | None = None
    transaction_cost_profile: str = "cn_futures_broker_v1"
    slippage_ticks_per_side: float = 0.5
    risk_overlays: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if float(self.capital) <= 0.0:
            raise ValueError("screening capital must be positive")
        margin = float(self.max_margin_utilization)
        reserve = float(self.minimum_cash_reserve)
        if not 0.0 < margin <= 1.0:
            raise ValueError("max_margin_utilization must be in (0, 1]")
        if not 0.0 <= reserve < 1.0:
            raise ValueError("minimum_cash_reserve must be in [0, 1)")
        if abs((margin + reserve) - 1.0) > 1e-12:
            raise ValueError(
                "minimum_cash_reserve must equal one minus max_margin_utilization"
            )
        if not 0.0 < float(self.max_contract_weight) <= 1.0:
            raise ValueError("max_contract_weight must be in (0, 1]")
        if float(self.slippage_ticks_per_side) < 0.0:
            raise ValueError("slippage_ticks_per_side cannot be negative")
        if tuple(self.risk_overlays):
            raise ValueError("the baseline daily screen does not permit risk overlays")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_overlays"] = list(self.risk_overlays)
        # AlphaEvaluator.record_blocked_run uses this explicit audit key.
        payload["risk_overlay_ids"] = list(self.risk_overlays)
        payload["risk_overlay_policy"] = (
            "No risk overlays are permitted in the frozen baseline screen; "
            "factor+sleeve behavior is evaluated without regime, volatility, "
            "or portfolio-level overlays."
        )
        return payload


@dataclass(frozen=True, slots=True)
class DailyDatasetAssumption:
    dataset_id: str
    market_vertical: str
    data_frequency: str
    data_file: str
    available_columns: tuple[str, ...]
    row_count: int | None = None
    expected_product_count: int | None = None
    source: str = "explicit"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available_columns"] = list(self.available_columns)
        return payload


@dataclass(frozen=True, slots=True)
class DailyFactorDescriptor:
    factor_id: str
    name: str
    source_path: str
    status: str
    implementation_status: str
    factor_family: str
    factor_subfamily: str
    data_frequency: str
    signal_frequency: str
    portfolio_layer: str
    evaluation_geometry: str
    execution_lag: str
    return_assumption: str
    signal_orientation: str
    requires_shorting: bool | None
    required_fields: tuple[str, ...]
    supported_markets: tuple[str, ...]
    metadata: Mapping[str, Any]
    contract: Mapping[str, Any]

    @property
    def short_name(self) -> str:
        return _component_short_name(self.factor_id, self.name)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("required_fields", "supported_markets"):
            payload[key] = list(payload[key])
        payload["metadata"] = dict(self.metadata)
        payload["contract"] = dict(self.contract)
        return payload


@dataclass(frozen=True, slots=True)
class SleeveMatch:
    sleeve_id: str
    sleeve_name: str
    match_class: str
    match_reason: str
    execution_ready: bool
    blocker_code: str = ""
    blocker_reason: str = ""

    @property
    def short_name(self) -> str:
        return _component_short_name(self.sleeve_id, self.sleeve_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScreeningPlanItem:
    factor: DailyFactorDescriptor
    sleeve: SleeveMatch
    strategy_id: str
    strategy_name: str
    strategy_config: Mapping[str, Any]
    strategy_config_fingerprint: str
    status: str
    blocker_codes: tuple[str, ...] = ()
    blocker_reasons: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    missing_required_fields: tuple[str, ...] = ()
    config_path: str = ""

    @property
    def primary_blocker_code(self) -> str:
        return self.blocker_codes[0] if self.blocker_codes else ""

    @property
    def suggested_action(self) -> str:
        return " ".join(self.blocker_reasons).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor.to_dict(),
            "sleeve": self.sleeve.to_dict(),
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "strategy_config": dict(self.strategy_config),
            "strategy_config_fingerprint": self.strategy_config_fingerprint,
            "status": self.status,
            "blocker_codes": list(self.blocker_codes),
            "blocker_reasons": list(self.blocker_reasons),
            "caveats": list(self.caveats),
            "missing_required_fields": list(self.missing_required_fields),
            "config_path": self.config_path,
        }

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "batch_strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "factor_id": self.factor.factor_id,
            "factor_name": self.factor.name,
            "factor_status": self.factor.status,
            "factor_implementation_status": self.factor.implementation_status,
            "factor_family": self.factor.factor_family,
            "factor_subfamily": self.factor.factor_subfamily,
            "data_frequency": self.factor.data_frequency,
            "signal_frequency": self.factor.signal_frequency,
            "evaluation_geometry": self.factor.evaluation_geometry,
            "execution_lag": self.factor.execution_lag,
            "return_assumption": self.factor.return_assumption,
            "signal_orientation": self.factor.signal_orientation,
            "requires_shorting": self.factor.requires_shorting,
            "required_fields": json.dumps(
                list(self.factor.required_fields), ensure_ascii=False
            ),
            "missing_required_fields": json.dumps(
                list(self.missing_required_fields), ensure_ascii=False
            ),
            "sleeve_id": self.sleeve.sleeve_id,
            "sleeve_name": self.sleeve.sleeve_name,
            "match_class": self.sleeve.match_class,
            "match_reason": self.sleeve.match_reason,
            "status": self.status,
            "blocker_codes": json.dumps(
                list(self.blocker_codes), ensure_ascii=False
            ),
            "blocker_reasons": json.dumps(
                list(self.blocker_reasons), ensure_ascii=False
            ),
            "caveats": json.dumps(list(self.caveats), ensure_ascii=False),
            "strategy_config_fingerprint": self.strategy_config_fingerprint,
            "config_path": self.config_path,
            "factor_source": self.factor.source_path,
        }


@dataclass(frozen=True, slots=True)
class DailyFactorScreeningPlan:
    batch_id: str
    market_vertical: str
    dataset: DailyDatasetAssumption
    assumptions: BaselineRiskAssumptions
    items: tuple[ScreeningPlanItem, ...]
    schema_version: int = DAILY_FACTOR_SCREEN_SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        return str(self.to_dict()["fingerprint"])

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "planner_id": "daily_single_factor_screening_v1",
            "batch_id": self.batch_id,
            "market_vertical": self.market_vertical,
            "dataset": self.dataset.to_dict(),
            "assumptions": self.assumptions.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }
        fingerprint_payload = json.loads(
            json.dumps(payload, sort_keys=True, default=str)
        )
        for item in fingerprint_payload["items"]:
            item.pop("config_path", None)
        payload["fingerprint"] = _stable_hash(fingerprint_payload)
        return payload


@dataclass(frozen=True, slots=True)
class ScreeningExecutionResult:
    factor_id: str
    strategy_id: str
    status: str
    run_id: str = ""
    return_code: int | None = None
    output: str = ""
    failure_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataset_columns_and_rows(path: Path) -> tuple[tuple[str, ...], int | None]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            return tuple(parquet.schema_arrow.names), int(parquet.metadata.num_rows)
        except ImportError:
            frame = pd.read_parquet(path)
            return tuple(str(column) for column in frame.columns), len(frame)
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path, nrows=0)
        return tuple(str(column) for column in frame.columns), None
    raise ValueError("daily screening data must be CSV or parquet")


def resolve_daily_dataset(
    *,
    market_vertical: str = DEFAULT_MARKET_VERTICAL,
    data_file: str | Path | None = None,
    dataset_id: str | None = None,
    workspace_root: str | Path = REPO_ROOT,
) -> DailyDatasetAssumption:
    """Resolve and inspect the deterministic daily dataset for one market."""

    market = str(market_vertical).strip().upper()
    root = Path(workspace_root).expanduser().resolve()
    source = "explicit"
    expected_products: int | None = None
    if data_file is None:
        if market != "FUTURES_CN":
            raise ValueError(
                f"{market} has no canonical daily screening dataset; pass --data-file"
            )
        path = root / CANONICAL_FUTURES_DAILY_DATA_PATH
        resolved_dataset_id = (
            str(dataset_id).strip()
            if dataset_id
            else CANONICAL_FUTURES_DAILY_DATASET_ID
        )
        expected_products = CANONICAL_FUTURES_DAILY_PRODUCT_COUNT
        source = "canonical_market_default"
    else:
        path = Path(data_file).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved_dataset_id = (
            str(dataset_id).strip()
            if dataset_id
            else f"{market}_DAILY_{_slug(path.stem, fallback='dataset').upper()}"
        )
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"daily screening data file does not exist: {path}")
    columns, rows = _dataset_columns_and_rows(path)
    core_missing = sorted({"date", "ticker", "close"}.difference(columns))
    if core_missing:
        raise ValueError(
            "daily screening dataset is missing core columns: "
            + ", ".join(core_missing)
        )
    return DailyDatasetAssumption(
        dataset_id=resolved_dataset_id,
        market_vertical=market,
        data_frequency="daily",
        data_file=_portable_path(path, root),
        available_columns=tuple(sorted(columns)),
        row_count=rows,
        expected_product_count=expected_products,
        source=source,
    )


def _describe_factor(path: Path) -> DailyFactorDescriptor:
    module = load_factor_module(path.stem, include_public_examples=False)
    metadata = getattr(module, "FACTOR_METADATA", {}) or {}
    contract = getattr(module, "FACTOR_CONTRACT", {}) or {}
    if not isinstance(metadata, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{path.stem} must declare mapping metadata and contract")
    factor_id = str(getattr(module, "FACTOR_ID", path.stem)).strip()
    if factor_id != path.stem:
        raise ValueError(
            f"{path.stem} declares mismatched FACTOR_ID {factor_id!r}"
        )
    orientation = str(
        getattr(module, "SIGNAL_ORIENTATION", None)
        or metadata.get("signal_orientation")
        or ""
    ).strip().lower()
    requires_shorting_value = metadata.get("requires_shorting")
    requires_shorting = (
        bool(requires_shorting_value)
        if isinstance(requires_shorting_value, bool)
        else None
    )
    name = str(
        getattr(module, "FACTOR_NAME", None)
        or metadata.get("name")
        or factor_id
    ).strip()
    return DailyFactorDescriptor(
        factor_id=factor_id,
        name=name,
        source_path=_portable_path(path, Path(REPO_ROOT)),
        status=str(metadata.get("status") or "unclassified").strip().lower(),
        implementation_status=str(
            metadata.get("implementation_status") or ""
        ).strip().lower(),
        factor_family=str(metadata.get("factor_family") or "").strip().lower(),
        factor_subfamily=str(
            metadata.get("factor_subfamily") or ""
        ).strip().lower(),
        data_frequency=str(metadata.get("data_frequency") or "").strip().lower(),
        signal_frequency=str(metadata.get("signal_frequency") or "").strip().lower(),
        portfolio_layer=str(metadata.get("portfolio_layer") or "").strip().lower(),
        evaluation_geometry=str(
            contract.get("evaluation_geometry") or ""
        ).strip().lower(),
        execution_lag=str(contract.get("execution_lag") or "").strip().lower(),
        return_assumption=str(
            contract.get("return_assumption") or ""
        ).strip().lower(),
        signal_orientation=orientation,
        requires_shorting=requires_shorting,
        required_fields=_text_tuple(metadata.get("required_fields")),
        supported_markets=tuple(
            value.upper()
            for value in _text_tuple(
                contract.get("supported_markets")
                or metadata.get("supported_markets")
            )
        ),
        metadata=dict(metadata),
        contract=dict(contract),
    )


def _is_active_daily_factor(
    factor: DailyFactorDescriptor,
    market_vertical: str,
) -> bool:
    if any(marker in factor.status for marker in INACTIVE_STATUS_MARKERS):
        return False
    if factor.data_frequency != "daily":
        return False
    if factor.portfolio_layer not in PURE_PORTFOLIO_LAYERS:
        return False
    market = str(market_vertical).strip().upper()
    return market in factor.supported_markets or "*" in factor.supported_markets


def enumerate_active_daily_factors(
    *,
    market_vertical: str = DEFAULT_MARKET_VERTICAL,
    factor_ids: Sequence[str] | None = None,
) -> tuple[DailyFactorDescriptor, ...]:
    """Enumerate loadable active private daily factors in stable-ID order."""

    paths = {
        path.stem: path
        for path in sorted(PRIVATE_FACTOR_ROOT.glob("fac_*.py"))
    }
    requested: tuple[str, ...] | None = None
    if factor_ids:
        requested = tuple(
            dict.fromkeys(canonical_factor_id(value) for value in factor_ids)
        )
        missing = sorted(set(requested).difference(paths))
        if missing:
            raise ValueError(
                "unknown active private factor IDs: " + ", ".join(missing)
            )
        selected_paths = [paths[factor_id] for factor_id in requested]
    else:
        selected_paths = list(paths.values())

    factors: list[DailyFactorDescriptor] = []
    ineligible: list[str] = []
    for path in selected_paths:
        factor = _describe_factor(path)
        if _is_active_daily_factor(factor, market_vertical):
            factors.append(factor)
        elif requested is not None:
            ineligible.append(
                f"{factor.factor_id}"
                f"(status={factor.status or 'missing'}, "
                f"frequency={factor.data_frequency or 'missing'}, "
                f"markets={','.join(factor.supported_markets) or 'missing'})"
            )
    if ineligible:
        raise ValueError(
            "selected factors are not active daily factors for "
            f"{str(market_vertical).upper()}: "
            + "; ".join(ineligible)
        )
    return tuple(sorted(factors, key=lambda factor: factor.factor_id))


def _factor_matching_text(factor: DailyFactorDescriptor) -> str:
    fields = (
        factor.factor_family,
        factor.factor_subfamily,
        str(factor.metadata.get("signal_horizon") or ""),
        str(factor.metadata.get("deduplication_cohort") or ""),
        str(factor.metadata.get("signal_shape") or ""),
        str(factor.metadata.get("normalization_hint") or ""),
    )
    return " ".join(fields).lower()


def _matched_markers(
    text: str,
    markers: Sequence[str],
) -> tuple[str, ...]:
    """Match metadata tokens without treating ``turn`` as part of ``return``."""

    found: list[str] = []
    for marker in markers:
        pattern = r"[\s_-]+".join(
            re.escape(part) for part in marker.split("_")
        )
        if re.search(
            rf"(?<![a-z0-9]){pattern}(?![a-z0-9])",
            text,
            flags=re.IGNORECASE,
        ):
            found.append(marker)
    return tuple(found)


@lru_cache(maxsize=128)
def _sleeve_name(sleeve_id: str) -> str:
    try:
        module = load_sleeve_module(sleeve_id)
    except (ImportError, ModuleNotFoundError, ValueError):
        return _component_short_name(sleeve_id)
    metadata = getattr(module, "SLEEVE_METADATA", {}) or {}
    return str(metadata.get("name") or sleeve_id)


def _market_scope_matches(scope: Any, market_vertical: str) -> bool:
    market = str(market_vertical).strip().upper()
    scope_text = str(scope or "").strip().upper()
    if market and market in scope_text:
        return True
    values = {
        value.upper()
        for value in _text_tuple(scope)
    }
    if not values:
        text = str(scope or "").strip().upper()
        values = {text} if text else set()
    return (
        not values
        or "AGNOSTIC" in values
        or "*" in values
        or market in values
    )


@lru_cache(maxsize=16)
def _extracted_sleeve_index(
    market_vertical: str,
) -> Mapping[str, tuple[tuple[str, str], ...]]:
    matches: dict[str, list[tuple[str, str]]] = {}
    for path in iter_sleeve_files():
        try:
            module = load_sleeve_module(path.stem)
        except (ImportError, ValueError):
            continue
        metadata = getattr(module, "SLEEVE_METADATA", {}) or {}
        if not isinstance(metadata, Mapping):
            continue
        if "compatible_factor_ids" in metadata:
            compatible_factor_ids = set(
                _text_tuple(metadata.get("compatible_factor_ids"))
            )
        else:
            # Backward-compatible default for sleeves that predate the
            # provenance/compatibility split.
            compatible_factor_ids = set(
                _text_tuple(
                    metadata.get("source_factor_ids")
                    or metadata.get("source_factor_id")
                )
            )
        if not compatible_factor_ids:
            continue
        if not _market_scope_matches(
            metadata.get("market_scope"), market_vertical
        ):
            continue
        frequency = str(
            metadata.get("frequency_scope") or "agnostic"
        ).strip().lower()
        if frequency not in {"daily", "agnostic", "*"}:
            continue
        for factor_id in compatible_factor_ids:
            matches.setdefault(factor_id, []).append(
                (
                    path.stem,
                    str(metadata.get("name") or path.stem),
                )
            )
    return {
        factor_id: tuple(sorted(values))
        for factor_id, values in matches.items()
    }


def _factor_specific_extracted_sleeve(
    factor: DailyFactorDescriptor,
    *,
    market_vertical: str,
) -> tuple[str, str] | None:
    matches = _extracted_sleeve_index(
        str(market_vertical).strip().upper()
    ).get(factor.factor_id, ())
    return matches[0] if matches else None


def _resolved_factor_specific_sleeve_match(
    factor: DailyFactorDescriptor,
    *,
    sleeve_id: str,
    sleeve_name: str,
    market_vertical: str,
) -> SleeveMatch:
    """Resolve one factor-specific sleeve through its executable config.

    ``historical_source_factor_ids`` records provenance, while
    ``compatible_factor_ids`` declares executable factor matches. Legacy
    sleeves without the new field continue to use ``source_factor_ids`` as
    their compatibility default. The reusable runner accepts standard and
    persistent sleeve configs, subject to the same geometry and return-horizon
    checks applied here. Extracted configs are executable only when they opt
    into a canonical rule-family adapter exposed by the sleeve engine.
    """

    orientation = (
        factor.signal_orientation
        if factor.signal_orientation in VALID_DIRECTIONAL_ORIENTATIONS
        else "higher_is_bullish"
    )
    base_reason = (
        f"{sleeve_id} explicitly lists {factor.factor_id} in "
        "compatible_factor_ids (or the legacy source-factor fallback) and "
        "preserves its governed score-to-position semantics."
    )
    try:
        module = load_sleeve_module(sleeve_id)
        config = module.build_config(
            factor.factor_id,
            market_vertical=str(market_vertical).strip().upper(),
            signal_orientation=orientation,
        )
    except (ImportError, TypeError, ValueError) as exc:
        return SleeveMatch(
            sleeve_id=sleeve_id,
            sleeve_name=sleeve_name,
            match_class="factor_specific_semantic_sleeve",
            match_reason=base_reason,
            execution_ready=False,
            blocker_code="factor_specific_sleeve_config_invalid",
            blocker_reason=(
                f"{sleeve_id} build_config() could not resolve an executable "
                f"config for {factor.factor_id}: {exc}"
            ),
        )

    if isinstance(config, SleeveConstructionConfig):
        if config.construction_geometry != factor.evaluation_geometry:
            return SleeveMatch(
                sleeve_id=sleeve_id,
                sleeve_name=sleeve_name,
                match_class="factor_specific_semantic_sleeve",
                match_reason=base_reason,
                execution_ready=False,
                blocker_code="factor_specific_sleeve_geometry_mismatch",
                blocker_reason=(
                    "Factor and sleeve evaluation geometries differ: "
                    f"{factor.evaluation_geometry or 'missing'} != "
                    f"{config.construction_geometry}."
                ),
            )
        if config.return_assumption != factor.return_assumption:
            return SleeveMatch(
                sleeve_id=sleeve_id,
                sleeve_name=sleeve_name,
                match_class="factor_specific_semantic_sleeve",
                match_reason=base_reason,
                execution_ready=False,
                blocker_code="factor_specific_sleeve_return_assumption_mismatch",
                blocker_reason=(
                    "Factor and sleeve return assumptions differ: "
                    f"{factor.return_assumption or 'missing'} != "
                    f"{config.return_assumption}."
                ),
            )
        return SleeveMatch(
            sleeve_id=sleeve_id,
            sleeve_name=sleeve_name,
            match_class="factor_specific_semantic_sleeve",
            match_reason=(
                f"{base_reason} build_config() resolves to "
                "SleeveConstructionConfig with compatible geometry and return "
                "horizon, so the canonical reusable-sleeve runner can execute it."
            ),
            execution_ready=True,
        )

    if isinstance(config, PersistentSleeveConfig):
        if factor.evaluation_geometry != "cross_sectional":
            return SleeveMatch(
                sleeve_id=sleeve_id,
                sleeve_name=sleeve_name,
                match_class="factor_specific_semantic_sleeve",
                match_reason=base_reason,
                execution_ready=False,
                blocker_code="factor_specific_sleeve_geometry_mismatch",
                blocker_reason=(
                    f"{sleeve_id} persistent construction requires a "
                    "cross-sectional factor, but "
                    f"{factor.factor_id} declares "
                    f"{factor.evaluation_geometry or 'missing'}."
                ),
            )
        return SleeveMatch(
            sleeve_id=sleeve_id,
            sleeve_name=sleeve_name,
            match_class="factor_specific_semantic_sleeve",
            match_reason=(
                f"{base_reason} build_config() resolves to "
                "PersistentSleeveConfig for a cross-sectional factor, so the "
                "canonical reusable-sleeve runner can execute it."
            ),
            execution_ready=True,
        )

    if isinstance(config, ExtractedSleeveConfig):
        if supports_extracted_sleeve_execution(config):
            construction_geometry = str(
                (config.parameters or {}).get("construction_geometry") or ""
            ).strip().lower()
            required_geometry = {
                "time_series_stateful": "time_series",
                "cross_sectional": "cross_sectional",
            }.get(construction_geometry)
            if required_geometry != factor.evaluation_geometry:
                return SleeveMatch(
                    sleeve_id=sleeve_id,
                    sleeve_name=sleeve_name,
                    match_class="factor_specific_semantic_sleeve",
                    match_reason=base_reason,
                    execution_ready=False,
                    blocker_code="factor_specific_sleeve_geometry_mismatch",
                    blocker_reason=(
                        "Factor and extracted sleeve evaluation geometries "
                        "differ: "
                        f"{factor.evaluation_geometry or 'missing'} != "
                        f"{construction_geometry or 'missing'}."
                    ),
                )
            return SleeveMatch(
                sleeve_id=sleeve_id,
                sleeve_name=sleeve_name,
                match_class="factor_specific_semantic_sleeve",
                match_reason=(
                    f"{base_reason} build_config() opts into the registered "
                    f"{config.rule_family} execution adapter with compatible "
                    "geometry. The return horizon is inherited from and "
                    "validated against the factor contract."
                ),
                execution_ready=True,
            )
        support = (
            "declares execution_supported=False"
            if not config.execution_supported
            else "does not have a registered execution adapter"
        )
        return SleeveMatch(
            sleeve_id=sleeve_id,
            sleeve_name=sleeve_name,
            match_class="factor_specific_semantic_sleeve",
            match_reason=base_reason,
            execution_ready=False,
            blocker_code="factor_specific_sleeve_execution_unsupported",
            blocker_reason=(
                f"{sleeve_id} build_config() resolves to ExtractedSleeveConfig "
                f"and {support}. Do not replace it silently with a generic "
                "sleeve."
            ),
        )

    return SleeveMatch(
        sleeve_id=sleeve_id,
        sleeve_name=sleeve_name,
        match_class="factor_specific_semantic_sleeve",
        match_reason=base_reason,
        execution_ready=False,
        blocker_code="factor_specific_sleeve_config_type_unsupported",
        blocker_reason=(
            f"{sleeve_id} build_config() returned unsupported config type "
            f"{type(config).__name__}; the canonical reusable-sleeve runner "
            "accepts SleeveConstructionConfig or PersistentSleeveConfig."
        ),
    )


def _planned_time_series_sleeve(
    factor: DailyFactorDescriptor,
    *,
    market_vertical: str,
) -> SleeveMatch:
    specific = _factor_specific_extracted_sleeve(
        factor,
        market_vertical=market_vertical,
    )
    if specific is not None:
        specific_id, name = specific
        return _resolved_factor_specific_sleeve_match(
            factor,
            sleeve_id=specific_id,
            sleeve_name=name,
            market_vertical=market_vertical,
        )
    if (
        factor.return_assumption
        == "close_signal_next_open_to_next_open"
    ):
        sleeve_id = TIME_SERIES_OVERNIGHT_SLEEVE_ID
        timing = "overnight next-open to next-open"
    elif factor.return_assumption == STANDARD_SLEEVE_RETURN_ASSUMPTION:
        sleeve_id = TIME_SERIES_SESSION_FLAT_SLEEVE_ID
        timing = "session-flat next-open to close"
    else:
        return SleeveMatch(
            sleeve_id=TIME_SERIES_SESSION_FLAT_SLEEVE_ID,
            sleeve_name=_sleeve_name(TIME_SERIES_SESSION_FLAT_SLEEVE_ID),
            match_class="unsupported_time_series_horizon",
            match_reason=(
                f"return_assumption={factor.return_assumption or 'missing'} "
                "has no registered executable generic daily sign sleeve."
            ),
            execution_ready=False,
            blocker_code="unsupported_time_series_return_assumption",
            blocker_reason=(
                "Register an executable time-series sleeve with exactly the "
                "factor's declared return assumption before screening it."
            ),
        )
    return SleeveMatch(
        sleeve_id=sleeve_id,
        sleeve_name=_sleeve_name(sleeve_id),
        match_class="generic_time_series_baseline",
        match_reason=(
            f"evaluation_geometry=time_series and return_assumption="
            f"{factor.return_assumption or 'missing'} select the executable "
            f"{timing} sign sleeve."
        ),
        execution_ready=True,
    )


def match_factor_to_sleeve(
    factor: DailyFactorDescriptor,
    *,
    market_vertical: str = DEFAULT_MARKET_VERTICAL,
) -> SleeveMatch:
    """Select one sleeve deterministically from factor contract and metadata."""

    geometry = factor.evaluation_geometry
    if geometry == "time_series":
        return _planned_time_series_sleeve(
            factor,
            market_vertical=market_vertical,
        )
    if geometry != "cross_sectional":
        return SleeveMatch(
            sleeve_id=QUINTILE_SLEEVE_ID,
            sleeve_name=_sleeve_name(QUINTILE_SLEEVE_ID),
            match_class="unsupported_geometry",
            match_reason=(
                f"evaluation_geometry={geometry or 'missing'} has no governed "
                "daily single-factor matching rule."
            ),
            execution_ready=False,
            blocker_code="unsupported_factor_geometry",
            blocker_reason=(
                "Declare cross_sectional or time_series geometry and provide "
                "a compatible executable sleeve."
            ),
        )

    specific = _factor_specific_extracted_sleeve(
        factor,
        market_vertical=market_vertical,
    )
    if specific is not None:
        sleeve_id, name = specific
        return _resolved_factor_specific_sleeve_match(
            factor,
            sleeve_id=sleeve_id,
            sleeve_name=name,
            market_vertical=market_vertical,
        )

    text = _factor_matching_text(factor)
    sparse_markers = _matched_markers(text, SPARSE_EVENT_MARKERS)
    continuous_markers = _matched_markers(text, CONTINUOUS_MARKERS)
    rank_markers = _matched_markers(text, RANK_MARKERS)

    if factor.requires_shorting is False and sparse_markers:
        sleeve_id = LONG_ONLY_EVENT_SLEEVE_ID
        match_class = "long_only_sparse_event"
        reason = (
            "metadata requires_shorting=false and sparse/event markers "
            f"{', '.join(sparse_markers)} select the 3-day positive-event "
            "long-only sleeve."
        )
    elif sparse_markers:
        sleeve_id = SIGNED_EVENT_SLEEVE_ID
        match_class = "signed_sparse_event"
        reason = (
            "cross-sectional signed factor with sparse/event markers "
            f"{', '.join(sparse_markers)} selects the persistent signed-score "
            "inverse-volatility sleeve."
        )
    elif continuous_markers:
        sleeve_id = ZSCORE_SLEEVE_ID
        match_class = "continuous_score"
        reason = (
            "continuous-score metadata markers "
            f"{', '.join(continuous_markers)} select the standard "
            "cross-sectional z-score sleeve."
        )
    elif rank_markers:
        sleeve_id = QUINTILE_SLEEVE_ID
        match_class = "rank_score"
        reason = (
            "rank/quantile metadata markers "
            f"{', '.join(rank_markers)} select the standard outer-quintile "
            "long-short sleeve."
        )
    else:
        sleeve_id = QUINTILE_SLEEVE_ID
        match_class = "default_cross_sectional_score"
        reason = (
            "cross-sectional factor has no declared sparse-event or continuous-"
            "zscore marker; the frozen standard is the simple outer-quintile "
            "long-short sleeve."
        )
    return SleeveMatch(
        sleeve_id=sleeve_id,
        sleeve_name=_sleeve_name(sleeve_id),
        match_class=match_class,
        match_reason=reason,
        execution_ready=True,
    )


def _strategy_id(factor_id: str, batch_id: str) -> str:
    return (
        "str_daily_screen_"
        f"{_slug(batch_id, fallback='batch')}_"
        f"{_slug(factor_id, fallback='factor')}"
    )


def build_typed_strategy_config(
    factor: DailyFactorDescriptor,
    sleeve: SleeveMatch,
    *,
    batch_id: str,
    market_vertical: str,
    assumptions: BaselineRiskAssumptions,
) -> StrategyBuilderConfig:
    """Build the canonical typed one-factor + one-sleeve strategy config."""

    strategy_name = (
        f"{factor.short_name} × {sleeve.short_name} [{batch_id}]"
    )
    return StrategyBuilderConfig(
        strategy_id=_strategy_id(factor.factor_id, batch_id),
        name=strategy_name,
        market_vertical=str(market_vertical).strip().upper(),
        core=StrategyCoreConfig(
            core_type=StrategyCoreType.FACTOR_SLEEVE,
            branches=(
                StrategyBranchConfig(
                    branch_id="core",
                    factor_ids=(factor.factor_id,),
                    sleeve_id=sleeve.sleeve_id,
                    execution_mode="risk_desk",
                ),
            ),
        ),
        risk_overlays=assumptions.risk_overlays,
        allocator=StrategyAllocatorConfig(
            max_gross_leverage=assumptions.max_gross_leverage,
            max_contract_weight=assumptions.max_contract_weight,
            max_margin_utilization=assumptions.max_margin_utilization,
        ),
        execution=StrategyExecutionConfig(
            capital=assumptions.capital,
            capital_currency=assumptions.capital_currency,
            transaction_cost_profile=assumptions.transaction_cost_profile,
            slippage_ticks_per_side=assumptions.slippage_ticks_per_side,
        ),
        research_mode="exploratory",
    )


def _preflight_blockers(
    factor: DailyFactorDescriptor,
    sleeve: SleeveMatch,
    dataset: DailyDatasetAssumption,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    codes: list[str] = []
    reasons: list[str] = []
    caveats: list[str] = []

    if not sleeve.execution_ready:
        codes.append(sleeve.blocker_code or "sleeve_not_executable")
        reasons.append(sleeve.blocker_reason or sleeve.match_reason)
    implementation_status = factor.implementation_status
    if (
        implementation_status.startswith("blocked_")
        or implementation_status == "implemented_requires_benchmark_adapter"
    ):
        codes.append(implementation_status)
        reasons.append(
            f"{factor.factor_id} declares FACTOR_METADATA.implementation_status="
            f"{implementation_status!r}; resolve that exact implementation gate "
            "before execution."
        )
    elif implementation_status == "implemented_data_provisional":
        caveats.append(
            "FACTOR_METADATA.implementation_status=implemented_data_provisional; "
            "results are exploratory until the provisional data semantics are "
            "replaced or attested."
        )
    if factor.signal_orientation not in VALID_DIRECTIONAL_ORIENTATIONS:
        codes.append("missing_or_unsupported_signal_orientation")
        reasons.append(
            f"{factor.factor_id} must declare SIGNAL_ORIENTATION as "
            "higher_is_bullish or higher_is_bearish before entering a sleeve."
        )
    if (
        factor.evaluation_geometry == "cross_sectional"
        and factor.execution_lag not in SUPPORTED_CROSS_SECTIONAL_LAGS
    ):
        codes.append("unsupported_execution_lag")
        reasons.append(
            f"Cross-sectional factor execution_lag={factor.execution_lag or 'missing'} "
            "is not supported by the canonical reusable-sleeve path."
        )
    if (
        sleeve.sleeve_id
        in {
            QUINTILE_SLEEVE_ID,
            ZSCORE_SLEEVE_ID,
            LONG_ONLY_EVENT_SLEEVE_ID,
            SIGNED_EVENT_SLEEVE_ID,
        }
        and factor.return_assumption != STANDARD_SLEEVE_RETURN_ASSUMPTION
    ):
        codes.append("sleeve_return_assumption_mismatch")
        reasons.append(
            f"Factor return_assumption={factor.return_assumption or 'missing'} "
            f"does not equal sleeve return assumption "
            f"{STANDARD_SLEEVE_RETURN_ASSUMPTION}."
        )

    available = set(dataset.available_columns)
    missing = tuple(
        sorted(set(factor.required_fields).difference(available))
    )
    if missing:
        codes.append("dataset_missing_required_fields")
        reasons.append(
            f"{dataset.dataset_id} lacks required factor fields: "
            + ", ".join(missing)
            + "."
        )
    if factor.requires_shorting is None:
        caveats.append(
            f"{factor.factor_id} does not declare boolean requires_shorting; "
            "the planner defaults it to the standard long-short score sleeve."
        )
    matching_text = _factor_matching_text(factor)
    if (
        sleeve.sleeve_id == QUINTILE_SLEEVE_ID
        and "sector" in matching_text
    ):
        caveats.append(
            "The factor score is sector-relative, but the generic quintile "
            "sleeve does not enforce portfolio-level sector neutrality; treat "
            "this as a factor screen rather than a sector-neutral strategy claim."
        )

    # Preserve deterministic first occurrence when two checks share a code.
    deduped_codes: list[str] = []
    deduped_reasons: list[str] = []
    for code, reason in zip(codes, reasons, strict=True):
        if code in deduped_codes:
            continue
        deduped_codes.append(code)
        deduped_reasons.append(reason)
    return (
        tuple(deduped_codes),
        tuple(deduped_reasons),
        missing,
        tuple(dict.fromkeys(caveats)),
    )


def build_daily_factor_screening_plan(
    *,
    batch_id: str = DEFAULT_BATCH_ID,
    market_vertical: str = DEFAULT_MARKET_VERTICAL,
    factor_ids: Sequence[str] | None = None,
    data_file: str | Path | None = None,
    dataset_id: str | None = None,
    assumptions: BaselineRiskAssumptions | None = None,
    workspace_root: str | Path = REPO_ROOT,
) -> DailyFactorScreeningPlan:
    """Build a deterministic plan without computing factor values."""

    resolved_batch = str(batch_id).strip()
    if not resolved_batch:
        raise ValueError("batch_id cannot be empty")
    market = str(market_vertical).strip().upper()
    baseline = assumptions or BaselineRiskAssumptions()
    dataset = resolve_daily_dataset(
        market_vertical=market,
        data_file=data_file,
        dataset_id=dataset_id,
        workspace_root=workspace_root,
    )
    factors = enumerate_active_daily_factors(
        market_vertical=market,
        factor_ids=factor_ids,
    )
    items: list[ScreeningPlanItem] = []
    for factor in factors:
        sleeve = match_factor_to_sleeve(
            factor,
            market_vertical=market,
        )
        config = build_typed_strategy_config(
            factor,
            sleeve,
            batch_id=resolved_batch,
            market_vertical=market,
            assumptions=baseline,
        )
        codes, reasons, missing, caveats = _preflight_blockers(
            factor,
            sleeve,
            dataset,
        )
        items.append(
            ScreeningPlanItem(
                factor=factor,
                sleeve=sleeve,
                strategy_id=config.strategy_id,
                strategy_name=config.name,
                strategy_config=config.to_dict(),
                strategy_config_fingerprint=config.fingerprint,
                status="blocked" if codes else "ready",
                blocker_codes=codes,
                blocker_reasons=reasons,
                caveats=caveats,
                missing_required_fields=missing,
            )
        )
    return DailyFactorScreeningPlan(
        batch_id=resolved_batch,
        market_vertical=market,
        dataset=dataset,
        assumptions=baseline,
        items=tuple(items),
    )


def _config_from_item(item: ScreeningPlanItem) -> StrategyBuilderConfig:
    factor = item.factor
    sleeve = item.sleeve
    payload = item.strategy_config
    allocator = payload["allocator"]
    execution = payload["execution"]
    return StrategyBuilderConfig(
        strategy_id=item.strategy_id,
        name=item.strategy_name,
        market_vertical=str(payload["market_vertical"]),
        core=StrategyCoreConfig(
            core_type=StrategyCoreType.FACTOR_SLEEVE,
            branches=(
                StrategyBranchConfig(
                    branch_id="core",
                    factor_ids=(factor.factor_id,),
                    sleeve_id=sleeve.sleeve_id,
                    execution_mode="risk_desk",
                ),
            ),
        ),
        risk_overlays=tuple(payload.get("risk_overlays") or ()),
        allocator=StrategyAllocatorConfig(**dict(allocator)),
        execution=StrategyExecutionConfig(**dict(execution)),
        research_mode=str(payload.get("research_mode") or "exploratory"),
        schema_version=int(payload.get("schema_version") or 1),
    )


def write_strategy_configs(
    plan: DailyFactorScreeningPlan,
    config_dir: str | Path,
    *,
    workspace_root: str | Path = REPO_ROOT,
) -> DailyFactorScreeningPlan:
    """Write one typed YAML per plan item, including auditable blocked plans."""

    destination = Path(config_dir).expanduser()
    if not destination.is_absolute():
        destination = Path(workspace_root) / destination
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(workspace_root).expanduser().resolve()
    updated: list[ScreeningPlanItem] = []
    for item in plan.items:
        path = destination / f"{item.strategy_id}.yaml"
        write_strategy_builder_config(_config_from_item(item), path)
        updated.append(
            replace(item, config_path=_portable_path(path, root))
        )
    return replace(plan, items=tuple(updated))


def write_screening_manifest(
    plan: DailyFactorScreeningPlan,
    output_dir: str | Path,
    *,
    workspace_root: str | Path = REPO_ROOT,
) -> tuple[Path, Path]:
    """Write deterministic JSON and flat CSV planning manifests."""

    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = Path(workspace_root) / destination
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "screening_manifest.json"
    csv_path = destination / "screening_manifest.csv"
    json_path.write_text(
        json.dumps(
            plan.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [item.to_manifest_row() for item in plan.items]
    fieldnames = list(rows[0]) if rows else [
        "batch_strategy_id",
        "strategy_name",
        "factor_id",
        "sleeve_id",
        "match_reason",
        "status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _default_evaluator(plan: DailyFactorScreeningPlan):
    from oqp.research.backtesting import AlphaEvaluator

    runtime = alpha_research_runtime_paths()
    return AlphaEvaluator(
        db_path=runtime.db_path,
        logs_dir=runtime.artifact_root,
        asset_class=plan.market_vertical,
    )


def record_blocked_plan_item(
    plan: DailyFactorScreeningPlan,
    item: ScreeningPlanItem,
    *,
    evaluator: Any | None = None,
    failure_code: str | None = None,
    suggested_action: str | None = None,
) -> str:
    """Record a preflight block through the evaluator's public hook."""

    resolved_evaluator = evaluator or _default_evaluator(plan)
    recorder = getattr(resolved_evaluator, "record_blocked_run", None)
    if not callable(recorder):
        raise RuntimeError(
            "Canonical evaluator does not expose record_blocked_run(); "
            f"cannot persist blocked preflight for {item.factor.factor_id}."
        )
    return str(
        recorder(
            strategy_id=item.strategy_id,
            strategy_name=item.strategy_name,
            component_factor_id=item.factor.factor_id,
            sleeve_id=item.sleeve.sleeve_id,
            failure_code=(
                failure_code
                or item.primary_blocker_code
                or "strategy_preflight_blocked"
            ),
            suggested_action=(
                suggested_action
                or item.suggested_action
                or "Resolve the recorded preflight failure before retrying."
            ),
            strategy_config=dict(item.strategy_config),
            strategy_config_fingerprint=item.strategy_config_fingerprint,
            batch_id=plan.batch_id,
            dataset_metadata=plan.dataset.to_dict(),
            assumptions=plan.assumptions.to_dict(),
            research_family=(
                f"daily_single_factor_screen::{item.factor.factor_id}"
            ),
        )
    )


def _command_for_item(
    item: ScreeningPlanItem,
    plan: DailyFactorScreeningPlan,
    *,
    python_executable: str,
    build_only: bool,
    start_date: str | None = None,
    end_date: str | None = None,
    split_date: str | None = None,
) -> list[str]:
    command = [
        python_executable,
        "scripts/research/run_strategy_backtest.py",
        "--config",
        item.config_path,
        "--data-file",
        plan.dataset.data_file,
    ]
    if build_only:
        command.append("--build-only")
    if start_date:
        command.extend(["--start-date", str(start_date)])
    if end_date:
        command.extend(["--end-date", str(end_date)])
    if split_date and not build_only:
        command.extend(["--split-date", str(split_date)])
    return command


def _process_output(completed: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        value.strip()
        for value in (completed.stdout, completed.stderr)
        if value and value.strip()
    )


def _run_id_from_output(output: str) -> str:
    matches = re.findall(r"\brun_[a-zA-Z0-9]+\b", output)
    return matches[-1] if matches else ""


def _active_target_summary(frame: pd.DataFrame) -> tuple[bool, str]:
    target_col = next(
        (
            column
            for column in (
                "final_target_weight",
                "routed_target_weight",
                "target_weight",
                "signal",
            )
            if column in frame.columns
        ),
        None,
    )
    if target_col is None:
        return False, "strategy output has no recognized target-weight column"
    targets = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0)
    active = targets.ne(0.0)
    if not bool(active.any()):
        return False, f"{target_col} contains no non-zero target"
    active_dates = int(frame.loc[active, "date"].nunique())
    return (
        True,
        f"{int(active.sum())} active rows across {active_dates} dates "
        f"from {target_col}",
    )


def _record_runtime_failure(
    plan: DailyFactorScreeningPlan,
    item: ScreeningPlanItem,
    *,
    evaluator: Any | None,
    failure_code: str,
    error: Exception | str,
) -> ScreeningExecutionResult:
    resolved_evaluator = evaluator or _default_evaluator(plan)
    message = str(error).strip() or type(error).__name__
    run_id = record_blocked_plan_item(
        plan,
        item,
        evaluator=resolved_evaluator,
        failure_code=failure_code,
        suggested_action=(
            f"{message} Inspect the factor contract, matching sleeve, dataset "
            "fields, and captured runtime error before retrying."
        ),
    )
    return ScreeningExecutionResult(
        factor_id=item.factor.factor_id,
        strategy_id=item.strategy_id,
        status="blocked_recorded",
        run_id=run_id,
        return_code=1,
        output=message,
        failure_code=failure_code,
    )


def execute_screening_plan(
    plan: DailyFactorScreeningPlan,
    *,
    evaluator: Any | None = None,
    workspace_root: str | Path = REPO_ROOT,
    start_date: str | None = None,
    end_date: str | None = None,
    split_date: str | None = None,
) -> tuple[ScreeningExecutionResult, ...]:
    """Run a plan in-process while sharing one prepared daily data load.

    The function uses the same ``FactorPortfolioRunner.build_with_sleeve`` and
    ``evaluate`` path as the canonical strategy command.  Only return-horizon
    views are copied per contract; data loading, aliases, capital, transaction
    costs, and liquidity eligibility are prepared once for the batch.
    """

    from oqp.commands.strategy_backtest import _factor_portfolio_config
    from oqp.execution.transaction_costs import attach_transaction_cost_policy
    from oqp.research.backtesting import (
        attach_capital_attrs,
        attach_return_horizon,
        resolve_execution_capital,
    )
    from oqp.research.factor_portfolios import (
        FactorPortfolioRunner,
        load_factor_portfolio_data,
    )
    from oqp.research.liquidity_eligibility import ensure_liquidity_eligibility

    root = Path(workspace_root).expanduser().resolve()
    for item in plan.items:
        if not item.config_path:
            raise ValueError(
                f"{item.factor.factor_id} has no config_path; call "
                "write_strategy_configs() before execution"
            )

    results_by_strategy: dict[str, ScreeningExecutionResult] = {}
    resolved_evaluator = evaluator
    ready_items: list[ScreeningPlanItem] = []
    for item in plan.items:
        if item.status != "blocked":
            ready_items.append(item)
            continue
        resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
        run_id = record_blocked_plan_item(
            plan,
            item,
            evaluator=resolved_evaluator,
        )
        results_by_strategy[item.strategy_id] = ScreeningExecutionResult(
            factor_id=item.factor.factor_id,
            strategy_id=item.strategy_id,
            status="blocked_recorded",
            run_id=run_id,
            failure_code=item.primary_blocker_code,
        )
    if not ready_items:
        return tuple(results_by_strategy[item.strategy_id] for item in plan.items)

    data_path = Path(plan.dataset.data_file).expanduser()
    if not data_path.is_absolute():
        data_path = root / data_path
    try:
        data = load_factor_portfolio_data(
            data_path,
            market_vertical=plan.market_vertical,
            return_horizon=STANDARD_SLEEVE_RETURN_ASSUMPTION,
            start_date=start_date,
            end_date=end_date,
            dataset_id=plan.dataset.dataset_id,
            workspace_root=root,
        )
        shared_frame = data.frame
        if "open_interest" not in shared_frame.columns:
            if "open_oi" in shared_frame.columns:
                shared_frame = shared_frame.copy()
                shared_frame.attrs.update(data.frame.attrs)
                shared_frame["open_interest"] = pd.to_numeric(
                    shared_frame["open_oi"], errors="coerce"
                )
            elif "oi" in shared_frame.columns:
                shared_frame = shared_frame.copy()
                shared_frame.attrs.update(data.frame.attrs)
                shared_frame["open_interest"] = pd.to_numeric(
                    shared_frame["oi"], errors="coerce"
                )
        shared_frame = attach_capital_attrs(
            shared_frame,
            resolve_execution_capital(
                asset_class=plan.market_vertical,
                initial_capital=plan.assumptions.capital,
                capital_currency=plan.assumptions.capital_currency,
                capital_profile="strategy_builder",
            ),
        )
        shared_frame = attach_transaction_cost_policy(
            shared_frame,
            market_vertical=plan.market_vertical,
            profile_id=plan.assumptions.transaction_cost_profile,
            use_case="research_net",
        )
        shared_frame.attrs["max_weight_per_asset"] = (
            plan.assumptions.max_contract_weight
        )
        shared_frame = ensure_liquidity_eligibility(
            shared_frame,
            market_vertical=plan.market_vertical,
            initial_capital=plan.assumptions.capital,
            capital_currency=plan.assumptions.capital_currency,
            max_position_weight=plan.assumptions.max_contract_weight,
        )
    except Exception as exc:
        resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
        for item in ready_items:
            results_by_strategy[item.strategy_id] = _record_runtime_failure(
                plan,
                item,
                evaluator=resolved_evaluator,
                failure_code="screening_shared_data_preparation_failed",
                error=exc,
            )
        return tuple(results_by_strategy[item.strategy_id] for item in plan.items)

    horizon_frames: dict[str, pd.DataFrame] = {
        STANDARD_SLEEVE_RETURN_ASSUMPTION: shared_frame
    }
    runtime = alpha_research_runtime_paths()
    for item in ready_items:
        horizon = item.factor.return_assumption
        try:
            if horizon not in horizon_frames:
                horizon_frames[horizon] = attach_return_horizon(
                    shared_frame,
                    return_horizon=horizon,
                    data_frequency=data.data_frequency,
                )
            config = _config_from_item(item)
            portfolio_config = _factor_portfolio_config(config)
            runner = FactorPortfolioRunner(portfolio_config)
            result = runner.build_with_sleeve(
                horizon_frames[horizon],
                factor_id=item.factor.factor_id,
                sleeve_id=item.sleeve.sleeve_id,
                strict_factor_contracts=True,
            )
            result.frame.attrs.update(
                {
                    "component_type": "strategy",
                    "strategy_id": config.strategy_id,
                    "strategy_core_type": config.core.core_type.value,
                    "strategy_config_fingerprint": config.fingerprint,
                    "screening_batch_id": plan.batch_id,
                    "screening_plan_fingerprint": plan.fingerprint,
                    "backtest_engine": "typed_strategy_composition",
                    "runner": "daily_factor_screening",
                }
            )
            has_targets, target_summary = _active_target_summary(result.frame)
            if not has_targets:
                raise RuntimeError(target_summary)
        except Exception as exc:
            resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
            results_by_strategy[item.strategy_id] = _record_runtime_failure(
                plan,
                item,
                evaluator=resolved_evaluator,
                failure_code="strategy_preflight_failed",
                error=exc,
            )
            continue

        try:
            run_id = runner.evaluate(
                result,
                db_path=runtime.db_path,
                logs_dir=runtime.artifact_root,
                crisis_period=data.crisis_period,
                split_date=split_date,
            )
            results_by_strategy[item.strategy_id] = ScreeningExecutionResult(
                factor_id=item.factor.factor_id,
                strategy_id=item.strategy_id,
                status="completed",
                run_id=str(run_id),
                return_code=0,
                output=target_summary,
            )
        except Exception as exc:
            resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
            results_by_strategy[item.strategy_id] = _record_runtime_failure(
                plan,
                item,
                evaluator=resolved_evaluator,
                failure_code="strategy_backtest_failed_after_preflight",
                error=exc,
            )
    return tuple(results_by_strategy[item.strategy_id] for item in plan.items)


def execute_screening_plan_subprocess(
    plan: DailyFactorScreeningPlan,
    *,
    evaluator: Any | None = None,
    python_executable: str = sys.executable,
    workspace_root: str | Path = REPO_ROOT,
    timeout_seconds: int = 3_600,
    start_date: str | None = None,
    end_date: str | None = None,
    split_date: str | None = None,
) -> tuple[ScreeningExecutionResult, ...]:
    """Preflight and run each ready item through the canonical subprocess."""

    root = Path(workspace_root).expanduser().resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    results: list[ScreeningExecutionResult] = []
    resolved_evaluator = evaluator

    for item in plan.items:
        if not item.config_path:
            raise ValueError(
                f"{item.factor.factor_id} has no config_path; call "
                "write_strategy_configs() before execution"
            )
        if item.status == "blocked":
            resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
            run_id = record_blocked_plan_item(
                plan,
                item,
                evaluator=resolved_evaluator,
            )
            results.append(
                ScreeningExecutionResult(
                    factor_id=item.factor.factor_id,
                    strategy_id=item.strategy_id,
                    status="blocked_recorded",
                    run_id=run_id,
                    failure_code=item.primary_blocker_code,
                )
            )
            continue

        try:
            preflight = subprocess.run(
                _command_for_item(
                    item,
                    plan,
                    python_executable=python_executable,
                    build_only=True,
                    start_date=start_date,
                    end_date=end_date,
                    split_date=split_date,
                ),
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            preflight_output = _process_output(preflight)
        except subprocess.TimeoutExpired as exc:
            resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
            run_id = record_blocked_plan_item(
                plan,
                item,
                evaluator=resolved_evaluator,
                failure_code="strategy_preflight_timeout",
                suggested_action=(
                    f"Build-only preflight exceeded {timeout_seconds} seconds; "
                    "inspect factor preparation cost and data volume."
                ),
            )
            results.append(
                ScreeningExecutionResult(
                    factor_id=item.factor.factor_id,
                    strategy_id=item.strategy_id,
                    status="blocked_recorded",
                    run_id=run_id,
                    return_code=124,
                    output=str(exc),
                    failure_code="strategy_preflight_timeout",
                )
            )
            continue

        no_targets = (
            "no active positions" in preflight_output.lower()
            or "not a tradable backtest" in preflight_output.lower()
        )
        if preflight.returncode != 0 or no_targets:
            code = (
                "strategy_preflight_no_active_targets"
                if no_targets
                else "strategy_preflight_failed"
            )
            resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
            run_id = record_blocked_plan_item(
                plan,
                item,
                evaluator=resolved_evaluator,
                failure_code=code,
                suggested_action=(
                    "Canonical build-only preflight failed. Review the captured "
                    "output, factor contract, dataset fields, and sleeve compatibility."
                ),
            )
            results.append(
                ScreeningExecutionResult(
                    factor_id=item.factor.factor_id,
                    strategy_id=item.strategy_id,
                    status="blocked_recorded",
                    run_id=run_id,
                    return_code=preflight.returncode,
                    output=preflight_output,
                    failure_code=code,
                )
            )
            continue

        try:
            completed = subprocess.run(
                _command_for_item(
                    item,
                    plan,
                    python_executable=python_executable,
                    build_only=False,
                    start_date=start_date,
                    end_date=end_date,
                    split_date=split_date,
                ),
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            output = _process_output(completed)
            if completed.returncode == 0:
                results.append(
                    ScreeningExecutionResult(
                        factor_id=item.factor.factor_id,
                        strategy_id=item.strategy_id,
                        status="completed",
                        run_id=_run_id_from_output(output),
                        return_code=completed.returncode,
                        output=output,
                    )
                )
            else:
                resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
                code = "strategy_backtest_failed_after_preflight"
                run_id = record_blocked_plan_item(
                    plan,
                    item,
                    evaluator=resolved_evaluator,
                    failure_code=code,
                    suggested_action=(
                        "Canonical backtest failed after build-only preflight. "
                        "Inspect the captured output and repair the runtime failure."
                    ),
                )
                results.append(
                    ScreeningExecutionResult(
                        factor_id=item.factor.factor_id,
                        strategy_id=item.strategy_id,
                        status="blocked_recorded",
                        run_id=run_id,
                        return_code=completed.returncode,
                        output=output,
                        failure_code=code,
                    )
                )
        except subprocess.TimeoutExpired as exc:
            resolved_evaluator = resolved_evaluator or _default_evaluator(plan)
            code = "strategy_backtest_timeout_after_preflight"
            run_id = record_blocked_plan_item(
                plan,
                item,
                evaluator=resolved_evaluator,
                failure_code=code,
                suggested_action=(
                    f"Canonical backtest exceeded {timeout_seconds} seconds "
                    "after preflight; inspect factor runtime and data volume."
                ),
            )
            results.append(
                ScreeningExecutionResult(
                    factor_id=item.factor.factor_id,
                    strategy_id=item.strategy_id,
                    status="blocked_recorded",
                    run_id=run_id,
                    return_code=124,
                    output=str(exc),
                    failure_code=code,
                )
            )
    return tuple(results)


def write_execution_results(
    results: Sequence[ScreeningExecutionResult],
    output_dir: str | Path,
    *,
    workspace_root: str | Path = REPO_ROOT,
) -> Path:
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = Path(workspace_root) / destination
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "execution_results.json"
    path.write_text(
        json.dumps(
            [result.to_dict() for result in results],
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "BaselineRiskAssumptions",
    "CANONICAL_FUTURES_DAILY_DATASET_ID",
    "CANONICAL_FUTURES_DAILY_DATA_PATH",
    "DAILY_FACTOR_SCREEN_SCHEMA_VERSION",
    "DEFAULT_BATCH_ID",
    "DEFAULT_MARKET_VERTICAL",
    "DailyDatasetAssumption",
    "DailyFactorDescriptor",
    "DailyFactorScreeningPlan",
    "ScreeningExecutionResult",
    "ScreeningPlanItem",
    "SleeveMatch",
    "build_daily_factor_screening_plan",
    "build_typed_strategy_config",
    "enumerate_active_daily_factors",
    "execute_screening_plan",
    "execute_screening_plan_subprocess",
    "match_factor_to_sleeve",
    "record_blocked_plan_item",
    "resolve_daily_dataset",
    "write_execution_results",
    "write_screening_manifest",
    "write_strategy_configs",
]
