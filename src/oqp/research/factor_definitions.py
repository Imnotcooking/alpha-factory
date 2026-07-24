"""Phase 1 contracts for pure, independently evaluable factor components."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from oqp.research.factor_governance import inspect_factor_governance
from oqp.research.factor_purity import inspect_factor_source_purity
from oqp.research.parameter_schema import (
    FactorParameterSchema,
    resolve_factor_parameter_schema,
)


FACTOR_DEFINITION_SCHEMA_VERSION = 1
VALID_SIGNAL_ORIENTATIONS = {
    "higher_is_bullish",
    "higher_is_bearish",
    "unsigned_event",
}
VALID_HORIZON_UNITS = {"bars", "sessions", "days", "weeks", "months"}
PURE_FACTOR_PORTFOLIO_LAYERS = {"alpha_score", "predictive_signal"}
ALLOCATION_PARAMETER_MARKERS = (
    "gross_leverage",
    "kelly",
    "max_weight",
    "portfolio_vol",
    "position_size",
    "risk_budget",
    "target_gross",
)


@dataclass(frozen=True, slots=True)
class ExpectedHoldingHorizon:
    """Economic horizon over which the factor expects its prediction to persist."""

    minimum: float
    maximum: float
    unit: str
    rationale: str = ""

    def __post_init__(self) -> None:
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        unit = str(self.unit).strip().lower()
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError("holding-horizon minimum must be positive and finite")
        if not math.isfinite(maximum) or maximum < minimum:
            raise ValueError("holding-horizon maximum must be finite and >= minimum")
        if unit not in VALID_HORIZON_UNITS:
            raise ValueError(
                f"holding-horizon unit must be one of {sorted(VALID_HORIZON_UNITS)}"
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "rationale", str(self.rationale).strip())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExpectedHoldingHorizon":
        return cls(
            minimum=float(payload.get("minimum")),
            maximum=float(payload.get("maximum")),
            unit=str(payload.get("unit") or ""),
            rationale=str(payload.get("rationale") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": self.unit,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    """Complete Phase 1 declaration for one predictive factor."""

    factor_id: str
    family: str
    economic_hypothesis: str
    required_columns: tuple[str, ...]
    native_market: str
    data_frequency: str
    signal_frequency: str
    parameter_schema: FactorParameterSchema
    signal_orientation: str
    evaluation_geometry: str
    expected_holding_horizon: ExpectedHoldingHorizon
    known_limitations: tuple[str, ...]
    source: str
    implementation_fingerprint: str
    schema_version: int = FACTOR_DEFINITION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "factor_id": self.factor_id,
            "family": self.family,
            "economic_hypothesis": self.economic_hypothesis,
            "required_columns": list(self.required_columns),
            "native_market": self.native_market,
            "data_frequency": self.data_frequency,
            "signal_frequency": self.signal_frequency,
            "default_parameters": self.parameter_schema.defaults,
            "parameter_schema_fingerprint": self.parameter_schema.fingerprint,
            "signal_orientation": self.signal_orientation,
            "evaluation_geometry": self.evaluation_geometry,
            "expected_holding_horizon": self.expected_holding_horizon.to_dict(),
            "known_limitations": list(self.known_limitations),
            "source": self.source,
            "implementation_fingerprint": self.implementation_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class FactorDefinitionInspection:
    """Non-throwing readiness result used by library audits and dashboards."""

    factor_id: str
    source: str
    family: str
    native_market: str
    data_frequency: str
    evaluation_geometry: str
    definition: FactorDefinition | None
    issues: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.definition is not None and not self.issues

    def to_manifest_row(self) -> dict[str, Any]:
        definition = self.definition
        return {
            "factor_id": self.factor_id,
            "phase_1_ready": self.ready,
            "factor_family": self.family,
            "native_market": self.native_market,
            "data_frequency": self.data_frequency,
            "evaluation_geometry": self.evaluation_geometry,
            "signal_orientation": (
                definition.signal_orientation if definition is not None else ""
            ),
            "holding_horizon": (
                _format_horizon(definition.expected_holding_horizon)
                if definition is not None
                else ""
            ),
            "parameter_count": (
                len(definition.parameter_schema.parameters)
                if definition is not None
                else 0
            ),
            "known_limitation_count": (
                len(definition.known_limitations) if definition is not None else 0
            ),
            "definition_fingerprint": (
                definition.fingerprint if definition is not None else ""
            ),
            "implementation_fingerprint": (
                definition.implementation_fingerprint
                if definition is not None
                else ""
            ),
            "issues": "; ".join(self.issues),
            "source": self.source,
        }


def inspect_factor_definition(
    path: Path,
    module: ModuleType | Any,
) -> FactorDefinitionInspection:
    """Inspect one factor without inferring declarations that researchers omitted."""

    governance = inspect_factor_governance(path, module)
    purity = inspect_factor_source_purity(path, module)
    source = _portable_source(path)
    metadata = getattr(module, "FACTOR_METADATA", {}) or {}
    contract = getattr(module, "FACTOR_CONTRACT", {}) or {}
    issues = list(governance.issues)
    issues.extend(purity.issues)

    hypothesis = _first_text(
        getattr(module, "ECONOMIC_HYPOTHESIS", None),
        getattr(module, "ECONOMIC_RATIONALE", None),
        getattr(module, "ECONOMIC_RATIONALE_EN", None),
    )
    if not hypothesis:
        issues.append("missing economic hypothesis")

    required_columns = _string_tuple(metadata.get("required_fields"))
    if not required_columns:
        issues.append("required_fields must contain at least one input column")

    signal_frequency = str(metadata.get("signal_frequency") or "").strip().lower()
    if not signal_frequency:
        issues.append("missing signal frequency")

    orientation = str(
        getattr(module, "SIGNAL_ORIENTATION", None)
        or metadata.get("signal_orientation")
        or ""
    ).strip().lower()
    if orientation not in VALID_SIGNAL_ORIENTATIONS:
        issues.append(
            "SIGNAL_ORIENTATION must be one of "
            + ", ".join(sorted(VALID_SIGNAL_ORIENTATIONS))
        )

    horizon: ExpectedHoldingHorizon | None = None
    raw_horizon = getattr(module, "EXPECTED_HOLDING_HORIZON", None)
    if not isinstance(raw_horizon, Mapping):
        issues.append("missing EXPECTED_HOLDING_HORIZON mapping")
    else:
        try:
            horizon = ExpectedHoldingHorizon.from_mapping(raw_horizon)
        except (TypeError, ValueError) as exc:
            issues.append(f"invalid EXPECTED_HOLDING_HORIZON: {exc}")

    limitations = _string_tuple(
        getattr(module, "KNOWN_LIMITATIONS", None)
        or metadata.get("known_limitations")
    )
    if not limitations:
        issues.append("KNOWN_LIMITATIONS must declare at least one limitation")

    parameter_schema: FactorParameterSchema | None = None
    if not hasattr(module, "FACTOR_PARAMETERS"):
        issues.append("missing FACTOR_PARAMETERS declaration")
    else:
        try:
            parameter_schema = resolve_factor_parameter_schema(module)
        except (TypeError, ValueError) as exc:
            issues.append(f"invalid FACTOR_PARAMETERS: {exc}")

    portfolio_layer = str(metadata.get("portfolio_layer") or "").strip().lower()
    if portfolio_layer not in PURE_FACTOR_PORTFOLIO_LAYERS:
        issues.append(
            f"portfolio_layer {portfolio_layer!r} is not a pure predictive layer"
        )
    if parameter_schema is not None:
        allocation_parameters = sorted(
            name
            for name in parameter_schema.by_name
            if any(marker in name.lower() for marker in ALLOCATION_PARAMETER_MARKERS)
        )
        if allocation_parameters:
            issues.append(
                "allocation parameters belong outside the factor: "
                + ", ".join(allocation_parameters)
            )

    definition: FactorDefinition | None = None
    if not issues and parameter_schema is not None and horizon is not None:
        definition = FactorDefinition(
            factor_id=governance.factor_id,
            family=governance.factor_family,
            economic_hypothesis=hypothesis,
            required_columns=required_columns,
            native_market=governance.native_market,
            data_frequency=governance.data_frequency,
            signal_frequency=signal_frequency,
            parameter_schema=parameter_schema,
            signal_orientation=orientation,
            evaluation_geometry=str(
                contract.get("evaluation_geometry") or ""
            ).strip().lower(),
            expected_holding_horizon=horizon,
            known_limitations=limitations,
            source=source,
            implementation_fingerprint=purity.implementation_fingerprint,
        )

    return FactorDefinitionInspection(
        factor_id=governance.factor_id or path.stem,
        source=source,
        family=governance.factor_family,
        native_market=governance.native_market,
        data_frequency=governance.data_frequency,
        evaluation_geometry=str(contract.get("evaluation_geometry") or "")
        .strip()
        .lower(),
        definition=definition,
        issues=tuple(dict.fromkeys(issues)),
    )


def resolve_factor_definition(
    path: Path,
    module: ModuleType | Any,
) -> FactorDefinition:
    inspection = inspect_factor_definition(path, module)
    if not inspection.ready or inspection.definition is None:
        raise ValueError(
            f"{inspection.factor_id} Phase 1 definition is incomplete: "
            + "; ".join(inspection.issues)
        )
    return inspection.definition


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple, set)):
        values = tuple(value)
    else:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _format_horizon(horizon: ExpectedHoldingHorizon) -> str:
    if horizon.minimum == horizon.maximum:
        return f"{horizon.minimum:g} {horizon.unit}"
    return f"{horizon.minimum:g}-{horizon.maximum:g} {horizon.unit}"


def _portable_source(path: Path) -> str:
    parts = path.parts
    for anchor in ("departments", "src", "tests"):
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.name


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ALLOCATION_PARAMETER_MARKERS",
    "ExpectedHoldingHorizon",
    "FACTOR_DEFINITION_SCHEMA_VERSION",
    "FactorDefinition",
    "FactorDefinitionInspection",
    "PURE_FACTOR_PORTFOLIO_LAYERS",
    "VALID_HORIZON_UNITS",
    "VALID_SIGNAL_ORIENTATIONS",
    "inspect_factor_definition",
    "resolve_factor_definition",
]
