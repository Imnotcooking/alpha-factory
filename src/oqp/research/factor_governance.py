"""Governance checks for stable factor identities and comparable metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


METADATA_SCHEMA_VERSION = 1
REQUIRED_METADATA_FIELDS = (
    "metadata_schema_version",
    "component_type",
    "status",
    "factor_family",
    "factor_subfamily",
    "native_market",
    "supported_markets",
    "data_frequency",
    "signal_frequency",
    "rebalance_frequency",
    "signal_horizon",
    "execution_style",
    "portfolio_layer",
    "deduplication_cohort",
    "cost_model",
    "required_fields",
    "legacy_ids",
)
REQUIRED_CONTRACT_FIELDS = (
    "evaluation_geometry",
    "execution_mode",
    "alpha_signal_col",
    "execution_weight_col",
    "execution_lag",
    "return_assumption",
    "supported_markets",
)


@dataclass(frozen=True, slots=True)
class FactorGovernanceRecord:
    factor_id: str
    source_stem: str
    metadata_schema_version: int | None
    component_type: str
    factor_family: str
    factor_subfamily: str
    native_market: str
    data_frequency: str
    portfolio_layer: str
    deduplication_cohort: str
    comparability_key: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def inspect_factor_governance(
    path: Path,
    module: ModuleType | Any,
) -> FactorGovernanceRecord:
    """Inspect one factor without changing or inferring missing declarations."""

    source_stem = path.stem
    factor_id = str(getattr(module, "FACTOR_ID", "")).strip()
    metadata = getattr(module, "FACTOR_METADATA", {}) or {}
    contract = getattr(module, "FACTOR_CONTRACT", {}) or {}
    issues: list[str] = []
    if factor_id != source_stem:
        issues.append(f"FACTOR_ID {factor_id!r} does not match filename {source_stem!r}")
    if not isinstance(metadata, Mapping):
        metadata = {}
        issues.append("FACTOR_METADATA must be a mapping")
    if not isinstance(contract, Mapping):
        contract = {}
        issues.append("FACTOR_CONTRACT must be a mapping")

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            issues.append(f"missing metadata field: {field}")
    for field in REQUIRED_CONTRACT_FIELDS:
        if field not in contract:
            issues.append(f"missing contract field: {field}")

    schema_version = _optional_int(metadata.get("metadata_schema_version"))
    if schema_version is not None and schema_version != METADATA_SCHEMA_VERSION:
        issues.append(
            f"unsupported metadata_schema_version: {schema_version}; "
            f"expected {METADATA_SCHEMA_VERSION}"
        )
    component_type = str(metadata.get("component_type", "")).strip().lower()
    if component_type and component_type != "factor":
        issues.append(f"component_type must be 'factor', got {component_type!r}")
    legacy_ids = metadata.get("legacy_ids")
    if legacy_ids is not None and not isinstance(legacy_ids, (list, tuple)):
        issues.append("legacy_ids must be a list or tuple")

    metadata_markets = _string_tuple(metadata.get("supported_markets"))
    contract_markets = _string_tuple(contract.get("supported_markets"))
    if metadata_markets and contract_markets and metadata_markets != contract_markets:
        issues.append("metadata and contract supported_markets differ")
    native_market = str(metadata.get("native_market", "")).strip().upper()
    if native_market and metadata_markets and native_market not in metadata_markets:
        issues.append("native_market is absent from supported_markets")

    comparability_key = (
        native_market,
        str(metadata.get("data_frequency", "")).strip().lower(),
        str(contract.get("evaluation_geometry", "")).strip().lower(),
        str(contract.get("execution_mode", "")).strip().lower(),
        str(contract.get("execution_lag", "")).strip().lower(),
        str(contract.get("return_assumption", "")).strip().lower(),
        str(metadata.get("portfolio_layer", "")).strip().lower(),
        str(metadata.get("deduplication_cohort", "")).strip().lower(),
    )
    return FactorGovernanceRecord(
        factor_id=factor_id,
        source_stem=source_stem,
        metadata_schema_version=schema_version,
        component_type=component_type,
        factor_family=str(metadata.get("factor_family", "")).strip().lower(),
        factor_subfamily=str(metadata.get("factor_subfamily", "")).strip().lower(),
        native_market=native_market,
        data_frequency=str(metadata.get("data_frequency", "")).strip().lower(),
        portfolio_layer=str(metadata.get("portfolio_layer", "")).strip().lower(),
        deduplication_cohort=str(metadata.get("deduplication_cohort", "")).strip(),
        comparability_key=comparability_key,
        issues=tuple(issues),
    )


def validate_factor_governance(
    path: Path,
    module: ModuleType | Any,
) -> FactorGovernanceRecord:
    record = inspect_factor_governance(path, module)
    if record.issues:
        joined = "; ".join(record.issues)
        raise ValueError(f"{record.source_stem} governance validation failed: {joined}")
    return record


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple, set)):
        values = tuple(value)
    else:
        return ()
    return tuple(str(item).strip().upper() for item in values if str(item).strip())


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "FactorGovernanceRecord",
    "METADATA_SCHEMA_VERSION",
    "REQUIRED_CONTRACT_FIELDS",
    "REQUIRED_METADATA_FIELDS",
    "inspect_factor_governance",
    "validate_factor_governance",
]
