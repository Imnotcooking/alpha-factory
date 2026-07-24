"""Validated loader for the Risk department limit catalog."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from oqp.config import REPO_ROOT
from oqp.risk.contracts import (
    RiskCalculationStatus,
    RiskEnforcementMode,
    RiskLimitCatalog,
    RiskLimitDefinition,
    RiskLimitDirection,
)


RISK_LIMIT_CATALOG_SCHEMA_VERSION = "oqp_risk_limit_catalog_v1"
DEFAULT_RISK_LIMIT_CATALOG_PATH = (
    REPO_ROOT / "departments" / "risk" / "limit_catalog.yaml"
)

_ALLOWED_SCOPES = frozenset(
    {"portfolio", "account", "strategy", "asset_class", "instrument", "order"}
)
_ALLOWED_UNITS = frozenset(
    {"ratio", "count", "currency", "currency_per_day", "days", "contracts"}
)


class RiskLimitCatalogError(ValueError):
    """Raised when the Risk limit catalog violates its contract."""


def load_risk_limit_catalog(
    path: str | Path | None = None,
) -> RiskLimitCatalog:
    """Load and validate the committed Risk control inventory."""

    source_path = Path(path) if path is not None else DEFAULT_RISK_LIMIT_CATALOG_PATH
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RiskLimitCatalogError(
            f"Cannot read risk limit catalog {source_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise RiskLimitCatalogError(
            f"Invalid YAML in risk limit catalog {source_path}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise RiskLimitCatalogError("Risk limit catalog root must be a mapping.")

    schema_version = _required_text(payload, "schema_version", context="catalog")
    if schema_version != RISK_LIMIT_CATALOG_SCHEMA_VERSION:
        raise RiskLimitCatalogError(
            f"Unsupported risk limit catalog schema_version: {schema_version!r}."
        )

    raw_controls = payload.get("controls")
    if not isinstance(raw_controls, list) or not raw_controls:
        raise RiskLimitCatalogError("Catalog controls must be a non-empty list.")

    controls: list[RiskLimitDefinition] = []
    control_ids: set[str] = set()
    for index, raw_control in enumerate(raw_controls):
        context = f"controls[{index}]"
        if not isinstance(raw_control, Mapping):
            raise RiskLimitCatalogError(f"{context} must be a mapping.")
        control = _parse_control(raw_control, context=context)
        if control.control_id in control_ids:
            raise RiskLimitCatalogError(
                f"Duplicate risk control id: {control.control_id!r}."
            )
        control_ids.add(control.control_id)
        controls.append(control)

    reporting_currency = _required_text(
        payload, "reporting_currency", context="catalog"
    ).upper()
    if len(reporting_currency) != 3 or not reporting_currency.isalpha():
        raise RiskLimitCatalogError(
            "catalog.reporting_currency must be a three-letter currency code."
        )

    return RiskLimitCatalog(
        schema_version=schema_version,
        catalog_owner=_required_text(payload, "catalog_owner", context="catalog"),
        reporting_currency=reporting_currency,
        controls=tuple(controls),
        source_path=source_path,
    )


def _parse_control(
    raw: Mapping[str, Any], *, context: str
) -> RiskLimitDefinition:
    direction = _enum_value(
        raw,
        "direction",
        context=context,
        enum_type=RiskLimitDirection,
    )
    calculation_status = _enum_value(
        raw,
        "calculation_status",
        context=context,
        enum_type=RiskCalculationStatus,
    )
    enforcement_mode = _enum_value(
        raw,
        "enforcement_mode",
        context=context,
        enum_type=RiskEnforcementMode,
    )
    scope = _choice(raw, "scope", context=context, allowed=_ALLOWED_SCOPES)
    unit = _choice(raw, "unit", context=context, allowed=_ALLOWED_UNITS)
    warning_threshold = _optional_threshold(
        raw.get("warning_threshold"), field=f"{context}.warning_threshold"
    )
    hard_threshold = _optional_threshold(
        raw.get("hard_threshold"), field=f"{context}.hard_threshold"
    )

    if calculation_status is not RiskCalculationStatus.ACTIVE and (
        enforcement_mode is not RiskEnforcementMode.OBSERVE
    ):
        raise RiskLimitCatalogError(
            f"{context} must remain observe until calculation_status is active."
        )
    if enforcement_mode is RiskEnforcementMode.WARN and warning_threshold is None:
        raise RiskLimitCatalogError(
            f"{context}.warning_threshold is required for warn mode."
        )
    if enforcement_mode is RiskEnforcementMode.BLOCK and hard_threshold is None:
        raise RiskLimitCatalogError(
            f"{context}.hard_threshold is required for block mode."
        )
    if warning_threshold is not None and hard_threshold is not None:
        if (
            direction is RiskLimitDirection.MAX
            and warning_threshold > hard_threshold
        ) or (
            direction is RiskLimitDirection.MIN
            and warning_threshold < hard_threshold
        ):
            raise RiskLimitCatalogError(
                f"{context} warning threshold must be reached before hard threshold."
            )

    source_path, source_symbol = _metric_source(
        _required_text(raw, "metric_source", context=context),
        field=f"{context}.metric_source",
    )
    return RiskLimitDefinition(
        control_id=_required_text(raw, "id", context=context),
        category=_required_text(raw, "category", context=context).lower(),
        scope=scope,
        unit=unit,
        direction=direction,
        calculation_status=calculation_status,
        enforcement_mode=enforcement_mode,
        warning_threshold=warning_threshold,
        hard_threshold=hard_threshold,
        metric_source_path=source_path,
        metric_source_symbol=source_symbol,
        owner=_required_text(raw, "owner", context=context),
        description=_required_text(raw, "description", context=context),
    )


def _enum_value(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    enum_type: type[RiskCalculationStatus]
    | type[RiskEnforcementMode]
    | type[RiskLimitDirection],
) -> RiskCalculationStatus | RiskEnforcementMode | RiskLimitDirection:
    raw_value = _required_text(payload, key, context=context).lower()
    try:
        return enum_type(raw_value)
    except ValueError as exc:
        allowed = sorted(member.value for member in enum_type)
        raise RiskLimitCatalogError(
            f"{context}.{key} must be one of {allowed}."
        ) from exc


def _choice(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    allowed: frozenset[str],
) -> str:
    value = _required_text(payload, key, context=context).lower()
    if value not in allowed:
        raise RiskLimitCatalogError(
            f"{context}.{key} must be one of {sorted(allowed)}."
        )
    return value


def _optional_threshold(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskLimitCatalogError(f"{field} must be a finite non-negative number.")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0:
        raise RiskLimitCatalogError(f"{field} must be a finite non-negative number.")
    return threshold


def _metric_source(value: str, *, field: str) -> tuple[PurePosixPath, str]:
    if value.count(":") != 1:
        raise RiskLimitCatalogError(
            f"{field} must use the format 'src/path.py:symbol'."
        )
    raw_path, symbol = value.split(":", maxsplit=1)
    path = _safe_relative_path(raw_path, field=field)
    if not path.parts or path.parts[0] != "src" or path.suffix != ".py":
        raise RiskLimitCatalogError(
            f"{field} path must be a Python file beneath 'src/'."
        )
    if not symbol.isidentifier():
        raise RiskLimitCatalogError(f"{field} symbol must be a Python identifier.")
    return path, symbol


def _required_text(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    if key not in payload:
        raise RiskLimitCatalogError(f"{context}.{key} is required.")
    return _nonempty_text(payload[key], f"{context}.{key}")


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RiskLimitCatalogError(f"{field} must be a non-empty string.")
    return value.strip()


def _safe_relative_path(value: str, *, field: str) -> PurePosixPath:
    if "\\" in value:
        raise RiskLimitCatalogError(f"{field} must use POSIX separators.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path == PurePosixPath("."):
        raise RiskLimitCatalogError(f"{field} must be a non-empty relative path.")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RiskLimitCatalogError(
            f"{field} cannot contain empty, '.' or '..' segments."
        )
    return path
