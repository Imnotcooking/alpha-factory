"""Component-neutral facade over declarative parameter schemas."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping

from oqp.research.parameter_schema import (
    FactorParameterSchema,
    FactorParameterSpec,
    build_parameter_schema,
    resolve_factor_parameter_schema,
    resolve_parameter_values,
)
from oqp.research.parameter_optimization import suggest_factor_parameters


ComponentParameterSpec = FactorParameterSpec


@dataclass(frozen=True, slots=True)
class ComponentParameterSchema:
    component_id: str
    component_type: str
    schema: FactorParameterSchema

    @property
    def parameters(self) -> tuple[ComponentParameterSpec, ...]:
        return self.schema.parameters

    @property
    def by_name(self) -> dict[str, ComponentParameterSpec]:
        return self.schema.by_name

    @property
    def defaults(self) -> dict[str, Any]:
        return self.schema.defaults

    @property
    def tunable_names(self) -> tuple[str, ...]:
        return self.schema.tunable_names

    @property
    def fingerprint(self) -> str:
        return self.schema.fingerprint

    def to_dict(self) -> dict[str, Any]:
        payload = self.schema.to_dict()
        payload["component_id"] = self.component_id
        payload["component_type"] = self.component_type
        return payload


def resolve_component_parameter_schema(
    component_module: ModuleType | Any,
    *,
    component_type: str,
    declaration_name: str | None = None,
    component_id: str | None = None,
) -> ComponentParameterSchema:
    component_type = str(component_type).strip().lower()
    if not component_type:
        raise ValueError("component_type cannot be empty")
    declaration_name = declaration_name or f"{component_type.upper()}_PARAMETERS"
    id_attribute = {
        "factor": "FACTOR_ID",
        "router": "ROUTER_ID",
        "risk_overlay": "OVERLAY_ID",
        "model": "MODEL_ID",
        "allocator": "ALLOCATOR_ID",
    }.get(component_type, "COMPONENT_ID")
    resolved_id = str(
        component_id or getattr(component_module, id_attribute, "")
    ).strip()
    if component_type == "factor" and declaration_name == "FACTOR_PARAMETERS":
        factor_schema = resolve_factor_parameter_schema(component_module)
        return ComponentParameterSchema(resolved_id, component_type, factor_schema)
    raw_schema = getattr(component_module, declaration_name, None)
    if not isinstance(raw_schema, Mapping):
        raise ValueError(
            f"{resolved_id or component_type} must declare {declaration_name}"
        )
    schema = build_parameter_schema(resolved_id, raw_schema)
    return ComponentParameterSchema(resolved_id, component_type, schema)


def build_component_parameter_schema(
    component_id: str,
    component_type: str,
    raw_schema: Mapping[str, Any],
) -> ComponentParameterSchema:
    """Build a validated component schema when ranges are configuration-driven."""

    resolved_id = str(component_id).strip()
    resolved_type = str(component_type).strip().lower()
    if not resolved_id or not resolved_type:
        raise ValueError("component_id and component_type cannot be empty")
    return ComponentParameterSchema(
        component_id=resolved_id,
        component_type=resolved_type,
        schema=build_parameter_schema(resolved_id, raw_schema),
    )


def resolve_component_parameter_values(
    schema: ComponentParameterSchema,
    overrides: Mapping[str, Any] | None = None,
    *,
    enforce_search_bounds: bool = False,
) -> dict[str, Any]:
    return resolve_parameter_values(
        schema.schema,
        overrides,
        enforce_search_bounds=enforce_search_bounds,
    )


def suggest_component_parameters(
    trial: Any,
    schema: ComponentParameterSchema,
) -> dict[str, Any]:
    return suggest_factor_parameters(trial, schema.schema)


__all__ = [
    "ComponentParameterSchema",
    "ComponentParameterSpec",
    "build_component_parameter_schema",
    "resolve_component_parameter_schema",
    "resolve_component_parameter_values",
    "suggest_component_parameters",
]
