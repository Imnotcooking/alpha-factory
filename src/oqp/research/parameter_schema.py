"""Declarative factor-parameter schemas and reproducibility metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from numbers import Integral, Real
from types import ModuleType
from typing import Any, Mapping

import pandas as pd


PARAMETER_SCHEMA_VERSION = "1.0"
VALID_PARAMETER_TYPES = {
    "int",
    "float",
    "bool",
    "str",
    "categorical",
    "mapping",
    "object",
}
VALID_SPEC_FIELDS = {
    "default",
    "type",
    "low",
    "high",
    "step",
    "log",
    "choices",
    "tunable",
    "description",
}


@dataclass(frozen=True, slots=True)
class FactorParameterSpec:
    """One factor input and, when tunable, its optimizer search domain."""

    name: str
    default: Any
    parameter_type: str
    tunable: bool = False
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "default": self.default,
            "type": self.parameter_type,
            "tunable": self.tunable,
        }
        if self.low is not None:
            payload["low"] = self.low
        if self.high is not None:
            payload["high"] = self.high
        if self.step is not None:
            payload["step"] = self.step
        if self.log:
            payload["log"] = True
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class FactorParameterSchema:
    """Validated, immutable parameter declaration for one factor."""

    factor_id: str
    parameters: tuple[FactorParameterSpec, ...]
    version: str = PARAMETER_SCHEMA_VERSION

    @property
    def by_name(self) -> dict[str, FactorParameterSpec]:
        return {spec.name: spec for spec in self.parameters}

    @property
    def defaults(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.parameters}

    @property
    def tunable_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.parameters if spec.tunable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "factor_id": self.factor_id,
            "parameters": {
                spec.name: spec.to_dict() for spec in self.parameters
            },
        }

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())


def resolve_factor_parameter_schema(
    factor_module: ModuleType | Any,
) -> FactorParameterSchema:
    """Validate and normalize a factor module's ``FACTOR_PARAMETERS`` mapping."""

    factor_id = str(getattr(factor_module, "FACTOR_ID", "")).strip()
    raw_schema = getattr(factor_module, "FACTOR_PARAMETERS", None)
    if not factor_id:
        raise ValueError("Factor module must declare FACTOR_ID")
    if not isinstance(raw_schema, Mapping):
        raise ValueError(f"{factor_id} must declare a FACTOR_PARAMETERS mapping")

    schema = build_parameter_schema(factor_id, raw_schema)
    _validate_compute_signature(factor_module, schema)
    return schema


def build_parameter_schema(
    component_id: str,
    raw_schema: Mapping[str, Any],
) -> FactorParameterSchema:
    """Build a validated schema for any declarative component parameter map."""

    component_id = str(component_id).strip()
    if not component_id:
        raise ValueError("component_id cannot be empty")
    if not isinstance(raw_schema, Mapping):
        raise ValueError(f"{component_id} must declare a parameter mapping")
    specs = tuple(
        _parse_parameter_spec(str(name), raw_spec)
        for name, raw_spec in raw_schema.items()
    )
    return FactorParameterSchema(factor_id=component_id, parameters=specs)


def resolve_parameter_values(
    schema: FactorParameterSchema,
    overrides: Mapping[str, Any] | None = None,
    *,
    enforce_search_bounds: bool = False,
) -> dict[str, Any]:
    """Return explicit run values without mutating the schema or factor defaults."""

    supplied = dict(overrides or {})
    unknown = sorted(set(supplied).difference(schema.by_name))
    if unknown:
        raise ValueError(
            f"{schema.factor_id} received undeclared parameter(s): {', '.join(unknown)}"
        )

    resolved = schema.defaults
    resolved.update(supplied)
    for name, value in resolved.items():
        spec = schema.by_name[name]
        _validate_parameter_value(spec, value)
        if enforce_search_bounds and name in supplied and spec.tunable:
            _validate_search_bounds(spec, value)
    return resolved


def attach_factor_parameter_attrs(
    frame: pd.DataFrame,
    factor_module: ModuleType | Any,
    *,
    supplied_parameters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach the declared schema and actual run values to a factor frame."""

    if getattr(factor_module, "FACTOR_PARAMETERS", None) is None:
        return frame
    schema = resolve_factor_parameter_schema(factor_module)
    actual: dict[str, Any] = {}
    factor_params = frame.attrs.get("factor_params", {})
    if isinstance(factor_params, Mapping):
        actual.update(
            {
                name: factor_params[name]
                for name in schema.by_name
                if name in factor_params
            }
        )
    actual.update(dict(supplied_parameters or {}))
    values = resolve_parameter_values(schema, actual)
    overrides = {
        name: value
        for name, value in values.items()
        if not _values_equal(value, schema.by_name[name].default)
    }
    frame.attrs["factor_parameter_schema_version"] = schema.version
    frame.attrs["factor_parameter_schema"] = schema.to_dict()
    frame.attrs["factor_parameter_schema_fingerprint"] = schema.fingerprint
    frame.attrs["factor_parameter_values"] = values
    frame.attrs["factor_parameter_values_fingerprint"] = _stable_hash(values)
    frame.attrs["factor_parameter_overrides"] = overrides
    return frame


def fingerprint_parameter_payload(payload: Any) -> str:
    """Return the canonical SHA-256 used for schema and selected-value records."""

    return _stable_hash(payload)


def _parse_parameter_spec(
    name: str,
    raw_spec: Any,
) -> FactorParameterSpec:
    if not name.isidentifier():
        raise ValueError(f"Invalid factor parameter name: {name!r}")
    if not isinstance(raw_spec, Mapping):
        raise ValueError(f"Parameter {name!r} specification must be a mapping")
    unknown_fields = sorted(set(raw_spec).difference(VALID_SPEC_FIELDS))
    if unknown_fields:
        raise ValueError(
            f"Parameter {name!r} has unknown field(s): {', '.join(unknown_fields)}"
        )
    if "default" not in raw_spec:
        raise ValueError(f"Parameter {name!r} must declare a default")

    default = raw_spec["default"]
    parameter_type = str(raw_spec.get("type") or _infer_parameter_type(default)).lower()
    if parameter_type not in VALID_PARAMETER_TYPES:
        raise ValueError(
            f"Parameter {name!r} type must be one of {sorted(VALID_PARAMETER_TYPES)}"
        )
    tunable = bool(raw_spec.get("tunable", False))
    choices = tuple(raw_spec.get("choices") or ())
    spec = FactorParameterSpec(
        name=name,
        default=default,
        parameter_type=parameter_type,
        tunable=tunable,
        low=raw_spec.get("low"),
        high=raw_spec.get("high"),
        step=raw_spec.get("step"),
        log=bool(raw_spec.get("log", False)),
        choices=choices,
        description=str(raw_spec.get("description") or "").strip(),
    )
    _validate_parameter_spec(spec)
    return spec


def _validate_parameter_spec(spec: FactorParameterSpec) -> None:
    _validate_parameter_value(spec, spec.default)
    if not spec.tunable:
        if any(value is not None for value in (spec.low, spec.high, spec.step)):
            raise ValueError(
                f"Fixed parameter {spec.name!r} cannot declare low/high/step"
            )
        if spec.log or spec.choices:
            raise ValueError(
                f"Fixed parameter {spec.name!r} cannot declare log/choices"
            )
        return

    if spec.parameter_type in {"int", "float"}:
        if spec.low is None or spec.high is None:
            raise ValueError(
                f"Tunable numeric parameter {spec.name!r} requires low and high"
            )
        if not isinstance(spec.low, Real) or not isinstance(spec.high, Real):
            raise ValueError(f"Parameter {spec.name!r} bounds must be numeric")
        if float(spec.low) >= float(spec.high):
            raise ValueError(f"Parameter {spec.name!r} requires low < high")
        if spec.step is not None and (
            not isinstance(spec.step, Real) or float(spec.step) <= 0
        ):
            raise ValueError(f"Parameter {spec.name!r} step must be positive")
        if spec.log and spec.step is not None:
            raise ValueError(f"Parameter {spec.name!r} cannot combine log and step")
        if spec.log and float(spec.low) <= 0:
            raise ValueError(f"Log parameter {spec.name!r} requires positive bounds")
        if spec.parameter_type == "int":
            for label, value in (
                ("low", spec.low),
                ("high", spec.high),
                ("step", spec.step),
            ):
                if value is not None and not _is_int(value):
                    raise ValueError(
                        f"Integer parameter {spec.name!r} {label} must be an integer"
                    )
        _validate_search_bounds(spec, spec.default)
        return

    if spec.parameter_type in {"categorical", "bool"}:
        if spec.parameter_type == "categorical" and not spec.choices:
            raise ValueError(
                f"Categorical parameter {spec.name!r} requires non-empty choices"
            )
        if spec.parameter_type == "bool" and spec.choices:
            raise ValueError(f"Boolean parameter {spec.name!r} cannot declare choices")
        if any(value is not None for value in (spec.low, spec.high, spec.step)) or spec.log:
            raise ValueError(
                f"Parameter {spec.name!r} cannot use numeric search fields"
            )
        if spec.choices and spec.default not in spec.choices:
            raise ValueError(
                f"Parameter {spec.name!r} default is absent from choices"
            )
        return

    raise ValueError(
        f"Tunable parameter {spec.name!r} cannot use type {spec.parameter_type!r}"
    )


def _validate_compute_signature(
    factor_module: ModuleType | Any,
    schema: FactorParameterSchema,
) -> None:
    compute = getattr(factor_module, "compute", None)
    if not callable(compute):
        raise ValueError(f"{schema.factor_id} must expose callable compute()")
    signature = inspect.signature(compute)
    parameters = list(signature.parameters.values())
    if not parameters:
        raise ValueError(f"{schema.factor_id} compute() must accept a data frame")

    declared_compute: dict[str, inspect.Parameter] = {}
    for parameter in parameters[1:]:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        declared_compute[parameter.name] = parameter

    schema_names = set(schema.by_name)
    compute_names = set(declared_compute)
    missing = sorted(compute_names.difference(schema_names))
    unknown = sorted(schema_names.difference(compute_names))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing from FACTOR_PARAMETERS: " + ", ".join(missing))
        if unknown:
            details.append("absent from compute(): " + ", ".join(unknown))
        raise ValueError(f"{schema.factor_id} parameter mismatch; {'; '.join(details)}")

    for name, parameter in declared_compute.items():
        if parameter.default is inspect.Parameter.empty:
            raise ValueError(
                f"{schema.factor_id} compute parameter {name!r} requires a source default"
            )
        schema_default = schema.by_name[name].default
        if not _values_equal(schema_default, parameter.default):
            raise ValueError(
                f"{schema.factor_id} parameter {name!r} schema default "
                f"{schema_default!r} differs from compute() default {parameter.default!r}"
            )


def _validate_parameter_value(spec: FactorParameterSpec, value: Any) -> None:
    valid = False
    if spec.parameter_type == "bool":
        valid = isinstance(value, bool)
    elif spec.parameter_type == "int":
        valid = _is_int(value)
    elif spec.parameter_type == "float":
        valid = isinstance(value, Real) and not isinstance(value, bool)
    elif spec.parameter_type == "str":
        valid = isinstance(value, str)
    elif spec.parameter_type == "categorical":
        valid = value in spec.choices
    elif spec.parameter_type == "mapping":
        valid = isinstance(value, Mapping)
    elif spec.parameter_type == "object":
        valid = True
    if not valid:
        raise ValueError(
            f"Parameter {spec.name!r} value {value!r} is not type {spec.parameter_type}"
        )


def _validate_search_bounds(spec: FactorParameterSpec, value: Any) -> None:
    if spec.parameter_type in {"bool", "categorical"}:
        return
    numeric = float(value)
    if spec.low is not None and numeric < float(spec.low):
        raise ValueError(f"Parameter {spec.name!r} is below its search lower bound")
    if spec.high is not None and numeric > float(spec.high):
        raise ValueError(f"Parameter {spec.name!r} is above its search upper bound")
    if spec.step is not None:
        low = float(spec.low)
        step = float(spec.step)
        step_count = (numeric - low) / step
        if not math.isclose(
            step_count,
            round(step_count),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Parameter {spec.name!r} does not align with search step {step}"
            )


def _infer_parameter_type(default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if _is_int(default):
        return "int"
    if isinstance(default, Real):
        return "float"
    if isinstance(default, str):
        return "str"
    if isinstance(default, Mapping):
        return "mapping"
    raise ValueError(
        f"Cannot infer parameter type from default {default!r}; declare type explicitly"
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _values_equal(left: Any, right: Any) -> bool:
    if (
        isinstance(left, Real)
        and isinstance(right, Real)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FactorParameterSchema",
    "FactorParameterSpec",
    "PARAMETER_SCHEMA_VERSION",
    "attach_factor_parameter_attrs",
    "build_parameter_schema",
    "fingerprint_parameter_payload",
    "resolve_factor_parameter_schema",
    "resolve_parameter_values",
]
