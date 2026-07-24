"""Deterministic preprocessing artifacts for ordered numeric matrices.

This module intentionally depends only on NumPy and the dependency-light
feature-schema contract.  It does not import pandas, scikit-learn, joblib, or
any paper-specific code, so the same fitted artifact can be used by research
jobs and operational inference services.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, ClassVar

import numpy as np

from oqp.contracts.regime_state import OrderedFeatureSchema


PREPROCESSOR_CORE_VERSION = "oqp_matrix_preprocessor_v1"
_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class PreprocessingError(ValueError):
    """Raised when a matrix or fitted preprocessing artifact is invalid."""


class MissingValuePolicy(str, Enum):
    """Training-fitted treatment of IEEE ``NaN`` feature values."""

    REJECT = "reject"
    CONSTANT = "constant"
    MEAN = "mean"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    """Versioned declaration of preprocessing behavior.

    Winsor bounds are estimated independently for each feature from the
    imputed training matrix using NumPy's deterministic ``linear`` quantile
    method.  Standardization uses population moments (``ddof=0``).
    """

    CONTRACT_VERSION: ClassVar[int] = 1
    QUANTILE_METHOD: ClassVar[str] = "linear"
    STANDARD_DEVIATION_DDOF: ClassVar[int] = 0

    standardize: bool = True
    winsor_quantiles: tuple[float, float] | None = None
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.REJECT
    constant_fill_value: float = 0.0
    scale_floor: float = 1e-12

    def __post_init__(self) -> None:
        if type(self.standardize) is not bool:
            raise TypeError("standardize must be boolean")
        policy = self.missing_value_policy
        if isinstance(policy, str):
            try:
                policy = MissingValuePolicy(policy)
            except ValueError as exc:
                raise PreprocessingError(
                    "missing_value_policy is not supported"
                ) from exc
            object.__setattr__(self, "missing_value_policy", policy)
        elif not isinstance(policy, MissingValuePolicy):
            raise TypeError("missing_value_policy must be a MissingValuePolicy")

        quantiles = self.winsor_quantiles
        if quantiles is not None:
            if isinstance(quantiles, (str, bytes)):
                raise TypeError("winsor_quantiles must contain two numbers")
            quantiles = tuple(quantiles)
            if len(quantiles) != 2:
                raise PreprocessingError(
                    "winsor_quantiles must contain exactly two values"
                )
            lower, upper = quantiles
            if any(type(value) not in (int, float) for value in quantiles):
                raise TypeError("winsor_quantiles must contain real numbers")
            lower = float(lower)
            upper = float(upper)
            if not (isfinite(lower) and isfinite(upper)):
                raise PreprocessingError("winsor_quantiles must be finite")
            if not 0.0 <= lower < upper <= 1.0:
                raise PreprocessingError(
                    "winsor_quantiles must satisfy 0 <= lower < upper <= 1"
                )
            object.__setattr__(self, "winsor_quantiles", (lower, upper))

        if type(self.constant_fill_value) not in (int, float):
            raise TypeError("constant_fill_value must be a real number")
        fill_value = float(self.constant_fill_value)
        if not isfinite(fill_value):
            raise PreprocessingError("constant_fill_value must be finite")
        object.__setattr__(self, "constant_fill_value", fill_value)

        if type(self.scale_floor) not in (int, float):
            raise TypeError("scale_floor must be a real number")
        scale_floor = float(self.scale_floor)
        if not isfinite(scale_floor) or scale_floor <= 0.0:
            raise PreprocessingError("scale_floor must be positive and finite")
        object.__setattr__(self, "scale_floor", scale_floor)

    def state_dict(self) -> dict[str, Any]:
        """Return the complete, canonical JSON-safe specification."""

        return {
            "contract_version": self.CONTRACT_VERSION,
            "standardize": self.standardize,
            "winsor_quantiles": (
                list(self.winsor_quantiles)
                if self.winsor_quantiles is not None
                else None
            ),
            "winsor_quantile_method": self.QUANTILE_METHOD,
            "missing_value_policy": self.missing_value_policy.value,
            "constant_fill_value": self.constant_fill_value,
            "standard_deviation_ddof": self.STANDARD_DEVIATION_DDOF,
            "scale_floor": self.scale_floor,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> PreprocessingSpec:
        """Restore a specification while rejecting silent semantic drift."""

        payload = _require_mapping(state, "preprocessing specification")
        _require_exact_keys(
            payload,
            {
                "contract_version",
                "standardize",
                "winsor_quantiles",
                "winsor_quantile_method",
                "missing_value_policy",
                "constant_fill_value",
                "standard_deviation_ddof",
                "scale_floor",
            },
            "preprocessing specification",
        )
        if (
            type(payload["contract_version"]) is not int
            or payload["contract_version"] != cls.CONTRACT_VERSION
        ):
            raise PreprocessingError("unsupported preprocessing specification version")
        if payload["winsor_quantile_method"] != cls.QUANTILE_METHOD:
            raise PreprocessingError("unsupported winsor quantile method")
        if (
            type(payload["standard_deviation_ddof"]) is not int
            or payload["standard_deviation_ddof"] != cls.STANDARD_DEVIATION_DDOF
        ):
            raise PreprocessingError("unsupported standard-deviation convention")
        quantiles = payload["winsor_quantiles"]
        if quantiles is not None and not isinstance(quantiles, list):
            raise TypeError("winsor_quantiles must be a JSON array or null")
        return cls(
            standardize=payload["standardize"],
            winsor_quantiles=(tuple(quantiles) if quantiles is not None else None),
            missing_value_policy=payload["missing_value_policy"],
            constant_fill_value=payload["constant_fill_value"],
            scale_floor=payload["scale_floor"],
        )


@dataclass(frozen=True, slots=True)
class FittedMatrixPreprocessor:
    """Immutable, authenticated parameters learned from one training matrix."""

    CONTRACT_VERSION: ClassVar[int] = 1

    artifact_id: str
    feature_schema: OrderedFeatureSchema
    spec: PreprocessingSpec
    fit_row_count: int
    training_data_sha256: str
    imputation_values: np.ndarray | None
    lower_bounds: np.ndarray | None
    upper_bounds: np.ndarray | None
    centers: np.ndarray
    scales: np.ndarray
    low_variance_mask: np.ndarray
    training_run_id: str | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id, "artifact_id")
        if not isinstance(self.feature_schema, OrderedFeatureSchema):
            raise TypeError("feature_schema must be an OrderedFeatureSchema")
        if not isinstance(self.spec, PreprocessingSpec):
            raise TypeError("spec must be a PreprocessingSpec")
        if type(self.fit_row_count) is not int or self.fit_row_count < 1:
            raise PreprocessingError("fit_row_count must be a positive integer")
        _require_sha256(self.training_data_sha256, "training_data_sha256")
        if self.training_run_id is not None:
            _require_identifier(self.training_run_id, "training_run_id")

        width = len(self.feature_schema.feature_names)
        imputation_values = _freeze_optional_float_vector(
            self.imputation_values,
            width=width,
            name="imputation_values",
        )
        lower_bounds = _freeze_optional_float_vector(
            self.lower_bounds,
            width=width,
            name="lower_bounds",
        )
        upper_bounds = _freeze_optional_float_vector(
            self.upper_bounds,
            width=width,
            name="upper_bounds",
        )
        centers = _freeze_float_vector(self.centers, width=width, name="centers")
        scales = _freeze_float_vector(self.scales, width=width, name="scales")
        low_variance_mask = _freeze_bool_vector(
            self.low_variance_mask,
            width=width,
            name="low_variance_mask",
        )

        if self.spec.missing_value_policy is MissingValuePolicy.REJECT:
            if imputation_values is not None:
                raise PreprocessingError(
                    "reject missing-value policy cannot store imputation values"
                )
        elif imputation_values is None:
            raise PreprocessingError(
                "imputing missing-value policies require imputation values"
            )

        if self.spec.winsor_quantiles is None:
            if lower_bounds is not None or upper_bounds is not None:
                raise PreprocessingError(
                    "clipping bounds require configured winsor quantiles"
                )
        elif lower_bounds is None or upper_bounds is None:
            raise PreprocessingError(
                "configured winsor quantiles require lower and upper bounds"
            )
        if lower_bounds is not None and upper_bounds is not None:
            if np.any(lower_bounds > upper_bounds):
                raise PreprocessingError(
                    "each lower clipping bound must not exceed its upper bound"
                )

        if np.any(scales <= 0.0):
            raise PreprocessingError("all stored scales must be positive")
        if not self.spec.standardize:
            if not np.array_equal(centers, np.zeros(width)):
                raise PreprocessingError(
                    "non-standardizing artifacts must store zero centers"
                )
            if not np.array_equal(scales, np.ones(width)):
                raise PreprocessingError(
                    "non-standardizing artifacts must store unit scales"
                )
            if np.any(low_variance_mask):
                raise PreprocessingError(
                    "non-standardizing artifacts cannot flag low-variance features"
                )

        object.__setattr__(self, "imputation_values", imputation_values)
        object.__setattr__(self, "lower_bounds", lower_bounds)
        object.__setattr__(self, "upper_bounds", upper_bounds)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "low_variance_mask", low_variance_mask)

        expected = self.calculate_artifact_sha256()
        if self.artifact_sha256 is None:
            object.__setattr__(self, "artifact_sha256", expected)
        else:
            _require_sha256(self.artifact_sha256, "artifact_sha256")
            if self.artifact_sha256 != expected:
                raise PreprocessingError(
                    "artifact_sha256 does not authenticate preprocessing state"
                )

    @property
    def lineage_sha256(self) -> str:
        """Digest for binding this exact preprocessing state into a model."""

        assert self.artifact_sha256 is not None
        return self.artifact_sha256

    def transform(
        self,
        values: Any,
        *,
        feature_schema: OrderedFeatureSchema,
    ) -> np.ndarray:
        """Apply only stored training parameters to a new ordered matrix."""

        self.require_integrity()
        _require_matching_schema(self.feature_schema, feature_schema)
        matrix = _coerce_matrix(
            values,
            width=len(self.feature_schema.feature_names),
            name="values",
        )
        transformed = matrix.copy()
        missing = np.isnan(transformed)
        if np.any(missing):
            if self.spec.missing_value_policy is MissingValuePolicy.REJECT:
                raise PreprocessingError(
                    "values contain NaN under the reject missing-value policy"
                )
            assert self.imputation_values is not None
            row_indices, column_indices = np.where(missing)
            transformed[row_indices, column_indices] = self.imputation_values[
                column_indices
            ]

        if self.lower_bounds is not None:
            assert self.upper_bounds is not None
            transformed = np.clip(
                transformed,
                self.lower_bounds,
                self.upper_bounds,
            )
        if self.spec.standardize:
            transformed = (transformed - self.centers) / self.scales
        if not np.isfinite(transformed).all():  # defensive invariant
            raise PreprocessingError("preprocessing produced non-finite values")
        return _freeze_float_matrix(transformed)

    def require_integrity(self) -> None:
        """Recalculate the semantic digest before the artifact is consumed."""

        if self.artifact_sha256 != self.calculate_artifact_sha256():
            raise PreprocessingError("preprocessing artifact integrity check failed")

    def calculate_artifact_sha256(self) -> str:
        """Hash every semantic field except the digest itself."""

        return hashlib.sha256(
            canonical_json_dumps(self._semantic_state_dict()).encode("utf-8")
        ).hexdigest()

    def _semantic_state_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.CONTRACT_VERSION,
            "core_version": PREPROCESSOR_CORE_VERSION,
            "artifact_id": self.artifact_id,
            "feature_schema": self.feature_schema.state_dict(),
            "spec": self.spec.state_dict(),
            "fit_row_count": self.fit_row_count,
            "training_data_sha256": self.training_data_sha256,
            "training_run_id": self.training_run_id,
            "statistics": {
                "imputation_values": _optional_vector_state(self.imputation_values),
                "lower_bounds": _optional_vector_state(self.lower_bounds),
                "upper_bounds": _optional_vector_state(self.upper_bounds),
                "centers": self.centers.tolist(),
                "scales": self.scales.tolist(),
                "low_variance_mask": self.low_variance_mask.tolist(),
            },
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a self-authenticating, JSON-safe artifact representation."""

        state = self._semantic_state_dict()
        state["artifact_sha256"] = self.artifact_sha256
        return state

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        expected_artifact_id: str,
        expected_artifact_sha256: str,
    ) -> FittedMatrixPreprocessor:
        """Restore only against identity and digest from a trusted registry."""

        _require_identifier(expected_artifact_id, "expected_artifact_id")
        _require_sha256(expected_artifact_sha256, "expected_artifact_sha256")
        payload = _require_mapping(state, "preprocessing artifact")
        _require_exact_keys(
            payload,
            {
                "contract_version",
                "core_version",
                "artifact_id",
                "artifact_sha256",
                "feature_schema",
                "spec",
                "fit_row_count",
                "training_data_sha256",
                "training_run_id",
                "statistics",
            },
            "preprocessing artifact",
        )
        if (
            type(payload["contract_version"]) is not int
            or payload["contract_version"] != cls.CONTRACT_VERSION
        ):
            raise PreprocessingError("unsupported preprocessing artifact version")
        if payload["core_version"] != PREPROCESSOR_CORE_VERSION:
            raise PreprocessingError("unsupported preprocessing core version")
        if payload["artifact_id"] != expected_artifact_id:
            raise PreprocessingError(
                "preprocessing artifact_id differs from expected_artifact_id"
            )
        if payload["artifact_sha256"] != expected_artifact_sha256:
            raise PreprocessingError(
                "preprocessing digest differs from expected_artifact_sha256"
            )
        statistics = _require_mapping(payload["statistics"], "statistics")
        _require_exact_keys(
            statistics,
            {
                "imputation_values",
                "lower_bounds",
                "upper_bounds",
                "centers",
                "scales",
                "low_variance_mask",
            },
            "preprocessing statistics",
        )
        try:
            schema = OrderedFeatureSchema.from_state_dict(payload["feature_schema"])
            spec = PreprocessingSpec.from_state_dict(payload["spec"])
            low_variance_mask = _require_bool_list(
                statistics["low_variance_mask"], "low_variance_mask"
            )
            return cls(
                artifact_id=payload["artifact_id"],
                feature_schema=schema,
                spec=spec,
                fit_row_count=payload["fit_row_count"],
                training_data_sha256=payload["training_data_sha256"],
                imputation_values=_require_optional_float_list(
                    statistics["imputation_values"], "imputation_values"
                ),
                lower_bounds=_require_optional_float_list(
                    statistics["lower_bounds"], "lower_bounds"
                ),
                upper_bounds=_require_optional_float_list(
                    statistics["upper_bounds"], "upper_bounds"
                ),
                centers=_require_float_list(statistics["centers"], "centers"),
                scales=_require_float_list(statistics["scales"], "scales"),
                low_variance_mask=low_variance_mask,
                training_run_id=payload["training_run_id"],
                artifact_sha256=payload["artifact_sha256"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PreprocessingError):
                raise
            raise PreprocessingError("invalid preprocessing artifact state") from exc


def fit_matrix_preprocessor(
    values: Any,
    *,
    feature_schema: OrderedFeatureSchema,
    spec: PreprocessingSpec,
    artifact_id: str,
    training_run_id: str | None = None,
) -> FittedMatrixPreprocessor:
    """Fit one deterministic artifact using only the supplied training rows."""

    if not isinstance(feature_schema, OrderedFeatureSchema):
        raise TypeError("feature_schema must be an OrderedFeatureSchema")
    if not isinstance(spec, PreprocessingSpec):
        raise TypeError("spec must be a PreprocessingSpec")
    _require_identifier(artifact_id, "artifact_id")
    if training_run_id is not None:
        _require_identifier(training_run_id, "training_run_id")
    matrix = _coerce_matrix(
        values,
        width=len(feature_schema.feature_names),
        name="training values",
    )
    training_data_sha256 = hash_numeric_matrix(
        matrix,
        feature_schema=feature_schema,
    )
    working = matrix.copy()
    missing = np.isnan(working)
    imputation_values: np.ndarray | None = None
    if np.any(missing) and spec.missing_value_policy is MissingValuePolicy.REJECT:
        raise PreprocessingError(
            "training values contain NaN under the reject missing-value policy"
        )
    if spec.missing_value_policy is not MissingValuePolicy.REJECT:
        imputation_values = np.empty(working.shape[1], dtype=np.float64)
        for column in range(working.shape[1]):
            observed = working[~missing[:, column], column]
            if spec.missing_value_policy is MissingValuePolicy.CONSTANT:
                imputation_values[column] = spec.constant_fill_value
            elif observed.size == 0:
                feature_name = feature_schema.feature_names[column]
                raise PreprocessingError(
                    f"cannot fit {spec.missing_value_policy.value} imputation for "
                    f"all-missing feature {feature_name!r}"
                )
            elif spec.missing_value_policy is MissingValuePolicy.MEAN:
                imputation_values[column] = float(np.mean(observed))
            else:
                imputation_values[column] = float(np.median(observed))
        if np.any(missing):
            row_indices, column_indices = np.where(missing)
            working[row_indices, column_indices] = imputation_values[column_indices]

    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    if spec.winsor_quantiles is not None:
        lower_quantile, upper_quantile = spec.winsor_quantiles
        lower_bounds = np.quantile(
            working,
            lower_quantile,
            axis=0,
            method=spec.QUANTILE_METHOD,
        )
        upper_bounds = np.quantile(
            working,
            upper_quantile,
            axis=0,
            method=spec.QUANTILE_METHOD,
        )
        working = np.clip(working, lower_bounds, upper_bounds)

    if spec.standardize:
        centers = np.mean(working, axis=0)
        raw_scales = np.std(
            working,
            axis=0,
            ddof=spec.STANDARD_DEVIATION_DDOF,
        )
        low_variance_mask = raw_scales <= spec.scale_floor
        scales = np.where(low_variance_mask, 1.0, raw_scales)
    else:
        centers = np.zeros(working.shape[1], dtype=np.float64)
        scales = np.ones(working.shape[1], dtype=np.float64)
        low_variance_mask = np.zeros(working.shape[1], dtype=np.bool_)

    return FittedMatrixPreprocessor(
        artifact_id=artifact_id,
        feature_schema=feature_schema,
        spec=spec,
        fit_row_count=matrix.shape[0],
        training_data_sha256=training_data_sha256,
        imputation_values=imputation_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        centers=centers,
        scales=scales,
        low_variance_mask=low_variance_mask,
        training_run_id=training_run_id,
    )


def hash_numeric_matrix(
    values: Any,
    *,
    feature_schema: OrderedFeatureSchema,
) -> str:
    """Hash ordered float64 matrix semantics, normalizing NaN and signed zero."""

    if not isinstance(feature_schema, OrderedFeatureSchema):
        raise TypeError("feature_schema must be an OrderedFeatureSchema")
    matrix = _coerce_matrix(
        values,
        width=len(feature_schema.feature_names),
        name="values",
    )
    canonical = np.array(matrix, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    canonical[np.isnan(canonical)] = np.nan
    header = canonical_json_dumps(
        {
            "dtype": "float64-little-endian",
            "feature_schema_sha256": feature_schema.schema_sha256,
            "shape": list(canonical.shape),
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_dumps(value: Any) -> str:
    """Serialize preprocessing state deterministically with finite numbers."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _coerce_matrix(values: Any, *, width: int, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        raise TypeError(f"{name} cannot be a masked array")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular numeric matrix") from exc
    if raw.dtype.kind not in "buif":
        raise TypeError(f"{name} must contain real numeric values")
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if matrix.ndim != 2:
        raise PreprocessingError(f"{name} must have exactly two dimensions")
    if matrix.shape[0] < 1:
        raise PreprocessingError(f"{name} must contain at least one row")
    if matrix.shape[1] != width:
        raise PreprocessingError(f"{name} width must match the ordered feature schema")
    if np.isinf(matrix).any():
        raise PreprocessingError(f"{name} cannot contain positive or negative infinity")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _freeze_float_matrix(values: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


def _freeze_float_vector(values: Any, *, width: int, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        raise TypeError(f"{name} cannot be a masked array")
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.shape != (width,):
        raise PreprocessingError(f"{name} must have one value per feature")
    if not np.isfinite(vector).all():
        raise PreprocessingError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(vector, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)


def _freeze_optional_float_vector(
    values: Any | None,
    *,
    width: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    return _freeze_float_vector(values, width=width, name=name)


def _freeze_bool_vector(values: Any, *, width: int, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        raise TypeError(f"{name} cannot be a masked array")
    vector = np.asarray(values)
    if vector.dtype.kind != "b":
        raise TypeError(f"{name} must contain booleans")
    if vector.ndim != 1 or vector.shape != (width,):
        raise PreprocessingError(f"{name} must have one value per feature")
    contiguous = np.ascontiguousarray(vector, dtype=np.bool_)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_)


def _optional_vector_state(values: np.ndarray | None) -> list[float] | None:
    return values.tolist() if values is not None else None


def _require_matching_schema(
    expected: OrderedFeatureSchema,
    observed: OrderedFeatureSchema,
) -> None:
    if not isinstance(observed, OrderedFeatureSchema):
        raise TypeError("feature_schema must be an OrderedFeatureSchema")
    if observed.schema_sha256 != expected.schema_sha256:
        raise PreprocessingError(
            "feature schema does not match the fitted ordered feature schema"
        )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise PreprocessingError(f"{name} has unknown or missing fields")


def _require_identifier(value: Any, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise PreprocessingError(f"{name} must be a non-empty string")


def _require_sha256(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise PreprocessingError(f"{name} must be a lowercase SHA-256 digest")


def _require_float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    if any(type(item) not in (int, float) for item in value):
        raise TypeError(f"{name} must contain real numbers")
    result = [float(item) for item in value]
    if not all(isfinite(item) for item in result):
        raise PreprocessingError(f"{name} must contain finite values")
    return result


def _require_optional_float_list(value: Any, name: str) -> list[float] | None:
    if value is None:
        return None
    return _require_float_list(value, name)


def _require_bool_list(value: Any, name: str) -> list[bool]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    if any(type(item) is not bool for item in value):
        raise TypeError(f"{name} must contain booleans")
    return value


__all__ = [
    "FittedMatrixPreprocessor",
    "MissingValuePolicy",
    "PREPROCESSOR_CORE_VERSION",
    "PreprocessingError",
    "PreprocessingSpec",
    "canonical_json_dumps",
    "fit_matrix_preprocessor",
    "hash_numeric_matrix",
]
