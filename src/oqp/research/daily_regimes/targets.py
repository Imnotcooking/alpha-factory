"""Fold-aware interfaces for forward target construction.

Stage 5 owns the preregistered target formulas, training-only tail thresholds,
purge horizons, and availability assertions.  The interfaces here make a
training-estimated threshold impossible to apply without an explicit fold,
training interval, training-row hash, and fitted-parameter hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd
from pandas.api.types import is_numeric_dtype


STAGE_OWNER = 5


class TargetKind(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


class TargetFitScope(str, Enum):
    """Whether construction requires parameters estimated from a training fold."""

    NO_ESTIMATION = "no_estimation"
    TRAINING_FOLD_ONLY = "training_fold_only"


class TargetConstructionUnavailableError(NotImplementedError):
    """Raised when preregistered targets are requested before Stage 5."""


@dataclass(frozen=True)
class TargetSpec:
    """Declaration of one forward target and its estimation scope."""

    name: str
    kind: TargetKind
    horizon_periods: int
    primary: bool = False
    formula_version: str = "pending_stage_5"
    fit_scope: TargetFitScope = TargetFitScope.NO_ESTIMATION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Target names must be non-empty.")
        if not isinstance(self.kind, TargetKind):
            raise TypeError("kind must be a TargetKind value.")
        if not isinstance(self.fit_scope, TargetFitScope):
            raise TypeError("fit_scope must be a TargetFitScope value.")
        if self.horizon_periods < 1:
            raise ValueError("horizon_periods must be at least one.")
        if (
            self.name == "next_day_tail_loss_event"
            and self.fit_scope is not TargetFitScope.TRAINING_FOLD_ONLY
        ):
            raise ValueError(
                "next_day_tail_loss_event requires a training-fold-only threshold."
            )


@dataclass(frozen=True)
class TargetBuildRequest:
    """Requested target set and its timing/provenance columns."""

    specs: tuple[TargetSpec, ...]
    product_column: str = "product"
    fold_column: str = "fold_id"
    information_date_column: str = "information_date"
    target_date_column: str = "target_date"

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("A target request must contain at least one target.")
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("Target names must be unique.")
        columns = (
            self.product_column,
            self.fold_column,
            self.information_date_column,
            self.target_date_column,
        )
        if any(not column.strip() for column in columns):
            raise ValueError("Target key-column names must be non-empty.")
        if len(columns) != len(set(columns)):
            raise ValueError("Target key-column names must be distinct.")

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def maximum_horizon(self) -> int:
        return max(spec.horizon_periods for spec in self.specs)

    @property
    def requires_training_fit(self) -> bool:
        return any(
            spec.fit_scope is TargetFitScope.TRAINING_FOLD_ONLY for spec in self.specs
        )


@dataclass(frozen=True)
class TargetTrainingContext:
    """Exact training sample authorized to estimate fold-local target parameters."""

    fold_id: str
    training_start: date
    training_end: date
    training_rows_hash: str
    training_row_count: int

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty.")
        if not isinstance(self.training_start, date) or not isinstance(
            self.training_end, date
        ):
            raise TypeError("Training boundaries must be datetime.date values.")
        if self.training_end < self.training_start:
            raise ValueError("training_end cannot precede training_start.")
        if self.training_row_count < 1:
            raise ValueError("training_row_count must be positive.")
        _require_sha256(self.training_rows_hash, "training_rows_hash")


@dataclass(frozen=True)
class TargetFitResult:
    """Fold-local fitted target parameters and immutable training provenance."""

    specs: tuple[TargetSpec, ...]
    context: TargetTrainingContext
    builder_id: str
    parameter_hash: str
    parameters_by_target: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("TargetFitResult.specs must be non-empty.")
        if not isinstance(self.context, TargetTrainingContext):
            raise TypeError("context must be a TargetTrainingContext.")
        if not self.builder_id.strip():
            raise ValueError("builder_id must be non-empty.")
        _require_sha256(self.parameter_hash, "parameter_hash")

        spec_names = {spec.name for spec in self.specs}
        if len(spec_names) != len(self.specs):
            raise ValueError("TargetFitResult specs must be unique.")
        unexpected = set(self.parameters_by_target).difference(spec_names)
        if unexpected:
            raise ValueError(
                f"Fitted parameters reference unrequested targets: {sorted(unexpected)}"
            )

        normalized: dict[str, Mapping[str, Any]] = {}
        for target_name, parameters in self.parameters_by_target.items():
            if not isinstance(parameters, Mapping):
                raise TypeError("Each target's fitted parameters must be a mapping.")
            normalized[target_name] = MappingProxyType(dict(parameters))
        object.__setattr__(
            self,
            "parameters_by_target",
            MappingProxyType(normalized),
        )

        for spec in self.specs:
            fitted_parameters = self.parameters_by_target.get(spec.name)
            if spec.fit_scope is TargetFitScope.TRAINING_FOLD_ONLY:
                if not fitted_parameters:
                    raise ValueError(
                        f"Training-scoped target {spec.name!r} requires fitted parameters."
                    )
            elif fitted_parameters:
                raise ValueError(
                    f"Target {spec.name!r} declares no training estimation but has "
                    "fold-fitted parameters."
                )


@dataclass(frozen=True)
class TargetBuildResult:
    """Forward outcomes tied to the exact fold-local fit used to construct them."""

    frame: pd.DataFrame
    specs: tuple[TargetSpec, ...]
    builder_id: str
    fit_result: TargetFitResult
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("TargetBuildResult.specs must be non-empty.")
        if not self.builder_id.strip():
            raise ValueError("builder_id must be non-empty.")
        if not isinstance(self.fit_result, TargetFitResult):
            raise TypeError("fit_result must be a TargetFitResult.")
        if self.builder_id != self.fit_result.builder_id:
            raise ValueError("Build and fit builder identifiers must match.")
        if self.specs != self.fit_result.specs:
            raise ValueError("Build and fit target specifications must match exactly.")

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def fold_id(self) -> str:
        return self.fit_result.context.fold_id


@runtime_checkable
class TargetBuilder(Protocol):
    """Two-phase protocol for the timing-audited Stage 5 target builder."""

    @property
    def builder_id(self) -> str:
        """Stable implementation identifier recorded in manifests."""

    def fit(
        self,
        training_panel: pd.DataFrame,
        *,
        request: TargetBuildRequest,
        context: TargetTrainingContext,
    ) -> TargetFitResult:
        """Estimate any threshold using exactly the declared training fold."""

    def build(
        self,
        panel: pd.DataFrame,
        *,
        request: TargetBuildRequest,
        fit_result: TargetFitResult,
    ) -> TargetBuildResult:
        """Construct outcomes using immutable parameters from ``fit``."""


def validate_target_result(
    result: TargetBuildResult,
    *,
    key_columns: Sequence[str] = (
        "product",
        "fold_id",
        "information_date",
        "target_date",
    ),
) -> None:
    """Validate target schema, fit provenance, timing, and elementary types."""

    if not isinstance(result.frame, pd.DataFrame):
        raise TypeError("TargetBuildResult.frame must be a pandas DataFrame.")
    if len(key_columns) != 4:
        raise ValueError(
            "key_columns must contain product, fold, information-date, and target-date."
        )

    required = tuple(key_columns) + result.target_columns
    missing = [column for column in required if column not in result.frame.columns]
    if missing:
        raise ValueError(f"Target result is missing columns: {missing}")

    product_column, fold_column, information_column, target_column = key_columns
    observed_folds = set(result.frame[fold_column].dropna().astype(str).unique())
    if observed_folds != {result.fold_id}:
        raise ValueError("Every target row must match its fitted training fold.")

    information_date = result.frame[information_column]
    target_date = result.frame[target_column]
    observed = information_date.notna() & target_date.notna()
    if (target_date[observed] <= information_date[observed]).any():
        raise ValueError("Every observed target date must follow its information date.")
    if result.frame.duplicated([product_column, fold_column, information_column]).any():
        raise ValueError("Target result has duplicate product-fold-information rows.")

    spec_by_name = {spec.name: spec for spec in result.specs}
    for column in result.target_columns:
        if not is_numeric_dtype(result.frame[column]):
            raise TypeError(f"Target column {column!r} must be numeric.")
        if spec_by_name[column].kind is TargetKind.BINARY:
            observed_values = set(result.frame[column].dropna().unique().tolist())
            if not observed_values.issubset({0, 1, False, True}):
                raise ValueError(f"Binary target {column!r} contains non-binary values.")


def fit_target_parameters(
    training_panel: pd.DataFrame,
    *,
    request: TargetBuildRequest,
    context: TargetTrainingContext,
) -> TargetFitResult:
    """Reserved production entry point for Stage 5 fold-local fitting."""

    del training_panel, request, context
    raise TargetConstructionUnavailableError(
        "Training-only target thresholds are fitted in Stage 5.  Stage 2 "
        "provides fold-aware interfaces only."
    )


def build_targets(
    panel: pd.DataFrame,
    *,
    request: TargetBuildRequest,
    fit_result: TargetFitResult,
) -> TargetBuildResult:
    """Reserved production entry point for Stage 5 target construction."""

    del panel, request, fit_result
    raise TargetConstructionUnavailableError(
        "Forward targets and their fold-local fitted thresholds are implemented "
        "in Stage 5.  Stage 2 provides interfaces only."
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")


__all__ = [
    "STAGE_OWNER",
    "TargetBuildRequest",
    "TargetBuildResult",
    "TargetBuilder",
    "TargetConstructionUnavailableError",
    "TargetFitResult",
    "TargetFitScope",
    "TargetKind",
    "TargetSpec",
    "TargetTrainingContext",
    "build_targets",
    "fit_target_parameters",
    "validate_target_result",
]
