"""Training-fold-only preprocessing for daily-regime representations.

Every learned quantity is fitted per product from rows inside one declared
training interval.  Validation rows can be transformed but can never alter the
stored clipping bounds, means, scales, or PCA loadings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


STAGE_OWNER = 4
FIT_SCOPE = "training_fold_only"
TRANSFORMER_VERSION = "fold_local_product_preprocessor_v1"
STANDARDIZED_COLUMN_NAMES: Mapping[str, str] = {
    "log_gk_gap_variance": "log_gk_gap_variance_z",
    "log_amihud": "log_amihud_z",
    "ker_20d": "ker_20d_z",
}


class PreprocessingUnavailableError(NotImplementedError):
    """Backward-compatible error type retained from the Stage 2 interface."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Leakage-safe preprocessing declaration."""

    fit_scope: str = FIT_SCOPE
    standardize: bool = True
    clip_quantiles: tuple[float, float] | None = None
    pca_components: int | None = None
    missing_value_policy: str = "reject_after_warmup"
    group_column: str = "product"
    date_column: str = "trading_date"

    def __post_init__(self) -> None:
        if self.fit_scope != FIT_SCOPE:
            raise ValueError("Preprocessing fit_scope must be 'training_fold_only'.")
        if self.clip_quantiles is not None:
            lower, upper = self.clip_quantiles
            if not 0.0 <= lower < upper <= 1.0:
                raise ValueError("clip_quantiles must satisfy 0 <= lower < upper <= 1.")
        if self.pca_components is not None and self.pca_components < 1:
            raise ValueError("pca_components must be positive when supplied.")
        if self.pca_components is not None and not self.standardize:
            raise ValueError("HPCA requires product-relative standardization.")
        if self.missing_value_policy != "reject_after_warmup":
            raise ValueError("Stage 4 supports only reject_after_warmup missingness.")
        if not self.group_column.strip() or not self.date_column.strip():
            raise ValueError("group_column and date_column must be non-empty.")


@dataclass(frozen=True)
class PreprocessingFitContext:
    """Training interval and optional caller-supplied row identity."""

    fold_id: str
    training_start: date
    training_end: date
    seed: int
    training_rows_hash: str | None = None
    training_row_count: int | None = None

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty.")
        if self.training_end < self.training_start:
            raise ValueError("training_end cannot precede training_start.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if self.training_rows_hash is not None:
            value = self.training_rows_hash.lower()
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(
                    "training_rows_hash must be a 64-character SHA-256 hex digest."
                )
        if self.training_row_count is not None and self.training_row_count < 1:
            raise ValueError("training_row_count must be positive when supplied.")


@dataclass(frozen=True)
class PreprocessingResult:
    """Transformed frame plus the exact columns handed to a model."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    transformer_id: str
    fold_id: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must be non-empty.")
        if not self.transformer_id.strip():
            raise ValueError("transformer_id must be non-empty.")
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty.")


@runtime_checkable
class FittedPreprocessor(Protocol):
    """A serializable transformer fitted on exactly one training fold."""

    @property
    def transformer_id(self) -> str:
        """Stable implementation identifier."""

    @property
    def fold_id(self) -> str:
        """Fold on which this object was fitted."""

    def transform(self, frame: pd.DataFrame) -> PreprocessingResult:
        """Apply immutable training parameters to a chronological frame."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return JSON-serializable learned parameters for audit artifacts."""


@runtime_checkable
class Preprocessor(Protocol):
    """Factory protocol that learns transformations from training rows only."""

    def fit(
        self,
        training_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        config: PreprocessingConfig,
        context: PreprocessingFitContext,
    ) -> FittedPreprocessor:
        """Fit without observing validation or holdout rows."""


@dataclass(frozen=True)
class _FeatureState:
    lower: float | None
    upper: float | None
    mean: float
    scale: float
    zero_variance: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "scale": self.scale,
            "zero_variance": bool(self.zero_variance),
        }


@dataclass(frozen=True)
class FittedFoldLocalPreprocessor:
    """Immutable product-relative scaler and optional global HPCA transform."""

    fold_id: str
    feature_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    config: PreprocessingConfig
    context: PreprocessingFitContext
    group_state: Mapping[str, Mapping[str, _FeatureState]]
    training_rows_hash: str
    training_row_count: int
    pca_mean: tuple[float, ...] | None = None
    pca_components: tuple[tuple[float, ...], ...] | None = None
    pca_explained_variance_ratio: tuple[float, ...] | None = None
    state_hash: str = ""

    @property
    def transformer_id(self) -> str:
        return f"{TRANSFORMER_VERSION}:{self.state_hash[:12]}"

    def transform(self, frame: pd.DataFrame) -> PreprocessingResult:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        required = {self.config.group_column, *self.feature_columns}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Preprocessing transform is missing columns: {missing}")

        transformed = frame.copy(deep=True)
        scaled = pd.DataFrame(
            np.nan,
            index=transformed.index,
            columns=self.feature_columns,
            dtype=float,
        )
        groups = transformed[self.config.group_column].astype(str)
        unseen_products: list[str] = []
        zero_variance_cells = 0
        for group_name, indices in groups.groupby(groups, sort=False).groups.items():
            if group_name not in self.group_state:
                unseen_products.append(group_name)
                continue
            group_parameters = self.group_state[group_name]
            index = pd.Index(indices)
            for feature in self.feature_columns:
                values = pd.to_numeric(transformed.loc[index, feature], errors="coerce")
                parameters = group_parameters[feature]
                if parameters.lower is not None:
                    values = values.clip(lower=parameters.lower, upper=parameters.upper)
                if parameters.zero_variance:
                    valid = values.notna() & np.isfinite(values)
                    scaled.loc[index[valid.to_numpy()], feature] = 0.0
                    zero_variance_cells += int(valid.sum())
                elif self.config.standardize:
                    scaled.loc[index, feature] = (
                        values - parameters.mean
                    ) / parameters.scale
                else:
                    scaled.loc[index, feature] = values

        finite_or_missing = scaled.apply(
            lambda values: values.isna() | np.isfinite(values), axis=0
        )
        if not bool(finite_or_missing.all(axis=None)):
            raise ValueError("Preprocessing produced non-finite transformed values.")

        if self.pca_components is not None:
            assert self.pca_mean is not None
            matrix = scaled.to_numpy(dtype=float)
            complete = np.isfinite(matrix).all(axis=1)
            pca_values = np.full((len(matrix), len(self.pca_components)), np.nan)
            if complete.any():
                centered = matrix[complete] - np.asarray(self.pca_mean, dtype=float)
                loadings = np.asarray(self.pca_components, dtype=float)
                pca_values[complete] = centered @ loadings.T
            for index, column in enumerate(self.output_columns):
                transformed[column] = pca_values[:, index]
        else:
            for source, destination in zip(self.feature_columns, self.output_columns):
                transformed[destination] = scaled[source]

        result = PreprocessingResult(
            frame=transformed,
            feature_columns=self.output_columns,
            transformer_id=self.transformer_id,
            fold_id=self.fold_id,
            diagnostics={
                "fit_scope": FIT_SCOPE,
                "training_rows_hash": self.training_rows_hash,
                "training_row_count": self.training_row_count,
                "unseen_products": sorted(set(unseen_products)),
                "incomplete_rows": int(
                    transformed[list(self.output_columns)].isna().any(axis=1).sum()
                ),
                "zero_variance_cells_set_to_zero": zero_variance_cells,
                "scientific_evidence": False,
            },
        )
        validate_preprocessing_result(result)
        return result

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "version": TRANSFORMER_VERSION,
            "state_hash": self.state_hash,
            "fold_id": self.fold_id,
            "fit_scope": FIT_SCOPE,
            "feature_columns": list(self.feature_columns),
            "output_columns": list(self.output_columns),
            "group_column": self.config.group_column,
            "date_column": self.config.date_column,
            "standardize": self.config.standardize,
            "clip_quantiles": (
                list(self.config.clip_quantiles)
                if self.config.clip_quantiles is not None
                else None
            ),
            "training_start": self.context.training_start.isoformat(),
            "training_end": self.context.training_end.isoformat(),
            "training_rows_hash": self.training_rows_hash,
            "training_row_count": self.training_row_count,
            "groups": {
                group: {
                    feature: parameters.to_dict()
                    for feature, parameters in sorted(features.items())
                }
                for group, features in sorted(self.group_state.items())
            },
            "pca": (
                {
                    "mean": list(self.pca_mean or ()),
                    "components": [
                        list(component) for component in self.pca_components or ()
                    ],
                    "explained_variance_ratio": list(
                        self.pca_explained_variance_ratio or ()
                    ),
                    "sign_rule": "largest_absolute_loading_positive",
                }
                if self.pca_components is not None
                else None
            ),
        }


class FoldLocalPreprocessor:
    """Fit clipping, product z-scores, and optional HPCA on training rows only."""

    def fit(
        self,
        training_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        config: PreprocessingConfig,
        context: PreprocessingFitContext,
    ) -> FittedFoldLocalPreprocessor:
        if not isinstance(training_frame, pd.DataFrame):
            raise TypeError("training_frame must be a pandas DataFrame.")
        features = tuple(feature_columns)
        if not features or len(features) != len(set(features)):
            raise ValueError("feature_columns must be non-empty and unique.")
        required = {config.group_column, config.date_column, *features}
        missing = sorted(required.difference(training_frame.columns))
        if missing:
            raise ValueError(f"Preprocessing fit is missing columns: {missing}")
        if training_frame.empty:
            raise ValueError("Preprocessing requires non-empty training rows.")

        dates = pd.to_datetime(
            training_frame[config.date_column], errors="raise"
        ).dt.normalize()
        start = pd.Timestamp(context.training_start)
        end = pd.Timestamp(context.training_end)
        if ((dates < start) | (dates > end)).any():
            raise ValueError(
                "training_frame contains rows outside the declared training interval."
            )

        groups = training_frame[config.group_column].astype(str)
        if groups.str.strip().eq("").any():
            raise ValueError("Training product/group identifiers must be non-empty.")
        numeric = training_frame[list(features)].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                "Training features contain missing or non-finite values; select the "
                "fold's common-complete post-warmup rows before fitting."
            )

        canonical = pd.DataFrame(
            {config.group_column: groups, config.date_column: dates},
            index=training_frame.index,
        )
        for feature in features:
            canonical[feature] = numeric[feature]
        if canonical.duplicated([config.group_column, config.date_column]).any():
            raise ValueError("Training features contain duplicate product-date rows.")
        rows_hash = _training_rows_hash(
            canonical,
            sort_columns=(config.group_column, config.date_column),
        )
        canonical_order = canonical.sort_values(
            [config.group_column, config.date_column], kind="mergesort"
        ).index
        groups = groups.loc[canonical_order]
        numeric = numeric.loc[canonical_order]
        row_count = int(len(training_frame))
        if (
            context.training_rows_hash is not None
            and context.training_rows_hash != rows_hash
        ):
            raise ValueError(
                "training_rows_hash does not match the supplied training frame."
            )
        if (
            context.training_row_count is not None
            and context.training_row_count != row_count
        ):
            raise ValueError(
                "training_row_count does not match the supplied training frame."
            )

        group_state: dict[str, dict[str, _FeatureState]] = {}
        fitted_scaled = pd.DataFrame(index=numeric.index, columns=features, dtype=float)
        for group_name, indices in groups.groupby(groups, sort=True).groups.items():
            index = pd.Index(indices)
            group_state[group_name] = {}
            for feature in features:
                series = numeric.loc[index, feature].astype(float)
                lower: float | None = None
                upper: float | None = None
                if config.clip_quantiles is not None:
                    lower = float(series.quantile(config.clip_quantiles[0]))
                    upper = float(series.quantile(config.clip_quantiles[1]))
                    series = series.clip(lower=lower, upper=upper)
                mean = float(series.mean()) if config.standardize else 0.0
                standard_deviation = (
                    float(series.std(ddof=0)) if config.standardize else 1.0
                )
                zero_variance = bool(
                    config.standardize and standard_deviation <= np.finfo(float).eps
                )
                scale = 1.0 if zero_variance else standard_deviation
                parameters = _FeatureState(
                    lower=lower,
                    upper=upper,
                    mean=mean,
                    scale=scale,
                    zero_variance=zero_variance,
                )
                group_state[group_name][feature] = parameters
                if zero_variance:
                    fitted_scaled.loc[index, feature] = 0.0
                elif config.standardize:
                    fitted_scaled.loc[index, feature] = (series - mean) / scale
                else:
                    fitted_scaled.loc[index, feature] = series

        pca_mean: tuple[float, ...] | None = None
        components: tuple[tuple[float, ...], ...] | None = None
        explained: tuple[float, ...] | None = None
        if config.pca_components is not None:
            if config.pca_components > len(features):
                raise ValueError(
                    "pca_components cannot exceed the source feature count."
                )
            if config.pca_components > len(fitted_scaled):
                raise ValueError("pca_components cannot exceed the training row count.")
            matrix = fitted_scaled.to_numpy(dtype=float)
            center = matrix.mean(axis=0)
            centered = matrix - center
            _, singular_values, right_vectors = np.linalg.svd(
                centered, full_matrices=False
            )
            selected = right_vectors[: config.pca_components].copy()
            for component in selected:
                pivot = int(np.argmax(np.abs(component)))
                if component[pivot] < 0.0:
                    component *= -1.0
            variance = singular_values**2 / max(len(matrix) - 1, 1)
            total = float(variance.sum())
            ratio = (
                variance[: config.pca_components] / total
                if total > 0.0
                else np.zeros(config.pca_components)
            )
            pca_mean = tuple(float(value) for value in center)
            components = tuple(tuple(float(value) for value in row) for row in selected)
            explained = tuple(float(value) for value in ratio)

        output_columns = (
            tuple(f"hpca_{index + 1}" for index in range(config.pca_components))
            if config.pca_components is not None
            else tuple(STANDARDIZED_COLUMN_NAMES.get(name, name) for name in features)
        )
        payload = _state_payload(
            fold_id=context.fold_id,
            feature_columns=features,
            output_columns=output_columns,
            config=config,
            context=context,
            group_state=group_state,
            training_rows_hash=rows_hash,
            training_row_count=row_count,
            pca_mean=pca_mean,
            pca_components=components,
            pca_explained_variance_ratio=explained,
        )
        state_hash = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        return FittedFoldLocalPreprocessor(
            fold_id=context.fold_id,
            feature_columns=features,
            output_columns=output_columns,
            config=config,
            context=context,
            group_state=group_state,
            training_rows_hash=rows_hash,
            training_row_count=row_count,
            pca_mean=pca_mean,
            pca_components=components,
            pca_explained_variance_ratio=explained,
            state_hash=state_hash,
        )


def fit_preprocessor(
    training_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: PreprocessingConfig,
    context: PreprocessingFitContext,
) -> FittedPreprocessor:
    """Fit the audited Stage 4 fold-local preprocessor."""

    return FoldLocalPreprocessor().fit(
        training_frame,
        feature_columns=feature_columns,
        config=config,
        context=context,
    )


def validate_preprocessing_result(result: PreprocessingResult) -> None:
    """Validate alignment and reject infinities while permitting warm-up missingness."""

    if not isinstance(result.frame, pd.DataFrame):
        raise TypeError("PreprocessingResult.frame must be a pandas DataFrame.")
    missing = [
        column
        for column in result.feature_columns
        if column not in result.frame.columns
    ]
    if missing:
        raise ValueError(f"Preprocessing result is missing features: {missing}")
    for column in result.feature_columns:
        numeric = pd.to_numeric(result.frame[column], errors="coerce")
        finite = numeric.dropna().to_numpy(dtype=float)
        if finite.size and not np.isfinite(finite).all():
            raise ValueError(f"Preprocessing feature {column!r} contains infinities.")


def _training_rows_hash(
    frame: pd.DataFrame,
    *,
    sort_columns: tuple[str, ...],
) -> str:
    ordered = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(
        drop=True
    )
    digest = hashlib.sha256()
    digest.update(
        "\x1f".join(str(column) for column in ordered.columns).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _state_payload(
    *,
    fold_id: str,
    feature_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    config: PreprocessingConfig,
    context: PreprocessingFitContext,
    group_state: Mapping[str, Mapping[str, _FeatureState]],
    training_rows_hash: str,
    training_row_count: int,
    pca_mean: tuple[float, ...] | None,
    pca_components: tuple[tuple[float, ...], ...] | None,
    pca_explained_variance_ratio: tuple[float, ...] | None,
) -> dict[str, Any]:
    return {
        "version": TRANSFORMER_VERSION,
        "fold_id": fold_id,
        "feature_columns": list(feature_columns),
        "output_columns": list(output_columns),
        "config": {
            "fit_scope": config.fit_scope,
            "standardize": config.standardize,
            "clip_quantiles": (
                list(config.clip_quantiles) if config.clip_quantiles else None
            ),
            "pca_components": config.pca_components,
            "missing_value_policy": config.missing_value_policy,
            "group_column": config.group_column,
            "date_column": config.date_column,
        },
        "context": {
            "training_start": context.training_start.isoformat(),
            "training_end": context.training_end.isoformat(),
            "seed": context.seed,
        },
        "training_rows_hash": training_rows_hash,
        "training_row_count": training_row_count,
        "groups": {
            group: {
                feature: parameters.to_dict()
                for feature, parameters in sorted(features.items())
            }
            for group, features in sorted(group_state.items())
        },
        "pca_mean": list(pca_mean) if pca_mean is not None else None,
        "pca_components": (
            [list(component) for component in pca_components]
            if pca_components is not None
            else None
        ),
        "pca_explained_variance_ratio": (
            list(pca_explained_variance_ratio)
            if pca_explained_variance_ratio is not None
            else None
        ),
    }


__all__ = [
    "FIT_SCOPE",
    "FittedFoldLocalPreprocessor",
    "FittedPreprocessor",
    "FoldLocalPreprocessor",
    "PreprocessingConfig",
    "PreprocessingFitContext",
    "PreprocessingResult",
    "PreprocessingUnavailableError",
    "Preprocessor",
    "STANDARDIZED_COLUMN_NAMES",
    "STAGE_OWNER",
    "TRANSFORMER_VERSION",
    "fit_preprocessor",
    "validate_preprocessing_result",
]
