"""Common interfaces for the preregistered baseline model ladder.

Stage 6 implements unconditional risk, EWMA, HAR, PCA plus k-means, and iid
Gaussian-mixture benchmarks.  Stage 2 supplies a single fit/predict/artifact
contract so later comparisons cannot quietly change their output geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


STAGE_OWNER = 6


class BaselineFamily(str, Enum):
    UNCONDITIONAL_RISK = "unconditional_risk"
    EWMA = "ewma"
    HAR = "har"
    PCA_KMEANS = "pca_kmeans"
    IID_GMM = "iid_gmm"


class BaselineUnavailableError(NotImplementedError):
    """Raised when baseline fitting is requested before Stage 6."""


@dataclass(frozen=True)
class BaselineConfig:
    """One deterministic baseline specification."""

    family: BaselineFamily
    seed: int
    feature_columns: tuple[str, ...] = ()
    state_count: int | None = None
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, BaselineFamily):
            raise TypeError("family must be a BaselineFamily value.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("feature_columns must be unique.")
        if self.state_count is not None and self.state_count < 2:
            raise ValueError("state_count must be at least two when supplied.")
        if self.family in {BaselineFamily.PCA_KMEANS, BaselineFamily.IID_GMM}:
            if not self.feature_columns:
                raise ValueError(f"{self.family.value} requires feature_columns.")
            if self.state_count is None:
                raise ValueError(f"{self.family.value} requires state_count.")


@dataclass(frozen=True)
class BaselineFitContext:
    """Fold and chronology recorded with every fitted baseline."""

    fold_id: str
    training_start: date
    training_end: date

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty.")
        if self.training_end < self.training_start:
            raise ValueError("training_end cannot precede training_start.")


@dataclass(frozen=True)
class BaselinePredictionResult:
    """Comparable fold-level predictions emitted by any baseline family."""

    frame: pd.DataFrame
    model_id: str
    family: BaselineFamily
    fold_id: str
    prediction_columns: tuple[str, ...]
    probability_columns: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, BaselineFamily):
            raise TypeError("family must be a BaselineFamily value.")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty.")
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty.")
        if not self.prediction_columns:
            raise ValueError("prediction_columns must be non-empty.")
        all_outputs = self.prediction_columns + self.probability_columns
        if len(all_outputs) != len(set(all_outputs)):
            raise ValueError("Prediction and probability column names must be unique.")


@runtime_checkable
class FittedBaseline(Protocol):
    """Immutable model fitted inside one training fold."""

    @property
    def model_id(self) -> str:
        """Stable model identifier."""

    @property
    def family(self) -> BaselineFamily:
        """Preregistered family represented by this object."""

    def predict(self, frame: pd.DataFrame) -> BaselinePredictionResult:
        """Emit prospective predictions without refitting."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return JSON-serializable fitted parameters and training metadata."""


@runtime_checkable
class BaselineEstimator(Protocol):
    """Factory protocol for a deterministic Stage 6 implementation."""

    def fit(
        self,
        training_frame: pd.DataFrame,
        *,
        config: BaselineConfig,
        context: BaselineFitContext,
    ) -> FittedBaseline:
        """Fit using training rows only."""


def validate_baseline_predictions(
    result: BaselinePredictionResult,
    *,
    key_columns: Sequence[str] = ("product", "information_date", "forecast_date"),
    probability_tolerance: float = 1e-10,
) -> None:
    """Validate common output geometry and probability invariants."""

    if not isinstance(result.frame, pd.DataFrame):
        raise TypeError("BaselinePredictionResult.frame must be a pandas DataFrame.")
    required = tuple(key_columns) + result.prediction_columns + result.probability_columns
    missing = [column for column in required if column not in result.frame.columns]
    if missing:
        raise ValueError(f"Baseline predictions are missing columns: {missing}")

    duplicate_keys = list(tuple(key_columns)[:2])
    if duplicate_keys and result.frame.duplicated(duplicate_keys).any():
        raise ValueError("Baseline predictions have duplicate product-information rows.")

    for column in result.prediction_columns + result.probability_columns:
        if not is_numeric_dtype(result.frame[column]):
            raise TypeError(f"Prediction column {column!r} must be numeric.")

    if result.probability_columns:
        probabilities = result.frame.loc[:, result.probability_columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Probability outputs must be finite.")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("Probability outputs must lie in [0, 1].")
        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=probability_tolerance,
        ):
            raise ValueError("Probability rows must sum to one.")


def create_baseline(config: BaselineConfig) -> BaselineEstimator:
    """Reserved production factory for the gated Stage 6 implementations."""

    del config
    raise BaselineUnavailableError(
        "Unconditional, EWMA, HAR, PCA-k-means, and iid-GMM baselines are "
        "implemented in Stage 6.  Stage 2 provides interfaces only."
    )


__all__ = [
    "BaselineConfig",
    "BaselineEstimator",
    "BaselineFamily",
    "BaselineFitContext",
    "BaselinePredictionResult",
    "BaselineUnavailableError",
    "FittedBaseline",
    "STAGE_OWNER",
    "create_baseline",
    "validate_baseline_predictions",
]
