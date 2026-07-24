"""Shared supervised-regression task and validation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from oqp.research.ml.features.governance import (
    detect_feature_columns,
    infer_feature_matrix_asset_class,
    scope_feature_matrix,
)


@dataclass(frozen=True)
class ValidationConfig:
    """Shared chronological validation policy for supervised ML experiments."""

    mode: Literal["walk_forward", "fixed_date"] = "walk_forward"
    min_train_days: int = 756
    test_window_days: int = 60
    purge_gap_days: int = 2
    include_prob_features: bool = False
    split_date: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"walk_forward", "fixed_date"}:
            raise ValueError(f"Unsupported validation mode: {self.mode}")
        if self.min_train_days <= 0:
            raise ValueError("min_train_days must be positive.")
        if self.test_window_days <= 0:
            raise ValueError("test_window_days must be positive.")
        if self.purge_gap_days < 0:
            raise ValueError("purge_gap_days cannot be negative.")
        if self.mode == "fixed_date" and not self.split_date:
            raise ValueError("fixed_date validation requires split_date.")

    def as_dict(self) -> dict[str, int | bool | str | None]:
        return {
            "mode": self.mode,
            "min_train_days": self.min_train_days,
            "test_window_days": self.test_window_days,
            "purge_gap_days": self.purge_gap_days,
            "include_prob_features": self.include_prob_features,
            "split_date": self.split_date,
        }


# Public compatibility name retained for existing research callers.
WalkForwardConfig = ValidationConfig


class SupervisedModelBase(ABC):
    """
    Shared base class for supervised alpha models.

    The class owns reusable research plumbing: loading a feature matrix,
    detecting valid feature columns, applying chronological walk-forward splits,
    and enforcing a purge gap between train and test.
    """

    def __init__(
        self,
        data_path: str | Path,
        target_col: str = "target_1d_rank",
        validation_config: ValidationConfig | None = None,
        walk_forward_config: ValidationConfig | None = None,
        asset_class: str | None = None,
        exclude_features: list[str] | tuple[str, ...] | None = None,
    ):
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.asset_class = asset_class
        self.exclude_features = tuple(exclude_features or ())
        if validation_config is not None and walk_forward_config is not None:
            raise ValueError(
                "Pass validation_config or walk_forward_config, not both."
            )
        self.validation_config = (
            validation_config or walk_forward_config or ValidationConfig()
        )
        self.walk_forward_config = self.validation_config
        self.model = None
        self.feature_cols: list[str] = []

    def load_feature_matrix(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path).copy()
        if self.asset_class:
            df = scope_feature_matrix(
                df,
                self.asset_class,
                default_asset_class=infer_feature_matrix_asset_class(self.data_path),
            )
            if df.empty:
                raise ValueError(
                    f"Feature matrix has no rows for asset class {self.asset_class}."
                )
        if "date" not in df.columns:
            raise ValueError("Feature matrix must contain a 'date' column.")
        if "ticker" not in df.columns:
            raise ValueError("Feature matrix must contain a 'ticker' column.")
        if self.target_col not in df.columns:
            raise ValueError(f"Feature matrix missing target column: {self.target_col}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["ticker"] = df["ticker"].astype(str)
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)

    def prepare_supervised_matrix(self) -> tuple[pd.DataFrame, list[str]]:
        df = self.load_feature_matrix()
        self.feature_cols = detect_feature_columns(
            df,
            include_prob_features=self.validation_config.include_prob_features,
        )
        self.feature_cols = [
            feature
            for feature in self.feature_cols
            if feature not in self.exclude_features
        ]
        if not self.feature_cols:
            raise ValueError("No usable feature columns were detected.")

        df = df.dropna(subset=[self.target_col, *self.feature_cols]).copy()
        if df.empty:
            raise ValueError("No rows remain after dropping missing features/target.")
        return df, self.feature_cols

    def generate_walk_forward_folds(
        self,
        min_train_days: int | None = None,
        test_window_days: int | None = None,
        purge_gap_days: int | None = None,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
        """
        Yield chronological expanding-window folds.

        Parameters are optional for compatibility with legacy alpha-lab callers.
        When omitted, values come from ``WalkForwardConfig``.
        """

        config = self.validation_config
        min_train = int(
            min_train_days if min_train_days is not None else config.min_train_days
        )
        test_window = int(
            test_window_days
            if test_window_days is not None
            else config.test_window_days
        )
        purge_gap = int(
            purge_gap_days if purge_gap_days is not None else config.purge_gap_days
        )

        if min_train <= 0:
            raise ValueError("min_train_days must be positive.")
        if test_window <= 0:
            raise ValueError("test_window_days must be positive.")
        if purge_gap < 0:
            raise ValueError("purge_gap_days cannot be negative.")

        df, features = self.prepare_supervised_matrix()
        unique_dates = pd.Index(df["date"].sort_values().dropna().unique())
        total_days = len(unique_dates)

        for start_test_idx in range(min_train, total_days, test_window):
            end_test_idx = min(start_test_idx + test_window, total_days)
            train_end_idx = max(0, start_test_idx - purge_gap)
            train_dates = unique_dates[:train_end_idx]
            test_dates = unique_dates[start_test_idx:end_test_idx]

            train_df = df[df["date"].isin(train_dates)].copy()
            test_df = df[df["date"].isin(test_dates)].copy()
            if not train_df.empty and not test_df.empty:
                yield train_df, test_df, features

    def generate_validation_folds(
        self,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
        """Yield folds under the trainer's declared chronological policy."""

        config = self.validation_config
        if config.mode == "walk_forward":
            yield from self.generate_walk_forward_folds()
            return

        df, features = self.prepare_supervised_matrix()
        split_date = pd.Timestamp(config.split_date)
        embargo_date = split_date - pd.Timedelta(days=config.purge_gap_days)
        train_df = df[df["date"] < embargo_date].copy()
        test_df = df[df["date"] >= split_date].copy()
        if train_df.empty or test_df.empty:
            raise ValueError("Fixed-date train/validation split is empty.")
        yield train_df, test_df, features

    def validation_policy(self) -> dict[str, int | bool | str | None]:
        """Return the serializable policy stored with every experiment."""

        return self.validation_config.as_dict()

    @abstractmethod
    def train(self) -> tuple:
        """Train the model and return model-specific artifacts."""
        raise NotImplementedError


class BaseMLModel(SupervisedModelBase):
    """Backwards-compatible alias for legacy supervised ML models."""


__all__ = [
    "BaseMLModel",
    "SupervisedModelBase",
    "ValidationConfig",
    "WalkForwardConfig",
]
