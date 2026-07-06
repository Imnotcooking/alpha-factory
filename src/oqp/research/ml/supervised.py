"""Shared supervised-model plumbing for research ML experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oqp.research.ml.feature_governance import detect_feature_columns


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for expanding-window supervised ML validation."""

    min_train_days: int = 756
    test_window_days: int = 60
    purge_gap_days: int = 2
    include_prob_features: bool = True


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
        walk_forward_config: WalkForwardConfig | None = None,
    ):
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.walk_forward_config = walk_forward_config or WalkForwardConfig()
        self.model = None
        self.feature_cols: list[str] = []

    def load_feature_matrix(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path).copy()
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
            include_prob_features=self.walk_forward_config.include_prob_features,
        )
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

        config = self.walk_forward_config
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

    @abstractmethod
    def train(self) -> tuple:
        """Train the model and return model-specific artifacts."""
        raise NotImplementedError


class BaseMLModel(SupervisedModelBase):
    """Backwards-compatible alias for legacy supervised ML models."""


__all__ = ["BaseMLModel", "SupervisedModelBase", "WalkForwardConfig"]
