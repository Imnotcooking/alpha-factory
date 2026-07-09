"""XGBoost research training engine for engineered alpha features."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from oqp.research.artifacts import ModelArtifactStore
from oqp.research.ml.feature_governance import detect_feature_columns
from oqp.research.model_registry import (
    build_data_fingerprint,
    record_from_artifact,
    register_model_artifact,
)

warnings.filterwarnings("ignore")


class XGBoostTrainingEngine:
    """
    Standalone XGBoost trainer for fixed-date alpha-rank validation.

    The engine trains on a feature matrix, applies a purged fixed-date split,
    exports the model/feature-importance artifacts, and registers the model in
    the promoted OQP model registry.
    """

    def __init__(
        self,
        data_path: str = "runtime/data/feature_store/ML_Feature_Matrix.parquet",
        model_output_path: str = "runtime/artifacts/research/models/xgb_base_model.json",
        importance_output_path: str = "runtime/artifacts/research/feature_importance/feature_importance_fac_057.csv",
        target_col: str = "target_4d_rank",
        split_date: str = "2024-01-01",
        model_name: str = "fac_057_xgboost_base",
        factor_id: str = "fac_057",
        target_horizon_days: int = 4,
        embargo_days: int = 8,
    ):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.importance_output_path = importance_output_path
        self.target_col = target_col
        self.split_date = pd.Timestamp(split_date)
        self.model_name = model_name
        self.factor_id = factor_id
        self.target_horizon_days = int(target_horizon_days)
        self.embargo_days = int(embargo_days)
        self.model: Any | None = None
        self.feature_cols: list[str] = []
        self.training_metrics: dict[str, Any] = {}
        self.split_policy = {
            "mode": "fixed_date_with_embargo",
            "split_date": str(self.split_date.date()),
            "embargo_days": self.embargo_days,
            "target_horizon_days": self.target_horizon_days,
        }

    def _load_and_prepare_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        df = pd.read_parquet(self.data_path).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        self.feature_cols = detect_feature_columns(df, include_prob_features=True)
        if not self.feature_cols:
            raise ValueError("[XGB] No numeric engineered feature columns found.")
        if self.target_col not in df.columns:
            raise ValueError(f"[XGB] Target column not found: {self.target_col}")

        df = df.dropna(subset=["date", *self.feature_cols, self.target_col]).copy()
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        embargo_date = self.split_date - pd.Timedelta(days=self.embargo_days)
        train_df = df[df["date"] < embargo_date].copy()
        valid_df = df[df["date"] >= self.split_date].copy()
        if train_df.empty or valid_df.empty:
            raise ValueError("[XGB] Train/validation split is empty.")

        return (
            train_df,
            train_df[self.feature_cols],
            train_df[self.target_col],
            valid_df[self.feature_cols],
            valid_df[self.target_col],
        )

    def _compute_sample_weights(
        self,
        train_df: pd.DataFrame,
        target_horizon: int | None = None,
    ) -> pd.Series:
        """Return uniqueness and time-decay sample weights."""

        horizon = int(target_horizon or self.target_horizon_days)
        if horizon <= 0:
            raise ValueError("target_horizon must be positive.")
        uniqueness_weight = 1.0 / horizon
        unique_dates = sorted(train_df["date"].dropna().unique())
        date_map = {date: idx for idx, date in enumerate(unique_dates)}
        time_steps = train_df["date"].map(date_map)
        max_step = len(unique_dates)

        if max_step > 1:
            time_decay = 0.5 + 0.5 * (time_steps / max_step)
        else:
            time_decay = pd.Series(1.0, index=train_df.index)
        return uniqueness_weight * time_decay

    def train(self) -> tuple[Any, pd.DataFrame]:
        from xgboost import XGBRegressor

        train_df, x_train, y_train, x_valid, y_valid = self._load_and_prepare_data()
        self.model = XGBRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.03,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        sample_weights = self._compute_sample_weights(train_df)
        self.model.fit(x_train, y_train, sample_weight=sample_weights)

        valid_pred = self.model.predict(x_valid)
        valid_ic = pd.DataFrame(
            {"pred": valid_pred, "target": y_valid.values}
        ).corr(method="spearman").iloc[0, 1]
        self.training_metrics = {
            "validation_spearman_ic": float(valid_ic),
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_valid)),
            "feature_count": int(len(self.feature_cols)),
        }

        importance_df = pd.DataFrame(
            {"feature": self.feature_cols, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)
        return self.model, importance_df

    def save_artifacts(self, importance_df: pd.DataFrame) -> None:
        if self.model is None:
            raise ValueError("Model must be trained before saving artifacts.")

        model_path = Path(self.model_output_path)
        importance_path = Path(self.importance_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        importance_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(model_path)
        importance_df.to_csv(importance_path, index=False)

        stored = ModelArtifactStore().archive_file(
            model_path,
            model_name=self.model_name,
        )
        data_fingerprint = build_data_fingerprint(self.data_path)
        record = record_from_artifact(
            artifact_id=stored.artifact_id,
            model_name=self.model_name,
            factor_id=self.factor_id,
            model_type="xgboost_regressor",
            artifact_path=stored.path,
            artifact_format="xgboost_json",
            legacy_path=model_path,
            source_module="oqp.research.ml.xgboost_model",
            data_fingerprint=data_fingerprint,
            feature_cols=list(self.feature_cols),
            target_col=self.target_col,
            split_policy=self.split_policy,
            metrics=self.training_metrics,
            hyperparams=self.model.get_params(),
            metadata={"importance_output_path": importance_path.as_posix()},
        )
        register_model_artifact(record)

    def run(self) -> tuple[Any, pd.DataFrame]:
        model, importance_df = self.train()
        self.save_artifacts(importance_df)
        return model, importance_df


__all__ = ["XGBoostTrainingEngine"]
