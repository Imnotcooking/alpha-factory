"""XGBoost implementation of the shared supervised-regression contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.research.artifacts import ModelArtifactStore
from oqp.research.ml.regression.experiments import (
    MLExperimentResult,
    mean_daily_rank_ic,
    persist_ml_experiment,
    register_failed_ml_experiment,
)
from oqp.research.ml.regression.base import SupervisedModelBase, ValidationConfig
from oqp.research.model_registry import DEFAULT_RESEARCH_DB_PATH


def _default_xgboost_validation() -> ValidationConfig:
    return ValidationConfig(
        mode="fixed_date",
        split_date="2024-01-01",
        purge_gap_days=8,
    )


@dataclass(frozen=True)
class XGBoostModelConfig:
    target_col: str = "target_4d_rank"
    target_horizon_days: int = 4
    validation: ValidationConfig = field(default_factory=_default_xgboost_validation)
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.03,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    def __post_init__(self) -> None:
        if self.target_horizon_days <= 0:
            raise ValueError("target_horizon_days must be positive.")


class XGBoostRegressorTrainer(SupervisedModelBase):
    """XGBoost regressor with standardized validation and experiment output."""

    model_type = "xgboost"
    artifact_format = "xgboost_json"

    def __init__(
        self,
        data_path: str | Path = "runtime/data/feature_store/ML_Feature_Matrix.parquet",
        config: XGBoostModelConfig | None = None,
        *,
        target_col: str | None = None,
        validation_config: ValidationConfig | None = None,
        model_output_path: str | Path = "runtime/artifacts/research/models/xgb_base_model.json",
        importance_output_path: str | Path = "runtime/artifacts/research/feature_importance/feature_importance_fac_057.csv",
        predictions_output_path: str | Path = "runtime/artifacts/research/predictions/predictions_fac_057.parquet",
        model_name: str = "fac_057_xgboost_base",
        factor_id: str | None = "fac_057",
        asset_class: str | None = None,
        exclude_features: list[str] | tuple[str, ...] | None = None,
        registry_db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
        artifact_store: ModelArtifactStore | None = None,
        target_horizon_days: int | None = None,
        split_date: str | None = None,
        embargo_days: int | None = None,
    ):
        base_config = config or XGBoostModelConfig()
        horizon = int(target_horizon_days or base_config.target_horizon_days)
        resolved_validation = validation_config or base_config.validation
        if validation_config is None and (split_date or embargo_days is not None):
            resolved_validation = ValidationConfig(
                mode="fixed_date",
                split_date=split_date or base_config.validation.split_date,
                purge_gap_days=(
                    int(embargo_days)
                    if embargo_days is not None
                    else base_config.validation.purge_gap_days
                ),
                include_prob_features=base_config.validation.include_prob_features,
            )

        self.config = base_config
        self.target_horizon_days = horizon
        self.model_name = model_name
        self.factor_id = factor_id
        self.model_output_path = Path(model_output_path)
        self.importance_output_path = Path(importance_output_path)
        self.predictions_output_path = Path(predictions_output_path)
        self.registry_db_path = Path(registry_db_path)
        self.artifact_store = artifact_store
        self.training_metrics: dict[str, Any] = {}
        super().__init__(
            data_path=data_path,
            target_col=target_col or base_config.target_col,
            validation_config=resolved_validation,
            asset_class=asset_class,
            exclude_features=exclude_features,
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

    def train(self) -> MLExperimentResult:
        from xgboost import XGBRegressor

        fold_importances: list[pd.DataFrame] = []
        oos_predictions: list[pd.DataFrame] = []
        train_rows = 0

        for fold_number, (train_df, test_df, features) in enumerate(
            self.generate_validation_folds(),
            start=1,
        ):
            model = XGBRegressor(**self.config.params)
            model.fit(
                train_df[features],
                train_df[self.target_col],
                sample_weight=self._compute_sample_weights(train_df),
            )
            self.model = model
            train_rows += len(train_df)

            fold_importances.append(
                pd.DataFrame(
                    {
                        "feature": features,
                        "importance": model.feature_importances_,
                        "fold": fold_number,
                    }
                )
            )
            oos_predictions.append(
                pd.DataFrame(
                    {
                        "date": test_df["date"].to_numpy(),
                        "ticker": test_df["ticker"].astype(str).to_numpy(),
                        "target": test_df[self.target_col].to_numpy(),
                        "prediction": model.predict(test_df[features]),
                        "fold": fold_number,
                    }
                )
            )

        if not oos_predictions or self.model is None:
            raise ValueError("[XGBoost] Validation produced no valid folds.")

        predictions = pd.concat(oos_predictions, ignore_index=True)
        importance = (
            pd.concat(fold_importances, ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        self.training_metrics = {
            "oos_mean_daily_spearman_ic": mean_daily_rank_ic(predictions),
            "train_rows_across_folds": int(train_rows),
            "oos_rows": int(len(predictions)),
            "fold_count": int(predictions["fold"].nunique()),
            "feature_count": int(len(self.feature_cols)),
        }
        return MLExperimentResult(
            model_type=self.model_type,
            model_name=self.model_name,
            model=self.model,
            factor_id=self.factor_id,
            asset_class=self.asset_class,
            data_path=self.data_path.as_posix(),
            target_col=self.target_col,
            feature_cols=list(self.feature_cols),
            predictions=predictions,
            feature_importance=importance,
            validation_policy=self.validation_policy(),
            metrics=dict(self.training_metrics),
            hyperparams=dict(self.config.params),
        )

    def run(self) -> MLExperimentResult:
        try:
            result = self.train()
            return persist_ml_experiment(
                result,
                model_output_path=self.model_output_path,
                importance_output_path=self.importance_output_path,
                predictions_output_path=self.predictions_output_path,
                artifact_format=self.artifact_format,
                source_module="oqp.research.ml.tree_based.xgboost",
                db_path=self.registry_db_path,
                artifact_store=self.artifact_store,
            )
        except Exception as exc:
            register_failed_ml_experiment(
                model_type=self.model_type,
                model_name=self.model_name,
                factor_id=self.factor_id,
                asset_class=self.asset_class,
                data_path=self.data_path,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                validation_policy=self.validation_policy(),
                hyperparams=self.config.params,
                error=exc,
                db_path=self.registry_db_path,
            )
            raise


# Historical public name retained as an identity alias.
XGBoostTrainingEngine = XGBoostRegressorTrainer


__all__ = [
    "XGBoostModelConfig",
    "XGBoostRegressorTrainer",
    "XGBoostTrainingEngine",
]
