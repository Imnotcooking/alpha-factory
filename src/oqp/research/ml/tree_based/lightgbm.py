"""LightGBM implementation of the shared supervised-regression contract."""

from __future__ import annotations

import os
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


@dataclass(frozen=True)
class LGBMModelConfig:
    target_col: str = "target_1d_rank"
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
    )
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50


class LightGBMRegressorTrainer(SupervisedModelBase):
    """LightGBM regressor with standardized validation and experiment output."""

    model_type = "lightgbm"
    artifact_format = "lightgbm_text"

    def __init__(
        self,
        data_path: str | Path,
        config: LGBMModelConfig | None = None,
        *,
        target_col: str | None = None,
        validation_config: ValidationConfig | None = None,
        model_name: str | None = None,
        factor_id: str | None = None,
        asset_class: str | None = None,
        exclude_features: list[str] | tuple[str, ...] | None = None,
        model_output_path: str | Path | None = None,
        importance_output_path: str | Path | None = None,
        predictions_output_path: str | Path | None = None,
        registry_db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
        artifact_store: ModelArtifactStore | None = None,
    ):
        self.config = config or LGBMModelConfig()
        resolved_target = target_col or self.config.target_col
        resolved_validation = validation_config or self.config.validation
        super().__init__(
            data_path=data_path,
            target_col=resolved_target,
            validation_config=resolved_validation,
            asset_class=asset_class,
            exclude_features=exclude_features,
        )
        self.model_name = model_name or "lightgbm_research_base"
        self.factor_id = factor_id
        self.model_output_path = Path(
            model_output_path
            or "runtime/artifacts/research/models/lightgbm_research_base.txt"
        )
        self.importance_output_path = Path(
            importance_output_path
            or "runtime/artifacts/research/feature_importance/lightgbm_research_base.csv"
        )
        self.predictions_output_path = Path(
            predictions_output_path
            or "runtime/artifacts/research/predictions/lightgbm_research_base.parquet"
        )
        self.registry_db_path = Path(registry_db_path)
        self.artifact_store = artifact_store

    def train(self) -> MLExperimentResult:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        import lightgbm as lgb

        fold_importances: list[pd.DataFrame] = []
        oos_predictions: list[pd.DataFrame] = []
        train_rows = 0

        for fold_number, (train_df, test_df, features) in enumerate(
            self.generate_validation_folds(),
            start=1,
        ):
            x_train = train_df[features].to_numpy(dtype="float64", copy=True)
            y_train = train_df[self.target_col].to_numpy(dtype="float64", copy=True)
            x_test = test_df[features].to_numpy(dtype="float64", copy=True)
            y_test = test_df[self.target_col].to_numpy(dtype="float64", copy=True)

            lgb_train = lgb.Dataset(x_train, y_train, feature_name=features)
            lgb_eval = lgb.Dataset(
                x_test,
                y_test,
                reference=lgb_train,
                feature_name=features,
            )
            model = lgb.train(
                self.config.params,
                lgb_train,
                num_boost_round=self.config.num_boost_round,
                valid_sets=[lgb_eval],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping_rounds,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.model = model
            train_rows += len(train_df)

            fold_importances.append(
                pd.DataFrame(
                    {
                        "feature": features,
                        "importance": model.feature_importance(
                            importance_type="gain"
                        ),
                        "fold": fold_number,
                    }
                )
            )
            oos_predictions.append(
                pd.DataFrame(
                    {
                        "date": test_df["date"].to_numpy(),
                        "ticker": test_df["ticker"].astype(str).to_numpy(),
                        "target": y_test,
                        "prediction": model.predict(x_test),
                        "fold": fold_number,
                    }
                )
            )

        if not oos_predictions or self.model is None:
            raise ValueError("[LightGBM] Validation produced no valid folds.")

        predictions = pd.concat(oos_predictions, ignore_index=True)
        importance = (
            pd.concat(fold_importances, ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        metrics = {
            "oos_mean_daily_spearman_ic": mean_daily_rank_ic(predictions),
            "train_rows_across_folds": int(train_rows),
            "oos_rows": int(len(predictions)),
            "fold_count": int(predictions["fold"].nunique()),
            "feature_count": int(len(self.feature_cols)),
        }
        hyperparams = {
            **self.config.params,
            "num_boost_round": self.config.num_boost_round,
            "early_stopping_rounds": self.config.early_stopping_rounds,
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
            metrics=metrics,
            hyperparams=hyperparams,
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
                source_module="oqp.research.ml.tree_based.lightgbm",
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


# Historical public name retained as an identity alias.  Persisted artifacts and
# caller-side ``isinstance`` checks therefore keep working while new code gets a
# name that states both the algorithm and the task.
LGBMModel = LightGBMRegressorTrainer


__all__ = [
    "LGBMModel",
    "LGBMModelConfig",
    "LightGBMRegressorTrainer",
]
