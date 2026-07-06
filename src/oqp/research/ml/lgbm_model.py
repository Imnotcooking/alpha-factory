"""LightGBM walk-forward research model."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from oqp.research.ml.supervised import SupervisedModelBase, WalkForwardConfig


@dataclass(frozen=True)
class LGBMModelConfig:
    target_col: str = "target_1d_rank"
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    params: dict = field(
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


class LGBMModel(SupervisedModelBase):
    """Expanding-window LightGBM regressor for alpha-rank targets."""

    def __init__(self, data_path: str, config: LGBMModelConfig | None = None):
        self.config = config or LGBMModelConfig()
        super().__init__(
            data_path=data_path,
            target_col=self.config.target_col,
            walk_forward_config=self.config.walk_forward,
        )
        self.fold_importances: list[pd.DataFrame] = []
        self.oos_predictions: list[pd.DataFrame] = []

    def train(self) -> tuple:
        import lightgbm as lgb

        fold_count = 1
        for train_df, test_df, features in self.generate_walk_forward_folds():
            x_train, y_train = train_df[features], train_df[self.target_col]
            x_test, y_test = test_df[features], test_df[self.target_col]

            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

            model = lgb.train(
                self.config.params,
                lgb_train,
                num_boost_round=self.config.num_boost_round,
                valid_sets=[lgb_train, lgb_eval],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping_rounds,
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.model = model

            importance = model.feature_importance(importance_type="gain")
            fold_df = pd.DataFrame(
                {"feature": features, f"importance_fold_{fold_count}": importance}
            )
            self.fold_importances.append(fold_df)

            test_df_copy = test_df.copy()
            test_df_copy["predicted_rank"] = model.predict(x_test)
            self.oos_predictions.append(test_df_copy)
            fold_count += 1

        if not self.oos_predictions:
            raise ValueError("[LGBM] Walk-forward produced no valid folds.")

        master_oos_df = pd.concat(self.oos_predictions, ignore_index=True)
        master_oos_df.attrs["oos_mean_ic"] = _mean_daily_ic(
            master_oos_df,
            pred_col="predicted_rank",
            target_col=self.target_col,
        )

        master_importance_df = self.fold_importances[0]
        for frame in self.fold_importances[1:]:
            master_importance_df = pd.merge(master_importance_df, frame, on="feature")

        importance_cols = [
            col
            for col in master_importance_df.columns
            if col.startswith("importance_fold")
        ]
        master_importance_df["average_importance"] = master_importance_df[
            importance_cols
        ].mean(axis=1)
        master_importance_df = (
            master_importance_df[["feature", "average_importance"]]
            .sort_values(by="average_importance", ascending=False)
            .reset_index(drop=True)
        )

        return self.model, master_oos_df, master_importance_df


def _mean_daily_ic(df: pd.DataFrame, *, pred_col: str, target_col: str) -> float:
    daily_ic = []
    for _, day in df.groupby("date", sort=False):
        value = day[pred_col].corr(day[target_col], method="spearman")
        if pd.notna(value):
            daily_ic.append(float(value))
    return float(pd.Series(daily_ic).mean()) if daily_ic else float("nan")


__all__ = ["LGBMModel", "LGBMModelConfig"]
