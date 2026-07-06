from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import optuna
except Exception:  # pragma: no cover - optional research dependency
    optuna = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional research dependency
    XGBClassifier = None

from oqp.research.tick_pulse.xgboost_model import (
    TickXGBoostConfig,
    TickXGBoostResearchEngine,
)

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


class TickXGBoostBayesianOptimizer:
    """
    Bayesian hyperparameter search for the tick XGBoost research model.

    The optimizer uses an embargoed train/validation split for tuning and leaves
    the final chronological test window untouched. After Optuna finds the best
    validation parameters, the final model is retrained/evaluated by
    TickXGBoostResearchEngine on its normal purged train/test split.
    """

    def __init__(
        self,
        base_config: TickXGBoostConfig,
        n_trials: int = 20,
        probability_threshold: float = 0.55,
    ):
        self.base_config = base_config
        self.n_trials = int(n_trials)
        self.probability_threshold = float(probability_threshold)
        self.feature_cols: list[str] = []

    def run(self, features: pd.DataFrame) -> dict:
        if optuna is None:
            raise ImportError("optuna is required for TickXGBoostBayesianOptimizer.")
        if XGBClassifier is None:
            raise ImportError("xgboost is required for TickXGBoostBayesianOptimizer.")
        prep_engine = TickXGBoostResearchEngine(self.base_config)
        dataset = prep_engine._prepare_dataset(features)
        self.feature_cols = prep_engine.feature_cols
        train_df, valid_df, split_info = self._calibration_split(dataset)

        if train_df["target"].nunique() < 2:
            raise ValueError("Bayesian calibration train target has only one class.")
        if valid_df["target"].nunique() < 2:
            raise ValueError("Bayesian calibration validation target has only one class.")

        sampler = optuna.samplers.TPESampler(seed=self.base_config.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: self._objective(trial, train_df, valid_df), n_trials=self.n_trials)

        best_params = self._normalized_params(study.best_params)
        final_result = self._fit_final_model(prep_engine, dataset, best_params)
        final_result["calibration"] = {
            "optimizer": "optuna_tpe",
            "objective": "validation_roc_auc",
            "n_trials": self.n_trials,
            "best_value": float(study.best_value),
            "best_params": best_params,
            "probability_threshold": self.probability_threshold,
            "split": split_info,
            "best_trial_metrics": dict(study.best_trial.user_attrs),
        }
        return final_result

    def _fit_final_model(
        self,
        prep_engine: TickXGBoostResearchEngine,
        dataset: pd.DataFrame,
        best_params: dict,
    ) -> dict:
        train_df, test_df = prep_engine._chronological_split(dataset)
        X_train = train_df[self.feature_cols]
        y_train = train_df["target"]
        X_test = test_df[self.feature_cols]
        y_test = test_df["target"]

        if y_train.nunique() < 2:
            raise ValueError("Final calibrated train target has only one class.")
        if y_test.nunique() < 2:
            raise ValueError("Final calibrated test target has only one class.")

        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        model = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.base_config.random_state,
            n_jobs=-1,
            scale_pos_weight=neg_count / max(pos_count, 1),
        )
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int)
        predictions = test_df[
            [
                "symbol",
                "datetime",
                "last_price",
                "mid_price",
                "future_datetime",
                "future_mid_price",
                "future_move_ticks",
                "target",
                *self.feature_cols,
            ]
        ].copy()
        predictions["ml_probability"] = pred_proba
        predictions["ml_label_50"] = pred_label

        return {
            "metrics": {
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "feature_count": int(len(self.feature_cols)),
                "train_target_rate": float(y_train.mean()),
                "test_target_rate": float(y_test.mean()),
                "roc_auc": float(roc_auc_score(y_test, pred_proba)),
                "accuracy_50": float(accuracy_score(y_test, pred_label)),
            },
            "importance": self._feature_importance(model),
            "thresholds": self._split_thresholds(model),
            "predictions": predictions,
            "feature_cols": list(self.feature_cols),
            "hyperparams": dict(best_params),
        }

    def _calibration_split(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        n_rows = len(dataset)
        test_start = int(n_rows * (1.0 - self.base_config.test_fraction))
        embargo = int(self.base_config.horizon_ticks)
        train_end = int(n_rows * 0.50)
        valid_start = train_end + embargo
        valid_end = test_start - embargo

        train_df = dataset.iloc[:train_end].copy()
        valid_df = dataset.iloc[valid_start:valid_end].copy()
        if len(train_df) < 1_000 or len(valid_df) < 500:
            raise ValueError(
                "Bayesian calibration split too small: "
                f"train={len(train_df):,}, valid={len(valid_df):,}."
            )

        return train_df, valid_df, {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "final_test_start_row": int(test_start),
            "embargo_gap_rows": int(embargo),
        }

    def _objective(self, trial: optuna.Trial, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        }

        X_train = train_df[self.feature_cols]
        y_train = train_df["target"]
        X_valid = valid_df[self.feature_cols]
        y_valid = valid_df["target"]
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.base_config.random_state,
            n_jobs=-1,
            scale_pos_weight=neg_count / max(pos_count, 1),
        )
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_valid)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_valid, pred_proba))
        accuracy = float(accuracy_score(y_valid, pred_label))
        signal_mask = pred_proba >= self.probability_threshold
        signal_accuracy = float(y_valid[signal_mask].mean()) if signal_mask.any() else np.nan

        trial.set_user_attr("roc_auc", auc)
        trial.set_user_attr("accuracy_50", accuracy)
        trial.set_user_attr("base_target_rate", float(y_valid.mean()))
        trial.set_user_attr("signal_count", int(signal_mask.sum()))
        trial.set_user_attr("signal_accuracy", signal_accuracy)
        return auc

    @staticmethod
    def _normalized_params(params: dict) -> dict:
        out = dict(params)
        out["n_estimators"] = int(out["n_estimators"])
        out["max_depth"] = int(out["max_depth"])
        return out

    def _feature_importance(self, model: XGBClassifier) -> pd.DataFrame:
        gain_scores = model.get_booster().get_score(importance_type="gain")
        weight_scores = model.get_booster().get_score(importance_type="weight")
        rows = []
        for idx, feature in enumerate(self.feature_cols):
            rows.append(
                {
                    "feature": feature,
                    "importance": float(model.feature_importances_[idx]),
                    "gain": float(gain_scores.get(feature, 0.0)),
                    "split_count": int(weight_scores.get(feature, 0)),
                }
            )
        return pd.DataFrame(rows).sort_values(["importance", "gain"], ascending=False).reset_index(drop=True)

    def _split_thresholds(self, model: XGBClassifier) -> pd.DataFrame:
        tree_df = model.get_booster().trees_to_dataframe()
        split_df = tree_df[tree_df["Feature"].isin(self.feature_cols)].copy()
        if split_df.empty:
            return pd.DataFrame(
                columns=[
                    "feature",
                    "split_count",
                    "split_median",
                    "split_q25",
                    "split_q75",
                    "split_min",
                    "split_max",
                    "total_gain",
                ]
            )

        return (
            split_df.groupby("Feature")
            .agg(
                split_count=("Split", "size"),
                split_median=("Split", "median"),
                split_q25=("Split", lambda s: float(s.quantile(0.25))),
                split_q75=("Split", lambda s: float(s.quantile(0.75))),
                split_min=("Split", "min"),
                split_max=("Split", "max"),
                total_gain=("Gain", "sum"),
            )
            .reset_index()
            .rename(columns={"Feature": "feature"})
            .sort_values(["total_gain", "split_count"], ascending=False)
            .reset_index(drop=True)
        )
