from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional research dependency
    XGBClassifier = None

from oqp.research.tick_pulse.xgboost_model import (
    TickXGBoostConfig,
    TickXGBoostResearchEngine,
)
from oqp.optimization import (
    FrozenResearchInputs,
    ObjectiveSpec,
    OptimizationComparisonResult,
    OptimizationStudyRunner,
    OptimizationStudySpec,
    OptimizationStudyStore,
    SearchBudget,
    TrialEvaluation,
    build_component_parameter_schema,
    require_dataset_fingerprint,
    stable_optimization_hash,
)


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
        *,
        sampler_id: str = "tpe",
        compare_random_baseline: bool = True,
        store: OptimizationStudyStore | None = None,
        study_id: str | None = None,
    ):
        self.base_config = base_config
        self.n_trials = int(n_trials)
        self.probability_threshold = float(probability_threshold)
        self.sampler_id = str(sampler_id).strip().lower()
        self.compare_random_baseline = bool(compare_random_baseline)
        self.store = store or OptimizationStudyStore()
        self.study_id = study_id
        self.feature_cols: list[str] = []

    def run(self, features: pd.DataFrame) -> dict:
        if XGBClassifier is None:
            raise ImportError("xgboost is required for TickXGBoostBayesianOptimizer.")
        dataset_fingerprint = require_dataset_fingerprint(features)
        prep_engine = TickXGBoostResearchEngine(self.base_config)
        dataset = prep_engine._prepare_dataset(features)
        self.feature_cols = prep_engine.feature_cols
        train_df, valid_df, split_info = self._calibration_split(dataset)

        if train_df["target"].nunique() < 2:
            raise ValueError("Bayesian calibration train target has only one class.")
        if valid_df["target"].nunique() < 2:
            raise ValueError("Bayesian calibration validation target has only one class.")

        schema = self._parameter_schema()
        identity_payload = {
            "component": "tick_xgboost_model",
            "dataset_fingerprint": dataset_fingerprint,
            "base_config": asdict(self.base_config),
            "probability_threshold": self.probability_threshold,
            "n_trials": self.n_trials,
            "sampler_id": self.sampler_id,
            "split": split_info,
            "schema": schema.to_dict(),
        }
        resolved_study_id = self.study_id or (
            "tick_xgboost_" + stable_optimization_hash(identity_payload)[:16]
        )
        spec = OptimizationStudySpec(
            study_id=resolved_study_id,
            purpose="model_hyperparameter",
            component_id=schema.component_id,
            sampler_id=self.sampler_id,
            objectives=(
                ObjectiveSpec("validation_roc_auc", "roc_auc", "maximize"),
            ),
            frozen_inputs=FrozenResearchInputs(
                dataset_fingerprint=dataset_fingerprint,
                holdout_fingerprint=stable_optimization_hash(split_info),
            ),
            budget=SearchBudget(max_trials=self.n_trials, n_jobs=1),
            seed=self.base_config.random_state,
            metadata={
                "probability_threshold": self.probability_threshold,
                "hypothesis": self.base_config.hypothesis,
            },
        )
        evaluator = lambda parameters, context: self._evaluate_candidate(
            parameters, context, train_df, valid_df
        )
        runner = OptimizationStudyRunner(self.store)
        if self.compare_random_baseline and self.sampler_id != "random":
            comparison = runner.run_with_random_baseline(spec, schema, evaluator)
        else:
            result = runner.run(spec, schema, evaluator)
            comparison = OptimizationComparisonResult(result, result)
        if not comparison.challenger.candidates:
            raise ValueError("XGBoost optimization produced no completed candidate")
        best_candidate = comparison.challenger.candidates[0]
        best_params = self._normalized_params(dict(best_candidate.parameters))
        final_result = self._fit_final_model(prep_engine, dataset, best_params)
        final_result["calibration"] = {
            "optimizer": f"shared_{self.sampler_id}",
            "objective": "validation_roc_auc",
            "n_trials": self.n_trials,
            "best_value": float(best_candidate.objective_values[0]),
            "best_params": best_params,
            "probability_threshold": self.probability_threshold,
            "split": split_info,
            "best_trial_metrics": dict(best_candidate.metrics),
            "optimization": comparison.challenger.to_dict(),
            "random_baseline": comparison.baseline.to_dict(),
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
            n_jobs=1,
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

        split_info = {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "final_test_start_row": int(test_start),
            "embargo_gap_rows": int(embargo),
        }
        if "datetime" in dataset.columns:
            split_info.update(
                {
                    "train_start": str(pd.to_datetime(train_df["datetime"]).min()),
                    "train_end": str(pd.to_datetime(train_df["datetime"]).max()),
                    "validation_start": str(
                        pd.to_datetime(valid_df["datetime"]).min()
                    ),
                    "validation_end": str(
                        pd.to_datetime(valid_df["datetime"]).max()
                    ),
                    "final_holdout_start": str(
                        pd.to_datetime(dataset.iloc[test_start:]["datetime"]).min()
                    ),
                    "final_holdout_end": str(
                        pd.to_datetime(dataset.iloc[test_start:]["datetime"]).max()
                    ),
                }
            )
        return train_df, valid_df, split_info

    def _evaluate_candidate(
        self,
        params: dict,
        context,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> TrialEvaluation:
        context.require_development_only()
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
            n_jobs=1,
            scale_pos_weight=neg_count / max(pos_count, 1),
        )
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_valid)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_valid, pred_proba))
        accuracy = float(accuracy_score(y_valid, pred_label))
        signal_mask = pred_proba >= self.probability_threshold
        signal_accuracy = float(y_valid[signal_mask].mean()) if signal_mask.any() else np.nan

        return TrialEvaluation(
            metrics={
                "roc_auc": auc,
                "accuracy_50": accuracy,
                "base_target_rate": float(y_valid.mean()),
                "signal_count": int(signal_mask.sum()),
                "signal_accuracy": (
                    signal_accuracy if np.isfinite(signal_accuracy) else 0.0
                ),
            },
            fold_metrics=(
                {
                    "fold": "fixed_validation",
                    "rows": int(len(valid_df)),
                    "roc_auc": auc,
                    "accuracy_50": accuracy,
                },
            ),
        )

    def _parameter_schema(self):
        defaults = self.base_config.hyperparams()
        return build_component_parameter_schema(
            "tick_xgboost_classifier",
            "model",
            {
                "n_estimators": {
                    "default": defaults["n_estimators"],
                    "type": "int",
                    "low": 80,
                    "high": 320,
                    "tunable": True,
                },
                "max_depth": {
                    "default": defaults["max_depth"],
                    "type": "int",
                    "low": 2,
                    "high": 6,
                    "tunable": True,
                },
                "learning_rate": {
                    "default": defaults["learning_rate"],
                    "type": "float",
                    "low": 0.01,
                    "high": 0.15,
                    "log": True,
                    "tunable": True,
                },
                "subsample": {
                    "default": defaults["subsample"],
                    "type": "float",
                    "low": 0.65,
                    "high": 1.0,
                    "tunable": True,
                },
                "colsample_bytree": {
                    "default": defaults["colsample_bytree"],
                    "type": "float",
                    "low": 0.65,
                    "high": 1.0,
                    "tunable": True,
                },
                "reg_alpha": {
                    "default": defaults["reg_alpha"],
                    "type": "float",
                    "low": 1e-4,
                    "high": 2.0,
                    "log": True,
                    "tunable": True,
                },
                "reg_lambda": {
                    "default": defaults["reg_lambda"],
                    "type": "float",
                    "low": 0.5,
                    "high": 8.0,
                    "log": True,
                    "tunable": True,
                },
                "min_child_weight": {
                    "default": defaults["min_child_weight"],
                    "type": "float",
                    "low": 1.0,
                    "high": 12.0,
                    "log": True,
                    "tunable": True,
                },
                "gamma": {
                    "default": defaults["gamma"],
                    "type": "float",
                    "low": 0.0,
                    "high": 3.0,
                    "tunable": True,
                },
            },
        )

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
