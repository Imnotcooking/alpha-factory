from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional research dependency
    XGBClassifier = None


TICK_ML_FEATURES = [
    "flow_imbalance",
    "book_imbalance",
    "volume_intensity",
    "rolling_mid_move_ticks",
    "price_shock",
    "spread",
    "volume_delta",
    "oi_delta",
    "mid_move_ticks",
    "last_price_delta",
    "mid_price_delta",
    "rolling_total_volume",
    "rolling_signed_volume",
    "rtv_fast_move_ticks",
    "rtv_abs_move_ticks",
    "rtv_threshold_ticks",
    "rtv_threshold_ratio",
]


@dataclass(frozen=True)
class TickXGBoostConfig:
    horizon_ticks: int
    min_success_ticks: float
    hypothesis: str
    max_rows: int = 200_000
    test_fraction: float = 0.30
    random_state: int = 42
    n_estimators: int = 180
    max_depth: int = 3
    learning_rate: float = 0.04
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.5
    reg_lambda: float = 2.0
    min_child_weight: float = 1.0
    gamma: float = 0.0

    def hyperparams(self) -> dict:
        return {
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "reg_alpha": float(self.reg_alpha),
            "reg_lambda": float(self.reg_lambda),
            "min_child_weight": float(self.min_child_weight),
            "gamma": float(self.gamma),
        }


class TickXGBoostResearchEngine:
    """
    Tick-level XGBoost research helper.

    This is intentionally separate from the daily alpha XGBoost engine. It trains
    on one selected tick feature frame and answers a local question: which current
    microstructure features help predict the selected forward tick outcome?
    """

    def __init__(self, config: TickXGBoostConfig):
        self.config = config
        self.feature_cols: list[str] = []
        self.model: XGBClassifier | None = None

    def run(self, features: pd.DataFrame) -> dict:
        if XGBClassifier is None:
            raise ImportError("xgboost is required for TickXGBoostResearchEngine.")
        dataset = self._prepare_dataset(features)
        train_df, test_df = self._chronological_split(dataset)

        X_train = train_df[self.feature_cols]
        y_train = train_df["target"]
        X_test = test_df[self.feature_cols]
        y_test = test_df["target"]

        if y_train.nunique() < 2:
            raise ValueError("XGBoost training target has only one class in the train window.")
        if y_test.nunique() < 2:
            raise ValueError("XGBoost validation target has only one class in the test window.")

        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = neg_count / max(pos_count, 1)

        self.model = XGBClassifier(
            **self.config.hyperparams(),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        self.model.fit(X_train, y_train)

        pred_proba = self.model.predict_proba(X_test)[:, 1]
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

        metrics = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(self.feature_cols)),
            "train_target_rate": float(y_train.mean()),
            "test_target_rate": float(y_test.mean()),
            "roc_auc": float(roc_auc_score(y_test, pred_proba)),
            "accuracy_50": float(accuracy_score(y_test, pred_label)),
        }

        return {
            "metrics": metrics,
            "importance": self._feature_importance(),
            "thresholds": self._split_thresholds(),
            "predictions": predictions,
            "feature_cols": self.feature_cols,
        }

    def _prepare_dataset(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.sort_values(["symbol", "datetime"]).copy()
        group_keys = ["symbol", "_session_id"] if "_session_id" in out.columns else ["symbol"]
        grouped = out.groupby(group_keys, sort=False)
        out["future_datetime"] = grouped["datetime"].shift(-self.config.horizon_ticks)
        out["future_mid_price"] = grouped["mid_price"].shift(-self.config.horizon_ticks)
        tick_size = out["tick_size_est"].replace(0, np.nan)
        out["future_move_ticks"] = (out["future_mid_price"] - out["mid_price"]) / tick_size

        if self.config.hypothesis in {"relative_velocity", "relative_velocity_fade"}:
            if "rtv_fast_move_ticks" not in out.columns:
                raise ValueError(f"{self.config.hypothesis} ML target requires rtv_fast_move_ticks.")
            if self.config.hypothesis == "relative_velocity_fade":
                out["target"] = (
                    ((out["rtv_fast_move_ticks"] > 0) & (out["future_move_ticks"] <= -self.config.min_success_ticks))
                    | ((out["rtv_fast_move_ticks"] < 0) & (out["future_move_ticks"] >= self.config.min_success_ticks))
                ).astype(int)
            else:
                out["target"] = (
                    ((out["rtv_fast_move_ticks"] > 0) & (out["future_move_ticks"] >= self.config.min_success_ticks))
                    | ((out["rtv_fast_move_ticks"] < 0) & (out["future_move_ticks"] <= -self.config.min_success_ticks))
                ).astype(int)
        elif self.config.hypothesis in {"bearish", "bearish_breakdown"}:
            out["target"] = (out["future_move_ticks"] <= -self.config.min_success_ticks).astype(int)
        else:
            out["target"] = (out["future_move_ticks"] >= self.config.min_success_ticks).astype(int)

        available_features = [col for col in TICK_ML_FEATURES if col in out.columns]
        if not available_features:
            raise ValueError("No tick ML features were available in the feature frame.")

        self.feature_cols = available_features
        required_cols = [
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
        out = out[required_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        out = out.sort_values(["datetime", "symbol"]).reset_index(drop=True)

        if len(out) > self.config.max_rows:
            keep_idx = np.linspace(0, len(out) - 1, self.config.max_rows, dtype=int)
            out = out.iloc[keep_idx].reset_index(drop=True)

        if len(out) < 2_000:
            raise ValueError(f"Not enough valid tick rows for ML training: {len(out):,}.")
        return out

    def _chronological_split(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(dataset) * (1.0 - self.config.test_fraction))
        train_end = max(0, split_idx - self.config.horizon_ticks)
        train_df = dataset.iloc[:train_end].copy()
        test_df = dataset.iloc[split_idx:].copy()

        if len(train_df) < 1_000 or len(test_df) < 500:
            raise ValueError(
                f"Train/test split too small: train={len(train_df):,}, test={len(test_df):,}."
            )
        return train_df, test_df

    def _feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not fitted.")

        gain_scores = self.model.get_booster().get_score(importance_type="gain")
        weight_scores = self.model.get_booster().get_score(importance_type="weight")
        rows = []
        for idx, feature in enumerate(self.feature_cols):
            rows.append(
                {
                    "feature": feature,
                    "importance": float(self.model.feature_importances_[idx]),
                    "gain": float(gain_scores.get(feature, 0.0)),
                    "split_count": int(weight_scores.get(feature, 0)),
                }
            )
        return pd.DataFrame(rows).sort_values(["importance", "gain"], ascending=False).reset_index(drop=True)

    def _split_thresholds(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not fitted.")

        tree_df = self.model.get_booster().trees_to_dataframe()
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

        threshold_df = (
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
        return threshold_df


def score_probability_threshold(predictions: pd.DataFrame, probability_threshold: float) -> dict:
    scored = predictions.copy()
    scored["ml_signal"] = scored["ml_probability"] >= probability_threshold
    fired = scored[scored["ml_signal"]]
    return {
        "rows": int(len(scored)),
        "signal_count": int(len(fired)),
        "signal_rate": float(len(fired) / len(scored)) if len(scored) else np.nan,
        "signal_accuracy": float(fired["target"].mean()) if len(fired) else np.nan,
        "target_rate": float(scored["target"].mean()) if len(scored) else np.nan,
        "scored": scored,
    }
