from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional research dependency
    XGBClassifier = None

from oqp.research.artifacts import ModelArtifactStore
from oqp.research.model_registry import (
    build_data_fingerprint,
    record_from_artifact,
    register_model_artifact,
)


TICK_GATEKEEPER_FEATURES = [
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
    "trend_1m_ticks",
    "trend_5m_ticks",
    "trend_15m_ticks",
    "vol_5m_ticks",
    "range_5m_ticks",
    "event_hour",
    "session_progress_ticks",
    "rtv_fast_move_ticks",
    "rtv_abs_move_ticks",
    "rtv_threshold_ticks",
    "rtv_threshold_ratio",
]


@dataclass(frozen=True)
class TickGatekeeperConfig:
    probability_threshold: float = 0.55
    test_fraction: float = 0.30
    random_state: int = 42
    n_estimators: int = 90
    max_depth: int = 3
    learning_rate: float = 0.035
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.5
    reg_lambda: float = 2.0
    min_child_weight: float = 3.0
    gamma: float = 0.0
    max_events: int = 5_000
    n_jobs: int = 4

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


class TickGatekeeperResearchEngine:
    """
    Event-conditional XGBoost helper.

    The normal tick model asks "which ticks predict the target?" This gatekeeper
    asks the narrower question "among already-detected hypothesis events, which
    ones should we actually trade?"
    """

    def __init__(self, config: TickGatekeeperConfig | None = None):
        self.config = config or TickGatekeeperConfig()
        self.feature_cols: list[str] = []
        self.model: XGBClassifier | None = None

    def run(self, features: pd.DataFrame, candidates: pd.DataFrame) -> dict:
        if XGBClassifier is None:
            raise ImportError("xgboost is required for TickGatekeeperResearchEngine.")
        dataset = self._prepare_event_dataset(features, candidates)
        train_df, test_df, split_info = self._chronological_split(dataset)

        X_train = train_df[self.feature_cols]
        y_train = train_df["target"]
        X_test = test_df[self.feature_cols]
        y_test = test_df["target"]

        if y_train.nunique() < 2:
            raise ValueError("Gatekeeper train target has only one class.")
        if y_test.nunique() < 2:
            raise ValueError("Gatekeeper test target has only one class.")

        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        self.model = XGBClassifier(
            **self.config.hyperparams(),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            scale_pos_weight=neg_count / max(pos_count, 1),
        )
        self.model.fit(X_train, y_train)

        pred_proba = self.model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int)
        gate_mask = pred_proba >= self.config.probability_threshold
        top20_threshold = float(np.quantile(pred_proba, 0.80))
        top20_mask = pred_proba >= top20_threshold

        scored_events = test_df[
            [
                "symbol",
                "datetime",
                "future_datetime",
                "last_price",
                "future_mid_price",
                "future_move_ticks",
                "expected_direction",
                "outcome",
                "target",
                *self.feature_cols,
            ]
        ].copy()
        scored_events["ml_probability"] = pred_proba
        scored_events["ml_label_50"] = pred_label
        scored_events["ml_gate"] = gate_mask

        metrics = {
            "events_total": int(len(dataset)),
            **split_info,
            "feature_count": int(len(self.feature_cols)),
            "train_base_rate": float(y_train.mean()),
            "test_base_rate": float(y_test.mean()),
            "roc_auc": float(roc_auc_score(y_test, pred_proba)),
            "accuracy_50": float(accuracy_score(y_test, pred_label)),
            "probability_threshold": float(self.config.probability_threshold),
            "gate_count": int(gate_mask.sum()),
            "gate_rate": float(gate_mask.mean()),
            "gate_accuracy": float(y_test[gate_mask].mean()) if gate_mask.any() else np.nan,
            "gate_avg_future_ticks": float(scored_events.loc[gate_mask, "future_move_ticks"].mean()) if gate_mask.any() else np.nan,
            "top20_threshold": top20_threshold,
            "top20_count": int(top20_mask.sum()),
            "top20_accuracy": float(y_test[top20_mask].mean()) if top20_mask.any() else np.nan,
            "test_avg_future_ticks": float(scored_events["future_move_ticks"].mean()),
        }

        return {
            "metrics": metrics,
            "importance": self._feature_importance(),
            "thresholds": self._split_thresholds(),
            "scored_events": scored_events,
            "feature_cols": list(self.feature_cols),
            "hyperparams": self.config.hyperparams(),
        }

    def _prepare_event_dataset(self, features: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            raise ValueError("No detected events available for gatekeeper training.")

        events = candidates.copy()
        events["target"] = events["is_correct"].astype(int)
        events = _add_event_context_features(features, events)

        available_features = [col for col in TICK_GATEKEEPER_FEATURES if col in events.columns]
        if not available_features:
            raise ValueError("No gatekeeper features were available.")

        self.feature_cols = available_features
        required_cols = [
            "symbol",
            "datetime",
            "future_datetime",
            "last_price",
            "future_mid_price",
            "future_move_ticks",
            "expected_direction",
            "outcome",
            "target",
            *self.feature_cols,
        ]
        dataset = events[required_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        dataset = dataset.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        if len(dataset) > self.config.max_events:
            keep_idx = np.linspace(0, len(dataset) - 1, self.config.max_events, dtype=int)
            dataset = dataset.iloc[keep_idx].reset_index(drop=True)
        if len(dataset) < 120:
            raise ValueError(f"Not enough detected events for gatekeeper training: {len(dataset):,}.")
        if dataset["target"].nunique() < 2:
            raise ValueError("Detected events contain only one realized outcome class.")
        return dataset

    def _chronological_split(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        split_idx = int(len(dataset) * (1.0 - self.config.test_fraction))
        embargo_events = max(5, min(50, int(round(len(dataset) * 0.02))))
        train_end = max(0, split_idx - embargo_events)
        train_df = dataset.iloc[:train_end].copy()
        test_df = dataset.iloc[split_idx:].copy()

        if len(train_df) < 60 or len(test_df) < 30:
            raise ValueError(
                f"Gatekeeper split too small: train={len(train_df):,}, test={len(test_df):,}."
            )
        return train_df, test_df, {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split_row": int(split_idx),
            "embargo_events": int(embargo_events),
        }

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

    def save_artifact(self, path: str, metadata: dict | None = None) -> None:
        if self.model is None:
            raise ValueError("Cannot save an unfitted gatekeeper model.")

        metadata = metadata or {}
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": list(self.feature_cols),
                "config": self.config.__dict__.copy(),
                "metadata": metadata,
            },
            path,
        )
        model_name = str(metadata.get("model_name") or os.path.splitext(os.path.basename(path))[0])
        stored = ModelArtifactStore().archive_file(path, model_name=model_name)
        data_fingerprint = build_data_fingerprint(
            metadata.get("data_path") or metadata.get("tick_file"),
            include_hash=bool(metadata.get("hash_data_file", False)),
        )
        record = record_from_artifact(
            artifact_id=stored.artifact_id,
            model_name=model_name,
            factor_id=metadata.get("factor_id"),
            model_type="xgboost_tick_gatekeeper",
            artifact_path=stored.path,
            artifact_format="joblib_pkl",
            legacy_path=path,
            source_module="oqp.research.tick_pulse.gatekeeper_model",
            data_fingerprint=data_fingerprint,
            feature_cols=list(self.feature_cols),
            target_col="is_correct",
            split_policy={
                "mode": "chronological_event_split",
                "test_fraction": self.config.test_fraction,
                "embargo": "event_count_adaptive",
            },
            metrics=metadata.get("metrics", {}),
            hyperparams=self.config.hyperparams(),
            metadata=metadata,
        )
        register_model_artifact(record)


def _add_event_context_features(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    context_pos_col = "_context_row_pos"
    for column in [
        "trend_1m_ticks",
        "trend_5m_ticks",
        "trend_15m_ticks",
        "vol_5m_ticks",
        "range_5m_ticks",
        "event_hour",
        "session_progress_ticks",
    ]:
        out[column] = np.nan

    feature_frame = features.sort_values(["symbol", "datetime"]).copy()
    feature_frame[context_pos_col] = feature_frame.groupby("symbol", sort=False).cumcount()
    if "_row_pos" in out.columns:
        out[context_pos_col] = pd.to_numeric(out["_row_pos"], errors="coerce")
    else:
        lookup = feature_frame[["symbol", "datetime", context_pos_col]].drop_duplicates(
            ["symbol", "datetime"],
            keep="first",
        )
        out = out.merge(
            lookup,
            on=["symbol", "datetime"],
            how="left",
        )

    for symbol, group in feature_frame.groupby("symbol", sort=False):
        if context_pos_col not in out.columns:
            continue

        event_mask = out["symbol"].eq(symbol) & out[context_pos_col].notna()
        if not event_mask.any():
            continue

        g = group.reset_index(drop=True)
        times = g["datetime"].astype("int64").to_numpy()
        mids = g["mid_price"].to_numpy(dtype=float)
        ticks = g["tick_size_est"].replace(0, np.nan).to_numpy(dtype=float)
        sessions = g["_session_id"].to_numpy() if "_session_id" in g.columns else np.zeros(len(g), dtype=int)
        session_starts = {
            session_id: int(np.flatnonzero(sessions == session_id)[0])
            for session_id in np.unique(sessions)
        }

        event_indices = out.index[event_mask].to_numpy()
        event_positions = out.loc[event_indices, context_pos_col].astype(int).to_numpy()
        for event_idx, pos in zip(event_indices, event_positions):
            if pos < 0 or pos >= len(g):
                continue

            tick_size = ticks[pos]
            if not np.isfinite(tick_size) or tick_size <= 0:
                continue

            event_time = times[pos]
            session_id = sessions[pos]
            session_start = session_starts[session_id]

            for minutes, column in [
                (1, "trend_1m_ticks"),
                (5, "trend_5m_ticks"),
                (15, "trend_15m_ticks"),
            ]:
                prior_time = event_time - int(minutes * 60 * 1_000_000_000)
                prior_pos = np.searchsorted(times, prior_time, side="left")
                prior_pos = min(max(prior_pos, session_start), pos)
                out.at[event_idx, column] = (mids[pos] - mids[prior_pos]) / tick_size

            start_5m = np.searchsorted(times, event_time - int(5 * 60 * 1_000_000_000), side="left")
            start_5m = min(max(start_5m, session_start), pos)
            window = mids[start_5m : pos + 1]
            if len(window) >= 3:
                out.at[event_idx, "vol_5m_ticks"] = np.nanstd(np.diff(window) / tick_size)
                out.at[event_idx, "range_5m_ticks"] = (np.nanmax(window) - np.nanmin(window)) / tick_size

            timestamp = out.at[event_idx, "datetime"]
            out.at[event_idx, "event_hour"] = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
            out.at[event_idx, "session_progress_ticks"] = pos - session_start

    return out.drop(columns=[context_pos_col], errors="ignore")
