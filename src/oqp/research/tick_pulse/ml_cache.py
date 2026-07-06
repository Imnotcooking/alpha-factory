from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from oqp.research.trials import record_research_trial


TICK_ML_MODEL_VERSION = "tick_xgboost_v3_rtv_features"


def ensure_tick_ml_tables(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tick_ml_studies (
                study_key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_mtime REAL NOT NULL,
                file_size INTEGER NOT NULL,
                product TEXT NOT NULL,
                symbol TEXT NOT NULL,
                window INTEGER NOT NULL,
                horizon_ticks INTEGER NOT NULL,
                hypothesis TEXT NOT NULL,
                min_success_ticks REAL NOT NULL,
                max_rows INTEGER NOT NULL,
                test_fraction REAL NOT NULL,
                hyperparams_json TEXT NOT NULL DEFAULT '{}',
                calibration_json TEXT,
                metrics_json TEXT NOT NULL,
                feature_cols_json TEXT NOT NULL,
                prediction_rows INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tick_ml_studies_lookup
            ON tick_ml_studies (
                file_path,
                symbol,
                hypothesis,
                window,
                horizon_ticks,
                min_success_ticks,
                model_version
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tick_ml_feature_importance (
                study_key TEXT NOT NULL,
                feature TEXT NOT NULL,
                importance REAL,
                gain REAL,
                split_count INTEGER,
                PRIMARY KEY (study_key, feature),
                FOREIGN KEY (study_key) REFERENCES tick_ml_studies (study_key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tick_ml_thresholds (
                study_key TEXT NOT NULL,
                feature TEXT NOT NULL,
                split_count INTEGER,
                split_median REAL,
                split_q25 REAL,
                split_q75 REAL,
                split_min REAL,
                split_max REAL,
                total_gain REAL,
                PRIMARY KEY (study_key, feature),
                FOREIGN KEY (study_key) REFERENCES tick_ml_studies (study_key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tick_ml_artifacts (
                study_key TEXT PRIMARY KEY,
                predictions_parquet BLOB NOT NULL,
                FOREIGN KEY (study_key) REFERENCES tick_ml_studies (study_key)
            )
            """
        )
        _ensure_columns(
            conn,
            "tick_ml_studies",
            {
                "hyperparams_json": "TEXT NOT NULL DEFAULT '{}'",
                "calibration_json": "TEXT",
            },
        )


def build_tick_ml_study_params(
    *,
    file_path: str,
    project_root: str,
    file_mtime: float,
    product: str,
    symbol: str,
    window: int,
    horizon_ticks: int,
    hypothesis: str,
    min_success_ticks: float,
    max_rows: int,
    test_fraction: float,
    model_version: str = TICK_ML_MODEL_VERSION,
) -> dict[str, Any]:
    abs_path = os.path.abspath(file_path)
    rel_path = os.path.relpath(abs_path, project_root)
    return {
        "model_version": model_version,
        "file_path": rel_path,
        "file_mtime": float(file_mtime),
        "file_size": int(os.path.getsize(abs_path)),
        "product": str(product),
        "symbol": str(symbol),
        "window": int(window),
        "horizon_ticks": int(horizon_ticks),
        "hypothesis": str(hypothesis),
        "min_success_ticks": float(min_success_ticks),
        "max_rows": int(max_rows),
        "test_fraction": float(test_fraction),
    }


def make_tick_ml_study_key(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_tick_ml_study(db_path: str, study_key: str) -> dict | None:
    ensure_tick_ml_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        study = conn.execute(
            "SELECT * FROM tick_ml_studies WHERE study_key = ?",
            (study_key,),
        ).fetchone()
        if study is None:
            return None

        importance = pd.read_sql_query(
            """
            SELECT feature, importance, gain, split_count
            FROM tick_ml_feature_importance
            WHERE study_key = ?
            ORDER BY importance DESC, gain DESC
            """,
            conn,
            params=(study_key,),
        )
        thresholds = pd.read_sql_query(
            """
            SELECT
                feature,
                split_count,
                split_median,
                split_q25,
                split_q75,
                split_min,
                split_max,
                total_gain
            FROM tick_ml_thresholds
            WHERE study_key = ?
            ORDER BY total_gain DESC, split_count DESC
            """,
            conn,
            params=(study_key,),
        )
        artifact = conn.execute(
            "SELECT predictions_parquet FROM tick_ml_artifacts WHERE study_key = ?",
            (study_key,),
        ).fetchone()

    if artifact is None:
        return None

    predictions = _dataframe_from_blob(artifact["predictions_parquet"])
    return {
        "metrics": json.loads(study["metrics_json"]),
        "importance": importance,
        "thresholds": thresholds,
        "predictions": predictions,
        "feature_cols": json.loads(study["feature_cols_json"]),
        "hyperparams": json.loads(study["hyperparams_json"] or "{}"),
        "calibration": json.loads(study["calibration_json"]) if study["calibration_json"] else None,
        "cache": {
            "study_key": study["study_key"],
            "created_at": study["created_at"],
            "updated_at": study["updated_at"],
            "prediction_rows": int(study["prediction_rows"]),
            "model_version": study["model_version"],
        },
    }


def save_tick_ml_study(
    db_path: str,
    study_key: str,
    params: dict[str, Any],
    result: dict,
    hyperparams: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> None:
    ensure_tick_ml_tables(db_path)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    hyperparams = hyperparams or result.get("hyperparams") or {}
    calibration = calibration if calibration is not None else result.get("calibration")
    hyperparams_json = json.dumps(hyperparams, default=_json_default, sort_keys=True)
    calibration_json = json.dumps(calibration, default=_json_default, sort_keys=True) if calibration else None
    metrics_json = json.dumps(result["metrics"], default=_json_default, sort_keys=True)
    feature_cols_json = json.dumps(result["feature_cols"], default=_json_default)
    predictions = result["predictions"].copy()
    prediction_blob = _dataframe_to_blob(predictions)

    with sqlite3.connect(db_path) as conn:
        existing = conn.execute(
            "SELECT created_at FROM tick_ml_studies WHERE study_key = ?",
            (study_key,),
        ).fetchone()
        created_at = existing[0] if existing else now

        conn.execute("DELETE FROM tick_ml_feature_importance WHERE study_key = ?", (study_key,))
        conn.execute("DELETE FROM tick_ml_thresholds WHERE study_key = ?", (study_key,))
        conn.execute("DELETE FROM tick_ml_artifacts WHERE study_key = ?", (study_key,))

        conn.execute(
            """
            INSERT OR REPLACE INTO tick_ml_studies (
                study_key,
                created_at,
                updated_at,
                model_version,
                file_path,
                file_mtime,
                file_size,
                product,
                symbol,
                window,
                horizon_ticks,
                hypothesis,
                min_success_ticks,
                max_rows,
                test_fraction,
                hyperparams_json,
                calibration_json,
                metrics_json,
                feature_cols_json,
                prediction_rows
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_key,
                created_at,
                now,
                params["model_version"],
                params["file_path"],
                params["file_mtime"],
                params["file_size"],
                params["product"],
                params["symbol"],
                params["window"],
                params["horizon_ticks"],
                params["hypothesis"],
                params["min_success_ticks"],
                params["max_rows"],
                params["test_fraction"],
                hyperparams_json,
                calibration_json,
                metrics_json,
                feature_cols_json,
                len(predictions),
            ),
        )
        conn.executemany(
            """
            INSERT INTO tick_ml_feature_importance (
                study_key,
                feature,
                importance,
                gain,
                split_count
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    study_key,
                    str(row["feature"]),
                    _sql_value(row["importance"]),
                    _sql_value(row["gain"]),
                    int(row["split_count"]),
                )
                for _, row in result["importance"].iterrows()
            ],
        )
        conn.executemany(
            """
            INSERT INTO tick_ml_thresholds (
                study_key,
                feature,
                split_count,
                split_median,
                split_q25,
                split_q75,
                split_min,
                split_max,
                total_gain
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    study_key,
                    str(row["feature"]),
                    int(row["split_count"]),
                    _sql_value(row["split_median"]),
                    _sql_value(row["split_q25"]),
                    _sql_value(row["split_q75"]),
                    _sql_value(row["split_min"]),
                    _sql_value(row["split_max"]),
                    _sql_value(row["total_gain"]),
                )
                for _, row in result["thresholds"].iterrows()
            ],
        )
        conn.execute(
            """
            INSERT INTO tick_ml_artifacts (study_key, predictions_parquet)
            VALUES (?, ?)
            """,
            (study_key, sqlite3.Binary(prediction_blob)),
        )
    _record_tick_ml_research_trial(
        db_path=db_path,
        study_key=study_key,
        params=params,
        result=result,
        hyperparams=hyperparams,
        calibration=calibration,
    )


def _dataframe_to_blob(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for column, column_type in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _dataframe_from_blob(blob: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(blob))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sql_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _record_tick_ml_research_trial(
    *,
    db_path: str,
    study_key: str,
    params: dict[str, Any],
    result: dict,
    hyperparams: dict[str, Any],
    calibration: dict[str, Any] | None,
) -> None:
    metrics = result.get("metrics", {})
    sample_size, metric_value, raw_p_value, metric_name = _tick_ml_p_value(metrics)
    signature_payload = {
        "experiment_source": "tick_ml_study_cache",
        "study_key": study_key,
        "model_version": params.get("model_version"),
        "file_path": params.get("file_path"),
        "file_mtime": params.get("file_mtime"),
        "file_size": params.get("file_size"),
        "product": params.get("product"),
        "symbol": params.get("symbol"),
        "window": params.get("window"),
        "horizon_ticks": params.get("horizon_ticks"),
        "hypothesis": params.get("hypothesis"),
        "min_success_ticks": params.get("min_success_ticks"),
        "max_rows": params.get("max_rows"),
        "test_fraction": params.get("test_fraction"),
        "hyperparams": hyperparams,
        "calibration": calibration,
    }
    record_research_trial(
        db_path,
        factor_id=f"tick_ml_{params.get('hypothesis', 'study')}",
        research_family="tick_pulse_lab",
        trial_signature_payload=signature_payload,
        params=signature_payload,
        experiment_source="tick_ml_study_cache",
        asset_class="FUTURES_CN",
        vertical_metadata={
            "market_vertical": "FUTURES_CN",
            "dataset_id": params.get("file_path", ""),
            "data_frequency": "tick",
            "dataset_role": "contract_tick",
            "data_tradability": "executable",
            "data_price_source": "contract_l1_tick",
            "data_vendor": "local_parquet",
            "execution_assumption": "purged_chronological_ml_holdout",
        },
        evaluation_geometry="time_series_event_classification",
        metric_name=metric_name,
        metric_value=metric_value,
        raw_p_value=raw_p_value,
        hit_rate_p_value=raw_p_value,
        sample_size=sample_size,
        metadata={
            "study_key": study_key,
            "metrics": metrics,
            "feature_count": len(result.get("feature_cols", [])),
        },
    )


def _tick_ml_p_value(metrics: dict[str, Any]) -> tuple[int | None, float, float, str]:
    if _safe_int(metrics.get("gate_count")) > 0 and np.isfinite(_safe_float(metrics.get("gate_accuracy"))):
        n = _safe_int(metrics.get("gate_count"))
        accuracy = _safe_float(metrics.get("gate_accuracy"))
        base_rate = _safe_float(metrics.get("test_base_rate", metrics.get("test_target_rate")))
        p_value = _binomial_p_value(int(round(accuracy * n)), n, base_rate)
        return n, accuracy, p_value, "gate_accuracy_vs_base_rate"

    n = _safe_int(metrics.get("test_rows"))
    accuracy = _safe_float(metrics.get("accuracy_50"))
    target_rate = _safe_float(metrics.get("test_target_rate", metrics.get("test_base_rate")))
    if n <= 0 or not np.isfinite(accuracy):
        return None, np.nan, np.nan, "ml_holdout_accuracy"
    majority_base = max(target_rate, 1.0 - target_rate) if np.isfinite(target_rate) else 0.5
    p_value = _binomial_p_value(int(round(accuracy * n)), n, majority_base)
    return n, accuracy, p_value, "ml_accuracy_vs_majority_base"


def _binomial_p_value(successes: int, total: int, base_rate: float) -> float:
    if total <= 0 or not np.isfinite(base_rate):
        return np.nan
    base_rate = min(max(float(base_rate), 0.0), 1.0)
    if base_rate <= 0.0:
        return 0.0 if successes > 0 else 1.0
    if base_rate >= 1.0:
        return 1.0
    return float(stats.binomtest(successes, total, base_rate, alternative="greater").pvalue)


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return -1
        return int(value)
    except (TypeError, ValueError):
        return -1
