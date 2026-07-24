"""Reproducible experiment contracts and persistence for ML regression."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from oqp.research.artifacts import ModelArtifactStore
from oqp.research.model_registry import (
    DEFAULT_RESEARCH_DB_PATH,
    build_data_fingerprint,
    record_from_artifact,
    register_model_artifact,
)


ML_EXPERIMENT_TABLE = "ml_experiments"


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def new_experiment_id(model_type: str) -> str:
    prefix = str(model_type).strip().lower().replace(" ", "_") or "model"
    return f"mlx_{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class MLExperimentResult:
    """Model-independent output produced by every promoted ML trainer."""

    model_type: str
    model_name: str
    model: Any = field(repr=False)
    data_path: str
    target_col: str
    feature_cols: list[str]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    validation_policy: dict[str, Any]
    metrics: dict[str, Any]
    hyperparams: dict[str, Any]
    factor_id: str | None = None
    asset_class: str | None = None
    experiment_id: str = ""
    status: str = "trained"
    artifact_id: str | None = None
    artifact_path: str | None = None
    importance_path: str | None = None
    predictions_path: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utc_timestamp)

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = new_experiment_id(self.model_type)


def ensure_ml_experiment_table(
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_file)) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ML_EXPERIMENT_TABLE} (
                experiment_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                factor_id TEXT,
                asset_class TEXT,
                data_path TEXT NOT NULL,
                data_sha256 TEXT,
                target_col TEXT NOT NULL,
                feature_count INTEGER NOT NULL,
                feature_cols_json TEXT NOT NULL,
                validation_policy_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                hyperparams_json TEXT NOT NULL,
                prediction_rows INTEGER NOT NULL,
                artifact_id TEXT,
                artifact_path TEXT,
                importance_path TEXT,
                predictions_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_ml_experiments_model "
            f"ON {ML_EXPERIMENT_TABLE}(model_type, created_at)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_ml_experiments_factor "
            f"ON {ML_EXPERIMENT_TABLE}(factor_id, created_at)"
        )
        columns = {
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({ML_EXPERIMENT_TABLE})"
            ).fetchall()
        }
        if "predictions_path" not in columns:
            conn.execute(
                f"ALTER TABLE {ML_EXPERIMENT_TABLE} ADD COLUMN predictions_path TEXT"
            )
        conn.commit()


def register_ml_experiment(
    result: MLExperimentResult,
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
) -> None:
    """Insert or update an experiment summary without storing bulky frames."""

    ensure_ml_experiment_table(db_path)
    fingerprint = build_data_fingerprint(result.data_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {ML_EXPERIMENT_TABLE} (
                experiment_id, status, model_type, model_name, factor_id,
                asset_class, data_path, data_sha256, target_col, feature_count,
                feature_cols_json, validation_policy_json, metrics_json,
                hyperparams_json, prediction_rows, artifact_id, artifact_path,
                importance_path, predictions_path, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.experiment_id,
                result.status,
                result.model_type,
                result.model_name,
                result.factor_id,
                result.asset_class,
                fingerprint.path if fingerprint else result.data_path,
                fingerprint.sha256 if fingerprint else None,
                result.target_col,
                len(result.feature_cols),
                _json(result.feature_cols),
                _json(result.validation_policy),
                _json(result.metrics),
                _json(result.hyperparams),
                len(result.predictions),
                result.artifact_id,
                result.artifact_path,
                result.importance_path,
                result.predictions_path,
                result.error,
                result.created_at,
            ),
        )
        conn.commit()


def register_failed_ml_experiment(
    *,
    model_type: str,
    model_name: str,
    data_path: str | Path,
    target_col: str,
    validation_policy: dict[str, Any],
    error: Exception | str,
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
    factor_id: str | None = None,
    asset_class: str | None = None,
    feature_cols: list[str] | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> MLExperimentResult:
    """Record a failed Python-level experiment for operational diagnosis."""

    result = MLExperimentResult(
        model_type=model_type,
        model_name=model_name,
        model=None,
        factor_id=factor_id,
        asset_class=asset_class,
        data_path=Path(data_path).as_posix(),
        target_col=target_col,
        feature_cols=list(feature_cols or []),
        predictions=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        validation_policy=dict(validation_policy),
        metrics={},
        hyperparams=dict(hyperparams or {}),
        status="failed",
        error=str(error),
    )
    register_ml_experiment(result, db_path=db_path)
    return result


def list_ml_experiments(
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
    *,
    model_type: str | None = None,
    factor_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return compact experiment-ledger rows for dashboard and CLI consumers."""

    ensure_ml_experiment_table(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if model_type:
        clauses.append("model_type = ?")
        params.append(str(model_type).lower())
    if factor_id:
        clauses.append("factor_id = ?")
        params.append(factor_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, int(limit)))

    with closing(sqlite3.connect(Path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT * FROM {ML_EXPERIMENT_TABLE}
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def latest_ml_experiment(
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
    *,
    model_type: str | None = None,
    factor_id: str | None = None,
) -> dict[str, Any] | None:
    rows = list_ml_experiments(
        db_path,
        model_type=model_type,
        factor_id=factor_id,
        limit=1,
    )
    return rows[0] if rows else None


def persist_ml_experiment(
    result: MLExperimentResult,
    *,
    model_output_path: str | Path,
    importance_output_path: str | Path,
    predictions_output_path: str | Path,
    artifact_format: str,
    source_module: str,
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
    artifact_store: ModelArtifactStore | None = None,
) -> MLExperimentResult:
    """Persist a trained model, its importance, registry record, and ledger row."""

    model_path = Path(model_output_path)
    importance_path = Path(importance_output_path)
    predictions_path = Path(predictions_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(result.model, "save_model"):
        raise TypeError(
            f"{result.model_type} model does not expose the required save_model() method."
        )
    result.model.save_model(str(model_path))
    result.feature_importance.to_csv(importance_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)

    store = artifact_store or ModelArtifactStore()
    stored = store.archive_file(model_path, model_name=result.model_name)
    stored_importance = store.archive_file(
        importance_path,
        model_name=result.model_name,
        artifact_id=stored.artifact_id,
    )
    stored_predictions = store.archive_file(
        predictions_path,
        model_name=result.model_name,
        artifact_id=stored.artifact_id,
    )
    data_fingerprint = build_data_fingerprint(result.data_path)
    registry_record = record_from_artifact(
        artifact_id=stored.artifact_id,
        model_name=result.model_name,
        factor_id=result.factor_id,
        model_type=f"{result.model_type}_regressor",
        artifact_path=stored.path,
        artifact_format=artifact_format,
        artifact_sha256=stored.sha256,
        artifact_size_bytes=stored.size_bytes,
        legacy_path=model_path,
        source_module=source_module,
        data_fingerprint=data_fingerprint,
        feature_cols=result.feature_cols,
        target_col=result.target_col,
        split_policy=result.validation_policy,
        metrics=result.metrics,
        hyperparams=result.hyperparams,
        metadata={
            "experiment_id": result.experiment_id,
            "asset_class": result.asset_class,
            "importance_output_path": importance_path.as_posix(),
            "predictions_output_path": predictions_path.as_posix(),
            "archived_importance_path": stored_importance.path,
            "archived_predictions_path": stored_predictions.path,
        },
    )
    register_model_artifact(registry_record, db_path=db_path)

    completed = replace(
        result,
        status="completed",
        artifact_id=stored.artifact_id,
        artifact_path=stored.path,
        importance_path=stored_importance.path,
        predictions_path=stored_predictions.path,
    )
    register_ml_experiment(completed, db_path=db_path)
    return completed


def mean_daily_rank_ic(
    predictions: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    target_col: str = "target",
) -> float:
    daily_ic: list[float] = []
    for _, day in predictions.groupby("date", sort=False):
        if (
            day[prediction_col].nunique(dropna=True) < 2
            or day[target_col].nunique(dropna=True) < 2
        ):
            continue
        value = day[prediction_col].corr(day[target_col], method="spearman")
        if pd.notna(value):
            daily_ic.append(float(value))
    return float(pd.Series(daily_ic).mean()) if daily_ic else float("nan")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


__all__ = [
    "ML_EXPERIMENT_TABLE",
    "MLExperimentResult",
    "ensure_ml_experiment_table",
    "latest_ml_experiment",
    "list_ml_experiments",
    "mean_daily_rank_ic",
    "new_experiment_id",
    "persist_ml_experiment",
    "register_failed_ml_experiment",
    "register_ml_experiment",
]
