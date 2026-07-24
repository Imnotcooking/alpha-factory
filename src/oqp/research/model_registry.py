from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oqp.research.artifacts import FileFingerprint, fingerprint_file, normalize_workspace_path


MODEL_REGISTRY_TABLE = "model_artifacts"
DEFAULT_RESEARCH_DB_PATH = Path(
    os.environ.get("ALPHA_RESEARCH_DB_PATH", "runtime/db/research/research_memory.db")
)


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ModelArtifactRecord:
    artifact_id: str
    model_name: str
    model_type: str
    artifact_path: str
    artifact_format: str
    factor_id: str | None = None
    legacy_path: str | None = None
    source_module: str | None = None
    data_path: str | None = None
    data_sha256: str | None = None
    data_mtime_ns: int | None = None
    data_size_bytes: int | None = None
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None
    feature_cols: list[str] = field(default_factory=list)
    target_col: str | None = None
    split_policy: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    hyperparams: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_timestamp)


def ensure_model_registry_tables(db_path: str | Path = DEFAULT_RESEARCH_DB_PATH) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_file)) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MODEL_REGISTRY_TABLE} (
                artifact_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                factor_id TEXT,
                model_type TEXT NOT NULL,
                artifact_format TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                legacy_path TEXT,
                source_module TEXT,
                data_path TEXT,
                data_sha256 TEXT,
                data_mtime_ns INTEGER,
                data_size_bytes INTEGER,
                artifact_sha256 TEXT,
                artifact_size_bytes INTEGER,
                feature_count INTEGER,
                feature_cols_json TEXT,
                target_col TEXT,
                split_policy_json TEXT,
                metrics_json TEXT,
                hyperparams_json TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_artifacts_name ON model_artifacts(model_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_artifacts_factor ON model_artifacts(factor_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_artifacts_created ON model_artifacts(created_at)")
        conn.commit()


def build_data_fingerprint(
    data_path: str | Path | None,
    *,
    include_hash: bool = True,
    workspace_root: str | Path | None = None,
) -> FileFingerprint | None:
    if data_path is None:
        return None
    return fingerprint_file(data_path, include_hash=include_hash, workspace_root=workspace_root)


def register_model_artifact(
    record: ModelArtifactRecord,
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
) -> None:
    ensure_model_registry_tables(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {MODEL_REGISTRY_TABLE} (
                artifact_id,
                model_name,
                factor_id,
                model_type,
                artifact_format,
                artifact_path,
                legacy_path,
                source_module,
                data_path,
                data_sha256,
                data_mtime_ns,
                data_size_bytes,
                artifact_sha256,
                artifact_size_bytes,
                feature_count,
                feature_cols_json,
                target_col,
                split_policy_json,
                metrics_json,
                hyperparams_json,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.artifact_id,
                record.model_name,
                record.factor_id,
                record.model_type,
                record.artifact_format,
                record.artifact_path,
                record.legacy_path,
                record.source_module,
                record.data_path,
                record.data_sha256,
                record.data_mtime_ns,
                record.data_size_bytes,
                record.artifact_sha256,
                record.artifact_size_bytes,
                int(len(record.feature_cols)),
                _json(record.feature_cols),
                record.target_col,
                _json(record.split_policy),
                _json(record.metrics),
                _json(record.hyperparams),
                _json(record.metadata),
                record.created_at,
            ),
        )
        conn.commit()


def latest_model_artifact(
    model_name: str,
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
) -> dict[str, Any] | None:
    ensure_model_registry_tables(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"""
            SELECT *
            FROM {MODEL_REGISTRY_TABLE}
            WHERE model_name = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (model_name,),
        ).fetchone()
    return dict(row) if row else None


def list_model_artifacts(
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
    *,
    model_type: str | None = None,
    factor_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent model artifacts for dashboard and audit consumers."""

    ensure_model_registry_tables(db_path)
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
            SELECT * FROM {MODEL_REGISTRY_TABLE}
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def record_from_artifact(
    *,
    artifact_id: str,
    model_name: str,
    model_type: str,
    artifact_path: str | Path,
    artifact_format: str,
    artifact_sha256: str | None = None,
    artifact_size_bytes: int | None = None,
    factor_id: str | None = None,
    legacy_path: str | Path | None = None,
    source_module: str | None = None,
    data_fingerprint: FileFingerprint | None = None,
    feature_cols: list[str] | None = None,
    target_col: str | None = None,
    split_policy: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    hyperparams: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> ModelArtifactRecord:
    return ModelArtifactRecord(
        artifact_id=artifact_id,
        model_name=model_name,
        factor_id=factor_id,
        model_type=model_type,
        artifact_format=artifact_format,
        artifact_path=normalize_workspace_path(artifact_path, workspace_root),
        legacy_path=normalize_workspace_path(legacy_path, workspace_root) if legacy_path else None,
        source_module=source_module,
        data_path=data_fingerprint.path if data_fingerprint else None,
        data_sha256=data_fingerprint.sha256 if data_fingerprint else None,
        data_mtime_ns=data_fingerprint.mtime_ns if data_fingerprint else None,
        data_size_bytes=data_fingerprint.size_bytes if data_fingerprint else None,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size_bytes,
        feature_cols=list(feature_cols or []),
        target_col=target_col,
        split_policy=dict(split_policy or {}),
        metrics=dict(metrics or {}),
        hyperparams=dict(hyperparams or {}),
        metadata=dict(metadata or {}),
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
