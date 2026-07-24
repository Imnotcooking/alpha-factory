"""Separate operational study storage from summarized research evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from oqp.optimization.contracts import OptimizationStudyResult, OptimizationStudySpec
from oqp.optimization.parameter_spaces import ComponentParameterSchema


class OptimizationStudyStore:
    def __init__(
        self,
        *,
        state_db_path: str | Path = "runtime/state/optimization/optuna.sqlite3",
        registry_db_path: str | Path = "runtime/research/alpha_lab.db",
        artifact_root: str | Path = "runtime/artifacts/research/optimization",
    ) -> None:
        self.state_db_path = Path(state_db_path)
        self.registry_db_path = Path(registry_db_path)
        self.artifact_root = Path(artifact_root)

    @property
    def optuna_storage_uri(self) -> str:
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.state_db_path.resolve().as_posix()}"

    def register_start(
        self,
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
    ) -> None:
        self.registry_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.registry_db_path) as conn:
            self._ensure_tables(conn)
            existing = conn.execute(
                """
                SELECT study_fingerprint, parameter_schema_fingerprint
                FROM optimization_studies
                WHERE study_id=?
                """,
                (spec.study_id,),
            ).fetchone()
            if existing is not None:
                existing_study, existing_schema = existing
                if existing_study != spec.fingerprint:
                    raise ValueError(
                        f"Study ID {spec.study_id!r} already exists with "
                        "different frozen inputs"
                    )
                if existing_schema != schema.fingerprint:
                    raise ValueError(
                        f"Study ID {spec.study_id!r} already exists with "
                        "a different parameter schema"
                    )
            conn.execute(
                """
                INSERT INTO optimization_studies (
                    study_id, study_fingerprint, component_id, component_type,
                    purpose, sampler_id, parameter_schema_fingerprint,
                    status, spec_json, schema_json, started_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(study_id) DO UPDATE SET
                    sampler_id=excluded.sampler_id,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    spec.study_id,
                    spec.fingerprint,
                    spec.component_id,
                    schema.component_type,
                    spec.purpose.value,
                    spec.sampler_id,
                    schema.fingerprint,
                    "running",
                    _json(spec.to_dict()),
                    _json(schema.to_dict()),
                    _utc_now(),
                    _utc_now(),
                ),
            )
            conn.commit()

    def persist_result(
        self,
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
        result_payload: Mapping[str, Any],
        trial_payload: list[Mapping[str, Any]],
    ) -> str:
        study_dir = self.artifact_root / spec.study_id
        study_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = study_dir / "study_result.json"
        trials_path = study_dir / "trials.json"
        _write_immutable_text(
            artifact_path,
            _json(dict(result_payload), indent=2),
        )
        _write_immutable_text(
            trials_path,
            _json(list(trial_payload), indent=2),
        )
        return artifact_path.as_posix()

    def register_complete(
        self,
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
        result: OptimizationStudyResult,
    ) -> None:
        self.registry_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.registry_db_path) as conn:
            self._ensure_tables(conn)
            conn.execute(
                """
                UPDATE optimization_studies
                SET status=?, trial_count=?, result_json=?, artifact_path=?,
                    completed_at=?, updated_at=?
                WHERE study_id=?
                """,
                (
                    "complete",
                    result.trial_count,
                    _json(result.to_dict()),
                    result.artifact_path,
                    _utc_now(),
                    _utc_now(),
                    spec.study_id,
                ),
            )
            conn.execute(
                "DELETE FROM optimization_candidates WHERE study_id=?",
                (spec.study_id,),
            )
            for candidate in result.candidates:
                conn.execute(
                    """
                    INSERT INTO optimization_candidates (
                        study_id, trial_number, parameters_json,
                        objective_values_json, metrics_json, feasible,
                        schema_fingerprint, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.study_id,
                        candidate.trial_number,
                        _json(dict(candidate.parameters)),
                        _json(list(candidate.objective_values)),
                        _json(dict(candidate.metrics)),
                        int(candidate.feasible),
                        schema.fingerprint,
                        _utc_now(),
                    ),
                )
            conn.commit()

    @staticmethod
    def _ensure_tables(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_studies (
                study_id TEXT PRIMARY KEY,
                study_fingerprint TEXT NOT NULL,
                component_id TEXT NOT NULL,
                component_type TEXT NOT NULL,
                purpose TEXT NOT NULL,
                sampler_id TEXT NOT NULL,
                parameter_schema_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                trial_count INTEGER DEFAULT 0,
                spec_json TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                result_json TEXT,
                artifact_path TEXT,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_candidates (
                study_id TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                parameters_json TEXT NOT NULL,
                objective_values_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                feasible INTEGER NOT NULL,
                schema_fingerprint TEXT NOT NULL,
                created_at TEXT,
                PRIMARY KEY (study_id, trial_number)
            )
            """
        )


def _json(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
        indent=indent,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_immutable_text(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(
                f"Immutable optimization artifact already exists with "
                f"different content: {path}"
            )
        return
    path.write_text(payload, encoding="utf-8")


__all__ = ["OptimizationStudyStore"]
