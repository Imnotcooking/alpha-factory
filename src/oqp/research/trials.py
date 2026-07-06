from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from oqp.research.multiple_testing import (
    benjamini_hochberg_q_values,
    bonferroni_p_value,
    holm_bonferroni_adjust,
    significance_label,
    stable_trial_hash,
)


VERTICAL_TRIAL_COLUMNS = {
    "market_vertical": "TEXT",
    "dataset_id": "TEXT",
    "universe_id": "TEXT",
    "data_frequency": "TEXT",
    "dataset_role": "TEXT",
    "data_tradability": "TEXT",
    "data_price_source": "TEXT",
    "data_roll_model": "TEXT",
    "data_liquidity_model": "TEXT",
    "data_execution_reality": "TEXT",
    "data_vendor": "TEXT",
    "execution_assumption": "TEXT",
}


@dataclass(frozen=True)
class ResearchTrialRecord:
    run_id: str
    factor_id: str
    research_family: str
    trial_signature: str
    trial_count: int
    adjusted_p_value: float
    fdr_q_value: float
    significance: str


@dataclass(frozen=True)
class EvidenceTicketRecord:
    ticket_id: str
    title: str
    source_page: str
    evidence_type: str
    stage: str
    status: str
    decision: str
    factor_id: str
    research_family: str
    run_id: str
    trial_signature: str
    metric_name: str
    metric_value: float
    confidence_score: float


def ensure_research_trial_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            factor_id TEXT,
            research_family TEXT,
            trial_signature TEXT,
            params_hash TEXT,
            asset_class TEXT,
            evaluation_geometry TEXT,
            metric_name TEXT,
            raw_p_value REAL,
            metric_p_value REAL,
            hit_rate_p_value REAL,
            sharpe_p_value REAL,
            bonferroni_p_value REAL,
            holm_p_value REAL,
            fdr_q_value REAL,
            trial_count_m INTEGER,
            significance TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_columns(
        conn,
        "research_trials",
        {
            **VERTICAL_TRIAL_COLUMNS,
            "experiment_source": "TEXT",
            "metric_value": "REAL",
            "sample_size": "INTEGER",
            "metadata_json": "TEXT",
        },
    )


def ensure_research_evidence_ticket_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_evidence_tickets (
            ticket_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_page TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            decision TEXT NOT NULL,
            thesis TEXT,
            factor_id TEXT,
            research_family TEXT,
            run_id TEXT,
            trial_signature TEXT,
            metric_name TEXT,
            metric_value REAL,
            confidence_score REAL,
            priority INTEGER,
            metrics_json TEXT,
            artifacts_json TEXT,
            context_json TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_columns(
        conn,
        "research_evidence_tickets",
        {
            "title": "TEXT",
            "source_page": "TEXT",
            "evidence_type": "TEXT",
            "stage": "TEXT",
            "status": "TEXT",
            "decision": "TEXT",
            "thesis": "TEXT",
            "factor_id": "TEXT",
            "research_family": "TEXT",
            "run_id": "TEXT",
            "trial_signature": "TEXT",
            "metric_name": "TEXT",
            "metric_value": "REAL",
            "confidence_score": "REAL",
            "priority": "INTEGER",
            "metrics_json": "TEXT",
            "artifacts_json": "TEXT",
            "context_json": "TEXT",
            "metadata_json": "TEXT",
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_evidence_tickets_status "
        "ON research_evidence_tickets(status, stage)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_evidence_tickets_factor "
        "ON research_evidence_tickets(factor_id, research_family)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_evidence_tickets_trial "
        "ON research_evidence_tickets(trial_signature, run_id)"
    )


def record_research_trial(
    db_path: str,
    *,
    factor_id: str,
    research_family: str,
    trial_signature_payload: dict[str, Any],
    params: dict[str, Any] | None = None,
    run_id: str | None = None,
    experiment_source: str = "",
    asset_class: str = "",
    vertical_metadata: dict[str, Any] | None = None,
    evaluation_geometry: str = "",
    metric_name: str = "",
    metric_value: float | None = None,
    raw_p_value: float | None = None,
    metric_p_value: float | None = None,
    hit_rate_p_value: float | None = None,
    sharpe_p_value: float | None = None,
    sample_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResearchTrialRecord:
    signature_payload = dict(trial_signature_payload)
    trial_signature = stable_trial_hash(signature_payload)
    run_id = run_id or f"{_slug(experiment_source or factor_id)}_{trial_signature}"
    params_hash = stable_trial_hash(params if params is not None else signature_payload)
    vertical_metadata = vertical_metadata or {}
    metadata = metadata or {}

    p_raw = _coerce_p_value(raw_p_value)
    p_metric = _coerce_p_value(metric_p_value)
    p_hit = _coerce_p_value(hit_rate_p_value)
    p_sharpe = _coerce_p_value(sharpe_p_value)
    if not np.isfinite(p_raw):
        p_raw = _first_finite(p_metric, p_hit, p_sharpe)

    with closing(sqlite3.connect(db_path)) as conn:
        ensure_research_trial_ledger(conn)
        existing = conn.execute(
            """
            SELECT trial_count_m, bonferroni_p_value, fdr_q_value, significance
            FROM research_trials
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if existing is not None:
            current_count = conn.execute(
                "SELECT COUNT(DISTINCT trial_signature) FROM research_trials WHERE research_family = ?",
                (research_family,),
            ).fetchone()[0]
            if int(existing[0] or 0) == int(current_count or 0) and current_count > 0:
                return ResearchTrialRecord(
                    run_id=run_id,
                    factor_id=factor_id,
                    research_family=research_family,
                    trial_signature=trial_signature,
                    trial_count=int(current_count),
                    adjusted_p_value=float(existing[1]) if existing[1] is not None else np.nan,
                    fdr_q_value=float(existing[2]) if existing[2] is not None else np.nan,
                    significance=str(existing[3] or "missing"),
                )
        conn.execute(
            """
            INSERT OR REPLACE INTO research_trials (
                run_id, factor_id, research_family, trial_signature, params_hash,
                asset_class, market_vertical, dataset_id, universe_id, data_frequency,
                dataset_role, data_tradability, data_price_source, data_roll_model,
                data_liquidity_model, data_execution_reality, data_vendor, execution_assumption,
                evaluation_geometry, metric_name, raw_p_value, metric_p_value, hit_rate_p_value,
                sharpe_p_value, trial_count_m, significance, experiment_source, metric_value,
                sample_size, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                factor_id,
                research_family,
                trial_signature,
                params_hash,
                asset_class,
                _text(vertical_metadata.get("market_vertical")),
                _text(vertical_metadata.get("dataset_id")),
                _text(vertical_metadata.get("universe_id")),
                _text(vertical_metadata.get("data_frequency")),
                _text(vertical_metadata.get("dataset_role")),
                _text(vertical_metadata.get("data_tradability")),
                _text(vertical_metadata.get("data_price_source")),
                _text(vertical_metadata.get("data_roll_model")),
                _text(vertical_metadata.get("data_liquidity_model")),
                _text(vertical_metadata.get("data_execution_reality")),
                _text(vertical_metadata.get("data_vendor")),
                _text(vertical_metadata.get("execution_assumption")),
                evaluation_geometry,
                metric_name,
                _sql_float(p_raw),
                _sql_float(p_metric),
                _sql_float(p_hit),
                _sql_float(p_sharpe),
                1,
                "pending",
                experiment_source,
                _sql_float(metric_value),
                int(sample_size) if sample_size is not None else None,
                json.dumps(metadata, default=str, sort_keys=True, ensure_ascii=False),
            ),
        )
        result = refresh_multiple_testing_adjustments(conn, research_family, trial_signature)
        conn.commit()
    return ResearchTrialRecord(
        run_id=run_id,
        factor_id=factor_id,
        research_family=research_family,
        trial_signature=trial_signature,
        trial_count=result["trial_count"],
        adjusted_p_value=result["adjusted_p_value"],
        fdr_q_value=result["fdr_q_value"],
        significance=result["significance"],
    )


def make_evidence_ticket_id(payload: dict[str, Any]) -> str:
    return f"ticket_{stable_trial_hash(payload)}"


def record_evidence_ticket(
    db_path: str,
    *,
    title: str,
    source_page: str,
    evidence_type: str,
    stage: str,
    status: str = "open",
    decision: str = "candidate",
    thesis: str = "",
    factor_id: str = "",
    research_family: str = "",
    run_id: str = "",
    trial_signature: str = "",
    metric_name: str = "",
    metric_value: float | None = None,
    confidence_score: float | None = None,
    priority: int | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    ticket_id: str | None = None,
) -> EvidenceTicketRecord:
    title = _required_text(title, "title")
    source_page = _required_text(source_page, "source_page")
    evidence_type = _required_text(evidence_type, "evidence_type")
    stage = _required_text(stage, "stage")
    status = _required_text(status, "status")
    decision = _required_text(decision, "decision")

    ticket_id = ticket_id or make_evidence_ticket_id(
        {
            "title": title,
            "source_page": source_page,
            "evidence_type": evidence_type,
            "stage": stage,
            "factor_id": factor_id,
            "research_family": research_family,
            "run_id": run_id,
            "trial_signature": trial_signature,
        }
    )

    with closing(sqlite3.connect(db_path)) as conn:
        ensure_research_evidence_ticket_ledger(conn)
        conn.execute(
            """
            INSERT INTO research_evidence_tickets (
                ticket_id, title, source_page, evidence_type, stage, status,
                decision, thesis, factor_id, research_family, run_id,
                trial_signature, metric_name, metric_value, confidence_score,
                priority, metrics_json, artifacts_json, context_json,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                title = excluded.title,
                source_page = excluded.source_page,
                evidence_type = excluded.evidence_type,
                stage = excluded.stage,
                status = excluded.status,
                decision = excluded.decision,
                thesis = excluded.thesis,
                factor_id = excluded.factor_id,
                research_family = excluded.research_family,
                run_id = excluded.run_id,
                trial_signature = excluded.trial_signature,
                metric_name = excluded.metric_name,
                metric_value = excluded.metric_value,
                confidence_score = excluded.confidence_score,
                priority = excluded.priority,
                metrics_json = excluded.metrics_json,
                artifacts_json = excluded.artifacts_json,
                context_json = excluded.context_json,
                metadata_json = excluded.metadata_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                ticket_id,
                title,
                source_page,
                evidence_type,
                stage,
                status,
                decision,
                thesis,
                factor_id,
                research_family,
                run_id,
                trial_signature,
                metric_name,
                _sql_float(metric_value),
                _sql_float(confidence_score),
                int(priority) if priority is not None else None,
                _json_text(metrics or {}),
                _json_text(artifacts or []),
                _json_text(context or {}),
                _json_text(metadata or {}),
            ),
        )
        conn.commit()

    return EvidenceTicketRecord(
        ticket_id=ticket_id,
        title=title,
        source_page=source_page,
        evidence_type=evidence_type,
        stage=stage,
        status=status,
        decision=decision,
        factor_id=factor_id,
        research_family=research_family,
        run_id=run_id,
        trial_signature=trial_signature,
        metric_name=metric_name,
        metric_value=float(metric_value) if metric_value is not None else np.nan,
        confidence_score=float(confidence_score) if confidence_score is not None else np.nan,
    )


def list_evidence_tickets(
    db_path: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    source_page: str | None = None,
    factor_id: str | None = None,
    research_family: str | None = None,
    limit: int | None = None,
    parse_json: bool = True,
) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as conn:
        ensure_research_evidence_ticket_ledger(conn)
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("status", status),
            ("stage", stage),
            ("source_page", source_page),
            ("factor_id", factor_id),
            ("research_family", research_family),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)

        query = "SELECT * FROM research_evidence_tickets"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY priority IS NULL, priority ASC, updated_at DESC, created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        frame = pd.read_sql_query(query, conn, params=params)

    if frame.empty or not parse_json:
        return frame
    for source_col, target_col, fallback in (
        ("metrics_json", "metrics", {}),
        ("artifacts_json", "artifacts", []),
        ("context_json", "context", {}),
        ("metadata_json", "metadata", {}),
    ):
        if source_col in frame.columns:
            frame[target_col] = frame[source_col].map(lambda value: _json_value(value, fallback))
    return frame


def get_evidence_ticket(db_path: str, ticket_id: str) -> dict[str, Any] | None:
    frame = list_evidence_tickets(db_path, parse_json=True)
    if frame.empty:
        return None
    match = frame[frame["ticket_id"] == ticket_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def update_evidence_ticket_status(
    db_path: str,
    ticket_id: str,
    *,
    status: str,
    decision: str | None = None,
    reviewer_note: str = "",
    metadata_patch: dict[str, Any] | None = None,
    reviewer: str = "research_dashboard",
) -> dict[str, Any]:
    ticket_id = _required_text(ticket_id, "ticket_id")
    status = _required_text(status, "status")
    timestamp = _utc_timestamp()

    with closing(sqlite3.connect(db_path)) as conn:
        ensure_research_evidence_ticket_ledger(conn)
        row = conn.execute(
            """
            SELECT metadata_json
            FROM research_evidence_tickets
            WHERE ticket_id = ?
            """,
            (ticket_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Evidence ticket not found: {ticket_id}")

        metadata = _json_value(row[0], {})
        if not isinstance(metadata, dict):
            metadata = {"legacy_metadata": metadata}
        if metadata_patch:
            metadata.update(metadata_patch)

        note = _text(reviewer_note).strip()
        if note:
            review_notes = metadata.get("review_notes")
            if not isinstance(review_notes, list):
                review_notes = []
            review_notes.append(
                {
                    "status": status,
                    "decision": decision or "",
                    "note": note,
                    "reviewer": _text(reviewer) or "research_dashboard",
                    "timestamp": timestamp,
                }
            )
            metadata["review_notes"] = review_notes

        metadata["last_review_status"] = status
        metadata["last_reviewed_at"] = timestamp
        metadata["last_reviewer"] = _text(reviewer) or "research_dashboard"
        if decision is not None:
            metadata["last_review_decision"] = decision

        if decision is None:
            conn.execute(
                """
                UPDATE research_evidence_tickets
                SET status = ?,
                    metadata_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE ticket_id = ?
                """,
                (status, _json_text(metadata), ticket_id),
            )
        else:
            conn.execute(
                """
                UPDATE research_evidence_tickets
                SET status = ?,
                    decision = ?,
                    metadata_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE ticket_id = ?
                """,
                (status, _required_text(decision, "decision"), _json_text(metadata), ticket_id),
            )
        conn.commit()

    updated = get_evidence_ticket(db_path, ticket_id)
    if updated is None:
        raise KeyError(f"Evidence ticket not found after update: {ticket_id}")
    return updated


def refresh_multiple_testing_adjustments(
    conn: sqlite3.Connection,
    research_family: str,
    target_signature: str | None = None,
) -> dict[str, Any]:
    trials = pd.read_sql_query(
        """
        SELECT trial_id, run_id, research_family, trial_signature, raw_p_value
        FROM research_trials
        WHERE research_family = ?
        """,
        conn,
        params=(research_family,),
    )
    if trials.empty:
        return {
            "trial_count": 0,
            "adjusted_p_value": np.nan,
            "fdr_q_value": np.nan,
            "significance": "missing",
        }

    trial_count = int(trials["trial_signature"].nunique())
    latest_by_signature = (
        trials.sort_values("trial_id")
        .drop_duplicates("trial_signature", keep="last")
        .set_index("trial_signature")
    )
    raw_p_values = pd.to_numeric(latest_by_signature["raw_p_value"], errors="coerce")
    holm_values = holm_bonferroni_adjust(raw_p_values)
    fdr_values = benjamini_hochberg_q_values(raw_p_values)

    result = {
        "trial_count": trial_count,
        "adjusted_p_value": np.nan,
        "fdr_q_value": np.nan,
        "significance": "missing",
    }
    backtest_exists = _table_exists(conn, "backtest_runs")
    for signature, raw_p_value in raw_p_values.items():
        bonferroni = bonferroni_p_value(float(raw_p_value), trial_count)
        holm = float(holm_values.loc[signature]) if signature in holm_values.index else np.nan
        fdr_q = float(fdr_values.loc[signature]) if signature in fdr_values.index else np.nan
        significance = significance_label(float(raw_p_value), bonferroni)
        conn.execute(
            """
            UPDATE research_trials
            SET trial_count_m = ?,
                bonferroni_p_value = ?,
                holm_p_value = ?,
                fdr_q_value = ?,
                significance = ?
            WHERE research_family = ? AND trial_signature = ?
            """,
            (trial_count, bonferroni, holm, fdr_q, significance, research_family, signature),
        )
        if backtest_exists:
            conn.execute(
                """
                UPDATE backtest_runs
                SET stat_trial_count = ?,
                    stat_adjusted_p_value = ?,
                    stat_holm_p_value = ?,
                    stat_fdr_q_value = ?,
                    stat_significance = ?
                WHERE stat_research_family = ? AND stat_trial_signature = ?
                """,
                (trial_count, bonferroni, holm, fdr_q, significance, research_family, signature),
            )
        if target_signature is None or signature == target_signature:
            result = {
                "trial_count": trial_count,
                "adjusted_p_value": bonferroni,
                "fdr_q_value": fdr_q,
                "significance": significance,
            }
    return result


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for column, column_type in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def _first_finite(*values: float) -> float:
    for value in values:
        if np.isfinite(value):
            return float(value)
    return np.nan


def _coerce_p_value(value: float | None) -> float:
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value):
        return np.nan
    return float(min(max(value, 0.0), 1.0))


def _sql_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _required_text(value: Any, name: str) -> str:
    text = _text(value).strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _json_text(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, ensure_ascii=False)


def _json_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        text = str(value).strip()
    except Exception:
        return fallback
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).lower())
    return cleaned.strip("_")[:32] or "trial"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
