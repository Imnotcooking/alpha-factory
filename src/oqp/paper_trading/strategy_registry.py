"""Runtime registry for strategies approved to run in paper trading."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.contracts import CandidateStatus, StrategyCandidate
from oqp.paper_trading.ledger import ensure_paper_trading_schema


class PaperStrategyStatus(str, Enum):
    CANDIDATE = "paper_candidate"
    RUNNING = "paper_running"
    PAUSED = "paused"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class PaperStrategyGateCheck:
    name: str
    passed: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PaperStrategyGateResult:
    strategy_id: str | None
    passed: bool
    message: str
    checks: tuple[PaperStrategyGateCheck, ...]
    record: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "passed": self.passed,
            "message": self.message,
            "checks": [check.to_dict() for check in self.checks],
            "record": self.record,
        }


@dataclass(frozen=True, slots=True)
class PaperStrategyRegistryWriteResult:
    db_path: Path
    strategy_id: str
    market_vertical: str
    candidate_id: str
    status: PaperStrategyStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "strategy_id": self.strategy_id,
            "market_vertical": self.market_vertical,
            "candidate_id": self.candidate_id,
            "status": self.status.value,
        }


def upsert_paper_strategy_from_candidate(
    db_path: str | Path,
    candidate: StrategyCandidate,
    *,
    status: PaperStrategyStatus | str = PaperStrategyStatus.RUNNING,
    max_order_notional: float | None = None,
    max_daily_notional: float | None = None,
    allowed_symbols: tuple[str, ...] | list[str] = (),
    rebalance_frequency: str | None = None,
    kill_switch: bool = False,
    approved_by: str | None = None,
    approved_at: datetime | str | None = None,
    notes: str | None = None,
    source_artifact: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PaperStrategyRegistryWriteResult:
    """Record that a strategy candidate is authorized for paper automation."""

    paper_status = _paper_strategy_status(status)
    if paper_status == PaperStrategyStatus.RUNNING and not (
        candidate.can_enter_paper_queue
        or candidate.promotion_status == CandidateStatus.PAPER_RUNNING
    ):
        raise ValueError(
            "Only paper-queue-eligible candidates can be approved for paper running."
        )

    path = ensure_paper_trading_schema(db_path)
    market_vertical = candidate.target_market_vertical
    timestamp = _datetime_text(approved_at or datetime.now(timezone.utc))
    max_order_value = (
        max_order_notional
        if max_order_notional is not None
        else candidate.safety_limits.max_order_notional
    )
    metadata_payload = {
        "native_market_vertical": candidate.native_market_vertical,
        "tested_market_vertical": candidate.tested_market_vertical,
        "intended_market_verticals": list(candidate.intended_market_verticals),
        "data_frequency": candidate.data_frequency,
        "data_vendor": candidate.data_vendor,
        "evaluation_geometry": candidate.evaluation_geometry,
        "ic_metric": candidate.ic_metric,
        "candidate_metrics": {
            "holdout_ic": candidate.metrics.holdout_ic,
            "sharpe_ratio": candidate.metrics.sharpe_ratio,
            "max_drawdown": candidate.metrics.max_drawdown,
            "metric_p_value": candidate.metrics.metric_p_value,
        },
        **dict(metadata or {}),
    }

    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT INTO paper_strategy_registry (
                strategy_id,
                market_vertical,
                candidate_id,
                status,
                source,
                research_run_id,
                approved_broker_profile,
                max_order_notional,
                max_daily_notional,
                allowed_symbols_json,
                rebalance_frequency,
                kill_switch,
                approved_by,
                approved_at,
                notes,
                source_artifact,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy_id, market_vertical) DO UPDATE SET
                candidate_id = excluded.candidate_id,
                status = excluded.status,
                source = excluded.source,
                research_run_id = excluded.research_run_id,
                approved_broker_profile = excluded.approved_broker_profile,
                max_order_notional = excluded.max_order_notional,
                max_daily_notional = excluded.max_daily_notional,
                allowed_symbols_json = excluded.allowed_symbols_json,
                rebalance_frequency = excluded.rebalance_frequency,
                kill_switch = excluded.kill_switch,
                approved_by = excluded.approved_by,
                approved_at = excluded.approved_at,
                notes = excluded.notes,
                source_artifact = excluded.source_artifact,
                metadata_json = excluded.metadata_json
            """,
            (
                candidate.strategy_id,
                market_vertical,
                candidate.candidate_id,
                paper_status.value,
                candidate.source,
                candidate.research_run_id,
                candidate.approved_broker_profile,
                _optional_float(max_order_value),
                _optional_float(max_daily_notional),
                json.dumps(_clean_symbols(allowed_symbols), sort_keys=True),
                rebalance_frequency,
                1 if kill_switch else 0,
                approved_by,
                timestamp,
                notes or candidate.notes,
                source_artifact,
                json.dumps(metadata_payload, sort_keys=True),
            ),
        )
        conn.commit()

    return PaperStrategyRegistryWriteResult(
        db_path=path,
        strategy_id=candidate.strategy_id,
        market_vertical=market_vertical,
        candidate_id=candidate.candidate_id,
        status=paper_status,
    )


def load_paper_strategy_registry(
    db_path: str | Path,
    *,
    status: PaperStrategyStatus | str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    columns = _registry_columns()
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    ensure_paper_trading_schema(path)
    where = ""
    params: list[Any] = []
    if status is not None:
        where = "WHERE status = ?"
        params.append(_paper_strategy_status(status).value)
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(
            f"""
            SELECT {", ".join(columns)}
            FROM paper_strategy_registry
            {where}
            ORDER BY approved_at DESC, strategy_id ASC
            LIMIT ?
            """,
            conn,
            params=(*params, max(int(limit), 1)),
        )


def load_paper_strategy_record(
    db_path: str | Path,
    strategy_id: str | None,
    *,
    market_vertical: str | None = None,
) -> dict[str, Any] | None:
    if not strategy_id:
        return None
    path = Path(db_path)
    if not path.exists():
        return None
    ensure_paper_trading_schema(path)
    clauses = ["strategy_id = ?"]
    params: list[Any] = [strategy_id]
    if market_vertical:
        clauses.append("market_vertical = ?")
        params.append(market_vertical)
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"""
            SELECT {", ".join(_registry_columns())}
            FROM paper_strategy_registry
            WHERE {" AND ".join(clauses)}
            ORDER BY
                CASE WHEN status = 'paper_running' THEN 0 ELSE 1 END,
                approved_at DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["allowed_symbols"] = _json_list(record.get("allowed_symbols_json"))
    record["metadata"] = _json_dict(record.get("metadata_json"))
    record["kill_switch"] = bool(record.get("kill_switch"))
    return record


def is_paper_strategy_running(record: dict[str, Any] | None) -> bool:
    return bool(
        record
        and record.get("status") == PaperStrategyStatus.RUNNING.value
        and not bool(record.get("kill_switch"))
    )


def review_paper_strategy_gate(
    db_path: str | Path,
    proposal: Any,
) -> PaperStrategyGateResult:
    strategy_ids = _proposal_strategy_ids(proposal)
    strategy_id = next(iter(strategy_ids)) if len(strategy_ids) == 1 else None
    record = load_paper_strategy_record(db_path, strategy_id)
    checks = [
        PaperStrategyGateCheck(
            name="Strategy ID present",
            passed=bool(strategy_ids),
            detail=", ".join(sorted(strategy_ids)) if strategy_ids else "missing",
        ),
        PaperStrategyGateCheck(
            name="Single strategy",
            passed=len(strategy_ids) == 1,
            detail=f"strategy_ids={', '.join(sorted(strategy_ids)) or 'none'}",
        ),
        PaperStrategyGateCheck(
            name="Strategy registered",
            passed=record is not None,
            detail=(
                f"{strategy_id} found in paper strategy registry"
                if record is not None
                else f"{strategy_id or 'missing'} is not registered"
            ),
        ),
        PaperStrategyGateCheck(
            name="Strategy paper-running",
            passed=is_paper_strategy_running(record),
            detail=_strategy_status_detail(record),
        ),
        _allowed_symbols_check(record, proposal),
        _max_order_notional_check(record, proposal),
    ]
    blockers = [check.name for check in checks if not check.passed]
    passed = not blockers
    return PaperStrategyGateResult(
        strategy_id=strategy_id,
        passed=passed,
        message=(
            "Paper strategy gate passed."
            if passed
            else "Blocked by: " + ", ".join(blockers)
        ),
        checks=tuple(checks),
        record=record,
    )


def _registry_columns() -> list[str]:
    return [
        "strategy_id",
        "market_vertical",
        "candidate_id",
        "status",
        "source",
        "research_run_id",
        "approved_broker_profile",
        "max_order_notional",
        "max_daily_notional",
        "allowed_symbols_json",
        "rebalance_frequency",
        "kill_switch",
        "approved_by",
        "approved_at",
        "notes",
        "source_artifact",
        "metadata_json",
    ]


def _proposal_strategy_ids(proposal: Any) -> set[str]:
    ids: set[str] = set()
    proposal_strategy = _text(getattr(proposal, "strategy_id", None))
    if proposal_strategy:
        ids.add(proposal_strategy)
    for intent in getattr(proposal, "intents", ()) or ():
        intent_strategy = _text(getattr(intent, "strategy_id", None))
        if intent_strategy:
            ids.add(intent_strategy)
    return ids


def _strategy_status_detail(record: dict[str, Any] | None) -> str:
    if not record:
        return "missing paper strategy registry record"
    return (
        f"status={record.get('status') or 'missing'}, "
        f"kill_switch={str(bool(record.get('kill_switch'))).lower()}"
    )


def _allowed_symbols_check(
    record: dict[str, Any] | None,
    proposal: Any,
) -> PaperStrategyGateCheck:
    allowed = tuple(record.get("allowed_symbols", ())) if record else ()
    symbols = tuple(
        str(getattr(getattr(intent, "instrument", None), "symbol", "")).upper()
        for intent in getattr(proposal, "intents", ()) or ()
    )
    symbols = tuple(symbol for symbol in symbols if symbol)
    if not allowed:
        return PaperStrategyGateCheck(
            name="Strategy symbol allowlist",
            passed=True,
            detail="no per-strategy symbol allowlist",
            metadata={"symbols": list(symbols)},
        )
    missing = sorted(set(symbols) - set(allowed))
    return PaperStrategyGateCheck(
        name="Strategy symbol allowlist",
        passed=not missing,
        detail=(
            f"symbols={', '.join(symbols) or 'none'} allowed={', '.join(allowed)}"
        ),
        metadata={"missing": missing, "symbols": list(symbols), "allowed": list(allowed)},
    )


def _max_order_notional_check(
    record: dict[str, Any] | None,
    proposal: Any,
) -> PaperStrategyGateCheck:
    limit = _optional_float(record.get("max_order_notional")) if record else None
    notionals = tuple(
        _optional_float(getattr(intent, "estimated_notional", None))
        for intent in getattr(proposal, "intents", ()) or ()
    )
    if limit is None:
        return PaperStrategyGateCheck(
            name="Strategy max order notional",
            passed=True,
            detail="no per-strategy max order notional",
        )
    missing = any(value is None for value in notionals)
    too_large = [value for value in notionals if value is not None and value > limit]
    passed = not missing and not too_large
    max_seen = max((value or 0.0 for value in notionals), default=0.0)
    return PaperStrategyGateCheck(
        name="Strategy max order notional",
        passed=passed,
        detail=f"max_intent_notional={max_seen:,.2f} limit={limit:,.2f}",
        metadata={"missing_notional": missing, "too_large": too_large},
    )


def _paper_strategy_status(value: PaperStrategyStatus | str) -> PaperStrategyStatus:
    if isinstance(value, PaperStrategyStatus):
        return value
    try:
        return PaperStrategyStatus(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown paper strategy status: {value!r}") from exc


def _clean_symbols(symbols: tuple[str, ...] | list[str]) -> list[str]:
    return sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _datetime_text(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    return str(value)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
