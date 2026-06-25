"""Paper order submission preflight checks.

This module deliberately does not place broker orders. It reviews approved
dry-run paper tickets and records why they are still blocked from submission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from oqp.accounts import (
    AccountEnvironment,
    TradeEvent,
    write_account_trade_events,
)
from oqp.brokers import BrokerConnectionConfig, BrokerEnvironment
from oqp.config import OQPSettings
from oqp.domain import utc_now
from oqp.paper_trading.order_router import PaperOrderTicketStatus


class PaperSubmissionDecision(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class PaperSubmissionCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "block"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PaperSubmissionPreflight:
    order_id: str
    checked_at: datetime
    decision: PaperSubmissionDecision
    message: str
    checks: tuple[PaperSubmissionCheck, ...]
    symbol: str
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    strategy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.decision == PaperSubmissionDecision.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "checked_at": self.checked_at.isoformat(),
            "decision": self.decision.value,
            "message": self.message,
            "checks": [check.to_dict() for check in self.checks],
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PaperSubmissionPreflightRecordResult:
    db_path: Path
    event_id: str
    order_id: str
    decision: PaperSubmissionDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "event_id": self.event_id,
            "order_id": self.order_id,
            "decision": self.decision.value,
        }


def review_paper_order_submission(
    ticket: Mapping[str, Any],
    *,
    settings: OQPSettings,
    broker_config: BrokerConnectionConfig,
    checked_at: datetime | None = None,
) -> PaperSubmissionPreflight:
    timestamp = (checked_at or utc_now()).replace(microsecond=0)
    metadata = _metadata_dict(ticket.get("metadata") or ticket.get("metadata_json"))
    order_id = _text(ticket.get("order_id")) or "unknown-paper-ticket"
    status = _text(ticket.get("status")) or ""
    symbol = _text(ticket.get("symbol")) or "UNKNOWN"
    side = _text(ticket.get("side"))
    quantity = _optional_float(ticket.get("quantity"))
    price = _optional_float(ticket.get("limit_price"))
    strategy_id = _text(ticket.get("strategy_id"))

    checks = (
        _check(
            "Ticket human-approved",
            status == PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
            status or "missing",
        ),
        _check(
            "Paper submit switch",
            settings.allow_paper_order_submit,
            f"ALLOW_PAPER_ORDER_SUBMIT={str(settings.allow_paper_order_submit).lower()}",
        ),
        _check(
            "Live trading disabled",
            not settings.allow_live_trading,
            f"ALLOW_LIVE_TRADING={str(settings.allow_live_trading).lower()}",
        ),
        _check(
            "Broker environment paper",
            broker_config.environment == BrokerEnvironment.PAPER,
            broker_config.environment.value,
        ),
        _check(
            "Broker profile write-enabled",
            not broker_config.readonly,
            f"readonly={str(broker_config.readonly).lower()}",
        ),
        _check(
            "Broker placement implementation",
            False,
            "disabled in current skeleton; no IBKR order can be submitted here",
        ),
    )
    blockers = [
        check.name
        for check in checks
        if not check.passed and check.severity == "block"
    ]
    decision = (
        PaperSubmissionDecision.READY
        if not blockers
        else PaperSubmissionDecision.BLOCKED
    )
    message = (
        "Paper ticket is submit-ready, but no broker call is made by this preflight."
        if decision == PaperSubmissionDecision.READY
        else "Blocked by: " + ", ".join(blockers)
    )

    return PaperSubmissionPreflight(
        order_id=order_id,
        checked_at=timestamp,
        decision=decision,
        message=message,
        checks=checks,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        strategy_id=strategy_id,
        metadata={
            "proposal_id": metadata.get("proposal_id"),
            "review_id": metadata.get("review_id"),
            "approval_status": metadata.get("approval_status"),
            "broker_submit_enabled": False,
            "submitter_skeleton": True,
        },
    )


def record_paper_submission_preflight(
    account_ledger_path: str | Path,
    preflight: PaperSubmissionPreflight,
    *,
    broker_config: BrokerConnectionConfig,
    account_id: str | None = None,
) -> PaperSubmissionPreflightRecordResult:
    profile = str(broker_config.metadata.get("profile") or "ibkr_paper_readonly")
    event = TradeEvent(
        event_id=(
            f"evt-paper-submit-preflight-{preflight.decision.value}-"
            f"{_safe_id(preflight.order_id)}-"
            f"{preflight.checked_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        ),
        event_type=f"paper_order_submission_preflight_{preflight.decision.value}",
        occurred_at=preflight.checked_at,
        account_id=account_id,
        broker=broker_config.broker,
        profile=profile,
        environment=AccountEnvironment.PAPER,
        symbol=preflight.symbol,
        side=preflight.side,
        quantity=preflight.quantity,
        price=preflight.price,
        currency=str(preflight.metadata.get("currency") or "USD"),
        strategy_id=preflight.strategy_id,
        order_id=preflight.order_id,
        metadata={
            "decision": preflight.decision.value,
            "message": preflight.message,
            "checks": [check.to_dict() for check in preflight.checks],
            "broker_submit_enabled": False,
            "event_source": "paper_order_submission_preflight",
            "submitter_skeleton": True,
            **preflight.metadata,
        },
    )
    result = write_account_trade_events(account_ledger_path, (event,))
    return PaperSubmissionPreflightRecordResult(
        db_path=result.db_path,
        event_id=result.event_ids[0],
        order_id=preflight.order_id,
        decision=preflight.decision,
    )


def _check(name: str, passed: bool, detail: str) -> PaperSubmissionCheck:
    return PaperSubmissionCheck(name=name, passed=passed, detail=detail)


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_id(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value)
    ).strip("-")
    return safe or "ticket"
