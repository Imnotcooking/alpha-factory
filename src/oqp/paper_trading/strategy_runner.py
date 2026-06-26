"""Automated dry-run runner for paper-running strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oqp.accounts import (
    account_trade_events_from_proposal_review,
    write_account_trade_events,
)
from oqp.brokers import BrokerConnectionConfig, get_broker_profile_config
from oqp.config import OQPSettings
from oqp.execution import (
    LoadedTradeProposal,
    TradeProposal,
    load_trade_proposal_artifacts,
    parse_trade_proposal,
)
from oqp.paper_trading.execution_safety import PaperExecutionReview
from oqp.paper_trading.ledger import (
    PaperExecutionReviewWriteResult,
    load_latest_paper_execution_reviews,
    paper_order_notional_today,
    write_paper_execution_review,
)
from oqp.paper_trading.execution_safety import review_paper_execution_proposal
from oqp.paper_trading.order_router import (
    PaperOrderTicketResult,
    create_dry_run_order_tickets,
)
from oqp.paper_trading.strategy_registry import (
    PaperStrategyGateResult,
    review_paper_strategy_gate,
)


@dataclass(frozen=True, slots=True)
class PaperStrategyRunnerItem:
    proposal_id: str
    artifact_path: Path | None
    action: str
    message: str
    strategy_gate: PaperStrategyGateResult | None = None
    review: PaperExecutionReview | None = None
    review_write: PaperExecutionReviewWriteResult | None = None
    account_event_count: int = 0
    ticket_result: PaperOrderTicketResult | None = None

    @property
    def ticket_count(self) -> int:
        return 0 if self.ticket_result is None else self.ticket_result.order_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "artifact_path": None if self.artifact_path is None else str(self.artifact_path),
            "action": self.action,
            "message": self.message,
            "strategy_gate": None
            if self.strategy_gate is None
            else self.strategy_gate.to_dict(),
            "review": None if self.review is None else self.review.to_dict(),
            "review_write": None
            if self.review_write is None
            else self.review_write.to_dict(),
            "account_event_count": self.account_event_count,
            "ticket_result": None
            if self.ticket_result is None
            else self.ticket_result.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PaperStrategyRunnerResult:
    proposal_path: Path
    loaded_count: int
    reviewed_count: int
    skipped_count: int
    ticket_count: int
    items: tuple[PaperStrategyRunnerItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_path": str(self.proposal_path),
            "loaded_count": self.loaded_count,
            "reviewed_count": self.reviewed_count,
            "skipped_count": self.skipped_count,
            "ticket_count": self.ticket_count,
            "items": [item.to_dict() for item in self.items],
        }


def run_paper_strategy_runner(
    proposal_path: str | Path,
    *,
    settings: OQPSettings,
    paper_ledger_path: str | Path,
    account_ledger_path: str | Path,
    broker_config: BrokerConnectionConfig | None = None,
    account_id: str | None = None,
    max_files: int = 50,
    skip_reviewed: bool = True,
) -> PaperStrategyRunnerResult:
    path = Path(proposal_path)
    paper_path = Path(paper_ledger_path)
    account_path = Path(account_ledger_path)
    active_broker_config = broker_config or get_broker_profile_config(
        "ibkr_paper_readonly",
        settings=settings,
    )
    loaded = _load_proposals(path, max_files=max_files)
    reviewed_ids = _reviewed_proposal_ids(paper_path) if skip_reviewed else set()
    items: list[PaperStrategyRunnerItem] = []
    daily_used = paper_order_notional_today(paper_path)

    for loaded_proposal in loaded:
        proposal = loaded_proposal.proposal
        if proposal.proposal_id in reviewed_ids:
            items.append(
                PaperStrategyRunnerItem(
                    proposal_id=proposal.proposal_id,
                    artifact_path=loaded_proposal.path,
                    action="skipped",
                    message="Proposal already has a recorded paper safety review.",
                )
            )
            continue

        strategy_gate = review_paper_strategy_gate(paper_path, proposal)
        if not strategy_gate.passed:
            items.append(
                PaperStrategyRunnerItem(
                    proposal_id=proposal.proposal_id,
                    artifact_path=loaded_proposal.path,
                    action="skipped",
                    message=strategy_gate.message,
                    strategy_gate=strategy_gate,
                )
            )
            continue

        review = review_paper_execution_proposal(
            proposal,
            settings=settings,
            broker_config=active_broker_config,
            daily_notional_used=daily_used,
        )
        review_write = write_paper_execution_review(
            paper_path,
            proposal_id=proposal.proposal_id,
            decision=review.decision.value,
            checks=[check.to_dict() for check in review.checks],
            estimated_notional=review.estimated_notional,
            order_count=review.order_count,
            message=review.message,
            reviewed_at=review.reviewed_at,
        )
        account_events = account_trade_events_from_proposal_review(
            proposal,
            decision=review.decision.value,
            reviewed_at=review.reviewed_at,
            environment="paper",
            profile=str(
                active_broker_config.metadata.get("profile") or "ibkr_paper_readonly"
            ),
            broker=active_broker_config.broker,
            account_id=account_id,
            review_id=review_write.review_id,
            message=review.message,
        )
        account_write = write_account_trade_events(account_path, account_events)
        ticket_result = create_dry_run_order_tickets(
            proposal,
            review=review,
            paper_ledger_path=paper_path,
            account_ledger_path=account_path,
            broker_config=active_broker_config,
            account_id=account_id,
            review_id=review_write.review_id,
            created_at=review.reviewed_at,
        )
        items.append(
            PaperStrategyRunnerItem(
                proposal_id=proposal.proposal_id,
                artifact_path=loaded_proposal.path,
                action="reviewed",
                message=ticket_result.message,
                strategy_gate=strategy_gate,
                review=review,
                review_write=review_write,
                account_event_count=account_write.event_count,
                ticket_result=ticket_result,
            )
        )
        if ticket_result.order_count:
            daily_used += review.estimated_notional or 0.0

    reviewed_count = sum(1 for item in items if item.action == "reviewed")
    skipped_count = sum(1 for item in items if item.action == "skipped")
    ticket_count = sum(item.ticket_count for item in items)
    return PaperStrategyRunnerResult(
        proposal_path=path,
        loaded_count=len(loaded),
        reviewed_count=reviewed_count,
        skipped_count=skipped_count,
        ticket_count=ticket_count,
        items=tuple(items),
    )


def _load_proposals(path: Path, *, max_files: int) -> tuple[LoadedTradeProposal, ...]:
    if path.is_file():
        proposal = parse_trade_proposal(json.loads(path.read_text(encoding="utf-8")))
        return (LoadedTradeProposal(proposal=proposal, path=path),)
    result = load_trade_proposal_artifacts(path, max_files=max_files)
    if result.issues:
        issue_text = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        raise ValueError(f"Proposal artifact issues: {issue_text}")
    return result.loaded


def _reviewed_proposal_ids(paper_ledger_path: Path) -> set[str]:
    reviews = load_latest_paper_execution_reviews(paper_ledger_path, limit=500)
    if reviews.empty or "proposal_id" not in reviews.columns:
        return set()
    return set(reviews["proposal_id"].dropna().astype(str))
