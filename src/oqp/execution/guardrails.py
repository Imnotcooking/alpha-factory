"""Safety checks for draft execution proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from oqp.brokers import BrokerConnectionConfig, BrokerEnvironment
from oqp.brokers.models import AccountSummary
from oqp.config import OQPSettings
from oqp.domain import utc_now
from oqp.execution.models import TradeProposal


class GuardrailSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class GuardrailCheck:
    name: str
    passed: bool
    severity: GuardrailSeverity
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GuardrailReport:
    proposal_id: str
    checks: tuple[GuardrailCheck, ...]
    checked_at: datetime = field(default_factory=utc_now)

    @property
    def hard_blocked(self) -> bool:
        return any(
            not check.passed and check.severity == GuardrailSeverity.BLOCK
            for check in self.checks
        )

    @property
    def passed(self) -> bool:
        return not self.hard_blocked


def evaluate_trade_proposal(
    proposal: TradeProposal,
    *,
    settings: OQPSettings,
    broker_config: BrokerConnectionConfig,
    account_summary: AccountSummary | None = None,
    broker_connected: bool = False,
    order_placement_enabled: bool = False,
) -> GuardrailReport:
    """Evaluate a proposal against current paper-execution constraints."""

    checks: list[GuardrailCheck] = []

    checks.append(
        GuardrailCheck(
            name="Paper broker profile",
            passed=broker_config.environment == BrokerEnvironment.PAPER,
            severity=GuardrailSeverity.BLOCK,
            detail=broker_config.environment.value,
        )
    )
    checks.append(
        GuardrailCheck(
            name="Broker read-only",
            passed=broker_config.readonly,
            severity=GuardrailSeverity.BLOCK,
            detail="readonly=" + str(broker_config.readonly).lower(),
        )
    )
    checks.append(
        GuardrailCheck(
            name="Live trading disabled",
            passed=not settings.allow_live_trading,
            severity=GuardrailSeverity.BLOCK,
            detail="ALLOW_LIVE_TRADING=" + str(settings.allow_live_trading).lower(),
        )
    )
    checks.append(
        GuardrailCheck(
            name="Paper-only proposal",
            passed=proposal.paper_only,
            severity=GuardrailSeverity.BLOCK,
            detail="paper_only=" + str(proposal.paper_only).lower(),
        )
    )
    checks.append(
        GuardrailCheck(
            name="Broker connected",
            passed=broker_connected,
            severity=GuardrailSeverity.WARNING,
            detail="connected=" + str(broker_connected).lower(),
        )
    )
    checks.append(
        GuardrailCheck(
            name="Order placement switch",
            passed=order_placement_enabled,
            severity=GuardrailSeverity.BLOCK,
            detail=(
                "enabled"
                if order_placement_enabled
                else "disabled until paper order router is implemented"
            ),
        )
    )

    estimated_notional = proposal.estimated_notional
    if settings.max_gross_exposure is not None:
        max_exposure = settings.max_gross_exposure
        passed = estimated_notional is not None and estimated_notional <= max_exposure
        detail = (
            f"estimated_notional={estimated_notional:,.2f}, max={max_exposure:,.2f}"
            if estimated_notional is not None
            else "estimated_notional unavailable"
        )
        checks.append(
            GuardrailCheck(
                name="Max gross exposure",
                passed=passed,
                severity=GuardrailSeverity.BLOCK,
                detail=detail,
            )
        )
    else:
        checks.append(
            GuardrailCheck(
                name="Max gross exposure",
                passed=True,
                severity=GuardrailSeverity.INFO,
                detail="not configured",
            )
        )

    if settings.max_daily_loss_pct is not None:
        nav = account_summary.net_liquidation if account_summary else None
        checks.append(
            GuardrailCheck(
                name="Max daily loss",
                passed=nav is not None,
                severity=GuardrailSeverity.WARNING,
                detail=(
                    f"limit={settings.max_daily_loss_pct:.2%}, nav={nav:,.2f}"
                    if nav is not None
                    else "account snapshot unavailable"
                ),
            )
        )
    else:
        checks.append(
            GuardrailCheck(
                name="Max daily loss",
                passed=True,
                severity=GuardrailSeverity.INFO,
                detail="not configured",
            )
        )

    return GuardrailReport(proposal_id=proposal.proposal_id, checks=tuple(checks))
