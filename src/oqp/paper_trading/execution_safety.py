"""Paper execution safety policy and proposal review."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from oqp.brokers import BrokerConnectionConfig, BrokerEnvironment
from oqp.config import OQPSettings
from oqp.domain import AssetClass, OrderType, utc_now
from oqp.execution import TradeProposal


class PaperExecutionDecisionStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


class PaperExecutionSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class PaperOptionsPolicy:
    enabled: bool = False
    allowed_underlyings: tuple[str, ...] = ()
    allowed_strategies: tuple[str, ...] = ()
    max_contracts: float | None = 1.0
    max_premium: float | None = 500.0
    max_defined_risk: float | None = 1_000.0
    max_spread_width: float | None = 10.0

    @classmethod
    def from_settings(cls, settings: OQPSettings) -> "PaperOptionsPolicy":
        return cls(
            enabled=settings.paper_options_enabled,
            allowed_underlyings=tuple(
                symbol.upper() for symbol in settings.paper_option_allowed_underlyings
            ),
            allowed_strategies=tuple(
                _normalize_policy_text(strategy)
                for strategy in settings.paper_option_allowed_strategies
            ),
            max_contracts=settings.paper_option_max_contracts,
            max_premium=settings.paper_option_max_premium,
            max_defined_risk=settings.paper_option_max_defined_risk,
            max_spread_width=settings.paper_option_max_spread_width,
        )


@dataclass(frozen=True, slots=True)
class PaperExecutionPolicy:
    allow_paper_trading: bool = False
    max_order_notional: float | None = 10_000.0
    max_daily_notional: float | None = 50_000.0
    allowed_symbols: tuple[str, ...] = ()
    allowed_asset_classes: tuple[str, ...] = ("equity", "etf")
    allow_market_orders: bool = False
    require_reference_price: bool = True
    options: PaperOptionsPolicy = field(default_factory=PaperOptionsPolicy)

    @classmethod
    def from_settings(cls, settings: OQPSettings) -> "PaperExecutionPolicy":
        return cls(
            allow_paper_trading=settings.allow_paper_trading,
            max_order_notional=settings.paper_max_order_notional,
            max_daily_notional=settings.paper_max_daily_notional,
            allowed_symbols=tuple(symbol.upper() for symbol in settings.paper_allowed_symbols),
            allowed_asset_classes=tuple(
                asset_class.lower()
                for asset_class in settings.paper_allowed_asset_classes
            ),
            options=PaperOptionsPolicy.from_settings(settings),
        )


@dataclass(frozen=True, slots=True)
class PaperExecutionCheck:
    name: str
    passed: bool
    severity: PaperExecutionSeverity
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity.value,
            "detail": self.detail,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PaperExecutionReview:
    proposal_id: str
    decision: PaperExecutionDecisionStatus
    checks: tuple[PaperExecutionCheck, ...]
    estimated_notional: float | None
    order_count: int
    reviewed_at: datetime = field(default_factory=utc_now)

    @property
    def passed(self) -> bool:
        return self.decision == PaperExecutionDecisionStatus.READY

    @property
    def message(self) -> str:
        if self.passed:
            return "Paper proposal passed safety review."
        blockers = [
            check.name
            for check in self.checks
            if not check.passed and check.severity == PaperExecutionSeverity.BLOCK
        ]
        return "Blocked by: " + ", ".join(blockers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "decision": self.decision.value,
            "estimated_notional": self.estimated_notional,
            "order_count": self.order_count,
            "reviewed_at": self.reviewed_at.isoformat(),
            "message": self.message,
            "checks": [check.to_dict() for check in self.checks],
        }


def review_paper_execution_proposal(
    proposal: TradeProposal,
    *,
    settings: OQPSettings,
    broker_config: BrokerConnectionConfig,
    policy: PaperExecutionPolicy | None = None,
    daily_notional_used: float = 0.0,
) -> PaperExecutionReview:
    active_policy = policy or PaperExecutionPolicy.from_settings(settings)
    checks: list[PaperExecutionCheck] = []

    checks.extend(
        [
            PaperExecutionCheck(
                name="Paper trading switch",
                passed=active_policy.allow_paper_trading,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    "ALLOW_PAPER_TRADING=true"
                    if active_policy.allow_paper_trading
                    else "ALLOW_PAPER_TRADING=false"
                ),
            ),
            PaperExecutionCheck(
                name="Live trading disabled",
                passed=not settings.allow_live_trading,
                severity=PaperExecutionSeverity.BLOCK,
                detail=f"ALLOW_LIVE_TRADING={str(settings.allow_live_trading).lower()}",
            ),
            PaperExecutionCheck(
                name="Paper broker profile",
                passed=broker_config.environment == BrokerEnvironment.PAPER,
                severity=PaperExecutionSeverity.BLOCK,
                detail=broker_config.environment.value,
            ),
            PaperExecutionCheck(
                name="Proposal paper-only",
                passed=proposal.paper_only,
                severity=PaperExecutionSeverity.BLOCK,
                detail=f"paper_only={str(proposal.paper_only).lower()}",
            ),
            PaperExecutionCheck(
                name="Has order intents",
                passed=len(proposal.intents) > 0,
                severity=PaperExecutionSeverity.BLOCK,
                detail=f"intents={len(proposal.intents)}",
            ),
        ]
    )

    estimated_notional = proposal.estimated_notional
    checks.append(
        PaperExecutionCheck(
            name="Proposal notional known",
            passed=estimated_notional is not None,
            severity=PaperExecutionSeverity.BLOCK,
            detail=(
                f"estimated_notional={estimated_notional:,.2f}"
                if estimated_notional is not None
                else "at least one intent lacks a reference/limit/stop price"
            ),
        )
    )

    option_intents = []
    for index, intent in enumerate(proposal.intents, start=1):
        symbol = intent.instrument.symbol.upper()
        asset_class = _asset_class_value(intent.instrument.asset_class)
        notional = intent.estimated_notional
        if asset_class == AssetClass.OPTION.value:
            option_intents.append((index, intent))

        if active_policy.allowed_symbols:
            symbol_allowed = symbol in active_policy.allowed_symbols
            underlying = _option_underlying(intent, proposal)
            if asset_class == AssetClass.OPTION.value and underlying:
                symbol_allowed = symbol_allowed or underlying in active_policy.allowed_symbols
            checks.append(
                PaperExecutionCheck(
                    name=f"Intent {index} symbol allowlist",
                    passed=symbol_allowed,
                    severity=PaperExecutionSeverity.BLOCK,
                    detail=f"{symbol} in {', '.join(active_policy.allowed_symbols)}",
                    metadata={"symbol": symbol, "underlying": underlying},
                )
            )

        asset_class_allowed = asset_class in active_policy.allowed_asset_classes
        if asset_class == AssetClass.OPTION.value and active_policy.options.enabled:
            asset_class_allowed = True
        checks.append(
            PaperExecutionCheck(
                name=f"Intent {index} asset class",
                passed=asset_class_allowed,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    f"{asset_class} allowed={', '.join(active_policy.allowed_asset_classes)}"
                    if asset_class != AssetClass.OPTION.value
                    else (
                        f"{asset_class} allowed by option policy"
                        if active_policy.options.enabled
                        else f"{asset_class} allowed={', '.join(active_policy.allowed_asset_classes)}"
                    )
                ),
                metadata={"symbol": symbol, "asset_class": asset_class},
            )
        )
        checks.append(
            PaperExecutionCheck(
                name=f"Intent {index} reference price",
                passed=(not active_policy.require_reference_price) or notional is not None,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    f"notional={notional:,.2f}"
                    if notional is not None
                    else "missing reference/limit/stop price"
                ),
                metadata={"symbol": symbol},
            )
        )
        checks.append(
            PaperExecutionCheck(
                name=f"Intent {index} market order",
                passed=active_policy.allow_market_orders
                or intent.order_type != OrderType.MARKET,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    "market orders allowed"
                    if active_policy.allow_market_orders
                    else f"order_type={intent.order_type.value}"
                ),
                metadata={"symbol": symbol, "order_type": intent.order_type.value},
            )
        )
        if active_policy.max_order_notional is not None:
            checks.append(
                PaperExecutionCheck(
                    name=f"Intent {index} max notional",
                    passed=notional is not None and notional <= active_policy.max_order_notional,
                    severity=PaperExecutionSeverity.BLOCK,
                    detail=(
                        f"notional={notional:,.2f} max={active_policy.max_order_notional:,.2f}"
                        if notional is not None
                        else f"notional unknown max={active_policy.max_order_notional:,.2f}"
                    ),
                    metadata={"symbol": symbol},
                )
            )

    if option_intents:
        checks.extend(
            _option_policy_checks(
                proposal,
                option_intents=tuple(option_intents),
                option_policy=active_policy.options,
            )
        )

    if active_policy.max_daily_notional is not None:
        proposed = estimated_notional or 0.0
        total = daily_notional_used + proposed
        checks.append(
            PaperExecutionCheck(
                name="Daily notional cap",
                passed=estimated_notional is not None and total <= active_policy.max_daily_notional,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    f"used={daily_notional_used:,.2f} proposed={proposed:,.2f} "
                    f"max={active_policy.max_daily_notional:,.2f}"
                ),
            )
        )

    blocked = any(
        not check.passed and check.severity == PaperExecutionSeverity.BLOCK
        for check in checks
    )
    return PaperExecutionReview(
        proposal_id=proposal.proposal_id,
        decision=(
            PaperExecutionDecisionStatus.BLOCKED
            if blocked
            else PaperExecutionDecisionStatus.READY
        ),
        checks=tuple(checks),
        estimated_notional=estimated_notional,
        order_count=len(proposal.intents),
    )


def _asset_class_value(asset_class: AssetClass | str) -> str:
    if isinstance(asset_class, AssetClass):
        return asset_class.value
    return str(asset_class).lower()


def _option_policy_checks(
    proposal: TradeProposal,
    *,
    option_intents: tuple[tuple[int, Any], ...],
    option_policy: PaperOptionsPolicy,
) -> list[PaperExecutionCheck]:
    checks: list[PaperExecutionCheck] = [
        PaperExecutionCheck(
            name="Options enabled",
            passed=option_policy.enabled,
            severity=PaperExecutionSeverity.BLOCK,
            detail=(
                "PAPER_OPTIONS_ENABLED=true"
                if option_policy.enabled
                else "PAPER_OPTIONS_ENABLED=false"
            ),
        )
    ]

    underlying = _proposal_option_underlying(proposal, option_intents)
    underlying_allowed = (
        bool(option_policy.allowed_underlyings)
        and underlying is not None
        and underlying in option_policy.allowed_underlyings
    )
    checks.append(
        PaperExecutionCheck(
            name="Option underlying allowlist",
            passed=underlying_allowed,
            severity=PaperExecutionSeverity.BLOCK,
            detail=(
                f"{underlying or 'unknown'} in {', '.join(option_policy.allowed_underlyings)}"
                if option_policy.allowed_underlyings
                else "no option underlyings configured"
            ),
            metadata={"underlying": underlying},
        )
    )

    strategy_aliases = _proposal_strategy_aliases(proposal)
    strategy_allowed = bool(option_policy.allowed_strategies) and bool(
        strategy_aliases.intersection(option_policy.allowed_strategies)
    )
    checks.append(
        PaperExecutionCheck(
            name="Option strategy allowlist",
            passed=strategy_allowed,
            severity=PaperExecutionSeverity.BLOCK,
            detail=(
                f"{', '.join(sorted(strategy_aliases)) or 'unknown'} in "
                f"{', '.join(option_policy.allowed_strategies)}"
                if option_policy.allowed_strategies
                else "no option strategies configured"
            ),
            metadata={"strategy_aliases": sorted(strategy_aliases)},
        )
    )

    if option_policy.max_contracts is not None:
        for index, intent in option_intents:
            checks.append(
                PaperExecutionCheck(
                    name=f"Option intent {index} max contracts",
                    passed=intent.quantity <= option_policy.max_contracts,
                    severity=PaperExecutionSeverity.BLOCK,
                    detail=(
                        f"contracts={intent.quantity:g} "
                        f"max={option_policy.max_contracts:g}"
                    ),
                    metadata={"symbol": intent.instrument.symbol},
                )
            )

    premium = _option_premium_at_risk(proposal, option_intents)
    if option_policy.max_premium is not None:
        checks.append(
            PaperExecutionCheck(
                name="Option max premium",
                passed=premium is not None and premium <= option_policy.max_premium,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    f"premium={premium:,.2f} max={option_policy.max_premium:,.2f}"
                    if premium is not None
                    else f"premium unknown max={option_policy.max_premium:,.2f}"
                ),
            )
        )

    defined_risk = _option_defined_risk(proposal)
    if option_policy.max_defined_risk is not None:
        checks.append(
            PaperExecutionCheck(
                name="Option max defined risk",
                passed=defined_risk is not None and defined_risk <= option_policy.max_defined_risk,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    f"defined_risk={defined_risk:,.2f} "
                    f"max={option_policy.max_defined_risk:,.2f}"
                    if defined_risk is not None
                    else f"defined risk unknown max={option_policy.max_defined_risk:,.2f}"
                ),
            )
        )

    width = _option_spread_width(proposal)
    if option_policy.max_spread_width is not None:
        checks.append(
            PaperExecutionCheck(
                name="Option max spread width",
                passed=width is None or width <= option_policy.max_spread_width,
                severity=PaperExecutionSeverity.BLOCK,
                detail=(
                    "not applicable"
                    if width is None
                    else f"width={width:,.2f} max={option_policy.max_spread_width:,.2f}"
                ),
            )
        )

    return checks


def _option_underlying(intent: Any, proposal: TradeProposal) -> str | None:
    value = intent.instrument.metadata.get("underlying")
    if not value:
        value = proposal.metadata.get("underlying")
    return str(value).upper() if value else None


def _proposal_option_underlying(
    proposal: TradeProposal,
    option_intents: tuple[tuple[int, Any], ...],
) -> str | None:
    value = proposal.metadata.get("underlying")
    if value:
        return str(value).upper()
    for _, intent in option_intents:
        underlying = _option_underlying(intent, proposal)
        if underlying:
            return underlying
    return None


def _proposal_strategy_aliases(proposal: TradeProposal) -> set[str]:
    values = {
        proposal.strategy_id,
        proposal.metadata.get("strategy"),
    }
    aliases: set[str] = set()
    for value in values:
        normalized = _normalize_policy_text(value)
        if not normalized:
            continue
        aliases.add(normalized)
        if normalized.startswith("options_"):
            aliases.add(normalized.removeprefix("options_"))
        else:
            aliases.add(f"options_{normalized}")
    return aliases


def _normalize_policy_text(value: Any) -> str:
    if value is None:
        return ""
    return "_".join(
        chunk
        for chunk in "".join(
            char.lower() if char.isalnum() else " "
            for char in str(value).strip()
        ).split()
        if chunk
    )


def _scanner_metric(proposal: TradeProposal, key: str) -> float | None:
    metrics = proposal.metadata.get("scanner_metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _option_premium_at_risk(
    proposal: TradeProposal,
    option_intents: tuple[tuple[int, Any], ...],
) -> float | None:
    debit_credit = _scanner_metric(proposal, "Debit/Credit")
    if debit_credit is not None:
        return max(-debit_credit, 0.0)

    buy_notionals = [
        intent.estimated_notional
        for _, intent in option_intents
        if str(intent.side.value).lower() == "buy"
    ]
    if any(value is None for value in buy_notionals):
        return None
    return sum(float(value) for value in buy_notionals)


def _option_defined_risk(proposal: TradeProposal) -> float | None:
    max_loss = _scanner_metric(proposal, "Max Loss")
    return abs(max_loss) if max_loss is not None else None


def _option_spread_width(proposal: TradeProposal) -> float | None:
    width = _scanner_metric(proposal, "Width")
    return abs(width) if width is not None else None
