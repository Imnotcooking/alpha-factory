"""Bridge options scanner candidates into paper trade proposal artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from oqp.config import load_settings
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType, utc_now
from oqp.execution.artifacts import trade_proposal_directory, write_trade_proposal_artifact
from oqp.execution.models import OrderIntent, TradeProposal


class OptionsProposalBridgeError(ValueError):
    """Raised when an options scanner row cannot become a proposal."""


@dataclass(frozen=True, slots=True)
class OptionsProposalResult:
    proposal: TradeProposal
    written_path: Path | None = None


def build_option_trade_proposal_from_candidate(
    candidate: Mapping[str, Any],
    *,
    underlying: str,
    contracts: float = 1.0,
    proposal_id: str | None = None,
    source: str = "options_desk",
    paper_only: bool = True,
    time_in_force: str = "DAY",
) -> TradeProposal:
    """Convert one options scanner candidate into a paper-only proposal.

    The generated proposal is still only a draft. Existing paper safety checks
    decide whether it is blocked or ready, and default policy blocks option
    asset classes unless explicitly enabled.
    """

    if contracts <= 0:
        raise OptionsProposalBridgeError("contracts must be positive")

    normalized_underlying = _required_text(underlying, "underlying").upper()
    strategy = _required_text(candidate.get("Strategy"), "Strategy")
    expiry = _required_text(candidate.get("Expiry"), "Expiry")
    legs = _legs(candidate)
    intents = tuple(
        _intent_from_leg(
            leg,
            underlying=normalized_underlying,
            default_expiry=expiry,
            strategy_id=_strategy_id(strategy),
            contracts=contracts,
            time_in_force=time_in_force,
            candidate=candidate,
        )
        for leg in legs
    )

    return TradeProposal(
        proposal_id=proposal_id or _default_option_proposal_id(normalized_underlying, strategy),
        source=source,
        intents=intents,
        paper_only=paper_only,
        strategy_id=_strategy_id(strategy),
        notes="Generated from Options Desk scanner candidate for paper-dashboard review.",
        metadata={
            "underlying": normalized_underlying,
            "strategy": strategy,
            "expiry": expiry,
            "structure": _optional_text(candidate.get("Structure")),
            "scanner_metrics": _scanner_metrics(candidate),
            "contracts_per_leg_unit": contracts,
            "safety_note": "Draft only. Review through paper-trading guardrails before any execution.",
        },
    )


def write_option_trade_proposal_from_candidate(
    candidate: Mapping[str, Any],
    *,
    underlying: str,
    contracts: float = 1.0,
    output_dir: Path | None = None,
    proposal_id: str | None = None,
    overwrite: bool = False,
) -> OptionsProposalResult:
    proposal = build_option_trade_proposal_from_candidate(
        candidate,
        underlying=underlying,
        contracts=contracts,
        proposal_id=proposal_id,
    )
    directory = output_dir or trade_proposal_directory(load_settings())
    path = write_trade_proposal_artifact(
        proposal,
        directory,
        overwrite=overwrite,
    )
    return OptionsProposalResult(proposal=proposal, written_path=path)


def _intent_from_leg(
    leg: Mapping[str, Any],
    *,
    underlying: str,
    default_expiry: str,
    strategy_id: str,
    contracts: float,
    time_in_force: str,
    candidate: Mapping[str, Any],
) -> OrderIntent:
    quantity_units = _positive_float(leg.get("quantity"), "leg.quantity")
    signed_quantity = float(leg["quantity"])
    premium = _positive_float(leg.get("premium"), "leg.premium")
    expiry = _optional_text(leg.get("expiry")) or default_expiry
    right = _option_type(leg.get("option_type"))
    strike = _positive_float(leg.get("strike"), "leg.strike")
    symbol = _option_symbol(underlying, expiry, right, strike)

    return OrderIntent(
        instrument=Instrument(
            symbol=symbol,
            asset_class=AssetClass.OPTION,
            exchange="SMART",
            currency="USD",
            broker_symbol=symbol,
            multiplier=100.0,
            metadata={
                "underlying": underlying,
                "expiry": expiry,
                "right": right,
                "strike": strike,
                "scanner_strategy": _optional_text(candidate.get("Strategy")),
            },
        ),
        side=OrderSide.BUY if signed_quantity > 0 else OrderSide.SELL,
        quantity=abs(quantity_units) * contracts,
        order_type=OrderType.LIMIT,
        limit_price=premium,
        reference_price=premium,
        time_in_force=time_in_force,
        strategy_id=strategy_id,
        signal_id=_optional_text(candidate.get("Structure")),
        confidence=_confidence(candidate),
        rationale=f"Options Desk candidate: {_optional_text(candidate.get('Strategy')) or 'option strategy'}",
        metadata={
            "leg_quantity": signed_quantity,
            "candidate_expiry": _optional_text(candidate.get("Expiry")),
            "candidate_structure": _optional_text(candidate.get("Structure")),
            "candidate_metrics": _scanner_metrics(candidate),
        },
    )


def _legs(candidate: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    value = candidate.get("Legs")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OptionsProposalBridgeError("candidate must include a Legs list")
    legs = tuple(item for item in value if isinstance(item, Mapping))
    if not legs:
        raise OptionsProposalBridgeError("candidate Legs list is empty")
    return legs


def _scanner_metrics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "Debit/Credit",
        "Max Profit",
        "Max Loss",
        "PoP",
        "EV",
        "VaR 95",
        "Edge",
        "Width",
        "Ratio",
        "Near Expiry",
        "Far Expiry",
    )
    return {key: _json_safe(candidate.get(key)) for key in keys if candidate.get(key) is not None}


def _confidence(candidate: Mapping[str, Any]) -> float | None:
    pop = candidate.get("PoP")
    try:
        parsed = float(pop)
    except (TypeError, ValueError):
        return None
    if 0 <= parsed <= 1:
        return parsed
    if 1 < parsed <= 100:
        return parsed / 100
    return None


def _option_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"call", "c"}:
        return "C"
    if normalized in {"put", "p"}:
        return "P"
    raise OptionsProposalBridgeError(f"unsupported option_type {value!r}")


def _option_symbol(underlying: str, expiry: str, right: str, strike: float) -> str:
    expiry_code = "".join(ch for ch in str(expiry) if ch.isdigit())[:8]
    if len(expiry_code) != 8:
        expiry_code = "UNKNOWN"
    strike_text = f"{strike:g}".replace(".", "_")
    return f"{underlying}_{expiry_code}_{right}{strike_text}"


def _strategy_id(strategy: str) -> str:
    safe = "".join(
        char.lower() if char.isalnum() else "_"
        for char in strategy.strip()
    ).strip("_")
    return f"options_{safe or 'strategy'}"


def _default_option_proposal_id(underlying: str, strategy: str) -> str:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"options-{underlying.lower()}-{_strategy_id(strategy).replace('_', '-')}-{stamp}"


def _required_text(value: Any, label: str) -> str:
    text = _optional_text(value)
    if not text:
        raise OptionsProposalBridgeError(f"{label} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OptionsProposalBridgeError(f"{label} must be numeric") from exc
    if parsed == 0:
        raise OptionsProposalBridgeError(f"{label} must be nonzero")
    if label != "leg.quantity" and parsed <= 0:
        raise OptionsProposalBridgeError(f"{label} must be positive")
    return parsed


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)
