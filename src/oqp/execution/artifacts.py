"""JSON artifacts for draft trade proposals."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from oqp.config import OQPSettings
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType
from oqp.execution.models import OrderIntent, ProposalStatus, TradeProposal


TRADE_PROPOSAL_DIRNAME = "trade_proposals"


class ProposalArtifactError(ValueError):
    """Raised when a trade proposal artifact cannot be parsed."""


@dataclass(frozen=True, slots=True)
class LoadedTradeProposal:
    proposal: TradeProposal
    path: Path


@dataclass(frozen=True, slots=True)
class ProposalArtifactIssue:
    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class ProposalLoadResult:
    directory: Path
    loaded: tuple[LoadedTradeProposal, ...] = ()
    issues: tuple[ProposalArtifactIssue, ...] = ()


def trade_proposal_directory(settings: OQPSettings) -> Path:
    return settings.artifact_root / TRADE_PROPOSAL_DIRNAME


def load_trade_proposal_artifacts(
    directory: Path,
    *,
    max_files: int | None = None,
) -> ProposalLoadResult:
    """Load all JSON proposal artifacts in newest-file-first order."""

    if not directory.exists():
        return ProposalLoadResult(directory=directory)
    if not directory.is_dir():
        return ProposalLoadResult(
            directory=directory,
            issues=(
                ProposalArtifactIssue(
                    path=directory,
                    message="proposal artifact path is not a directory",
                ),
            ),
        )

    paths = sorted(
        directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if max_files is not None:
        paths = paths[:max_files]

    loaded: list[LoadedTradeProposal] = []
    issues: list[ProposalArtifactIssue] = []

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for proposal_payload in _iter_proposal_payloads(payload):
                proposal = parse_trade_proposal(proposal_payload)
                loaded.append(LoadedTradeProposal(proposal=proposal, path=path))
        except Exception as exc:
            issues.append(ProposalArtifactIssue(path=path, message=str(exc)))

    return ProposalLoadResult(
        directory=directory,
        loaded=tuple(loaded),
        issues=tuple(issues),
    )


def parse_trade_proposal(payload: Mapping[str, Any]) -> TradeProposal:
    _require_mapping(payload, "proposal")
    intents = tuple(parse_order_intent(item) for item in _list(payload, "intents"))

    kwargs: dict[str, Any] = {
        "proposal_id": _required_str(payload, "proposal_id"),
        "source": _required_str(payload, "source"),
        "intents": intents,
        "status": _enum(
            ProposalStatus,
            payload.get("status", ProposalStatus.DRAFT.value),
            "status",
        ),
        "paper_only": _bool(payload.get("paper_only", True), "paper_only"),
        "strategy_id": _optional_str(payload.get("strategy_id")),
        "research_run_id": _optional_str(payload.get("research_run_id")),
        "notes": _optional_str(payload.get("notes")),
        "metadata": _dict(payload.get("metadata"), "metadata"),
    }

    if payload.get("created_at") is not None:
        kwargs["created_at"] = _datetime(payload["created_at"], "created_at")

    return TradeProposal(**kwargs)


def parse_order_intent(payload: Mapping[str, Any]) -> OrderIntent:
    _require_mapping(payload, "intent")

    return OrderIntent(
        instrument=parse_instrument(payload),
        side=_enum(OrderSide, _required_str(payload, "side"), "side"),
        quantity=_positive_float(payload.get("quantity"), "quantity"),
        order_type=_enum(
            OrderType,
            payload.get("order_type", OrderType.MARKET.value),
            "order_type",
        ),
        limit_price=_optional_positive_float(payload.get("limit_price"), "limit_price"),
        stop_price=_optional_positive_float(payload.get("stop_price"), "stop_price"),
        time_in_force=_optional_str(payload.get("time_in_force")) or "DAY",
        strategy_id=_optional_str(payload.get("strategy_id")),
        signal_id=_optional_str(payload.get("signal_id")),
        target_weight=_optional_float(payload.get("target_weight"), "target_weight"),
        reference_price=_optional_positive_float(
            payload.get("reference_price"),
            "reference_price",
        ),
        confidence=_optional_float(payload.get("confidence"), "confidence"),
        rationale=_optional_str(payload.get("rationale")),
        metadata=_dict(payload.get("metadata"), "metadata"),
    )


def parse_instrument(payload: Mapping[str, Any]) -> Instrument:
    instrument_payload = payload.get("instrument")
    if instrument_payload is not None:
        _require_mapping(instrument_payload, "instrument")
        source = instrument_payload
    else:
        source = payload

    return Instrument(
        symbol=_required_str(source, "symbol").upper(),
        asset_class=_enum(AssetClass, _required_str(source, "asset_class"), "asset_class"),
        exchange=_optional_str(source.get("exchange")),
        currency=(_optional_str(source.get("currency")) or "USD").upper(),
        broker_symbol=_optional_str(source.get("broker_symbol")),
        multiplier=_optional_positive_float(source.get("multiplier"), "multiplier") or 1.0,
        metadata=_dict(source.get("metadata"), "metadata"),
    )


def trade_proposal_to_dict(proposal: TradeProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "source": proposal.source,
        "status": proposal.status.value,
        "paper_only": proposal.paper_only,
        "created_at": proposal.created_at.isoformat(),
        "strategy_id": proposal.strategy_id,
        "research_run_id": proposal.research_run_id,
        "notes": proposal.notes,
        "metadata": proposal.metadata,
        "intents": [order_intent_to_dict(intent) for intent in proposal.intents],
    }


def order_intent_to_dict(intent: OrderIntent) -> dict[str, Any]:
    return {
        "instrument": instrument_to_dict(intent.instrument),
        "side": intent.side.value,
        "quantity": intent.quantity,
        "order_type": intent.order_type.value,
        "limit_price": intent.limit_price,
        "stop_price": intent.stop_price,
        "time_in_force": intent.time_in_force,
        "strategy_id": intent.strategy_id,
        "signal_id": intent.signal_id,
        "target_weight": intent.target_weight,
        "reference_price": intent.reference_price,
        "confidence": intent.confidence,
        "rationale": intent.rationale,
        "metadata": intent.metadata,
    }


def instrument_to_dict(instrument: Instrument) -> dict[str, Any]:
    return {
        "symbol": instrument.symbol,
        "asset_class": instrument.asset_class.value,
        "exchange": instrument.exchange,
        "currency": instrument.currency,
        "broker_symbol": instrument.broker_symbol,
        "multiplier": instrument.multiplier,
        "metadata": instrument.metadata,
    }


def write_trade_proposal_artifact(
    proposal: TradeProposal,
    directory: Path,
    *,
    filename: str | None = None,
    overwrite: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in proposal.proposal_id
    ).strip("-")
    path = directory / (filename or f"{safe_id or 'trade-proposal'}.json")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.write_text(
        json.dumps(trade_proposal_to_dict(proposal), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _iter_proposal_payloads(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("proposals"), list):
        for item in payload["proposals"]:
            _require_mapping(item, "proposal")
            yield item
        return
    _require_mapping(payload, "proposal")
    yield payload


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ProposalArtifactError(f"{label} must be an object")


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProposalArtifactError(f"{key} is required")
    return value.strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    value = value.strip()
    return value or None


def _enum(enum_type: type[Any], value: Any, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    normalized = str(value).strip().lower().replace(" ", "_")
    try:
        return enum_type(normalized)
    except ValueError as exc:
        raise ProposalArtifactError(
            f"{field_name} has unsupported value {value!r}"
        ) from exc


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _optional_positive_float(value, field_name)
    if parsed is None:
        raise ProposalArtifactError(f"{field_name} is required")
    return parsed


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    parsed = _optional_float(value, field_name)
    if parsed is not None and parsed <= 0:
        raise ProposalArtifactError(f"{field_name} must be positive")
    return parsed


def _optional_float(value: Any, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProposalArtifactError(f"{field_name} must be numeric") from exc


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ProposalArtifactError(f"{field_name} must be boolean")


def _datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ProposalArtifactError(f"{field_name} must be an ISO datetime string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProposalArtifactError(f"{field_name} must be an ISO datetime string") from exc


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProposalArtifactError(f"{field_name} must be an object")
    return dict(value)


def _list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ProposalArtifactError(f"{key} must be a list")
    return value
