"""Bridge research signal artifacts into paper trade proposals."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from oqp.config import load_settings
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType, Signal, utc_now
from oqp.execution.artifacts import trade_proposal_directory, write_trade_proposal_artifact
from oqp.execution.models import OrderIntent, TradeProposal


SIGNAL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "asset", "instrument"),
    "asset_class": ("asset_class", "asset_type", "type"),
    "direction": ("direction", "signal", "side", "trade_direction", "position"),
    "strength": ("strength", "score", "confidence", "signal_strength", "alpha_score"),
    "strategy_id": ("strategy_id", "strategy", "model", "factor_id", "factor"),
    "research_run_id": ("research_run_id", "run_id", "experiment_id"),
    "signal_id": ("signal_id", "id", "artifact_id"),
    "target_weight": ("target_weight", "weight", "target_pct"),
    "reference_price": ("reference_price", "price", "close", "last_price", "mark"),
    "quantity": ("quantity", "shares", "contracts", "units"),
    "target_notional": ("target_notional", "notional", "gross_notional"),
    "order_type": ("order_type", "order_style"),
    "limit_price": ("limit_price", "limit", "limit_px"),
    "stop_price": ("stop_price", "stop", "stop_px"),
    "time_in_force": ("time_in_force", "tif"),
    "exchange": ("exchange", "primary_exchange"),
    "currency": ("currency", "ccy"),
    "broker_symbol": ("broker_symbol", "ibkr_symbol", "contract_symbol"),
    "multiplier": ("multiplier", "contract_multiplier"),
    "horizon": ("horizon", "holding_period"),
    "rationale": ("rationale", "reason", "notes"),
    "generated_at": ("generated_at", "as_of", "timestamp", "datetime"),
}


class ResearchSignalBridgeError(ValueError):
    """Raised when signal artifacts cannot be loaded or converted."""


@dataclass(frozen=True, slots=True)
class ResearchSignalIssue:
    row_number: int | None
    message: str
    row: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SignalProposalConfig:
    source: str = "research_pipeline"
    default_strategy_id: str = "unassigned_strategy"
    default_asset_class: AssetClass = AssetClass.EQUITY
    default_order_type: OrderType = OrderType.MARKET
    default_time_in_force: str = "DAY"
    default_quantity: float = 1.0
    default_notional_per_signal: float | None = None
    min_strength: float = 0.0
    allow_short: bool = True
    write_empty: bool = False
    paper_only: bool = True


@dataclass(frozen=True, slots=True)
class SignalProposalResult:
    proposal: TradeProposal
    issues: tuple[ResearchSignalIssue, ...] = ()
    written_path: Path | None = None


DEFAULT_SIGNAL_PROPOSAL_CONFIG = SignalProposalConfig()


def load_research_signal_rows(path: Path) -> list[dict[str, Any]]:
    """Load research signal rows from JSON, JSONL, or CSV."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _rows_from_json_payload(payload)
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ResearchSignalBridgeError(
                    f"line {line_number} must contain a JSON object"
                )
            rows.append(dict(value))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ResearchSignalBridgeError(
        f"unsupported signal artifact extension {suffix!r}; use .json, .jsonl, or .csv"
    )


def build_trade_proposal_from_signal_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    proposal_id: str | None = None,
    config: SignalProposalConfig | None = None,
    input_path: Path | None = None,
) -> SignalProposalResult:
    """Convert research signal rows into a paper-only trade proposal."""

    resolved_config = config or SignalProposalConfig()
    intents: list[OrderIntent] = []
    issues: list[ResearchSignalIssue] = []
    strategy_ids: set[str] = set()
    research_run_ids: set[str] = set()

    for row_number, row in enumerate(rows, 1):
        row_dict = dict(row)
        try:
            signal = signal_from_row(row_dict, config=resolved_config)
            if signal.direction == 0:
                issues.append(_issue(row_number, "direction is flat; skipped", row_dict))
                continue
            if abs(signal.strength) < resolved_config.min_strength:
                issues.append(
                    _issue(
                        row_number,
                        f"strength {signal.strength:.4f} below minimum",
                        row_dict,
                    )
                )
                continue
            if signal.direction < 0 and not resolved_config.allow_short:
                issues.append(_issue(row_number, "short signals disabled; skipped", row_dict))
                continue

            intent = order_intent_from_signal_row(
                signal,
                row_dict,
                config=resolved_config,
            )
            intents.append(intent)
            strategy_ids.add(signal.strategy_id)
            if signal.research_run_id:
                research_run_ids.add(signal.research_run_id)
        except Exception as exc:
            issues.append(_issue(row_number, str(exc), row_dict))

    proposal = TradeProposal(
        proposal_id=proposal_id or _default_proposal_id(),
        source=resolved_config.source,
        intents=tuple(intents),
        paper_only=resolved_config.paper_only,
        strategy_id=next(iter(strategy_ids)) if len(strategy_ids) == 1 else None,
        research_run_id=next(iter(research_run_ids)) if len(research_run_ids) == 1 else None,
        notes="Generated from research signal rows for paper-dashboard review.",
        metadata={
            "input_path": str(input_path) if input_path else None,
            "skipped_rows": len(issues),
            "raw_rows": len(rows) if isinstance(rows, Sequence) else None,
        },
    )
    return SignalProposalResult(proposal=proposal, issues=tuple(issues))


def write_research_signal_proposal(
    signal_path: Path,
    *,
    output_dir: Path | None = None,
    proposal_id: str | None = None,
    config: SignalProposalConfig | None = None,
    overwrite: bool = False,
) -> SignalProposalResult:
    """Load a research signal artifact and write a paper proposal JSON artifact."""

    rows = load_research_signal_rows(signal_path)
    resolved_config = config or SignalProposalConfig()
    result = build_trade_proposal_from_signal_rows(
        rows,
        proposal_id=proposal_id,
        config=resolved_config,
        input_path=signal_path,
    )
    if not result.proposal.intents and not resolved_config.write_empty:
        return result

    directory = output_dir or trade_proposal_directory(load_settings())
    path = write_trade_proposal_artifact(
        result.proposal,
        directory,
        overwrite=overwrite,
    )
    return SignalProposalResult(
        proposal=result.proposal,
        issues=result.issues,
        written_path=path,
    )


def signal_from_row(
    row: Mapping[str, Any],
    *,
    config: SignalProposalConfig | None = None,
) -> Signal:
    resolved_config = config or SignalProposalConfig()
    symbol = _required_text(row, "symbol").upper()
    strategy_id = _text(row, "strategy_id") or resolved_config.default_strategy_id
    direction = _direction(row)
    strength = _strength(row, direction)
    asset_class = _asset_class(row, resolved_config.default_asset_class)

    return Signal(
        instrument=Instrument(
            symbol=symbol,
            asset_class=asset_class,
            exchange=_text(row, "exchange"),
            currency=(_text(row, "currency") or "USD").upper(),
            broker_symbol=_text(row, "broker_symbol"),
            multiplier=_positive_float(row, "multiplier") or 1.0,
            metadata={},
        ),
        direction=direction,
        strength=strength,
        strategy_id=strategy_id,
        generated_at=_datetime(row, "generated_at") or utc_now(),
        target_weight=_float(row, "target_weight"),
        horizon=_text(row, "horizon"),
        research_run_id=_text(row, "research_run_id"),
        metadata={"source_row": dict(row)},
    )


def order_intent_from_signal_row(
    signal: Signal,
    row: Mapping[str, Any],
    *,
    config: SignalProposalConfig | None = None,
) -> OrderIntent:
    resolved_config = config or SignalProposalConfig()
    reference_price = _positive_float(row, "reference_price")
    limit_price = _positive_float(row, "limit_price")
    stop_price = _positive_float(row, "stop_price")
    order_type = _order_type(row, resolved_config.default_order_type)
    if limit_price is not None and _raw_value(row, "order_type") is None:
        order_type = OrderType.LIMIT
    if order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and limit_price is None:
        limit_price = reference_price

    quantity = _quantity(row, config=resolved_config, reference_price=reference_price)

    return OrderIntent(
        instrument=signal.instrument,
        side=OrderSide.BUY if signal.direction > 0 else OrderSide.SELL,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=_text(row, "time_in_force") or resolved_config.default_time_in_force,
        strategy_id=signal.strategy_id,
        signal_id=_text(row, "signal_id"),
        target_weight=signal.target_weight,
        reference_price=reference_price,
        confidence=signal.strength,
        rationale=_text(row, "rationale"),
        metadata={
            "research_run_id": signal.research_run_id,
            "horizon": signal.horizon,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert research signal rows into a paper trade proposal artifact."
    )
    parser.add_argument("signal_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--proposal-id")
    parser.add_argument("--source", default=DEFAULT_SIGNAL_PROPOSAL_CONFIG.source)
    parser.add_argument(
        "--strategy-id",
        default=DEFAULT_SIGNAL_PROPOSAL_CONFIG.default_strategy_id,
    )
    parser.add_argument(
        "--min-strength",
        type=float,
        default=DEFAULT_SIGNAL_PROPOSAL_CONFIG.min_strength,
    )
    parser.add_argument(
        "--default-quantity",
        type=float,
        default=DEFAULT_SIGNAL_PROPOSAL_CONFIG.default_quantity,
    )
    parser.add_argument("--default-notional", type=float)
    parser.add_argument("--no-shorts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = SignalProposalConfig(
        source=args.source,
        default_strategy_id=args.strategy_id,
        min_strength=args.min_strength,
        default_quantity=args.default_quantity,
        default_notional_per_signal=args.default_notional,
        allow_short=not args.no_shorts,
    )
    result = write_research_signal_proposal(
        args.signal_path,
        output_dir=args.output_dir,
        proposal_id=args.proposal_id,
        config=config,
        overwrite=args.overwrite,
    )

    print(f"proposal_id={result.proposal.proposal_id}")
    print(f"intents={len(result.proposal.intents)}")
    print(f"issues={len(result.issues)}")
    print(f"written_path={result.written_path or ''}")
    for issue in result.issues:
        print(f"issue row={issue.row_number}: {issue.message}")
    return 0 if result.written_path or result.proposal.intents else 1


def _rows_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("signals", "rows", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return _mapping_rows(value)
        return [dict(payload)]
    if isinstance(payload, list):
        return _mapping_rows(payload)
    raise ResearchSignalBridgeError("JSON signal artifact must be an object or list")


def _mapping_rows(values: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            raise ResearchSignalBridgeError(f"row {index} must be an object")
        rows.append(dict(value))
    return rows


def _raw_value(row: Mapping[str, Any], field: str) -> Any:
    for key in SIGNAL_FIELD_ALIASES[field]:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _text(row: Mapping[str, Any], field: str) -> str | None:
    value = _raw_value(row, field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(row: Mapping[str, Any], field: str) -> str:
    text = _text(row, field)
    if text is None:
        raise ResearchSignalBridgeError(f"{field} is required")
    return text


def _float(row: Mapping[str, Any], field: str) -> float | None:
    value = _raw_value(row, field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchSignalBridgeError(f"{field} must be numeric") from exc


def _positive_float(row: Mapping[str, Any], field: str) -> float | None:
    value = _float(row, field)
    if value is not None and value <= 0:
        raise ResearchSignalBridgeError(f"{field} must be positive")
    return value


def _direction(row: Mapping[str, Any]) -> int:
    value = _raw_value(row, "direction")
    if value is None:
        score = _float(row, "strength")
        if score is None:
            raise ResearchSignalBridgeError("direction or score is required")
        value = score

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"buy", "long", "bullish", "+", "+1", "1"}:
            return 1
        if normalized in {"sell", "short", "bearish", "-", "-1"}:
            return -1
        if normalized in {"flat", "hold", "neutral", "0"}:
            return 0

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchSignalBridgeError("direction must be buy/sell/flat or numeric") from exc
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


def _strength(row: Mapping[str, Any], direction: int) -> float:
    value = _float(row, "strength")
    if value is None:
        return 1.0 if direction else 0.0
    strength = abs(value)
    if strength > 1 and strength <= 100:
        strength = strength / 100
    if not 0 <= strength <= 1:
        raise ResearchSignalBridgeError("strength must be between 0 and 1")
    return strength


def _asset_class(row: Mapping[str, Any], default: AssetClass) -> AssetClass:
    value = _text(row, "asset_class")
    if value is None:
        return default
    normalized = value.lower().replace(" ", "_")
    try:
        return AssetClass(normalized)
    except ValueError as exc:
        raise ResearchSignalBridgeError(f"asset_class has unsupported value {value!r}") from exc


def _order_type(row: Mapping[str, Any], default: OrderType) -> OrderType:
    value = _text(row, "order_type")
    if value is None:
        return default
    normalized = value.lower().replace(" ", "_")
    try:
        return OrderType(normalized)
    except ValueError as exc:
        raise ResearchSignalBridgeError(f"order_type has unsupported value {value!r}") from exc


def _datetime(row: Mapping[str, Any], field: str) -> datetime | None:
    text = _text(row, field)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchSignalBridgeError(f"{field} must be an ISO datetime string") from exc


def _quantity(
    row: Mapping[str, Any],
    *,
    config: SignalProposalConfig,
    reference_price: float | None,
) -> float:
    explicit_quantity = _positive_float(row, "quantity")
    if explicit_quantity is not None:
        return explicit_quantity

    notional = _positive_float(row, "target_notional") or config.default_notional_per_signal
    if notional is not None:
        if reference_price is None:
            raise ResearchSignalBridgeError(
                "reference_price is required when sizing by notional"
            )
        multiplier = _positive_float(row, "multiplier") or 1.0
        return notional / (reference_price * multiplier)

    if config.default_quantity <= 0:
        raise ResearchSignalBridgeError("default_quantity must be positive")
    return config.default_quantity


def _issue(row_number: int | None, message: str, row: Mapping[str, Any]) -> ResearchSignalIssue:
    return ResearchSignalIssue(row_number=row_number, message=message, row=dict(row))


def _default_proposal_id() -> str:
    return "research-signals-" + utc_now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
