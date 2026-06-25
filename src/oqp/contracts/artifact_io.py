"""JSON artifact I/O for strategy candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from oqp.config import OQPSettings
from oqp.contracts.strategy_candidate import (
    CandidateMetrics,
    CandidateSafetyLimits,
    CandidateStatus,
    StrategyCandidate,
)


STRATEGY_CANDIDATE_DIRNAME = "strategy_candidates"


class StrategyCandidateArtifactError(ValueError):
    """Raised when a strategy-candidate artifact cannot be parsed."""


@dataclass(frozen=True, slots=True)
class LoadedStrategyCandidate:
    candidate: StrategyCandidate
    path: Path


@dataclass(frozen=True, slots=True)
class StrategyCandidateArtifactIssue:
    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class StrategyCandidateLoadResult:
    directory: Path
    loaded: tuple[LoadedStrategyCandidate, ...] = ()
    issues: tuple[StrategyCandidateArtifactIssue, ...] = ()


def strategy_candidate_directory(settings: OQPSettings) -> Path:
    return settings.artifact_root / STRATEGY_CANDIDATE_DIRNAME


def load_strategy_candidate_artifacts(
    directory: Path,
    *,
    max_files: int | None = None,
) -> StrategyCandidateLoadResult:
    """Load all JSON strategy-candidate artifacts in newest-file-first order."""

    if not directory.exists():
        return StrategyCandidateLoadResult(directory=directory)
    if not directory.is_dir():
        return StrategyCandidateLoadResult(
            directory=directory,
            issues=(
                StrategyCandidateArtifactIssue(
                    path=directory,
                    message="strategy candidate artifact path is not a directory",
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

    loaded: list[LoadedStrategyCandidate] = []
    issues: list[StrategyCandidateArtifactIssue] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for candidate_payload in _iter_candidate_payloads(payload):
                loaded.append(
                    LoadedStrategyCandidate(
                        candidate=parse_strategy_candidate(candidate_payload),
                        path=path,
                    )
                )
        except Exception as exc:
            issues.append(StrategyCandidateArtifactIssue(path=path, message=str(exc)))

    return StrategyCandidateLoadResult(
        directory=directory,
        loaded=tuple(loaded),
        issues=tuple(issues),
    )


def parse_strategy_candidate(payload: Mapping[str, Any]) -> StrategyCandidate:
    _require_mapping(payload, "strategy candidate")
    metrics = _dict(payload.get("metrics"), "metrics")
    safety_limits = _dict(payload.get("safety_limits"), "safety_limits")

    kwargs: dict[str, Any] = {
        "candidate_id": _required_str(payload, "candidate_id"),
        "strategy_id": _required_str(payload, "strategy_id"),
        "source": _required_str(payload, "source"),
        "promotion_status": _candidate_status(
            payload.get("promotion_status", CandidateStatus.RESEARCH_ONLY.value)
        ),
        "native_market_vertical": _optional_str(payload.get("native_market_vertical"))
        or "UNKNOWN",
        "tested_market_vertical": _optional_str(payload.get("tested_market_vertical")),
        "target_market_vertical": _optional_str(payload.get("target_market_vertical")),
        "intended_market_verticals": tuple(_list(payload, "intended_market_verticals")),
        "research_run_id": _optional_str(payload.get("research_run_id")),
        "dataset_id": _optional_str(payload.get("dataset_id")),
        "universe_id": _optional_str(payload.get("universe_id")),
        "data_frequency": _optional_str(payload.get("data_frequency")),
        "data_vendor": _optional_str(payload.get("data_vendor")),
        "execution_assumption": _optional_str(payload.get("execution_assumption")),
        "evaluation_geometry": _optional_str(payload.get("evaluation_geometry")),
        "ic_metric": _optional_str(payload.get("ic_metric")),
        "metrics": _parse_candidate_metrics(metrics),
        "safety_limits": CandidateSafetyLimits(
            paper_only=_bool(safety_limits.get("paper_only", True), "paper_only"),
            allow_live_trading=_bool(
                safety_limits.get("allow_live_trading", False),
                "allow_live_trading",
            ),
            max_gross_exposure=_optional_float(
                safety_limits.get("max_gross_exposure"),
                "max_gross_exposure",
            ),
            max_single_position=_optional_float(
                safety_limits.get("max_single_position"),
                "max_single_position",
            ),
            max_daily_loss_pct=_optional_float(
                safety_limits.get("max_daily_loss_pct"),
                "max_daily_loss_pct",
            ),
            max_order_notional=_optional_float(
                safety_limits.get("max_order_notional"),
                "max_order_notional",
            ),
        ),
        "instrument_mapping_required": _bool(
            payload.get("instrument_mapping_required", False),
            "instrument_mapping_required",
        ),
        "approved_broker_profile": _optional_str(payload.get("approved_broker_profile")),
        "notes": _optional_str(payload.get("notes")),
        "tags": tuple(_list(payload, "tags")),
        "metadata": _dict(payload.get("metadata"), "metadata"),
    }

    if payload.get("created_at") is not None:
        kwargs["created_at"] = _datetime(payload["created_at"], "created_at")

    return StrategyCandidate(**kwargs)


def strategy_candidate_to_dict(candidate: StrategyCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["promotion_status"] = candidate.promotion_status.value
    payload["created_at"] = candidate.created_at.isoformat()
    payload["tags"] = list(candidate.tags)
    return payload


def _parse_candidate_metrics(payload: Mapping[str, Any]) -> CandidateMetrics:
    numeric_fields = {
        "validation_ic",
        "holdout_ic",
        "crisis_ic",
        "validation_hit_rate",
        "holdout_hit_rate",
        "sharpe_ratio",
        "annualized_return",
        "max_drawdown",
        "turnover_rate",
        "avg_daily_cost_bps",
        "metric_p_value",
        "sharpe_p_value",
    }
    values = {
        field_name: _optional_float(payload.get(field_name), field_name)
        for field_name in numeric_fields
    }
    values["significance"] = _optional_str(payload.get("significance"))
    return CandidateMetrics(**values)


def write_strategy_candidate_artifact(
    candidate: StrategyCandidate,
    directory: Path,
    *,
    filename: str | None = None,
    overwrite: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in candidate.candidate_id
    ).strip("-")
    path = directory / (filename or f"{safe_id or 'strategy-candidate'}.json")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.write_text(
        json.dumps(strategy_candidate_to_dict(candidate), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _iter_candidate_payloads(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("candidates"), list):
        for item in payload["candidates"]:
            _require_mapping(item, "strategy candidate")
            yield item
        return
    _require_mapping(payload, "strategy candidate")
    yield payload


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise StrategyCandidateArtifactError(f"{label} must be an object")


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StrategyCandidateArtifactError(f"{key} is required")
    return value.strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _candidate_status(value: Any) -> CandidateStatus:
    if isinstance(value, CandidateStatus):
        return value
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return CandidateStatus(normalized)
    except ValueError as exc:
        raise StrategyCandidateArtifactError(
            f"promotion_status has unsupported value {value!r}"
        ) from exc


def _datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise StrategyCandidateArtifactError(
            f"{field_name} must be an ISO datetime string"
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyCandidateArtifactError(
            f"{field_name} must be an ISO datetime string"
        ) from exc


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise StrategyCandidateArtifactError(f"{field_name} must be an object")
    return dict(value)


def _list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise StrategyCandidateArtifactError(f"{key} must be a list")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise StrategyCandidateArtifactError(f"{field_name} must be boolean")


def _optional_float(value: Any, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise StrategyCandidateArtifactError(f"{field_name} must be numeric") from exc
