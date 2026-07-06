"""Shared contracts for advisory intelligence engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from oqp.intelligence.context import EngineContext


class EngineStatus(str, Enum):
    """Common status vocabulary for dashboard-facing engine output."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class EngineHealth:
    engine_id: str
    status: EngineStatus
    message: str
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "status": self.status.value,
            "message": self.message,
            "checked_at": self.checked_at.isoformat(timespec="seconds"),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Structured output produced by an intelligence engine."""

    engine_id: str
    engine_name: str
    status: EngineStatus
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    health: tuple[EngineHealth, ...] = field(default_factory=tuple)

    def frame(self, name: str) -> pd.DataFrame:
        return self.frames.get(name, pd.DataFrame())

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "summary": self.summary,
            "metrics": dict(self.metrics),
            "signals": dict(self.signals),
            "metadata": dict(self.metadata),
            "health": [item.to_dict() for item in self.health],
            "frames": {name: len(frame) for name, frame in self.frames.items()},
        }


class BaseEngine(ABC):
    """Base class for modular advisory engines."""

    engine_id = "base"
    engine_name = "Base Engine"
    category = "base"
    version = "0.1.0"

    def healthcheck(self, context: EngineContext) -> EngineHealth:
        return EngineHealth(
            engine_id=self.engine_id,
            status=EngineStatus.PASS,
            message="Engine is available.",
            metadata={"category": self.category, "version": self.version},
        )

    @abstractmethod
    def run(self, context: EngineContext) -> EngineResult:
        raise NotImplementedError

    def result(
        self,
        *,
        status: EngineStatus,
        summary: str,
        metrics: dict[str, Any] | None = None,
        frames: dict[str, pd.DataFrame] | None = None,
        signals: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        health: tuple[EngineHealth, ...] | None = None,
    ) -> EngineResult:
        return EngineResult(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            status=status,
            summary=summary,
            metrics=dict(metrics or {}),
            frames=dict(frames or {}),
            signals=dict(signals or {}),
            metadata={
                "category": self.category,
                "version": self.version,
                **dict(metadata or {}),
            },
            health=health or (EngineHealth(self.engine_id, status, summary),),
        )
