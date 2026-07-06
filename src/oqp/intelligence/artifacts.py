"""Lightweight artifact references for intelligence engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EngineArtifactRef:
    """Pointer to a model/report artifact without loading the artifact itself."""

    artifact_id: str
    engine_id: str
    path: str
    artifact_type: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return Path(self.path).expanduser().exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "engine_id": self.engine_id,
            "path": self.path,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "exists": self.exists,
            "metadata": dict(self.metadata),
        }
