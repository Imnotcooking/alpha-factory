"""Framework-neutral contracts for adaptive state-space filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StateSpaceSchema:
    """Column contract for state-space feature builders."""

    date_col: str = "date"
    y_col: str = "y"
    x_cols: tuple[str, ...] = field(default_factory=tuple)
    group_cols: tuple[str, ...] = field(default_factory=lambda: ("ticker",))

    def validate(self) -> None:
        if not self.y_col:
            raise ValueError("y_col is required.")
        if not self.x_cols:
            raise ValueError("At least one x_col is required.")


@dataclass(frozen=True)
class StateSpaceArtifact:
    """Metadata returned when deterministic state-space outputs are persisted."""

    output_path: str
    metadata_path: str
    row_count: int
    feature_columns: list[str]
    metadata: dict[str, Any]


class StateSpaceFilter(ABC):
    """Small interface shared by adaptive state-space filters."""

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return input-index-aligned state-space features."""
        raise NotImplementedError


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert nested dataclasses to plain dictionaries for JSON metadata."""

    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported metadata value: {type(value)!r}")


__all__ = [
    "StateSpaceArtifact",
    "StateSpaceFilter",
    "StateSpaceSchema",
    "dataclass_to_dict",
]
