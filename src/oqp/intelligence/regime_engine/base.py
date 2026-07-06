"""Base contracts for market regime engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from oqp.intelligence.base import BaseEngine


@dataclass(frozen=True, slots=True)
class RegimeState:
    label: str
    probability: float | None = None
    as_of: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRegimeEngine(BaseEngine):
    category = "regime_engine"

    def state_frame(self, states: tuple[RegimeState, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Label": state.label,
                    "Probability": state.probability,
                    "As Of": state.as_of,
                    "Metadata": state.metadata,
                }
                for state in states
            ]
        )
