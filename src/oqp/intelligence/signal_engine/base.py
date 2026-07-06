"""Base contracts for signal scoring and ensemble engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from oqp.intelligence.base import BaseEngine


@dataclass(frozen=True, slots=True)
class SignalRequest:
    raw_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SignalResult:
    scored_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BaseSignalEngine(BaseEngine):
    category = "signal_engine"
