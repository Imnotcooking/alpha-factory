"""Base contracts for portfolio allocation engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from oqp.intelligence.base import BaseEngine


@dataclass(frozen=True, slots=True)
class AllocationRequest:
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AllocationResult:
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BaseAllocationEngine(BaseEngine):
    category = "allocation_engine"
