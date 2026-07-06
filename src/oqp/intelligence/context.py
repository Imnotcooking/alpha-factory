"""Runtime context passed into advisory intelligence engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class EngineContext:
    """One immutable bundle of runtime data for all advisory engines."""

    environment: str = "ops"
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    settings: Mapping[str, Any] = field(default_factory=dict)
    live_summary: Mapping[str, Any] = field(default_factory=dict)
    paper_summary: Mapping[str, Any] = field(default_factory=dict)
    live_nav_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_nav_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    live_positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    live_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    approved_strategies: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def account_summary(self, environment: str) -> Mapping[str, Any]:
        value = environment.strip().lower()
        if value == "live":
            return self.live_summary
        if value == "paper":
            return self.paper_summary
        raise ValueError(f"Unknown account environment: {environment!r}")

    def positions(self, environment: str) -> pd.DataFrame:
        value = environment.strip().lower()
        if value == "live":
            return self.live_positions
        if value == "paper":
            return self.paper_positions
        raise ValueError(f"Unknown account environment: {environment!r}")

    def nav_history(self, environment: str) -> pd.DataFrame:
        value = environment.strip().lower()
        if value == "live":
            return self.live_nav_history
        if value == "paper":
            return self.paper_nav_history
        raise ValueError(f"Unknown account environment: {environment!r}")
