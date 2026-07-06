"""Typed contracts for research execution backtests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class BacktestBackendMetadata:
    backend_id: str
    backend_name: str
    native_module: str | None = None
    native_source_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionBacktestRequest:
    """Array-level request passed to a research execution simulator."""

    asset_ids: Sequence[int]
    prices: Sequence[float]
    target_weights: Sequence[float]
    volumes: Sequence[float] | None = None
    volatilities: Sequence[float] | None = None
    hursts: Sequence[float] | None = None
    date_ids: Sequence[int] | None = None
    time_ids: Sequence[int] | None = None
    period_returns: Sequence[float] | None = None
    multipliers: Sequence[float] | None = None
    fee_types: Sequence[int] | None = None
    fee_open: Sequence[float] | None = None
    fee_close_history: Sequence[float] | None = None
    fee_close_today: Sequence[float] | None = None
    tick_sizes: Sequence[float] | None = None
    initial_capital: float = 1_000_000.0
    deadband: float = 0.015
    integer_lots: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_rows = len(self.asset_ids)
        if n_rows == 0:
            raise ValueError("ExecutionBacktestRequest requires at least one row")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.deadband < 0:
            raise ValueError("deadband cannot be negative")

        for name, values in self._all_series().items():
            if values is not None and len(values) != n_rows:
                raise ValueError(f"{name} length must match asset_ids length")

    @property
    def n_rows(self) -> int:
        return len(self.asset_ids)

    @property
    def has_cost_inputs(self) -> bool:
        return all(
            values is not None
            for values in (
                self.multipliers,
                self.fee_types,
                self.fee_open,
                self.fee_close_history,
                self.fee_close_today,
            )
        )

    def _all_series(self) -> Mapping[str, Sequence[Any] | None]:
        return {
            "prices": self.prices,
            "target_weights": self.target_weights,
            "volumes": self.volumes,
            "volatilities": self.volatilities,
            "hursts": self.hursts,
            "date_ids": self.date_ids,
            "time_ids": self.time_ids,
            "period_returns": self.period_returns,
            "multipliers": self.multipliers,
            "fee_types": self.fee_types,
            "fee_open": self.fee_open,
            "fee_close_history": self.fee_close_history,
            "fee_close_today": self.fee_close_today,
            "tick_sizes": self.tick_sizes,
        }


@dataclass(frozen=True, slots=True)
class ExecutionBacktestResult:
    equity_curve: np.ndarray
    backend: BacktestBackendMetadata
    gross_equity_curve: np.ndarray | None = None
    slippage_cost: np.ndarray | None = None
    exchange_fee: np.ndarray | None = None
    total_cost: np.ndarray | None = None
    executed_weight: np.ndarray | None = None
    trade_notional: np.ndarray | None = None
    trade_contracts: np.ndarray | None = None
    portfolio_leverage: np.ndarray | None = None
    diagnostics: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve[-1])
