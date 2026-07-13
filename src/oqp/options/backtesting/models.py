"""Typed models for event-driven listed-options backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from oqp.contracts.market_vertical import normalize_market_vertical
from oqp.options.liquidity import OptionLiquidityRule
from oqp.options.margin import OptionMarginPolicy


@dataclass(frozen=True, slots=True)
class OptionBacktestConfig:
    min_dte: int = 7
    max_dte: int = 60
    target_moneyness: float = 1.0
    contracts_per_signal: int = 1
    allow_multiple_positions_per_underlying: bool = False
    rate: float = 0.045

    def __post_init__(self) -> None:
        if self.min_dte < 0:
            raise ValueError("min_dte cannot be negative.")
        if self.max_dte < self.min_dte:
            raise ValueError("max_dte must be >= min_dte.")
        if self.target_moneyness <= 0:
            raise ValueError("target_moneyness must be positive.")
        if self.contracts_per_signal <= 0:
            raise ValueError("contracts_per_signal must be positive.")


@dataclass(frozen=True, slots=True)
class OptionBacktestRequest:
    chain: pd.DataFrame
    underlying: pd.DataFrame
    signals: pd.DataFrame
    market_vertical: str = "OPTIONS_US"
    initial_capital: float = 100_000.0
    config: OptionBacktestConfig = field(default_factory=OptionBacktestConfig)
    liquidity: OptionLiquidityRule = field(default_factory=OptionLiquidityRule)
    margin: OptionMarginPolicy = field(default_factory=OptionMarginPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        market_vertical = normalize_market_vertical(self.market_vertical)
        if market_vertical not in {"OPTIONS_US", "OPTIONS_CN"}:
            raise ValueError("OptionBacktestRequest.market_vertical must be OPTIONS_US or OPTIONS_CN.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        object.__setattr__(self, "market_vertical", market_vertical)


@dataclass(frozen=True, slots=True)
class OptionBacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        if self.equity_curve.empty:
            return float("nan")
        return float(self.equity_curve["equity"].iloc[-1])

    def to_returns_frame(self) -> pd.DataFrame:
        if self.equity_curve.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "gross_return",
                    "net_return",
                    "portfolio_leverage",
                    "daily_turnover",
                    "long_weight",
                    "short_weight",
                    "long_notional",
                    "short_notional",
                    "gross_notional",
                    "net_notional",
                ]
            )
        out = self.equity_curve.loc[:, ["date", "equity", "gross_equity", "turnover", "gross_exposure"]].copy()
        out["gross_return"] = out["gross_equity"].pct_change().fillna(0.0)
        out["net_return"] = out["equity"].pct_change().fillna(0.0)
        out["portfolio_leverage"] = out["gross_exposure"] / out["equity"].replace(0, pd.NA)
        out["daily_turnover"] = out["turnover"] / out["equity"].replace(0, pd.NA)
        out["long_notional"] = pd.to_numeric(out["gross_exposure"], errors="coerce").fillna(0.0)
        out["short_notional"] = 0.0
        out["gross_notional"] = out["long_notional"].abs() + out["short_notional"].abs()
        out["net_notional"] = out["long_notional"] + out["short_notional"]
        equity = pd.to_numeric(out["equity"], errors="coerce").replace(0.0, pd.NA)
        out["long_weight"] = out["long_notional"] / equity
        out["short_weight"] = out["short_notional"] / equity
        return out.loc[
            :,
            [
                "date",
                "gross_return",
                "net_return",
                "portfolio_leverage",
                "daily_turnover",
                "long_weight",
                "short_weight",
                "long_notional",
                "short_notional",
                "gross_notional",
                "net_notional",
            ],
        ]
