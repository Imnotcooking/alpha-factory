"""Adapters between option backtests and research-dashboard artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.research.backtesting.models import BacktestBackendMetadata, ExecutionBacktestResult

from oqp.options.backtesting.models import OptionBacktestResult


def option_result_to_execution_result(result: OptionBacktestResult) -> ExecutionBacktestResult:
    equity = result.equity_curve["equity"].to_numpy(dtype=float)
    gross = result.equity_curve["gross_equity"].to_numpy(dtype=float)
    turnover = result.equity_curve["turnover"].to_numpy(dtype=float)
    leverage = (
        result.equity_curve["gross_exposure"].replace(0, np.nan)
        / result.equity_curve["equity"].replace(0, np.nan)
    ).fillna(0.0)
    return ExecutionBacktestResult(
        equity_curve=equity,
        gross_equity_curve=gross,
        total_cost=turnover * 0.0,
        portfolio_leverage=leverage.to_numpy(dtype=float),
        backend=BacktestBackendMetadata(
            backend_id="options_event_driven",
            backend_name="Options Event-Driven Python",
            metadata={"backtest_route": "event_driven_options", **result.diagnostics},
        ),
    )


def option_result_to_returns_frame(result: OptionBacktestResult) -> pd.DataFrame:
    return result.to_returns_frame()
