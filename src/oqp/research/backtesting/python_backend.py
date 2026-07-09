"""Deterministic Python fallback for research execution simulations."""

from __future__ import annotations

import numpy as np

from oqp.research.backtesting.models import (
    BacktestBackendMetadata,
    ExecutionBacktestRequest,
    ExecutionBacktestResult,
)


class PythonBacktestBackend:
    backend_id = "python"
    backend_name = "Python Research Backtest Fallback"

    def run(self, request: ExecutionBacktestRequest) -> ExecutionBacktestResult:
        asset_ids = np.asarray(request.asset_ids, dtype=np.int32)
        prices = np.asarray(request.prices, dtype=np.float64)
        target_weights = np.asarray(request.target_weights, dtype=np.float64)
        period_returns = (
            np.asarray(request.period_returns, dtype=np.float64)
            if request.period_returns is not None
            else None
        )

        equity_curve = np.empty(request.n_rows, dtype=np.float64)
        gross_equity_curve = np.empty(request.n_rows, dtype=np.float64)
        executed_weight = np.empty(request.n_rows, dtype=np.float64)
        total_cost = np.zeros(request.n_rows, dtype=np.float64)
        portfolio_leverage = np.empty(request.n_rows, dtype=np.float64)

        last_price_by_asset: dict[int, float] = {}
        weight_by_asset: dict[int, float] = {}
        equity = float(request.initial_capital)
        gross_equity = float(request.initial_capital)

        for i, asset_id in enumerate(asset_ids):
            asset_key = int(asset_id)
            price = float(prices[i])
            previous_price = last_price_by_asset.get(asset_key, price)
            asset_return = (
                float(period_returns[i])
                if period_returns is not None
                else price / previous_price - 1.0 if previous_price else 0.0
            )

            previous_weight = weight_by_asset.get(asset_key, 0.0)
            desired_weight = float(target_weights[i])
            delta = desired_weight - previous_weight
            if abs(delta) >= request.deadband:
                current_weight = desired_weight
                turnover_cost = abs(delta) * 0.0001 * equity
            else:
                current_weight = previous_weight
                turnover_cost = 0.0

            gross_equity *= 1.0 + current_weight * asset_return
            equity = equity * (1.0 + current_weight * asset_return) - turnover_cost

            weight_by_asset[asset_key] = current_weight
            last_price_by_asset[asset_key] = price

            equity_curve[i] = equity
            gross_equity_curve[i] = gross_equity
            executed_weight[i] = current_weight
            total_cost[i] = turnover_cost
            portfolio_leverage[i] = sum(abs(weight) for weight in weight_by_asset.values())

        return ExecutionBacktestResult(
            equity_curve=equity_curve,
            backend=BacktestBackendMetadata(
                backend_id=self.backend_id,
                backend_name=self.backend_name,
                metadata={
                    "asset_class": request.asset_class,
                    "backtest_route": request.backtest_route,
                    "cost_model": "1bp turnover fallback",
                },
            ),
            gross_equity_curve=gross_equity_curve,
            total_cost=total_cost,
            executed_weight=executed_weight,
            portfolio_leverage=portfolio_leverage,
        )
