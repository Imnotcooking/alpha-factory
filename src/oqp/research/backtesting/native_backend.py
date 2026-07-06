"""Native C++ backend adapter for research execution simulations."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np

from oqp.native import load_quant_core
from oqp.research.backtesting.models import (
    BacktestBackendMetadata,
    ExecutionBacktestRequest,
    ExecutionBacktestResult,
)


class NativeBacktestBackend:
    backend_id = "native"
    backend_name = "OQP Native C++ Backtest Backend"
    required_features = ("ExecutionEngine", "FuturesMargin", "StochasticTCAWrapper")

    def __init__(
        self,
        *,
        quant_core: ModuleType | None = None,
        legacy_paths: tuple[str | Path, ...] = (),
    ) -> None:
        self._quant_core = quant_core
        self.legacy_paths = tuple(legacy_paths)

    @property
    def quant_core(self) -> ModuleType:
        if self._quant_core is None:
            self._quant_core = load_quant_core(
                self.required_features,
                legacy_paths=self.legacy_paths,
            )
        return self._quant_core

    def run(self, request: ExecutionBacktestRequest) -> ExecutionBacktestResult:
        qc = self.quant_core
        engine = self._build_default_engine(qc, request)

        arrays = _PreparedArrays.from_request(request)
        if request.period_returns is not None and hasattr(engine, "run_simulation_with_costs_and_returns"):
            native_result = engine.run_simulation_with_costs_and_returns(
                arrays.asset_ids,
                arrays.prices,
                arrays.target_weights,
                arrays.period_returns,
                arrays.volumes,
                arrays.volatilities,
                arrays.hursts,
                arrays.time_ids,
                arrays.date_ids,
                arrays.multipliers,
                arrays.fee_types,
                arrays.fee_open,
                arrays.fee_close_history,
                arrays.fee_close_today,
                arrays.tick_sizes,
                bool(request.integer_lots),
            )
        elif request.has_cost_inputs and hasattr(engine, "run_simulation_with_costs"):
            native_result = engine.run_simulation_with_costs(
                arrays.asset_ids,
                arrays.prices,
                arrays.target_weights,
                arrays.volumes,
                arrays.volatilities,
                arrays.hursts,
                arrays.date_ids,
                arrays.multipliers,
                arrays.fee_types,
                arrays.fee_open,
                arrays.fee_close_history,
                arrays.fee_close_today,
                arrays.tick_sizes,
                bool(request.integer_lots),
            )
        else:
            native_result = engine.run_simulation(
                arrays.asset_ids,
                arrays.prices,
                arrays.target_weights,
                arrays.volumes,
                arrays.volatilities,
                arrays.hursts,
                arrays.date_ids,
            )

        return self._coerce_result(native_result, qc)

    def _build_default_engine(self, qc: ModuleType, request: ExecutionBacktestRequest):
        tca = qc.StochasticTCAWrapper(1e-4, 0.1, 2.0, 60)
        margin = qc.FuturesMargin(maintenance_req=0.05)
        return qc.ExecutionEngine(
            tca_model=tca,
            margin_model=margin,
            initial_capital=float(request.initial_capital),
            deadband=float(request.deadband),
            enforce_price_limits=False,
            enforce_t1=False,
        )

    def _coerce_result(self, native_result, qc: ModuleType) -> ExecutionBacktestResult:
        backend = BacktestBackendMetadata(
            backend_id=self.backend_id,
            backend_name=self.backend_name,
            native_module=qc.__name__,
            native_source_path=str(getattr(qc, "__file__", "")) or None,
        )

        if not isinstance(native_result, dict):
            return ExecutionBacktestResult(
                equity_curve=np.asarray(native_result, dtype=np.float64),
                backend=backend,
            )

        diagnostics = {
            key: np.asarray(value)
            for key, value in native_result.items()
            if key
            not in {
                "equity_curve",
                "gross_equity_curve",
                "slippage_cost",
                "exchange_fee",
                "total_cost",
                "executed_weight",
                "trade_notional",
                "trade_contracts",
                "portfolio_leverage",
            }
        }
        return ExecutionBacktestResult(
            equity_curve=np.asarray(native_result["equity_curve"], dtype=np.float64),
            backend=backend,
            gross_equity_curve=_optional_array(native_result, "gross_equity_curve"),
            slippage_cost=_optional_array(native_result, "slippage_cost"),
            exchange_fee=_optional_array(native_result, "exchange_fee"),
            total_cost=_optional_array(native_result, "total_cost"),
            executed_weight=_optional_array(native_result, "executed_weight"),
            trade_notional=_optional_array(native_result, "trade_notional"),
            trade_contracts=_optional_array(native_result, "trade_contracts"),
            portfolio_leverage=_optional_array(native_result, "portfolio_leverage"),
            diagnostics=diagnostics,
        )


class _PreparedArrays:
    def __init__(self, request: ExecutionBacktestRequest) -> None:
        self.asset_ids = np.ascontiguousarray(request.asset_ids, dtype=np.int32)
        self.prices = np.ascontiguousarray(request.prices, dtype=np.float64)
        self.target_weights = np.ascontiguousarray(request.target_weights, dtype=np.float64)
        self.volumes = _float_array(request.volumes, request.n_rows, 1_000.0)
        self.volatilities = _float_array(request.volatilities, request.n_rows, 0.01)
        self.hursts = _float_array(request.hursts, request.n_rows, 0.5)
        self.date_ids = _int_array(request.date_ids, request.n_rows, arange_default=True)
        self.time_ids = _int_array(request.time_ids, request.n_rows, arange_default=True)
        self.period_returns = _float_array(request.period_returns, request.n_rows, 0.0)
        self.multipliers = _float_array(request.multipliers, request.n_rows, 1.0)
        self.fee_types = _int_array(request.fee_types, request.n_rows)
        self.fee_open = _float_array(request.fee_open, request.n_rows, 0.0)
        self.fee_close_history = _float_array(request.fee_close_history, request.n_rows, 0.0)
        self.fee_close_today = _float_array(request.fee_close_today, request.n_rows, 0.0)
        self.tick_sizes = _float_array(request.tick_sizes, request.n_rows, 0.0)

    @classmethod
    def from_request(cls, request: ExecutionBacktestRequest) -> "_PreparedArrays":
        return cls(request)


def _float_array(values, n_rows: int, default: float) -> np.ndarray:
    if values is None:
        values = np.full(n_rows, default)
    return np.ascontiguousarray(values, dtype=np.float64)


def _int_array(values, n_rows: int, *, arange_default: bool = False) -> np.ndarray:
    if values is None:
        values = np.arange(n_rows) if arange_default else np.zeros(n_rows)
    return np.ascontiguousarray(values, dtype=np.int32)


def _optional_array(payload: dict, key: str) -> np.ndarray | None:
    if key not in payload:
        return None
    return np.asarray(payload[key], dtype=np.float64)
