"""Backend-selecting research backtest engine."""

from __future__ import annotations

from pathlib import Path

from oqp.native import QuantCoreUnavailable
from oqp.research.backtesting.models import ExecutionBacktestRequest, ExecutionBacktestResult
from oqp.research.backtesting.native_backend import NativeBacktestBackend
from oqp.research.backtesting.python_backend import PythonBacktestBackend


class BacktestEngine:
    """Run execution backtests with native acceleration when available."""

    def __init__(
        self,
        *,
        prefer_native: bool = True,
        legacy_native_paths: tuple[str | Path, ...] = (),
    ) -> None:
        self.prefer_native = bool(prefer_native)
        self.legacy_native_paths = tuple(legacy_native_paths)
        self.python_backend = PythonBacktestBackend()

    def run(self, request: ExecutionBacktestRequest) -> ExecutionBacktestResult:
        if self.prefer_native and request.native_compatible:
            try:
                return NativeBacktestBackend(legacy_paths=self.legacy_native_paths).run(request)
            except QuantCoreUnavailable:
                pass
        return self.python_backend.run(request)
