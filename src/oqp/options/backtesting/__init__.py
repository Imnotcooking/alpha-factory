"""Event-driven options backtesting primitives."""

from oqp.options.backtesting.adapters import (
    option_result_to_execution_result,
    option_result_to_returns_frame,
)
from oqp.options.backtesting.engine import OptionBacktestEngine
from oqp.options.backtesting.ledger import OptionBacktestLedger, OptionPositionLot
from oqp.options.backtesting.models import (
    OptionBacktestConfig,
    OptionBacktestRequest,
    OptionBacktestResult,
)

__all__ = [
    "OptionBacktestConfig",
    "OptionBacktestEngine",
    "OptionBacktestLedger",
    "OptionBacktestRequest",
    "OptionBacktestResult",
    "OptionPositionLot",
    "option_result_to_execution_result",
    "option_result_to_returns_frame",
]
