"""Broker adapter interfaces and implementations."""

from oqp.brokers.base import BrokerAdapter, BrokerAdapterError
from oqp.brokers.models import (
    AccountSummary,
    BrokerConnectionConfig,
    BrokerConnectionStatus,
    BrokerEnvironment,
    BrokerHealth,
    BrokerSnapshot,
    CancelResult,
    CashBalance,
    ExecutionReport,
    OrderReceipt,
)
from oqp.brokers.registry import (
    BrokerProfileError,
    get_broker_adapter,
    get_broker_connection_config,
    get_broker_profile_config,
    register_broker_adapter,
)
from oqp.brokers.ibkr import (
    IBKRBrokerAdapter,
    IBKRReadOnlyPortfolioSnapshot,
    fetch_ibkr_readonly_portfolio_snapshot,
    ibkr_account_summary_to_middle_office_metrics,
    ibkr_position_to_middle_office_row,
)

__all__ = [
    "AccountSummary",
    "BrokerAdapter",
    "BrokerAdapterError",
    "BrokerConnectionConfig",
    "BrokerConnectionStatus",
    "BrokerEnvironment",
    "BrokerHealth",
    "BrokerSnapshot",
    "BrokerProfileError",
    "CancelResult",
    "CashBalance",
    "ExecutionReport",
    "get_broker_adapter",
    "get_broker_connection_config",
    "get_broker_profile_config",
    "IBKRBrokerAdapter",
    "IBKRReadOnlyPortfolioSnapshot",
    "fetch_ibkr_readonly_portfolio_snapshot",
    "ibkr_account_summary_to_middle_office_metrics",
    "ibkr_position_to_middle_office_row",
    "OrderReceipt",
    "register_broker_adapter",
]
