"""Abstract interfaces for broker adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from oqp.brokers.models import (
    AccountSummary,
    BrokerConnectionConfig,
    BrokerHealth,
    BrokerSnapshot,
    CancelResult,
    CashBalance,
    ExecutionReport,
    OrderReceipt,
)
from oqp.domain import Order, Position


class BrokerAdapterError(RuntimeError):
    """Base error for broker adapter failures."""


class BrokerAdapter(ABC):
    """Common contract for paper/live broker integrations."""

    broker: str

    @abstractmethod
    def connect(self, config: BrokerConnectionConfig) -> BrokerHealth:
        """Connect to the broker gateway or API."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection."""

    @abstractmethod
    def healthcheck(self) -> BrokerHealth:
        """Return the current connection/account health."""

    @abstractmethod
    def get_account_summary(self) -> AccountSummary:
        """Fetch account-level values such as NAV and buying power."""

    @abstractmethod
    def get_cash_balances(self) -> Sequence[CashBalance]:
        """Fetch cash balances by currency."""

    @abstractmethod
    def get_positions(self) -> Sequence[Position]:
        """Fetch current broker positions."""

    @abstractmethod
    def get_open_orders(self) -> Sequence[OrderReceipt]:
        """Fetch currently open broker orders."""

    @abstractmethod
    def place_order(self, order: Order) -> OrderReceipt:
        """Submit an order to the broker."""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> CancelResult:
        """Cancel an open order by broker order id."""

    @abstractmethod
    def get_executions(self) -> Sequence[ExecutionReport]:
        """Fetch recent executions/fills."""

    def get_snapshot(self) -> BrokerSnapshot:
        """Build a point-in-time account snapshot from primitive calls."""

        account = self.get_account_summary()
        positions = tuple(self.get_positions())
        cash_balances = tuple(self.get_cash_balances())
        open_orders = tuple(self.get_open_orders())
        return BrokerSnapshot(
            broker=self.broker,
            account=account,
            positions=positions,
            cash_balances=cash_balances,
            open_orders=open_orders,
        )
