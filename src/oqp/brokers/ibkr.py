"""Interactive Brokers read-only adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from oqp.brokers.base import BrokerAdapter, BrokerAdapterError
from oqp.brokers.models import (
    AccountSummary,
    BrokerConnectionConfig,
    BrokerEnvironment,
    BrokerConnectionStatus,
    BrokerHealth,
    CashBalance,
    CancelResult,
    ExecutionReport,
    OrderReceipt,
)
from oqp.domain import (
    AssetClass,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


@dataclass(frozen=True, slots=True)
class IBKRReadOnlyPortfolioSnapshot:
    """Middle-office friendly IBKR account extract."""

    health: BrokerHealth
    position_rows: tuple[dict[str, Any], ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class IBKRBrokerAdapter(BrokerAdapter):
    """Read-only adapter for a locally running TWS or IB Gateway API session."""

    broker = "ibkr"

    def __init__(self, connect_timeout: float = 4.0) -> None:
        self.config: BrokerConnectionConfig | None = None
        self.connect_timeout = connect_timeout
        self._ib: Any | None = None
        self._last_health: BrokerHealth | None = None

    def connect(self, config: BrokerConnectionConfig) -> BrokerHealth:
        self.disconnect()
        self.config = config

        ib_class = self._load_ib_class()
        if ib_class is None:
            self._last_health = BrokerHealth(
                broker=self.broker,
                status=BrokerConnectionStatus.ERROR,
                account_id=config.account_id,
                message=(
                    "ib_insync is not installed in this Python environment. "
                    "Install dependencies in the Streamlit runtime before connecting."
                ),
                metadata={
                    "environment": config.environment.value,
                    "readonly": config.readonly,
                    "implemented": True,
                },
            )
            return self._last_health

        self._ib = ib_class()
        try:
            self._ib.connect(
                config.host,
                config.port,
                clientId=config.client_id,
                timeout=self.connect_timeout,
                readonly=config.readonly,
                account=config.account_id or "",
            )
        except Exception as exc:
            self.disconnect()
            self._last_health = BrokerHealth(
                broker=self.broker,
                status=BrokerConnectionStatus.ERROR,
                account_id=config.account_id,
                message=(
                    "Could not connect to IBKR API. Log in to TWS or IB Gateway "
                    "locally and confirm API socket access is enabled."
                ),
                metadata={
                    "environment": config.environment.value,
                    "host": config.host,
                    "port": config.port,
                    "client_id": config.client_id,
                    "readonly": config.readonly,
                    "implemented": True,
                    "error": str(exc),
                },
            )
            return self._last_health

        account_id = config.account_id or self._first_account_id()
        self._last_health = BrokerHealth(
            broker=self.broker,
            status=BrokerConnectionStatus.CONNECTED,
            account_id=account_id,
            message="Connected to IBKR API in read-only mode.",
            metadata={
                "environment": config.environment.value,
                "host": config.host,
                "port": config.port,
                "client_id": config.client_id,
                "readonly": config.readonly,
                "implemented": True,
            },
        )
        return self._last_health

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                if self._ib.isConnected():
                    self._ib.disconnect()
            except Exception:
                pass
        self._ib = None

    def healthcheck(self) -> BrokerHealth:
        if self._is_connected():
            account_id = (
                self.config.account_id
                if self.config and self.config.account_id
                else self._first_account_id()
            )
            return BrokerHealth(
                broker=self.broker,
                status=BrokerConnectionStatus.CONNECTED,
                account_id=account_id,
                message="Connected to IBKR API in read-only mode.",
                metadata={"implemented": True, "readonly": True},
            )

        return self._last_health or BrokerHealth(
            broker=self.broker,
            status=BrokerConnectionStatus.DISCONNECTED,
            account_id=self.config.account_id if self.config else None,
            message="IBKR adapter available; not connected to TWS or IB Gateway.",
            metadata={"implemented": True, "readonly": True},
        )

    def get_account_summary(self) -> AccountSummary:
        self._require_connection()
        summary = list(self._ib.accountSummary())
        account_id = self.config.account_id if self.config else None
        account_id = account_id or self._first_item_attr(summary, "account") or "unknown"

        net_liquidation, currency = self._summary_value(
            summary, ("NetLiquidation",), with_currency=True
        )
        cash = self._summary_value(summary, ("TotalCashValue",))
        buying_power = self._summary_value(summary, ("BuyingPower", "AvailableFunds"))
        available_funds = self._summary_value(summary, ("AvailableFunds",))
        excess_liquidity = self._summary_value(summary, ("ExcessLiquidity",))
        gross_position_value = self._summary_value(summary, ("GrossPositionValue",))

        return AccountSummary(
            broker=self.broker,
            account_id=account_id,
            currency=currency or "USD",
            net_liquidation=net_liquidation,
            cash=cash,
            buying_power=buying_power,
            gross_position_value=gross_position_value,
            metadata={
                "available_funds": available_funds,
                "excess_liquidity": excess_liquidity,
            },
        )

    def get_cash_balances(self) -> Sequence[CashBalance]:
        self._require_connection()
        summary = list(self._ib.accountSummary())
        currencies = sorted(
            {
                item.currency
                for item in summary
                if getattr(item, "currency", None)
                and item.currency not in {"BASE", ""}
                and item.tag in {"TotalCashValue", "SettledCash", "BuyingPower"}
            }
        )
        if not currencies:
            currencies = ["USD"]

        balances: list[CashBalance] = []
        for currency in currencies:
            cash = self._summary_value(summary, ("TotalCashValue",), currency=currency)
            settled_cash = self._summary_value(summary, ("SettledCash",), currency=currency)
            buying_power = self._summary_value(
                summary, ("BuyingPower", "AvailableFunds"), currency=currency
            )
            if cash is None and settled_cash is None and buying_power is None:
                continue

            balances.append(
                CashBalance(
                    currency=currency,
                    cash=cash or 0.0,
                    settled_cash=settled_cash,
                    buying_power=buying_power,
                )
            )
        return balances

    def get_positions(self) -> Sequence[Position]:
        self._require_connection()
        positions: list[Position] = []

        for item in self._ib.portfolio():
            quantity = self._float(getattr(item, "position", None))
            if quantity is None or quantity == 0:
                continue

            contract = item.contract
            if getattr(contract, "secType", "") == "CASH":
                continue

            instrument = self._instrument_from_contract(contract)
            market_price = self._float(getattr(item, "marketPrice", None))
            average_cost = self._normalize_average_cost(
                self._float(getattr(item, "averageCost", None)) or 0.0,
                instrument,
            )

            positions.append(
                Position(
                    instrument=instrument,
                    quantity=quantity,
                    average_cost=average_cost,
                    market_price=market_price if market_price and market_price > 0 else None,
                    account_id=getattr(item, "account", None),
                    broker=self.broker,
                    metadata={
                        "market_value": self._float(getattr(item, "marketValue", None)),
                        "unrealized_pnl": self._float(
                            getattr(item, "unrealizedPNL", None)
                        ),
                        "realized_pnl": self._float(getattr(item, "realizedPNL", None)),
                    },
                )
            )

        return positions

    def get_open_orders(self) -> Sequence[OrderReceipt]:
        self._require_connection()
        receipts: list[OrderReceipt] = []

        for trade in self._ib.openTrades():
            order = getattr(trade, "order", None)
            contract = getattr(trade, "contract", None)
            status = getattr(trade, "orderStatus", None)
            if order is None or contract is None:
                continue

            receipt = self._order_receipt_from_trade(order, contract, status)
            if receipt is not None:
                receipts.append(receipt)

        return receipts

    def place_order(self, order: Order) -> OrderReceipt:
        self._require_connection()
        if self.config is None:
            raise BrokerAdapterError("IBKR order placement requires a broker config.")
        if self.config.environment != BrokerEnvironment.PAPER:
            raise BrokerAdapterError("IBKR order placement is only enabled for paper profiles.")
        if self.config.readonly:
            raise BrokerAdapterError("IBKR order placement is blocked for read-only profiles.")
        if order.instrument.asset_class not in {AssetClass.EQUITY, AssetClass.ETF}:
            raise BrokerAdapterError("IBKR paper order placement currently supports equities and ETFs only.")
        if order.order_type != OrderType.LIMIT:
            raise BrokerAdapterError("IBKR paper order placement currently requires limit orders.")
        if order.limit_price is None or order.limit_price <= 0:
            raise BrokerAdapterError("IBKR paper limit orders require a positive limit price.")

        try:
            from ib_insync import LimitOrder, Stock
        except ImportError as exc:
            raise BrokerAdapterError("ib_insync is required for IBKR paper order placement.") from exc

        symbol = order.instrument.broker_symbol or order.instrument.symbol
        contract = Stock(
            symbol,
            order.instrument.exchange or "SMART",
            order.instrument.currency,
        )
        try:
            qualified = self._ib.qualifyContracts(contract)
            if qualified:
                contract = qualified[0]
        except Exception:
            # IBKR can still accept a SMART stock contract without qualification.
            pass

        ib_order = LimitOrder(
            order.side.value.upper(),
            float(order.quantity),
            float(order.limit_price),
            tif=order.time_in_force or "DAY",
        )
        account_id = self.config.account_id or order.account_id
        if account_id:
            ib_order.account = account_id
        if order.client_order_id:
            ib_order.orderRef = order.client_order_id
        elif order.strategy_id:
            ib_order.orderRef = f"oqp:{order.strategy_id}"

        try:
            trade = self._ib.placeOrder(contract, ib_order)
            try:
                self._ib.sleep(0.5)
            except Exception:
                pass
        except Exception as exc:
            raise BrokerAdapterError(f"IBKR paper order placement failed: {exc}") from exc

        status = getattr(trade, "orderStatus", None)
        broker_order = getattr(trade, "order", ib_order)
        broker_order_id = str(getattr(broker_order, "orderId", "")) or None
        status_text = getattr(status, "status", None)
        receipt_status = self._order_status(status_text)
        if receipt_status == OrderStatus.DRAFT:
            receipt_status = OrderStatus.SUBMITTED

        return OrderReceipt(
            order=order,
            status=receipt_status,
            broker_order_id=broker_order_id,
            client_order_id=order.client_order_id,
            message=status_text or "submitted",
            metadata={
                "profile": self.config.metadata.get("profile"),
                "environment": self.config.environment.value,
                "readonly": self.config.readonly,
                "contract_symbol": getattr(contract, "localSymbol", None)
                or getattr(contract, "symbol", None),
                "order_ref": getattr(broker_order, "orderRef", None),
                "perm_id": getattr(status, "permId", None),
            },
        )

    def cancel_order(self, broker_order_id: str) -> CancelResult:
        raise BrokerAdapterError(
            "Order cancellation is disabled in the read-only IBKR adapter path."
        )

    def get_executions(self) -> Sequence[ExecutionReport]:
        self._require_connection()
        return []

    def _load_ib_class(self) -> Any | None:
        self._ensure_event_loop()
        try:
            from ib_insync import IB, util
        except ImportError:
            return None

        try:
            util.patchAsyncio()
        except Exception:
            pass
        return IB

    @staticmethod
    def _ensure_event_loop() -> None:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    def _require_connection(self) -> None:
        if not self._is_connected():
            raise BrokerAdapterError("IBKR is not connected.")

    def _is_connected(self) -> bool:
        try:
            return bool(self._ib is not None and self._ib.isConnected())
        except Exception:
            return False

    def _first_account_id(self) -> str | None:
        if not self._is_connected():
            return None
        try:
            accounts = list(self._ib.managedAccounts())
        except Exception:
            return None
        return accounts[0] if accounts else None

    def _instrument_from_contract(self, contract: Any) -> Instrument:
        sec_type = getattr(contract, "secType", "")
        asset_class = self._asset_class(sec_type)
        multiplier = self._contract_multiplier(contract, asset_class)
        symbol = (
            getattr(contract, "localSymbol", None)
            or getattr(contract, "symbol", None)
            or str(getattr(contract, "conId", "UNKNOWN"))
        )

        return Instrument(
            symbol=symbol,
            asset_class=asset_class,
            exchange=getattr(contract, "primaryExchange", None)
            or getattr(contract, "exchange", None),
            currency=getattr(contract, "currency", None) or "USD",
            broker_symbol=getattr(contract, "localSymbol", None),
            multiplier=multiplier,
            metadata={
                "con_id": getattr(contract, "conId", None),
                "sec_type": sec_type,
                "ib_symbol": getattr(contract, "symbol", None),
                "expiry": getattr(contract, "lastTradeDateOrContractMonth", None),
                "strike": self._float(getattr(contract, "strike", None)),
                "right": getattr(contract, "right", None),
                "trading_class": getattr(contract, "tradingClass", None),
            },
        )

    def _order_receipt_from_trade(
        self,
        ib_order: Any,
        contract: Any,
        order_status: Any,
    ) -> OrderReceipt | None:
        quantity = self._float(getattr(ib_order, "totalQuantity", None))
        if quantity is None or quantity <= 0:
            return None

        order_type = self._order_type(getattr(ib_order, "orderType", "MKT"))
        limit_price = self._positive_float(getattr(ib_order, "lmtPrice", None))
        stop_price = self._positive_float(getattr(ib_order, "auxPrice", None))

        if order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and limit_price is None:
            return None
        if order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and stop_price is None:
            return None

        domain_order = Order(
            instrument=self._instrument_from_contract(contract),
            side=self._order_side(getattr(ib_order, "action", "BUY")),
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=getattr(ib_order, "tif", None) or "DAY",
            status=self._order_status(getattr(order_status, "status", None)),
            account_id=getattr(ib_order, "account", None) or None,
            broker=self.broker,
        )

        return OrderReceipt(
            order=domain_order,
            status=domain_order.status,
            broker_order_id=str(getattr(ib_order, "orderId", "")) or None,
            message=getattr(order_status, "status", None),
        )

    def _summary_value(
        self,
        summary: Iterable[Any],
        tags: tuple[str, ...],
        *,
        currency: str | None = None,
        with_currency: bool = False,
    ) -> float | tuple[float | None, str | None] | None:
        items = list(summary)
        currency_order = (currency,) if currency else ("USD", "BASE", "")

        for preferred_currency in currency_order:
            for item in items:
                if item.tag not in tags:
                    continue
                if preferred_currency and item.currency != preferred_currency:
                    continue
                value = self._float(item.value)
                if value is not None:
                    if with_currency:
                        return value, item.currency if item.currency != "BASE" else "USD"
                    return value

        if with_currency:
            return None, None
        return None

    def _first_item_attr(self, items: Iterable[Any], attr: str) -> str | None:
        for item in items:
            value = getattr(item, attr, None)
            if value:
                return str(value)
        return None

    def _asset_class(self, sec_type: str) -> AssetClass:
        return {
            "STK": AssetClass.EQUITY,
            "OPT": AssetClass.OPTION,
            "FUT": AssetClass.FUTURE,
            "CASH": AssetClass.CASH,
            "CFD": AssetClass.EQUITY,
            "CRYPTO": AssetClass.CRYPTO,
        }.get(sec_type, AssetClass.EQUITY)

    def _contract_multiplier(self, contract: Any, asset_class: AssetClass) -> float:
        raw_multiplier = getattr(contract, "multiplier", None)
        multiplier = self._float(raw_multiplier)
        if multiplier and multiplier > 0:
            return multiplier
        if asset_class == AssetClass.OPTION:
            return 100.0
        return 1.0

    def _normalize_average_cost(self, average_cost: float, instrument: Instrument) -> float:
        if instrument.asset_class == AssetClass.OPTION and instrument.multiplier:
            return average_cost / instrument.multiplier
        return average_cost

    def _order_side(self, action: str) -> OrderSide:
        return OrderSide.SELL if str(action).upper() == "SELL" else OrderSide.BUY

    def _order_type(self, order_type: str) -> OrderType:
        normalized = str(order_type).upper().replace(" ", "_")
        return {
            "LMT": OrderType.LIMIT,
            "MKT": OrderType.MARKET,
            "STP": OrderType.STOP,
            "STP_LMT": OrderType.STOP_LIMIT,
        }.get(normalized, OrderType.MARKET)

    def _order_status(self, status: str | None) -> OrderStatus:
        normalized = (status or "").lower()
        if normalized in {"filled"}:
            return OrderStatus.FILLED
        if normalized in {"cancelled", "apicancelled"}:
            return OrderStatus.CANCELLED
        if normalized in {"inactive", "rejected"}:
            return OrderStatus.REJECTED
        if normalized in {"submitted", "presubmitted", "pendingsubmit"}:
            return OrderStatus.SUBMITTED
        return OrderStatus.DRAFT

    def _positive_float(self, value: Any) -> float | None:
        parsed = self._float(value)
        return parsed if parsed and parsed > 0 else None

    def _float(self, value: Any) -> float | None:
        if value in (None, "", "nan"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def fetch_ibkr_readonly_portfolio_snapshot(
    config: BrokerConnectionConfig,
    *,
    adapter: BrokerAdapter | None = None,
) -> IBKRReadOnlyPortfolioSnapshot:
    """Connect read-only and return the IBKR extract used by portfolio ETL."""

    broker = adapter or IBKRBrokerAdapter()
    health = broker.connect(config)
    if health.status != BrokerConnectionStatus.CONNECTED:
        return IBKRReadOnlyPortfolioSnapshot(
            health=health,
            error=health.message,
        )

    try:
        account = broker.get_account_summary()
        positions = tuple(broker.get_positions())
        return IBKRReadOnlyPortfolioSnapshot(
            health=health,
            position_rows=tuple(
                ibkr_position_to_middle_office_row(position)
                for position in positions
            ),
            metrics=ibkr_account_summary_to_middle_office_metrics(account),
        )
    except Exception as exc:
        return IBKRReadOnlyPortfolioSnapshot(
            health=health,
            error=str(exc),
        )
    finally:
        broker.disconnect()


def ibkr_account_summary_to_middle_office_metrics(
    account: AccountSummary,
) -> dict[str, float]:
    """Convert generic account summary fields to the legacy metrics JSON."""

    return {
        "Total_NAV_USD": _float_or_zero(account.net_liquidation),
        "Available_Cash_USD": _float_or_zero(account.cash),
        "Margin_Buffer_USD": _float_or_zero(
            account.metadata.get("excess_liquidity")
            or account.metadata.get("available_funds")
            or account.buying_power
        ),
    }


def ibkr_position_to_middle_office_row(position: Position) -> dict[str, Any]:
    """Convert a generic IBKR position into the shared live-position shape."""

    instrument = position.instrument
    market_price = position.market_price
    if market_price is None or market_price == 0:
        market_price = position.average_cost

    return {
        "Ticker": instrument.broker_symbol or instrument.symbol,
        "Shares": float(position.quantity),
        "AvgPrice": round(float(position.average_cost), 4),
        "Broker_Price": float(market_price),
        "Broker_PnL": _float_or_zero(position.metadata.get("unrealized_pnl")),
        "Currency": instrument.currency,
        "AssetType": _middle_office_asset_type(instrument.asset_class),
        "Multiplier": float(instrument.multiplier),
        "Broker": "IBKR Live",
    }


def _middle_office_asset_type(asset_class: AssetClass) -> str:
    if asset_class == AssetClass.OPTION:
        return "Option"
    if asset_class == AssetClass.FUTURE:
        return "Future"
    if asset_class == AssetClass.CASH:
        return "Cash"
    if asset_class == AssetClass.CRYPTO:
        return "Crypto"
    return "Equity"


def _float_or_zero(value: Any) -> float:
    try:
        if value in (None, "", "nan"):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
