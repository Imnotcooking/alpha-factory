"""QMT connector-backed broker adapter.

This module deliberately does not import ``xtquant``. MiniQMT is expected to
run on a Windows host, with a small local connector service translating HTTP
requests into XtQuant calls. The adapter below is the OQP-side bridge.
"""

from __future__ import annotations

import json as json_lib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oqp.brokers.base import BrokerAdapter, BrokerAdapterError
from oqp.brokers.models import (
    AccountSummary,
    BrokerConnectionConfig,
    BrokerConnectionStatus,
    BrokerEnvironment,
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
from oqp.qmt_connector.security import qmt_json_body_bytes, qmt_signature_headers


QMT_BROKER = "qmt"
DEFAULT_QMT_CONNECTOR_URL = "http://127.0.0.1:58668"


@dataclass(frozen=True, slots=True)
class QMTConnectorRequest:
    method: str
    path: str
    params: dict[str, Any] | None = None
    json: dict[str, Any] | None = None


class QMTBrokerAdapter(BrokerAdapter):
    """Broker adapter that talks to the Windows-hosted QMT connector."""

    broker = QMT_BROKER

    def __init__(self, session: Any | None = None, timeout: float = 5.0) -> None:
        self.config: BrokerConnectionConfig | None = None
        self._session = session
        self.timeout = timeout
        self._last_health: BrokerHealth | None = None

    def connect(self, config: BrokerConnectionConfig) -> BrokerHealth:
        self.config = config
        try:
            payload = self._request("GET", "/health")
        except Exception as exc:
            self._last_health = BrokerHealth(
                broker=self.broker,
                status=BrokerConnectionStatus.ERROR,
                account_id=config.account_id,
                message=f"QMT connector is unavailable: {exc}",
                metadata=self._health_metadata(config),
            )
            return self._last_health

        status_text = str(
            payload.get("status")
            or payload.get("connector_status")
            or payload.get("qmt_status")
            or ""
        ).lower()
        connected = bool(payload.get("connected")) or status_text in {
            "ok",
            "pass",
            "ready",
            "connected",
        }
        self._last_health = BrokerHealth(
            broker=self.broker,
            status=(
                BrokerConnectionStatus.CONNECTED
                if connected
                else BrokerConnectionStatus.ERROR
            ),
            account_id=_text(payload.get("account_id")) or config.account_id,
            message=_text(payload.get("message"))
            or ("Connected to QMT connector." if connected else "QMT connector is not ready."),
            metadata={
                **self._health_metadata(config),
                "connector_mode": payload.get("mode"),
                "mini_qmt_connected": payload.get("mini_qmt_connected"),
                "session_id": payload.get("session_id"),
            },
        )
        return self._last_health

    def disconnect(self) -> None:
        self._last_health = None

    def healthcheck(self) -> BrokerHealth:
        if self.config is None:
            return BrokerHealth(
                broker=self.broker,
                status=BrokerConnectionStatus.DISCONNECTED,
                message="QMT adapter available; no connector profile has been used.",
                metadata={"implemented": True},
            )
        return self.connect(self.config)

    def get_account_summary(self) -> AccountSummary:
        self._require_config()
        payload = self._request(
            "GET",
            "/account",
            params=self._account_params(),
        )
        account = _payload_item(payload, "account")
        account_id = (
            _text(account.get("account_id"))
            or _text(account.get("fund_account"))
            or self.config.account_id
            or "unknown"
        )
        currency = (_text(account.get("currency")) or "CNY").upper()
        cash = _first_float(account, "cash", "available_cash", "available", "m_dAvailable")
        market_value = _first_float(account, "market_value", "m_dMarketValue")
        total_asset = _first_float(
            account,
            "net_liquidation",
            "total_asset",
            "balance",
            "m_dBalance",
        )
        frozen_cash = _first_float(account, "frozen_cash", "frozen", "m_dFrozenCash")

        return AccountSummary(
            broker=self.broker,
            account_id=account_id,
            currency=currency,
            net_liquidation=total_asset,
            cash=cash,
            buying_power=_first_float(account, "buying_power", "enable_balance"),
            gross_position_value=market_value,
            metadata={
                "source": "qmt_connector",
                "account_type": account.get("account_type") or self._account_type(),
                "frozen_cash": frozen_cash,
                "raw": account,
            },
        )

    def get_cash_balances(self) -> Sequence[CashBalance]:
        self._require_config()
        payload = self._request(
            "GET",
            "/account",
            params=self._account_params(),
        )
        account = _payload_item(payload, "account")
        rows = _payload_list(account, "cash_balances") or _payload_list(payload, "cash_balances")
        if not rows:
            cash = _first_float(account, "cash", "available_cash", "available", "m_dAvailable")
            if cash is None:
                return ()
            rows = ({"currency": account.get("currency") or "CNY", "cash": cash},)

        balances: list[CashBalance] = []
        for row in rows:
            cash = _first_float(row, "cash", "available_cash", "available")
            if cash is None:
                continue
            balances.append(
                CashBalance(
                    currency=(_text(row.get("currency")) or "CNY").upper(),
                    cash=cash,
                    settled_cash=_first_float(row, "settled_cash"),
                    buying_power=_first_float(row, "buying_power", "available"),
                    metadata={"raw": dict(row)},
                )
            )
        return tuple(balances)

    def get_positions(self) -> Sequence[Position]:
        self._require_config()
        payload = self._request(
            "GET",
            "/positions",
            params=self._account_params(),
        )
        rows = _payload_list(payload, "positions")
        return tuple(self._position_from_payload(row) for row in rows)

    def get_open_orders(self) -> Sequence[OrderReceipt]:
        self._require_config()
        payload = self._request(
            "GET",
            "/orders",
            params={**self._account_params(), "open_only": True},
        )
        rows = _payload_list(payload, "orders")
        return tuple(self._receipt_from_order_payload(row) for row in rows)

    def place_order(self, order: Order) -> OrderReceipt:
        self._require_config()
        if self.config.readonly:
            raise BrokerAdapterError("QMT order placement is blocked for read-only profiles.")
        if self.config.environment == BrokerEnvironment.LIVE and not self.config.metadata.get(
            "allow_qmt_live_trading"
        ):
            raise BrokerAdapterError("QMT live order placement is not enabled for this profile.")
        if order.order_type != OrderType.LIMIT:
            raise BrokerAdapterError("QMT skeleton only allows limit orders.")
        if order.limit_price is None or order.limit_price <= 0:
            raise BrokerAdapterError("QMT limit orders require a positive limit price.")
        if not _text(order.client_order_id):
            raise BrokerAdapterError("QMT write profiles require client_order_id for idempotency.")

        request = self._order_request_payload(order)
        try:
            payload = self._request("POST", "/submit_order", json=request)
        except Exception as exc:
            self._write_audit_event(
                "oqp_submit_order",
                "/submit_order",
                request_payload=request,
                error=str(exc),
            )
            raise
        self._write_audit_event(
            "oqp_submit_order",
            "/submit_order",
            request_payload=request,
            response_payload=payload,
        )
        receipt_payload = _payload_item(payload, "order")
        if not receipt_payload:
            receipt_payload = dict(payload)
        receipt_payload.setdefault("symbol", order.instrument.symbol)
        receipt_payload.setdefault("asset_class", order.instrument.asset_class.value)
        receipt_payload.setdefault("side", order.side.value)
        receipt_payload.setdefault("quantity", order.quantity)
        receipt_payload.setdefault("order_type", order.order_type.value)
        receipt_payload.setdefault("limit_price", order.limit_price)

        receipt = self._receipt_from_order_payload(receipt_payload, fallback_order=order)
        if receipt.status == OrderStatus.DRAFT:
            receipt = OrderReceipt(
                order=receipt.order,
                status=OrderStatus.SUBMITTED,
                broker_order_id=receipt.broker_order_id,
                client_order_id=receipt.client_order_id,
                submitted_at=receipt.submitted_at,
                message=receipt.message or "submitted",
                metadata=receipt.metadata,
            )
        return receipt

    def cancel_order(self, broker_order_id: str) -> CancelResult:
        self._require_config()
        if self.config.readonly:
            raise BrokerAdapterError("QMT cancellation is blocked for read-only profiles.")
        request_payload = {**self._account_params(), "broker_order_id": broker_order_id}
        try:
            payload = self._request(
                "POST",
                "/cancel_order",
                json=request_payload,
            )
        except Exception as exc:
            self._write_audit_event(
                "oqp_cancel_order",
                "/cancel_order",
                request_payload=request_payload,
                error=str(exc),
            )
            raise
        self._write_audit_event(
            "oqp_cancel_order",
            "/cancel_order",
            request_payload=request_payload,
            response_payload=payload,
        )
        cancelled = bool(payload.get("cancelled")) or payload.get("result") in {0, "0", "ok"}
        return CancelResult(
            broker_order_id=broker_order_id,
            cancelled=cancelled,
            status=OrderStatus.CANCELLED if cancelled else None,
            message=_text(payload.get("message")) or _text(payload.get("status_msg")),
            metadata={"raw": payload},
        )

    def get_executions(self) -> Sequence[ExecutionReport]:
        self._require_config()
        payload = self._request(
            "GET",
            "/trades",
            params=self._account_params(),
        )
        rows = _payload_list(payload, "trades")
        reports: list[ExecutionReport] = []
        for row in rows:
            order = self._order_from_payload(row, status=OrderStatus.FILLED)
            quantity = _first_float(row, "filled_quantity", "traded_volume", "quantity")
            average_price = _first_float(row, "average_price", "traded_price", "price")
            if quantity is None or quantity <= 0 or average_price is None or average_price <= 0:
                continue
            reports.append(
                ExecutionReport(
                    order=order,
                    filled_quantity=quantity,
                    average_price=average_price,
                    broker_execution_id=_text(row.get("trade_id"))
                    or _text(row.get("traded_id")),
                    broker_order_id=_text(row.get("broker_order_id"))
                    or _text(row.get("order_id"))
                    or _text(row.get("order_sysid")),
                    commission=_first_float(row, "commission", "used_commission"),
                    currency=(_text(row.get("currency")) or order.instrument.currency).upper(),
                    metadata={"raw": dict(row)},
                )
            )
        return tuple(reports)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = QMTConnectorRequest(method=method, path=path, params=params, json=json)
        session = self._session
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise BrokerAdapterError("requests is required for the QMT connector.") from exc
            session = requests

        url = self._url(path)
        headers = {"Accept": "application/json"}
        token = self._api_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        body_bytes = qmt_json_body_bytes(request.json)
        signing_secret = self._signing_secret()
        if signing_secret:
            headers.update(
                qmt_signature_headers(
                    signing_secret,
                    request.method,
                    path,
                    params=request.params,
                    body=body_bytes,
                )
            )

        request_kwargs: dict[str, Any] = {
            "params": request.params,
            "headers": headers,
            "timeout": self.timeout,
        }
        if request.json is not None and signing_secret:
            headers["Content-Type"] = "application/json"
            request_kwargs["data"] = body_bytes
        else:
            request_kwargs["json"] = request.json

        response = session.request(request.method, url, **request_kwargs)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        payload = response.json() if hasattr(response, "json") else response
        if not isinstance(payload, Mapping):
            raise BrokerAdapterError(f"QMT connector returned non-object payload for {path}.")
        error = payload.get("error")
        if error:
            raise BrokerAdapterError(str(error))
        return dict(payload)

    def _position_from_payload(self, row: Mapping[str, Any]) -> Position:
        instrument = self._instrument_from_payload(row)
        quantity = _signed_quantity(
            _first_float(row, "quantity", "volume", "position") or 0.0,
            row.get("direction"),
        )
        market_price = _first_float(row, "market_price", "last_price", "price")
        average_cost = _first_float(row, "average_cost", "avg_price", "open_price") or 0.0
        return Position(
            instrument=instrument,
            quantity=quantity,
            average_cost=average_cost,
            market_price=market_price,
            account_id=_text(row.get("account_id")) or (self.config.account_id if self.config else None),
            broker=self.broker,
            metadata={
                "market_value": _first_float(row, "market_value", "instrument_value"),
                "unrealized_pnl": _first_float(
                    row,
                    "unrealized_pnl",
                    "float_profit",
                    "position_profit",
                ),
                "available_quantity": _first_float(row, "can_use_volume", "can_close_vol"),
                "frozen_quantity": _first_float(row, "frozen_volume"),
                "direction": row.get("direction"),
                "offset_flag": row.get("offset_flag"),
                "used_margin": _first_float(row, "used_margin"),
                "raw": dict(row),
            },
        )

    def _instrument_from_payload(self, row: Mapping[str, Any]) -> Instrument:
        symbol = (
            _text(row.get("symbol"))
            or _text(row.get("stock_code"))
            or _text(row.get("instrument_id"))
            or "UNKNOWN"
        )
        asset_class = _asset_class(row.get("asset_class") or row.get("asset_type"))
        multiplier = _first_float(row, "multiplier", "volume_multiple") or 1.0
        return Instrument(
            symbol=symbol,
            asset_class=asset_class,
            exchange=_text(row.get("exchange")) or _text(row.get("exchange_id")),
            currency=(_text(row.get("currency")) or "CNY").upper(),
            broker_symbol=_text(row.get("broker_symbol")) or symbol,
            multiplier=multiplier,
            metadata={
                "qmt_account_type": row.get("account_type") or self._account_type(),
                "product_id": row.get("product_id"),
                "instrument_name": row.get("instrument_name"),
                "direction": row.get("direction"),
                "offset_flag": row.get("offset_flag"),
            },
        )

    def _receipt_from_order_payload(
        self,
        row: Mapping[str, Any],
        *,
        fallback_order: Order | None = None,
    ) -> OrderReceipt:
        order = fallback_order or self._order_from_payload(row)
        return OrderReceipt(
            order=order,
            status=_order_status(row.get("status") or row.get("order_status")),
            broker_order_id=_text(row.get("broker_order_id"))
            or _text(row.get("order_id"))
            or _text(row.get("order_sysid")),
            client_order_id=_text(row.get("client_order_id"))
            or _text(row.get("order_remark")),
            message=_text(row.get("message")) or _text(row.get("status_msg")),
            metadata={
                "order_sysid": row.get("order_sysid"),
                "strategy_name": row.get("strategy_name"),
                "qmt_order_type": row.get("qmt_order_type") or row.get("order_type"),
                "raw": dict(row),
            },
        )

    def _order_from_payload(
        self,
        row: Mapping[str, Any],
        *,
        status: OrderStatus | None = None,
    ) -> Order:
        instrument = self._instrument_from_payload(row)
        quantity = abs(_first_float(row, "quantity", "order_volume", "volume") or 0.0)
        limit_price = _first_float(row, "limit_price", "price")
        order_type = _order_type(row.get("order_type"), limit_price=limit_price)
        return Order(
            instrument=instrument,
            side=_order_side(row.get("side") or row.get("order_side") or row.get("order_type")),
            quantity=quantity or 1.0,
            order_type=order_type,
            limit_price=limit_price if order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} else None,
            stop_price=_first_float(row, "stop_price"),
            time_in_force=_text(row.get("time_in_force")) or "DAY",
            status=status or _order_status(row.get("status") or row.get("order_status")),
            strategy_id=_text(row.get("strategy_id")) or _text(row.get("strategy_name")),
            account_id=_text(row.get("account_id")) or (self.config.account_id if self.config else None),
            broker=self.broker,
            client_order_id=_text(row.get("client_order_id")) or _text(row.get("order_remark")),
            metadata={
                "qmt_order_type": row.get("qmt_order_type") or row.get("order_type"),
                "direction": row.get("direction"),
                "offset_flag": row.get("offset_flag"),
            },
        )

    def _order_request_payload(self, order: Order) -> dict[str, Any]:
        instrument = order.instrument
        metadata = dict(order.metadata or {})
        return {
            **self._account_params(),
            "symbol": instrument.broker_symbol or instrument.symbol,
            "asset_class": instrument.asset_class.value,
            "exchange": instrument.exchange,
            "currency": instrument.currency,
            "multiplier": instrument.multiplier,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "limit_price": order.limit_price,
            "time_in_force": order.time_in_force,
            "strategy_id": order.strategy_id,
            "client_order_id": order.client_order_id,
            "qmt_order_type": metadata.get("qmt_order_type"),
            "price_type": metadata.get("price_type", "FIX_PRICE"),
            "direction": metadata.get("direction"),
            "offset_flag": metadata.get("offset_flag"),
            "order_remark": metadata.get("order_remark") or order.client_order_id,
        }

    def _account_params(self) -> dict[str, Any]:
        return {
            "account_id": self.config.account_id if self.config else None,
            "account_type": self._account_type(),
        }

    def _account_type(self) -> str:
        if self.config is None:
            return "STOCK"
        return str(self.config.metadata.get("account_type") or "STOCK").upper()

    def _api_token(self) -> str | None:
        if self.config is None:
            return None
        return _text(self.config.metadata.get("api_token"))

    def _signing_secret(self) -> str | None:
        if self.config is None:
            return None
        return _text(self.config.metadata.get("request_signing_secret"))

    def _audit_log_path(self) -> Path | None:
        if self.config is None:
            return None
        path = _text(self.config.metadata.get("audit_log_path"))
        return Path(path) if path else None

    def _write_audit_event(
        self,
        event_type: str,
        endpoint: str,
        *,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        path = self._audit_log_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self.config
        event = {
            "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "event": event_type,
            "endpoint": endpoint,
            "broker": self.broker,
            "profile": config.metadata.get("profile") if config else None,
            "environment": config.environment.value if config else None,
            "account_id": config.account_id if config else None,
            "request": dict(request_payload),
            "response": dict(response_payload or {}),
            "error": error,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json_lib.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _url(self, path: str) -> str:
        base_url = DEFAULT_QMT_CONNECTOR_URL
        if self.config is not None:
            base_url = str(self.config.metadata.get("connector_url") or base_url)
        return base_url.rstrip("/") + "/" + path.lstrip("/")

    def _require_config(self) -> None:
        if self.config is None:
            raise BrokerAdapterError("QMT adapter has not been connected with a profile.")

    def _health_metadata(self, config: BrokerConnectionConfig) -> dict[str, Any]:
        return {
            "implemented": True,
            "connector_url": config.metadata.get("connector_url"),
            "environment": config.environment.value,
            "readonly": config.readonly,
            "profile": config.metadata.get("profile"),
            "account_type": config.metadata.get("account_type"),
        }


def _payload_item(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else dict(payload)


def _payload_list(payload: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if value is None and key == "positions":
        value = payload.get("data")
    if value is None and key == "orders":
        value = payload.get("data")
    if value is None and key == "trades":
        value = payload.get("data")
    if isinstance(value, Mapping):
        value = value.get("rows") or value.get("items")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _asset_class(value: Any) -> AssetClass:
    text = str(value or "").strip().lower()
    if text in {"stock", "stk", "equity", "a_share", "ashare"}:
        return AssetClass.EQUITY
    if text in {"fund", "etf"}:
        return AssetClass.ETF
    if text in {"future", "futures", "fut"}:
        return AssetClass.FUTURE
    if text in {"option", "options", "opt", "stock_option", "future_option"}:
        return AssetClass.OPTION
    if text in {"cash", "money"}:
        return AssetClass.CASH
    return AssetClass.EQUITY


def _order_side(value: Any) -> OrderSide:
    text = str(value or "").strip().lower()
    if text in {"sell", "short", "stock_sell", "24", "49"}:
        return OrderSide.SELL
    return OrderSide.BUY


def _order_type(value: Any, *, limit_price: float | None) -> OrderType:
    text = str(value or "").strip().lower()
    if text in {"market", "mkt"}:
        return OrderType.MARKET
    return OrderType.LIMIT if limit_price is not None else OrderType.MARKET


def _order_status(value: Any) -> OrderStatus:
    text = str(value or "").strip().lower()
    if text in {"56", "filled", "succeeded", "all_traded"}:
        return OrderStatus.FILLED
    if text in {"55", "partially_filled", "part_succ", "partial"}:
        return OrderStatus.PARTIALLY_FILLED
    if text in {"54", "cancelled", "canceled"}:
        return OrderStatus.CANCELLED
    if text in {"57", "rejected", "junk", "error"}:
        return OrderStatus.REJECTED
    if text in {"48", "49", "50", "51", "52", "submitted", "reported", "pending"}:
        return OrderStatus.SUBMITTED
    return OrderStatus.DRAFT


def _signed_quantity(quantity: float, direction: Any) -> float:
    text = str(direction or "").strip().lower()
    if text in {"short", "sell", "49", "direction_flag_short"}:
        return -abs(quantity)
    return float(quantity)


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, "", "nan"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
