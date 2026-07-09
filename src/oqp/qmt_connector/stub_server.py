"""Stdlib-only QMT connector stub.

The real connector will run on Windows beside MiniQMT and translate this same
HTTP contract into ``xtquant`` calls. This stub keeps development unblocked
before QMT registration and exercises the same security boundary we want in
production: private bind, bearer auth, HMAC request signing, connector-side
risk checks, idempotency, and durable write audit logs.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from oqp.qmt_connector.security import (
    DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
    verify_qmt_signature,
)


READONLY_MODE = "readonly"
PAPER_SUBMIT_MODE = "paper_submit"
VALID_MODES = (READONLY_MODE, PAPER_SUBMIT_MODE)
DEFAULT_READONLY_PORT = 58668
DEFAULT_SUBMIT_PORT = 58669
WRITE_ENDPOINTS = {"/submit_order", "/cancel_order"}
TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")


@dataclass(slots=True)
class ConnectorRiskPolicy:
    """Connector-side limits that must pass before any submit is accepted."""

    allowed_symbols: tuple[str, ...] = ()
    allowed_asset_classes: tuple[str, ...] = ()
    allowed_account_types: tuple[str, ...] = ()
    max_quantity: float | None = None
    max_notional: float | None = None
    min_limit_price: float | None = None
    max_limit_price: float | None = None

    def __post_init__(self) -> None:
        self.allowed_symbols = tuple(
            _text(symbol).upper() for symbol in self.allowed_symbols if _text(symbol)
        )
        self.allowed_asset_classes = tuple(
            _text(asset_class).lower()
            for asset_class in self.allowed_asset_classes
            if _text(asset_class)
        )
        self.allowed_account_types = tuple(
            _text(account_type).upper()
            for account_type in self.allowed_account_types
            if _text(account_type)
        )

    def startup_error(self, *, account_type: str, mode: str) -> str | None:
        if mode != PAPER_SUBMIT_MODE:
            return None
        missing: list[str] = []
        if not self.allowed_symbols:
            missing.append("allowed symbol")
        if not self.allowed_asset_classes:
            missing.append("allowed asset class")
        if self.max_quantity is None or self.max_quantity <= 0:
            missing.append("positive max quantity")
        if self.max_notional is None or self.max_notional <= 0:
            missing.append("positive max notional")
        if self.min_limit_price is None or self.min_limit_price <= 0:
            missing.append("positive min limit price")
        if self.max_limit_price is None or self.max_limit_price <= 0:
            missing.append("positive max limit price")
        if (
            self.min_limit_price is not None
            and self.max_limit_price is not None
            and self.min_limit_price >= self.max_limit_price
        ):
            missing.append("valid min/max limit price band")
        account_types = self.allowed_account_types or (account_type.upper(),)
        if account_type.upper() not in account_types:
            missing.append("matching account type allowlist")
        if not missing:
            return None
        return "paper_submit mode requires connector risk policy: " + ", ".join(missing)

    def violation(self, request: Mapping[str, Any], state: StubConnectorState) -> str | None:
        symbol = (_text(request.get("symbol")) or _text(request.get("stock_code")) or "").upper()
        asset_class = (_text(request.get("asset_class")) or "equity").lower()
        account_id = _text(request.get("account_id")) or state.account_id
        account_type = (_text(request.get("account_type")) or state.account_type).upper()
        quantity = _float_value(request.get("quantity")) or _float_value(request.get("order_volume"))
        limit_price = _float_value(request.get("limit_price")) or _float_value(request.get("price"))
        multiplier = _float_value(request.get("multiplier")) or 1.0
        account_types = self.allowed_account_types or (state.account_type,)

        if account_id != state.account_id:
            return f"account_id {account_id} is not the connector account"
        if account_type not in account_types:
            return f"account_type {account_type} is not allowed"
        if symbol not in self.allowed_symbols:
            return f"symbol {symbol} is not allowlisted"
        if asset_class not in self.allowed_asset_classes:
            return f"asset_class {asset_class} is not allowlisted"
        if quantity is None or quantity <= 0:
            return "quantity must be positive"
        if self.max_quantity is not None and quantity > self.max_quantity:
            return f"quantity {quantity:g} exceeds max {self.max_quantity:g}"
        if limit_price is None or limit_price <= 0:
            return "limit_price must be positive"
        if self.min_limit_price is not None and limit_price < self.min_limit_price:
            return f"limit_price {limit_price:g} is below min {self.min_limit_price:g}"
        if self.max_limit_price is not None and limit_price > self.max_limit_price:
            return f"limit_price {limit_price:g} exceeds max {self.max_limit_price:g}"
        notional = quantity * limit_price * multiplier
        if self.max_notional is not None and notional > self.max_notional:
            return f"notional {notional:g} exceeds max {self.max_notional:g}"
        return None


@dataclass(slots=True)
class StubConnectorState:
    """Mutable fake QMT state served by the stub connector."""

    account_id: str = "QMT123456"
    account_type: str = "STOCK"
    session_id: int = 880001
    mode: str = READONLY_MODE
    currency: str = "CNY"
    cash: float = 900_000.0
    frozen_cash: float = 0.0
    positions: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    risk_policy: ConnectorRiskPolicy = field(default_factory=ConnectorRiskPolicy)
    audit_log_path: Path | str | None = None
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    seen_nonces: set[str] = field(default_factory=set)
    started_at: str = field(default_factory=lambda: _utc_iso())
    next_order_id: int = 9001

    def __post_init__(self) -> None:
        self.account_id = self.account_id.strip()
        self.account_type = self.account_type.upper()
        self.mode = self.mode.lower()
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported QMT stub mode: {self.mode}")
        if self.audit_log_path is not None:
            self.audit_log_path = Path(self.audit_log_path)
        if not self.positions:
            self.positions.extend(_default_positions(self.account_id, self.account_type))

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "connected": True,
            "mini_qmt_connected": False,
            "mode": self.mode,
            "account_id": self.account_id,
            "account_type": self.account_type,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "message": "QMT connector stub is running; MiniQMT is not connected.",
        }

    def account_payload(self, query: Mapping[str, str]) -> dict[str, Any]:
        account_id = query.get("account_id") or self.account_id
        account_type = (query.get("account_type") or self.account_type).upper()
        market_value = sum(_position_market_value(row) for row in self.positions)
        return {
            "account": {
                "account_id": account_id,
                "account_type": account_type,
                "currency": self.currency,
                "cash": self.cash,
                "available_cash": max(self.cash - self.frozen_cash, 0.0),
                "frozen_cash": self.frozen_cash,
                "buying_power": max(self.cash - self.frozen_cash, 0.0),
                "market_value": market_value,
                "total_asset": self.cash + market_value,
                "cash_balances": [
                    {
                        "currency": self.currency,
                        "cash": self.cash,
                        "settled_cash": self.cash,
                        "buying_power": max(self.cash - self.frozen_cash, 0.0),
                    }
                ],
            }
        }

    def positions_payload(self, query: Mapping[str, str]) -> dict[str, Any]:
        account_id = query.get("account_id") or self.account_id
        account_type = (query.get("account_type") or self.account_type).upper()
        return {
            "positions": [
                self._with_account_defaults(row, account_id, account_type)
                for row in self.positions
            ]
        }

    def orders_payload(self, query: Mapping[str, str]) -> dict[str, Any]:
        account_id = query.get("account_id") or self.account_id
        account_type = (query.get("account_type") or self.account_type).upper()
        rows = [
            self._with_account_defaults(row, account_id, account_type)
            for row in self.orders
        ]
        if _truthy(query.get("open_only")):
            rows = [row for row in rows if _is_open_order(row)]
        return {"orders": rows}

    def trades_payload(self, query: Mapping[str, str]) -> dict[str, Any]:
        account_id = query.get("account_id") or self.account_id
        account_type = (query.get("account_type") or self.account_type).upper()
        return {
            "trades": [
                self._with_account_defaults(row, account_id, account_type)
                for row in self.trades
            ]
        }

    def submit_order(self, request: Mapping[str, Any]) -> tuple[HTTPStatus, dict[str, Any]]:
        status, payload = self._submit_order(request)
        self.audit_event(
            "connector_submit_order",
            "/submit_order",
            status,
            request_payload=request,
            response_payload=payload,
        )
        return status, payload

    def cancel_order(self, request: Mapping[str, Any]) -> tuple[HTTPStatus, dict[str, Any]]:
        status, payload = self._cancel_order(request)
        self.audit_event(
            "connector_cancel_order",
            "/cancel_order",
            status,
            request_payload=request,
            response_payload=payload,
        )
        return status, payload

    def audit_event(
        self,
        event_type: str,
        endpoint: str,
        status: HTTPStatus | int,
        *,
        request_payload: Mapping[str, Any] | None = None,
        response_payload: Mapping[str, Any] | None = None,
        client_address: str | None = None,
    ) -> None:
        status_code = status.value if isinstance(status, HTTPStatus) else int(status)
        event = {
            "ts": _utc_iso(),
            "event": event_type,
            "endpoint": endpoint,
            "status_code": status_code,
            "mode": self.mode,
            "account_id": self.account_id,
            "account_type": self.account_type,
            "client_address": client_address,
            "request": dict(request_payload or {}),
            "response": dict(response_payload or {}),
        }
        self.audit_events.append(event)
        if self.audit_log_path is None:
            return
        path = Path(self.audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _submit_order(self, request: Mapping[str, Any]) -> tuple[HTTPStatus, dict[str, Any]]:
        if self.mode != PAPER_SUBMIT_MODE:
            return _error(
                "QMT stub is in readonly mode; start with --mode paper_submit to accept paper orders.",
                HTTPStatus.FORBIDDEN,
            )

        validation_error = _order_validation_error(request)
        if validation_error:
            return _error(validation_error, HTTPStatus.BAD_REQUEST)

        risk_error = self.risk_policy.violation(request, self)
        if risk_error:
            return _error(f"risk check failed: {risk_error}", HTTPStatus.FORBIDDEN)

        client_order_id = _text(request.get("client_order_id"))
        existing_order = self._order_by_client_id(client_order_id)
        fingerprint = _idempotency_fingerprint(request)
        if existing_order is not None:
            if existing_order.get("_idempotency_fingerprint") != fingerprint:
                return _error(
                    f"client_order_id {client_order_id} already exists with different order terms",
                    HTTPStatus.CONFLICT,
                )
            replay = {key: value for key, value in existing_order.items() if not key.startswith("_")}
            replay["idempotent_replay"] = True
            replay["message"] = "idempotent replay; original stub order returned"
            return HTTPStatus.OK, {"order": replay}

        order_id = str(self.next_order_id)
        self.next_order_id += 1
        limit_price = _float_value(request.get("limit_price")) or _float_value(request.get("price"))
        quantity = _float_value(request.get("quantity")) or _float_value(request.get("order_volume"))
        order = {
            "order_id": order_id,
            "broker_order_id": order_id,
            "order_sysid": f"STUB-{order_id}",
            "order_status": 50,
            "status": "submitted",
            "message": "stub paper order accepted",
            "account_id": request.get("account_id") or self.account_id,
            "account_type": request.get("account_type") or self.account_type,
            "symbol": request.get("symbol") or request.get("stock_code"),
            "asset_class": request.get("asset_class") or "equity",
            "exchange": request.get("exchange"),
            "currency": request.get("currency") or self.currency,
            "side": request.get("side") or request.get("order_side") or "buy",
            "quantity": quantity,
            "order_type": request.get("order_type") or "limit",
            "price": limit_price,
            "limit_price": limit_price,
            "multiplier": _float_value(request.get("multiplier")) or 1.0,
            "price_type": request.get("price_type") or "FIX_PRICE",
            "strategy_id": request.get("strategy_id"),
            "client_order_id": client_order_id,
            "order_remark": request.get("order_remark") or client_order_id,
            "created_at": _utc_iso(),
            "source": "qmt_stub",
            "_idempotency_fingerprint": fingerprint,
        }
        self.orders.append(order)
        response_order = {key: value for key, value in order.items() if not key.startswith("_")}
        return HTTPStatus.OK, {"order": response_order}

    def _cancel_order(self, request: Mapping[str, Any]) -> tuple[HTTPStatus, dict[str, Any]]:
        if self.mode != PAPER_SUBMIT_MODE:
            return _error(
                "QMT stub is in readonly mode; start with --mode paper_submit to accept cancels.",
                HTTPStatus.FORBIDDEN,
            )

        broker_order_id = str(
            request.get("broker_order_id")
            or request.get("order_id")
            or request.get("order_sysid")
            or ""
        ).strip()
        if not broker_order_id:
            return _error("broker_order_id is required.", HTTPStatus.BAD_REQUEST)

        for order in self.orders:
            identifiers = {
                str(order.get("broker_order_id") or ""),
                str(order.get("order_id") or ""),
                str(order.get("order_sysid") or ""),
            }
            if broker_order_id in identifiers:
                order.update(
                    {
                        "order_status": 54,
                        "status": "cancelled",
                        "message": "stub cancel request accepted",
                        "cancelled_at": _utc_iso(),
                    }
                )
                return HTTPStatus.OK, {
                    "cancelled": True,
                    "message": "cancel request accepted",
                }

        return HTTPStatus.OK, {"cancelled": False, "message": "order not found"}

    def _order_by_client_id(self, client_order_id: str | None) -> dict[str, Any] | None:
        if not client_order_id:
            return None
        for order in self.orders:
            if order.get("client_order_id") == client_order_id:
                return order
        return None

    def _with_account_defaults(
        self,
        row: Mapping[str, Any],
        account_id: str,
        account_type: str,
    ) -> dict[str, Any]:
        output = dict(row)
        output.setdefault("account_id", account_id)
        output.setdefault("account_type", account_type)
        output.setdefault("currency", self.currency)
        return {key: value for key, value in output.items() if not key.startswith("_")}


@dataclass(slots=True)
class RunningQMTConnectorStub:
    """Handle returned by ``start_qmt_connector_stub`` for tests and demos."""

    server: ThreadingHTTPServer
    thread: threading.Thread
    base_url: str
    state: StubConnectorState

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def create_qmt_connector_stub_server(
    host: str = "127.0.0.1",
    port: int = DEFAULT_READONLY_PORT,
    *,
    state: StubConnectorState | None = None,
    api_token: str | None = None,
    signing_secret: str | None = None,
    risk_policy: ConnectorRiskPolicy | None = None,
    allow_public_bind: bool = False,
    signature_tolerance_seconds: int = DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
    quiet: bool = True,
) -> ThreadingHTTPServer:
    active_state = state or StubConnectorState()
    if risk_policy is not None:
        active_state.risk_policy = risk_policy
    _validate_connector_startup(
        host,
        active_state,
        api_token=api_token,
        signing_secret=signing_secret,
        allow_public_bind=allow_public_bind,
    )
    handler = _build_handler(
        active_state,
        api_token=api_token,
        signing_secret=signing_secret,
        signature_tolerance_seconds=signature_tolerance_seconds,
        quiet=quiet,
    )
    server = _ReusableThreadingHTTPServer((host, port), handler)
    setattr(server, "qmt_stub_state", active_state)
    return server


def start_qmt_connector_stub(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    state: StubConnectorState | None = None,
    api_token: str | None = None,
    signing_secret: str | None = None,
    risk_policy: ConnectorRiskPolicy | None = None,
    allow_public_bind: bool = False,
    signature_tolerance_seconds: int = DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
    quiet: bool = True,
) -> RunningQMTConnectorStub:
    active_state = state or StubConnectorState()
    server = create_qmt_connector_stub_server(
        host,
        port,
        state=active_state,
        api_token=api_token,
        signing_secret=signing_secret,
        risk_policy=risk_policy,
        allow_public_bind=allow_public_bind,
        signature_tolerance_seconds=signature_tolerance_seconds,
        quiet=quiet,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    actual_host, actual_port = server.server_address[:2]
    return RunningQMTConnectorStub(
        server=server,
        thread=thread,
        base_url=f"http://{actual_host}:{actual_port}",
        state=active_state,
    )


def serve_qmt_connector_stub(
    host: str = "127.0.0.1",
    port: int = DEFAULT_READONLY_PORT,
    *,
    state: StubConnectorState | None = None,
    api_token: str | None = None,
    signing_secret: str | None = None,
    risk_policy: ConnectorRiskPolicy | None = None,
    allow_public_bind: bool = False,
    signature_tolerance_seconds: int = DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
    quiet: bool = False,
) -> None:
    server = create_qmt_connector_stub_server(
        host,
        port,
        state=state,
        api_token=api_token,
        signing_secret=signing_secret,
        risk_policy=risk_policy,
        allow_public_bind=allow_public_bind,
        signature_tolerance_seconds=signature_tolerance_seconds,
        quiet=quiet,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_handler(
    state: StubConnectorState,
    *,
    api_token: str | None,
    signing_secret: str | None,
    signature_tolerance_seconds: int,
    quiet: bool,
) -> type[BaseHTTPRequestHandler]:
    expected_authorization = f"Bearer {api_token}" if api_token else None

    class QMTConnectorStubHandler(BaseHTTPRequestHandler):
        server_version = "OQPQMTConnectorStub/0.2"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if not self._authorized(expected_authorization, path):
                return
            if not self._signature_valid(signing_secret, path, parsed.query, b""):
                return

            query = _single_value_query(parsed.query)
            if path == "/health":
                self._send_json(HTTPStatus.OK, state.health_payload())
            elif path == "/account":
                self._send_json(HTTPStatus.OK, state.account_payload(query))
            elif path == "/positions":
                self._send_json(HTTPStatus.OK, state.positions_payload(query))
            elif path == "/orders":
                self._send_json(HTTPStatus.OK, state.orders_payload(query))
            elif path == "/trades":
                self._send_json(HTTPStatus.OK, state.trades_payload(query))
            else:
                self._send_json(*_error(f"unknown endpoint: {path}", HTTPStatus.NOT_FOUND))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            raw_body = self._read_raw_body()
            if not self._authorized(expected_authorization, path, raw_body=raw_body):
                return
            if not self._signature_valid(signing_secret, path, parsed.query, raw_body):
                return

            body = self._parse_json_body(raw_body)
            if body is None:
                status, payload = _error("request body must be a JSON object.", HTTPStatus.BAD_REQUEST)
                state.audit_event(
                    "connector_request_rejected",
                    path,
                    status,
                    response_payload=payload,
                    client_address=self._client_address(),
                )
                self._send_json(status, payload)
                return

            if path == "/submit_order":
                self._send_json(*state.submit_order(body))
            elif path == "/cancel_order":
                self._send_json(*state.cancel_order(body))
            else:
                status, payload = _error(f"unknown endpoint: {path}", HTTPStatus.NOT_FOUND)
                if path in WRITE_ENDPOINTS:
                    state.audit_event(
                        "connector_request_rejected",
                        path,
                        status,
                        request_payload=body,
                        response_payload=payload,
                        client_address=self._client_address(),
                    )
                self._send_json(status, payload)

        def log_message(self, format: str, *args: Any) -> None:
            if not quiet:
                super().log_message(format, *args)

        def _authorized(
            self,
            expected: str | None,
            path: str,
            *,
            raw_body: bytes = b"",
        ) -> bool:
            if expected is None:
                return True
            if self.headers.get("Authorization") == expected:
                return True
            status, payload = _error("unauthorized", HTTPStatus.UNAUTHORIZED)
            if path in WRITE_ENDPOINTS:
                state.audit_event(
                    "connector_auth_rejected",
                    path,
                    status,
                    request_payload=_body_preview(raw_body),
                    response_payload=payload,
                    client_address=self._client_address(),
                )
            self._send_json(status, payload)
            return False

        def _signature_valid(
            self,
            secret: str | None,
            path: str,
            query: str,
            body: bytes,
        ) -> bool:
            if not secret:
                return True
            result = verify_qmt_signature(
                self.headers,
                secret,
                self.command,
                path,
                params=query,
                body=body,
                tolerance_seconds=signature_tolerance_seconds,
            )
            if result.ok and result.nonce not in state.seen_nonces:
                if result.nonce:
                    state.seen_nonces.add(result.nonce)
                return True
            message = "replayed signature nonce" if result.ok else result.error or "invalid signature"
            status, payload = _error(message, HTTPStatus.UNAUTHORIZED)
            if path in WRITE_ENDPOINTS:
                state.audit_event(
                    "connector_signature_rejected",
                    path,
                    status,
                    request_payload=_body_preview(body),
                    response_payload=payload,
                    client_address=self._client_address(),
                )
            self._send_json(status, payload)
            return False

        def _read_raw_body(self) -> bytes:
            try:
                content_length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                return b""
            return self.rfile.read(content_length) if content_length else b"{}"

        def _parse_json_body(self, raw_body: bytes) -> dict[str, Any] | None:
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                return None
            return dict(payload) if isinstance(payload, Mapping) else None

        def _client_address(self) -> str:
            return str(self.client_address[0]) if self.client_address else ""

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return QMTConnectorStubHandler


def _validate_connector_startup(
    host: str,
    state: StubConnectorState,
    *,
    api_token: str | None,
    signing_secret: str | None,
    allow_public_bind: bool,
) -> None:
    if not allow_public_bind and not _is_private_bind_host(host):
        raise ValueError(
            "QMT connector must bind to localhost, a private LAN IP, or a Tailscale/WireGuard IP."
        )
    if state.mode != PAPER_SUBMIT_MODE:
        return
    if not _text(api_token):
        raise ValueError("paper_submit mode requires --api-token.")
    if not _text(signing_secret):
        raise ValueError("paper_submit mode requires --signing-secret.")
    startup_error = state.risk_policy.startup_error(
        account_type=state.account_type,
        mode=state.mode,
    )
    if startup_error:
        raise ValueError(startup_error)


def _is_private_bind_host(host: str) -> bool:
    text = host.strip().lower()
    if text in {"localhost"}:
        return True
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if ip.is_unspecified:
        return False
    return bool(ip.is_loopback or ip.is_private or ip in TAILSCALE_CGNAT)


def _default_positions(account_id: str, account_type: str) -> list[dict[str, Any]]:
    return [
        {
            "account_id": account_id,
            "account_type": account_type,
            "symbol": "600000.SH",
            "asset_class": "equity",
            "quantity": 1000,
            "avg_price": 10.0,
            "market_price": 10.5,
            "market_value": 10_500.0,
            "unrealized_pnl": 500.0,
            "currency": "CNY",
            "multiplier": 1,
            "instrument_name": "QMT stub equity",
        },
        {
            "account_id": account_id,
            "account_type": account_type,
            "symbol": "rb2401.SF",
            "asset_class": "future",
            "position": 2,
            "direction": "short",
            "avg_price": 3600.0,
            "last_price": 3580.0,
            "instrument_value": 71_600.0,
            "float_profit": 400.0,
            "currency": "CNY",
            "multiplier": 10,
            "instrument_name": "QMT stub future",
        },
    ]


def _single_value_query(query: str) -> dict[str, str]:
    parsed = parse_qs(query, keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items() if values}


def _error(message: str, status: HTTPStatus) -> tuple[HTTPStatus, dict[str, Any]]:
    return status, {"error": message, "status": "error"}


def _is_open_order(row: Mapping[str, Any]) -> bool:
    status_text = str(row.get("status") or "").lower()
    if status_text in {"cancelled", "canceled", "filled", "rejected"}:
        return False
    return str(row.get("order_status") or "") not in {"54", "56", "57"}


def _order_validation_error(request: Mapping[str, Any]) -> str | None:
    symbol = str(request.get("symbol") or request.get("stock_code") or "").strip()
    if not symbol:
        return "symbol is required."
    client_order_id = str(request.get("client_order_id") or "").strip()
    if not client_order_id:
        return "client_order_id is required for idempotency."
    quantity = _float_value(request.get("quantity")) or _float_value(request.get("order_volume"))
    if quantity is None or quantity <= 0:
        return "quantity must be positive."
    order_type = str(request.get("order_type") or "limit").lower()
    limit_price = _float_value(request.get("limit_price")) or _float_value(request.get("price"))
    if order_type != "limit":
        return "stub connector only accepts limit orders."
    if limit_price is None or limit_price <= 0:
        return "limit_price must be positive."
    return None


def _idempotency_fingerprint(request: Mapping[str, Any]) -> str:
    keys = (
        "account_id",
        "account_type",
        "symbol",
        "asset_class",
        "side",
        "quantity",
        "order_type",
        "limit_price",
        "price",
        "multiplier",
        "strategy_id",
    )
    payload = {key: request.get(key) for key in keys if request.get(key) is not None}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _position_market_value(row: Mapping[str, Any]) -> float:
    explicit = _float_value(row.get("market_value")) or _float_value(row.get("instrument_value"))
    if explicit is not None:
        return abs(explicit)
    quantity = (
        _float_value(row.get("quantity"))
        or _float_value(row.get("position"))
        or _float_value(row.get("volume"))
        or 0.0
    )
    price = (
        _float_value(row.get("market_price"))
        or _float_value(row.get("last_price"))
        or _float_value(row.get("price"))
        or 0.0
    )
    multiplier = _float_value(row.get("multiplier")) or 1.0
    return abs(quantity * price * multiplier)


def _float_value(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _body_preview(raw_body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OQP QMT connector stub.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mode", choices=VALID_MODES, default=READONLY_MODE)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--signing-secret", default=None)
    parser.add_argument("--account-id", default="QMT123456")
    parser.add_argument("--account-type", default="STOCK")
    parser.add_argument("--session-id", type=int, default=880001)
    parser.add_argument("--cash", type=float, default=900_000.0)
    parser.add_argument("--allowed-symbol", action="append", default=[])
    parser.add_argument("--allowed-asset-class", action="append", default=[])
    parser.add_argument("--allowed-account-type", action="append", default=[])
    parser.add_argument("--max-quantity", type=float, default=None)
    parser.add_argument("--max-notional", type=float, default=None)
    parser.add_argument("--min-limit-price", type=float, default=None)
    parser.add_argument("--max-limit-price", type=float, default=None)
    parser.add_argument("--audit-log-path", default=None)
    parser.add_argument("--allow-public-bind", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    port = args.port or (
        DEFAULT_SUBMIT_PORT if args.mode == PAPER_SUBMIT_MODE else DEFAULT_READONLY_PORT
    )
    risk_policy = ConnectorRiskPolicy(
        allowed_symbols=tuple(args.allowed_symbol),
        allowed_asset_classes=tuple(args.allowed_asset_class),
        allowed_account_types=tuple(args.allowed_account_type),
        max_quantity=args.max_quantity,
        max_notional=args.max_notional,
        min_limit_price=args.min_limit_price,
        max_limit_price=args.max_limit_price,
    )
    state = StubConnectorState(
        account_id=args.account_id,
        account_type=args.account_type,
        session_id=args.session_id,
        mode=args.mode,
        cash=args.cash,
        risk_policy=risk_policy,
        audit_log_path=args.audit_log_path,
    )
    server = create_qmt_connector_stub_server(
        args.host,
        port,
        state=state,
        api_token=args.api_token,
        signing_secret=args.signing_secret,
        allow_public_bind=args.allow_public_bind,
        quiet=not args.verbose,
    )
    host, actual_port = server.server_address[:2]
    print(
        f"QMT connector stub serving on http://{host}:{actual_port} "
        f"mode={state.mode} account_id={state.account_id}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("QMT connector stub shutting down.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
