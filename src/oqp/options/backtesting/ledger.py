"""Position ledger for daily event-driven option backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from oqp.options.lifecycle import expiry_settlement_value
from oqp.options.liquidity import fill_price, mark_price
from oqp.options.margin import OptionMarginPolicy, premium_cashflow


@dataclass(slots=True)
class OptionPositionLot:
    option_symbol: str
    underlying_symbol: str
    expiry: date
    right: str
    strike: float
    quantity: float
    entry_price: float
    multiplier: float
    opened_at: date
    market_vertical: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gross_contracts(self) -> float:
        return abs(float(self.quantity))

    def row(self) -> dict[str, Any]:
        return {
            "option_symbol": self.option_symbol,
            "underlying_symbol": self.underlying_symbol,
            "expiry": self.expiry,
            "right": self.right,
            "strike": self.strike,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "multiplier": self.multiplier,
            "opened_at": self.opened_at,
            "market_vertical": self.market_vertical,
            **self.metadata,
        }


class OptionBacktestLedger:
    def __init__(
        self,
        *,
        initial_capital: float,
        margin_policy: OptionMarginPolicy | None = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.margin_policy = margin_policy or OptionMarginPolicy()
        self.positions: dict[str, OptionPositionLot] = {}
        self.trades: list[dict[str, Any]] = []

    def open_long(
        self,
        row: pd.Series,
        *,
        quantity: float,
        trade_date: date,
        fill: float,
        reason: str,
    ) -> None:
        option_symbol = str(row["option_symbol"])
        multiplier = float(row.get("multiplier") or 100.0)
        cashflow = premium_cashflow(
            quantity=quantity,
            price=fill,
            multiplier=multiplier,
            commission_per_contract=self.margin_policy.commission_per_contract,
        )
        self.cash += cashflow
        lot = OptionPositionLot(
            option_symbol=option_symbol,
            underlying_symbol=str(row["underlying_symbol"]),
            expiry=pd.to_datetime(row["expiry"]).date(),
            right=str(row["right"]),
            strike=float(row["strike"]),
            quantity=float(quantity),
            entry_price=float(fill),
            multiplier=multiplier,
            opened_at=trade_date,
            market_vertical=str(row.get("market_vertical") or "OPTIONS_US"),
            metadata={"quote_source": row.get("quote_source"), "entry_reason": reason},
        )
        self.positions[option_symbol] = lot
        self._record_trade(
            trade_date=trade_date,
            row=row,
            quantity=quantity,
            price=fill,
            cashflow=cashflow,
            reason=reason,
        )

    def close_position(
        self,
        option_symbol: str,
        row: pd.Series | dict[str, Any],
        *,
        trade_date: date,
        price: float,
        reason: str,
    ) -> None:
        lot = self.positions.pop(option_symbol, None)
        if lot is None:
            return
        closing_quantity = -lot.quantity
        cashflow = premium_cashflow(
            quantity=closing_quantity,
            price=price,
            multiplier=lot.multiplier,
            commission_per_contract=self.margin_policy.commission_per_contract,
        )
        self.cash += cashflow
        self._record_trade(
            trade_date=trade_date,
            row=row,
            quantity=closing_quantity,
            price=price,
            cashflow=cashflow,
            reason=reason,
            opened_at=lot.opened_at,
            entry_price=lot.entry_price,
        )

    def settle_expired(
        self,
        *,
        trade_date: date,
        underlying_prices: dict[str, float],
    ) -> None:
        for symbol, lot in list(self.positions.items()):
            if lot.expiry > trade_date:
                continue
            settlement = expiry_settlement_value(lot.row(), underlying_prices.get(lot.underlying_symbol))
            self.close_position(symbol, lot.row(), trade_date=trade_date, price=settlement, reason="expiry")

    def mark_to_market(
        self,
        *,
        trade_date: date,
        chain_lookup: dict[str, pd.Series],
        underlying_prices: dict[str, float],
    ) -> tuple[float, float, list[dict[str, Any]]]:
        market_value = 0.0
        gross_exposure = 0.0
        rows: list[dict[str, Any]] = []
        for lot in self.positions.values():
            quote = chain_lookup.get(lot.option_symbol)
            price = mark_price(quote) if quote is not None else None
            if price is None and lot.expiry <= trade_date:
                price = expiry_settlement_value(lot.row(), underlying_prices.get(lot.underlying_symbol))
            if price is None:
                price = lot.entry_price
            value = lot.quantity * price * lot.multiplier
            market_value += value
            gross_exposure += abs(value)
            rows.append(
                {
                    **lot.row(),
                    "date": trade_date,
                    "mark": price,
                    "market_value": value,
                    "unrealized_pnl": lot.quantity * (price - lot.entry_price) * lot.multiplier,
                }
            )
        return market_value, gross_exposure, rows

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def positions_frame(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([lot.row() for lot in self.positions.values()])

    def _record_trade(
        self,
        *,
        trade_date: date,
        row: pd.Series | dict[str, Any],
        quantity: float,
        price: float,
        cashflow: float,
        reason: str,
        opened_at: date | None = None,
        entry_price: float | None = None,
    ) -> None:
        self.trades.append(
            {
                "date": trade_date,
                "option_symbol": row.get("option_symbol"),
                "underlying_symbol": row.get("underlying_symbol"),
                "expiry": row.get("expiry"),
                "right": row.get("right"),
                "strike": row.get("strike"),
                "quantity": quantity,
                "price": price,
                "cashflow": cashflow,
                "commission": abs(quantity) * self.margin_policy.commission_per_contract,
                "reason": reason,
                "opened_at": opened_at,
                "entry_price": entry_price,
            }
        )
