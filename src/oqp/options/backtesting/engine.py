"""Daily event-driven options backtesting engine."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from oqp.options.backtesting.ledger import OptionBacktestLedger
from oqp.options.backtesting.models import OptionBacktestRequest, OptionBacktestResult
from oqp.options.chain_loader import OptionChainStore, normalize_option_chain_frame
from oqp.options.lifecycle import days_to_expiry
from oqp.options.liquidity import fill_price, mark_price, passes_liquidity


class OptionBacktestEngine:
    """Conservative v1 engine for listed-options research.

    Supported behavior:
    - daily chain/underlying data
    - long call/put entries from directional signals
    - bid/ask-aware fills or explicit settlement proxy fallback
    - exit signals and expiry settlement
    """

    backend_id = "options_event_driven"

    def run(self, request: OptionBacktestRequest) -> OptionBacktestResult:
        chain = normalize_option_chain_frame(
            request.chain,
            market_vertical=request.market_vertical,
        )
        signals = _normalize_signals(request.signals)
        underlying = _normalize_underlying(request.underlying)
        store = OptionChainStore(chain)
        ledger = OptionBacktestLedger(
            initial_capital=request.initial_capital,
            margin_policy=request.margin,
        )

        dates = _event_dates(chain, signals, underlying)
        equity_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []
        prev_equity = request.initial_capital
        for current_date in dates:
            underlying_prices = _underlying_prices_for_date(underlying, current_date)
            chain_lookup = _chain_lookup_for_date(store, current_date)
            turnover_before = sum(abs(trade.get("cashflow", 0.0)) for trade in ledger.trades)

            ledger.settle_expired(trade_date=current_date, underlying_prices=underlying_prices)
            todays_signals = signals.loc[signals["date"].eq(current_date)]
            for _, signal in todays_signals.iterrows():
                self._process_signal(
                    signal,
                    current_date=current_date,
                    store=store,
                    ledger=ledger,
                    request=request,
                    underlying_prices=underlying_prices,
                )

            chain_lookup = _chain_lookup_for_date(store, current_date)
            option_value, gross_exposure, marks = ledger.mark_to_market(
                trade_date=current_date,
                chain_lookup=chain_lookup,
                underlying_prices=underlying_prices,
            )
            position_rows.extend(marks)
            equity = ledger.cash + option_value
            turnover_after = sum(abs(trade.get("cashflow", 0.0)) for trade in ledger.trades)
            equity_rows.append(
                {
                    "date": current_date,
                    "cash": ledger.cash,
                    "option_market_value": option_value,
                    "equity": equity,
                    "gross_equity": request.initial_capital * (equity / request.initial_capital),
                    "gross_exposure": gross_exposure,
                    "turnover": max(turnover_after - turnover_before, 0.0),
                    "daily_pnl": equity - prev_equity,
                }
            )
            prev_equity = equity

        equity_curve = pd.DataFrame(equity_rows)
        return OptionBacktestResult(
            equity_curve=equity_curve,
            trades=ledger.trades_frame(),
            positions=pd.DataFrame(position_rows),
            diagnostics={
                "backend_id": self.backend_id,
                "market_vertical": request.market_vertical,
                "chain_rows": int(len(chain)),
                "signal_rows": int(len(signals)),
                "underlying_rows": int(len(underlying)),
            },
        )

    def _process_signal(
        self,
        signal: pd.Series,
        *,
        current_date: date,
        store: OptionChainStore,
        ledger: OptionBacktestLedger,
        request: OptionBacktestRequest,
        underlying_prices: dict[str, float],
    ) -> None:
        underlying = str(signal["underlying_symbol"]).upper()
        direction = float(signal.get("direction") or 0.0)
        if direction == 0:
            self._close_underlying_positions(
                underlying,
                current_date=current_date,
                store=store,
                ledger=ledger,
                request=request,
            )
            return
        if (
            not request.config.allow_multiple_positions_per_underlying
            and any(lot.underlying_symbol == underlying for lot in ledger.positions.values())
        ):
            return
        contract = self._select_contract(
            signal,
            current_date=current_date,
            store=store,
            request=request,
            underlying_price=underlying_prices.get(underlying),
        )
        if contract is None:
            return
        fill = fill_price(contract, "buy", request.liquidity)
        quantity = float(signal.get("contracts") or request.config.contracts_per_signal)
        ledger.open_long(
            contract,
            quantity=quantity,
            trade_date=current_date,
            fill=fill,
            reason="signal_entry",
        )

    def _select_contract(
        self,
        signal: pd.Series,
        *,
        current_date: date,
        store: OptionChainStore,
        request: OptionBacktestRequest,
        underlying_price: float | None,
    ) -> pd.Series | None:
        explicit_symbol = str(signal.get("option_symbol") or "").strip()
        if explicit_symbol:
            quote = store.latest_quote_on_or_before(explicit_symbol, current_date)
            if quote is not None and passes_liquidity(quote, request.liquidity):
                return quote
            return None

        direction = float(signal.get("direction") or 0.0)
        right = str(signal.get("right") or ("call" if direction > 0 else "put")).lower()
        min_dte = int(signal.get("min_dte") or request.config.min_dte)
        max_dte = int(signal.get("max_dte") or request.config.max_dte)
        snapshot = store.snapshot(
            current_date,
            underlying_symbol=str(signal["underlying_symbol"]),
            right=right,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if snapshot.empty:
            return None
        liquid = snapshot.loc[snapshot.apply(lambda row: passes_liquidity(row, request.liquidity), axis=1)].copy()
        if liquid.empty:
            return None
        target_moneyness = float(signal.get("target_moneyness") or request.config.target_moneyness)
        if underlying_price is None or underlying_price <= 0:
            liquid["selection_score"] = liquid["strike"].rank(method="first")
        else:
            target_strike = underlying_price * target_moneyness
            liquid["selection_score"] = (liquid["strike"] - target_strike).abs()
        liquid["dte"] = liquid["expiry"].map(lambda expiry: days_to_expiry(current_date, expiry))
        liquid = liquid.sort_values(["selection_score", "dte", "open_interest", "volume"], ascending=[True, True, False, False])
        return liquid.iloc[0]

    def _close_underlying_positions(
        self,
        underlying: str,
        *,
        current_date: date,
        store: OptionChainStore,
        ledger: OptionBacktestLedger,
        request: OptionBacktestRequest,
    ) -> None:
        for symbol, lot in list(ledger.positions.items()):
            if lot.underlying_symbol != underlying:
                continue
            quote = store.latest_quote_on_or_before(symbol, current_date)
            if quote is None:
                continue
            price = fill_price(quote, "sell", request.liquidity)
            ledger.close_position(symbol, quote, trade_date=current_date, price=price, reason="signal_exit")


def _normalize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    if "date" not in out.columns:
        raise ValueError("Option signals require a date column.")
    if "underlying_symbol" not in out.columns:
        if "underlying" in out.columns:
            out["underlying_symbol"] = out["underlying"]
        else:
            raise ValueError("Option signals require underlying_symbol or underlying.")
    if "direction" not in out.columns:
        out["direction"] = out.get("signal", 0)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.upper()
    out["direction"] = pd.to_numeric(out["direction"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["date", "underlying_symbol"]).sort_values("date").reset_index(drop=True)


def _normalize_underlying(underlying: pd.DataFrame) -> pd.DataFrame:
    out = underlying.copy()
    if out.empty:
        return pd.DataFrame(columns=["date", "underlying_symbol", "close"])
    if "underlying_symbol" not in out.columns:
        if "ticker" in out.columns:
            out["underlying_symbol"] = out["ticker"]
        elif "symbol" in out.columns:
            out["underlying_symbol"] = out["symbol"]
        else:
            raise ValueError("Underlying data requires underlying_symbol, ticker, or symbol.")
    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = out["datetime"]
        else:
            raise ValueError("Underlying data requires date or datetime.")
    if "close" not in out.columns:
        if "settlement" in out.columns:
            out["close"] = out["settlement"]
        elif "last_price" in out.columns:
            out["close"] = out["last_price"]
        else:
            raise ValueError("Underlying data requires close, settlement, or last_price.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "underlying_symbol", "close"]).sort_values("date").reset_index(drop=True)


def _event_dates(chain: pd.DataFrame, signals: pd.DataFrame, underlying: pd.DataFrame) -> list[date]:
    values = set(chain.get("date", pd.Series(dtype=object)).dropna().tolist())
    values.update(signals.get("date", pd.Series(dtype=object)).dropna().tolist())
    values.update(underlying.get("date", pd.Series(dtype=object)).dropna().tolist())
    return sorted(values)


def _underlying_prices_for_date(underlying: pd.DataFrame, current_date: date) -> dict[str, float]:
    rows = underlying.loc[underlying["date"].eq(current_date)]
    return {
        str(row["underlying_symbol"]).upper(): float(row["close"])
        for _, row in rows.iterrows()
        if pd.notna(row.get("close"))
    }


def _chain_lookup_for_date(store: OptionChainStore, current_date: date) -> dict[str, pd.Series]:
    rows = store.snapshot(current_date)
    return {str(row["option_symbol"]): row for _, row in rows.iterrows()}
