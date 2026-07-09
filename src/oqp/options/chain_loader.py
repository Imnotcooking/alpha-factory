"""Option-chain normalization and lookup utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from oqp.options.contracts import CHAIN_COLUMNS, OptionQuoteSnapshot, normalize_option_right


_COLUMN_ALIASES = {
    "option_symbol": ("option_symbol", "symbol", "ticker", "contract_symbol", "option_ticker", "代码", "合约"),
    "vendor_symbol": ("vendor_symbol", "massive_symbol", "polygon_symbol", "raw_symbol"),
    "underlying_symbol": ("underlying_symbol", "underlying", "underlying_ticker", "root", "标的", "标的代码"),
    "expiry": ("expiry", "expiration", "expiration_date", "expire_date", "maturity", "到期日"),
    "right": ("right", "option_type", "contract_type", "put_call", "cp", "call_put", "类型"),
    "strike": ("strike", "strike_price", "行权价"),
    "multiplier": ("multiplier", "contract_multiplier", "shares_per_contract", "合约乘数"),
    "exchange": ("exchange", "primary_exchange", "交易所"),
    "market_vertical": ("market_vertical", "asset_class"),
    "currency": ("currency", "币种"),
    "date": ("date", "trading_date", "trade_date", "as_of", "日期"),
    "timestamp": ("timestamp", "datetime", "quote_datetime", "ts"),
    "bid": ("bid", "bid_price", "买价"),
    "ask": ("ask", "ask_price", "卖价"),
    "mid": ("mid", "mid_price"),
    "last": ("last", "last_price", "lastPrice", "最新价"),
    "mark": ("mark", "fair_value", "fmv"),
    "open": ("open", "open_price", "开盘价"),
    "high": ("high", "high_price", "最高价"),
    "low": ("low", "low_price", "最低价"),
    "close": ("close", "close_price", "settlement", "settle", "settlement_price", "收盘价", "结算价"),
    "volume": ("volume", "vol", "成交量"),
    "open_interest": ("open_interest", "oi", "持仓量"),
    "implied_volatility": ("implied_volatility", "iv", "隐含波动率"),
    "delta": ("delta",),
    "gamma": ("gamma",),
    "theta": ("theta",),
    "vega": ("vega",),
    "quote_timestamp": ("quote_timestamp", "quote_time", "last_updated"),
    "quote_source": ("quote_source", "source", "vendor"),
}


NUMERIC_COLUMNS = (
    "strike",
    "multiplier",
    "bid",
    "ask",
    "mid",
    "last",
    "mark",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
)


def normalize_option_chain_frame(
    frame: pd.DataFrame,
    *,
    market_vertical: str = "OPTIONS_US",
    source: str | None = None,
    default_multiplier: float | None = None,
) -> pd.DataFrame:
    """Return a vendor-neutral option-chain frame.

    The function accepts Massive-style columns, Yahoo-style columns, and common
    Chinese static-parquet names. Missing bid/ask data is allowed, but ``mark``
    is then treated as a settlement/close proxy by the execution layer.
    """

    if frame.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    out = pd.DataFrame(index=frame.index)
    for canonical, aliases in _COLUMN_ALIASES.items():
        column = _first_existing(frame, aliases)
        if column is not None:
            out[canonical] = frame[column]

    for column in CHAIN_COLUMNS:
        if column not in out.columns:
            out[column] = None

    if out["timestamp"].isna().all() and not out["date"].isna().all():
        out["timestamp"] = out["date"]
    if out["date"].isna().all() and not out["timestamp"].isna().all():
        out["date"] = out["timestamp"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["quote_timestamp"] = pd.to_datetime(out["quote_timestamp"], errors="coerce")

    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date
    out["right"] = out["right"].map(_safe_right)
    out["underlying_symbol"] = out["underlying_symbol"].fillna("").astype(str).str.upper().str.strip()
    out["option_symbol"] = out["option_symbol"].fillna("").astype(str).str.strip()
    out["vendor_symbol"] = out["vendor_symbol"].where(out["vendor_symbol"].notna(), out["option_symbol"])
    out["market_vertical"] = out["market_vertical"].fillna(market_vertical)
    out["currency"] = out["currency"].fillna("USD" if market_vertical == "OPTIONS_US" else "CNY")
    out["quote_source"] = out["quote_source"].fillna(source or "")
    out["underlying_type"] = out["underlying_type"].fillna("unknown")
    out["exercise_style"] = out["exercise_style"].fillna("american" if market_vertical == "OPTIONS_US" else "unknown")
    out["settlement_style"] = out["settlement_style"].fillna("physical")

    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if default_multiplier is not None:
        out["multiplier"] = out["multiplier"].fillna(float(default_multiplier))
    out["multiplier"] = out["multiplier"].fillna(100.0 if market_vertical == "OPTIONS_US" else 1.0)

    out["mid"] = out["mid"].where(out["mid"].gt(0), _mid_from_bid_ask(out))
    out["mark"] = _first_positive_series(out, ("mark", "mid", "close", "last"))
    out["option_symbol"] = out.apply(_fill_option_symbol, axis=1)
    out = out.dropna(subset=["date", "expiry", "right", "strike"])
    out = out[out["option_symbol"].astype(str).ne("")]
    out = out.sort_values(["date", "underlying_symbol", "expiry", "right", "strike", "option_symbol"])
    return out.loc[:, list(CHAIN_COLUMNS)].reset_index(drop=True)


def option_quotes_to_frame(
    quotes: Iterable[Any],
    *,
    market_vertical: str = "OPTIONS_US",
) -> pd.DataFrame:
    rows = []
    for quote in quotes:
        snapshot = quote if isinstance(quote, OptionQuoteSnapshot) else None
        if snapshot is None:
            from oqp.options.contracts import option_quote_to_snapshot

            snapshot = option_quote_to_snapshot(quote, market_vertical=market_vertical)
        rows.append(snapshot.to_chain_row())
    return normalize_option_chain_frame(pd.DataFrame(rows), market_vertical=market_vertical)


def load_option_chain_file(
    path: str | Path,
    *,
    market_vertical: str = "OPTIONS_US",
    source: str | None = None,
    default_multiplier: float | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported option chain file type: {path.suffix}")
    return normalize_option_chain_frame(
        frame,
        market_vertical=market_vertical,
        source=source or path.name,
        default_multiplier=default_multiplier,
    )


class OptionChainStore:
    """In-memory normalized option-chain lookup."""

    def __init__(self, chain: pd.DataFrame):
        self.chain = normalize_option_chain_frame(chain)

    @property
    def available_dates(self) -> list[Any]:
        return sorted(self.chain["date"].dropna().unique().tolist())

    def snapshot(
        self,
        as_of: Any,
        *,
        underlying_symbol: str | None = None,
        right: str | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        min_strike: float | None = None,
        max_strike: float | None = None,
    ) -> pd.DataFrame:
        if self.chain.empty:
            return self.chain.copy()
        as_of_date = pd.to_datetime(as_of).date()
        frame = self.chain.loc[self.chain["date"].eq(as_of_date)].copy()
        if underlying_symbol:
            frame = frame.loc[frame["underlying_symbol"].astype(str).str.upper().eq(underlying_symbol.upper())]
        if right:
            wanted = normalize_option_right(right).value
            frame = frame.loc[frame["right"].eq(wanted)]
        if min_dte is not None or max_dte is not None:
            dte = (pd.to_datetime(frame["expiry"]) - pd.Timestamp(as_of_date)).dt.days
            if min_dte is not None:
                frame = frame.loc[dte.ge(min_dte)]
                dte = dte.loc[frame.index]
            if max_dte is not None:
                frame = frame.loc[dte.le(max_dte)]
        if min_strike is not None:
            frame = frame.loc[frame["strike"].ge(float(min_strike))]
        if max_strike is not None:
            frame = frame.loc[frame["strike"].le(float(max_strike))]
        return frame.reset_index(drop=True)

    def contract_history(self, option_symbol: str) -> pd.DataFrame:
        return self.chain.loc[self.chain["option_symbol"].astype(str).eq(str(option_symbol))].copy()

    def latest_quote_on_or_before(self, option_symbol: str, as_of: Any) -> pd.Series | None:
        as_of_date = pd.to_datetime(as_of).date()
        history = self.contract_history(option_symbol)
        history = history.loc[history["date"].le(as_of_date)].sort_values("date")
        if history.empty:
            return None
        return history.iloc[-1]


def _first_existing(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    columns = {str(column).lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        lowered = alias.lower()
        if lowered in columns:
            return columns[lowered]
    return None


def _safe_right(value: Any) -> str | None:
    try:
        return normalize_option_right(value).value
    except ValueError:
        return None


def _mid_from_bid_ask(frame: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(frame["bid"], errors="coerce")
    ask = pd.to_numeric(frame["ask"], errors="coerce")
    mid = (bid + ask) / 2.0
    return mid.where(bid.gt(0) & ask.gt(0))


def _first_positive_series(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series(index=frame.index, dtype="float64")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result = result.where(result.notna() & result.gt(0), values.where(values.gt(0)))
    return result


def _fill_option_symbol(row: pd.Series) -> str:
    current = str(row.get("option_symbol") or "").strip()
    if current:
        return current
    underlying = str(row.get("underlying_symbol") or "OPT").upper()
    expiry = pd.to_datetime(row.get("expiry")).strftime("%y%m%d")
    right = "C" if row.get("right") == "call" else "P"
    strike = int(round(float(row.get("strike") or 0.0) * 1000))
    return f"{underlying}{expiry}{right}{strike:08d}"
