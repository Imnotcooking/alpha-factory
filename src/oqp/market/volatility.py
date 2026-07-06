"""Historical volatility helpers for portfolio reporting."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_PRICE_HISTORY_PATHS = (
    REPO_ROOT / "runtime" / "market" / "price_history.parquet",
    REPO_ROOT / "runtime" / "market" / "price_history.csv",
    REPO_ROOT / "runtime" / "data" / "market" / "price_history.parquet",
    REPO_ROOT / "runtime" / "data" / "market" / "price_history.csv",
)


def load_price_history(paths: Iterable[str | Path] | None = None) -> pd.DataFrame:
    """Load the first available long-form price history cache."""

    configured = os.getenv("OQP_PRICE_HISTORY_PATH")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(Path(path) for path in (paths or DEFAULT_PRICE_HISTORY_PATHS))

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".txt"}:
            frame = pd.read_csv(path)
        else:
            continue
        return normalize_price_history(frame)
    return pd.DataFrame(columns=["symbol", "date", "close"])


def normalize_price_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common price-history column spellings."""

    columns = {str(column).lower(): column for column in frame.columns}
    symbol_col = _first_existing(columns, ("symbol", "ticker", "underlying"))
    date_col = _first_existing(columns, ("date", "datetime", "timestamp"))
    close_col = _first_existing(columns, ("close", "adj close", "adj_close", "price"))
    if symbol_col is None or date_col is None or close_col is None:
        return pd.DataFrame(columns=["symbol", "date", "close"])

    out = frame[[symbol_col, date_col, close_col]].copy()
    out.columns = ["symbol", "date", "close"]
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out[out["symbol"].ne("")]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def historical_volatility(
    close: pd.Series,
    *,
    window: int,
    periods_per_year: int = 252,
) -> float | None:
    """Annualized close-to-close historical volatility."""

    prices = pd.to_numeric(close, errors="coerce").dropna()
    returns = prices.pct_change().dropna()
    if len(returns) < max(int(window), 2):
        return None
    value = returns.tail(int(window)).std() * math.sqrt(periods_per_year)
    if pd.isna(value):
        return None
    return float(value)


def historical_volatility_frame(
    price_history: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 20),
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Return one row per symbol with annualized HV columns."""

    history = normalize_price_history(price_history)
    columns = ["symbol", *(f"hv_{window}d" for window in windows)]
    if history.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for symbol, group in history.groupby("symbol"):
        row = {"symbol": symbol}
        for window in windows:
            row[f"hv_{window}d"] = historical_volatility(
                group["close"],
                window=window,
                periods_per_year=periods_per_year,
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def enrich_with_historical_volatility(
    positions: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    underlying_col: str = "underlying",
) -> pd.DataFrame:
    """Attach HV columns to positions by underlying first, symbol second."""

    out = positions.copy()
    if volatility.empty:
        out["hv_5d"] = pd.NA
        out["hv_20d"] = pd.NA
        return out

    vol = volatility.copy()
    vol["symbol"] = vol["symbol"].astype(str).str.upper().str.strip()
    lookup = vol.set_index("symbol")
    keys = (
        out.get(underlying_col, out.get(symbol_col, pd.Series("", index=out.index)))
        .fillna(out.get(symbol_col, pd.Series("", index=out.index)))
        .astype(str)
        .str.upper()
        .str.strip()
    )
    fallback_keys = out.get(symbol_col, pd.Series("", index=out.index)).astype(str).str.upper().str.strip()

    for column in [col for col in vol.columns if col.startswith("hv_")]:
        primary = keys.map(lookup[column])
        fallback = fallback_keys.map(lookup[column])
        out[column] = primary.fillna(fallback)
    return out


def _first_existing(columns: dict[str, object], candidates: tuple[str, ...]) -> object | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None
