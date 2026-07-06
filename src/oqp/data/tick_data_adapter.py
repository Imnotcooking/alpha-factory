from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


TICK_REQUIRED_COLUMNS = {
    "symbol",
    "datetime",
    "last_price",
    "volume",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    "oi",
}


class TickUniverseSelector:
    """Selects a product or main contract from a long-form L1 tick frame."""

    def __init__(self, product: str = "", symbol: str = ""):
        self.product = str(product or "").strip()
        self.symbol = str(symbol or "").strip()

    @staticmethod
    def looks_like_tick_columns(columns: set[str] | list[str]) -> bool:
        return TICK_REQUIRED_COLUMNS.issubset(set(columns))

    @staticmethod
    def product_prefix(symbol: str) -> str:
        match = re.match(r"^([A-Za-z]+)", str(symbol))
        return match.group(1) if match else ""

    @staticmethod
    def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "symbol" not in out.columns and "ticker" in out.columns:
            out["symbol"] = out["ticker"].astype(str)
        if "ticker" not in out.columns and "symbol" in out.columns:
            out["ticker"] = out["symbol"].astype(str)

        if "datetime" not in out.columns and "date" in out.columns:
            out["datetime"] = pd.to_datetime(out["date"])
        if "date" not in out.columns and "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"])

        if "last_price" not in out.columns and "close" in out.columns:
            out["last_price"] = pd.to_numeric(out["close"], errors="coerce")
        if "close" not in out.columns and "last_price" in out.columns:
            out["close"] = pd.to_numeric(out["last_price"], errors="coerce")

        if "symbol" in out.columns:
            out["symbol"] = out["symbol"].astype(str)
            out["ticker"] = out["ticker"].astype(str)
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")

        return out

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.normalize_schema(df)
        missing = sorted(TICK_REQUIRED_COLUMNS.difference(out.columns))
        if missing:
            raise ValueError(f"Tick data missing required columns: {missing}")

        if self.symbol:
            selected = out[out["symbol"].str.lower().eq(self.symbol.lower())].copy()
            if selected.empty:
                raise ValueError(f"Symbol {self.symbol!r} not found in tick data.")
        else:
            selected = self._select_product(out)
            selected_symbol = self._main_contract_symbol(selected)
            selected = selected[selected["symbol"].eq(selected_symbol)].copy()

        selected["product"] = selected["symbol"].map(self.product_prefix)
        selected = selected.dropna(subset=["datetime", "last_price"]).sort_values(["symbol", "datetime"])
        selected.attrs["selected_symbol"] = str(selected["symbol"].iloc[0]) if not selected.empty else ""
        selected.attrs["selected_product"] = str(selected["product"].iloc[0]) if not selected.empty else self.product
        selected.attrs["data_frequency"] = "tick"
        return selected.reset_index(drop=True)

    def _select_product(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.product:
            return df.copy()

        product = self.product.lower()
        product_values = df["symbol"].map(self.product_prefix).str.lower()
        selected = df[product_values.eq(product)].copy()
        if selected.empty:
            raise ValueError(f"Product {self.product!r} not found in tick data.")
        return selected

    @staticmethod
    def _main_contract_symbol(df: pd.DataFrame) -> str:
        if df.empty:
            raise ValueError("Cannot select main contract from an empty tick frame.")

        work = df.sort_values(["symbol", "datetime"]).copy()
        volume = pd.to_numeric(work["volume"], errors="coerce")
        volume_delta = volume.groupby(work["symbol"], sort=False).diff().fillna(0.0).clip(lower=0.0)
        scores = volume_delta.groupby(work["symbol"], sort=True).sum()
        if scores.notna().any() and float(scores.max()) > 0:
            return str(scores.idxmax())
        counts = work["symbol"].value_counts()
        return str(counts.idxmax())


class TickForwardReturnBuilder:
    """Builds non-overlapping-looking-ahead tick horizon labels by group."""

    def __init__(self, horizon_ticks: int = 1, price_col: str = "close"):
        self.horizon_ticks = max(int(horizon_ticks), 1)
        self.price_col = price_col

    def add_forward_return(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.price_col not in out.columns:
            raise ValueError(f"Cannot build tick forward_return: missing {self.price_col!r}.")

        group_keys = self._group_keys(out)
        price = pd.to_numeric(out[self.price_col], errors="coerce").replace(0, np.nan)
        future_price = price.groupby([out[key] for key in group_keys], sort=False).shift(-self.horizon_ticks)
        out["forward_return"] = future_price / price - 1.0
        return out

    @staticmethod
    def _group_keys(df: pd.DataFrame) -> list[str]:
        if "symbol" in df.columns and "_session_id" in df.columns:
            return ["symbol", "_session_id"]
        if "ticker" in df.columns and "_session_id" in df.columns:
            return ["ticker", "_session_id"]
        if "symbol" in df.columns:
            return ["symbol"]
        return ["ticker"]


def parquet_columns(path: str | Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set(pd.read_parquet(path, engine="pyarrow").columns)
