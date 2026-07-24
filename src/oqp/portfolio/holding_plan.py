"""Persistent discretionary labels and ATR reference levels for live holdings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_HOLDING_PLAN_PATH = REPO_ROOT / "runtime" / "state" / "portfolio" / "holding_plans.json"
HOLDING_STYLES = ("", "Trading", "Investing")


def holding_plan_key(broker: Any, symbol: Any) -> str:
    return f"{str(broker or 'unknown').strip().lower()}|{str(symbol or '').strip().upper()}"


def load_holding_styles(path: str | Path = DEFAULT_HOLDING_PLAN_PATH) -> dict[str, str]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    values = payload.get("styles", payload) if isinstance(payload, dict) else {}
    if not isinstance(values, dict):
        raise ValueError("Holding plan JSON must contain a styles object.")
    return {
        str(key): str(value)
        for key, value in values.items()
        if str(value) in HOLDING_STYLES and str(value)
    }


def save_holding_styles(
    styles: dict[str, str],
    path: str | Path = DEFAULT_HOLDING_PLAN_PATH,
) -> Path:
    cleaned: dict[str, str] = {}
    for key, value in styles.items():
        style = str(value or "")
        if style not in HOLDING_STYLES:
            raise ValueError(f"Unsupported holding style: {style}")
        if style:
            cleaned[str(key)] = style
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps({"styles": cleaned}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def add_holding_plan_columns(
    holdings: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    styles: dict[str, str] | None = None,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Add editable style plus 1x ATR TP/SL reference prices."""

    if holdings.empty:
        return holdings.copy()
    out = holdings.copy()
    style_map = styles or {}
    out["Plan Key"] = out.apply(
        lambda row: holding_plan_key(row.get("Broker"), row.get("Symbol")), axis=1
    )
    # New option rows start as Trading; other new/unclassified holdings start as
    # Investing. Any explicit saved choice always wins.
    asset_classes = out.get("Asset Class", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    default_styles = pd.Series("Investing", index=out.index)
    default_styles.loc[asset_classes.str.contains("option", regex=False)] = "Trading"
    out["Holding Style"] = out["Plan Key"].map(style_map).fillna(default_styles)

    atr_by_symbol = _latest_atr_by_symbol(price_history, window=atr_window)
    take_profit: list[float | None] = []
    stop_loss: list[float | None] = []
    for _, row in out.iterrows():
        asset_class = str(row.get("Asset Class") or "").lower()
        symbol = str(row.get("Symbol") or "").upper().strip()
        market_price = pd.to_numeric(pd.Series([row.get("Market Price")]), errors="coerce").iloc[0]
        quantity = pd.to_numeric(pd.Series([row.get("Quantity")]), errors="coerce").iloc[0]
        atr = atr_by_symbol.get(symbol)
        valid_asset = "equity" in asset_class or "stock" in asset_class or "etf" in asset_class
        if not valid_asset or pd.isna(market_price) or atr is None or atr <= 0:
            take_profit.append(None)
            stop_loss.append(None)
            continue
        direction = -1.0 if pd.notna(quantity) and float(quantity) < 0 else 1.0
        take_profit.append(float(market_price) + direction * atr)
        stop_loss.append(float(market_price) - direction * atr)
    out["TP 1x ATR"] = take_profit
    out["SL 1x ATR"] = stop_loss
    return out


def _latest_atr_by_symbol(price_history: pd.DataFrame, *, window: int) -> dict[str, float]:
    if price_history.empty:
        return {}
    frame = price_history.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"symbol", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return {}
    output: dict[str, float] = {}
    for symbol, group in frame.groupby(frame["symbol"].astype(str).str.upper().str.strip()):
        ordered = group.sort_values("date") if "date" in group else group
        high = pd.to_numeric(ordered["high"], errors="coerce")
        low = pd.to_numeric(ordered["low"], errors="coerce")
        close = pd.to_numeric(ordered["close"], errors="coerce")
        true_range = pd.concat(
            [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(max(int(window), 1)).mean().dropna()
        if not atr.empty and pd.notna(atr.iloc[-1]):
            output[str(symbol)] = float(atr.iloc[-1])
    return output
