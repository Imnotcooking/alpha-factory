"""Chinese futures market-data normalization helpers."""

from __future__ import annotations

import pandas as pd


FUTURES_CN_DAILY_REQUIRED_COLUMNS = {"date", "ticker", "close"}

_COLUMN_ALIASES = {
    "date": ("date", "datetime", "trade_date", "trading_date", "交易日期"),
    "ticker": ("ticker", "symbol", "contract", "code", "wind_code", "合约代码"),
    "close": ("close", "settle", "settlement", "收盘价", "结算价"),
    "open": ("open", "开盘价"),
    "high": ("high", "最高价"),
    "low": ("low", "最低价"),
    "volume": ("volume", "vol", "成交量"),
    "turnover": ("turnover", "amount", "成交额"),
    "open_interest": ("open_interest", "oi", "持仓量"),
    "exchange": ("exchange", "交易所"),
}


def normalize_futures_cn_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CN futures daily vendor schemas to OQP columns."""

    work = df.copy()
    rename_map: dict[str, str] = {}
    lower_to_original = {str(col).strip().lower(): col for col in work.columns}

    for target, aliases in _COLUMN_ALIASES.items():
        if target in work.columns:
            continue
        for alias in aliases:
            original = lower_to_original.get(alias.lower())
            if original is not None:
                rename_map[original] = target
                break

    if rename_map:
        work = work.rename(columns=rename_map)

    missing = sorted(FUTURES_CN_DAILY_REQUIRED_COLUMNS - set(work.columns))
    if missing:
        raise ValueError(f"Chinese futures daily data is missing columns: {missing}")

    if "oi" not in work.columns and "open_interest" in work.columns:
        work["oi"] = work["open_interest"]
    if "open_interest" not in work.columns and "oi" in work.columns:
        work["open_interest"] = work["oi"]

    cols = [
        col
        for col in [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "oi",
            "open_interest",
            "exchange",
            "sector",
        ]
        if col in work.columns
    ]
    out = work[cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.strip()

    for col in ["open", "high", "low", "close", "volume", "turnover", "oi", "open_interest"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "ticker", "close"])
    out = out[(out["ticker"] != "") & (out["close"] > 0)]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
