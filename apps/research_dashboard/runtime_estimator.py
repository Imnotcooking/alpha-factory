from __future__ import annotations

import math

import pandas as pd

try:
    from oqp.contracts.market_vertical import ASSET_TAXONOMY
except Exception:  # pragma: no cover - dashboard standalone fallback
    ASSET_TAXONOMY = {}


def runtime_estimate_frame(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact runtime guide from recent research ledger rows."""

    columns = ["Asset", "Rows", "Frequency", "Route", "Estimate", "Notes"]
    if runs_df.empty or "asset_class" not in runs_df.columns:
        return pd.DataFrame(columns=columns)

    frame = runs_df.copy()
    frame["asset_class"] = frame["asset_class"].fillna("UNKNOWN").astype(str)
    frame["_row_count"] = _numeric(frame.get("validation_rows")) + _numeric(frame.get("holdout_rows"))
    frame["_row_count"] = frame["_row_count"].where(frame["_row_count"] > 0, _numeric(frame.get("sample_size")))
    frame["_row_count"] = frame["_row_count"].fillna(0).astype(int)
    if "timestamp" in frame.columns:
        frame["_timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("_timestamp", ascending=False, na_position="last")

    rows = []
    for asset_class, group in frame.groupby("asset_class", sort=True):
        latest = group.iloc[0]
        row_count = int(latest.get("_row_count") or 0)
        frequency = str(latest.get("data_frequency") or "daily")
        execution_mode = str(latest.get("execution_mode") or "")
        estimate, note = estimate_runtime(row_count, asset_class, frequency, execution_mode)
        taxonomy = ASSET_TAXONOMY.get(asset_class, {})
        route = str(taxonomy.get("backtest_route") or ("event_driven_options" if "OPTIONS" in asset_class else "vectorized"))
        rows.append(
            {
                "Asset": asset_class,
                "Rows": f"{row_count:,}" if row_count else "Unknown",
                "Frequency": frequency,
                "Route": route,
                "Estimate": estimate,
                "Notes": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def estimate_runtime(row_count: int, asset_class: str, frequency: str, execution_mode: str = "") -> tuple[str, str]:
    """Conservative wall-clock estimate for one research backtest."""

    if row_count <= 0:
        return "Unknown", "No row-count metadata yet"

    asset = str(asset_class or "").upper()
    freq = str(frequency or "daily").lower()
    mode = str(execution_mode or "").lower()
    rows_m = max(row_count / 1_000_000.0, 0.001)

    if "OPTIONS" in asset:
        low, high = rows_m * 240, rows_m * 900
        note = "Event-driven option-chain route"
    elif freq == "tick":
        low, high = rows_m * 60, rows_m * 240
        note = "Tick route; prefiltering matters"
    elif freq == "intraday":
        low, high = rows_m * 45, rows_m * 180
        note = "Intraday panel; factor cost dominates"
    elif "EQUITY_CN" in asset:
        low, high = rows_m * 30, rows_m * 75
        note = "Full A-share panels are factor-compute heavy"
    elif "direct" in mode:
        low, high = rows_m * 12, rows_m * 45
        note = "Direct weights; execution is vectorized"
    else:
        low, high = rows_m * 8, rows_m * 35
        note = "Vectorized route; factor complexity drives spread"

    return f"{_format_seconds(low)}-{_format_seconds(high)}", note


def _numeric(series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 1.0)
    if seconds < 90:
        return f"{math.ceil(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"
