"""Portfolio margin-budget controls for futures research targets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from oqp.data.instruments import InstrumentMaster


DEFAULT_TARGET_COLUMNS = (
    "routed_target_weight",
    "final_target_weight",
    "target_weight",
    "signal",
)


def apply_margin_utilization_cap(
    frame: pd.DataFrame,
    *,
    market_vertical: str,
    source_weight_col: str,
    max_margin_utilization: float | None,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
) -> pd.DataFrame:
    """Scale futures targets so configured initial margin stays below capital."""

    out = frame.copy()
    out.attrs.update(frame.attrs)
    if max_margin_utilization is None:
        out.attrs["margin_budget_status"] = "disabled"
        return out
    limit = float(max_margin_utilization)
    if not 0.0 < limit <= 1.0:
        raise ValueError("max_margin_utilization must be between zero and one")
    if not str(market_vertical).upper().startswith("FUTURES_"):
        raise ValueError("margin utilization caps currently support futures only")
    required = {"date", "ticker", source_weight_col}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"margin utilization cap is missing columns: {missing}")

    master = InstrumentMaster(market_vertical)
    fallback_rates = out["ticker"].astype(str).map(
        lambda ticker: master.get_profile(ticker).margin_rate
    )
    if "margin_rate" in out.columns:
        rates = pd.to_numeric(out["margin_rate"], errors="coerce")
        out["margin_rate"] = rates.fillna(fallback_rates).astype(float)
    else:
        out["margin_rate"] = fallback_rates.astype(float)
    if out["margin_rate"].le(0.0).any():
        raise ValueError("margin rates must be positive")

    source = pd.to_numeric(out[source_weight_col], errors="coerce").fillna(0.0)
    out["pre_margin_target_weight"] = source
    out["pre_margin_contribution"] = source.abs() * out["margin_rate"]
    daily_margin = out.groupby("date")["pre_margin_contribution"].transform("sum")
    safe_margin = daily_margin.where(daily_margin.gt(0.0), np.nan)
    out["margin_scale"] = np.minimum(1.0, limit / safe_margin).fillna(1.0)
    out["margin_cap_bound"] = out["margin_scale"].lt(1.0 - 1e-12)

    scaled_columns: list[str] = []
    for column in dict.fromkeys((*target_columns, source_weight_col)):
        if column not in out.columns:
            continue
        backup = f"pre_margin_{column}"
        if backup not in out.columns:
            out[backup] = out[column]
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        out[column] = values * out["margin_scale"]
        scaled_columns.append(column)

    final_source = pd.to_numeric(
        out[source_weight_col], errors="coerce"
    ).fillna(0.0)
    out["margin_contribution"] = final_source.abs() * out["margin_rate"]
    out["margin_utilization"] = out.groupby("date")[
        "margin_contribution"
    ].transform("sum")
    out.attrs["margin_budget_status"] = "applied"
    out.attrs["max_margin_utilization"] = limit
    out.attrs["minimum_cash_reserve"] = 1.0 - limit
    out.attrs["margin_scaled_weight_columns"] = scaled_columns
    return out


__all__ = ["apply_margin_utilization_cap"]
