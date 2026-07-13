"""Realized-volatility diagnostics across market-data imputation views."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from oqp.data.brownian_bridge import BrownianBridgeConfig
from oqp.data.views import build_market_data_views


@dataclass(frozen=True)
class RealizedVolatilityConfig:
    timestamp_col: str = "date"
    asset_col: str = "ticker"
    price_col: str = "close"
    annualization: float = 252.0
    max_stale_bars: int = 3
    bridge_max_gap_bars: int = 20
    bridge_seed: int = 42
    zero_return_tolerance: float = 1e-12


def realized_volatility_by_asset(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "date",
    asset_col: str = "ticker",
    price_col: str = "close",
    annualization: float = 252.0,
    zero_return_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return close-to-close realized-volatility diagnostics by asset.

    ``annualized_vol`` uses the root-mean-square log return scaled by
    ``sqrt(annualization)``. Set ``annualization=1`` for per-bar comparisons
    across raw, forward-filled, and Brownian Bridge views.
    """

    required = {timestamp_col, asset_col, price_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Realized volatility input missing required columns: {missing}")
    if annualization <= 0:
        raise ValueError("annualization must be positive.")

    if frame.empty:
        return pd.DataFrame(
            columns=[
                asset_col,
                "observations",
                "return_observations",
                "realized_variance",
                "step_vol",
                "annualized_vol",
                "zero_return_pct",
                "missing_return_pct",
            ]
        )

    work = frame[[timestamp_col, asset_col, price_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[asset_col] = work[asset_col].astype(str).str.strip()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, asset_col])
    work = work[work[asset_col] != ""]
    if work.empty:
        return realized_volatility_by_asset(
            pd.DataFrame(columns=[timestamp_col, asset_col, price_col]),
            timestamp_col=timestamp_col,
            asset_col=asset_col,
            price_col=price_col,
            annualization=annualization,
            zero_return_tolerance=zero_return_tolerance,
        )

    prices = (
        work.sort_values([asset_col, timestamp_col])
        .groupby([timestamp_col, asset_col], as_index=False)[price_col]
        .last()
        .pivot(index=timestamp_col, columns=asset_col, values=price_col)
        .sort_index()
    )
    prices = prices.where(prices > 0.0)
    returns = np.log(prices).diff()

    rows: list[dict[str, Any]] = []
    for asset in returns.columns:
        asset_returns = pd.to_numeric(returns[asset], errors="coerce")
        valid = asset_returns.dropna()
        observations = int(prices[asset].notna().sum())
        return_observations = int(valid.shape[0])
        if return_observations == 0:
            realized_variance = np.nan
            step_vol = np.nan
            annualized_vol = np.nan
            zero_return_pct = np.nan
        else:
            values = valid.to_numpy(dtype="float64")
            realized_variance = float(np.square(values).sum())
            step_vol = float(np.sqrt(np.square(values).mean()))
            annualized_vol = float(step_vol * np.sqrt(annualization))
            zero_return_pct = float((np.abs(values) <= zero_return_tolerance).mean())
        rows.append(
            {
                asset_col: asset,
                "observations": observations,
                "return_observations": return_observations,
                "realized_variance": realized_variance,
                "step_vol": step_vol,
                "annualized_vol": annualized_vol,
                "zero_return_pct": zero_return_pct,
                "missing_return_pct": float(asset_returns.isna().mean()) if len(asset_returns) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compare_risk_imputation_views(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "date",
    asset_col: str = "ticker",
    price_col: str = "close",
    max_stale_bars: int = 3,
    bridge_max_gap_bars: int = 20,
    bridge_seed: int = 42,
    annualization: float = 252.0,
    calendar: Iterable[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """Compare raw, forward-filled, and Brownian Bridge realized-vol views."""

    ffill_views = build_market_data_views(
        frame,
        timestamp_col=timestamp_col,
        asset_col=asset_col,
        price_cols=(price_col,),
        max_stale_bars=max_stale_bars,
        calendar=calendar,
        risk_imputation="ffill",
    )
    bridge_views = build_market_data_views(
        frame,
        timestamp_col=timestamp_col,
        asset_col=asset_col,
        price_cols=(price_col,),
        max_stale_bars=max_stale_bars,
        calendar=calendar,
        risk_imputation="brownian_bridge",
        bridge_config=BrownianBridgeConfig(
            timestamp_col=timestamp_col,
            asset_col=asset_col,
            value_cols=(price_col,),
            max_gap_bars=bridge_max_gap_bars,
            seed=bridge_seed,
        ),
    )

    raw_vol = _rename_mode_columns(
        realized_volatility_by_asset(
            frame,
            timestamp_col=timestamp_col,
            asset_col=asset_col,
            price_col=price_col,
            annualization=annualization,
        ),
        asset_col,
        "raw",
    )
    ffill_vol = _rename_mode_columns(
        realized_volatility_by_asset(
            ffill_views.risk,
            timestamp_col=timestamp_col,
            asset_col=asset_col,
            price_col=price_col,
            annualization=annualization,
        ),
        asset_col,
        "ffill",
    )
    bridge_vol = _rename_mode_columns(
        realized_volatility_by_asset(
            bridge_views.risk,
            timestamp_col=timestamp_col,
            asset_col=asset_col,
            price_col=price_col,
            annualization=annualization,
        ),
        asset_col,
        "bridge",
    )

    asset_summary = raw_vol.merge(ffill_vol, on=asset_col, how="outer").merge(
        bridge_vol,
        on=asset_col,
        how="outer",
    )
    asset_summary["bridge_vs_ffill_pct"] = _relative_change(
        asset_summary.get("bridge_annualized_vol"),
        asset_summary.get("ffill_annualized_vol"),
    )
    asset_summary["ffill_vs_raw_pct"] = _relative_change(
        asset_summary.get("ffill_annualized_vol"),
        asset_summary.get("raw_annualized_vol"),
    )

    bridge_synthetic = (
        bridge_views.risk.get("is_synthetic", pd.Series(False, index=bridge_views.risk.index))
        .fillna(False)
        .astype(bool)
    )
    summary = {
        "asset_count": int(asset_summary[asset_col].nunique(dropna=True)) if asset_col in asset_summary else 0,
        "raw_median_rv": _nanmedian(asset_summary.get("raw_annualized_vol")),
        "ffill_median_rv": _nanmedian(asset_summary.get("ffill_annualized_vol")),
        "bridge_median_rv": _nanmedian(asset_summary.get("bridge_annualized_vol")),
        "bridge_vs_ffill_pct": _safe_ratio_delta(
            _nanmedian(asset_summary.get("bridge_annualized_vol")),
            _nanmedian(asset_summary.get("ffill_annualized_vol")),
        ),
        "ffill_zero_return_pct": _nanmedian(asset_summary.get("ffill_zero_return_pct")),
        "bridge_zero_return_pct": _nanmedian(asset_summary.get("bridge_zero_return_pct")),
        "bridge_synthetic_rows": int(bridge_synthetic.sum()),
        "bridge_synthetic_pct": float(bridge_synthetic.mean()) if len(bridge_synthetic) else np.nan,
        "bridge_expired_rows": int(
            bridge_views.risk.get("quality_state", pd.Series("", index=bridge_views.risk.index))
            .astype(str)
            .eq("stale_expired")
            .sum()
        ),
    }
    return {
        "asset_summary": asset_summary,
        "summary": summary,
        "ffill_quality": ffill_views.quality_summary,
        "bridge_attrs": dict(bridge_views.risk.attrs),
    }


def _rename_mode_columns(frame: pd.DataFrame, asset_col: str, mode: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[asset_col])
    rename = {
        col: f"{mode}_{col}"
        for col in [
            "observations",
            "return_observations",
            "realized_variance",
            "step_vol",
            "annualized_vol",
            "zero_return_pct",
            "missing_return_pct",
        ]
        if col in frame.columns
    }
    return frame.rename(columns=rename)


def _relative_change(numerator: pd.Series | None, denominator: pd.Series | None) -> pd.Series:
    if numerator is None or denominator is None:
        return pd.Series(dtype="float64")
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.where(denominator.abs() > 0.0, numerator / denominator - 1.0, np.nan)


def _nanmedian(values: pd.Series | None) -> float:
    if values is None:
        return np.nan
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(np.nanmedian(numeric.to_numpy(dtype="float64")))


def _safe_ratio_delta(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return float(numerator / denominator - 1.0)
