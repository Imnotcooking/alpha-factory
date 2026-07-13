"""Brownian Bridge risk-view reconstruction for missing market data.

This module is intentionally risk-only. It creates synthetic paths inside
closed gaps so risk models do not see artificial zero volatility from flat
forward-filled marks. Alpha and execution logic should continue to use fresh
observed data guards.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oqp.data.quality_flags import (
    QUALITY_BRIDGE_SYNTHETIC,
    QUALITY_FRESH,
    QUALITY_MISSING,
    QUALITY_STALE_EXPIRED,
)


@dataclass(frozen=True)
class BrownianBridgeConfig:
    timestamp_col: str = "date"
    asset_col: str = "ticker"
    value_cols: tuple[str, ...] = ("close",)
    max_gap_bars: int = 20
    seed: int = 42
    path_id: int = 0
    sigma_floor: float = 1e-8


def build_brownian_bridge_view(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "date",
    asset_col: str = "ticker",
    value_cols: Sequence[str] = ("close",),
    max_gap_bars: int = 20,
    calendar: Iterable[pd.Timestamp] | None = None,
    seed: int = 42,
    path_id: int = 0,
    sigma_floor: float = 1e-8,
) -> pd.DataFrame:
    """Build a risk-only view with Brownian Bridge gap reconstruction.

    The returned frame is long-form on the requested timestamp/asset grid.
    Observed prices are preserved exactly. Missing values are reconstructed only
    when they sit between two observed positive endpoints and the closed gap is
    no longer than ``max_gap_bars``.
    """

    cols = tuple(dict.fromkeys(str(col) for col in value_cols))
    if not cols:
        raise ValueError("build_brownian_bridge_view requires at least one value column.")
    if max_gap_bars < 0:
        raise ValueError("max_gap_bars cannot be negative.")
    if sigma_floor < 0:
        raise ValueError("sigma_floor cannot be negative.")

    required = {timestamp_col, asset_col, *cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Brownian Bridge view missing required columns: {missing}")

    if frame.empty:
        return _empty_bridge_view(timestamp_col, asset_col, cols)

    work = frame[[timestamp_col, asset_col, *cols]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[asset_col] = work[asset_col].astype(str).str.strip()
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, asset_col])
    work = work[work[asset_col] != ""]
    if work.empty:
        return _empty_bridge_view(timestamp_col, asset_col, cols)

    work = (
        work.sort_values([asset_col, timestamp_col])
        .groupby([timestamp_col, asset_col], as_index=False)[list(cols)]
        .last()
    )
    timestamps = _timestamp_grid(work[timestamp_col], calendar)
    assets = pd.Index(sorted(work[asset_col].dropna().astype(str).unique()), name=asset_col)
    full_index = pd.MultiIndex.from_product(
        [timestamps, assets],
        names=[timestamp_col, asset_col],
    )

    wide = work.set_index([timestamp_col, asset_col]).sort_index()[list(cols)].reindex(full_index)
    primary_col = cols[0]
    observed = wide[primary_col].notna()
    bridged = wide.copy()
    synthetic = pd.Series(False, index=wide.index, dtype=bool)
    bridge_sigma = pd.Series(np.nan, index=wide.index, dtype="float64")
    bridge_step = pd.Series(pd.NA, index=wide.index, dtype="Int64")
    bridge_steps = pd.Series(pd.NA, index=wide.index, dtype="Int64")
    bridge_gap_id = pd.Series(pd.NA, index=wide.index, dtype="object")

    rng = np.random.default_rng(seed)
    global_sigmas = {
        col: _global_sigma_step(wide[col], assets, sigma_floor=sigma_floor) for col in cols
    }

    for asset in assets:
        asset_frame = wide.xs(asset, level=asset_col, drop_level=True)[list(cols)]
        asset_observed = asset_frame[primary_col].notna().to_numpy()
        observed_locs = np.flatnonzero(asset_observed)
        if len(observed_locs) < 2:
            continue

        for gap_number, (left, right) in enumerate(zip(observed_locs[:-1], observed_locs[1:]), start=1):
            gap_len = int(right - left - 1)
            if gap_len <= 0 or gap_len > max_gap_bars:
                continue

            fill_positions = np.arange(left + 1, right)
            fill_timestamps = asset_frame.index[fill_positions]
            filled_any_primary = False
            primary_sigma = np.nan

            for col in cols:
                series = asset_frame[col]
                left_value = series.iloc[left]
                right_value = series.iloc[right]
                if not _can_bridge_endpoint(left_value) or not _can_bridge_endpoint(right_value):
                    continue

                sigma_step = _asset_sigma_step(series, sigma_floor=global_sigmas[col])
                path = _bridge_log_prices(
                    float(left_value),
                    float(right_value),
                    gap_len,
                    sigma_step=sigma_step,
                    rng=rng,
                )
                bridged.loc[pd.MultiIndex.from_product([fill_timestamps, [asset]]), col] = path
                if col == primary_col:
                    filled_any_primary = True
                    primary_sigma = sigma_step

            if filled_any_primary:
                fill_index = pd.MultiIndex.from_product([fill_timestamps, [asset]])
                synthetic.loc[fill_index] = True
                bridge_sigma.loc[fill_index] = primary_sigma
                bridge_step.loc[fill_index] = np.arange(1, gap_len + 1, dtype="int64")
                bridge_steps.loc[fill_index] = gap_len
                bridge_gap_id.loc[fill_index] = f"{asset}:bridge:{gap_number}"

    out = bridged.reset_index()
    flags = _bridge_flags(
        out[[timestamp_col, asset_col]],
        observed.reset_index(drop=True),
        synthetic.reset_index(drop=True),
        bridge_sigma.reset_index(drop=True),
        bridge_step.reset_index(drop=True),
        bridge_steps.reset_index(drop=True),
        bridge_gap_id.reset_index(drop=True),
        timestamp_col=timestamp_col,
        asset_col=asset_col,
    )
    out = pd.concat([out, flags], axis=1)
    out.attrs["view_type"] = "risk"
    out.attrs["fill_policy"] = "brownian_bridge_risk_only"
    out.attrs["risk_imputation"] = "brownian_bridge"
    out.attrs["max_bridge_gap_bars"] = int(max_gap_bars)
    out.attrs["bridge_seed"] = int(seed)
    out.attrs["bridge_path_id"] = int(path_id)
    return out


def build_brownian_bridge_view_from_config(
    frame: pd.DataFrame,
    config: BrownianBridgeConfig,
    *,
    calendar: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build a Brownian Bridge view from a reusable config object."""

    return build_brownian_bridge_view(
        frame,
        timestamp_col=config.timestamp_col,
        asset_col=config.asset_col,
        value_cols=config.value_cols,
        max_gap_bars=config.max_gap_bars,
        calendar=calendar,
        seed=config.seed,
        path_id=config.path_id,
        sigma_floor=config.sigma_floor,
    )


def _timestamp_grid(
    observed_timestamps: pd.Series,
    calendar: Iterable[pd.Timestamp] | None,
) -> pd.DatetimeIndex:
    if calendar is not None:
        timestamps = pd.to_datetime(pd.Index(list(calendar)), errors="coerce").dropna().unique()
        return pd.DatetimeIndex(sorted(timestamps), name=observed_timestamps.name)
    return pd.DatetimeIndex(sorted(observed_timestamps.dropna().unique()), name=observed_timestamps.name)


def _can_bridge_endpoint(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric) and numeric > 0.0


def _bridge_log_prices(
    start_price: float,
    end_price: float,
    gap_len: int,
    *,
    sigma_step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    intervals = gap_len + 1
    steps = np.arange(1, intervals, dtype="float64")
    weight = steps / float(intervals)
    start_log = np.log(start_price)
    end_log = np.log(end_price)
    expected = start_log + weight * (end_log - start_log)

    if sigma_step <= 0.0:
        return np.exp(expected)

    increments = rng.normal(loc=0.0, scale=sigma_step, size=intervals)
    walk = np.cumsum(increments)
    noise = walk[:-1] - weight * walk[-1]
    return np.exp(expected + noise)


def _global_sigma_step(
    series: pd.Series,
    assets: pd.Index,
    *,
    sigma_floor: float,
) -> float:
    sigmas = []
    for asset in assets:
        sigma = _asset_sigma_step(
            series.xs(asset, level=series.index.names[1], drop_level=True),
            sigma_floor=np.nan,
        )
        if np.isfinite(sigma) and sigma > 0.0:
            sigmas.append(float(sigma))
    if sigmas:
        return max(float(np.nanmedian(sigmas)), sigma_floor)
    return sigma_floor


def _asset_sigma_step(series: pd.Series, *, sigma_floor: float) -> float:
    valid = pd.to_numeric(series, errors="coerce")
    valid = valid[valid.gt(0.0)]
    if valid.shape[0] < 3:
        return 0.0 if np.isnan(sigma_floor) else float(sigma_floor)

    positions = np.flatnonzero(series.index.isin(valid.index))
    logs = np.log(valid.to_numpy(dtype="float64"))
    step_counts = np.diff(positions).astype("float64")
    returns = np.diff(logs)
    scaled = returns / np.sqrt(np.maximum(step_counts, 1.0))
    scaled = scaled[np.isfinite(scaled)]
    if scaled.shape[0] < 2:
        return 0.0 if np.isnan(sigma_floor) else float(sigma_floor)
    sigma = float(np.nanstd(scaled, ddof=1))
    if not np.isfinite(sigma):
        return 0.0 if np.isnan(sigma_floor) else float(sigma_floor)
    if np.isnan(sigma_floor):
        return sigma
    return max(sigma, float(sigma_floor))


def _bridge_flags(
    keys: pd.DataFrame,
    observed: pd.Series,
    synthetic: pd.Series,
    bridge_sigma: pd.Series,
    bridge_step: pd.Series,
    bridge_steps: pd.Series,
    bridge_gap_id: pd.Series,
    *,
    timestamp_col: str,
    asset_col: str,
) -> pd.DataFrame:
    flags = keys.reset_index(drop=True).copy()
    flags["_observed"] = observed.fillna(False).astype(bool).to_numpy()
    flags["_synthetic"] = synthetic.fillna(False).astype(bool).to_numpy()
    flags["_obs_group"] = flags.groupby(asset_col, sort=False)["_observed"].cumsum()
    missing_after_obs = (~flags["_observed"]) & flags["_obs_group"].gt(0)

    stale_bars = pd.Series(pd.NA, index=flags.index, dtype="Int64")
    stale_bars.loc[flags["_observed"]] = 0
    if missing_after_obs.any():
        stale_bars.loc[missing_after_obs] = (
            flags.loc[missing_after_obs]
            .groupby([asset_col, "_obs_group"], sort=False)
            .cumcount()
            .add(1)
            .astype("Int64")
        )

    last_observed_ts = flags[timestamp_col].where(flags["_observed"])
    last_observed_ts = last_observed_ts.groupby(flags[asset_col], sort=False).ffill()
    gap_id = pd.Series(pd.NA, index=flags.index, dtype="object")
    gap_id.loc[missing_after_obs] = (
        flags.loc[missing_after_obs, asset_col].astype(str)
        + ":"
        + flags.loc[missing_after_obs, "_obs_group"].astype("int64").astype(str)
    )
    gap_id.loc[flags["_synthetic"]] = bridge_gap_id.loc[flags["_synthetic"]].astype("object")

    quality_state = np.select(
        [
            flags["_observed"].to_numpy(dtype=bool),
            flags["_synthetic"].to_numpy(dtype=bool),
            missing_after_obs.to_numpy(dtype=bool),
        ],
        [QUALITY_FRESH, QUALITY_BRIDGE_SYNTHETIC, QUALITY_STALE_EXPIRED],
        default=QUALITY_MISSING,
    )
    fill_method = np.where(flags["_synthetic"], "brownian_bridge", "none")

    return pd.DataFrame(
        {
            "is_fresh": flags["_observed"].astype(bool),
            "is_synthetic": flags["_synthetic"].astype(bool),
            "stale_bars": stale_bars,
            "last_observed_ts": pd.to_datetime(last_observed_ts, errors="coerce"),
            "gap_id": gap_id,
            "fill_method": fill_method,
            "quality_state": quality_state,
            "bridge_sigma": bridge_sigma,
            "bridge_step": bridge_step.astype("Int64"),
            "bridge_steps": bridge_steps.astype("Int64"),
        }
    )


def _empty_bridge_view(timestamp_col: str, asset_col: str, value_cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(
        columns=[
            timestamp_col,
            asset_col,
            *value_cols,
            "is_fresh",
            "is_synthetic",
            "stale_bars",
            "last_observed_ts",
            "gap_id",
            "fill_method",
            "quality_state",
            "bridge_sigma",
            "bridge_step",
            "bridge_steps",
        ]
    )
    out.attrs["view_type"] = "risk"
    out.attrs["fill_policy"] = "brownian_bridge_risk_only"
    out.attrs["risk_imputation"] = "brownian_bridge"
    return out
