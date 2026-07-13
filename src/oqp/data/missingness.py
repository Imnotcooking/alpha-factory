"""Missing-data view builders for market data.

The default accounting treatment is capped forward-fill with explicit
freshness flags. It is meant for valuation and matrix survival, not alpha
generation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oqp.data.quality_flags import (
    QUALITY_FRESH,
    QUALITY_MISSING,
    QUALITY_STALE_EXPIRED,
    QUALITY_STALE_WITHIN_LIMIT,
)


@dataclass(frozen=True)
class ForwardFillConfig:
    timestamp_col: str = "date"
    asset_col: str = "ticker"
    value_cols: tuple[str, ...] = ("close",)
    max_stale_bars: int = 3


def build_accounting_view(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "date",
    asset_col: str = "ticker",
    value_cols: Sequence[str] = ("close",),
    max_stale_bars: int = 3,
    calendar: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build a forward-filled accounting view with freshness metadata.

    The returned frame is long-form and expands each asset to the shared
    timestamp grid. Synthetic values are capped by ``max_stale_bars`` and are
    always tagged with ``is_fresh=False``.
    """

    cols = tuple(dict.fromkeys(str(col) for col in value_cols))
    if not cols:
        raise ValueError("build_accounting_view requires at least one value column.")
    if max_stale_bars < 0:
        raise ValueError("max_stale_bars cannot be negative.")

    required = {timestamp_col, asset_col, *cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Accounting view missing required columns: {missing}")

    if frame.empty:
        return _empty_view(timestamp_col, asset_col, cols)

    work = frame[[timestamp_col, asset_col, *cols]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[asset_col] = work[asset_col].astype(str).str.strip()
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, asset_col])
    work = work[work[asset_col] != ""]
    if work.empty:
        return _empty_view(timestamp_col, asset_col, cols)

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
    if max_stale_bars == 0:
        filled = wide.copy()
    else:
        filled = wide.groupby(level=asset_col, sort=False).ffill(limit=max_stale_bars)

    out = filled.reset_index()
    flags = _freshness_frame(
        out[[timestamp_col, asset_col]],
        observed.reset_index(drop=True),
        timestamp_col=timestamp_col,
        asset_col=asset_col,
        max_stale_bars=max_stale_bars,
        filled_primary=out[primary_col],
    )
    out = pd.concat([out, flags], axis=1)
    out.attrs["view_type"] = "accounting"
    out.attrs["fill_policy"] = "ffill_with_freshness_flags"
    out.attrs["max_stale_bars"] = int(max_stale_bars)
    return out


def build_alpha_view(
    accounting_view: pd.DataFrame,
    *,
    value_cols: Sequence[str] = ("close",),
) -> pd.DataFrame:
    """Create an alpha-safe view by masking synthetic accounting marks."""

    cols = [str(col) for col in value_cols if col in accounting_view.columns]
    if "is_fresh" not in accounting_view.columns:
        raise ValueError("Alpha view requires an accounting view with is_fresh.")

    out = accounting_view.copy()
    stale_mask = ~out["is_fresh"].fillna(False).astype(bool)
    if cols:
        out.loc[stale_mask, cols] = np.nan
    out["alpha_can_update"] = ~stale_mask
    out["alpha_block_reason"] = np.where(stale_mask, out["quality_state"].astype(str), "")
    out.attrs.update(accounting_view.attrs)
    out.attrs["view_type"] = "alpha"
    return out


def _timestamp_grid(
    observed_timestamps: pd.Series,
    calendar: Iterable[pd.Timestamp] | None,
) -> pd.DatetimeIndex:
    if calendar is not None:
        timestamps = pd.to_datetime(pd.Index(list(calendar)), errors="coerce").dropna().unique()
        return pd.DatetimeIndex(sorted(timestamps), name=observed_timestamps.name)
    return pd.DatetimeIndex(sorted(observed_timestamps.dropna().unique()), name=observed_timestamps.name)


def _freshness_frame(
    keys: pd.DataFrame,
    observed: pd.Series,
    *,
    timestamp_col: str,
    asset_col: str,
    max_stale_bars: int,
    filled_primary: pd.Series,
) -> pd.DataFrame:
    flags = keys.reset_index(drop=True).copy()
    flags["_observed"] = observed.fillna(False).astype(bool).to_numpy()
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
    is_synthetic = (
        missing_after_obs
        & filled_primary.notna().reset_index(drop=True)
        & stale_bars.le(max_stale_bars).fillna(False)
    )
    stale_expired = missing_after_obs & ~is_synthetic
    gap_id = pd.Series(pd.NA, index=flags.index, dtype="object")
    gap_id.loc[missing_after_obs] = (
        flags.loc[missing_after_obs, asset_col].astype(str)
        + ":"
        + flags.loc[missing_after_obs, "_obs_group"].astype("int64").astype(str)
    )

    quality_state = np.select(
        [
            flags["_observed"].to_numpy(dtype=bool),
            is_synthetic.to_numpy(dtype=bool),
            stale_expired.to_numpy(dtype=bool),
        ],
        [QUALITY_FRESH, QUALITY_STALE_WITHIN_LIMIT, QUALITY_STALE_EXPIRED],
        default=QUALITY_MISSING,
    )
    fill_method = np.where(is_synthetic, "ffill", "none")

    return pd.DataFrame(
        {
            "is_fresh": flags["_observed"].astype(bool),
            "is_synthetic": is_synthetic.astype(bool),
            "stale_bars": stale_bars,
            "last_observed_ts": pd.to_datetime(last_observed_ts, errors="coerce"),
            "gap_id": gap_id,
            "fill_method": fill_method,
            "quality_state": quality_state,
        }
    )


def _empty_view(timestamp_col: str, asset_col: str, value_cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
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
        ]
    )
