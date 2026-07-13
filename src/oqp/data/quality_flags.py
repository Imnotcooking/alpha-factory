"""Quality flags for derived market-data views.

Raw vendor data should stay untouched. These helpers describe derived rows so
downstream accounting, alpha, and risk modules can tell observed data from
synthetic marks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


QUALITY_FRESH = "fresh"
QUALITY_STALE_WITHIN_LIMIT = "stale_within_limit"
QUALITY_STALE_EXPIRED = "stale_expired"
QUALITY_MISSING = "missing"
QUALITY_BRIDGE_SYNTHETIC = "bridge_synthetic"

QUALITY_COLUMNS = (
    "is_fresh",
    "is_synthetic",
    "stale_bars",
    "last_observed_ts",
    "gap_id",
    "fill_method",
    "quality_state",
)


def alpha_fresh_mask(
    frame: pd.DataFrame,
    *,
    required_cols: Sequence[str] | None = None,
    fresh_col: str = "is_fresh",
) -> pd.Series:
    """Return rows that are safe for alpha updates.

    A row is alpha-safe only when it is fresh and all requested input columns are
    present. This deliberately rejects forward-filled accounting marks.
    """

    if fresh_col not in frame.columns:
        raise ValueError(f"Alpha freshness guard requires {fresh_col!r}.")

    mask = frame[fresh_col].fillna(False).astype(bool)
    for col in required_cols or ():
        if col not in frame.columns:
            raise ValueError(f"Alpha freshness guard missing required column {col!r}.")
        mask &= frame[col].notna()
    return mask


def require_fresh_alpha_inputs(
    frame: pd.DataFrame,
    *,
    required_cols: Sequence[str] | None = None,
    fresh_col: str = "is_fresh",
) -> pd.DataFrame:
    """Return only rows that can be used by alpha logic."""

    return frame.loc[
        alpha_fresh_mask(frame, required_cols=required_cols, fresh_col=fresh_col)
    ].copy()


def summarize_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize standard quality flags for dashboards and audit metadata."""

    rows = int(len(frame))
    if rows == 0:
        return {
            "rows": 0,
            "fresh_rows": 0,
            "synthetic_rows": 0,
            "stale_expired_rows": 0,
            "missing_rows": 0,
            "fresh_pct": 0.0,
            "synthetic_pct": 0.0,
            "max_stale_bars": 0,
        }

    fresh = frame.get("is_fresh", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    synthetic = (
        frame.get("is_synthetic", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    )
    state = frame.get("quality_state", pd.Series("", index=frame.index)).astype(str)
    stale_bars = pd.to_numeric(frame.get("stale_bars", pd.Series(0, index=frame.index)), errors="coerce")

    return {
        "rows": rows,
        "fresh_rows": int(fresh.sum()),
        "synthetic_rows": int(synthetic.sum()),
        "stale_expired_rows": int(state.eq(QUALITY_STALE_EXPIRED).sum()),
        "missing_rows": int(state.eq(QUALITY_MISSING).sum()),
        "fresh_pct": float(fresh.mean()),
        "synthetic_pct": float(synthetic.mean()),
        "max_stale_bars": int(stale_bars.fillna(0).max()),
    }
