"""Leakage-aware transforms for comparable factor scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.research.factor_portfolios.contracts import VALID_NORMALIZATIONS


def normalize_factor_signal(
    df: pd.DataFrame,
    signal_col: str,
    *,
    method: str,
    date_col: str = "date",
    winsor_limit: float | None = 3.0,
) -> pd.Series:
    """Normalize a factor without using observations from future timestamps."""

    if method not in VALID_NORMALIZATIONS:
        raise ValueError(f"unknown factor normalization: {method}")
    if signal_col not in df.columns:
        raise ValueError(f"factor frame is missing signal column {signal_col!r}")

    values = pd.to_numeric(df[signal_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if method == "raw":
        normalized = values.astype(float)
    elif method == "cross_sectional_rank":
        ranks = values.groupby(df[date_col], sort=False).rank(
            method="average",
            pct=True,
        )
        normalized = (ranks - 0.5) * 2.0
    else:
        grouped = values.groupby(df[date_col], sort=False)
        means = grouped.transform("mean")
        stds = grouped.transform(lambda series: series.std(ddof=0))
        normalized = (values - means) / stds.replace(0.0, np.nan)

    if winsor_limit is not None:
        limit = abs(float(winsor_limit))
        normalized = normalized.clip(-limit, limit)
    return normalized.astype(float)


__all__ = ["normalize_factor_signal"]
