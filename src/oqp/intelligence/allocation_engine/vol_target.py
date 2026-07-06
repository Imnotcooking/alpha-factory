"""Volatility target scaling helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def portfolio_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> float:
    assets = list(weights.index)
    if not assets:
        return 0.0
    cov = covariance.reindex(index=assets, columns=assets).fillna(0.0).astype(float)
    value = float(weights.values.T @ cov.values @ weights.values)
    return math.sqrt(max(value, 0.0) * periods_per_year)


def scale_to_vol_target(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    target_volatility: float = 0.10,
    periods_per_year: int = 252,
    max_leverage: float = 1.0,
) -> pd.Series:
    current_vol = portfolio_volatility(
        weights,
        covariance,
        periods_per_year=periods_per_year,
    )
    if current_vol <= 0 or target_volatility <= 0:
        return weights * 0.0
    scale = min(float(target_volatility) / current_vol, float(max_leverage))
    return (weights * scale).replace([np.inf, -np.inf], 0.0).fillna(0.0)
