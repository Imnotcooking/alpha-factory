"""Kelly sizing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.intelligence.allocation_engine.constraints import apply_weight_constraints


def kelly_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    kelly_fraction: float = 0.5,
    max_abs_weight: float | None = 0.25,
    max_gross: float | None = 1.0,
    ridge: float = 1e-6,
) -> pd.Series:
    """Estimate fractional Kelly weights from expected returns and covariance."""

    assets = list(expected_returns.index)
    if not assets:
        return pd.Series(dtype=float)

    cov = covariance.reindex(index=assets, columns=assets).fillna(0.0).astype(float)
    mu = expected_returns.reindex(assets).fillna(0.0).astype(float)
    matrix = cov.values + np.eye(len(assets)) * ridge
    try:
        raw = np.linalg.solve(matrix, mu.values)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(matrix) @ mu.values
    weights = pd.Series(raw * float(kelly_fraction), index=assets)
    return apply_weight_constraints(
        weights,
        max_abs_weight=max_abs_weight,
        max_gross=max_gross,
    )
