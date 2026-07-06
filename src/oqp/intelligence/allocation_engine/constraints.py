"""Portfolio allocation constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_weight_constraints(
    weights: pd.Series,
    *,
    max_abs_weight: float | None = 0.25,
    max_gross: float | None = 1.0,
    long_only: bool = False,
) -> pd.Series:
    """Clip and gross-scale a weight vector."""

    out = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    if long_only:
        out = out.clip(lower=0.0)
    if max_abs_weight is not None and max_abs_weight > 0:
        out = out.clip(lower=-float(max_abs_weight), upper=float(max_abs_weight))
    if max_gross is not None and max_gross > 0:
        gross = float(out.abs().sum())
        if gross > float(max_gross):
            out = out * (float(max_gross) / gross)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def normalize_weights(weights: pd.Series, *, long_only: bool = False) -> pd.Series:
    out = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    if long_only:
        out = out.clip(lower=0.0)
        total = float(out.sum())
    else:
        total = float(out.abs().sum())
    if total <= 0:
        return pd.Series(0.0, index=out.index)
    return out / total
