from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = ["coefficient_columns", "summarize_dual_kalman_output"]


def summarize_dual_kalman_output(
    features: pd.DataFrame,
    *,
    prefix: str = "dkf",
    group_col: str | None = "ticker",
) -> pd.DataFrame:
    """Return compact quality diagnostics for DKF feature output."""

    if features.empty:
        return pd.DataFrame()

    group_keys = [group_col] if group_col and group_col in features.columns else []
    grouped = features.groupby(group_keys, dropna=False) if group_keys else [(None, features)]
    rows = []
    for key, group in grouped:
        residual = pd.to_numeric(group.get(f"{prefix}_residual"), errors="coerce")
        innovation_z = pd.to_numeric(group.get(f"{prefix}_innovation_z"), errors="coerce")
        uncertainty = pd.to_numeric(group.get(f"{prefix}_state_uncertainty"), errors="coerce")
        drift = pd.to_numeric(group.get(f"{prefix}_beta_l1_change"), errors="coerce")
        row = {
            "rows": int(len(group)),
            "residual_mean": float(residual.mean()) if residual.notna().any() else np.nan,
            "residual_std": float(residual.std()) if residual.notna().sum() > 1 else np.nan,
            "mean_abs_innovation_z": float(innovation_z.abs().mean()) if innovation_z.notna().any() else np.nan,
            "median_state_uncertainty": float(uncertainty.median()) if uncertainty.notna().any() else np.nan,
            "mean_beta_l1_change": float(drift.mean()) if drift.notna().any() else np.nan,
            "valid_residual_rate": float(residual.notna().mean()),
        }
        if group_keys:
            row[group_col] = key[0] if isinstance(key, tuple) else key
        rows.append(row)

    return pd.DataFrame(rows)


def coefficient_columns(features: pd.DataFrame, prefix: str = "dkf") -> list[str]:
    """List posterior beta columns while excluding variance diagnostics."""

    return [
        col
        for col in features.columns
        if col.startswith(f"{prefix}_beta_")
        and not col.endswith("_var")
        and col != f"{prefix}_beta_l1_change"
    ]
