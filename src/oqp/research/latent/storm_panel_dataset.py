from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from oqp.research.ml.feature_governance import detect_feature_columns


@dataclass(frozen=True)
class TemporalPanelConfig:
    window_size: int = 20
    stride: int = 1
    max_samples: int | None = 50_000
    random_state: int = 42
    include_prob_features: bool = True
    feature_prefixes: tuple[str, ...] = ("f_", "prob_")


def detect_storm_feature_columns(
    df: pd.DataFrame,
    include_prob_features: bool = True,
) -> list[str]:
    return detect_feature_columns(df, include_prob_features=include_prob_features)


def build_temporal_windows(
    matrix: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    config: TemporalPanelConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Build one rolling time-series sample per (ticker, date).

    Output X has shape [samples, window_size, features]. Metadata is aligned to
    X and identifies the ticker/date represented by the final row of each
    window. This is the temporal half of the STORM-style research path.
    """
    cfg = config or TemporalPanelConfig()
    required = {"date", "ticker"}
    missing = required.difference(matrix.columns)
    if missing:
        raise ValueError(f"Temporal panel missing columns: {sorted(missing)}")

    features = list(feature_cols) if feature_cols is not None else detect_storm_feature_columns(
        matrix,
        include_prob_features=cfg.include_prob_features,
    )
    if not features:
        raise ValueError("No numeric feature columns available for temporal windows.")

    work = matrix.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["ticker"] = work["ticker"].astype(str)
    work = work.sort_values(["ticker", "date"]).reset_index(drop=True)

    samples: list[np.ndarray] = []
    meta_rows: list[dict] = []
    for ticker, group in work.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        values = (
            group[features]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        if len(values) < cfg.window_size:
            continue

        for end_idx in range(cfg.window_size - 1, len(values), max(cfg.stride, 1)):
            start_idx = end_idx - cfg.window_size + 1
            samples.append(values[start_idx : end_idx + 1])
            row = group.iloc[end_idx]
            meta = {
                "date": row["date"],
                "ticker": ticker,
                "_source_index": int(group.index[end_idx]),
            }
            for col in ("target_1d_rank", "target_4d_rank", "sector"):
                if col in group.columns:
                    meta[col] = row[col]
            meta_rows.append(meta)

    if not samples:
        raise ValueError("No temporal windows were created. Try a smaller window size.")

    x = np.stack(samples, axis=0)
    meta_df = pd.DataFrame(meta_rows)

    if cfg.max_samples is not None and len(meta_df) > cfg.max_samples:
        rng = np.random.default_rng(cfg.random_state)
        keep = np.sort(rng.choice(len(meta_df), size=cfg.max_samples, replace=False))
        x = x[keep]
        meta_df = meta_df.iloc[keep].reset_index(drop=True)
    else:
        meta_df = meta_df.reset_index(drop=True)

    return x, meta_df, features


def flatten_temporal_windows(x: np.ndarray, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Flatten [samples, window, features] into a tabular matrix for simple encoders."""
    if x.ndim != 3:
        raise ValueError("Expected X with shape [samples, window, features].")
    features = list(feature_cols)
    if x.shape[2] != len(features):
        raise ValueError("Feature column count does not match the X tensor.")

    columns = []
    for lag in range(x.shape[1] - 1, -1, -1):
        for feature in features:
            columns.append(f"{feature}_lag_{lag:02d}")
    flat = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    return pd.DataFrame(flat, columns=columns)


def build_cross_sectional_snapshot(
    matrix: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    include_prob_features: bool = True,
) -> pd.DataFrame:
    """
    Produce a light spatial summary per date.

    This is not the full STORM cross-sectional transformer. It gives the UI a
    transparent first spatial check: market-wide mean/std/rank dispersion of
    each feature on each date.
    """
    required = {"date", "ticker"}
    missing = required.difference(matrix.columns)
    if missing:
        raise ValueError(f"Cross-sectional snapshot missing columns: {sorted(missing)}")

    features = list(feature_cols) if feature_cols is not None else detect_storm_feature_columns(
        matrix,
        include_prob_features=include_prob_features,
    )
    if not features:
        return pd.DataFrame()

    work = matrix.copy()
    work["date"] = pd.to_datetime(work["date"])
    numeric = work[["date", *features]].replace([np.inf, -np.inf], np.nan)
    rows = []
    for date, day in numeric.groupby("date", sort=False):
        row = {"date": date}
        for feature in features:
            values = pd.to_numeric(day[feature], errors="coerce").dropna()
            row[f"cs_{feature}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"cs_{feature}_std"] = float(values.std()) if len(values) > 1 else 0.0
            row[f"cs_{feature}_iqr"] = (
                float(values.quantile(0.75) - values.quantile(0.25))
                if not values.empty
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
