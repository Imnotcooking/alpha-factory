"""Information-coefficient decay diagnostics for engineered features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.stats as stats


DEFAULT_MATRIX_PATH = "runtime/data/feature_store/ML_Feature_Matrix.parquet"
DEFAULT_FEATURE = "f_oi_growth_10d"
DEFAULT_HORIZONS = tuple(range(1, 11))


def list_feature_columns(matrix_path: str = DEFAULT_MATRIX_PATH) -> list[str]:
    """Return available engineered feature columns from a feature matrix."""

    df = pd.read_parquet(matrix_path)
    return sorted(col for col in df.columns if col.startswith("f_"))


def compute_ic_decay(
    feature: str = DEFAULT_FEATURE,
    matrix_path: str = DEFAULT_MATRIX_PATH,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    tickers: Iterable[str] | None = None,
    min_assets_per_day: int = 5,
) -> pd.DataFrame:
    """
    Compute a daily cross-sectional Spearman IC decay curve.

    Forward return is measured from next open to future close when an ``open``
    column exists, matching next-open execution assumptions and avoiding
    same-close leakage.
    """

    df = pd.read_parquet(matrix_path)
    required = {"date", "ticker", "close", feature}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"IC decay missing columns: {sorted(missing)}")

    has_open = "open" in df.columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    df = df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"]).copy()

    if tickers:
        ticker_set = {str(ticker) for ticker in tickers}
        df = df[df["ticker"].isin(ticker_set)].copy()
    if df.empty:
        return pd.DataFrame()

    feature_values = pd.to_numeric(df[feature], errors="coerce")
    grouped = df.groupby("ticker", sort=False)
    entry_price = grouped["open"].shift(-1) if has_open else grouped["close"].shift(-1)

    rows = []
    for horizon in horizons:
        horizon = int(horizon)
        if horizon < 1:
            continue

        exit_price = grouped["close"].shift(-horizon)
        work = pd.DataFrame(
            {
                "date": df["date"],
                "feature": feature_values,
                "forward_return": exit_price / entry_price - 1.0,
            }
        ).replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=["feature", "forward_return"])

        daily_ics = []
        for _, day_slice in work.groupby("date", sort=False):
            if len(day_slice) < min_assets_per_day:
                continue
            if (
                day_slice["feature"].nunique() < 2
                or day_slice["forward_return"].nunique() < 2
            ):
                continue
            ic, _ = stats.spearmanr(day_slice["feature"], day_slice["forward_return"])
            if np.isfinite(ic):
                daily_ics.append(float(ic))

        if daily_ics:
            ic_array = np.asarray(daily_ics, dtype=float)
            ic_mean = float(ic_array.mean())
            ic_std = float(ic_array.std(ddof=1)) if len(ic_array) > 1 else 0.0
            ic_ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
            positive_rate = float((ic_array > 0).mean())
        else:
            ic_mean = np.nan
            ic_ir = np.nan
            positive_rate = np.nan

        rows.append(
            {
                "horizon": horizon,
                "ic": ic_mean,
                "ic_ir": ic_ir,
                "positive_day_rate": positive_rate,
                "valid_days": len(daily_ics),
                "sample_count": len(work),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_FEATURE",
    "DEFAULT_HORIZONS",
    "DEFAULT_MATRIX_PATH",
    "compute_ic_decay",
    "list_feature_columns",
]
