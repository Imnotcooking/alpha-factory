"""Dashboard-specific file discovery and loading for relationship research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oqp.data.futures_cn import normalize_futures_cn_daily_frame
from oqp.data.runtime_paths import (
    default_futures_cn_index_daily_file,
    discover_futures_cn_daily_files,
)

__all__ = ["available_tickers", "list_daily_price_files", "load_daily_price_frame"]


def list_daily_price_files(base_dir: str | Path) -> list[Path]:
    repo_root = Path(base_dir)
    patterns = ("*1d*index*.parquet", "*1d*main*.parquet", "*daily*.parquet", "*day*.parquet")
    seen: dict[Path, None] = {}
    for path in discover_futures_cn_daily_files(patterns=patterns):
        seen[path.resolve()] = None
    root_matrix = repo_root / "runtime" / "data" / "feature_store" / "ML_Feature_Matrix.parquet"
    if root_matrix.exists():
        seen[root_matrix.resolve()] = None
    files = list(seen.keys())
    default_source = default_futures_cn_index_daily_file()
    if default_source.exists():
        default_resolved = default_source.resolve()
        files = [default_resolved, *[path for path in files if path != default_resolved]]
    return files


def load_daily_price_frame(path: str | Path) -> pd.DataFrame:
    return normalize_futures_cn_daily_frame(pd.read_parquet(path))


def available_tickers(df: pd.DataFrame, min_observations: int = 252) -> pd.DataFrame:
    counts = (
        df.dropna(subset=["close"])
        .groupby("ticker", as_index=False)
        .agg(
            observations=("close", "size"),
            start=("date", "min"),
            end=("date", "max"),
        )
        .sort_values(["observations", "ticker"], ascending=[False, True])
    )
    counts["eligible"] = counts["observations"] >= int(min_observations)
    return counts
