"""Runtime data path contracts for local/private market files."""

from __future__ import annotations

import os
from pathlib import Path

from oqp.config import REPO_ROOT
from oqp.contracts.market_vertical import normalize_market_vertical


DEFAULT_FUTURES_CN_DAILY_FILENAME = "全市场_1d_index_20180101_20260602.parquet"


def _path_from_env(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


RUNTIME_DATA_ROOT = _path_from_env(
    os.environ.get("ALPHA_RUNTIME_DATA_ROOT", "runtime/data")
)


FUTURES_CN_DAILY_DATA_ROOT = _path_from_env(
    os.environ.get(
        "FUTURES_CN_DAILY_DATA_ROOT",
        RUNTIME_DATA_ROOT / "futures_cn" / "daily",
    )
)
FUTURES_CN_INTRADAY_DATA_ROOT = _path_from_env(
    os.environ.get(
        "FUTURES_CN_INTRADAY_DATA_ROOT",
        RUNTIME_DATA_ROOT / "futures_cn" / "intraday",
    )
)
FUTURES_CN_TICK_DATA_ROOT = _path_from_env(
    os.environ.get(
        "FUTURES_CN_TICK_DATA_ROOT",
        RUNTIME_DATA_ROOT / "futures_cn" / "tick",
    )
)


def asset_class_runtime_data_root(asset_class: str, timeframe: str = "daily") -> Path:
    """Return the canonical runtime data folder for an asset class/timeframe."""

    normalized = normalize_market_vertical(asset_class).lower()
    timeframe = str(timeframe).strip().lower()
    if not timeframe:
        timeframe = "daily"
    return RUNTIME_DATA_ROOT / normalized / timeframe


def futures_cn_daily_roots() -> tuple[Path, ...]:
    """Return canonical daily CN futures data roots."""

    return (FUTURES_CN_DAILY_DATA_ROOT,)


def futures_cn_intraday_roots() -> tuple[Path, ...]:
    """Return canonical intraday CN futures data roots."""

    return (FUTURES_CN_INTRADAY_DATA_ROOT,)


def futures_cn_tick_roots() -> tuple[Path, ...]:
    """Return canonical tick CN futures data roots."""

    return (FUTURES_CN_TICK_DATA_ROOT,)


def discover_futures_cn_daily_files(
    *,
    patterns: tuple[str, ...] | None = None,
    include_csv: bool = False,
) -> list[Path]:
    """Find local CN futures daily market files."""

    search_patterns = patterns or ("*.parquet",)
    if include_csv:
        search_patterns = (*search_patterns, "*.csv")

    seen: dict[Path, tuple[int, float]] = {}
    for root_rank, root in enumerate(futures_cn_daily_roots()):
        for pattern in search_patterns:
            for path in root.glob(pattern) if root.exists() else []:
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen[resolved] = (root_rank, path.stat().st_mtime)

    files = list(seen)
    files.sort(
        key=lambda path: (
            seen[path][0],
            -seen[path][1],
            path.name.lower(),
        )
    )
    return files


def discover_asset_class_files(
    asset_class: str,
    *,
    timeframe: str = "daily",
    patterns: tuple[str, ...] | None = None,
    include_csv: bool = False,
) -> list[Path]:
    """Find local market data files for a taxonomy asset class."""

    normalized = normalize_market_vertical(asset_class)
    if normalized == "FUTURES_CN" and timeframe == "daily":
        return discover_futures_cn_daily_files(patterns=patterns, include_csv=include_csv)

    search_patterns = patterns or ("*.parquet",)
    if include_csv:
        search_patterns = (*search_patterns, "*.csv")
    return _discover_files_from_roots(
        (asset_class_runtime_data_root(normalized, timeframe),),
        search_patterns,
    )


def _discover_files_from_roots(
    roots: tuple[Path, ...],
    patterns: tuple[str, ...],
) -> list[Path]:
    seen: dict[Path, tuple[int, float]] = {}
    for root_rank, root in enumerate(roots):
        for pattern in patterns:
            for path in root.glob(pattern) if root.exists() else []:
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen[resolved] = (root_rank, path.stat().st_mtime)
    files = list(seen)
    files.sort(key=lambda path: (seen[path][0], -seen[path][1], path.name.lower()))
    return files


def discover_futures_cn_intraday_files(
    *,
    patterns: tuple[str, ...] | None = None,
) -> list[Path]:
    """Find local CN futures intraday market files."""

    return _discover_files_from_roots(
        futures_cn_intraday_roots(),
        patterns or ("*.parquet",),
    )


def discover_futures_cn_tick_files(
    *,
    patterns: tuple[str, ...] | None = None,
) -> list[Path]:
    """Find local CN futures tick market files."""

    return _discover_files_from_roots(
        futures_cn_tick_roots(),
        patterns or ("*tick*.parquet", "*.parquet"),
    )


def default_futures_cn_daily_file() -> Path:
    """Return the newest preferred local CN futures daily file."""

    files = discover_futures_cn_daily_files()
    if files:
        return files[0]
    return FUTURES_CN_DAILY_DATA_ROOT / DEFAULT_FUTURES_CN_DAILY_FILENAME


def default_futures_cn_index_daily_file() -> Path:
    """Return the preferred continuous/index-style CN futures daily file."""

    files = discover_futures_cn_daily_files(
        patterns=(
            "*1d*index*.parquet",
            "*index*.parquet",
            "*指数*.parquet",
            "*全市场*.parquet",
        ),
    )
    if files:
        return files[0]
    return FUTURES_CN_DAILY_DATA_ROOT / DEFAULT_FUTURES_CN_DAILY_FILENAME
