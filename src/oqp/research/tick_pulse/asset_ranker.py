from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from oqp.data.futures_cn import normalize_futures_cn_daily_frame
from oqp.data.runtime_paths import futures_cn_daily_roots, futures_cn_intraday_roots
from oqp.risk.factor_breadth import extract_base_symbol, map_chinese_futures_sector


REQUIRED_DAILY_COLUMNS = {"date", "ticker", "close"}

__all__ = [
    "REQUIRED_DAILY_COLUMNS",
    "discover_daily_universe_files",
    "discover_intraday_universe_files",
    "filter_ranked_assets",
    "load_daily_universe",
    "rank_daily_asset_volatility",
]


def load_daily_universe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Daily universe file not found: {path}")

    return normalize_futures_cn_daily_frame(pd.read_parquet(path))


def discover_daily_universe_files(project_root: str | Path) -> list[dict]:
    return _discover_bar_universe_files(
        project_root,
        futures_cn_daily_roots(),
        ("*1d*.parquet", "*daily*.parquet", "*universe*.parquet"),
        include_universe_root=True,
    )


def discover_intraday_universe_files(project_root: str | Path) -> list[dict]:
    return _discover_bar_universe_files(
        project_root,
        futures_cn_intraday_roots(),
        ("*1m*.parquet", "*intraday*.parquet", "*.parquet"),
        include_universe_root=False,
    )


def _discover_bar_universe_files(
    project_root: str | Path,
    search_roots: tuple[Path, ...],
    patterns: tuple[str, ...],
    include_universe_root: bool,
) -> list[dict]:
    repo_root = Path(project_root)

    candidates: dict[Path, None] = {}
    roots = search_roots
    if include_universe_root:
        roots = (*roots, repo_root / "runtime" / "data" / "universes")
    for data_dir in roots:
        for pattern in patterns:
            for path in data_dir.glob(pattern) if data_dir.exists() else []:
                if path.is_file() and "tick" not in path.name.lower():
                    candidates[path.resolve()] = None

    files = []
    for path in candidates:
        stat = path.stat()
        try:
            display_path = path.relative_to(repo_root)
        except ValueError:
            display_path = path
        files.append(
            {
                "path": str(display_path),
                "label": f"{path.name} ({stat.st_size / 1024 / 1024:.1f} MB)",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

    files.sort(key=lambda item: ("全市场" not in item["path"], -item["mtime"], item["path"]))
    return files


def rank_daily_asset_volatility(
    daily_df: pd.DataFrame,
    *,
    lookback_days: int = 252,
    min_observations: int = 120,
    annualization: int = 252,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    missing = sorted(REQUIRED_DAILY_COLUMNS - set(daily_df.columns))
    if missing:
        raise ValueError(f"Daily universe frame missing required columns: {missing}")

    work = daily_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ticker"] = work["ticker"].astype(str)
    for column in ["open", "high", "low", "close", "volume", "oi"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.dropna(subset=["date", "ticker", "close"])
    work = work[work["close"] > 0].sort_values(["ticker", "date"])
    if work.empty:
        return pd.DataFrame()

    work["log_return"] = work.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s).diff()
    )
    if {"high", "low", "close"}.issubset(work.columns):
        valid_range = (work["high"] > 0) & (work["low"] > 0) & (work["high"] >= work["low"])
        work["intraday_range_pct"] = np.where(
            valid_range,
            np.log(work["high"] / work["low"]).replace([np.inf, -np.inf], np.nan),
            np.nan,
        )
    else:
        work["intraday_range_pct"] = np.nan

    lookback = work.groupby("ticker", sort=False, group_keys=False).tail(lookback_days)

    rows = []
    for ticker, group in lookback.groupby("ticker", sort=True):
        group = group.sort_values("date")
        returns = group["log_return"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < min_observations:
            continue

        full_returns = work.loc[work["ticker"] == ticker, "log_return"].replace([np.inf, -np.inf], np.nan).dropna()
        volume = group["volume"] if "volume" in group.columns else pd.Series(dtype=float)
        oi = group["oi"] if "oi" in group.columns else pd.Series(dtype=float)
        range_pct = group["intraday_range_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        last_row = group.iloc[-1]

        recent_ann_vol = float(returns.std(ddof=1) * np.sqrt(annualization))
        full_ann_vol = float(full_returns.std(ddof=1) * np.sqrt(annualization)) if len(full_returns) > 1 else np.nan
        rows.append(
            {
                "ticker": ticker,
                "base_symbol": extract_base_symbol(ticker),
                "sector": map_chinese_futures_sector(ticker),
                "last_date": last_row["date"],
                "last_close": float(last_row["close"]),
                "recent_ann_vol": recent_ann_vol,
                "full_ann_vol": full_ann_vol,
                "avg_abs_return_bps": float(returns.abs().mean() * 10000.0),
                "avg_intraday_range_pct": float(range_pct.mean()) if len(range_pct) else np.nan,
                "avg_daily_volume": float(volume.dropna().mean()) if len(volume.dropna()) else np.nan,
                "avg_oi": float(oi.dropna().mean()) if len(oi.dropna()) else np.nan,
                "valid_return_days": int(len(returns)),
                "coverage": float(min(1.0, len(returns) / max(1, lookback_days))),
            }
        )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked

    ranked["vol_rank"] = ranked["recent_ann_vol"].rank(method="min", ascending=False).astype(int)
    ranked["vol_percentile"] = ranked["recent_ann_vol"].rank(pct=True)
    ranked["liquidity_percentile"] = np.log1p(ranked["avg_daily_volume"].fillna(0)).rank(pct=True)
    ranked["oi_percentile"] = np.log1p(ranked["avg_oi"].fillna(0)).rank(pct=True)
    ranked["download_priority_score"] = (
        0.65 * ranked["vol_percentile"]
        + 0.20 * ranked["liquidity_percentile"]
        + 0.10 * ranked["oi_percentile"]
        + 0.05 * ranked["coverage"].clip(upper=1.0)
    )
    ranked["download_priority_rank"] = ranked["download_priority_score"].rank(
        method="min",
        ascending=False,
    ).astype(int)
    ranked["download_symbol_hint"] = ranked["base_symbol"]
    return ranked.sort_values(["download_priority_rank", "vol_rank", "ticker"]).reset_index(drop=True)


def filter_ranked_assets(
    ranked: pd.DataFrame,
    *,
    sectors: Iterable[str] | None = None,
    min_volume: float = 0.0,
    top_n: int = 30,
    sort_by: str = "download_priority_score",
) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    out = ranked.copy()
    if sectors:
        sector_set = set(sectors)
        out = out[out["sector"].isin(sector_set)]
    if min_volume > 0 and "avg_daily_volume" in out.columns:
        out = out[out["avg_daily_volume"].fillna(0) >= min_volume]

    if sort_by not in out.columns:
        sort_by = "download_priority_score"
    out = out.sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)
    return out
