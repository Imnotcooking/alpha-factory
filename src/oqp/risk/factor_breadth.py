from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.data.instruments import InstrumentMaster


__all__ = [
    "RiskBreadthConfig",
    "compute_breadth_metrics",
    "compute_log_return_matrix",
    "compute_risk_factor_breadth",
    "compute_rolling_breadth",
    "extract_base_symbol",
    "infer_component_labels",
    "load_futures_daily_data",
    "map_chinese_futures_sector",
    "prepare_pca_matrix",
    "run_covariance_pca",
]


@dataclass(frozen=True)
class RiskBreadthConfig:
    variance_threshold: float = 0.95
    max_components: int = 20
    min_history_pct: float = 0.60
    min_observations: int = 252
    rolling_window: int = 504
    rolling_step: int = 21
    rolling_min_assets: int = 20


def extract_base_symbol(ticker: str) -> str:
    """Extract the futures symbol from Chinese index labels or contract codes."""
    if ticker is None or pd.isna(ticker):
        return ""

    text = str(ticker).strip()
    paren_match = re.search(r"\(([^()]+)\)", text)
    if paren_match:
        return paren_match.group(1).strip()

    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"【[^】]*】", "", text)
    text = re.sub(r"\d+", "", text)
    symbol_match = re.search(r"[A-Za-z]+", text)
    return symbol_match.group(0) if symbol_match else text


def map_chinese_futures_sector(ticker: str, sector_map: dict[str, str] | None = None) -> str:
    sector_map = sector_map or InstrumentMaster("FUTURES_CN").get_sector_map()
    symbol = extract_base_symbol(ticker)
    return sector_map.get(symbol, sector_map.get(symbol.lower(), sector_map.get(symbol.upper(), "Unknown")))


def load_futures_daily_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Risk breadth data source not found: {path}")

    df = pd.read_parquet(path)
    required = {"date", "ticker", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Risk breadth input is missing columns: {missing}")

    out = df[["date", "ticker", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "close"])
    out = out[out["close"] > 0]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_log_return_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "ticker", "close"}.issubset(df.columns):
        raise ValueError("Expected columns: date, ticker, close")

    work = df[["date", "ticker", "close"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["date", "ticker", "close"])
    work = work[work["close"] > 0]

    prices = (
        work.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
    )
    log_returns = np.log(prices).diff()
    return log_returns.replace([np.inf, -np.inf], np.nan)


def prepare_pca_matrix(
    returns: pd.DataFrame,
    config: RiskBreadthConfig,
) -> tuple[pd.DataFrame, list[str]]:
    if returns.empty:
        return returns.copy(), []

    min_obs = max(1, min(config.min_observations, len(returns)))
    min_obs = max(min_obs, int(np.ceil(len(returns) * config.min_history_pct)))
    valid_assets = returns.columns[returns.notna().sum() >= min_obs].tolist()
    selected = returns[valid_assets].copy()
    selected = selected.dropna(how="all")
    if selected.empty:
        return selected, []

    demeaned = selected.sub(selected.mean(axis=0), axis=1)
    return demeaned.fillna(0.0), valid_assets


def compute_breadth_metrics(
    eigenvalues: np.ndarray,
    *,
    variance_threshold: float = 0.95,
    naive_breadth: int | None = None,
) -> dict[str, float | int]:
    eig = np.asarray(eigenvalues, dtype=float)
    eig = np.clip(eig[np.isfinite(eig)], 0.0, None)
    total = eig.sum()
    if total <= 0 or eig.size == 0:
        return {
            "br_threshold": 0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "naive_breadth": int(naive_breadth or 0),
            "breadth_haircut": np.nan,
        }

    weights = eig / total
    cumulative = np.cumsum(weights)
    br_threshold = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    positive_weights = weights[weights > 0]
    entropy = -np.sum(positive_weights * np.log(positive_weights))
    effective_rank = float(np.exp(entropy))
    participation_ratio = float(1.0 / np.sum(weights**2)) if np.sum(weights**2) > 0 else 0.0
    naive = int(naive_breadth if naive_breadth is not None else eig.size)
    haircut = float(br_threshold / naive) if naive > 0 else np.nan
    return {
        "br_threshold": br_threshold,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "naive_breadth": naive,
        "breadth_haircut": haircut,
    }


def run_covariance_pca(
    returns: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    config: RiskBreadthConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RiskBreadthConfig()
    sector_map = sector_map or InstrumentMaster("FUTURES_CN").get_sector_map()
    x, assets = prepare_pca_matrix(returns, cfg)
    if x.empty or len(assets) < 2:
        raise ValueError("Not enough valid assets for covariance PCA.")

    covariance = np.cov(x.to_numpy(dtype=float), rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]

    total_var = eigenvalues.sum()
    explained = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(explained)
    components = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    spectrum = pd.DataFrame(
        {
            "component": components,
            "component_idx": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "explained_variance_ratio": explained,
            "cumulative_variance": cumulative,
        }
    )

    loadings = _build_asset_loadings(assets, eigenvectors, explained, sector_map, cfg.max_components)
    sector_signed, sector_abs = _build_sector_summaries(loadings)
    component_labels = infer_component_labels(sector_abs, loadings, cfg.max_components)
    metrics = compute_breadth_metrics(
        eigenvalues,
        variance_threshold=cfg.variance_threshold,
        naive_breadth=len(assets),
    )
    metrics.update(
        {
            "valid_assets": len(assets),
            "observations": int(len(x)),
            "date_min": x.index.min(),
            "date_max": x.index.max(),
            "variance_threshold": cfg.variance_threshold,
        }
    )

    return {
        "metrics": metrics,
        "spectrum": spectrum,
        "asset_loadings": loadings,
        "sector_signed": sector_signed,
        "sector_abs": sector_abs,
        "component_labels": component_labels,
        "returns": x,
        "skipped_assets": [col for col in returns.columns if col not in assets],
    }


def compute_rolling_breadth(
    returns: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    config: RiskBreadthConfig | None = None,
) -> tuple[pd.DataFrame, int]:
    cfg = config or RiskBreadthConfig()
    sector_map = sector_map or InstrumentMaster("FUTURES_CN").get_sector_map()
    dates = returns.dropna(how="all").index
    if len(dates) < cfg.rolling_window:
        return pd.DataFrame(), 1

    rows = []
    skipped = 0
    for end_pos in range(cfg.rolling_window, len(dates) + 1, cfg.rolling_step):
        window_dates = dates[end_pos - cfg.rolling_window : end_pos]
        window = returns.loc[window_dates]
        try:
            result = run_covariance_pca(window, sector_map=sector_map, config=cfg)
        except ValueError:
            skipped += 1
            continue
        metrics = result["metrics"]
        if metrics["valid_assets"] < cfg.rolling_min_assets:
            skipped += 1
            continue
        rows.append(
            {
                "date": window_dates[-1],
                "br95": metrics["br_threshold"],
                "effective_rank": metrics["effective_rank"],
                "participation_ratio": metrics["participation_ratio"],
                "naive_breadth": metrics["naive_breadth"],
                "breadth_haircut": metrics["breadth_haircut"],
                "valid_assets": metrics["valid_assets"],
            }
        )

    return pd.DataFrame(rows), skipped


def compute_risk_factor_breadth(
    source_path: str | Path,
    config: RiskBreadthConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RiskBreadthConfig()
    raw = load_futures_daily_data(source_path)
    sector_map = InstrumentMaster("FUTURES_CN").get_sector_map()
    returns = compute_log_return_matrix(raw)
    full = run_covariance_pca(returns, sector_map=sector_map, config=cfg)
    rolling, skipped = compute_rolling_breadth(returns, sector_map=sector_map, config=cfg)
    full["rolling_breadth"] = rolling
    full["rolling_skipped_windows"] = skipped
    full["source_rows"] = len(raw)
    full["source_assets"] = raw["ticker"].nunique()
    return full


def _build_asset_loadings(
    assets: list[str],
    eigenvectors: np.ndarray,
    explained: np.ndarray,
    sector_map: dict[str, str],
    max_components: int,
) -> pd.DataFrame:
    rows = []
    n_components = min(max_components, eigenvectors.shape[1])
    for component_idx in range(n_components):
        component = f"PC{component_idx + 1}"
        for asset_idx, ticker in enumerate(assets):
            symbol = extract_base_symbol(ticker)
            loading = float(eigenvectors[asset_idx, component_idx])
            rows.append(
                {
                    "ticker": ticker,
                    "base_symbol": symbol,
                    "sector": map_chinese_futures_sector(ticker, sector_map),
                    "component": component,
                    "component_idx": component_idx + 1,
                    "loading": loading,
                    "abs_loading": abs(loading),
                    "explained_variance_ratio": float(explained[component_idx]),
                }
            )
    return pd.DataFrame(rows)


def _build_sector_summaries(loadings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if loadings.empty:
        return pd.DataFrame(), pd.DataFrame()

    signed = (
        loadings.groupby(["component", "component_idx", "sector"], as_index=False)
        .agg(
            signed_mean_loading=("loading", "mean"),
            signed_sum_loading=("loading", "sum"),
            asset_count=("ticker", "nunique"),
        )
        .sort_values(["component_idx", "sector"])
    )

    absolute = (
        loadings.groupby(["component", "component_idx", "sector"], as_index=False)
        .agg(
            abs_loading_sum=("abs_loading", "sum"),
            abs_loading_mean=("abs_loading", "mean"),
            asset_count=("ticker", "nunique"),
        )
        .sort_values(["component_idx", "sector"])
    )
    total_abs = absolute.groupby("component")["abs_loading_sum"].transform("sum")
    absolute["abs_loading_share"] = np.where(total_abs > 0, absolute["abs_loading_sum"] / total_abs, 0.0)
    return signed, absolute


def infer_component_labels(
    sector_abs: pd.DataFrame,
    asset_loadings: pd.DataFrame,
    max_components: int = 20,
) -> pd.DataFrame:
    """Create human-readable PCA component labels from dominant sectors/assets."""
    if sector_abs.empty or asset_loadings.empty:
        return pd.DataFrame()

    industrial = {"化工", "黑色", "能源", "有色", "建材"}
    agriculture = {"油脂油料", "软商品", "生鲜"}
    financial = {"国债", "股指"}

    rows = []
    components = (
        sector_abs[["component", "component_idx"]]
        .drop_duplicates()
        .sort_values("component_idx")
        .head(max_components)
    )
    for _, component_row in components.iterrows():
        component = component_row["component"]
        component_idx = int(component_row["component_idx"])
        sector_slice = sector_abs[sector_abs["component"] == component].sort_values(
            "abs_loading_share", ascending=False
        )
        asset_slice = asset_loadings[asset_loadings["component"] == component].sort_values(
            "abs_loading", ascending=False
        )

        top_sectors = sector_slice.head(3)["sector"].tolist()
        top_assets = asset_slice.head(5)["base_symbol"].tolist()
        top_share = float(sector_slice.head(3)["abs_loading_share"].sum())
        top_set = set(top_sectors)

        if len(top_set & industrial) >= 2:
            label_en = "Cyclical industrial commodity beta"
            label_zh = "工业品周期共同因子"
            interpretation_en = (
                "Dominated by industrial commodity sectors. Treat it as the broad macro cycle "
                "linking energy, black metals, chemicals, and base metals."
            )
            interpretation_zh = "主要由工业品板块驱动，可理解为能源、黑色、化工、有色共同波动的宏观周期因子。"
        elif len(top_set & agriculture) >= 2:
            label_en = "Agricultural and soft-commodity beta"
            label_zh = "农产品与软商品共同因子"
            interpretation_en = "Dominated by oils, soft commodities, or fresh-product contracts."
            interpretation_zh = "主要由油脂油料、软商品或生鲜类合约驱动。"
        elif "国债" in top_set:
            label_en = "Rates duration factor"
            label_zh = "利率久期因子"
            interpretation_en = "Dominated by Chinese government bond futures and rate sensitivity."
            interpretation_zh = "主要由国债期货与利率久期变化驱动。"
        elif "股指" in top_set:
            label_en = "Equity index beta"
            label_zh = "股指风险因子"
            interpretation_en = "Dominated by equity-index futures and equity-market beta."
            interpretation_zh = "主要由股指期货与权益市场 beta 驱动。"
        elif "贵金属" in top_set:
            label_en = "Precious-metals factor"
            label_zh = "贵金属因子"
            interpretation_en = "Dominated by gold/silver-style safe-haven or real-rate sensitivity."
            interpretation_zh = "主要由黄金、白银等贵金属的避险或实际利率敏感性驱动。"
        else:
            label_en = "Mixed sector factor"
            label_zh = "混合板块因子"
            interpretation_en = "No single economic family dominates the component cleanly."
            interpretation_zh = "没有单一经济板块能清晰解释该主成分。"

        rows.append(
            {
                "component": component,
                "component_idx": component_idx,
                "label_en": label_en,
                "label_zh": label_zh,
                "top_sectors": ", ".join(top_sectors),
                "top_assets": ", ".join(top_assets),
                "top3_sector_share": top_share,
                "interpretation_en": interpretation_en,
                "interpretation_zh": interpretation_zh,
            }
        )

    return pd.DataFrame(rows)
