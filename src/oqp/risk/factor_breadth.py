from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.contracts.market_vertical import normalize_market_vertical
from oqp.data.brownian_bridge import BrownianBridgeConfig
from oqp.data.instruments import InstrumentMaster
from oqp.data.views import build_market_data_views


__all__ = [
    "RiskBreadthConfig",
    "compute_breadth_metrics",
    "compute_component_stability",
    "compute_log_return_matrix",
    "compute_risk_factor_breadth",
    "compute_rolling_breadth",
    "classify_breadth_regimes",
    "extract_base_symbol",
    "infer_component_labels",
    "load_daily_market_data",
    "load_futures_daily_data",
    "map_chinese_futures_sector",
    "prepare_pca_matrix",
    "run_covariance_pca",
    "summarize_breadth_regime_periods",
    "translate_sector_label",
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
    risk_imputation: str = "ffill"
    bridge_max_gap_bars: int = 20
    bridge_seed: int = 42
    asset_class: str = "FUTURES_CN"
    max_assets: int | None = None
    stability_components: int = 3
    stability_max_windows: int = 36


def extract_base_symbol(ticker: str) -> str:
    """Extract the futures symbol from Chinese index labels or contract codes."""
    if ticker is None or pd.isna(ticker):
        return ""

    text = str(ticker).strip()
    if re.match(r"^[A-Z]{2,6}\.\d{4,8}$", text):
        return text

    kq_match = re.search(r"@[A-Za-z]+\.(?P<symbol>[A-Za-z]+)", text)
    if kq_match:
        return kq_match.group("symbol").strip()

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
    return load_daily_market_data(path, asset_class="FUTURES_CN")[["date", "ticker", "close"]].copy()


def load_daily_market_data(path: str | Path, *, asset_class: str = "FUTURES_CN") -> pd.DataFrame:
    """Load a taxonomy daily panel into date/ticker/close form."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Risk breadth data source not found: {path}")

    raw = _read_daily_market_columns(path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume", "sector", "name", "asset_class"])

    normalized = _normalize_daily_market_columns(raw)
    normalized["asset_class"] = normalize_market_vertical(asset_class)
    normalized = normalized.dropna(subset=["date", "ticker", "close"])
    normalized = normalized[normalized["ticker"].astype(str).str.strip().ne("")]
    normalized = normalized[normalized["close"] > 0]
    return normalized.sort_values(["ticker", "date"]).reset_index(drop=True)


def _read_daily_market_columns(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        available = _parquet_columns(path)
        selected = _selected_market_columns(available)
        if len(selected) < 3:
            raise ValueError(f"Daily market file does not expose date/ticker/close columns: {path}")
        return pd.read_parquet(path, columns=selected)
    if path.suffix.lower() == ".csv":
        preview = pd.read_csv(path, nrows=1)
        selected = _selected_market_columns(list(preview.columns))
        if len(selected) < 3:
            raise ValueError(f"Daily market CSV does not expose date/ticker/close columns: {path}")
        return pd.read_csv(path, usecols=selected)
    raise ValueError(f"Unsupported market breadth data file: {path}")


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def _selected_market_columns(columns: list[str]) -> list[str]:
    required = [
        _first_present(columns, ["date", "datetime", "timestamp", "trading_day", "trade_date"]),
        _first_present(columns, ["ticker", "symbol", "wind_code", "instrument_id", "instrument", "contract"]),
        _first_present(columns, ["close", "adjusted_close", "adj_close", "settle", "settlement", "last_price", "price"]),
    ]
    optional = [
        _first_present(columns, ["volume", "vol", "turnover_volume"]),
        _first_present(columns, ["sector", "industry", "sw_l1_name"]),
        _first_present(columns, ["name", "security_name", "instrument_name"]),
    ]
    return list(dict.fromkeys([col for col in [*required, *optional] if col]))


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    by_lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found:
            return found
    return None


def _normalize_daily_market_columns(raw: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw.columns)
    date_col = _first_present(columns, ["date", "datetime", "timestamp", "trading_day", "trade_date"])
    ticker_col = _first_present(columns, ["ticker", "symbol", "wind_code", "instrument_id", "instrument", "contract"])
    close_col = _first_present(columns, ["close", "adjusted_close", "adj_close", "settle", "settlement", "last_price", "price"])
    if not date_col or not ticker_col or not close_col:
        raise ValueError("Daily market frame requires date, ticker, and close-like columns.")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "ticker": raw[ticker_col].astype(str).str.strip(),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
        }
    )
    volume_col = _first_present(columns, ["volume", "vol", "turnover_volume"])
    if volume_col:
        out["volume"] = pd.to_numeric(raw[volume_col], errors="coerce")
    sector_col = _first_present(columns, ["sector", "industry", "sw_l1_name"])
    if sector_col:
        out["sector"] = raw[sector_col].astype(str).str.strip()
    name_col = _first_present(columns, ["name", "security_name", "instrument_name"])
    if name_col:
        out["name"] = raw[name_col].astype(str).str.strip()
    return out


def compute_log_return_matrix(
    df: pd.DataFrame,
    *,
    max_stale_bars: int = 3,
    risk_imputation: str = "ffill",
    bridge_max_gap_bars: int | None = None,
    bridge_seed: int = 42,
) -> pd.DataFrame:
    if not {"date", "ticker", "close"}.issubset(df.columns):
        raise ValueError("Expected columns: date, ticker, close")

    work = df[["date", "ticker", "close"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["date", "ticker"])
    work = work[work["close"] > 0]
    risk_mode = str(risk_imputation).strip().lower()

    views = build_market_data_views(
        work,
        timestamp_col="date",
        asset_col="ticker",
        price_cols=("close",),
        max_stale_bars=max_stale_bars,
        risk_imputation=risk_mode,
        bridge_config=(
            BrownianBridgeConfig(
                timestamp_col="date",
                asset_col="ticker",
                value_cols=("close",),
                max_gap_bars=int(bridge_max_gap_bars if bridge_max_gap_bars is not None else max_stale_bars),
                seed=bridge_seed,
            )
            if risk_mode == "brownian_bridge"
            else None
        ),
    )
    prices = views.risk.pivot(index="date", columns="ticker", values="close").sort_index()
    log_returns = np.log(prices).diff()
    return log_returns.replace([np.inf, -np.inf], np.nan)


def select_assets_for_breadth(df: pd.DataFrame, config: RiskBreadthConfig) -> pd.DataFrame:
    """Optionally cap a large universe before covariance PCA."""

    if df.empty or config.max_assets is None or int(config.max_assets) <= 0:
        return df.copy()

    max_assets = int(config.max_assets)
    if df["ticker"].nunique(dropna=True) <= max_assets:
        return df.copy()

    work = df.copy()
    if "volume" not in work.columns:
        work["volume"] = np.nan
    work["dollar_volume"] = pd.to_numeric(work.get("volume"), errors="coerce") * pd.to_numeric(
        work.get("close"), errors="coerce"
    )
    rank = (
        work.groupby("ticker", as_index=False)
        .agg(
            observations=("close", "size"),
            avg_dollar_volume=("dollar_volume", "mean"),
            last_date=("date", "max"),
        )
        .sort_values(["observations", "avg_dollar_volume", "last_date", "ticker"], ascending=[False, False, False, True])
        .head(max_assets)
    )
    selected = set(rank["ticker"].astype(str))
    return work[work["ticker"].astype(str).isin(selected)].drop(columns=["dollar_volume"], errors="ignore")


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


def classify_breadth_regimes(
    rolling: pd.DataFrame,
    *,
    metric: str = "breadth_haircut",
    low_quantile: float = 0.20,
    high_quantile: float = 0.80,
) -> pd.DataFrame:
    """Classify rolling breadth into Low/Normal/High regimes by adaptive quantiles."""

    if rolling.empty or metric not in rolling.columns:
        return pd.DataFrame()

    out = rolling.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=["date", metric]).sort_values("date").reset_index(drop=True)
    if out.empty:
        return out

    low_threshold = float(out[metric].quantile(low_quantile))
    high_threshold = float(out[metric].quantile(high_quantile))
    if not np.isfinite(low_threshold) or not np.isfinite(high_threshold) or low_threshold >= high_threshold:
        median = float(out[metric].median())
        out["breadth_regime"] = np.where(out[metric] < median, "Low", np.where(out[metric] > median, "High", "Normal"))
    else:
        out["breadth_regime"] = np.select(
            [out[metric] <= low_threshold, out[metric] >= high_threshold],
            ["Low", "High"],
            default="Normal",
        )
    out["breadth_regime_rank"] = out["breadth_regime"].map({"Low": 0, "Normal": 1, "High": 2}).astype(int)
    out["breadth_regime_metric"] = metric
    out["breadth_low_threshold"] = low_threshold
    out["breadth_high_threshold"] = high_threshold
    return out


def summarize_breadth_regime_periods(classified: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive rolling breadth labels into date blocks for research planning."""

    if classified.empty or "breadth_regime" not in classified.columns:
        return pd.DataFrame()

    work = classified.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "breadth_regime"]).sort_values("date").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    group_id = work["breadth_regime"].ne(work["breadth_regime"].shift()).cumsum()
    rows = []
    use_en = {
        "Low": "Stress-test diversification and check whether alpha is just macro beta.",
        "Normal": "Use as baseline train/test material.",
        "High": "Check whether alpha survives diversified, less crowded conditions.",
    }
    use_zh = {
        "Low": "用于压力测试分散化，并检查 alpha 是否只是宏观 beta。",
        "Normal": "作为基准训练/测试样本。",
        "High": "检查 alpha 在更分散、不拥挤的环境中是否仍然有效。",
    }
    for _, group in work.groupby(group_id, sort=False):
        regime = str(group["breadth_regime"].iloc[0])
        rows.append(
            {
                "start": group["date"].min(),
                "end": group["date"].max(),
                "breadth_regime": regime,
                "windows": int(len(group)),
                "avg_br95": float(pd.to_numeric(group.get("br95"), errors="coerce").mean())
                if "br95" in group
                else np.nan,
                "avg_effective_rank": float(pd.to_numeric(group.get("effective_rank"), errors="coerce").mean())
                if "effective_rank" in group
                else np.nan,
                "avg_participation_ratio": float(pd.to_numeric(group.get("participation_ratio"), errors="coerce").mean())
                if "participation_ratio" in group
                else np.nan,
                "avg_breadth_haircut": float(pd.to_numeric(group.get("breadth_haircut"), errors="coerce").mean())
                if "breadth_haircut" in group
                else np.nan,
                "research_use_en": use_en.get(regime, ""),
                "research_use_zh": use_zh.get(regime, ""),
            }
        )
    return pd.DataFrame(rows)


def compute_component_stability(
    returns: pd.DataFrame,
    *,
    baseline_loadings: pd.DataFrame,
    baseline_labels: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    config: RiskBreadthConfig | None = None,
) -> tuple[pd.DataFrame, int]:
    """Measure whether rolling PCA components keep the same economic meaning."""

    cfg = config or RiskBreadthConfig()
    sector_map = sector_map or InstrumentMaster("FUTURES_CN").get_sector_map()
    dates = returns.dropna(how="all").index
    if len(dates) < cfg.rolling_window or baseline_loadings.empty or baseline_labels.empty:
        return pd.DataFrame(), 1

    components = [
        f"PC{i + 1}"
        for i in range(
            min(
                int(cfg.stability_components),
                int(cfg.max_components),
                int(baseline_loadings["component_idx"].max()),
            )
        )
    ]
    if not components:
        return pd.DataFrame(), 1

    end_positions = list(range(cfg.rolling_window, len(dates) + 1, cfg.rolling_step))
    if cfg.stability_max_windows and len(end_positions) > cfg.stability_max_windows:
        end_positions = end_positions[-int(cfg.stability_max_windows) :]

    rows = []
    skipped = 0
    for end_pos in end_positions:
        window_dates = dates[end_pos - cfg.rolling_window : end_pos]
        window = returns.loc[window_dates]
        try:
            rolling_result = run_covariance_pca(window, sector_map=sector_map, config=cfg)
        except ValueError:
            skipped += 1
            continue

        rolling_loadings = rolling_result["asset_loadings"]
        rolling_labels = rolling_result["component_labels"]
        rolling_spectrum = rolling_result["spectrum"].set_index("component")

        for component in components:
            baseline_row = _component_label_row(baseline_labels, component)
            rolling_row = _component_label_row(rolling_labels, component)
            if baseline_row is None or rolling_row is None or component not in rolling_spectrum.index:
                skipped += 1
                continue

            baseline_sector = _dominant_sector_from_label_row(baseline_row)
            rolling_sector = _dominant_sector_from_label_row(rolling_row)
            similarity = _component_loading_similarity(
                baseline_loadings,
                rolling_loadings,
                component,
            )
            rows.append(
                {
                    "date": window_dates[-1],
                    "component": component,
                    "loading_similarity": similarity,
                    "label_match": bool(rolling_row.get("label_en") == baseline_row.get("label_en")),
                    "dominant_sector_match": bool(
                        baseline_sector and rolling_sector and baseline_sector == rolling_sector
                    ),
                    "baseline_dominant_sector": baseline_sector,
                    "baseline_dominant_sector_en": translate_sector_label(baseline_sector, "en"),
                    "baseline_dominant_sector_zh": translate_sector_label(baseline_sector, "zh"),
                    "window_dominant_sector": rolling_sector,
                    "window_dominant_sector_en": translate_sector_label(rolling_sector, "en"),
                    "window_dominant_sector_zh": translate_sector_label(rolling_sector, "zh"),
                    "baseline_label_en": baseline_row.get("label_en", ""),
                    "baseline_label_zh": baseline_row.get("label_zh", ""),
                    "window_label_en": rolling_row.get("label_en", ""),
                    "window_label_zh": rolling_row.get("label_zh", ""),
                    "label_confidence": float(rolling_row.get("label_confidence", np.nan)),
                    "explained_variance_ratio": float(
                        rolling_spectrum.loc[component, "explained_variance_ratio"]
                    ),
                }
            )

    return pd.DataFrame(rows), skipped


def compute_risk_factor_breadth(
    source_path: str | Path,
    config: RiskBreadthConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RiskBreadthConfig()
    raw = load_daily_market_data(source_path, asset_class=cfg.asset_class)
    selected_raw = select_assets_for_breadth(raw, cfg)
    sector_map = _sector_map_for_asset_class(cfg.asset_class, selected_raw)
    returns = compute_log_return_matrix(
        selected_raw,
        risk_imputation=cfg.risk_imputation,
        bridge_max_gap_bars=cfg.bridge_max_gap_bars,
        bridge_seed=cfg.bridge_seed,
    )
    quality_views = build_market_data_views(
        selected_raw,
        timestamp_col="date",
        asset_col="ticker",
        price_cols=("close",),
    )
    full = run_covariance_pca(returns, sector_map=sector_map, config=cfg)
    rolling, skipped = compute_rolling_breadth(returns, sector_map=sector_map, config=cfg)
    stability, stability_skipped = compute_component_stability(
        returns,
        baseline_loadings=full["asset_loadings"],
        baseline_labels=full["component_labels"],
        sector_map=sector_map,
        config=cfg,
    )
    full["rolling_breadth"] = rolling
    full["rolling_skipped_windows"] = skipped
    full["component_stability"] = stability
    full["component_stability_skipped_windows"] = stability_skipped
    full["source_rows"] = len(raw)
    full["source_assets"] = raw["ticker"].nunique()
    full["selected_rows"] = len(selected_raw)
    full["selected_assets"] = selected_raw["ticker"].nunique()
    full["asset_class"] = normalize_market_vertical(cfg.asset_class)
    full["data_quality"] = quality_views.quality_summary
    return full


def _sector_map_for_asset_class(asset_class: str, df: pd.DataFrame) -> dict[str, str]:
    normalized = normalize_market_vertical(asset_class)
    if normalized == "FUTURES_CN":
        return InstrumentMaster("FUTURES_CN").get_sector_map()
    if "sector" in df.columns:
        scoped = df.dropna(subset=["ticker", "sector"]).copy()
        if not scoped.empty:
            return (
                scoped.groupby(scoped["ticker"].astype(str))["sector"]
                .agg(lambda values: str(values.mode().iat[0]) if not values.mode().empty else "Unknown")
                .to_dict()
            )
    return {}


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


def _component_label_row(labels: pd.DataFrame, component: str) -> pd.Series | None:
    if labels.empty or "component" not in labels.columns:
        return None
    rows = labels[labels["component"] == component]
    if rows.empty:
        return None
    return rows.iloc[0]


def _dominant_sector_from_label_row(row: pd.Series) -> str:
    top_sectors = str(row.get("top_sectors", "")).strip()
    if not top_sectors:
        return ""
    return top_sectors.split(",")[0].strip()


def _component_loading_similarity(
    baseline_loadings: pd.DataFrame,
    rolling_loadings: pd.DataFrame,
    component: str,
) -> float:
    required = {"ticker", "component", "loading"}
    if not required.issubset(baseline_loadings.columns) or not required.issubset(rolling_loadings.columns):
        return np.nan
    baseline = baseline_loadings.loc[
        baseline_loadings["component"].eq(component), ["ticker", "loading"]
    ].rename(columns={"loading": "baseline_loading"})
    rolling = rolling_loadings.loc[
        rolling_loadings["component"].eq(component), ["ticker", "loading"]
    ].rename(columns={"loading": "rolling_loading"})
    merged = baseline.merge(rolling, on="ticker", how="inner").dropna()
    if len(merged) < 2:
        return np.nan

    a = merged["baseline_loading"].to_numpy(dtype=float)
    b = merged["rolling_loading"].to_numpy(dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return np.nan
    # PCA signs are arbitrary, so abs(cosine) is the stable orientation-free comparison.
    return float(np.clip(abs(np.dot(a, b) / denom), 0.0, 1.0))


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


_SECTOR_LABEL_EN = {
    "化工": "Chemicals",
    "黑色": "Ferrous metals",
    "能源": "Energy",
    "有色": "Base metals",
    "建材": "Building materials",
    "新能源": "New energy",
    "油脂油料": "Oils/oilseeds",
    "软商品": "Soft commodities",
    "生鲜": "Fresh products",
    "国债": "Rates",
    "股指": "Equity index",
    "航运": "Shipping",
    "贵金属": "Precious metals",
    "Unknown": "Unknown",
}


def translate_sector_label(sector: str, language: str = "en") -> str:
    """Translate internal sector labels for dashboard display."""

    if str(language).upper().startswith("ZH"):
        return str(sector)
    return _SECTOR_LABEL_EN.get(str(sector), str(sector))


def _sector_en(sector: str) -> str:
    return translate_sector_label(sector, "en")


def _signed_basket(asset_slice: pd.DataFrame, *, sign: int, top_n: int = 4) -> tuple[list[str], list[str]]:
    if "loading" not in asset_slice.columns:
        return [], []
    if sign > 0:
        scoped = asset_slice[asset_slice["loading"] > 0].sort_values("loading", ascending=False)
    else:
        scoped = asset_slice[asset_slice["loading"] < 0].sort_values("loading", ascending=True)
    if scoped.empty:
        return [], []

    assets = scoped.head(top_n)["base_symbol"].astype(str).tolist()
    sector_scores = (
        scoped.assign(abs_loading=scoped["loading"].abs())
        .groupby("sector", as_index=False)["abs_loading"]
        .sum()
        .sort_values("abs_loading", ascending=False)
    )
    sectors = sector_scores.head(3)["sector"].astype(str).tolist()
    return assets, sectors


def _basket_label(assets: list[str], sectors: list[str], *, language: str) -> str:
    if not assets:
        return "N/A"
    sector_text = " / ".join(sectors if language == "zh" else [_sector_en(sector) for sector in sectors])
    return f"{', '.join(assets)} ({sector_text})" if sector_text else ", ".join(assets)


def _component_label_confidence(top_share: float, label_en: str, positive_assets: list[str], negative_assets: list[str]) -> float:
    """Heuristic semantic-label confidence, not a statistical confidence interval."""
    concentration_score = np.clip(top_share, 0.0, 1.0)
    side_depth = min(len(positive_assets) + len(negative_assets), 8) / 8
    spread_bonus = 0.10 if positive_assets and negative_assets else 0.0
    mixed_penalty = 0.25 if label_en == "Mixed sector factor" else 0.0
    confidence = 0.20 + 0.55 * concentration_score + 0.15 * side_depth + spread_bonus - mixed_penalty
    return float(np.clip(confidence, 0.0, 1.0))


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
        positive_assets, positive_sectors = _signed_basket(asset_slice, sign=1)
        negative_assets, negative_sectors = _signed_basket(asset_slice, sign=-1)
        positive_top = positive_sectors[0] if positive_sectors else ""
        negative_top = negative_sectors[0] if negative_sectors else ""

        if positive_top and negative_top and positive_top != negative_top:
            label_en = f"{_sector_en(positive_top)} vs {_sector_en(negative_top)} spread"
            label_zh = f"{positive_top} vs {negative_top} 相对价差因子"
            interpretation_en = (
                "The component separates the positive-loading basket from the "
                "negative-loading basket. Treat it as a relative market spread, "
                "not a one-way beta."
            )
            interpretation_zh = "该主成分区分正载荷篮子与负载荷篮子，更像相对价差因子，而不是单边 beta。"
        elif len(top_set & industrial) >= 2:
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
                "positive_basket_en": _basket_label(positive_assets, positive_sectors, language="en"),
                "positive_basket_zh": _basket_label(positive_assets, positive_sectors, language="zh"),
                "negative_basket_en": _basket_label(negative_assets, negative_sectors, language="en"),
                "negative_basket_zh": _basket_label(negative_assets, negative_sectors, language="zh"),
                "top3_sector_share": top_share,
                "label_confidence": _component_label_confidence(
                    top_share,
                    label_en,
                    positive_assets,
                    negative_assets,
                ),
                "interpretation_en": interpretation_en,
                "interpretation_zh": interpretation_zh,
            }
        )

    return pd.DataFrame(rows)
