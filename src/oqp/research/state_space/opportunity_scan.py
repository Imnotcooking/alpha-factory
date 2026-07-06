from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.data.instruments import InstrumentMaster
from oqp.risk.factor_breadth import extract_base_symbol, map_chinese_futures_sector
from oqp.research.state_space.opportunity_scoring import liquidity_score_from_rank, score_opportunity
from oqp.research.state_space.spread_models import (
    SpreadModelConfig,
    construct_pair_spread,
    estimate_half_life,
    estimate_ols_beta,
    log_return_matrix,
    rolling_zscore,
)


ARBITRAGE_CALENDAR = "calendar"
ARBITRAGE_CROSS_PRODUCT = "cross_product"
ARBITRAGE_STATISTICAL = "statistical"

__all__ = [
    "ARBITRAGE_CALENDAR",
    "ARBITRAGE_CROSS_PRODUCT",
    "ARBITRAGE_STATISTICAL",
    "DataAuditConfig",
    "OpportunityScanConfig",
    "build_asset_metadata",
    "classify_arbitrage_type",
    "compute_data_audit",
    "construct_spread_for_candidate",
    "normalize_daily_market_frame",
    "run_opportunity_scan",
    "selected_candidate_config",
]


@dataclass(frozen=True)
class OpportunityScanConfig:
    min_observations: int = 252
    lookback: int = 504
    zscore_window: int = 126
    max_assets: int = 40
    min_abs_correlation: float = 0.15
    include_cross_sector: bool = True
    asset_class: str = "FUTURES_CN"


@dataclass(frozen=True)
class DataAuditConfig:
    min_observations: int = 252
    stale_days: int = 45


def normalize_daily_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Daily market frame missing required columns: {missing}")

    cols = [
        col
        for col in ["date", "ticker", "open", "high", "low", "close", "volume", "oi", "open_interest", "sector"]
        if col in df.columns
    ]
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str)
    for col in ["open", "high", "low", "close", "volume", "oi", "open_interest"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "close"])
    out = out[out["close"] > 0]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_asset_metadata(df: pd.DataFrame, asset_class: str = "FUTURES_CN") -> pd.DataFrame:
    market = normalize_daily_market_frame(df)
    if market.empty:
        return pd.DataFrame()

    master = InstrumentMaster(asset_class)
    grouped = market.groupby("ticker", as_index=False).agg(
        observations=("close", "size"),
        start=("date", "min"),
        end=("date", "max"),
        avg_close=("close", "mean"),
        avg_volume=("volume", "mean") if "volume" in market.columns else ("close", "size"),
    )
    rows = []
    for row in grouped.itertuples(index=False):
        ticker = str(row.ticker)
        profile = master.get_profile(ticker)
        rows.append(
            {
                "ticker": ticker,
                "base_symbol": extract_base_symbol(ticker),
                "sector": profile.sector or map_chinese_futures_sector(ticker),
                "exchange": profile.exchange,
                "multiplier": float(profile.multiplier),
                "tick_size": float(profile.tick_size),
                "margin_rate": float(profile.margin_rate),
                "fee_type": profile.fee_type,
                "fee_open": float(profile.fee_open),
                "fee_close_history": float(profile.fee_close_history),
                "observations": int(row.observations),
                "start": row.start,
                "end": row.end,
                "avg_close": float(row.avg_close),
                "avg_volume": float(row.avg_volume) if pd.notna(row.avg_volume) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_opportunity_scan(
    df: pd.DataFrame,
    config: OpportunityScanConfig | None = None,
) -> dict[str, Any]:
    cfg = config or OpportunityScanConfig()
    market = normalize_daily_market_frame(df)
    metadata = build_asset_metadata(market, asset_class=cfg.asset_class)
    if market.empty or metadata.empty:
        return {"candidates": pd.DataFrame(), "metadata": metadata, "returns": pd.DataFrame()}

    eligible = metadata[metadata["observations"] >= int(cfg.min_observations)].copy()
    if eligible.empty or len(eligible) < 2:
        return {"candidates": pd.DataFrame(), "metadata": metadata, "returns": pd.DataFrame()}

    eligible = _rank_assets_for_scan(eligible).head(int(cfg.max_assets))
    selected = eligible["ticker"].tolist()
    prices = _price_matrix(market, selected)
    returns = log_return_matrix(prices)
    rows = []
    meta = eligible.set_index("ticker").to_dict(orient="index")
    for y_idx, y_ticker in enumerate(selected):
        for x_ticker in selected[y_idx + 1 :]:
            metrics = _pair_metrics(
                prices[[y_ticker, x_ticker]],
                returns[[y_ticker, x_ticker]],
                y_ticker,
                x_ticker,
                meta,
                cfg,
            )
            if not metrics:
                continue
            if abs(metrics["correlation"]) < float(cfg.min_abs_correlation):
                continue
            if not cfg.include_cross_sector and metrics["y_sector"] != metrics["x_sector"]:
                continue
            rows.append(metrics)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return {"candidates": candidates, "metadata": metadata, "returns": returns}

    candidates["liquidity_score"] = liquidity_score_from_rank(np.log1p(candidates["avg_pair_volume"].fillna(0.0)))
    scored = candidates.apply(score_opportunity, axis=1, result_type="expand")
    candidates = pd.concat([candidates, scored], axis=1)
    candidates = candidates.sort_values(
        ["opportunity_score", "abs_latest_z", "stability_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return {"candidates": candidates, "metadata": metadata, "returns": returns}


def classify_arbitrage_type(y_meta: dict[str, Any], x_meta: dict[str, Any]) -> str:
    if y_meta.get("base_symbol") == x_meta.get("base_symbol") and y_meta.get("ticker") != x_meta.get("ticker"):
        return ARBITRAGE_CALENDAR
    if y_meta.get("sector") == x_meta.get("sector"):
        return ARBITRAGE_CROSS_PRODUCT
    return ARBITRAGE_STATISTICAL


def compute_data_audit(df: pd.DataFrame, config: DataAuditConfig | None = None) -> dict[str, Any]:
    cfg = config or DataAuditConfig()
    market = normalize_daily_market_frame(df)
    if market.empty:
        return {"summary": {}, "assets": pd.DataFrame(), "schema": pd.DataFrame()}

    latest_date = market["date"].max()
    assets = market.groupby("ticker", as_index=False).agg(
        observations=("close", "size"),
        start=("date", "min"),
        end=("date", "max"),
        missing_close=("close", lambda s: int(s.isna().sum())),
        duplicate_dates=("date", lambda s: int(s.duplicated().sum())),
    )
    assets["base_symbol"] = assets["ticker"].map(extract_base_symbol)
    assets["sector"] = assets["ticker"].map(map_chinese_futures_sector)
    assets["eligible"] = assets["observations"] >= int(cfg.min_observations)
    assets["days_since_last"] = (latest_date - assets["end"]).dt.days
    assets["stale"] = assets["days_since_last"] > int(cfg.stale_days)

    schema = pd.DataFrame(
        {
            "column": list(df.columns),
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": [int(df[col].notna().sum()) for col in df.columns],
        }
    )
    summary = {
        "rows": int(len(market)),
        "assets": int(market["ticker"].nunique()),
        "eligible_assets": int(assets["eligible"].sum()),
        "date_min": market["date"].min(),
        "date_max": latest_date,
        "has_contract_level_duplicates": bool(assets["base_symbol"].duplicated().any()),
    }
    return {"summary": summary, "assets": assets, "schema": schema}


def selected_candidate_config(
    candidate: pd.Series,
    *,
    method: str,
    hedge_method: str = "ols",
    hedge_lookback: int = 504,
    zscore_window: int = 126,
) -> SpreadModelConfig:
    return SpreadModelConfig(
        y_ticker=str(candidate["y_ticker"]),
        x_ticker=str(candidate["x_ticker"]),
        method=method,
        hedge_method=hedge_method,
        hedge_lookback=hedge_lookback,
        zscore_window=zscore_window,
        y_multiplier=float(candidate.get("y_multiplier", 1.0) or 1.0),
        x_multiplier=float(candidate.get("x_multiplier", 1.0) or 1.0),
    )


def construct_spread_for_candidate(
    df: pd.DataFrame,
    candidate: pd.Series,
    *,
    method: str,
    hedge_method: str = "ols",
    hedge_lookback: int = 504,
    zscore_window: int = 126,
) -> pd.DataFrame:
    cfg = selected_candidate_config(
        candidate,
        method=method,
        hedge_method=hedge_method,
        hedge_lookback=hedge_lookback,
        zscore_window=zscore_window,
    )
    return construct_pair_spread(df, cfg)


def _rank_assets_for_scan(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    out["volume_rank"] = np.log1p(out["avg_volume"].fillna(0.0)).rank(pct=True)
    out["history_rank"] = out["observations"].rank(pct=True)
    out["scan_rank"] = 0.65 * out["history_rank"] + 0.35 * out["volume_rank"]
    return out.sort_values(["scan_rank", "observations", "ticker"], ascending=[False, False, True])


def _price_matrix(market: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    scoped = market[market["ticker"].isin(selected)]
    return (
        scoped.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )


def _pair_metrics(
    price_pair: pd.DataFrame,
    return_pair: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    meta: dict[str, dict[str, Any]],
    cfg: OpportunityScanConfig,
) -> dict[str, Any]:
    aligned_returns = return_pair.dropna().tail(int(cfg.lookback))
    aligned_prices = price_pair.dropna().tail(int(cfg.lookback))
    if len(aligned_returns) < int(cfg.min_observations):
        return {}

    y_ret = aligned_returns[y_ticker]
    x_ret = aligned_returns[x_ticker]
    correlation = float(y_ret.corr(x_ret))
    beta = estimate_ols_beta(y_ret, x_ret)
    if not np.isfinite(correlation) or not np.isfinite(beta):
        return {}

    residual = y_ret - beta * x_ret
    residual_z = rolling_zscore(residual, int(cfg.zscore_window))
    latest_z = float(residual_z.dropna().iloc[-1]) if residual_z.notna().any() else np.nan
    half_life = estimate_half_life(residual)
    beta_drift = _beta_drift(y_ret, x_ret)
    y_meta = {**meta[y_ticker], "ticker": y_ticker}
    x_meta = {**meta[x_ticker], "ticker": x_ticker}
    arb_type = classify_arbitrage_type(y_meta, x_meta)
    round_turn_cost = _pair_round_turn_cost_bps(y_meta, x_meta)
    avg_pair_volume = float(np.nanmean([y_meta.get("avg_volume", np.nan), x_meta.get("avg_volume", np.nan)]))

    return {
        "candidate_id": f"{y_ticker} ~ {x_ticker}",
        "y_ticker": y_ticker,
        "x_ticker": x_ticker,
        "arbitrage_type": arb_type,
        "y_sector": y_meta.get("sector", "Unknown"),
        "x_sector": x_meta.get("sector", "Unknown"),
        "sector_pair": _sector_pair(y_meta.get("sector", "Unknown"), x_meta.get("sector", "Unknown")),
        "y_base": y_meta.get("base_symbol", ""),
        "x_base": x_meta.get("base_symbol", ""),
        "observations": int(len(aligned_returns)),
        "correlation": correlation,
        "abs_correlation": abs(correlation),
        "beta": float(beta),
        "beta_drift": float(beta_drift),
        "latest_z": latest_z,
        "abs_latest_z": abs(latest_z) if np.isfinite(latest_z) else np.nan,
        "residual_vol": float(residual.std(ddof=1)),
        "half_life": float(half_life) if np.isfinite(half_life) else np.inf,
        "round_turn_cost_bps": round_turn_cost,
        "avg_pair_volume": avg_pair_volume,
        "y_multiplier": float(y_meta.get("multiplier", 1.0) or 1.0),
        "x_multiplier": float(x_meta.get("multiplier", 1.0) or 1.0),
        "latest_y_price": float(aligned_prices[y_ticker].dropna().iloc[-1]) if not aligned_prices.empty else np.nan,
        "latest_x_price": float(aligned_prices[x_ticker].dropna().iloc[-1]) if not aligned_prices.empty else np.nan,
    }


def _beta_drift(y: pd.Series, x: pd.Series) -> float:
    if len(y) < 80:
        return np.nan
    split = len(y) // 2
    first = estimate_ols_beta(y.iloc[:split], x.iloc[:split])
    second = estimate_ols_beta(y.iloc[split:], x.iloc[split:])
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(abs(second - first))


def _sector_pair(y_sector: str, x_sector: str) -> str:
    parts = sorted([str(y_sector), str(x_sector)])
    return f"{parts[0]} / {parts[1]}"


def _pair_round_turn_cost_bps(y_meta: dict[str, Any], x_meta: dict[str, Any]) -> float:
    return float(_one_leg_round_turn_bps(y_meta) + _one_leg_round_turn_bps(x_meta))


def _one_leg_round_turn_bps(meta: dict[str, Any]) -> float:
    fee_type = str(meta.get("fee_type", "ratio"))
    open_fee = float(meta.get("fee_open", 0.0) or 0.0)
    close_fee = float(meta.get("fee_close_history", 0.0) or 0.0)
    if fee_type == "ratio":
        return (open_fee + close_fee) * 10000.0
    notional = float(meta.get("avg_close", np.nan)) * float(meta.get("multiplier", 1.0) or 1.0)
    if not np.isfinite(notional) or notional <= 0:
        return np.nan
    return ((open_fee + close_fee) / notional) * 10000.0
