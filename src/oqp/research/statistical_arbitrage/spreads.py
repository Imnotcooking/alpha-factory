"""Reusable deterministic pair-spread mathematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


SPREAD_RETURN_RESIDUAL = "return_residual"
SPREAD_PRICE_RATIO = "price_ratio"
SPREAD_LINEAR_PRICE = "linear_price"
SPREAD_CONTRACT_VALUE = "contract_value"

__all__ = [
    "SPREAD_CONTRACT_VALUE",
    "SPREAD_LINEAR_PRICE",
    "SPREAD_PRICE_RATIO",
    "SPREAD_RETURN_RESIDUAL",
    "SpreadModelConfig",
    "build_price_matrix",
    "construct_pair_spread",
    "contract_multipliers_for_pair",
    "estimate_half_life",
    "estimate_ols_beta",
    "log_return_matrix",
    "rolling_zscore",
]


@dataclass(frozen=True)
class SpreadModelConfig:
    y_ticker: str
    x_ticker: str
    method: str = SPREAD_RETURN_RESIDUAL
    hedge_method: str = "ols"
    hedge_lookback: int = 504
    zscore_window: int = 126
    fixed_beta: float | None = None
    y_multiplier: float = 1.0
    x_multiplier: float = 1.0


def build_price_matrix(
    df: pd.DataFrame,
    *,
    value_col: str = "close",
) -> pd.DataFrame:
    """Return a date x ticker price matrix from long-form market data."""

    required = {"date", "ticker", value_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Market frame missing required columns: {missing}")

    work = df[["date", "ticker", value_col]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ticker"] = work["ticker"].astype(str)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["date", "ticker", value_col])
    work = work[work[value_col] > 0]
    if work.empty:
        return pd.DataFrame()

    return (
        work.pivot_table(index="date", columns="ticker", values=value_col, aggfunc="last")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )


def log_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    safe = prices.where(prices > 0)
    return np.log(safe / safe.shift(1)).replace([np.inf, -np.inf], np.nan)


def estimate_ols_beta(y: pd.Series, x: pd.Series) -> float:
    aligned = pd.concat([y, x], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    yv = aligned.iloc[:, 0].to_numpy(dtype=float)
    xv = aligned.iloc[:, 1].to_numpy(dtype=float)
    x_var = float(np.var(xv, ddof=1))
    if not np.isfinite(x_var) or x_var <= 0:
        return np.nan
    return float(np.cov(yv, xv, ddof=1)[0, 1] / x_var)


def rolling_zscore(series: pd.Series, window: int = 126) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_periods = max(20, min(int(window), max(2, int(window) // 3)))
    mean = values.rolling(int(window), min_periods=min_periods).mean()
    std = values.rolling(int(window), min_periods=min_periods).std(ddof=1)
    return ((values - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def estimate_half_life(series: pd.Series) -> float:
    """Estimate AR(1) mean-reversion half-life in observations."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 20:
        return np.nan

    lagged = values.shift(1)
    aligned = pd.concat([values, lagged], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan

    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    x = aligned.iloc[:, 1].to_numpy(dtype=float)
    x_var = float(np.var(x, ddof=1))
    if not np.isfinite(x_var) or x_var <= 0:
        return np.nan

    phi = float(np.cov(y, x, ddof=1)[0, 1] / x_var)
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.inf
    return float(-np.log(2.0) / np.log(phi))


def construct_pair_spread(
    df: pd.DataFrame,
    config: SpreadModelConfig,
) -> pd.DataFrame:
    prices = build_price_matrix(df)
    missing = [ticker for ticker in [config.y_ticker, config.x_ticker] if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Selected ticker(s) unavailable: {missing}")

    pair = prices[[config.y_ticker, config.x_ticker]].rename(
        columns={config.y_ticker: "y_price", config.x_ticker: "x_price"}
    )
    returns = log_return_matrix(pair).rename(columns={"y_price": "y_return", "x_price": "x_return"})
    out = pair.join(returns).dropna(subset=["y_price", "x_price"]).reset_index()

    beta_source_y: pd.Series
    beta_source_x: pd.Series
    if config.method == SPREAD_RETURN_RESIDUAL:
        beta_source_y = out["y_return"].tail(config.hedge_lookback)
        beta_source_x = out["x_return"].tail(config.hedge_lookback)
    elif config.method == SPREAD_CONTRACT_VALUE:
        beta_source_y = (out["y_price"] * float(config.y_multiplier)).tail(config.hedge_lookback)
        beta_source_x = (out["x_price"] * float(config.x_multiplier)).tail(config.hedge_lookback)
    else:
        beta_source_y = out["y_price"].tail(config.hedge_lookback)
        beta_source_x = out["x_price"].tail(config.hedge_lookback)

    beta = _resolve_beta(config, beta_source_y, beta_source_x)
    out["hedge_beta"] = beta
    out["spread_method"] = config.method

    if config.method == SPREAD_RETURN_RESIDUAL:
        out["spread"] = out["y_return"] - beta * out["x_return"]
        out["spread_units"] = "return_residual"
    elif config.method == SPREAD_PRICE_RATIO:
        out["spread"] = np.log(out["y_price"] / out["x_price"])
        out["spread_units"] = "log_price_ratio"
    elif config.method == SPREAD_LINEAR_PRICE:
        out["spread"] = out["y_price"] - beta * out["x_price"]
        out["spread_units"] = "price_points"
    elif config.method == SPREAD_CONTRACT_VALUE:
        y_value = out["y_price"] * float(config.y_multiplier)
        x_value = out["x_price"] * float(config.x_multiplier)
        out["spread"] = y_value - beta * x_value
        out["spread_units"] = "contract_value"
    else:
        raise ValueError(f"Unknown spread method: {config.method}")

    out["spread_z"] = rolling_zscore(out["spread"], config.zscore_window)
    out["spread_change"] = out["spread"].diff()
    out["half_life"] = estimate_half_life(out["spread"].tail(config.hedge_lookback))
    out["relationship"] = f"{config.y_ticker} ~ {config.x_ticker}"
    return out.replace([np.inf, -np.inf], np.nan)


def contract_multipliers_for_pair(
    y_ticker: str,
    x_ticker: str,
    multiplier_map: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    if not multiplier_map:
        return 1.0, 1.0
    return float(multiplier_map.get(y_ticker, 1.0)), float(multiplier_map.get(x_ticker, 1.0))


def _resolve_beta(config: SpreadModelConfig, y: pd.Series, x: pd.Series) -> float:
    if config.hedge_method == "fixed":
        return float(1.0 if config.fixed_beta is None else config.fixed_beta)
    beta = estimate_ols_beta(y, x)
    if not np.isfinite(beta):
        return float(1.0 if config.fixed_beta is None else config.fixed_beta)
    return beta
