from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.research.state_space.base_filter import StateSpaceSchema
from oqp.research.state_space.dual_kalman_regression import DualKalmanRegressionConfig
from oqp.research.state_space.feature_adapter import build_dual_kalman_features


__all__ = [
    "RelationshipLabConfig",
    "available_tickers",
    "build_relationship_frame",
    "list_daily_price_files",
    "load_daily_price_frame",
    "relationship_summary",
    "run_relationship_dkf",
]


@dataclass(frozen=True)
class RelationshipLabConfig:
    y_ticker: str
    x_ticker: str
    y_label: str = "Y"
    x_label: str = "X"
    return_type: str = "log"
    process_noise: float = 1e-4
    observation_noise: float = 1e-4
    initial_state_covariance: float = 10.0


def list_daily_price_files(base_dir: str | Path) -> list[Path]:
    repo_root = Path(base_dir)
    runtime_daily = repo_root / "runtime" / "data" / "alpha_lab" / "market_data" / "daily"
    patterns = ["*1d*index*.parquet", "*1d*main*.parquet", "*daily*.parquet"]
    seen: dict[Path, None] = {}
    for pattern in patterns:
        for path in sorted(runtime_daily.glob(pattern) if runtime_daily.exists() else []):
            if path.is_file():
                seen[path.resolve()] = None
    root_matrix = repo_root / "runtime" / "data" / "alpha_lab" / "feature_store" / "ML_Feature_Matrix.parquet"
    if root_matrix.exists():
        seen[root_matrix.resolve()] = None
    return list(seen.keys())


def load_daily_price_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Daily price file missing required columns: {sorted(missing)}")
    cols = [col for col in ["date", "ticker", "open", "high", "low", "close", "volume", "oi", "open_interest"] if col in df.columns]
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


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


def build_relationship_frame(df: pd.DataFrame, config: RelationshipLabConfig) -> pd.DataFrame:
    returns = _wide_returns(df, return_type=config.return_type)
    missing = [ticker for ticker in [config.y_ticker, config.x_ticker] if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Selected ticker(s) unavailable after return calculation: {missing}")

    pair = (
        returns[[config.y_ticker, config.x_ticker]]
        .rename(columns={config.y_ticker: "y_return", config.x_ticker: "x_return"})
        .dropna()
        .reset_index()
    )
    pair["relationship"] = f"{config.y_label} ~ {config.x_label}"
    pair["ticker"] = config.y_ticker
    return pair


def run_relationship_dkf(df: pd.DataFrame, config: RelationshipLabConfig) -> dict[str, Any]:
    pair = build_relationship_frame(df, config)
    dkf_config = DualKalmanRegressionConfig(
        schema=StateSpaceSchema(
            date_col="date",
            y_col="y_return",
            x_cols=("x_return",),
            group_cols=("ticker",),
        ),
        process_noise=float(config.process_noise),
        observation_noise=float(config.observation_noise),
        initial_state_covariance=float(config.initial_state_covariance),
    )
    features = build_dual_kalman_features(pair, dkf_config)
    merged = pair.merge(
        features.drop(columns=["y_return"], errors="ignore"),
        on=["date", "ticker"],
        how="left",
    )
    merged["spread_return"] = merged["dkf_residual"]
    merged["residual_z"] = merged["dkf_innovation_z"]
    merged["dynamic_beta"] = merged["dkf_beta_x_return"]
    merged["dynamic_alpha"] = merged["dkf_beta_intercept"]
    merged["state_uncertainty"] = merged["dkf_state_uncertainty"]
    merged["beta_l1_change"] = merged["dkf_beta_l1_change"]
    return {
        "pair": merged,
        "features": features,
        "config": dkf_config,
        "summary": relationship_summary(merged),
    }


def relationship_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    beta = pd.to_numeric(frame.get("dynamic_beta"), errors="coerce")
    residual_z = pd.to_numeric(frame.get("residual_z"), errors="coerce")
    residual = pd.to_numeric(frame.get("spread_return"), errors="coerce")
    return {
        "rows": float(len(frame)),
        "beta_latest": float(beta.dropna().iloc[-1]) if beta.notna().any() else np.nan,
        "beta_median": float(beta.median()) if beta.notna().any() else np.nan,
        "beta_change": float(beta.dropna().iloc[-1] - beta.dropna().iloc[0]) if beta.notna().sum() > 1 else np.nan,
        "residual_z_latest": float(residual_z.dropna().iloc[-1]) if residual_z.notna().any() else np.nan,
        "residual_vol": float(residual.std()) if residual.notna().sum() > 1 else np.nan,
        "extreme_z_rate": float(residual_z.abs().gt(2.0).mean()) if residual_z.notna().any() else np.nan,
    }


def _wide_returns(df: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close.where(close > 0)
    close = close.dropna(axis=1, how="all")
    if return_type == "log":
        returns = np.log(close / close.shift(1))
    elif return_type == "pct":
        returns = close.pct_change()
    else:
        raise ValueError("return_type must be 'log' or 'pct'.")
    return returns.replace([np.inf, -np.inf], np.nan)
