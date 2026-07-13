#!/usr/bin/env python3
"""Reproduce the volatility-quartile switching table for the CN futures sleeves."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_FILE = REPO_ROOT / (
    "runtime/data/futures_cn/intraday/futures_data_futures_main_1m_adj_20260710_090753.parquet"
)
FACTOR_FILE = REPO_ROOT / (
    "departments/research/factors/daily_signals/"
    "fac_095_Intraday_Volatility_Quartile_Router_Futures_CN.py"
)
TREND_RETURNS = REPO_ROOT / "runtime/artifacts/research/returns/returns_run_5ba5a760.csv"
MEAN_RETURNS = REPO_ROOT / "runtime/artifacts/research/returns/returns_run_ee854132.csv"
OUTPUT_DIR = REPO_ROOT / "runtime/artifacts/research/volatility_quartile_router"
DAILY_FILE = OUTPUT_DIR / "cn_futures_volatility_quartile_router_daily.csv"
QUARTILE_FILE = OUTPUT_DIR / "cn_futures_volatility_quartile_sleeve_table.csv"
SUMMARY_FILE = OUTPUT_DIR / "cn_futures_volatility_quartile_router_summary.json"
REGIME_FILE = OUTPUT_DIR / "cn_futures_volatility_quartile_regimes.csv"


def load_router():
    spec = importlib.util.spec_from_file_location("fac095_quartile_study", FACTOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FACTOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_regimes(router, *, rebuild: bool) -> pd.DataFrame:
    if not rebuild and REGIME_FILE.exists():
        return pd.read_csv(
            REGIME_FILE,
            parse_dates=["router_calendar_day", "volatility_observation_day"],
        )
    if not rebuild and DAILY_FILE.exists():
        cached = pd.read_csv(
            DAILY_FILE,
            parse_dates=["router_calendar_day", "volatility_observation_day"],
        )
        columns = [
            "router_calendar_day",
            "volatility_observation_day",
            "market_realized_volatility",
            "market_volatility_q25",
            "market_volatility_q50",
            "market_volatility_q75",
            "market_volatility_quartile",
        ]
        regimes = cached[columns].drop_duplicates("router_calendar_day")
        regimes.to_csv(REGIME_FILE, index=False)
        return regimes
    raw = pl.read_parquet(DATA_FILE).to_pandas()
    if rebuild:
        import os

        os.environ["FAC_095_REBUILD_REGIME_CACHE"] = "1"
    prepared = router.prepare_data(raw)
    records = prepared.attrs.get("fac_095_regime_records", [])
    if not records:
        raise RuntimeError("fac_095 did not publish volatility regime records")
    regimes = pd.DataFrame(records).sort_values("router_calendar_day")
    regimes.to_csv(REGIME_FILE, index=False)
    return regimes


def annual_stats(returns: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    mean = float(clean.mean()) if len(clean) else np.nan
    volatility = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "observations": int(len(clean)),
        "total_return": float((1.0 + clean).prod() - 1.0) if len(clean) else np.nan,
        "annual_return_arithmetic": mean * 252.0,
        "annual_volatility": volatility * math.sqrt(252.0) if np.isfinite(volatility) else np.nan,
        "sharpe": mean / volatility * math.sqrt(252.0) if volatility > 0.0 else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-regimes",
        action="store_true",
        help="Recompute regimes from the full minute parquet. Memory intensive; cached regimes are the default.",
    )
    args = parser.parse_args()
    router = load_router()
    regimes = load_regimes(router, rebuild=args.rebuild_regimes)
    trend = pd.read_csv(TREND_RETURNS, parse_dates=["date"])
    mean = pd.read_csv(MEAN_RETURNS, parse_dates=["date"])
    trend = trend.rename(columns={column: f"trend_{column}" for column in trend.columns if column != "date"})
    mean = mean.rename(
        columns={column: f"mean_reversion_{column}" for column in mean.columns if column != "date"}
    )
    daily = trend.merge(mean, on="date", how="inner")
    daily = daily.merge(regimes, left_on="date", right_on="router_calendar_day", how="left")
    daily["static_50_50_return"] = 0.5 * (
        daily["trend_net_return"] + daily["mean_reversion_net_return"]
    )
    daily["paper_router_return"] = np.where(
        daily["market_volatility_quartile"].eq(4.0),
        daily["mean_reversion_net_return"],
        daily["trend_net_return"],
    )
    daily.loc[daily["market_volatility_quartile"].isna(), "paper_router_return"] = np.nan
    high_volatility = daily["market_volatility_quartile"].eq(4.0)
    for column in [
        "gross_return",
        "benchmark_return",
        "daily_turnover",
        "daily_total_cost",
        "daily_cost_bps",
        "portfolio_leverage",
    ]:
        daily[f"paper_router_{column}"] = np.where(
            high_volatility,
            daily[f"mean_reversion_{column}"],
            daily[f"trend_{column}"],
        )

    trend_volatility = float(daily["trend_net_return"].std(ddof=1))
    mean_volatility = float(daily["mean_reversion_net_return"].std(ddof=1))
    inverse_vol_mean_weight = trend_volatility / (trend_volatility + mean_volatility)
    inverse_vol_trend_weight = 1.0 - inverse_vol_mean_weight
    daily["static_inverse_vol_return"] = (
        inverse_vol_trend_weight * daily["trend_net_return"]
        + inverse_vol_mean_weight * daily["mean_reversion_net_return"]
    )

    quartile_rows: list[dict[str, float | int]] = []
    for quartile in range(1, 5):
        sample = daily.loc[daily["market_volatility_quartile"].eq(float(quartile))]
        spread = sample["mean_reversion_net_return"] - sample["trend_net_return"]
        spread_standard_error = (
            float(spread.std(ddof=1) / math.sqrt(len(spread))) if len(spread) > 1 else np.nan
        )
        quartile_rows.append(
            {
                "quartile": quartile,
                "days": int(len(sample)),
                "avg_trend_bps": float(sample["trend_net_return"].mean() * 10_000.0),
                "avg_mean_reversion_bps": float(sample["mean_reversion_net_return"].mean() * 10_000.0),
                "avg_mean_minus_trend_bps": float(spread.mean() * 10_000.0),
                "mean_minus_trend_t_stat": (
                    float(spread.mean() / spread_standard_error)
                    if np.isfinite(spread_standard_error) and spread_standard_error > 0.0
                    else np.nan
                ),
                "trend_hit_rate": float(sample["trend_net_return"].gt(0.0).mean()),
                "mean_reversion_hit_rate": float(sample["mean_reversion_net_return"].gt(0.0).mean()),
            }
        )
    quartile_table = pd.DataFrame(quartile_rows)

    summary = {
        "method": {
            "paper_rule": "route next period to trend in Q1-Q3 and reversal in Q4",
            "clock_mapping": "paper month -> CN futures observed calendar day",
            "volatility_mapping": "paper daily market returns -> equal-weight one-minute futures-market returns",
            "history_mapping": "paper rolling 5 years -> rolling 252 observed days",
            "lookahead_control": "day-t realized volatility routes only the next observed day",
        },
        "trend": annual_stats(daily["trend_net_return"]),
        "mean_reversion": annual_stats(daily["mean_reversion_net_return"]),
        "static_50_50": annual_stats(daily["static_50_50_return"]),
        "static_inverse_vol_ex_post": {
            **annual_stats(daily["static_inverse_vol_return"]),
            "trend_weight": inverse_vol_trend_weight,
            "mean_reversion_weight": inverse_vol_mean_weight,
            "warning": "Weights use full-window realized volatility and are a descriptive control, not deployable ex ante.",
        },
        "paper_router_return_level_estimate": annual_stats(daily["paper_router_return"]),
        "paper_router_execution_proxy": {
            "annual_gross_return_arithmetic": float(daily["paper_router_gross_return"].mean() * 252.0),
            "average_daily_turnover": float(daily["paper_router_daily_turnover"].mean()),
            "annual_turnover": float(daily["paper_router_daily_turnover"].mean() * 252.0),
            "average_daily_cost_bps": float(daily["paper_router_daily_cost_bps"].mean()),
            "total_modeled_cost_cny": float(daily["paper_router_daily_total_cost"].sum()),
            "max_gross_leverage": float(daily["paper_router_portfolio_leverage"].max()),
            "selected_same_horizon_benchmark": annual_stats(daily["paper_router_benchmark_return"]),
            "assumption": (
                "Selects each standalone run's gross return, net return, turnover, and cost on the routed day. "
                "The sleeves are session-local and the regime is carried across split Friday-night/Saturday rows."
            ),
        },
        "daily_return_correlation": float(
            daily["trend_net_return"].corr(daily["mean_reversion_net_return"])
        ),
        "quartiles": quartile_rows,
        "warning": (
            "The artifact router selects already-costed standalone sleeve days. A combined signal-engine run "
            "would be preferable, but is intentionally not attempted on the 8 GB workstation because loading "
            "both 4.3-million-row factor states together exhausts memory."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_FILE, index=False)
    quartile_table.to_csv(QUARTILE_FILE, index=False)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(quartile_table.to_string(index=False))
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"saved daily: {DAILY_FILE}")
    print(f"saved quartiles: {QUARTILE_FILE}")
    print(f"saved summary: {SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
