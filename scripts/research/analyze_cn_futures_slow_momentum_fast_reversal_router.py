#!/usr/bin/env python3
"""Test slow cross-sectional momentum against fast reversal under frozen volatility routing."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MATCHED_SCRIPT = REPO_ROOT / "scripts/research/analyze_cn_futures_matched_momentum_reversal_router.py"
OUTPUT_DIR = REPO_ROOT / "runtime/artifacts/research/slow_momentum_fast_reversal_router"
DAILY_FILE = OUTPUT_DIR / "cn_futures_slow_momentum_fast_reversal_daily.csv"
QUARTILE_FILE = OUTPUT_DIR / "cn_futures_slow_momentum_fast_reversal_quartiles.csv"
SUMMARY_FILE = OUTPUT_DIR / "cn_futures_slow_momentum_fast_reversal_summary.json"

MOMENTUM_HORIZONS = (1, 3, 5)
DAILY_VOLATILITY_LOOKBACK = 20
REGIME_LOOKBACK_DAYS = 252
LIQUID_CONTRACTS = 40
TAIL_FRACTION = 0.20
MIN_CROSS_SECTION = 20
HOLDOUT_START = pd.Timestamp("2025-10-09")


def load_matched_module():
    spec = importlib.util.spec_from_file_location("matched_router_cache", MATCHED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import compact-cache builder: {MATCHED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trading_day(timestamps: pd.Series) -> pd.Series:
    values = pd.to_datetime(timestamps, errors="coerce")
    days = values.dt.normalize() + pd.to_timedelta(values.dt.hour.ge(21).astype(int), unit="D")
    weekday = days.dt.weekday
    weekend_adjustment = np.select([weekday.eq(5), weekday.eq(6)], [2, 1], default=0)
    return days + pd.to_timedelta(weekend_adjustment, unit="D")


def trade_day_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    matched = load_matched_module()
    matched.build_bar_cache(rebuild=False)
    bars = matched.prepare_bars()
    bars["trading_day"] = trading_day(bars["date"])
    bars = bars.sort_values(["ticker", "date"]).reset_index(drop=True)

    daily = (
        bars.groupby(["ticker", "trading_day"], sort=False, observed=True)
        .agg(
            open=("open", "first"),
            close=("close", "last"),
            volume=("volume", "sum"),
            notional=("bar_notional", "sum"),
            roll_count=("roll_count", "sum"),
            open_cost_rate=("one_way_cost_rate", "first"),
            close_cost_rate=("one_way_cost_rate", "last"),
            source_bars=("date", "size"),
        )
        .reset_index()
        .sort_values(["ticker", "trading_day"])
        .reset_index(drop=True)
    )
    ticker_group = daily.groupby("ticker", sort=False, observed=True)
    previous_close = ticker_group["close"].shift(1)
    daily["daily_log_return"] = np.log(daily["close"] / previous_close)
    daily["lagged_daily_sigma"] = ticker_group["daily_log_return"].transform(
        lambda values: values.rolling(
            DAILY_VOLATILITY_LOOKBACK,
            min_periods=DAILY_VOLATILITY_LOOKBACK // 2,
        ).std().shift(1)
    )
    daily["next_trading_day"] = ticker_group["trading_day"].shift(-1)
    daily["next_open"] = ticker_group["open"].shift(-1)
    daily["next_close"] = ticker_group["close"].shift(-1)
    daily["next_open_cost_rate"] = ticker_group["open_cost_rate"].shift(-1)
    daily["next_close_cost_rate"] = ticker_group["close_cost_rate"].shift(-1)
    daily["forward_return"] = daily["next_close"] / daily["next_open"] - 1.0
    daily.loc[daily["forward_return"].abs().gt(0.15), "forward_return"] = np.nan

    daily["lagged_median_notional"] = ticker_group["notional"].transform(
        lambda values: values.rolling(20, min_periods=10).median().shift(1)
    )
    daily["liquidity_rank"] = daily.groupby("trading_day", sort=False, observed=True)[
        "lagged_median_notional"
    ].rank(method="first", ascending=False)

    market_rows = bars.loc[
        bars["bar_log_return"].notna() & bars["roll_count"].eq(0.0),
        ["date", "trading_day", "bar_log_return"],
    ]
    market_bars = market_rows.groupby(["trading_day", "date"], sort=True).agg(
        market_return=("bar_log_return", "mean"),
        contracts=("bar_log_return", "size"),
    )
    market_bars = market_bars.loc[market_bars["contracts"].ge(MIN_CROSS_SECTION)].reset_index()
    market_volatility = market_bars.assign(square=market_bars["market_return"].pow(2)).groupby(
        "trading_day", sort=True
    )["square"].sum().pow(0.5)
    rolling = market_volatility.rolling(REGIME_LOOKBACK_DAYS, min_periods=REGIME_LOOKBACK_DAYS)
    q25 = rolling.quantile(0.25)
    q50 = rolling.quantile(0.50)
    q75 = rolling.quantile(0.75)
    quartile = pd.Series(np.nan, index=market_volatility.index, dtype=float)
    ready = q75.notna()
    quartile.loc[ready & market_volatility.le(q25)] = 1.0
    quartile.loc[ready & market_volatility.gt(q25) & market_volatility.le(q50)] = 2.0
    quartile.loc[ready & market_volatility.gt(q50) & market_volatility.le(q75)] = 3.0
    quartile.loc[ready & market_volatility.gt(q75)] = 4.0
    regimes = pd.DataFrame(
        {
            "signal_day": market_volatility.index,
            "market_realized_volatility": market_volatility.to_numpy(),
            "market_volatility_q25": q25.to_numpy(),
            "market_volatility_q50": q50.to_numpy(),
            "market_volatility_q75": q75.to_numpy(),
            "market_volatility_quartile": quartile.to_numpy(),
        }
    )
    return daily, regimes


def cross_sectional_weights(daily: pd.DataFrame, *, momentum_horizon: int) -> pd.DataFrame:
    out = daily.copy()
    ticker_group = out.groupby("ticker", sort=False, observed=True)
    end = ticker_group["close"].shift(1)
    start = ticker_group["close"].shift(int(momentum_horizon) + 1)
    out["momentum_formation_return"] = np.log(end / start)
    out["momentum_formation_roll_count"] = ticker_group["roll_count"].transform(
        lambda values: values.shift(1).rolling(
            int(momentum_horizon),
            min_periods=int(momentum_horizon),
        ).sum()
    )
    sigma = out["lagged_daily_sigma"].replace(0.0, np.nan)
    out["momentum_score"] = out["momentum_formation_return"] / (
        sigma * math.sqrt(float(momentum_horizon))
    )
    out["reversal_score"] = -out["daily_log_return"] / sigma
    common = (
        out["liquidity_rank"].le(LIQUID_CONTRACTS)
        & out["momentum_score"].replace([np.inf, -np.inf], np.nan).notna()
        & out["reversal_score"].replace([np.inf, -np.inf], np.nan).notna()
        & out["forward_return"].notna()
        & out["momentum_formation_roll_count"].eq(0.0)
        & out["roll_count"].eq(0.0)
    )
    count = common.groupby(out["trading_day"], sort=False).transform("sum")
    common &= count.ge(MIN_CROSS_SECTION)

    for sleeve in ["momentum", "reversal"]:
        score_column = f"{sleeve}_score"
        residual = out.loc[common, score_column] - out.loc[common].groupby(
            "trading_day", sort=False
        )[score_column].transform("mean")
        percentile = residual.groupby(out.loc[common, "trading_day"], sort=False).rank(pct=True)
        long = percentile.ge(1.0 - TAIL_FRACTION)
        short = percentile.le(TAIL_FRACTION)
        long_count = long.groupby(out.loc[common, "trading_day"], sort=False).transform("sum")
        short_count = short.groupby(out.loc[common, "trading_day"], sort=False).transform("sum")
        weights = pd.Series(0.0, index=out.index, dtype="float32")
        weights.loc[long.index[long]] = (0.5 / long_count.loc[long]).astype("float32")
        weights.loc[short.index[short]] = (-0.5 / short_count.loc[short]).astype("float32")
        out[f"{sleeve}_weight"] = weights
    return out


def sleeve_returns(weighted: pd.DataFrame, sleeve: str) -> pd.DataFrame:
    weight = pd.to_numeric(weighted[f"{sleeve}_weight"], errors="coerce").fillna(0.0)
    gross_contribution = weight * weighted["forward_return"].fillna(0.0)
    round_trip_cost = weight.abs() * (
        weighted["next_open_cost_rate"].fillna(0.0)
        + weighted["next_close_cost_rate"].fillna(0.0)
    )
    rows = pd.DataFrame(
        {
            "signal_day": weighted["trading_day"],
            "execution_day": weighted["next_trading_day"],
            "gross_contribution": gross_contribution,
            "cost": round_trip_cost,
            "gross": weight.abs(),
            "net": weight,
        }
    )
    portfolio = rows.groupby("signal_day", sort=True).agg(
        execution_day=("execution_day", "max"),
        gross_return=("gross_contribution", "sum"),
        cost=("cost", "sum"),
        gross_exposure=("gross", "sum"),
        net_exposure=("net", "sum"),
    )
    portfolio["net_return"] = portfolio["gross_return"] - portfolio["cost"]
    portfolio["turnover"] = 2.0 * portfolio["gross_exposure"]
    return portfolio.add_prefix(f"{sleeve}_")


def build_horizon_result(
    daily: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    momentum_horizon: int,
) -> pd.DataFrame:
    weighted = cross_sectional_weights(daily, momentum_horizon=momentum_horizon)
    momentum = sleeve_returns(weighted, "momentum")
    reversal = sleeve_returns(weighted, "reversal")
    result = momentum.join(reversal, how="outer").reset_index()
    result["execution_day"] = result["momentum_execution_day"].combine_first(
        result["reversal_execution_day"]
    )
    result = result.merge(regimes, on="signal_day", how="left")
    high_volatility = result["market_volatility_quartile"].eq(4.0)
    routable = result["market_volatility_quartile"].between(1.0, 4.0, inclusive="both")
    for field in ["gross_return", "net_return", "cost", "turnover", "gross_exposure"]:
        result[f"router_{field}"] = np.where(
            high_volatility,
            result[f"reversal_{field}"],
            result[f"momentum_{field}"],
        )
        result.loc[~routable, f"router_{field}"] = np.nan
        result[f"static_50_50_{field}"] = 0.5 * (
            result[f"momentum_{field}"] + result[f"reversal_{field}"]
        )
    momentum_vol = result["momentum_net_return"].rolling(63, min_periods=20).std().shift(1)
    reversal_vol = result["reversal_net_return"].rolling(63, min_periods=20).std().shift(1)
    denominator = 1.0 / momentum_vol.replace(0.0, np.nan) + 1.0 / reversal_vol.replace(0.0, np.nan)
    result["inverse_vol_momentum_weight"] = (1.0 / momentum_vol) / denominator
    result["inverse_vol_reversal_weight"] = (1.0 / reversal_vol) / denominator
    missing = result["inverse_vol_momentum_weight"].isna()
    result.loc[missing, ["inverse_vol_momentum_weight", "inverse_vol_reversal_weight"]] = 0.5
    result["lagged_inverse_vol_net_return"] = (
        result["inverse_vol_momentum_weight"] * result["momentum_net_return"]
        + result["inverse_vol_reversal_weight"] * result["reversal_net_return"]
    )
    result["momentum_horizon_sessions"] = int(momentum_horizon)
    return result


def metrics(values: pd.Series) -> dict[str, float | int]:
    returns = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(returns.mean()) if len(returns) else np.nan
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "days": int(len(returns)),
        "compounded_return": float(wealth.iloc[-1] - 1.0) if len(wealth) else np.nan,
        "annual_return_arithmetic": mean * 252.0,
        "annual_volatility": volatility * math.sqrt(252.0) if np.isfinite(volatility) else np.nan,
        "sharpe": mean / volatility * math.sqrt(252.0) if volatility > 0.0 else np.nan,
        "mean_t_stat": (
            mean / (volatility / math.sqrt(len(returns)))
            if volatility > 0.0 and len(returns) > 1
            else np.nan
        ),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    selected = frame.loc[frame["market_volatility_quartile"].notna()].copy()
    summary: dict[str, object] = {
        "start": str(selected["execution_day"].min().date()) if len(selected) else None,
        "end": str(selected["execution_day"].max().date()) if len(selected) else None,
        "sleeve_correlation": float(
            selected["momentum_net_return"].corr(selected["reversal_net_return"])
        ),
    }
    for sleeve in ["momentum", "reversal", "router", "static_50_50"]:
        summary[sleeve] = metrics(selected[f"{sleeve}_net_return"])
        summary[sleeve]["annual_gross_return_arithmetic"] = float(
            selected[f"{sleeve}_gross_return"].mean() * 252.0
        )
        summary[sleeve]["average_daily_turnover"] = float(selected[f"{sleeve}_turnover"].mean())
        summary[sleeve]["average_daily_cost_bps"] = float(selected[f"{sleeve}_cost"].mean() * 10_000.0)
        summary[sleeve]["max_gross"] = float(selected[f"{sleeve}_gross_exposure"].max())
    summary["lagged_inverse_vol"] = metrics(selected["lagged_inverse_vol_net_return"])
    summary["router_paired_comparisons"] = {}
    for comparator in ["momentum", "reversal", "static_50_50"]:
        spread = selected["router_net_return"] - selected[f"{comparator}_net_return"]
        standard_error = spread.std(ddof=1) / math.sqrt(len(spread)) if len(spread) > 1 else np.nan
        summary["router_paired_comparisons"][comparator] = {
            "annual_excess_return": float(spread.mean() * 252.0),
            "mean_spread_bps": float(spread.mean() * 10_000.0),
            "t_stat": float(spread.mean() / standard_error) if standard_error > 0.0 else np.nan,
        }
    return summary


def quartile_rows(frame: pd.DataFrame, *, horizon: int, window: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for quartile in range(1, 5):
        sample = frame.loc[frame["market_volatility_quartile"].eq(float(quartile))]
        spread = sample["reversal_net_return"] - sample["momentum_net_return"]
        standard_error = spread.std(ddof=1) / math.sqrt(len(spread)) if len(spread) > 1 else np.nan
        rows.append(
            {
                "momentum_horizon_sessions": horizon,
                "window": window,
                "quartile": quartile,
                "days": int(len(sample)),
                "momentum_bps": float(sample["momentum_net_return"].mean() * 10_000.0),
                "reversal_bps": float(sample["reversal_net_return"].mean() * 10_000.0),
                "reversal_minus_momentum_bps": float(spread.mean() * 10_000.0),
                "spread_t_stat": float(spread.mean() / standard_error) if standard_error > 0.0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    daily, regimes = trade_day_panel()
    all_daily: list[pd.DataFrame] = []
    all_quartiles: list[pd.DataFrame] = []
    horizon_summaries: dict[str, object] = {}
    for horizon in MOMENTUM_HORIZONS:
        result = build_horizon_result(daily, regimes, momentum_horizon=horizon)
        all_daily.append(result)
        windows = {
            "all_routable": result,
            "development": result.loc[result["execution_day"].lt(HOLDOUT_START)],
            "holdout": result.loc[result["execution_day"].ge(HOLDOUT_START)],
        }
        horizon_summaries[str(horizon)] = {
            name: summarize(frame) for name, frame in windows.items()
        }
        all_quartiles.extend(
            quartile_rows(frame, horizon=horizon, window=name) for name, frame in windows.items()
        )

    combined_daily = pd.concat(all_daily, ignore_index=True)
    combined_quartiles = pd.concat(all_quartiles, ignore_index=True)
    summary = {
        "specification": {
            "bar_source": "cached 15-minute adjusted-main panel",
            "trade_day": "night session mapped to the following business day; weekend fragments rolled to Monday",
            "momentum_horizons_sessions": list(MOMENTUM_HORIZONS),
            "momentum_skip_sessions": 1,
            "reversal_horizon_sessions": 1,
            "execution": "signal after trade day t; hold next trade-day open to close",
            "portfolio": "top/bottom quintile, +0.5/-0.5, 1.0 gross",
            "regime": "rolling 252-trade-day market-volatility quartiles; Q1-Q3 momentum, Q4 reversal",
            "selection": "all three horizons pre-declared; no return-based parameter search",
        },
        "horizons": horizon_summaries,
        "paper_success_by_horizon": {
            str(horizon): bool(
                horizon_summaries[str(horizon)]["holdout"]["router"]["sharpe"]
                > max(
                    horizon_summaries[str(horizon)]["holdout"]["momentum"]["sharpe"],
                    horizon_summaries[str(horizon)]["holdout"]["reversal"]["sharpe"],
                    horizon_summaries[str(horizon)]["holdout"]["static_50_50"]["sharpe"],
                )
            )
            for horizon in MOMENTUM_HORIZONS
        },
        "resource_note": "Uses only the existing 15-minute cache; the 16-million-row minute parquet is not scanned.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_daily.to_csv(DAILY_FILE, index=False)
    combined_quartiles.to_csv(QUARTILE_FILE, index=False)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(combined_quartiles.loc[combined_quartiles["window"].eq("holdout")].to_string(index=False))
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"saved daily: {DAILY_FILE}")
    print(f"saved quartiles: {QUARTILE_FILE}")
    print(f"saved summary: {SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
