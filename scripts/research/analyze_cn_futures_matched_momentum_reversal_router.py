#!/usr/bin/env python3
"""Test a paper-matched long-short momentum/reversal volatility router."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.data.instruments import InstrumentMaster  # noqa: E402


DATA_FILE = REPO_ROOT / (
    "runtime/data/futures_cn/intraday/futures_data_futures_main_1m_adj_20260710_090753.parquet"
)
OUTPUT_DIR = REPO_ROOT / "runtime/artifacts/research/matched_momentum_reversal_router"
BAR_FILE = OUTPUT_DIR / "cn_futures_adjusted_main_15m_bars.parquet"
DAILY_FILE = OUTPUT_DIR / "cn_futures_matched_momentum_reversal_daily.csv"
QUARTILE_FILE = OUTPUT_DIR / "cn_futures_matched_momentum_reversal_quartiles.csv"
SUMMARY_FILE = OUTPUT_DIR / "cn_futures_matched_momentum_reversal_summary.json"

LIQUID_CONTRACTS = 40
LIQUIDITY_LOOKBACK_DAYS = 20
VOLATILITY_LOOKBACK_BARS = 64
MOMENTUM_FORMATION_BARS = 8
MOMENTUM_SKIP_BARS = 1
HOLD_BARS = 4
TAIL_FRACTION = 0.20
MIN_CROSS_SECTION = 20
REGIME_LOOKBACK_DAYS = 252
SLIPPAGE_TICKS_PER_SIDE = 0.5
SESSION_GAP_MINUTES = 180
HOLDOUT_START = pd.Timestamp("2025-10-09")


def build_bar_cache(*, rebuild: bool) -> None:
    if BAR_FILE.exists() and not rebuild:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = (
        pl.scan_parquet(DATA_FILE)
        .select(
            "symbol",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "month_change",
        )
        .filter(
            pl.col("datetime").is_not_null()
            & pl.col("symbol").is_not_null()
            & pl.col("close").is_finite()
            & pl.col("close").gt(0.0)
            & pl.col("volume").is_finite()
            & pl.col("volume").ge(0.0)
        )
        .with_columns(pl.col("datetime").dt.truncate("15m").alias("bar_time"))
        .group_by(["symbol", "bar_time"])
        .agg(
            pl.col("open").sort_by("datetime").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").sort_by("datetime").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("open_interest").sort_by("datetime").last().alias("open_interest"),
            pl.col("month_change").abs().sum().alias("roll_count"),
            pl.len().alias("source_minutes"),
        )
        .sort(["symbol", "bar_time"])
        .collect(engine="streaming")
    )
    bars.write_parquet(BAR_FILE, compression="zstd")


def instrument_maps() -> tuple[dict[str, float], dict[str, str], dict[str, dict[str, float | str]]]:
    master = InstrumentMaster("FUTURES_CN")
    multipliers = {str(key): float(value) for key, value in master.get_multiplier_map().items()}
    sectors = {str(key): str(value) for key, value in master.get_sector_map().items()}
    costs: dict[str, dict[str, float | str]] = {}
    for base in master.get_sector_map():
        profile = master.get_profile(base)
        costs[str(base)] = {
            "tick_size": float(profile.tick_size),
            "multiplier": float(profile.multiplier),
            "fee_type": str(profile.fee_type),
            "fee_open": float(profile.fee_open),
            "fee_close": float(profile.fee_close_today),
        }
    return multipliers, sectors, costs


def prepare_bars() -> pd.DataFrame:
    bars = pl.read_parquet(BAR_FILE).to_pandas()
    bars = bars.rename(columns={"symbol": "ticker", "bar_time": "date"})
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
    bars["ticker"] = bars["ticker"].astype("category")
    bars["base"] = bars["ticker"].astype(str).str.extract(r"\.([A-Za-z]+)$", expand=False)
    multipliers, sectors, costs = instrument_maps()
    bars["multiplier"] = bars["base"].map(multipliers).fillna(1.0).astype("float32")
    bars["sector"] = bars["base"].map(sectors).fillna("unknown").astype("category")
    bars = bars.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_group = bars.groupby("ticker", sort=False, observed=True)
    gap = ticker_group["date"].diff().gt(pd.Timedelta(minutes=SESSION_GAP_MINUTES)).fillna(True)
    bars["session_id"] = gap.groupby(bars["ticker"], observed=True).cumsum().astype("int32")
    session_group = bars.groupby(["ticker", "session_id"], sort=False, observed=True)
    bars["session_bar"] = session_group.cumcount().astype("int16")
    bars["bars_remaining"] = session_group.cumcount(ascending=False).astype("int16")
    previous_close = session_group["close"].shift(1)
    next_close = session_group["close"].shift(-1)
    bars["bar_log_return"] = np.log(bars["close"] / previous_close)
    bars["forward_return"] = next_close / bars["close"] - 1.0
    bars.loc[bars["forward_return"].abs().gt(0.10), "forward_return"] = np.nan

    prior_close = ticker_group["close"].shift(MOMENTUM_SKIP_BARS)
    formation_start = ticker_group["close"].shift(
        MOMENTUM_FORMATION_BARS + MOMENTUM_SKIP_BARS
    )
    bars["momentum_formation_return"] = np.log(prior_close / formation_start)
    bars["momentum_formation_roll_count"] = ticker_group["roll_count"].transform(
        lambda values: values.rolling(
            MOMENTUM_FORMATION_BARS + MOMENTUM_SKIP_BARS,
            min_periods=MOMENTUM_FORMATION_BARS + MOMENTUM_SKIP_BARS,
        ).sum()
    )
    bars["lagged_bar_sigma"] = ticker_group["bar_log_return"].transform(
        lambda values: values.rolling(
            VOLATILITY_LOOKBACK_BARS,
            min_periods=VOLATILITY_LOOKBACK_BARS // 2,
        ).std().shift(1)
    )
    bars["calendar_day"] = bars["date"].dt.normalize()
    bars["bar_notional"] = bars["close"] * bars["volume"] * bars["multiplier"]

    daily_liquidity = (
        bars.groupby(["ticker", "calendar_day"], sort=False, observed=True)["bar_notional"]
        .sum()
        .rename("daily_notional")
        .reset_index()
        .sort_values(["ticker", "calendar_day"])
    )
    daily_liquidity["lagged_median_notional"] = daily_liquidity.groupby(
        "ticker", sort=False, observed=True
    )["daily_notional"].transform(
        lambda values: values.rolling(
            LIQUIDITY_LOOKBACK_DAYS,
            min_periods=LIQUIDITY_LOOKBACK_DAYS // 2,
        ).median().shift(1)
    )
    daily_liquidity["liquidity_rank"] = daily_liquidity.groupby(
        "calendar_day", sort=False, observed=True
    )["lagged_median_notional"].rank(method="first", ascending=False)
    bars = bars.merge(
        daily_liquidity[["ticker", "calendar_day", "lagged_median_notional", "liquidity_rank"]],
        on=["ticker", "calendar_day"],
        how="left",
        sort=False,
    )

    price = pd.to_numeric(bars["close"], errors="coerce")
    tick = bars["base"].map({key: float(value["tick_size"]) for key, value in costs.items()})
    multiplier = bars["base"].map({key: float(value["multiplier"]) for key, value in costs.items()})
    fee_type = bars["base"].map({key: str(value["fee_type"]) for key, value in costs.items()})
    mean_fee = bars["base"].map(
        {
            key: 0.5 * (float(value["fee_open"]) + float(value["fee_close"]))
            for key, value in costs.items()
        }
    )
    fixed_fee_rate = mean_fee / (price * multiplier).replace(0.0, np.nan)
    fee_rate = pd.Series(np.where(fee_type.eq("fixed"), fixed_fee_rate, mean_fee), index=bars.index)
    bars["one_way_cost_rate"] = (
        SLIPPAGE_TICKS_PER_SIDE * tick / price.replace(0.0, np.nan) + fee_rate
    ).fillna(0.0)
    return bars


def decision_weights(bars: pd.DataFrame, *, hold_bars: int) -> pd.DataFrame:
    sigma = bars["lagged_bar_sigma"].replace(0.0, np.nan)
    bars["momentum_score"] = bars["momentum_formation_return"] / (
        sigma * math.sqrt(MOMENTUM_FORMATION_BARS)
    )
    bars["reversal_score"] = -bars["bar_log_return"] / sigma
    intraday_bucket = (bars["date"].dt.hour * 60 + bars["date"].dt.minute) // 15
    decision = intraday_bucket.mod(int(hold_bars)).eq(int(hold_bars) - 1)
    common = (
        decision
        & bars["bars_remaining"].ge(int(hold_bars))
        & bars["liquidity_rank"].le(LIQUID_CONTRACTS)
        & bars["momentum_score"].replace([np.inf, -np.inf], np.nan).notna()
        & bars["reversal_score"].replace([np.inf, -np.inf], np.nan).notna()
        & bars["forward_return"].notna()
        & bars["momentum_formation_roll_count"].eq(0.0)
    )
    cross_section_count = common.groupby(bars["date"], sort=False).transform("sum")
    common &= cross_section_count.ge(MIN_CROSS_SECTION)

    for sleeve, score_column in [
        ("momentum", "momentum_score"),
        ("reversal", "reversal_score"),
    ]:
        residual = bars.loc[common, score_column] - bars.loc[common].groupby(
            "date", sort=False
        )[score_column].transform("mean")
        percentile = residual.groupby(bars.loc[common, "date"], sort=False).rank(pct=True)
        long = percentile.ge(1.0 - TAIL_FRACTION)
        short = percentile.le(TAIL_FRACTION)
        long_count = long.groupby(bars.loc[common, "date"], sort=False).transform("sum")
        short_count = short.groupby(bars.loc[common, "date"], sort=False).transform("sum")
        selected_weight = pd.Series(0.0, index=percentile.index, dtype="float32")
        selected_weight.loc[long] = (0.5 / long_count.loc[long]).astype("float32")
        selected_weight.loc[short] = (-0.5 / short_count.loc[short]).astype("float32")

        decision_column = f"{sleeve}_decision_weight"
        weight_column = f"{sleeve}_weight"
        bars[decision_column] = np.nan
        bars.loc[decision, decision_column] = 0.0
        bars.loc[selected_weight.index, decision_column] = selected_weight
        bars[weight_column] = bars.groupby(
            ["ticker", "session_id"], sort=False, observed=True
        )[decision_column].ffill(limit=int(hold_bars) - 1).fillna(0.0).astype("float32")
        bars.loc[bars["forward_return"].isna(), weight_column] = 0.0
    return bars


def portfolio_bars(bars: pd.DataFrame, sleeve: str) -> pd.DataFrame:
    weight = pd.to_numeric(bars[f"{sleeve}_weight"], errors="coerce").fillna(0.0)
    prior_weight = weight.groupby(bars["ticker"], sort=False, observed=True).shift(1).fillna(0.0)
    delta = weight - prior_weight
    contribution = weight * bars["forward_return"].fillna(0.0)
    cost = delta.abs() * bars["one_way_cost_rate"]
    row = pd.DataFrame(
        {
            "date": bars["date"],
            "gross_contribution": contribution,
            "cost": cost,
            "turnover": delta.abs(),
            "gross_exposure": weight.abs(),
        }
    )
    grouped = row.groupby("date", sort=True).agg(
        gross_return=("gross_contribution", "sum"),
        cost=("cost", "sum"),
        turnover=("turnover", "sum"),
        gross_exposure=("gross_exposure", "sum"),
    )
    grouped["net_return"] = grouped["gross_return"] - grouped["cost"]
    return grouped


def compound(values: pd.Series) -> float:
    return float((1.0 + pd.to_numeric(values, errors="coerce").fillna(0.0)).prod() - 1.0)


def daily_portfolios(bars: pd.DataFrame) -> pd.DataFrame:
    momentum = portfolio_bars(bars, "momentum").add_prefix("momentum_")
    reversal = portfolio_bars(bars, "reversal").add_prefix("reversal_")
    intraday = momentum.join(reversal, how="outer").fillna(0.0).reset_index()
    intraday["calendar_day"] = intraday["date"].dt.normalize()
    daily = intraday.groupby("calendar_day", sort=True).agg(
        momentum_gross_return=("momentum_gross_return", compound),
        momentum_net_return=("momentum_net_return", compound),
        momentum_cost=("momentum_cost", "sum"),
        momentum_turnover=("momentum_turnover", "sum"),
        momentum_max_gross=("momentum_gross_exposure", "max"),
        reversal_gross_return=("reversal_gross_return", compound),
        reversal_net_return=("reversal_net_return", compound),
        reversal_cost=("reversal_cost", "sum"),
        reversal_turnover=("reversal_turnover", "sum"),
        reversal_max_gross=("reversal_gross_exposure", "max"),
    ).reset_index()
    return daily


def volatility_regimes(bars: pd.DataFrame, calendar_days: pd.Series) -> pd.DataFrame:
    liquid = (
        bars["liquidity_rank"].le(LIQUID_CONTRACTS)
        & bars["bar_log_return"].notna()
        & bars["roll_count"].eq(0.0)
    )
    market = bars.loc[liquid, ["date", "bar_log_return"]].groupby("date", sort=True).agg(
        market_return=("bar_log_return", "mean"),
        contracts=("bar_log_return", "size"),
    )
    market = market.loc[market["contracts"].ge(MIN_CROSS_SECTION)].copy()
    market["calendar_day"] = market.index.normalize()
    daily_volatility = market.assign(square=market["market_return"].pow(2)).groupby(
        "calendar_day", sort=True
    )["square"].sum().pow(0.5)
    rolling = daily_volatility.rolling(REGIME_LOOKBACK_DAYS, min_periods=REGIME_LOOKBACK_DAYS)
    q25 = rolling.quantile(0.25)
    q50 = rolling.quantile(0.50)
    q75 = rolling.quantile(0.75)
    quartile = pd.Series(np.nan, index=daily_volatility.index)
    ready = q75.notna()
    quartile.loc[ready & daily_volatility.le(q25)] = 1.0
    quartile.loc[ready & daily_volatility.gt(q25) & daily_volatility.le(q50)] = 2.0
    quartile.loc[ready & daily_volatility.gt(q50) & daily_volatility.le(q75)] = 3.0
    quartile.loc[ready & daily_volatility.gt(q75)] = 4.0
    state = pd.DataFrame(
        {
            "volatility_observation_day": daily_volatility.index,
            "market_realized_volatility": daily_volatility.to_numpy(),
            "market_volatility_q25": q25.to_numpy(),
            "market_volatility_q50": q50.to_numpy(),
            "market_volatility_q75": q75.to_numpy(),
            "market_volatility_quartile": quartile.to_numpy(),
        }
    )
    state["calendar_day"] = state["volatility_observation_day"].shift(-1)
    state = state.dropna(subset=["calendar_day"]).set_index("calendar_day")
    all_days = pd.Index(pd.to_datetime(calendar_days).dropna().unique()).sort_values()
    return state.reindex(all_days).ffill(limit=2).rename_axis("calendar_day").reset_index()


def add_allocations(daily: pd.DataFrame) -> pd.DataFrame:
    high_volatility = daily["market_volatility_quartile"].eq(4.0)
    routable = daily["market_volatility_quartile"].between(1.0, 4.0, inclusive="both")
    for field in ["gross_return", "net_return", "cost", "turnover", "max_gross"]:
        daily[f"router_{field}"] = np.where(
            high_volatility,
            daily[f"reversal_{field}"],
            daily[f"momentum_{field}"],
        )
        daily.loc[~routable, f"router_{field}"] = np.nan
        daily[f"static_50_50_{field}"] = 0.5 * (
            daily[f"momentum_{field}"] + daily[f"reversal_{field}"]
        )

    momentum_vol = daily["momentum_net_return"].rolling(63, min_periods=20).std().shift(1)
    reversal_vol = daily["reversal_net_return"].rolling(63, min_periods=20).std().shift(1)
    total_inverse = 1.0 / momentum_vol.replace(0.0, np.nan) + 1.0 / reversal_vol.replace(0.0, np.nan)
    daily["inverse_vol_momentum_weight"] = (1.0 / momentum_vol) / total_inverse
    daily["inverse_vol_reversal_weight"] = (1.0 / reversal_vol) / total_inverse
    missing = daily["inverse_vol_momentum_weight"].isna()
    daily.loc[missing, ["inverse_vol_momentum_weight", "inverse_vol_reversal_weight"]] = 0.5
    daily["lagged_inverse_vol_net_return"] = (
        daily["inverse_vol_momentum_weight"] * daily["momentum_net_return"]
        + daily["inverse_vol_reversal_weight"] * daily["reversal_net_return"]
    )
    return daily


def metrics(returns: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else np.nan
    volatility = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "days": int(len(values)),
        "compounded_return": float(wealth.iloc[-1] - 1.0) if len(wealth) else np.nan,
        "annual_return_arithmetic": mean * 252.0,
        "annual_volatility": volatility * math.sqrt(252.0) if np.isfinite(volatility) else np.nan,
        "sharpe": mean / volatility * math.sqrt(252.0) if volatility > 0.0 else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def quartile_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for quartile in range(1, 5):
        sample = frame.loc[frame["market_volatility_quartile"].eq(float(quartile))]
        spread = sample["reversal_net_return"] - sample["momentum_net_return"]
        standard_error = spread.std(ddof=1) / math.sqrt(len(spread)) if len(spread) > 1 else np.nan
        rows.append(
            {
                "window": label,
                "quartile": quartile,
                "days": int(len(sample)),
                "momentum_bps": float(sample["momentum_net_return"].mean() * 10_000.0),
                "reversal_bps": float(sample["reversal_net_return"].mean() * 10_000.0),
                "reversal_minus_momentum_bps": float(spread.mean() * 10_000.0),
                "spread_t_stat": float(spread.mean() / standard_error) if standard_error > 0.0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_window(frame: pd.DataFrame) -> dict[str, object]:
    selected = frame.loc[frame["market_volatility_quartile"].notna()].copy()
    summary: dict[str, object] = {
        "start": str(selected["calendar_day"].min().date()) if len(selected) else None,
        "end": str(selected["calendar_day"].max().date()) if len(selected) else None,
        "momentum": metrics(selected["momentum_net_return"]),
        "reversal": metrics(selected["reversal_net_return"]),
        "router": metrics(selected["router_net_return"]),
        "static_50_50": metrics(selected["static_50_50_net_return"]),
        "lagged_inverse_vol": metrics(selected["lagged_inverse_vol_net_return"]),
        "sleeve_correlation": float(
            selected["momentum_net_return"].corr(selected["reversal_net_return"])
        ),
    }
    for sleeve in ["momentum", "reversal", "router", "static_50_50"]:
        summary[sleeve]["annual_gross_return_arithmetic"] = float(
            selected[f"{sleeve}_gross_return"].mean() * 252.0
        )
        summary[sleeve]["average_daily_turnover"] = float(selected[f"{sleeve}_turnover"].mean())
        summary[sleeve]["average_daily_cost_bps"] = float(selected[f"{sleeve}_cost"].mean() * 10_000.0)
        summary[sleeve]["max_gross"] = float(selected[f"{sleeve}_max_gross"].max())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-bars", action="store_true")
    parser.add_argument("--hold-bars", type=int, choices=[4, 8], default=HOLD_BARS)
    args = parser.parse_args()
    build_bar_cache(rebuild=args.rebuild_bars)
    bars = decision_weights(prepare_bars(), hold_bars=args.hold_bars)
    daily = daily_portfolios(bars)
    regimes = volatility_regimes(bars, daily["calendar_day"])
    daily = daily.merge(regimes, on="calendar_day", how="left").sort_values("calendar_day")
    daily = add_allocations(daily)

    windows = {
        "all_routable": daily,
        "development": daily.loc[daily["calendar_day"].lt(HOLDOUT_START)],
        "holdout": daily.loc[daily["calendar_day"].ge(HOLDOUT_START)],
    }
    summaries = {name: summarize_window(frame) for name, frame in windows.items()}
    quartiles = pd.concat(
        [quartile_table(frame, name) for name, frame in windows.items()],
        ignore_index=True,
    )
    summary = {
        "specification": {
            "bar_minutes": 15,
            "liquid_contracts": LIQUID_CONTRACTS,
            "liquidity_lookback_days": LIQUIDITY_LOOKBACK_DAYS,
            "momentum_formation_minutes": MOMENTUM_FORMATION_BARS * 15,
            "momentum_skip_minutes": MOMENTUM_SKIP_BARS * 15,
            "reversal_formation_minutes": 15,
            "holding_minutes": args.hold_bars * 15,
            "decision_clock": (
                f"synchronized non-overlapping decisions every {args.hold_bars * 15} minutes"
            ),
            "session_gap_minutes": SESSION_GAP_MINUTES,
            "portfolio": "cross-sectional top/bottom quintile, 0.5 long plus 0.5 short",
            "regime": "rolling 252-day market-volatility quartiles; Q1-Q3 momentum, Q4 reversal next day",
            "selection": (
                "60-minute primary specification; 120-minute choice is a slower-clock robustness check"
            ),
        },
        "windows": summaries,
        "paper_hypothesis_supported": bool(
            quartiles.loc[
                (quartiles["window"].eq("holdout")) & quartiles["quartile"].eq(4),
                "reversal_minus_momentum_bps",
            ].iloc[0]
            > 0.0
            and summaries["holdout"]["router"]["sharpe"]
            > max(
                summaries["holdout"]["momentum"]["sharpe"],
                summaries["holdout"]["reversal"]["sharpe"],
                summaries["holdout"]["static_50_50"]["sharpe"],
            )
        ),
        "resource_note": "The full minute parquet is scanned only when the compact 15-minute cache is absent or --rebuild-bars is supplied.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.hold_bars == HOLD_BARS else f"_{args.hold_bars * 15}m_hold"
    daily_file = DAILY_FILE.with_name(f"{DAILY_FILE.stem}{suffix}{DAILY_FILE.suffix}")
    quartile_file = QUARTILE_FILE.with_name(f"{QUARTILE_FILE.stem}{suffix}{QUARTILE_FILE.suffix}")
    summary_file = SUMMARY_FILE.with_name(f"{SUMMARY_FILE.stem}{suffix}{SUMMARY_FILE.suffix}")
    daily.to_csv(daily_file, index=False)
    quartiles.to_csv(quartile_file, index=False)
    summary_file.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(quartiles.to_string(index=False))
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"saved bars: {BAR_FILE}")
    print(f"saved daily: {daily_file}")
    print(f"saved quartiles: {quartile_file}")
    print(f"saved summary: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
