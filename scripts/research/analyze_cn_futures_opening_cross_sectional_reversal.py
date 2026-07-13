#!/usr/bin/env python3
"""Test a liquid-universe opening-hour cross-sectional reversal in CN futures."""

from __future__ import annotations

import itertools
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
OUTPUT_FILE = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/"
    "cn_futures_opening_cross_sectional_reversal_search.csv"
)
PERIODS = (
    ("opening_0945_to_morning_close", "09:45", "09:45", "11:25"),
    ("opening_1000_to_morning_close", "10:00", "10:00", "11:25"),
    ("opening_1015_to_morning_close", "10:15", "10:15", "11:25"),
    ("morning_1125_to_afternoon", "11:25", "13:30", "14:55"),
    ("opening_1015_to_afternoon", "10:15", "13:30", "14:55"),
    ("morning_1125_to_late_close", "11:25", "14:00", "14:55"),
)
LIQUIDITY_MINIMUMS = (0.50, 0.70, 0.80)
TAIL_FRACTIONS = (0.10, 0.20, 0.30)
VOLUME_MAXIMUMS = (1.0, 1.5, math.inf)


def instrument_costs() -> pd.DataFrame:
    master = InstrumentMaster("FUTURES_CN")
    rows: list[dict[str, object]] = []
    for base in master.get_sector_map():
        profile = master.get_profile(base)
        rows.append(
            {
                "base": str(base),
                "sector": str(master.get_sector_map()[base]),
                "tick_size": float(profile.tick_size),
                "multiplier": float(profile.multiplier),
                "fee_type": str(profile.fee_type),
                "fee_open": float(profile.fee_open),
                "fee_close_today": float(profile.fee_close_today),
            }
        )
    return pd.DataFrame(rows)


def build_daily_panel(signal_time: str, entry_time: str, exit_time: str) -> pd.DataFrame:
    signal_hour, signal_minute = map(int, signal_time.split(":"))
    entry_hour, entry_minute = map(int, entry_time.split(":"))
    exit_hour, exit_minute = map(int, exit_time.split(":"))
    raw = (
        pl.scan_parquet(DATA_FILE)
        .select("symbol", "datetime", "close", "volume", "month_change")
        .with_columns(
            pl.col("datetime").dt.date().alias("day"),
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.minute().alias("minute"),
            pl.col("symbol").str.extract(r"\.([A-Za-z]+)$", 1).alias("base"),
        )
        .filter(
            (pl.col("hour") == 9)
            | (pl.col("hour") == 10)
            | (pl.col("hour") == 11)
            | (pl.col("hour") == 13)
            | (pl.col("hour") == 14)
        )
    )
    clock = pl.col("hour").cast(pl.Int32) * 60 + pl.col("minute").cast(pl.Int32)
    signal_clock = signal_hour * 60 + signal_minute
    entry_clock = entry_hour * 60 + entry_minute
    exit_clock = exit_hour * 60 + exit_minute
    daily = (
        raw.group_by(["symbol", "base", "day"])
        .agg(
            pl.col("close").filter(clock >= 540).first().alias("open_price"),
            pl.col("close").filter(clock <= signal_clock).last().alias("signal_price"),
            pl.col("close").filter(clock >= entry_clock).first().alias("entry_price"),
            pl.col("close").filter(clock <= exit_clock).last().alias("exit_price"),
            pl.col("volume").filter(clock <= signal_clock).sum().alias("opening_volume"),
            (pl.col("close") * pl.col("volume")).sum().alias("daily_price_volume"),
            pl.col("month_change").abs().sum().alias("roll_count"),
            pl.len().alias("rows"),
        )
        .filter(
            pl.col("open_price").is_not_null()
            & pl.col("signal_price").is_not_null()
            & pl.col("entry_price").is_not_null()
            & pl.col("exit_price").is_not_null()
            & pl.col("roll_count").eq(0)
        )
        .sort(["symbol", "day"])
        .collect(engine="streaming")
        .to_pandas()
    )
    costs = instrument_costs()
    daily = daily.merge(costs, on="base", how="left")
    daily["daily_notional"] = daily["daily_price_volume"] * daily["multiplier"].fillna(1.0)
    daily["lagged_notional"] = daily.groupby("symbol", sort=False)["daily_notional"].transform(
        lambda s: s.rolling(20, min_periods=10).median().shift(1)
    )
    daily["liquidity_rank"] = daily.groupby("day", sort=False)["lagged_notional"].rank(pct=True)
    daily["lagged_opening_volume"] = daily.groupby("symbol", sort=False)["opening_volume"].transform(
        lambda s: s.rolling(20, min_periods=10).median().shift(1)
    )
    daily["volume_ratio"] = daily["opening_volume"] / daily["lagged_opening_volume"].replace(0.0, np.nan)
    daily["opening_return"] = np.log(daily["signal_price"] / daily["open_price"])
    daily["holding_return"] = np.log(daily["exit_price"] / daily["entry_price"])
    daily["sector_median_return"] = daily.groupby(["day", "sector"], sort=False)["opening_return"].transform(
        "median"
    )
    daily["residual_opening_return"] = daily["opening_return"] - daily["sector_median_return"]
    daily["residual_rank"] = daily.groupby("day", sort=False)["residual_opening_return"].rank(pct=True)
    fixed_fee = (daily["fee_open"] + daily["fee_close_today"]) / (
        daily["entry_price"] * daily["multiplier"]
    )
    ratio_fee = daily["fee_open"] + daily["fee_close_today"]
    fees = np.where(daily["fee_type"].eq("fixed"), fixed_fee, ratio_fee)
    daily["round_trip_cost"] = fees + 1.25 * daily["tick_size"] / daily["entry_price"]
    daily["day"] = pd.to_datetime(daily["day"])
    return daily


def stats(values: pd.Series) -> tuple[int, float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2 or values.std(ddof=1) <= 0:
        return len(values), np.nan, np.nan, np.nan
    return (
        len(values),
        float(values.mean() * 252),
        float(values.mean() / values.std(ddof=1) * math.sqrt(252)),
        float(values.sum()),
    )


def main() -> int:
    train_end = pd.Timestamp("2025-01-01")
    validation_end = pd.Timestamp("2025-10-01")
    rows: list[dict[str, object]] = []
    for period_name, signal_time, entry_time, exit_time in PERIODS:
        panel = build_daily_panel(signal_time, entry_time, exit_time)
        for liquidity_min, tail, volume_max in itertools.product(
            LIQUIDITY_MINIMUMS, TAIL_FRACTIONS, VOLUME_MAXIMUMS
        ):
            eligible = panel["liquidity_rank"].ge(liquidity_min) & panel["volume_ratio"].le(volume_max)
            signal = pd.Series(0.0, index=panel.index)
            signal.loc[eligible & panel["residual_rank"].le(tail)] = 1.0
            signal.loc[eligible & panel["residual_rank"].ge(1.0 - tail)] = -1.0
            selected = panel.loc[signal.ne(0.0)].copy()
            selected["direction"] = signal.loc[selected.index]
            selected["net_asset_return"] = (
                selected["direction"] * selected["holding_return"] - selected["round_trip_cost"]
            )
            counts = selected.groupby("day")["symbol"].transform("size")
            selected["portfolio_contribution"] = selected["net_asset_return"] / counts
            daily_return = selected.groupby("day")["portfolio_contribution"].sum()
            full_calendar = pd.bdate_range(panel["day"].min(), panel["day"].max())
            daily_return = daily_return.reindex(full_calendar, fill_value=0.0)
            split = {
                "train": daily_return[daily_return.index < train_end],
                "validation": daily_return[
                    (daily_return.index >= train_end) & (daily_return.index < validation_end)
                ],
                "audit": daily_return[daily_return.index >= validation_end],
            }
            row: dict[str, object] = {
                "period": period_name,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "liquidity_min": liquidity_min,
                "tail_fraction": tail,
                "volume_max": volume_max,
                "selected_assets": int(len(selected)),
                "distinct_symbols": int(selected["symbol"].nunique()),
            }
            for name, returns in split.items():
                count, annual, sharpe, total = stats(returns)
                row.update(
                    {
                        f"{name}_days": count,
                        f"{name}_annual_return": annual,
                        f"{name}_sharpe": sharpe,
                        f"{name}_total_return": total,
                    }
                )
            row["selection_min_sharpe"] = min(row["train_sharpe"], row["validation_sharpe"])
            rows.append(row)
    output = pd.DataFrame(rows).sort_values("selection_min_sharpe", ascending=False)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(output.head(30).to_string(index=False))
    print(f"saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
