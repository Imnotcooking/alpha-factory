#!/usr/bin/env python3
"""Test night-return to day-return reversal across liquid CN futures."""

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
    "runtime/artifacts/research/minute_reversion/cn_futures_night_day_reversal_search.csv"
)
Z_MINIMUMS = (0.0, 0.5, 1.0, 1.5)
LIQUIDITY_MINIMUMS = (0.50, 0.70, 0.80)
NIGHT_VOLUME_MINIMUMS = (0.0, 0.8, 1.0, 1.2)
RANGE_RATIO_MINIMUMS = (0.0, 0.8, 1.0)
EDGE_GATES = ((0, -math.inf), (40, 0.0), (80, 0.0), (120, 0.0), (80, 0.0002))
MODES = ("night_to_day", "overnight_open_to_last_half_hour")


def instrument_costs() -> pd.DataFrame:
    master = InstrumentMaster("FUTURES_CN")
    sectors = master.get_sector_map()
    rows: list[dict[str, object]] = []
    for base, sector in sectors.items():
        profile = master.get_profile(base)
        rows.append(
            {
                "base": str(base),
                "sector": str(sector),
                "tick_size": float(profile.tick_size),
                "multiplier": float(profile.multiplier),
                "fee_type": str(profile.fee_type),
                "fee_open": float(profile.fee_open),
                "fee_close_today": float(profile.fee_close_today),
            }
        )
    return pd.DataFrame(rows)


def build_panel() -> pd.DataFrame:
    raw = (
        pl.scan_parquet(DATA_FILE)
        .select("symbol", "datetime", "close", "high", "low", "volume", "month_change")
        .with_columns(
            pl.col("datetime").dt.date().alias("calendar_day"),
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.minute().alias("minute"),
            pl.col("symbol").str.extract(r"\.([A-Za-z]+)$", 1).alias("base"),
        )
    )
    clock = pl.col("hour").cast(pl.Int32) * 60 + pl.col("minute").cast(pl.Int32)
    day = (
        raw.filter((pl.col("hour") >= 9) & (pl.col("hour") < 16))
        .group_by(["symbol", "base", "calendar_day"])
        .agg(
            pl.col("close").first().alias("day_open"),
            pl.col("close").filter(clock <= 570).last().alias("open_half_hour_close"),
            pl.col("close").filter(clock >= 870).first().alias("last_half_hour_open"),
            pl.col("close").last().alias("day_close"),
            pl.col("volume").sum().alias("day_volume"),
            (pl.col("close") * pl.col("volume")).sum().alias("day_price_volume"),
            pl.col("month_change").abs().sum().alias("day_roll_count"),
            pl.len().alias("day_rows"),
        )
        .sort(["symbol", "calendar_day"])
        .with_columns(
            pl.col("day_close").shift(1).over("symbol").alias("previous_day_close")
        )
        .rename({"calendar_day": "trade_day"})
    )
    night = (
        raw.filter((pl.col("hour") >= 21) | (pl.col("hour") < 3))
        .with_columns(
            pl.when(pl.col("hour") >= 21)
            .then(pl.col("calendar_day") + pl.duration(days=1))
            .otherwise(pl.col("calendar_day"))
            .alias("trade_day")
        )
        .group_by(["symbol", "base", "trade_day"])
        .agg(
            pl.col("close").last().alias("night_close"),
            pl.col("high").max().alias("night_high"),
            pl.col("low").min().alias("night_low"),
            pl.col("volume").sum().alias("night_volume"),
            pl.col("month_change").abs().sum().alias("night_roll_count"),
            pl.len().alias("night_rows"),
        )
    )
    panel = day.join(night, on=["symbol", "base", "trade_day"], how="inner").sort(
        ["symbol", "trade_day"]
    )
    frame = panel.collect(engine="streaming").to_pandas().merge(instrument_costs(), on="base", how="left")
    frame["trade_day"] = pd.to_datetime(frame["trade_day"])
    frame["daily_notional"] = frame["day_price_volume"] * frame["multiplier"]
    frame["lagged_notional"] = frame.groupby("symbol", sort=False)["daily_notional"].transform(
        lambda s: s.rolling(20, min_periods=10).median().shift(1)
    )
    frame["liquidity_rank"] = frame.groupby("trade_day", sort=False)["lagged_notional"].rank(pct=True)
    frame["night_return"] = np.log(frame["night_close"] / frame["previous_day_close"])
    frame["day_return"] = np.log(frame["day_close"] / frame["day_open"])
    frame["overnight_open_return"] = np.log(frame["open_half_hour_close"] / frame["previous_day_close"])
    frame["last_half_hour_return"] = np.log(frame["day_close"] / frame["last_half_hour_open"])
    frame["night_sigma"] = frame.groupby("symbol", sort=False)["night_return"].transform(
        lambda s: s.rolling(40, min_periods=20).std().shift(1)
    )
    frame["night_z"] = frame["night_return"] / frame["night_sigma"].replace(0.0, np.nan)
    frame["overnight_open_sigma"] = frame.groupby("symbol", sort=False)["overnight_open_return"].transform(
        lambda s: s.rolling(40, min_periods=20).std().shift(1)
    )
    frame["overnight_open_z"] = frame["overnight_open_return"] / frame[
        "overnight_open_sigma"
    ].replace(0.0, np.nan)
    frame["night_volume_median"] = frame.groupby("symbol", sort=False)["night_volume"].transform(
        lambda s: s.rolling(20, min_periods=10).median().shift(1)
    )
    frame["night_volume_ratio"] = frame["night_volume"] / frame["night_volume_median"].replace(0.0, np.nan)
    frame["night_range_pct"] = (frame["night_high"] - frame["night_low"]) / frame["night_close"]
    frame["night_range_median"] = frame.groupby("symbol", sort=False)["night_range_pct"].transform(
        lambda s: s.rolling(20, min_periods=10).median().shift(1)
    )
    frame["night_range_ratio"] = frame["night_range_pct"] / frame["night_range_median"].replace(0.0, np.nan)
    fixed_fee = (frame["fee_open"] + frame["fee_close_today"]) / (
        frame["day_open"] * frame["multiplier"]
    )
    ratio_fee = frame["fee_open"] + frame["fee_close_today"]
    fees = np.where(frame["fee_type"].eq("fixed"), fixed_fee, ratio_fee)
    frame["round_trip_cost"] = fees + 1.25 * frame["tick_size"] / frame["day_open"]
    fixed_last_fee = (frame["fee_open"] + frame["fee_close_today"]) / (
        frame["last_half_hour_open"] * frame["multiplier"]
    )
    last_fees = np.where(frame["fee_type"].eq("fixed"), fixed_last_fee, ratio_fee)
    frame["last_half_round_trip_cost"] = (
        last_fees + 1.25 * frame["tick_size"] / frame["last_half_hour_open"]
    )
    frame["raw_reversal_net_night_to_day"] = -np.sign(frame["night_return"]) * frame["day_return"] - frame[
        "round_trip_cost"
    ]
    frame["raw_reversal_net_overnight_open_to_last_half_hour"] = (
        -np.sign(frame["overnight_open_return"]) * frame["last_half_hour_return"]
        - frame["last_half_round_trip_cost"]
    )
    for window in {window for window, _ in EDGE_GATES if window > 0}:
        for mode in MODES:
            frame[f"lagged_edge_{mode}_{window}"] = frame.groupby("symbol", sort=False)[
                f"raw_reversal_net_{mode}"
            ].transform(lambda s, w=window: s.rolling(w, min_periods=max(20, w // 2)).mean().shift(1))
    return frame.loc[
        frame["day_roll_count"].eq(0)
        & frame["night_roll_count"].eq(0)
        & frame["previous_day_close"].gt(0)
        & frame["day_open"].gt(0)
        & frame["day_close"].gt(0)
    ].copy()


def stats(values: pd.Series) -> tuple[int, float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    mean = float(values.mean()) if len(values) else np.nan
    return (
        len(values),
        mean * 252,
        mean / std * math.sqrt(252) if std > 0 else np.nan,
        float(values.sum()),
    )


def main() -> int:
    panel = build_panel()
    train_end = pd.Timestamp("2025-01-01")
    validation_end = pd.Timestamp("2025-10-01")
    full_calendar = pd.bdate_range(panel["trade_day"].min(), panel["trade_day"].max())
    rows: list[dict[str, object]] = []
    for mode, z_min, liquidity_min, volume_min, range_min, (edge_window, edge_min) in itertools.product(
        MODES, Z_MINIMUMS, LIQUIDITY_MINIMUMS, NIGHT_VOLUME_MINIMUMS, RANGE_RATIO_MINIMUMS, EDGE_GATES
    ):
        predictor = "night_return" if mode == "night_to_day" else "overnight_open_return"
        predictor_z = "night_z" if mode == "night_to_day" else "overnight_open_z"
        outcome = "day_return" if mode == "night_to_day" else "last_half_hour_return"
        cost = "round_trip_cost" if mode == "night_to_day" else "last_half_round_trip_cost"
        eligible = (
            panel[predictor_z].abs().ge(z_min)
            & panel["liquidity_rank"].ge(liquidity_min)
            & panel["night_volume_ratio"].ge(volume_min)
            & panel["night_range_ratio"].ge(range_min)
        )
        if edge_window > 0:
            eligible &= panel[f"lagged_edge_{mode}_{edge_window}"].ge(edge_min)
        selected = panel.loc[eligible].copy()
        selected["direction"] = -np.sign(selected[predictor])
        selected["net_asset_return"] = (
            selected["direction"] * selected[outcome] - selected[cost]
        )
        counts = selected.groupby("trade_day")["symbol"].transform("size")
        selected["portfolio_contribution"] = selected["net_asset_return"] / counts
        daily = selected.groupby("trade_day")["portfolio_contribution"].sum().reindex(
            full_calendar, fill_value=0.0
        )
        split = {
            "train": daily[daily.index < train_end],
            "validation": daily[(daily.index >= train_end) & (daily.index < validation_end)],
            "audit": daily[daily.index >= validation_end],
        }
        row: dict[str, object] = {
            "mode": mode,
            "z_min": z_min,
            "liquidity_min": liquidity_min,
            "night_volume_min": volume_min,
            "night_range_min": range_min,
            "edge_window": edge_window,
            "edge_min_bps": edge_min * 10_000 if np.isfinite(edge_min) else -math.inf,
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
    print(output.head(40).to_string(index=False))
    print(f"panel rows: {len(panel):,}; symbols: {panel['symbol'].nunique()}")
    print(f"saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
