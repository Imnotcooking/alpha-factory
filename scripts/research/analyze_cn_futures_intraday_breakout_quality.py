#!/usr/bin/env python3
"""Search cost-aware entry gates for the existing intraday breadth breakout."""

from __future__ import annotations

import importlib.util
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
FACTOR_FILE = REPO_ROOT / (
    "departments/research/factors/daily_signals/"
    "fac_090_Intraday_Breadth_Anchored_Breakout_Futures_CN.py"
)
EVENT_FILE = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/cn_futures_intraday_breakout_quality_events.parquet"
)
SEARCH_FILE = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/cn_futures_intraday_breakout_quality_search.csv"
)

VOLUME_MINIMUMS = (0.75, 1.0, 1.25, 1.5)
KER_MINIMUMS = (0.30, 0.45, 0.60, 0.75)
TREND_MINIMUMS = (0.35, 0.60, 0.90, 1.20)
BREADTH_MINIMUMS = (0.0, 0.05, 0.10, 0.20)
EDGE_COST_MINIMUMS = (0.0, 2.0, 4.0, 6.0)


def load_factor():
    spec = importlib.util.spec_from_file_location("fac090_quality_study", FACTOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FACTOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instrument_cost_frame() -> pd.DataFrame:
    master = InstrumentMaster("FUTURES_CN")
    rows: list[dict[str, object]] = []
    for base in master.get_sector_map():
        profile = master.get_profile(base)
        rows.append(
            {
                "base": str(base),
                "tick_size": float(profile.tick_size),
                "multiplier": float(profile.multiplier),
                "fee_type": str(profile.fee_type),
                "fee_open": float(profile.fee_open),
                "fee_close_today": float(profile.fee_close_today),
            }
        )
    return pd.DataFrame(rows)


def build_events() -> pd.DataFrame:
    raw = (
        pl.scan_parquet(DATA_FILE)
        .filter(
            (pl.col("datetime") >= pl.datetime(2025, 10, 1))
            & (pl.col("datetime") < pl.datetime(2026, 7, 10))
        )
        .collect(engine="streaming")
        .to_pandas()
    )
    factor = load_factor()
    prepared = factor.prepare_data(raw)
    result = factor.compute(prepared)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    state = pd.to_numeric(result["breadth_breakout_active_state"], errors="coerce").fillna(0.0)
    prior_state = state.groupby(result["ticker"], sort=False).shift(1).fillna(0.0)
    start = state.ne(0.0) & prior_state.eq(0.0)
    result["episode_id"] = start.groupby(result["ticker"], sort=False).cumsum()
    next_close = result.groupby("ticker", sort=False)["close"].shift(-1)
    result["signed_period_return"] = state * (next_close / result["close"] - 1.0)
    decision_features = [
        "breadth_breakout_atr",
        "breadth_breakout_trend_atr",
        "breadth_breakout_ker",
        "breadth_breakout_volume_ratio",
        "breadth_breakout_market_breadth",
        "breadth_breakout_breadth_aligned",
    ]
    for column in decision_features:
        result[f"entry_{column}"] = result.groupby("ticker", sort=False)[column].shift(1)
    active = result.loc[state.ne(0.0)].copy()
    payoff = (
        active.groupby(["ticker", "episode_id"], sort=False)["signed_period_return"]
        .sum()
        .rename("gross_trade_return")
        .reset_index()
    )
    entry_columns = [
        "ticker",
        "episode_id",
        "date",
        "close",
        *[f"entry_{column}" for column in decision_features],
        "breadth_breakout_exec_quality",
    ]
    entries = result.loc[start, entry_columns].merge(payoff, on=["ticker", "episode_id"], how="inner")
    entries = entries.rename(columns={f"entry_{column}": column for column in decision_features})
    entries["base"] = entries["ticker"].str.extract(r"\.([A-Za-z]+)$", expand=False)
    entries = entries.merge(instrument_cost_frame(), on="base", how="left")
    fixed_fee = (entries["fee_open"] + entries["fee_close_today"]) / (
        entries["close"] * entries["multiplier"]
    )
    ratio_fee = entries["fee_open"] + entries["fee_close_today"]
    fees = np.where(entries["fee_type"].eq("fixed"), fixed_fee, ratio_fee)
    entries["round_trip_cost"] = fees + entries["tick_size"] / entries["close"]
    entries["atr_pct"] = entries["breadth_breakout_atr"] / entries["close"]
    entries["edge_cost_ratio"] = entries["atr_pct"] / entries["round_trip_cost"].replace(0.0, np.nan)
    entries["net_trade_return"] = entries["gross_trade_return"] - entries["round_trip_cost"]
    entries["date"] = pd.to_datetime(entries["date"])
    return entries


def daily_stats(events: pd.DataFrame, start: str, end: str) -> tuple[int, int, float, float, float]:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    selected = events.loc[events["date"].ge(start_time) & events["date"].lt(end_time)].copy()
    selected["day"] = selected["date"].dt.normalize()
    daily = selected.groupby("day")["net_trade_return"].sum() * 0.10
    calendar = pd.bdate_range(start_time, end_time - pd.Timedelta(days=1))
    daily = daily.reindex(calendar, fill_value=0.0)
    std = float(daily.std(ddof=1))
    mean = float(daily.mean())
    return (
        int(len(selected)),
        int(len(daily)),
        mean * 252.0,
        mean / std * math.sqrt(252.0) if std > 0 else np.nan,
        float(selected["net_trade_return"].mean() * 10_000) if len(selected) else np.nan,
    )


def main() -> int:
    events = build_events()
    EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(events).write_parquet(EVENT_FILE, compression="zstd")
    rows: list[dict[str, object]] = []
    for volume_min, ker_min, trend_min, breadth_min, edge_cost_min in itertools.product(
        VOLUME_MINIMUMS,
        KER_MINIMUMS,
        TREND_MINIMUMS,
        BREADTH_MINIMUMS,
        EDGE_COST_MINIMUMS,
    ):
        mask = (
            events["breadth_breakout_volume_ratio"].ge(volume_min)
            & events["breadth_breakout_ker"].ge(ker_min)
            & events["breadth_breakout_trend_atr"].ge(trend_min)
            & events["breadth_breakout_market_breadth"].ge(breadth_min)
            & events["edge_cost_ratio"].ge(edge_cost_min)
        )
        sample = events.loc[mask].copy()
        train = daily_stats(sample, "2025-10-01", "2026-01-01")
        validation = daily_stats(sample, "2026-01-01", "2026-04-01")
        audit = daily_stats(sample, "2026-04-01", "2026-07-10")
        if train[0] < 25 or validation[0] < 25:
            continue
        rows.append(
            {
                "volume_min": volume_min,
                "ker_min": ker_min,
                "trend_min": trend_min,
                "breadth_min": breadth_min,
                "edge_cost_min": edge_cost_min,
                "train_events": train[0],
                "train_annual_return": train[2],
                "train_sharpe": train[3],
                "train_net_bps": train[4],
                "validation_events": validation[0],
                "validation_annual_return": validation[2],
                "validation_sharpe": validation[3],
                "validation_net_bps": validation[4],
                "audit_events": audit[0],
                "audit_annual_return": audit[2],
                "audit_sharpe": audit[3],
                "audit_net_bps": audit[4],
                "selection_min_sharpe": min(train[3], validation[3]),
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["selection_min_sharpe", "validation_sharpe"], ascending=False
    )
    output.to_csv(SEARCH_FILE, index=False)
    print(output.head(40).to_string(index=False))
    print(f"events: {len(events):,}; symbols: {events['ticker'].nunique()}")
    print(f"saved events: {EVENT_FILE}")
    print(f"saved search: {SEARCH_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
