#!/usr/bin/env python3
"""Reconcile fac092 minute states with its coarse event-study timestamps."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / (
    "runtime/data/futures_cn/intraday/futures_data_futures_main_1m_adj_20260710_090753.parquet"
)
EVENT_FILE = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/"
    "cn_futures_indicator_reentry_full_2024-01-01_2026-07-09_events.parquet"
)
FACTOR_FILE = REPO_ROOT / (
    "departments/research/factors/daily_signals/"
    "fac_092_Intraday_Liquid_Universe_Indicator_Reentry_Futures_CN.py"
)
OUTPUT_FILE = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/fac092_event_alignment.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-start", default="2025-08-01")
    parser.add_argument("--audit-start", default="2025-10-09")
    parser.add_argument("--end", default="2026-07-09")
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def load_factor():
    spec = importlib.util.spec_from_file_location("fac092_alignment", FACTOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FACTOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    end_exclusive = pd.Timestamp(args.end) + pd.Timedelta(days=1)
    raw = (
        pl.scan_parquet(DATA_FILE)
        .filter(
            (pl.col("datetime") >= pl.lit(args.load_start).str.to_datetime("%Y-%m-%d"))
            & (pl.col("datetime") < pl.lit(str(end_exclusive.date())).str.to_datetime("%Y-%m-%d"))
        )
        .collect(engine="streaming")
        .to_pandas()
    )
    factor = load_factor()
    prepared = factor.prepare_data(raw)
    result = factor.compute(prepared)
    result["previous_signal"] = result.groupby("ticker", sort=False)["signal"].shift(1).fillna(0.0)
    entries = result.loc[
        result["signal"].ne(0.0) & result["previous_signal"].eq(0.0),
        [
            "ticker",
            "date",
            "close",
            "signal",
            "bb_z",
            "rsi",
            "kdj_j",
            "volume_ratio",
            "ker",
            "liquidity_rank",
            "remaining_session_buckets",
        ],
    ].copy()
    entries = entries.loc[pd.to_datetime(entries["date"]).ge(args.audit_start)]
    entries["event_datetime"] = pd.to_datetime(entries["date"]) + pd.Timedelta(minutes=1)

    events = pl.read_parquet(EVENT_FILE).to_pandas()
    events["datetime"] = pd.to_datetime(events["datetime"])
    horizon = int(result.attrs["factor_params"]["hold_buckets"])
    events = events.loc[
        events["datetime"].ge(args.audit_start)
        & events["entry_abs_z"].ge(result.attrs["factor_params"]["entry_z_band"][0])
        & events["entry_abs_z"].lt(result.attrs["factor_params"]["entry_z_band"][1])
        & events["liquidity_rank"].ge(result.attrs["factor_params"]["liquidity_rank_min"])
        & events["volume_ratio"].le(result.attrs["factor_params"]["volume_ratio_max"])
        & events["confirmation_count"].le(result.attrs["factor_params"]["confirmation_max"])
        & events[f"net_reversal_{horizon}"].notna()
    ].copy()
    event_columns = [
        "symbol",
        "datetime",
        "entry_abs_z",
        "liquidity_rank",
        "volume_ratio",
        "confirmation_count",
        f"gross_reversal_{horizon}",
        f"net_reversal_{horizon}",
        f"forward_datetime_{horizon}",
    ]
    events = events[event_columns]
    aligned = entries.merge(
        events,
        left_on=["ticker", "event_datetime"],
        right_on=["symbol", "datetime"],
        how="left",
        indicator=True,
    )
    aligned["matched"] = aligned["_merge"].eq("both")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(args.output_file, index=False)
    print(f"factor entries: {len(entries):,}")
    print(f"eligible study events: {len(events):,}")
    print(f"matched entries: {int(aligned['matched'].sum()):,}")
    matched_payoff = pd.to_numeric(aligned[f"net_reversal_{horizon}"], errors="coerce")
    print(f"matched event net mean: {matched_payoff.mean() * 10_000:.3f} bps")
    print(f"saved: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
