#!/usr/bin/env python3
"""Select a liquid-universe indicator reentry rule by daily portfolio behavior."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/"
    "cn_futures_indicator_reentry_full_2024-01-01_2026-07-09_events.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/"
    "cn_futures_indicator_reentry_portfolio_search.csv"
)

ENTRY_BANDS = ((1.50, 1.75), (1.50, 2.00), (1.50, 2.50), (1.75, 2.50))
LIQUIDITY_MINIMUMS = (0.70, 0.80, 0.90)
VOLUME_MAXIMUMS = (0.80, 1.00, 1.20, 1.50)
CONFIRMATION_MAXIMUMS = (0, 1, 2)
HORIZONS = (24, 36, 48)
WEIGHT_MODES = ("equal", "inverse_atr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-file", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--validation-end", default="2025-10-01")
    parser.add_argument("--audit-end", default="2026-07-10")
    parser.add_argument("--base-weight", type=float, default=0.05)
    parser.add_argument("--min-train-events", type=int, default=300)
    parser.add_argument("--min-validation-events", type=int, default=200)
    return parser.parse_args()


def daily_stats(events: pd.DataFrame, payoff: str, weight: np.ndarray) -> dict[str, float]:
    values = pd.to_numeric(events[payoff], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(weight)
    if not finite.any():
        return {"events": 0, "days": 0, "net_bps": np.nan, "annual_return": np.nan, "sharpe": np.nan}
    selected = events.loc[finite, ["datetime"]].copy()
    selected["pnl"] = values[finite] * weight[finite]
    selected["day"] = selected["datetime"].dt.normalize()
    daily = selected.groupby("day", sort=True)["pnl"].sum()
    # Include inactive exchange weekdays so sparse rules do not receive a Sharpe bonus.
    calendar = pd.bdate_range(events["datetime"].min().normalize(), events["datetime"].max().normalize())
    daily = daily.reindex(calendar, fill_value=0.0)
    std = float(daily.std(ddof=1))
    mean = float(daily.mean())
    return {
        "events": int(finite.sum()),
        "days": int(len(daily)),
        "net_bps": float(np.nanmean(values[finite]) * 10_000),
        "annual_return": mean * 252.0,
        "sharpe": mean / std * math.sqrt(252.0) if std > 0 else np.nan,
    }


def event_weights(events: pd.DataFrame, mode: str, base_weight: float) -> np.ndarray:
    if mode == "equal":
        return np.full(len(events), base_weight, dtype=float)
    atr = pd.to_numeric(events["atr_pct"], errors="coerce").to_numpy(dtype=float)
    # A 10 bp five-minute ATR is the risk anchor; clipping prevents tiny ATR estimates
    # from creating leverage and keeps the rule implementable without portfolio resizing.
    return base_weight * np.clip(0.001 / atr, 0.25, 2.0)


def non_overlapping(events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep the first eligible event until its session-aware forward exit."""
    exit_column = f"forward_datetime_{horizon}"
    kept: list[int] = []
    ordered = events.sort_values(["symbol", "datetime"])
    for _, group in ordered.groupby("symbol", sort=False):
        entries = group["datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        exits = group[exit_column].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        indexes = group.index.to_numpy()
        next_entry = np.iinfo(np.int64).min
        nat = np.datetime64("NaT", "ns").astype(np.int64)
        for index, entry_time, exit_time in zip(indexes, entries, exits, strict=True):
            if exit_time == nat or entry_time < next_entry:
                continue
            kept.append(int(index))
            next_entry = exit_time
    return events.loc[kept].sort_values("datetime")


def main() -> int:
    args = parse_args()
    events = pl.read_parquet(args.events_file).to_pandas()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    audit_end = pd.Timestamp(args.audit_end)
    split_masks = {
        "train": events["datetime"].ge(train_start) & events["datetime"].lt(train_end),
        "validation": events["datetime"].ge(train_end) & events["datetime"].lt(validation_end),
        "audit": events["datetime"].ge(validation_end) & events["datetime"].lt(audit_end),
    }
    z = pd.to_numeric(events["entry_abs_z"], errors="coerce")
    liquidity = pd.to_numeric(events["liquidity_rank"], errors="coerce")
    volume = pd.to_numeric(events["volume_ratio"], errors="coerce")
    confirmations = pd.to_numeric(events["confirmation_count"], errors="coerce")

    rows: list[dict[str, object]] = []
    grid = itertools.product(
        ENTRY_BANDS,
        LIQUIDITY_MINIMUMS,
        VOLUME_MAXIMUMS,
        CONFIRMATION_MAXIMUMS,
        HORIZONS,
        WEIGHT_MODES,
    )
    for (entry_min, entry_max), liquidity_min, volume_max, confirmation_max, horizon, weight_mode in grid:
        spec = (
            z.ge(entry_min)
            & z.lt(entry_max)
            & liquidity.ge(liquidity_min)
            & volume.le(volume_max)
            & confirmations.le(confirmation_max)
        )
        payoff = f"net_reversal_{horizon}"
        row: dict[str, object] = {
            "entry_min": entry_min,
            "entry_max": entry_max,
            "liquidity_min": liquidity_min,
            "volume_max": volume_max,
            "confirmation_max": confirmation_max,
            "horizon_bars": horizon,
            "weight_mode": weight_mode,
        }
        eligible = True
        for split, split_mask in split_masks.items():
            sample = non_overlapping(events.loc[spec & split_mask].copy(), horizon)
            weights = event_weights(sample, weight_mode, args.base_weight)
            result = daily_stats(sample, payoff, weights)
            row.update({f"{split}_{key}": value for key, value in result.items()})
            if split == "train" and result["events"] < args.min_train_events:
                eligible = False
            if split == "validation" and result["events"] < args.min_validation_events:
                eligible = False
        if eligible:
            row["selection_min_sharpe"] = min(float(row["train_sharpe"]), float(row["validation_sharpe"]))
            row["selection_mean_sharpe"] = (float(row["train_sharpe"]) + float(row["validation_sharpe"])) / 2.0
            rows.append(row)

    output = pd.DataFrame(rows).sort_values(
        ["selection_min_sharpe", "selection_mean_sharpe"], ascending=False
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)
    print(output.head(30).to_string(index=False))
    print(f"\nSaved {len(output):,} eligible specifications to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
