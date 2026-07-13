#!/usr/bin/env python3
"""Train/validation search for a liquid whole-universe indicator reentry rule."""

from __future__ import annotations

import argparse
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
    "cn_futures_indicator_reentry_spec_search.csv"
)
ENTRY_BANDS = (
    (1.50, 1.75),
    (1.50, 2.00),
    (1.50, 2.50),
    (1.50, math.inf),
    (1.75, 2.50),
    (2.00, math.inf),
)
LIQUIDITY_MINIMUMS = (0.60, 0.70, 0.80, 0.90)
VOLUME_MAXIMUMS = (0.80, 1.00, 1.20, 1.50, math.inf)
HORIZON_BARS = (12, 24)
INDICATOR_MODES = (
    "all",
    "no_volume_spike",
    "confirmation_zero",
    "confirmation_at_most_one",
    "oscillators_not_confirmed",
    "rsi_turn_without_full_extreme",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-file", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--validation-end", default="2025-10-01")
    parser.add_argument("--min-train-events", type=int, default=300)
    parser.add_argument("--min-validation-events", type=int, default=200)
    return parser.parse_args()


def stats(values: np.ndarray) -> tuple[int, float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return 0, float("nan"), float("nan"), float("nan")
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    t_stat = float(np.mean(values) / (std / math.sqrt(len(values)))) if std > 0 else float("nan")
    return len(values), float(np.mean(values) * 10_000), float(np.mean(values > 0)), t_stat


def indicator_mask(events: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones(len(events), dtype=bool)
    if mode == "no_volume_spike":
        return events["volume_confirm"].eq(0).to_numpy()
    if mode == "confirmation_zero":
        return events["confirmation_count"].eq(0).to_numpy()
    if mode == "confirmation_at_most_one":
        return events["confirmation_count"].le(1).to_numpy()
    if mode == "oscillators_not_confirmed":
        return (events["rsi_confirm"].eq(0) & events["kdj_confirm"].eq(0)).to_numpy()
    if mode == "rsi_turn_without_full_extreme":
        return (events["rsi_turn"].gt(0) & events["rsi_confirm"].eq(0)).to_numpy()
    raise ValueError(f"Unknown indicator mode: {mode}")


def main() -> int:
    args = parse_args()
    columns = [
        "datetime",
        "quarter",
        "entry_abs_z",
        "liquidity_rank",
        "volume_ratio",
        "rsi_turn",
        "rsi_confirm",
        "kdj_confirm",
        "volume_confirm",
        "confirmation_count",
        "net_reversal_12",
        "net_reversal_24",
    ]
    events = pl.read_parquet(args.events_file, columns=columns).to_pandas()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    train_mask = events["datetime"].lt(train_end).to_numpy()
    validation_mask = events["datetime"].ge(train_end).to_numpy() & events["datetime"].lt(validation_end).to_numpy()
    selection_mask = train_mask | validation_mask
    z = pd.to_numeric(events["entry_abs_z"], errors="coerce").to_numpy()
    liquidity = pd.to_numeric(events["liquidity_rank"], errors="coerce").to_numpy()
    volume = pd.to_numeric(events["volume_ratio"], errors="coerce").to_numpy()

    rows: list[dict[str, object]] = []
    for indicator_mode in INDICATOR_MODES:
        indicators = indicator_mask(events, indicator_mode)
        for entry_min, entry_max in ENTRY_BANDS:
            entry = (z >= entry_min) & (z < entry_max)
            for liquidity_min in LIQUIDITY_MINIMUMS:
                liquid = liquidity >= liquidity_min
                for volume_max in VOLUME_MAXIMUMS:
                    spec = indicators & entry & liquid & (volume <= volume_max)
                    for horizon in HORIZON_BARS:
                        values = pd.to_numeric(events[f"net_reversal_{horizon}"], errors="coerce").to_numpy()
                        train = stats(values[spec & train_mask])
                        validation = stats(values[spec & validation_mask])
                        if train[0] < args.min_train_events or validation[0] < args.min_validation_events:
                            continue
                        quarter_means = (
                            pd.DataFrame(
                                {
                                    "quarter": events.loc[spec & selection_mask, "quarter"],
                                    "net": values[spec & selection_mask],
                                }
                            )
                            .groupby("quarter", observed=True)["net"]
                            .mean()
                        )
                        positive_quarter_rate = float((quarter_means > 0).mean()) if len(quarter_means) else float("nan")
                        score = (
                            min(train[1], validation[1])
                            + 0.25 * (train[1] + validation[1])
                            - 0.25 * abs(train[1] - validation[1])
                            + max(positive_quarter_rate - 0.5, -0.5)
                        )
                        rows.append(
                            {
                                "indicator_mode": indicator_mode,
                                "entry_z_min": entry_min,
                                "entry_z_max": entry_max,
                                "liquidity_rank_min": liquidity_min,
                                "volume_ratio_max": volume_max,
                                "horizon_minutes": horizon * 5,
                                "train_events": train[0],
                                "train_net_mean_bps": train[1],
                                "train_hit_rate": train[2],
                                "train_t_stat": train[3],
                                "validation_events": validation[0],
                                "validation_net_mean_bps": validation[1],
                                "validation_hit_rate": validation[2],
                                "validation_t_stat": validation[3],
                                "positive_quarter_rate": positive_quarter_rate,
                                "selection_score": score,
                            }
                        )

    result = pd.DataFrame(rows).sort_values(
        ["selection_score", "validation_net_mean_bps", "train_net_mean_bps"], ascending=False
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_file, index=False)
    robust = result.loc[
        result["train_net_mean_bps"].gt(0)
        & result["validation_net_mean_bps"].gt(0)
        & result["positive_quarter_rate"].ge(0.60)
    ]
    print(f"Wrote {len(result):,} eligible specifications to {args.output_file.resolve()}")
    print(f"Robust positive specifications: {len(robust):,}")
    print(robust.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
