#!/usr/bin/env python3
"""Select robust session-gap reversal specifications without using the audit split."""

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
    "cn_futures_session_open_reversal_development_pre_audit_"
    "2024-01-01_2026-03-31_events.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/"
    "cn_futures_session_open_reversal_spec_search.csv"
)
DECISION_MINUTES = (0, 3, 5, 10, 15, 30)
OPEN_TYPES = ("day_open", "night_open", "afternoon_open")
GAP_BANDS = (
    (0.50, 1.00),
    (0.50, 1.50),
    (0.50, 2.00),
    (0.75, 1.00),
    (0.75, 1.50),
    (0.75, 2.00),
    (0.75, 3.00),
    (1.00, 1.50),
    (1.00, 2.00),
    (1.00, 3.00),
    (1.50, 2.00),
    (1.50, 3.00),
    (2.00, 3.00),
    (2.00, math.inf),
    (3.00, math.inf),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-file", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--validation-end", default="2025-10-01")
    parser.add_argument("--min-train-events", type=int, default=100)
    parser.add_argument("--min-validation-events", type=int, default=75)
    return parser.parse_args()


def sample_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"events": 0, "mean_bps": float("nan"), "hit_rate": float("nan"), "t_stat": float("nan")}
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    t_stat = float(np.mean(finite) / (std / math.sqrt(len(finite)))) if std > 0 else float("nan")
    return {
        "events": int(len(finite)),
        "mean_bps": float(np.mean(finite) * 10_000),
        "hit_rate": float(np.mean(finite > 0)),
        "t_stat": t_stat,
    }


def main() -> int:
    args = parse_args()
    columns = [
        "datetime",
        "quarter",
        "open_type",
        "session_minute",
        "sector",
        "gap_z",
        "net_reversal_to_session_close",
    ]
    events = pl.read_parquet(args.events_file, columns=columns).to_pandas()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    events["abs_gap_z"] = pd.to_numeric(events["gap_z"], errors="coerce").abs()
    events["net"] = pd.to_numeric(events["net_reversal_to_session_close"], errors="coerce")
    events = events.dropna(subset=["datetime", "abs_gap_z", "net"])

    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    train_mask = events["datetime"].lt(train_end).to_numpy()
    validation_mask = events["datetime"].ge(train_end).to_numpy() & events["datetime"].lt(validation_end).to_numpy()
    selection_mask = train_mask | validation_mask

    sectors = ["ALL"] + sorted(
        sector
        for sector, count in events.loc[selection_mask, "sector"].value_counts().items()
        if int(count) >= args.min_train_events + args.min_validation_events
    )
    abs_gap_z = events["abs_gap_z"].to_numpy()
    net = events["net"].to_numpy()
    rows: list[dict[str, object]] = []

    for open_type in OPEN_TYPES:
        open_mask = events["open_type"].eq(open_type).to_numpy()
        for decision_minute in DECISION_MINUTES:
            time_mask = events["session_minute"].eq(decision_minute).to_numpy()
            for sector in sectors:
                sector_mask = np.ones(len(events), dtype=bool) if sector == "ALL" else events["sector"].eq(sector).to_numpy()
                base_mask = open_mask & time_mask & sector_mask
                if int(np.sum(base_mask & selection_mask)) < args.min_train_events + args.min_validation_events:
                    continue
                for gap_min, gap_max in GAP_BANDS:
                    spec_mask = base_mask & (abs_gap_z >= gap_min) & (abs_gap_z < gap_max)
                    train = sample_stats(net[spec_mask & train_mask])
                    validation = sample_stats(net[spec_mask & validation_mask])
                    if train["events"] < args.min_train_events or validation["events"] < args.min_validation_events:
                        continue

                    quarter_means = (
                        events.loc[spec_mask & selection_mask, ["quarter", "net"]]
                        .groupby("quarter", observed=True)["net"]
                        .mean()
                    )
                    positive_quarter_rate = float((quarter_means > 0).mean()) if len(quarter_means) else float("nan")
                    train_mean = float(train["mean_bps"])
                    validation_mean = float(validation["mean_bps"])
                    score = (
                        min(train_mean, validation_mean)
                        + 0.25 * (train_mean + validation_mean)
                        - 0.25 * abs(train_mean - validation_mean)
                        + 0.5 * max(positive_quarter_rate - 0.5, -0.5)
                    )
                    rows.append(
                        {
                            "open_type": open_type,
                            "decision_minute": decision_minute,
                            "sector": sector,
                            "gap_z_min": gap_min,
                            "gap_z_max": gap_max,
                            "train_events": train["events"],
                            "train_net_mean_bps": train_mean,
                            "train_hit_rate": train["hit_rate"],
                            "train_t_stat": train["t_stat"],
                            "validation_events": validation["events"],
                            "validation_net_mean_bps": validation_mean,
                            "validation_hit_rate": validation["hit_rate"],
                            "validation_t_stat": validation["t_stat"],
                            "positive_quarter_rate": positive_quarter_rate,
                            "selection_score": score,
                        }
                    )

    result = pd.DataFrame(rows).sort_values(
        ["selection_score", "validation_net_mean_bps", "train_net_mean_bps"],
        ascending=False,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_file, index=False)
    print(f"Wrote {len(result):,} eligible specifications to {args.output_file.resolve()}")
    robust = result.loc[
        result["train_net_mean_bps"].gt(0)
        & result["validation_net_mean_bps"].gt(0)
        & result["positive_quarter_rate"].ge(0.60)
    ]
    print(f"Robust positive specifications: {len(robust):,}")
    print(robust.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
