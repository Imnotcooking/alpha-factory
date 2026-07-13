#!/usr/bin/env python3
"""Walk-forward specification search for CN futures pair-reversion events."""

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
    "cn_futures_pair_reversion_full_with_regime_"
    "2024-01-01_2026-07-09_events.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "runtime/artifacts/research/minute_reversion/"
    "cn_futures_pair_reversion_spec_search.csv"
)
HORIZONS = (3, 5, 10, 15, 30, 60)
SHOCK_BANDS = (
    (1.5, 2.0),
    (1.5, 2.5),
    (1.5, 3.0),
    (1.5, 4.0),
    (1.5, math.inf),
    (2.0, 2.5),
    (2.0, 3.0),
    (2.0, 4.0),
    (2.0, math.inf),
    (2.5, 3.0),
    (2.5, 4.0),
    (2.5, math.inf),
    (3.0, 4.0),
    (3.0, math.inf),
    (4.0, math.inf),
)
REGIME_SPECS = (
    ("none", math.inf, math.inf, -math.inf),
    ("negative_autocorr", 0.0, math.inf, -math.inf),
    ("low_variance_ratio", math.inf, 0.90, -math.inf),
    ("mean_reverting", 0.0, 1.00, -math.inf),
    ("strong_mean_reverting", -0.05, 1.00, -math.inf),
    ("correlated_mean_reverting", 0.0, 1.00, 0.30),
    ("strong_correlated_mean_reverting", 0.0, 0.90, 0.50),
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


def stats(values: np.ndarray) -> tuple[int, float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return 0, float("nan"), float("nan"), float("nan")
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    t_stat = float(np.mean(values) / (std / math.sqrt(len(values)))) if std > 0 else float("nan")
    return len(values), float(np.mean(values) * 10_000), float(np.mean(values > 0)), t_stat


def main() -> int:
    args = parse_args()
    columns = [
        "pair",
        "datetime",
        "quarter",
        "shock_z",
        "spread_autocorr_1",
        "leg_correlation",
        "variance_ratio",
    ] + [f"net_reversal_{h}" for h in HORIZONS]
    events = pl.read_parquet(args.events_file, columns=columns).to_pandas()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    events["abs_shock_z"] = pd.to_numeric(events["shock_z"], errors="coerce").abs()
    autocorr = pd.to_numeric(events["spread_autocorr_1"], errors="coerce").to_numpy()
    leg_correlation = pd.to_numeric(events["leg_correlation"], errors="coerce").to_numpy()
    variance_ratio = pd.to_numeric(events["variance_ratio"], errors="coerce").to_numpy()
    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    train_mask = events["datetime"].lt(train_end).to_numpy()
    validation_mask = events["datetime"].ge(train_end).to_numpy() & events["datetime"].lt(validation_end).to_numpy()
    selection_mask = train_mask | validation_mask
    shock_z = events["abs_shock_z"].to_numpy()

    rows: list[dict[str, object]] = []
    for pair in sorted(events["pair"].dropna().unique()):
        pair_mask = events["pair"].eq(pair).to_numpy()
        for shock_min, shock_max in SHOCK_BANDS:
            shock_mask = pair_mask & (shock_z >= shock_min) & (shock_z < shock_max)
            for regime_name, autocorr_max, variance_ratio_max, leg_correlation_min in REGIME_SPECS:
                regime_mask = (
                    np.isfinite(autocorr)
                    & np.isfinite(leg_correlation)
                    & np.isfinite(variance_ratio)
                    & (autocorr <= autocorr_max)
                    & (variance_ratio <= variance_ratio_max)
                    & (leg_correlation >= leg_correlation_min)
                )
                spec_mask = shock_mask & regime_mask
                for horizon in HORIZONS:
                    values = pd.to_numeric(events[f"net_reversal_{horizon}"], errors="coerce").to_numpy()
                    train = stats(values[spec_mask & train_mask])
                    validation = stats(values[spec_mask & validation_mask])
                    if train[0] < args.min_train_events or validation[0] < args.min_validation_events:
                        continue
                    quarter_means = (
                        pd.DataFrame(
                            {
                                "quarter": events.loc[spec_mask & selection_mask, "quarter"],
                                "net": values[spec_mask & selection_mask],
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
                        + 0.5 * max(positive_quarter_rate - 0.5, -0.5)
                    )
                    rows.append(
                        {
                            "pair": pair,
                            "shock_z_min": shock_min,
                            "shock_z_max": shock_max,
                            "regime": regime_name,
                            "autocorr_max": autocorr_max,
                            "variance_ratio_max": variance_ratio_max,
                            "leg_correlation_min": leg_correlation_min,
                            "horizon_minutes": horizon,
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

    result = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
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
