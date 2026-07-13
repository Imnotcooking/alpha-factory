#!/usr/bin/env python3
"""Evaluate a lagged realized-edge gate for a frozen pair-reversion rule."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-file", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--pair", default="precious_metals")
    parser.add_argument("--shock-z-min", type=float, default=4.0)
    parser.add_argument("--horizon", type=int, default=60)
    return parser.parse_args()


def stats(values: pd.Series) -> tuple[int, float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if not len(values):
        return 0, float("nan"), float("nan"), float("nan")
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    t_stat = float(np.mean(values) / (std / math.sqrt(len(values)))) if std > 0 else float("nan")
    return len(values), float(np.mean(values) * 10_000), float(np.mean(values > 0)), t_stat


def main() -> int:
    args = parse_args()
    net_col = f"net_reversal_{args.horizon}"
    columns = ["pair", "datetime", "shock_z", net_col]
    events = pl.read_parquet(args.events_file, columns=columns).to_pandas()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    events = events.loc[
        events["pair"].eq(args.pair)
        & pd.to_numeric(events["shock_z"], errors="coerce").abs().ge(args.shock_z_min)
    ].copy()
    events["trade_date"] = events["datetime"].dt.normalize()
    events["quarter"] = events["datetime"].dt.to_period("Q").astype(str)
    events["split"] = np.select(
        [
            events["datetime"].lt(pd.Timestamp("2025-01-01")),
            events["datetime"].lt(pd.Timestamp("2025-10-01")),
            events["datetime"].lt(pd.Timestamp("2026-04-01")),
        ],
        ["train_2024", "validation_2025_01_09", "audit_2025Q4_2026Q1"],
        default="latest_2026Q2_Q3",
    )

    daily = events.groupby("trade_date", observed=True)[net_col].mean().sort_index().rename("daily_net")
    rows: list[dict[str, object]] = []
    for window in (10, 20, 40, 60):
        gate = (
            daily.shift(1)
            .rolling(window=window, min_periods=max(5, window // 2))
            .mean()
            .rename("trailing_edge")
        )
        evaluated = events.join(gate, on="trade_date")
        for threshold_bps in (0.0, 0.5, 1.0, 2.0):
            traded = evaluated.loc[evaluated["trailing_edge"].gt(threshold_bps / 10_000)].copy()
            for split, group in traded.groupby("split", observed=True):
                count, mean_bps, hit_rate, t_stat = stats(group[net_col])
                rows.append(
                    {
                        "window_active_days": window,
                        "threshold_bps": threshold_bps,
                        "split": split,
                        "events": count,
                        "net_mean_bps": mean_bps,
                        "hit_rate": hit_rate,
                        "t_stat": t_stat,
                    }
                )
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    frozen = 20
    frozen_threshold = 0.0
    frozen_gate = daily.shift(1).rolling(window=frozen, min_periods=10).mean().rename("trailing_edge")
    frozen_events = events.join(frozen_gate, on="trade_date")
    frozen_events = frozen_events.loc[frozen_events["trailing_edge"].gt(frozen_threshold)]
    print("\nFrozen 20-active-day, positive-edge gate by quarter:")
    quarter_rows = []
    for quarter, group in frozen_events.groupby("quarter", observed=True):
        count, mean_bps, hit_rate, t_stat = stats(group[net_col])
        quarter_rows.append(
            {
                "quarter": quarter,
                "events": count,
                "net_mean_bps": mean_bps,
                "hit_rate": hit_rate,
                "t_stat": t_stat,
            }
        )
    print(pd.DataFrame(quarter_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
