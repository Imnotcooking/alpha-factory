#!/usr/bin/env python3
"""Event study for one-minute failed-auction reversal in CN futures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.data.instruments import InstrumentMaster  # noqa: E402


DEFAULT_DATA = REPO_ROOT / (
    "runtime/data/futures_cn/intraday/"
    "futures_data_futures_main_1m_adj_20260710_090753.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/artifacts/research/minute_reversion"
HORIZONS = (1, 3, 5, 10, 15, 30, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--range-window", type=int, default=20)
    parser.add_argument("--atr-window", type=int, default=30)
    parser.add_argument("--volume-window", type=int, default=480)
    parser.add_argument("--min-excursion-atr", type=float, default=0.05)
    parser.add_argument("--min-rejection-atr", type=float, default=0.25)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--slippage-ticks-round-trip", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="development_failed_auction")
    return parser.parse_args()


def instrument_frame() -> pl.DataFrame:
    master = InstrumentMaster("FUTURES_CN")
    rows: list[dict[str, object]] = []
    for base, sector in master.get_sector_map().items():
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
    return pl.DataFrame(rows)


def build_events(args: argparse.Namespace) -> pl.DataFrame:
    group = ["ticker", "session_id"]
    start = pl.lit(args.start_date).str.to_datetime("%Y-%m-%d")
    end = pl.lit(args.end_date).str.to_datetime("%Y-%m-%d") + pl.duration(days=1)
    frame = (
        pl.scan_parquet(args.data_file)
        .select(
            "symbol",
            "datetime",
            "month_change",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        )
        .filter((pl.col("datetime") >= start) & (pl.col("datetime") < end))
        .rename({"symbol": "ticker"})
        .with_columns(pl.col("ticker").str.extract(r"\.([A-Za-z]+)$", 1).alias("base"))
        .join(instrument_frame().lazy(), on="base", how="left")
        .sort(["ticker", "datetime"])
        .with_columns(
            pl.col("datetime").shift(1).over("ticker").alias("prev_datetime"),
            pl.col("close").shift(1).over("ticker").alias("prev_close"),
        )
        .with_columns(
            (
                pl.col("prev_datetime").is_null()
                | (
                    (pl.col("datetime") - pl.col("prev_datetime")).dt.total_minutes()
                    > args.session_gap_minutes
                )
            ).alias("session_start"),
            pl.max_horizontal(
                (pl.col("high") - pl.col("low")).abs(),
                (pl.col("high") - pl.col("prev_close")).abs(),
                (pl.col("low") - pl.col("prev_close")).abs(),
            ).alias("true_range"),
        )
        .with_columns(pl.col("session_start").cum_sum().over("ticker").alias("session_id"))
        .with_columns(
            pl.int_range(pl.len()).over(group).alias("session_minute"),
            pl.col("high")
            .rolling_max(window_size=args.range_window, min_samples=args.range_window)
            .shift(1)
            .over(group)
            .alias("prior_range_high"),
            pl.col("low")
            .rolling_min(window_size=args.range_window, min_samples=args.range_window)
            .shift(1)
            .over(group)
            .alias("prior_range_low"),
            pl.col("true_range")
            .ewm_mean(span=args.atr_window, adjust=False, min_samples=max(10, args.atr_window // 3))
            .shift(1)
            .over(group)
            .alias("atr"),
            pl.col("volume")
            .rolling_median(window_size=args.volume_window, min_samples=max(120, args.volume_window // 4))
            .shift(1)
            .over("ticker")
            .alias("median_volume"),
            (pl.col("open_interest") - pl.col("open_interest").shift(1).over(group)).alias("delta_oi"),
            pl.col("month_change")
            .abs()
            .rolling_sum(window_size=args.range_window + 1, min_samples=1)
            .over(group)
            .alias("roll_in_window"),
            *[
                (pl.col("close").shift(-h).over(group) / pl.col("close"))
                .log()
                .alias(f"forward_return_{h}")
                for h in HORIZONS
            ],
        )
        .with_columns(
            ((pl.col("high") - pl.col("prior_range_high")) / pl.col("atr")).alias("up_excursion_atr"),
            ((pl.col("prior_range_low") - pl.col("low")) / pl.col("atr")).alias("down_excursion_atr"),
            ((pl.col("high") - pl.col("close")) / pl.col("atr")).alias("upper_rejection_atr"),
            ((pl.col("close") - pl.col("low")) / pl.col("atr")).alias("lower_rejection_atr"),
            ((pl.col("high") - pl.col("low")) / pl.col("atr")).alias("bar_range_atr"),
            (pl.col("volume") / pl.col("median_volume")).alias("volume_ratio"),
            (pl.col("delta_oi").abs() / pl.col("volume").clip(lower_bound=1.0)).alias("oi_participation"),
            pl.when((pl.col("high") - pl.col("low")) > 0)
            .then((2 * pl.col("close") - pl.col("high") - pl.col("low")) / (pl.col("high") - pl.col("low")))
            .otherwise(0.0)
            .alias("close_location"),
        )
        .with_columns(
            (
                pl.col("up_excursion_atr").ge(args.min_excursion_atr)
                & pl.col("upper_rejection_atr").ge(args.min_rejection_atr)
                & pl.col("close").lt(pl.col("prior_range_high"))
            ).alias("failed_up_auction"),
            (
                pl.col("down_excursion_atr").ge(args.min_excursion_atr)
                & pl.col("lower_rejection_atr").ge(args.min_rejection_atr)
                & pl.col("close").gt(pl.col("prior_range_low"))
            ).alias("failed_down_auction"),
            (
                args.slippage_ticks_round_trip * pl.col("tick_size") / pl.col("close")
                + pl.when(pl.col("fee_type") == "fixed")
                .then((pl.col("fee_open") + pl.col("fee_close_today")) / (pl.col("close") * pl.col("multiplier")))
                .otherwise(pl.col("fee_open") + pl.col("fee_close_today"))
            ).alias("estimated_round_trip_cost"),
        )
        .filter(
            (pl.col("failed_up_auction") ^ pl.col("failed_down_auction"))
            & pl.col("roll_in_window").eq(0)
            & pl.col("estimated_round_trip_cost").is_finite()
        )
        .with_columns(
            pl.when(pl.col("failed_down_auction")).then(1.0).otherwise(-1.0).alias("position_direction"),
            pl.when(pl.col("failed_down_auction"))
            .then(pl.col("down_excursion_atr"))
            .otherwise(pl.col("up_excursion_atr"))
            .alias("excursion_atr"),
            pl.when(pl.col("failed_down_auction"))
            .then(pl.col("lower_rejection_atr"))
            .otherwise(pl.col("upper_rejection_atr"))
            .alias("rejection_atr"),
            (
                pl.col("datetime").dt.year().cast(pl.String)
                + "-Q"
                + (((pl.col("datetime").dt.month() - 1) // 3) + 1).cast(pl.String)
            ).alias("quarter"),
        )
        .with_columns(
            pl.col("excursion_atr").cut([0.10, 0.20, 0.35, 0.50, 0.75]).alias("excursion_bin"),
            pl.col("rejection_atr").cut([0.50, 0.75, 1.00, 1.50, 2.00]).alias("rejection_bin"),
            pl.col("bar_range_atr").cut([1.0, 1.5, 2.0, 3.0]).alias("range_bin"),
            pl.col("volume_ratio").cut([0.8, 1.2, 1.5, 2.0]).alias("volume_bin"),
            pl.col("oi_participation").cut([0.1, 0.25, 0.5, 1.0]).alias("oi_bin"),
            *[
                (pl.col("position_direction") * pl.col(f"forward_return_{h}")).alias(f"gross_reversal_{h}")
                for h in HORIZONS
            ],
        )
        .with_columns(
            *[
                (pl.col(f"gross_reversal_{h}") - pl.col("estimated_round_trip_cost")).alias(f"net_reversal_{h}")
                for h in HORIZONS
            ]
        )
        .select(
            "ticker",
            "base",
            "sector",
            "datetime",
            "quarter",
            "session_minute",
            "close",
            "position_direction",
            "excursion_atr",
            "rejection_atr",
            "bar_range_atr",
            "volume_ratio",
            "oi_participation",
            "close_location",
            "estimated_round_trip_cost",
            "excursion_bin",
            "rejection_bin",
            "range_bin",
            "volume_bin",
            "oi_bin",
            *[f"gross_reversal_{h}" for h in HORIZONS],
            *[f"net_reversal_{h}" for h in HORIZONS],
        )
    )
    return frame.collect(engine="streaming")


def summarize(events: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    dimensions = [
        "quarter",
        "sector",
        "base",
        "excursion_bin",
        "rejection_bin",
        "range_bin",
        "volume_bin",
        "oi_bin",
    ]
    slices: list[tuple[str, pl.DataFrame]] = [("all", events)]
    for dimension in dimensions:
        for key, group in events.partition_by(dimension, as_dict=True, maintain_order=True).items():
            value = key[0] if isinstance(key, tuple) else key
            slices.append((f"{dimension}={value}", group))
    for slice_name, group in slices:
        for horizon in HORIZONS:
            gross = group[f"gross_reversal_{horizon}"].drop_nulls().to_numpy()
            net = group[f"net_reversal_{horizon}"].drop_nulls().to_numpy()
            std = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
            rows.append(
                {
                    "slice": slice_name,
                    "horizon_minutes": horizon,
                    "events": int(len(net)),
                    "gross_mean_bps": float(np.mean(gross) * 10_000) if len(gross) else None,
                    "net_mean_bps": float(np.mean(net) * 10_000) if len(net) else None,
                    "net_hit_rate": float(np.mean(net > 0)) if len(net) else None,
                    "net_t_stat": float(np.mean(net) / (std / math.sqrt(len(net)))) if std > 0 else None,
                }
            )
    return pl.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = build_events(args)
    if events.is_empty():
        raise RuntimeError("No failed-auction events passed the base definition.")
    stem = f"cn_futures_failed_auction_{args.label}_{args.start_date}_{args.end_date}"
    event_path = args.output_dir / f"{stem}_events.parquet"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    events.write_parquet(event_path, compression="zstd")
    summary = summarize(events)
    summary.write_csv(summary_path)
    metadata = {
        "data_file": str(args.data_file.resolve()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "range_window": args.range_window,
        "atr_window": args.atr_window,
        "min_excursion_atr": args.min_excursion_atr,
        "min_rejection_atr": args.min_rejection_atr,
        "slippage_ticks_round_trip": args.slippage_ticks_round_trip,
        "event_count": events.height,
        "event_file": str(event_path.resolve()),
        "summary_file": str(summary_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    print("\nOverall:")
    print(summary.filter(pl.col("slice") == "all"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
