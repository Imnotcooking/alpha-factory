#!/usr/bin/env python3
"""Diagnose conditional minute-level reversal in CN futures OHLCV data.

The script deliberately stops at event-study evidence. It measures a shock on
bar t-1, requires the first opposing bar at t, and evaluates returns beginning
after t. This geometry keeps signal construction separate from the future path
that is used to judge the hypothesis.
"""

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
HORIZONS = (1, 3, 5, 10, 15, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--fast-window", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=120)
    parser.add_argument("--volume-window", type=int, default=480)
    parser.add_argument("--min-shock-z", type=float, default=1.5)
    parser.add_argument(
        "--event-mode",
        choices=("absolute", "sector_residual"),
        default="absolute",
        help="Fade the contract move or only its move relative to the sector median.",
    )
    parser.add_argument(
        "--require-confirmation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require one opposing minute after the shock before measuring the trade.",
    )
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--slippage-ticks-round-trip", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="development")
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


def build_event_frame(args: argparse.Namespace) -> pl.DataFrame:
    group = ["ticker", "session_id"]
    event_shift = 1 if args.require_confirmation else 0
    start = pl.lit(args.start_date).str.to_datetime("%Y-%m-%d")
    end = pl.lit(args.end_date).str.to_datetime("%Y-%m-%d") + pl.duration(days=1)

    frame = (
        pl.scan_parquet(args.data_file)
        .select(
            "symbol",
            "datetime",
            "underlying_symbol",
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
        .with_columns(
            pl.col("ticker").str.extract(r"\.([A-Za-z]+)$", 1).alias("base"),
        )
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
            ).alias("session_start")
        )
        .with_columns(
            pl.col("session_start").cum_sum().over("ticker").alias("session_id")
        )
        .with_columns(
            pl.int_range(pl.len()).over(group).alias("session_minute"),
            (pl.col("close") / pl.col("prev_close")).log().alias("ret_1"),
            pl.when((pl.col("high") - pl.col("low")) > 0)
            .then((2 * pl.col("close") - pl.col("high") - pl.col("low")) / (pl.col("high") - pl.col("low")))
            .otherwise(0.0)
            .alias("close_location"),
            (pl.col("open_interest") - pl.col("open_interest").shift(1).over(group)).alias("delta_oi"),
        )
        .with_columns(
            pl.col("ret_1")
            .rolling_std(window_size=args.vol_window, min_samples=max(30, args.vol_window // 4))
            .shift(1)
            .over(group)
            .alias("sigma_1m"),
            pl.col("volume")
            .rolling_median(window_size=args.volume_window, min_samples=max(60, args.volume_window // 4))
            .shift(1)
            .over("ticker")
            .alias("median_volume"),
            (pl.col("close") / pl.col("close").shift(args.fast_window).over(group))
            .log()
            .alias("shock_return"),
            pl.col("ret_1")
            .abs()
            .rolling_sum(window_size=args.fast_window, min_samples=args.fast_window)
            .over(group)
            .alias("path_length"),
            pl.col("month_change")
            .abs()
            .rolling_sum(window_size=args.fast_window + 1, min_samples=1)
            .over(group)
            .alias("roll_in_window"),
        )
        .with_columns(
            (pl.col("shock_return") / (pl.col("sigma_1m") * math.sqrt(args.fast_window))).alias("shock_z"),
            (pl.col("shock_return").abs() / pl.col("path_length")).alias("path_efficiency"),
            (pl.col("volume") / pl.col("median_volume")).alias("volume_ratio"),
            (pl.col("delta_oi").abs() / pl.col("volume").clip(lower_bound=1.0)).alias("oi_participation"),
            pl.col("shock_return").median().over(["sector", "datetime"]).alias("sector_shock_return"),
            pl.col("ret_1").median().over(["sector", "datetime"]).alias("sector_ret_1"),
            pl.len().over(["sector", "datetime"]).alias("sector_contract_count"),
        )
        .with_columns(
            (pl.col("shock_return") - pl.col("sector_shock_return")).alias("residual_shock_return"),
            (
                (pl.col("shock_return") - pl.col("sector_shock_return")).abs()
                / pl.col("shock_return").abs()
            ).alias("residual_share"),
            *[
                (pl.col("close").shift(-h).over(group) / pl.col("close"))
                .log()
                .alias(f"forward_return_{h}")
                for h in HORIZONS
            ],
        )
        .with_columns(
            (pl.col("residual_shock_return") / (pl.col("sigma_1m") * math.sqrt(args.fast_window)))
            .alias("residual_shock_z"),
            *[
                pl.col(f"forward_return_{h}")
                .median()
                .over(["sector", "datetime"])
                .alias(f"sector_forward_return_{h}")
                for h in HORIZONS
            ],
        )
        .with_columns(
            *[
                (pl.col(f"forward_return_{h}") - pl.col(f"sector_forward_return_{h}"))
                .alias(f"residual_forward_return_{h}")
                for h in HORIZONS
            ],
            (
                pl.col("residual_shock_return")
                if args.event_mode == "sector_residual"
                else pl.col("shock_return")
            )
            .shift(event_shift)
            .over(group)
            .alias("event_shock_return"),
            (
                pl.col("residual_shock_z")
                if args.event_mode == "sector_residual"
                else pl.col("shock_z")
            )
            .shift(event_shift)
            .over(group)
            .alias("event_shock_z"),
            pl.col("path_efficiency").shift(event_shift).over(group).alias("event_path_efficiency"),
            pl.col("volume_ratio").shift(event_shift).over(group).alias("event_volume_ratio"),
            pl.col("oi_participation").shift(event_shift).over(group).alias("event_oi_participation"),
            pl.col("residual_share").shift(event_shift).over(group).alias("event_residual_share"),
            pl.col("roll_in_window").shift(event_shift).over(group).alias("event_roll_in_window"),
        )
        .with_columns(
            pl.col("event_shock_return").sign().alias("event_direction"),
            (
                -pl.col("event_shock_return").sign()
                * (
                    (pl.col("ret_1") - pl.col("sector_ret_1"))
                    if args.event_mode == "sector_residual"
                    else pl.col("ret_1")
                )
            ).alias("confirmation_return"),
            (-pl.col("event_shock_return").sign() * pl.col("close_location")).alias("rejection_clv"),
            (
                args.slippage_ticks_round_trip * pl.col("tick_size") / pl.col("close")
                + pl.when(pl.col("fee_type") == "fixed")
                .then((pl.col("fee_open") + pl.col("fee_close_today")) / (pl.col("close") * pl.col("multiplier")))
                .otherwise(pl.col("fee_open") + pl.col("fee_close_today"))
            ).alias("estimated_round_trip_cost"),
        )
        .filter(
            pl.col("event_shock_z").abs().ge(args.min_shock_z)
            & (pl.col("confirmation_return").gt(0) if args.require_confirmation else pl.lit(True))
            & pl.col("event_roll_in_window").eq(0)
            & pl.col("session_minute").gt(args.fast_window)
            & (
                pl.col("sector_contract_count").gt(1)
                if args.event_mode == "sector_residual"
                else pl.lit(True)
            )
            & pl.col("estimated_round_trip_cost").is_finite()
        )
        .with_columns(
            (pl.col("confirmation_return") / pl.col("event_shock_return").abs()).alias("confirmation_fraction"),
            pl.when(pl.col("session_minute") < 15)
            .then(pl.lit("open_00_14"))
            .when(pl.col("session_minute") < 60)
            .then(pl.lit("early_15_59"))
            .when(pl.col("session_minute") < 180)
            .then(pl.lit("middle_60_179"))
            .otherwise(pl.lit("late_180_plus"))
            .alias("session_phase"),
            (
                pl.col("datetime").dt.strftime("%Y-Q")
                + (((pl.col("datetime").dt.month() - 1) // 3) + 1).cast(pl.String)
            ).alias("quarter"),
            pl.col("event_shock_z").abs().cut([2.0, 2.5, 3.0, 4.0]).alias("shock_z_bin"),
            pl.col("event_volume_ratio").cut([0.8, 1.2, 1.5, 2.0]).alias("volume_bin"),
            pl.col("event_oi_participation").cut([0.1, 0.25, 0.5, 1.0]).alias("oi_bin"),
            pl.col("event_path_efficiency").cut([0.35, 0.55, 0.75, 0.9]).alias("efficiency_bin"),
            pl.col("event_residual_share").cut([0.25, 0.5, 0.75, 1.0]).alias("residual_bin"),
            pl.col("rejection_clv").cut([-0.5, 0.0, 0.5]).alias("rejection_bin"),
        )
        .with_columns(
            pl.col("confirmation_fraction").cut([0.05, 0.1, 0.2, 0.4]).alias("confirmation_bin"),
        )
        .with_columns(
            *[
                (
                    -pl.col("event_direction")
                    * (
                        pl.col(f"residual_forward_return_{h}")
                        if args.event_mode == "sector_residual"
                        else pl.col(f"forward_return_{h}")
                    )
                ).alias(f"gross_reversal_{h}")
                for h in HORIZONS
            ]
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
            "session_phase",
            "close",
            "event_direction",
            "event_shock_return",
            "event_shock_z",
            "event_path_efficiency",
            "event_volume_ratio",
            "event_oi_participation",
            "event_residual_share",
            "confirmation_return",
            "confirmation_fraction",
            "rejection_clv",
            "estimated_round_trip_cost",
            "shock_z_bin",
            "volume_bin",
            "oi_bin",
            "efficiency_bin",
            "residual_bin",
            "rejection_bin",
            "confirmation_bin",
            *[f"gross_reversal_{h}" for h in HORIZONS],
            *[f"net_reversal_{h}" for h in HORIZONS],
        )
    )
    return frame.collect(engine="streaming")


def mean_tstat(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan")
    std = float(np.std(finite, ddof=1))
    return float(np.mean(finite) / (std / math.sqrt(len(finite)))) if std > 0 else float("nan")


def summarize(events: pl.DataFrame, dimensions: list[str]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    slices: list[tuple[str, pl.DataFrame]] = [("all", events)]
    for dimension in dimensions:
        for key, group in events.partition_by(dimension, as_dict=True, maintain_order=True).items():
            value = key[0] if isinstance(key, tuple) else key
            slices.append((f"{dimension}={value}", group))

    for slice_name, group in slices:
        for horizon in HORIZONS:
            gross = group[f"gross_reversal_{horizon}"].to_numpy()
            net = group[f"net_reversal_{horizon}"].to_numpy()
            finite_net = net[np.isfinite(net)]
            rows.append(
                {
                    "slice": slice_name,
                    "horizon_minutes": horizon,
                    "events": int(len(finite_net)),
                    "gross_mean_bps": float(np.nanmean(gross) * 10_000) if len(gross) else None,
                    "net_mean_bps": float(np.nanmean(net) * 10_000) if len(net) else None,
                    "net_median_bps": float(np.nanmedian(net) * 10_000) if len(net) else None,
                    "net_hit_rate": float(np.mean(finite_net > 0)) if len(finite_net) else None,
                    "net_t_stat": mean_tstat(net),
                }
            )
    return pl.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.fast_window < 1 or args.vol_window < 30 or args.volume_window < 60:
        raise ValueError("Window arguments are too short for stable minute diagnostics.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = build_event_frame(args)
    if events.is_empty():
        raise RuntimeError("No events passed the base shock and confirmation filters.")

    stem = f"cn_futures_minute_reversion_{args.label}_{args.start_date}_{args.end_date}"
    event_path = args.output_dir / f"{stem}_events.parquet"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    events.write_parquet(event_path, compression="zstd")

    dimensions = [
        "quarter",
        "sector",
        "base",
        "session_phase",
        "shock_z_bin",
        "volume_bin",
        "oi_bin",
        "efficiency_bin",
        "residual_bin",
        "rejection_bin",
        "confirmation_bin",
    ]
    summary = summarize(events, dimensions)
    summary.write_csv(summary_path)

    metadata = {
        "data_file": str(args.data_file.resolve()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "fast_window": args.fast_window,
        "vol_window": args.vol_window,
        "volume_window": args.volume_window,
        "min_shock_z": args.min_shock_z,
        "event_mode": args.event_mode,
        "require_confirmation": args.require_confirmation,
        "slippage_ticks_round_trip": args.slippage_ticks_round_trip,
        "event_count": events.height,
        "event_definition": (
            "shock on t-1, first opposing return on t, evaluate after t"
            if args.require_confirmation
            else "shock through t, enter after t without an extra confirmation bar"
        ),
        "estimated_cost_definition": "round-trip exchange fees plus configured tick slippage",
        "event_file": str(event_path.resolve()),
        "summary_file": str(summary_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    print("\nOverall results:")
    print(summary.filter(pl.col("slice") == "all"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
