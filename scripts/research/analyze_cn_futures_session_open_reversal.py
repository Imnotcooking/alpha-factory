#!/usr/bin/env python3
"""Event study for minute-level reversal after CN futures session gaps."""

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
DECISION_MINUTES = (0, 1, 3, 5, 10, 15, 30)
HORIZONS = (1, 3, 5, 10, 15, 30, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--gap-history", type=int, default=60)
    parser.add_argument("--min-gap-z", type=float, default=0.5)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--slippage-ticks-round-trip", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="development_open_gap")
    parser.add_argument("--debug", action="store_true")
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


def _open_type() -> pl.Expr:
    hour = pl.col("datetime").dt.hour()
    return (
        pl.when(hour >= 20)
        .then(pl.lit("night_open"))
        .when(hour < 11)
        .then(pl.lit("day_open"))
        .when(hour < 15)
        .then(pl.lit("afternoon_open"))
        .otherwise(pl.lit("other_open"))
    )


def build_event_frame(args: argparse.Namespace) -> pl.DataFrame:
    group = ["ticker", "session_id"]
    start = pl.lit(args.start_date).str.to_datetime("%Y-%m-%d")
    end = pl.lit(args.end_date).str.to_datetime("%Y-%m-%d") + pl.duration(days=1)

    bars = (
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
            ).alias("session_start")
        )
        .with_columns(pl.col("session_start").cum_sum().over("ticker").alias("session_id"))
        .with_columns(
            pl.int_range(pl.len()).over(group).alias("session_minute"),
            pl.col("datetime").first().over(group).alias("session_start_time"),
            pl.col("open").first().over(group).alias("session_open"),
            pl.col("prev_close").first().over(group).alias("previous_session_close"),
            pl.col("month_change").first().over(group).alias("session_roll_flag"),
            pl.col("close").last().over(group).alias("session_close"),
            _open_type().first().over(group).alias("open_type"),
            (pl.col("close") / pl.col("prev_close")).log().alias("ret_1"),
            pl.col("volume")
            .rolling_median(window_size=480, min_samples=120)
            .shift(1)
            .over("ticker")
            .alias("median_volume"),
        )
        .with_columns(
            (pl.col("session_open") / pl.col("previous_session_close")).log().alias("gap_return"),
            (pl.col("close") / pl.col("previous_session_close")).log().alias("displacement_return"),
            (pl.col("close") / pl.col("session_open")).log().alias("from_open_return"),
            (pl.col("volume") / pl.col("median_volume")).alias("volume_ratio"),
            *[
                (pl.col("close").shift(-h).over(group) / pl.col("close"))
                .log()
                .alias(f"forward_return_{h}")
                for h in HORIZONS
            ],
        )
    )

    sessions = (
        bars.filter(pl.col("session_minute") == 0)
        .select("ticker", "session_start_time", "open_type", "gap_return")
        .sort(["ticker", "open_type", "session_start_time"])
        .with_columns(
            pl.col("gap_return")
            .abs()
            .rolling_quantile(
                quantile=0.75,
                window_size=args.gap_history,
                min_samples=max(20, args.gap_history // 3),
            )
            .shift(1)
            .over(["ticker", "open_type"])
            .alias("historical_gap_scale")
        )
        .select("ticker", "session_start_time", "historical_gap_scale")
    )
    if args.debug:
        session_diagnostics = sessions.select(
            pl.len().alias("sessions"),
            pl.col("historical_gap_scale").is_not_null().sum().alias("scaled_sessions"),
        ).collect(engine="streaming")
        print("Session-scale diagnostics:")
        print(session_diagnostics)

    candidates = (
        bars.join(sessions, on=["ticker", "session_start_time"], how="left")
        .filter(pl.col("session_minute").is_in(DECISION_MINUTES))
        .with_columns(
            (pl.col("gap_return") / pl.col("historical_gap_scale")).alias("gap_z"),
            (pl.col("displacement_return").abs() / pl.col("gap_return").abs()).alias("retained_gap_fraction"),
            (
                args.slippage_ticks_round_trip * pl.col("tick_size") / pl.col("close")
                + pl.when(pl.col("fee_type") == "fixed")
                .then((pl.col("fee_open") + pl.col("fee_close_today")) / (pl.col("close") * pl.col("multiplier")))
                .otherwise(pl.col("fee_open") + pl.col("fee_close_today"))
            ).alias("estimated_round_trip_cost"),
        )
        .with_columns(
            pl.col("gap_return").sign().alias("event_direction"),
            (-pl.col("gap_return").sign() * pl.col("ret_1")).alias("last_bar_confirmation"),
            (
                (pl.col("displacement_return").sign() == pl.col("gap_return").sign())
                & pl.col("displacement_return").ne(0)
            ).alias("gap_still_open"),
            (pl.col("from_open_return").sign() == pl.col("gap_return").sign()).alias("continued_from_open"),
            (
                pl.col("datetime").dt.strftime("%Y-Q")
                + (((pl.col("datetime").dt.month() - 1) // 3) + 1).cast(pl.String)
            ).alias("quarter"),
            pl.col("gap_z").abs().cut([0.75, 1.0, 1.5, 2.0, 3.0]).alias("gap_z_bin"),
            pl.col("retained_gap_fraction").cut([0.25, 0.5, 0.75, 1.0, 1.5]).alias("retained_gap_bin"),
            pl.col("volume_ratio").cut([0.8, 1.2, 1.5, 2.0]).alias("volume_bin"),
        )
    )
    if args.debug:
        diagnostics = candidates.select(
            pl.len().alias("candidate_rows"),
            pl.col("gap_return").is_not_null().sum().alias("gap_rows"),
            pl.col("historical_gap_scale").is_not_null().sum().alias("scaled_rows"),
            pl.col("gap_z").is_finite().sum().alias("finite_gap_z_rows"),
            pl.col("gap_still_open").fill_null(False).sum().alias("open_gap_rows"),
            pl.col("session_roll_flag").fill_null(0).eq(0).sum().alias("non_roll_rows"),
            pl.col("estimated_round_trip_cost").is_finite().sum().alias("costed_rows"),
        ).collect(engine="streaming")
        print("Candidate diagnostics:")
        print(diagnostics)

    events = (
        candidates
        .filter(
            pl.col("gap_z").abs().ge(args.min_gap_z)
            & pl.col("gap_still_open")
            & pl.col("session_roll_flag").fill_null(0).eq(0)
            & pl.col("estimated_round_trip_cost").is_finite()
        )
        .with_columns(
            *[
                (-pl.col("event_direction") * pl.col(f"forward_return_{h}")).alias(f"gross_reversal_{h}")
                for h in HORIZONS
            ],
            (-pl.col("event_direction") * (pl.col("session_close") / pl.col("close")).log()).alias(
                "gross_reversal_to_session_close"
            ),
        )
        .with_columns(
            *[
                (pl.col(f"gross_reversal_{h}") - pl.col("estimated_round_trip_cost")).alias(f"net_reversal_{h}")
                for h in HORIZONS
            ],
            (pl.col("gross_reversal_to_session_close") - pl.col("estimated_round_trip_cost")).alias(
                "net_reversal_to_session_close"
            ),
        )
        .select(
            "ticker",
            "base",
            "sector",
            "datetime",
            "quarter",
            "open_type",
            "session_minute",
            "close",
            "gap_return",
            "gap_z",
            "displacement_return",
            "from_open_return",
            "retained_gap_fraction",
            "last_bar_confirmation",
            "continued_from_open",
            "volume_ratio",
            "estimated_round_trip_cost",
            "gap_z_bin",
            "retained_gap_bin",
            "volume_bin",
            *[f"gross_reversal_{h}" for h in HORIZONS],
            *[f"net_reversal_{h}" for h in HORIZONS],
            "gross_reversal_to_session_close",
            "net_reversal_to_session_close",
        )
    )
    return events.collect(engine="streaming")


def _stats(values: np.ndarray) -> tuple[int, float, float, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    t_stat = float(np.mean(finite) / (std / math.sqrt(len(finite)))) if std > 0 else float("nan")
    return (
        len(finite),
        float(np.mean(finite) * 10_000),
        float(np.median(finite) * 10_000),
        float(np.mean(finite > 0)),
        t_stat,
    )


def summarize(events: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    dimensions = [
        "quarter",
        "open_type",
        "session_minute",
        "sector",
        "base",
        "gap_z_bin",
        "retained_gap_bin",
        "volume_bin",
        "continued_from_open",
    ]
    slices: list[tuple[str, pl.DataFrame]] = [("all", events)]
    for dimension in dimensions:
        for key, group in events.partition_by(dimension, as_dict=True, maintain_order=True).items():
            value = key[0] if isinstance(key, tuple) else key
            slices.append((f"{dimension}={value}", group))

    for slice_name, group in slices:
        for horizon in HORIZONS:
            count, net_mean, net_median, hit_rate, t_stat = _stats(group[f"net_reversal_{horizon}"].to_numpy())
            gross = group[f"gross_reversal_{horizon}"].to_numpy()
            rows.append(
                {
                    "slice": slice_name,
                    "horizon_minutes": horizon,
                    "events": count,
                    "gross_mean_bps": float(np.nanmean(gross) * 10_000) if np.isfinite(gross).any() else None,
                    "net_mean_bps": net_mean,
                    "net_median_bps": net_median,
                    "net_hit_rate": hit_rate,
                    "net_t_stat": t_stat,
                }
            )
    return pl.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = build_event_frame(args)
    if events.is_empty():
        raise RuntimeError("No opening-gap events passed the base filters.")

    stem = f"cn_futures_session_open_reversal_{args.label}_{args.start_date}_{args.end_date}"
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
        "gap_history": args.gap_history,
        "min_gap_z": args.min_gap_z,
        "slippage_ticks_round_trip": args.slippage_ticks_round_trip,
        "event_count": events.height,
        "event_definition": "session gap remains open at a fixed decision minute; fade gap direction",
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
