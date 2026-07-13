#!/usr/bin/env python3
"""Study liquid-universe Bollinger/RSI/KDJ/ATR/volume reentry events."""

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
HORIZON_BARS = (1, 3, 6, 12, 24, 36, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-07-09")
    parser.add_argument("--decision-minutes", type=int, default=5)
    parser.add_argument("--bb-window", type=int, default=24)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--kdj-window", type=int, default=9)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--ker-window", type=int, default=20)
    parser.add_argument("--volume-window", type=int, default=48)
    parser.add_argument("--liquidity-days", type=int, default=20)
    parser.add_argument("--min-entry-z", type=float, default=1.5)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--slippage-ticks-round-trip", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="full_indicator_reentry")
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
    every = f"{int(args.decision_minutes)}m"
    start = pl.lit(args.start_date).str.to_datetime("%Y-%m-%d")
    end = pl.lit(args.end_date).str.to_datetime("%Y-%m-%d") + pl.duration(days=1)
    raw = (
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
        .with_columns(pl.col("symbol").str.extract(r"\.([A-Za-z]+)$", 1).alias("base"))
        .join(instrument_frame().lazy(), on="base", how="left")
        .sort(["symbol", "datetime"])
    )
    coarse = (
        raw.group_by_dynamic(
            "datetime",
            every=every,
            period=every,
            group_by=[
                "symbol",
                "base",
                "sector",
                "tick_size",
                "multiplier",
                "fee_type",
                "fee_open",
                "fee_close_today",
            ],
            closed="left",
            label="right",
        )
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("open_interest").last().alias("open_interest"),
            pl.col("month_change").abs().sum().alias("roll_count"),
            pl.len().alias("source_rows"),
        )
        .filter(pl.col("source_rows") > 0)
        .sort(["symbol", "datetime"])
        .with_columns(
            pl.col("datetime").shift(1).over("symbol").alias("prev_datetime"),
            pl.col("close").shift(1).over("symbol").alias("prev_close"),
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
            (pl.col("close") - pl.col("prev_close")).alias("close_change"),
            pl.col("datetime").dt.date().alias("calendar_date"),
        )
        .with_columns(pl.col("session_start").cum_sum().over("symbol").alias("session_id"))
    )

    daily_liquidity = (
        coarse.group_by(["symbol", "calendar_date"])
        .agg((pl.col("close") * pl.col("volume") * pl.col("multiplier")).sum().alias("daily_notional"))
        .sort(["symbol", "calendar_date"])
        .with_columns(
            pl.col("daily_notional")
            .rolling_median(
                window_size=args.liquidity_days,
                min_samples=max(5, args.liquidity_days // 2),
            )
            .shift(1)
            .over("symbol")
            .alias("lagged_median_daily_notional")
        )
        .with_columns(
            (
                pl.col("lagged_median_daily_notional").rank(method="average").over("calendar_date")
                / pl.len().over("calendar_date")
            ).alias("liquidity_rank")
        )
    )

    group = ["symbol", "session_id"]
    features = (
        coarse.join(daily_liquidity, on=["symbol", "calendar_date"], how="left")
        .sort(["symbol", "datetime"])
        .with_columns(
            pl.int_range(pl.len()).over(group).alias("session_bar"),
            pl.col("close")
            .rolling_mean(window_size=args.bb_window, min_samples=args.bb_window)
            .over("symbol")
            .alias("bb_center"),
            pl.col("close")
            .rolling_std(window_size=args.bb_window, min_samples=args.bb_window)
            .over("symbol")
            .alias("bb_std"),
            pl.col("true_range")
            .ewm_mean(alpha=1.0 / args.atr_window, adjust=False, min_samples=args.atr_window)
            .over("symbol")
            .alias("atr"),
            pl.when(pl.col("close_change") > 0).then(pl.col("close_change")).otherwise(0.0)
            .ewm_mean(alpha=1.0 / args.rsi_window, adjust=False, min_samples=args.rsi_window)
            .over("symbol")
            .alias("avg_gain"),
            pl.when(pl.col("close_change") < 0).then(-pl.col("close_change")).otherwise(0.0)
            .ewm_mean(alpha=1.0 / args.rsi_window, adjust=False, min_samples=args.rsi_window)
            .over("symbol")
            .alias("avg_loss"),
            pl.col("high")
            .rolling_max(window_size=args.kdj_window, min_samples=args.kdj_window)
            .over("symbol")
            .alias("kdj_high"),
            pl.col("low")
            .rolling_min(window_size=args.kdj_window, min_samples=args.kdj_window)
            .over("symbol")
            .alias("kdj_low"),
            pl.col("volume")
            .rolling_median(window_size=args.volume_window, min_samples=max(12, args.volume_window // 4))
            .shift(1)
            .over("symbol")
            .alias("lagged_median_volume"),
            pl.col("close_change")
            .abs()
            .rolling_sum(window_size=args.ker_window, min_samples=args.ker_window)
            .over("symbol")
            .alias("ker_path"),
            pl.col("close").shift(args.ker_window).over("symbol").alias("ker_lag_close"),
            *[
                (pl.col("close").shift(-bars).over(group) / pl.col("close"))
                .log()
                .alias(f"forward_return_{bars}")
                for bars in HORIZON_BARS
            ],
            *[
                pl.col("datetime").shift(-bars).over(group).alias(f"forward_datetime_{bars}")
                for bars in HORIZON_BARS
            ],
        )
        .with_columns(
            ((pl.col("close") - pl.col("bb_center")) / pl.col("bb_std")).alias("bb_z"),
            (100.0 - 100.0 / (1.0 + pl.col("avg_gain") / pl.col("avg_loss"))).alias("rsi"),
            (
                100.0 * (pl.col("close") - pl.col("kdj_low"))
                / (pl.col("kdj_high") - pl.col("kdj_low"))
            ).alias("rsv"),
            (pl.col("volume") / pl.col("lagged_median_volume")).alias("volume_ratio"),
            ((pl.col("close") - pl.col("ker_lag_close")).abs() / pl.col("ker_path")).alias("ker"),
            (pl.col("atr") / pl.col("close")).alias("atr_pct"),
            ((2.0 * pl.col("bb_std")) / pl.col("atr")).alias("half_band_atr"),
            (
                (pl.col("bb_center") - pl.col("bb_center").shift(5).over("symbol")).abs()
                / pl.col("atr")
            ).alias("bb_slope_atr"),
        )
        .with_columns(
            pl.col("rsv").ewm_mean(alpha=1.0 / 3.0, adjust=False, min_samples=3).over("symbol").alias("kdj_k")
        )
        .with_columns(
            pl.col("kdj_k").ewm_mean(alpha=1.0 / 3.0, adjust=False, min_samples=3).over("symbol").alias("kdj_d")
        )
        .with_columns(
            (3.0 * pl.col("kdj_k") - 2.0 * pl.col("kdj_d")).alias("kdj_j"),
            pl.col("bb_z").shift(1).over(group).alias("previous_bb_z"),
            pl.col("rsi").shift(1).over(group).alias("previous_rsi"),
            pl.col("kdj_k").shift(1).over(group).alias("previous_kdj_k"),
            pl.col("kdj_d").shift(1).over(group).alias("previous_kdj_d"),
        )
        .with_columns(
            (
                pl.col("previous_bb_z").le(-args.min_entry_z)
                & pl.col("bb_z").gt(-args.min_entry_z)
                & pl.col("bb_z").lt(0)
            ).alias("long_reentry"),
            (
                pl.col("previous_bb_z").ge(args.min_entry_z)
                & pl.col("bb_z").lt(args.min_entry_z)
                & pl.col("bb_z").gt(0)
            ).alias("short_reentry"),
        )
        .filter(
            (pl.col("long_reentry") ^ pl.col("short_reentry"))
            & pl.col("roll_count").eq(0)
            & pl.col("source_rows").ge(max(1, args.decision_minutes - 1))
            & pl.col("lagged_median_daily_notional").gt(0)
        )
        .with_columns(
            pl.when(pl.col("long_reentry")).then(1.0).otherwise(-1.0).alias("position_direction"),
            (
                args.slippage_ticks_round_trip * pl.col("tick_size") / pl.col("close")
                + pl.when(pl.col("fee_type") == "fixed")
                .then((pl.col("fee_open") + pl.col("fee_close_today")) / (pl.col("close") * pl.col("multiplier")))
                .otherwise(pl.col("fee_open") + pl.col("fee_close_today"))
            ).alias("estimated_round_trip_cost"),
        )
        .with_columns(
            (-pl.col("position_direction") * (pl.col("rsi") - 50.0)).alias("rsi_extremeness"),
            (pl.col("position_direction") * (pl.col("rsi") - pl.col("previous_rsi"))).alias("rsi_turn"),
            (-pl.col("position_direction") * (pl.col("kdj_j") - 50.0)).alias("kdj_extremeness"),
            (pl.col("position_direction") * (pl.col("kdj_k") - pl.col("kdj_d"))).alias("kd_cross_strength"),
            pl.col("previous_bb_z").abs().alias("entry_abs_z"),
            (
                pl.col("datetime").dt.year().cast(pl.String)
                + "-Q"
                + (((pl.col("datetime").dt.month() - 1) // 3) + 1).cast(pl.String)
            ).alias("quarter"),
            *[
                (pl.col("position_direction") * pl.col(f"forward_return_{bars}")).alias(f"gross_reversal_{bars}")
                for bars in HORIZON_BARS
            ],
        )
        .with_columns(
            (
                (pl.col("rsi_extremeness") >= 15) & (pl.col("rsi_turn") > 0)
            ).cast(pl.Int8).alias("rsi_confirm"),
            (
                (pl.col("kdj_extremeness") >= 20) & (pl.col("kd_cross_strength") > 0)
            ).cast(pl.Int8).alias("kdj_confirm"),
            (pl.col("volume_ratio") >= 1.2).cast(pl.Int8).alias("volume_confirm"),
            (
                (pl.col("ker") <= 0.5) & (pl.col("bb_slope_atr") <= 1.0)
            ).cast(pl.Int8).alias("range_confirm"),
            *[
                (pl.col(f"gross_reversal_{bars}") - pl.col("estimated_round_trip_cost")).alias(
                    f"net_reversal_{bars}"
                )
                for bars in HORIZON_BARS
            ],
        )
        .with_columns(
            (
                pl.col("rsi_confirm")
                + pl.col("kdj_confirm")
                + pl.col("volume_confirm")
                + pl.col("range_confirm")
            ).alias("confirmation_count")
        )
        .select(
            "symbol",
            "base",
            "sector",
            "datetime",
            "quarter",
            "position_direction",
            "entry_abs_z",
            "bb_z",
            "rsi",
            "rsi_extremeness",
            "rsi_turn",
            "kdj_k",
            "kdj_d",
            "kdj_j",
            "kdj_extremeness",
            "kd_cross_strength",
            "atr_pct",
            "half_band_atr",
            "bb_slope_atr",
            "ker",
            "volume_ratio",
            "lagged_median_daily_notional",
            "liquidity_rank",
            "estimated_round_trip_cost",
            "rsi_confirm",
            "kdj_confirm",
            "volume_confirm",
            "range_confirm",
            "confirmation_count",
            *[f"gross_reversal_{bars}" for bars in HORIZON_BARS],
            *[f"net_reversal_{bars}" for bars in HORIZON_BARS],
            *[f"forward_datetime_{bars}" for bars in HORIZON_BARS],
        )
    )
    return features.collect(engine="streaming")


def summary(events: pl.DataFrame, decision_minutes: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in events.partition_by(
        ["quarter", "sector", "confirmation_count"], as_dict=True, maintain_order=True
    ).items():
        quarter, sector, confirmations = keys
        for bars in HORIZON_BARS:
            net = group[f"net_reversal_{bars}"].drop_nulls().to_numpy()
            gross = group[f"gross_reversal_{bars}"].drop_nulls().to_numpy()
            std = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
            rows.append(
                {
                    "quarter": quarter,
                    "sector": sector,
                    "confirmation_count": confirmations,
                    "horizon_minutes": bars * decision_minutes,
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
        raise RuntimeError("No indicator reentry events were produced.")
    stem = f"cn_futures_indicator_reentry_{args.label}_{args.start_date}_{args.end_date}"
    event_path = args.output_dir / f"{stem}_events.parquet"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    events.write_parquet(event_path, compression="zstd")
    summary(events, args.decision_minutes).write_csv(summary_path)
    metadata = {
        "data_file": str(args.data_file.resolve()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "decision_minutes": args.decision_minutes,
        "bb_window": args.bb_window,
        "rsi_window": args.rsi_window,
        "kdj_window": args.kdj_window,
        "atr_window": args.atr_window,
        "ker_window": args.ker_window,
        "volume_window": args.volume_window,
        "liquidity_days": args.liquidity_days,
        "min_entry_z": args.min_entry_z,
        "event_count": events.height,
        "event_file": str(event_path.resolve()),
        "summary_file": str(summary_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    print("\nOverall by confirmation count and horizon:")
    print(
        events.group_by("confirmation_count")
        .agg(
            pl.len().alias("events"),
            *[
                pl.col(f"net_reversal_{bars}").mean().mul(10_000).alias(
                    f"net_{bars * args.decision_minutes}m_bps"
                )
                for bars in HORIZON_BARS
            ],
        )
        .sort("confirmation_count")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
