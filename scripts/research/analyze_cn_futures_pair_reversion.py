#!/usr/bin/env python3
"""Event study for risk-balanced relative-value reversal in CN futures pairs."""

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
PAIR_SPECS = (
    ("KQ.m@DCE.m", "KQ.m@CZCE.RM", "protein_meals"),
    ("KQ.m@DCE.y", "KQ.m@CZCE.OI", "vegetable_oils_cross_exchange"),
    ("KQ.m@DCE.p", "KQ.m@DCE.y", "vegetable_oils_dce"),
    ("KQ.m@SHFE.rb", "KQ.m@SHFE.hc", "flat_long_steel"),
    ("KQ.m@INE.bc", "KQ.m@SHFE.cu", "copper_contracts"),
    ("KQ.m@CFFEX.IF", "KQ.m@CFFEX.IH", "large_cap_indices"),
    ("KQ.m@CFFEX.IC", "KQ.m@CFFEX.IM", "small_cap_indices"),
    ("KQ.m@CFFEX.T", "KQ.m@CFFEX.TF", "government_bonds_10y_5y"),
    ("KQ.m@CFFEX.T", "KQ.m@CFFEX.TL", "government_bonds_10y_30y"),
    ("KQ.m@SHFE.au", "KQ.m@SHFE.ag", "precious_metals"),
    ("KQ.m@INE.sc", "KQ.m@SHFE.bu", "crude_bitumen"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--fast-window", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=240)
    parser.add_argument("--spread-window", type=int, default=480)
    parser.add_argument("--regime-window", type=int, default=2400)
    parser.add_argument("--min-shock-z", type=float, default=1.5)
    parser.add_argument("--entry-mode", choices=("immediate", "reentry"), default="immediate")
    parser.add_argument("--reentry-z", type=float, default=3.0)
    parser.add_argument("--pending-minutes", type=int, default=30)
    parser.add_argument("--min-leg-correlation", type=float, default=-1.0)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--slippage-ticks-round-trip", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="development_pair_reversion")
    return parser.parse_args()


def base_symbol(ticker: str) -> str:
    return ticker.rsplit(".", 1)[-1]


def cost_expr(ticker: str, price_col: str, slippage_ticks: float) -> pl.Expr:
    profile = InstrumentMaster("FUTURES_CN").get_profile(base_symbol(ticker))
    slippage = float(slippage_ticks) * float(profile.tick_size) / pl.col(price_col)
    if str(profile.fee_type) == "fixed":
        fees = (float(profile.fee_open) + float(profile.fee_close_today)) / (
            pl.col(price_col) * float(profile.multiplier)
        )
    else:
        fees = pl.lit(float(profile.fee_open) + float(profile.fee_close_today))
    return slippage + fees


def pair_events(
    args: argparse.Namespace,
    ticker_a: str,
    ticker_b: str,
    pair_name: str,
) -> pl.DataFrame:
    start = pl.lit(args.start_date).str.to_datetime("%Y-%m-%d")
    end = pl.lit(args.end_date).str.to_datetime("%Y-%m-%d") + pl.duration(days=1)
    source = (
        pl.scan_parquet(args.data_file)
        .select("symbol", "datetime", "close", "month_change")
        .filter(
            pl.col("symbol").is_in([ticker_a, ticker_b])
            & (pl.col("datetime") >= start)
            & (pl.col("datetime") < end)
        )
    )
    left = source.filter(pl.col("symbol") == ticker_a).select(
        "datetime",
        pl.col("close").alias("close_a"),
        pl.col("month_change").alias("roll_a"),
    )
    right = source.filter(pl.col("symbol") == ticker_b).select(
        "datetime",
        pl.col("close").alias("close_b"),
        pl.col("month_change").alias("roll_b"),
    )
    frame = (
        left.join(right, on="datetime", how="inner")
        .sort("datetime")
        .with_columns(pl.col("datetime").shift(1).alias("prev_datetime"))
        .with_columns(
            (
                pl.col("prev_datetime").is_null()
                | (
                    (pl.col("datetime") - pl.col("prev_datetime")).dt.total_minutes()
                    > args.session_gap_minutes
                )
            ).alias("session_start")
        )
        .with_columns(pl.col("session_start").cum_sum().alias("session_id"))
        .with_columns(
            (pl.col("close_a") / pl.col("close_a").shift(1).over("session_id")).log().alias("ret_a"),
            (pl.col("close_b") / pl.col("close_b").shift(1).over("session_id")).log().alias("ret_b"),
            (pl.col("close_a") / pl.col("close_a").shift(args.fast_window).over("session_id"))
            .log()
            .alias("fast_a"),
            (pl.col("close_b") / pl.col("close_b").shift(args.fast_window).over("session_id"))
            .log()
            .alias("fast_b"),
            (pl.col("roll_a").abs() + pl.col("roll_b").abs())
            .rolling_sum(window_size=args.fast_window + 1, min_samples=1)
            .over("session_id")
            .alias("roll_in_window"),
            *[
                (pl.col("close_a").shift(-h).over("session_id") / pl.col("close_a"))
                .log()
                .alias(f"forward_a_{h}")
                for h in HORIZONS
            ],
            *[
                (pl.col("close_b").shift(-h).over("session_id") / pl.col("close_b"))
                .log()
                .alias(f"forward_b_{h}")
                for h in HORIZONS
            ],
        )
        .with_columns(
            pl.col("ret_a")
            .rolling_std(window_size=args.vol_window, min_samples=max(60, args.vol_window // 4))
            .shift(1)
            .alias("sigma_a"),
            pl.col("ret_b")
            .rolling_std(window_size=args.vol_window, min_samples=max(60, args.vol_window // 4))
            .shift(1)
            .alias("sigma_b"),
        )
        .with_columns(
            (pl.col("sigma_b") / (pl.col("sigma_a") + pl.col("sigma_b"))).alias("weight_a"),
            (pl.col("sigma_a") / (pl.col("sigma_a") + pl.col("sigma_b"))).alias("weight_b"),
        )
        .with_columns(
            (pl.col("weight_a") * pl.col("ret_a") - pl.col("weight_b") * pl.col("ret_b")).alias(
                "spread_ret_1"
            ),
            (pl.col("weight_a") * pl.col("fast_a") - pl.col("weight_b") * pl.col("fast_b")).alias(
                "spread_shock"
            ),
        )
        .with_columns(
            pl.col("spread_ret_1").shift(1).alias("spread_ret_lag_1"),
            (pl.col("ret_a") * pl.col("ret_b")).alias("leg_cross_product"),
            pl.col("ret_a").pow(2).alias("ret_a_sq"),
            pl.col("ret_b").pow(2).alias("ret_b_sq"),
        )
        .with_columns(
            (pl.col("spread_ret_1") * pl.col("spread_ret_lag_1")).alias("spread_lag_product"),
            pl.col("spread_ret_1").pow(2).alias("spread_ret_sq"),
            pl.col("spread_ret_lag_1").pow(2).alias("spread_lag_sq"),
        )
        .with_columns(
            pl.col("spread_ret_1")
            .rolling_std(window_size=args.spread_window, min_samples=max(120, args.spread_window // 4))
            .shift(1)
            .alias("spread_sigma"),
            pl.col("spread_ret_1")
            .rolling_std(window_size=args.regime_window, min_samples=max(480, args.regime_window // 4))
            .shift(1)
            .alias("regime_spread_sigma_1"),
            pl.col("spread_shock")
            .rolling_std(window_size=args.regime_window, min_samples=max(480, args.regime_window // 4))
            .shift(1)
            .alias("regime_spread_sigma_fast"),
            *[
                pl.col(column)
                .rolling_mean(window_size=args.regime_window, min_samples=max(480, args.regime_window // 4))
                .shift(1)
                .alias(f"mean_{column}")
                for column in (
                    "spread_ret_1",
                    "spread_ret_lag_1",
                    "spread_lag_product",
                    "spread_ret_sq",
                    "spread_lag_sq",
                    "ret_a",
                    "ret_b",
                    "leg_cross_product",
                    "ret_a_sq",
                    "ret_b_sq",
                )
            ],
        )
        .with_columns(
            (
                (pl.col("mean_spread_lag_product") - pl.col("mean_spread_ret_1") * pl.col("mean_spread_ret_lag_1"))
                / (
                    (pl.col("mean_spread_ret_sq") - pl.col("mean_spread_ret_1").pow(2)).sqrt()
                    * (pl.col("mean_spread_lag_sq") - pl.col("mean_spread_ret_lag_1").pow(2)).sqrt()
                )
            ).alias("spread_autocorr_1"),
            (
                (pl.col("mean_leg_cross_product") - pl.col("mean_ret_a") * pl.col("mean_ret_b"))
                / (
                    (pl.col("mean_ret_a_sq") - pl.col("mean_ret_a").pow(2)).sqrt()
                    * (pl.col("mean_ret_b_sq") - pl.col("mean_ret_b").pow(2)).sqrt()
                )
            ).alias("leg_correlation"),
            (
                pl.col("regime_spread_sigma_fast").pow(2)
                / (float(args.fast_window) * pl.col("regime_spread_sigma_1").pow(2))
            ).alias("variance_ratio"),
        )
        .with_columns(
            (pl.col("spread_shock") / (pl.col("spread_sigma") * math.sqrt(args.fast_window))).alias("shock_z"),
            (
                pl.col("weight_a") * cost_expr(ticker_a, "close_a", args.slippage_ticks_round_trip)
                + pl.col("weight_b") * cost_expr(ticker_b, "close_b", args.slippage_ticks_round_trip)
            ).alias("estimated_round_trip_cost"),
            (
                pl.col("datetime").dt.year().cast(pl.String)
                + "-Q"
                + (((pl.col("datetime").dt.month() - 1) // 3) + 1).cast(pl.String)
            ).alias("quarter"),
        )
        .with_columns(
            pl.col("shock_z").shift(1).over("session_id").alias("previous_shock_z"),
            pl.col("shock_z")
            .shift(1)
            .rolling_max(window_size=args.pending_minutes, min_samples=1)
            .over("session_id")
            .alias("recent_max_shock_z"),
            pl.col("shock_z")
            .shift(1)
            .rolling_min(window_size=args.pending_minutes, min_samples=1)
            .over("session_id")
            .alias("recent_min_shock_z"),
        )
        .filter(
            (
                pl.col("shock_z").abs().ge(args.min_shock_z)
                if args.entry_mode == "immediate"
                else (
                    pl.col("shock_z").abs().le(args.reentry_z)
                    & pl.col("previous_shock_z").abs().gt(args.reentry_z)
                    & (
                        (pl.col("shock_z").gt(0) & pl.col("recent_max_shock_z").ge(args.min_shock_z))
                        | (pl.col("shock_z").lt(0) & pl.col("recent_min_shock_z").le(-args.min_shock_z))
                    )
                )
            )
            & pl.col("leg_correlation").ge(args.min_leg_correlation)
            & pl.col("roll_in_window").eq(0)
            & pl.col("estimated_round_trip_cost").is_finite()
        )
        .with_columns(
            pl.lit(pair_name).alias("pair"),
            pl.lit(ticker_a).alias("ticker_a"),
            pl.lit(ticker_b).alias("ticker_b"),
            pl.col("shock_z").sign().alias("event_direction"),
            pl.col("shock_z").abs().cut([2.0, 2.5, 3.0, 4.0]).alias("shock_z_bin"),
            *[
                (
                    -pl.col("shock_z").sign()
                    * (
                        pl.col("weight_a") * pl.col(f"forward_a_{h}")
                        - pl.col("weight_b") * pl.col(f"forward_b_{h}")
                    )
                ).alias(f"gross_reversal_{h}")
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
            "pair",
            "ticker_a",
            "ticker_b",
            "datetime",
            "quarter",
            "close_a",
            "close_b",
            "weight_a",
            "weight_b",
            "spread_shock",
            "shock_z",
            "shock_z_bin",
            "event_direction",
            "spread_autocorr_1",
            "leg_correlation",
            "variance_ratio",
            "estimated_round_trip_cost",
            *[f"gross_reversal_{h}" for h in HORIZONS],
            *[f"net_reversal_{h}" for h in HORIZONS],
        )
    )
    return frame.collect(engine="streaming")


def summarize(events: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in events.partition_by(["pair", "quarter", "shock_z_bin"], as_dict=True).items():
        pair, quarter, shock_bin = keys
        for horizon in HORIZONS:
            net = group[f"net_reversal_{horizon}"].drop_nulls().to_numpy()
            gross = group[f"gross_reversal_{horizon}"].drop_nulls().to_numpy()
            std = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
            rows.append(
                {
                    "pair": pair,
                    "quarter": quarter,
                    "shock_z_bin": str(shock_bin),
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
    pair_frames: list[pl.DataFrame] = []
    for ticker_a, ticker_b, pair_name in PAIR_SPECS:
        events = pair_events(args, ticker_a, ticker_b, pair_name)
        print(f"{pair_name}: {events.height:,} events")
        pair_frames.append(events)
    all_events = pl.concat(pair_frames, how="vertical")

    stem = f"cn_futures_pair_reversion_{args.label}_{args.start_date}_{args.end_date}"
    event_path = args.output_dir / f"{stem}_events.parquet"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    all_events.write_parquet(event_path, compression="zstd")
    summary = summarize(all_events)
    summary.write_csv(summary_path)
    metadata = {
        "data_file": str(args.data_file.resolve()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "fast_window": args.fast_window,
        "vol_window": args.vol_window,
        "spread_window": args.spread_window,
        "regime_window": args.regime_window,
        "min_shock_z": args.min_shock_z,
        "entry_mode": args.entry_mode,
        "reentry_z": args.reentry_z,
        "pending_minutes": args.pending_minutes,
        "min_leg_correlation": args.min_leg_correlation,
        "slippage_ticks_round_trip": args.slippage_ticks_round_trip,
        "pairs": [name for _, _, name in PAIR_SPECS],
        "event_count": all_events.height,
        "event_file": str(event_path.resolve()),
        "summary_file": str(summary_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
