"""CLI for publishing immutable FMP and Massive research datasets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from oqp.research.api_datasets import (
    materialize_fmp_us_equity_daily,
    materialize_massive_us_option_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch vendor data into an immutable, quality-checked and "
            "fingerprinted local dataset. API credentials are read from the environment."
        )
    )
    subparsers = parser.add_subparsers(dest="dataset_type", required=True)

    equity = subparsers.add_parser("fmp-equity-daily")
    equity.add_argument("symbols", nargs="+", help="Explicit US equity symbols.")
    equity.add_argument("--start-date")
    equity.add_argument("--end-date")
    equity.add_argument(
        "--adjustment-method",
        choices=("provider_default", "non_split_adjusted", "dividend_adjusted"),
        default="provider_default",
    )
    equity.add_argument("--dataset-id", default="fmp_us_equity_daily")
    equity.add_argument("--point-in-time-universe", action="store_true")
    equity.add_argument("--universe-as-of")
    equity.add_argument("--allow-quality-errors", action="store_true")

    options = subparsers.add_parser("massive-options-snapshot")
    options.add_argument("underlyings", nargs="+", help="Explicit US underlyings.")
    options.add_argument("--expiration")
    options.add_argument("--min-strike", type=float)
    options.add_argument("--max-strike", type=float)
    options.add_argument("--dataset-id", default="massive_us_option_snapshot")
    options.add_argument("--allow-quality-errors", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset_type == "fmp-equity-daily":
        bundle = materialize_fmp_us_equity_daily(
            args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            adjustment_method=args.adjustment_method,
            dataset_id=args.dataset_id,
            point_in_time_universe=args.point_in_time_universe,
            universe_as_of=args.universe_as_of,
            strict_quality=not args.allow_quality_errors,
        )
    else:
        bundle = materialize_massive_us_option_snapshot(
            args.underlyings,
            expiration=args.expiration,
            min_strike=args.min_strike,
            max_strike=args.max_strike,
            dataset_id=args.dataset_id,
            strict_quality=not args.allow_quality_errors,
        )

    print(f"dataset_id={bundle.dataset_id}")
    print(f"dataset_version={bundle.dataset_version}")
    print(f"dataset_fingerprint={bundle.dataset_fingerprint}")
    print(f"historical_backtest_eligible={bundle.historical_backtest_eligible}")
    print(f"descriptor={bundle.descriptor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
