"""CLI for composing several research factors into one backtested strategy."""

from __future__ import annotations

import argparse
import time

from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical
from oqp.research.factor_portfolios import (
    FactorPortfolioRunner,
    load_factor_portfolio_data,
    load_factor_portfolio_config,
    load_router_state_data,
)
from oqp.research_runtime import alpha_research_runtime_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose multiple factor scores and backtest one integrated strategy."
    )
    parser.add_argument("--config", required=True, help="Factor portfolio YAML file.")
    parser.add_argument(
        "--data-file",
        required=True,
        help=(
            "Long-format parquet/CSV input or an immutable API materialization "
            "directory containing materialization.json."
        ),
    )
    parser.add_argument(
        "--router-state-file",
        default=None,
        help="Optional CSV/parquet state input overriding router.state_file.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument(
        "--split-mode",
        default="auto",
        choices=("auto", "date", "ratio"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.70)
    parser.add_argument("--purge-periods", type=int, default=0)
    parser.add_argument("--embargo-periods", type=int, default=0)
    parser.add_argument("--purge-unit", default="auto")
    parser.add_argument(
        "--strict-factor-contracts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Validate and compose signals without writing a backtest run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    config = load_factor_portfolio_config(args.config)
    market_vertical = normalize_market_vertical(config.market_vertical)
    if market_vertical not in ASSET_TAXONOMY:
        raise ValueError(f"unsupported market vertical: {market_vertical}")
    if market_vertical.startswith("OPTIONS_"):
        raise ValueError(
            "factor-level option portfolios are not yet supported by this tabular route; "
            "compose option strategies as event-driven sleeves instead"
        )

    print(f"Building factor portfolio: {config.strategy_id} ({config.name})")
    print(f"Market: {market_vertical}")
    if config.factors:
        for spec in config.factors:
            weight = config.normalized_weights[spec.factor_id]
            print(
                f"  - {spec.factor_id}: weight={weight:.2%}, "
                f"orientation={spec.orientation:+d}"
            )
    else:
        for sleeve in config.sleeves:
            factor_ids = ", ".join(spec.factor_id for spec in sleeve.factors)
            print(f"  - sleeve {sleeve.sleeve_id}: {factor_ids}")
        print(f"  - router: {config.router.router_id}")

    data = load_factor_portfolio_data(
        args.data_file,
        market_vertical=market_vertical,
        return_horizon=config.return_horizon,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    runner = FactorPortfolioRunner(config)
    router_states = None
    if config.router is not None:
        state_file = args.router_state_file or config.router.state_file
        if not state_file:
            raise ValueError(
                "routed strategy requires --router-state-file or router.state_file"
            )
        router_states = load_router_state_data(state_file)
    result = runner.build(
        data.frame,
        strict_factor_contracts=args.strict_factor_contracts,
        router_states=router_states,
    )
    print(
        f"Composite rows={len(result.frame):,}, "
        f"valid_scores={result.frame['composite_score'].notna().sum():,}"
    )
    print(f"Execution: {result.execution_detail}")
    if args.build_only:
        print(f"Build-only validation complete in {time.time() - started:.2f}s")
        return 0

    runtime = alpha_research_runtime_paths()
    run_id = runner.evaluate(
        result,
        db_path=runtime.db_path,
        logs_dir=runtime.artifact_root,
        crisis_period=data.crisis_period,
        split_date=args.split_date,
        split_mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        purge_periods=args.purge_periods,
        embargo_periods=args.embargo_periods,
        purge_unit=args.purge_unit,
    )
    print(f"Backtest complete: run_id={run_id} in {time.time() - started:.2f}s")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
