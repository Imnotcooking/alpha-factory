"""Dispatch a typed strategy draft to the existing research backtest engine."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import time

import pandas as pd

from oqp.execution.transaction_costs import attach_transaction_cost_policy
from oqp.research.backtesting import (
    RETURN_HORIZON_AUTO,
    attach_capital_attrs,
    normalize_return_horizon,
    resolve_execution_capital,
)
from oqp.research.factor_portfolios import (
    FactorPortfolioConfig,
    FactorPortfolioRunner,
    FactorSpec,
    RouterSpec,
    SleeveSpec,
    StrategyRiskOverlaySpec,
    load_factor_portfolio_data,
    load_router_state_data,
)
from oqp.research.factors import load_factor_module
from oqp.research.strategy_composition import (
    StrategyBuilderConfig,
    StrategyCoreType,
    load_strategy_builder_config,
    strategy_execution_support,
)
from oqp.research_runtime import alpha_research_runtime_paths


MARGIN_BUDGET_GROSS_SAFETY_CEILING = 100.0
CONTRACT_TIMED_REUSABLE_CORE_TYPES = {
    StrategyCoreType.FACTOR_SLEEVE,
    StrategyCoreType.ML_PREDICTIVE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and backtest one typed research strategy draft."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--strict-factor-contracts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _factor_portfolio_config(config: StrategyBuilderConfig) -> FactorPortfolioConfig:
    core = config.core
    overlays = tuple(
        StrategyRiskOverlaySpec(overlay_id=value) for value in config.risk_overlays
    )
    common = {
        "strategy_id": config.strategy_id,
        "name": config.name,
        "market_vertical": config.market_vertical,
        "risk_overlays": overlays,
        "max_gross_leverage": (
            config.allocator.max_gross_leverage
            if config.allocator.max_gross_leverage is not None
            else MARGIN_BUDGET_GROSS_SAFETY_CEILING
        ),
        "max_weight_per_asset": config.allocator.max_contract_weight,
        "max_margin_utilization": config.allocator.max_margin_utilization,
        "return_horizon": "auto",
        "description": (
            f"Typed {core.core_type.value} strategy draft; "
            f"fingerprint={config.fingerprint}."
        ),
    }
    if core.core_type == StrategyCoreType.ROUTED_COMPONENTS:
        sleeves = tuple(
            SleeveSpec(
                sleeve_id=branch.branch_id,
                factors=tuple(FactorSpec(factor_id=value) for value in branch.factor_ids),
                weighting_method="equal",
                normalization=(
                    "raw" if branch.execution_mode in {"direct", "statarb"}
                    else "cross_sectional_zscore"
                ),
            )
            for branch in core.branches
        )
        return FactorPortfolioConfig(
            **common,
            sleeves=sleeves,
            router=RouterSpec(
                router_id=str(core.router_id),
                state_file=core.router_state_file,
            ),
            execution_mode="risk_desk",
            neutralize=False,
        )

    branch = core.branches[0]
    factors = tuple(FactorSpec(factor_id=value) for value in branch.factor_ids)
    return FactorPortfolioConfig(
        **common,
        factors=factors,
        weighting_method="equal",
        normalization=(
            "raw" if branch.execution_mode in {"direct", "statarb"}
            else "cross_sectional_zscore"
        ),
        execution_mode=branch.execution_mode,
        neutralize=branch.execution_mode == "risk_desk",
    )


def _attach_ml_predictions(
    frame: pd.DataFrame,
    config: StrategyBuilderConfig,
) -> pd.DataFrame:
    path = Path(str(config.core.ml_predictions_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"registered ML predictions do not exist: {path}")
    predictions = pd.read_parquet(path)
    required = {"date", "ticker", "prediction"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"ML prediction artifact is missing columns: {missing}")
    prediction_frame = predictions[["date", "ticker", "prediction"]].copy()
    prediction_frame["date"] = pd.to_datetime(prediction_frame["date"], errors="raise")
    prediction_frame["ticker"] = prediction_frame["ticker"].astype(str)
    if prediction_frame.duplicated(["date", "ticker"]).any():
        raise ValueError("ML prediction artifact contains duplicate date/ticker rows")
    out = frame.merge(
        prediction_frame.rename(columns={"prediction": "ml_prediction"}),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    out.attrs.update(frame.attrs)
    out.attrs["ml_experiment_id"] = config.core.ml_experiment_id
    out.attrs["ml_predictions_path"] = str(path)
    return out


def _resolve_data_return_horizon(config: StrategyBuilderConfig) -> str:
    """Resolve the data clock before loading a reusable-sleeve strategy.

    A registered ML prediction artifact replaces the factor's score source,
    not its execution clock, so ML and non-ML reusable sleeves follow the same
    explicit factor-contract rule here.
    """

    core = config.core
    if core.core_type not in CONTRACT_TIMED_REUSABLE_CORE_TYPES:
        return RETURN_HORIZON_AUTO
    branch = core.branches[0]
    if branch.sleeve_id is None:
        return RETURN_HORIZON_AUTO

    factor_id = branch.factor_ids[0]
    try:
        factor_module = load_factor_module(factor_id)
    except (ImportError, ModuleNotFoundError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot resolve the return horizon for {factor_id}: "
            "the registered factor module could not be loaded."
        ) from exc

    contract = getattr(factor_module, "FACTOR_CONTRACT", None)
    if not isinstance(contract, Mapping):
        raise ValueError(
            f"Cannot resolve the return horizon for {factor_id}: "
            "reusable-sleeve strategies require an explicit FACTOR_CONTRACT."
        )
    declared = str(contract.get("return_assumption") or "").strip()
    if not declared:
        raise ValueError(
            f"Cannot resolve the return horizon for {factor_id}: "
            "FACTOR_CONTRACT.return_assumption is required."
        )
    try:
        horizon = normalize_return_horizon(declared)
    except ValueError as exc:
        raise ValueError(
            f"Cannot resolve the return horizon for {factor_id}: "
            f"FACTOR_CONTRACT.return_assumption={declared!r} is not a "
            "loadable market-data horizon."
        ) from exc
    if horizon == RETURN_HORIZON_AUTO:
        raise ValueError(
            f"Cannot resolve the return horizon for {factor_id}: "
            "FACTOR_CONTRACT.return_assumption must be explicit, not 'auto'."
        )
    return horizon


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    config = load_strategy_builder_config(args.config)
    support = strategy_execution_support(config)
    if not support.runnable:
        raise RuntimeError(f"Strategy is not executable yet: {support.reason}")

    uses_reusable_sleeve = (
        config.core.core_type in CONTRACT_TIMED_REUSABLE_CORE_TYPES
        and config.core.branches[0].sleeve_id is not None
    )
    data_return_horizon = _resolve_data_return_horizon(config)
    data = load_factor_portfolio_data(
        args.data_file,
        market_vertical=config.market_vertical,
        return_horizon=data_return_horizon,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    frame = data.frame
    if config.core.core_type == StrategyCoreType.ML_PREDICTIVE:
        frame = _attach_ml_predictions(frame, config)
    frame = attach_capital_attrs(
        frame,
        resolve_execution_capital(
            asset_class=config.market_vertical,
            initial_capital=config.execution.capital,
            capital_currency=config.execution.capital_currency,
            capital_profile="strategy_builder",
        ),
    )
    frame = attach_transaction_cost_policy(
        frame,
        market_vertical=config.market_vertical,
        profile_id=config.execution.transaction_cost_profile,
        use_case="research_net",
    )

    portfolio_config = _factor_portfolio_config(config)
    runner = FactorPortfolioRunner(portfolio_config)
    router_states = None
    if config.core.core_type == StrategyCoreType.ROUTED_COMPONENTS:
        router_states = load_router_state_data(str(config.core.router_state_file))
    if uses_reusable_sleeve:
        branch = config.core.branches[0]
        result = runner.build_with_sleeve(
            frame,
            factor_id=branch.factor_ids[0],
            sleeve_id=str(branch.sleeve_id),
            strict_factor_contracts=args.strict_factor_contracts,
        )
    else:
        result = runner.build(
            frame,
            strict_factor_contracts=args.strict_factor_contracts,
            router_states=router_states,
        )
    result.frame.attrs.update(
        {
            "component_type": "strategy",
            "strategy_id": config.strategy_id,
            "strategy_core_type": config.core.core_type.value,
            "strategy_config_fingerprint": config.fingerprint,
            "backtest_engine": "typed_strategy_composition",
            "runner": "strategy_backtest",
        }
    )
    print(f"Strategy: {config.strategy_id} ({config.core.core_type.value})")
    print(f"Backend: {support.backend}")
    print(f"Rows: {len(result.frame):,}")
    print(f"Execution: {result.execution_detail}")
    target_col = next(
        (
            column
            for column in (
                "final_target_weight",
                "routed_target_weight",
                "target_weight",
                "signal",
            )
            if column in result.frame.columns
        ),
        None,
    )
    has_active_targets: bool | None = None
    if target_col is not None:
        targets = pd.to_numeric(result.frame[target_col], errors="coerce").fillna(0.0)
        active = targets.ne(0.0)
        has_active_targets = bool(active.any())
        active_dates = int(result.frame.loc[active, "date"].nunique())
        gross_by_date = targets.abs().groupby(result.frame["date"]).sum()
        print(
            "Targets: "
            f"{int(active.sum()):,} active rows across {active_dates:,} dates; "
            f"mean gross={float(gross_by_date.mean()):.2%}"
        )
        if (
            config.allocator.max_margin_utilization is not None
            and "margin_utilization" in result.frame.columns
        ):
            margin = pd.to_numeric(
                result.frame["margin_utilization"], errors="coerce"
            ).groupby(result.frame["date"]).first()
            print(
                "Margin budget: "
                f"mean={float(margin.mean()):.2%}; "
                f"maximum={float(margin.max()):.2%}; "
                f"limit={config.allocator.max_margin_utilization:.2%}"
            )
        if not has_active_targets:
            print(
                "WARNING: the selected components and data produced no active "
                "positions; this is not a tradable backtest."
            )
    if args.build_only:
        print(f"Build-only validation complete in {time.time() - started:.2f}s")
        return 0
    if has_active_targets is False:
        raise RuntimeError(
            "Backtest stopped because the strategy produced no active targets. "
            "Use build-only diagnostics to correct the universe, liquidity, or "
            "component eligibility first."
        )

    runtime = alpha_research_runtime_paths()
    run_id = runner.evaluate(
        result,
        db_path=runtime.db_path,
        logs_dir=runtime.artifact_root,
        crisis_period=data.crisis_period,
        split_date=args.split_date,
    )
    print(f"Backtest complete: run_id={run_id} in {time.time() - started:.2f}s")
    return 0


__all__ = ["build_parser", "main"]
