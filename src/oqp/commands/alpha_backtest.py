from __future__ import annotations

import os
import argparse
import sqlite3
import time
from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical
from oqp.research_runtime import alpha_research_runtime_paths


ALPHA_RUNTIME_PATHS = alpha_research_runtime_paths()
ALPHA_RUNTIME_ARTIFACT_ROOT = ALPHA_RUNTIME_PATHS.artifact_root
ALPHA_RESEARCH_DB_PATH = ALPHA_RUNTIME_PATHS.db_path
DEFAULT_DATA_FILE = ALPHA_RUNTIME_PATHS.default_daily_data_file

def get_factor_metadata(factor_module, factor_module_name: str) -> tuple[str, str, str, str, int, str | None]:
    factor_id = getattr(factor_module, "FACTOR_ID", factor_module_name)
    name = getattr(factor_module, "NAME_EN", getattr(factor_module, "FACTOR_NAME", factor_id))
    category = getattr(factor_module, "CATEGORY", "Uncategorized")
    economic_rationale = getattr(
        factor_module,
        "ECONOMIC_RATIONALE_EN",
        getattr(factor_module, "ECONOMIC_RATIONALE", ""),
    )
    complexity = getattr(factor_module, "COMPLEXITY", 0)
    geometry = getattr(
        factor_module,
        "EVALUATION_GEOMETRY",
        getattr(factor_module, "STRATEGY_GEOMETRY", None),
    )
    return factor_id, name, category, economic_rationale, complexity, geometry


def infer_data_frequency(df: pd.DataFrame) -> str:
    from oqp.research.backtesting import infer_frame_frequency

    return infer_frame_frequency(df)


def apply_vertical_attrs(df: pd.DataFrame, attrs: dict[str, str]) -> pd.DataFrame:
    for key, value in attrs.items():
        if value and key not in df.attrs:
            df.attrs[key] = value
    return df


def build_execution_mode_config(factor_module, args, execution_weight_col: str | None = None) -> ExecutionModeConfig:
    from oqp.research.backtesting import ExecutionModeConfig

    raw_config = getattr(factor_module, "EXECUTION_MODE_CONFIG", {}) or {}
    if not isinstance(raw_config, dict):
        print("⚠️ WARNING: EXECUTION_MODE_CONFIG must be a dict. Ignoring it.")
        raw_config = {}
    config_data = dict(raw_config)

    if execution_weight_col:
        config_data["source_col"] = execution_weight_col
    if args.max_gross_leverage is not None:
        config_data["max_gross_leverage"] = args.max_gross_leverage
    if args.max_weight_per_asset is not None:
        config_data["max_weight_per_asset"] = args.max_weight_per_asset
    if getattr(args, "sizing_modules", None) is not None:
        config_data["sizing_modules"] = args.sizing_modules
    if getattr(args, "kelly_fraction", None) is not None:
        config_data["kelly_fraction"] = args.kelly_fraction

    allowed = set(ExecutionModeConfig.__dataclass_fields__)
    unknown = sorted(set(config_data) - allowed)
    if unknown:
        print(f"⚠️ WARNING: Ignoring unknown execution mode config keys: {unknown}")
    filtered = {key: value for key, value in config_data.items() if key in allowed}
    return ExecutionModeConfig(**filtered)


def resolve_execution_mode(factor_module, df: pd.DataFrame, requested_mode: str) -> str:
    if requested_mode and requested_mode != "auto":
        return requested_mode
    return (
        getattr(factor_module, "EXECUTION_MODE", None)
        or df.attrs.get("execution_mode")
        or "risk_desk"
    )


def is_options_market(asset: str) -> bool:
    return normalize_market_vertical(asset) in {"OPTIONS_US", "OPTIONS_CN"}


def load_tabular_file(path: str) -> pd.DataFrame:
    import pandas as pd

    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular file type: {suffix}")


def normalize_underlying_input(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    out = df.copy()
    if "ticker" not in out.columns:
        if "underlying_symbol" in out.columns:
            out["ticker"] = out["underlying_symbol"]
        elif "symbol" in out.columns:
            out["ticker"] = out["symbol"]
    if "underlying_symbol" not in out.columns and "ticker" in out.columns:
        out["underlying_symbol"] = out["ticker"]
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = out["datetime"]
    if "close" not in out.columns:
        for candidate in ("settlement", "last_price", "mark", "price"):
            if candidate in out.columns:
                out["close"] = out[candidate]
                break
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.upper()
    if "underlying_symbol" in out.columns:
        out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.upper()
    return out


def build_option_backtest_signals(
    factor_df: pd.DataFrame,
    *,
    contract,
    contracts_per_signal: int,
    target_moneyness: float,
    min_dte: int,
    max_dte: int,
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = factor_df.copy()
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = out["datetime"]
    if "underlying_symbol" not in out.columns:
        if "ticker" in out.columns:
            out["underlying_symbol"] = out["ticker"]
        elif "symbol" in out.columns:
            out["underlying_symbol"] = out["symbol"]
    if "date" not in out.columns or "underlying_symbol" not in out.columns:
        raise ValueError("Option factor output requires date plus underlying_symbol, ticker, or symbol.")

    direction_col = next(
        (
            col
            for col in ("direction", "signal_direction", "option_direction")
            if col in out.columns
        ),
        None,
    )
    if direction_col is not None:
        raw_direction = pd.to_numeric(out[direction_col], errors="coerce").fillna(0.0)
    else:
        signal_col = contract.execution_weight_col or contract.alpha_signal_col
        if signal_col not in out.columns:
            raise ValueError(f"Option factor output missing signal column {signal_col!r}.")
        raw_direction = pd.to_numeric(out[signal_col], errors="coerce").fillna(0.0)

    out["direction"] = np.sign(raw_direction).astype(float)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.upper()
    out = out.dropna(subset=["date", "underlying_symbol"]).sort_values(["underlying_symbol", "date"])

    event_rows: list[dict[str, object]] = []
    passthrough_cols = [
        "option_symbol",
        "right",
        "contracts",
        "target_moneyness",
        "min_dte",
        "max_dte",
    ]
    for _, group in out.groupby("underlying_symbol", sort=True):
        previous_direction = 0.0
        for _, row in group.iterrows():
            desired_direction = float(row["direction"])
            base = {
                "date": row["date"],
                "underlying_symbol": row["underlying_symbol"],
                "contracts": float(row.get("contracts", contracts_per_signal) or contracts_per_signal),
                "target_moneyness": float(row.get("target_moneyness", target_moneyness) or target_moneyness),
                "min_dte": int(row.get("min_dte", min_dte) or min_dte),
                "max_dte": int(row.get("max_dte", max_dte) or max_dte),
            }
            for col in passthrough_cols:
                if col in row and col not in base and not pd.isna(row[col]):
                    base[col] = row[col]
            if previous_direction != 0.0 and desired_direction != previous_direction:
                event_rows.append({**base, "direction": 0.0})
            if desired_direction != 0.0 and desired_direction != previous_direction:
                event_rows.append({**base, "direction": desired_direction})
            previous_direction = desired_direction

    signals = pd.DataFrame(event_rows)
    if signals.empty:
        return pd.DataFrame(columns=["date", "underlying_symbol", "direction"])
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.date
    return signals.dropna(subset=["date", "underlying_symbol"]).reset_index(drop=True)


def run_options_backtest_route(
    *,
    args,
    factor_module,
    factor_id: str,
    factor_name: str,
    factor_category: str,
    factor_rationale: str,
    factor_complexity: int,
    capital_profile,
    trade_policy,
    start_time: float,
) -> None:
    from oqp.options import (
        OptionBacktestConfig,
        OptionBacktestEngine,
        OptionBacktestRequest,
        OptionLiquidityRule,
        OptionMarginPolicy,
        load_option_chain_file,
    )
    from oqp.research import attach_factor_contract_attrs, resolve_factor_contract
    from oqp.research.backtesting import AlphaEvaluator
    from oqp.research.parameter_schema import attach_factor_parameter_attrs

    if not args.option_chain_file:
        print("❌ ERROR: Option backtests require --option_chain_file.")
        return
    underlying_path = args.option_underlying_file or args.data_file
    if not underlying_path:
        print("❌ ERROR: Option backtests require --option_underlying_file or --data_file for underlying bars.")
        return
    if not os.path.exists(args.option_chain_file):
        print(f"❌ ERROR: Option chain file {args.option_chain_file} not found.")
        return
    if not os.path.exists(underlying_path):
        print(f"❌ ERROR: Underlying file {underlying_path} not found.")
        return

    print("   ⚡ Loading option chain and underlying bars...")
    try:
        chain = load_option_chain_file(
            args.option_chain_file,
            market_vertical=args.asset,
            default_multiplier=args.option_default_multiplier,
        )
        underlying = normalize_underlying_input(load_tabular_file(underlying_path))
    except Exception as e:
        print(f"❌ [OPTIONS DATA ERROR] {e}")
        return

    if hasattr(factor_module, "prepare_data"):
        try:
            print("   -> Applying factor data adapter to underlying bars...")
            underlying = factor_module.prepare_data(underlying)
            underlying = normalize_underlying_input(underlying)
        except Exception as e:
            print(f"❌ [FACTOR DATA ADAPTER ERROR] {e}")
            return

    if underlying.empty or chain.empty:
        print("❌ ERROR: Option chain and underlying data must both be non-empty.")
        return

    print(f"   ✅ Option chain rows: {len(chain):,}; underlying rows: {len(underlying):,}")
    print("   -> Computing option-direction factor scores...")
    try:
        factor_df = factor_module.compute(underlying.copy())
        factor_df = normalize_underlying_input(factor_df)
        factor_df = attach_factor_parameter_attrs(factor_df, factor_module)
    except Exception as e:
        print(f"❌ [FACTOR COMPUTE ERROR] {e}")
        return

    factor_df.attrs.update(
        {
            "market_vertical": args.asset,
            "source_path": os.path.abspath(args.option_chain_file),
            "dataset_id": os.path.splitext(os.path.basename(args.option_chain_file))[0],
            "data_file": os.path.abspath(underlying_path),
            "data_vendor": "local_option_chain",
            "data_frequency": "daily",
            "dataset_role": "option_chain",
            "data_tradability": "executable_option_chain",
            "data_price_source": "option_bid_ask_or_mark",
            "data_roll_model": "contract_selection_by_signal",
            "data_liquidity_model": "option_bid_ask_volume_oi_filter",
            "data_execution_reality": "event_driven_options_v1",
            "execution_assumption": "daily_signal_option_fill_then_mark",
            "initial_capital": capital_profile.initial_capital,
            "capital_currency": capital_profile.currency,
            "capital_profile": capital_profile.profile,
            "capital_source": capital_profile.source,
            "min_trade_weight_delta": trade_policy.min_trade_weight_delta,
            "min_trade_weight_delta_source": trade_policy.source,
        }
    )

    try:
        contract = resolve_factor_contract(
            factor_module,
            factor_df,
            factor_id=factor_id,
            requested_execution_mode=args.execution_mode,
            default_return_assumption="custom_forward_return",
            market_vertical=args.asset,
            strict=args.strict_factor_contract,
        )
    except Exception as e:
        print(f"❌ [FACTOR CONTRACT ERROR] {e}")
        return
    factor_df = attach_factor_contract_attrs(factor_df, contract)
    if contract.contract_source != "explicit":
        print("   ⚠️ Factor Contract: inferred legacy contract. Add FACTOR_CONTRACT for strict mode.")
        for warning in contract.warnings[:4]:
            print(f"      - {warning}")
    print(
        "   -> 📜 Option Factor Contract: "
        f"alpha={contract.alpha_signal_col}, "
        f"weight={contract.execution_weight_col}, "
        f"supported={', '.join(contract.supported_markets)}"
    )

    try:
        signals = build_option_backtest_signals(
            factor_df,
            contract=contract,
            contracts_per_signal=args.option_contracts_per_signal,
            target_moneyness=args.option_target_moneyness,
            min_dte=args.option_min_dte,
            max_dte=args.option_max_dte,
        )
    except Exception as e:
        print(f"❌ [OPTION SIGNAL ERROR] {e}")
        return
    if signals.empty:
        print("❌ ERROR: Option factor produced no entry/exit events after signal normalization.")
        return

    print(f"   -> Event-driven option signals: {len(signals):,}")
    request = OptionBacktestRequest(
        chain=chain,
        underlying=underlying,
        signals=signals,
        market_vertical=args.asset,
        initial_capital=capital_profile.initial_capital,
        config=OptionBacktestConfig(
            min_dte=args.option_min_dte,
            max_dte=args.option_max_dte,
            target_moneyness=args.option_target_moneyness,
            contracts_per_signal=args.option_contracts_per_signal,
        ),
        liquidity=OptionLiquidityRule(
            min_volume=args.option_min_volume,
            min_open_interest=args.option_min_open_interest,
            max_spread_pct=args.option_max_spread_pct,
            allow_settlement_proxy=args.option_allow_settlement_proxy,
        ),
        margin=OptionMarginPolicy(
            commission_per_contract=args.option_commission_per_contract,
        ),
        metadata={
            "factor_id": factor_id,
            "option_chain_file": os.path.abspath(args.option_chain_file),
            "underlying_file": os.path.abspath(underlying_path),
            "option_min_dte": args.option_min_dte,
            "option_max_dte": args.option_max_dte,
            "option_target_moneyness": args.option_target_moneyness,
            "option_contracts_per_signal": args.option_contracts_per_signal,
            "option_min_volume": args.option_min_volume,
            "option_min_open_interest": args.option_min_open_interest,
            "option_max_spread_pct": args.option_max_spread_pct,
            "option_allow_settlement_proxy": args.option_allow_settlement_proxy,
            "option_commission_per_contract": args.option_commission_per_contract,
        },
    )

    print("   -> Running Options Event-Driven Engine...")
    try:
        result = OptionBacktestEngine().run(request)
    except Exception as e:
        print(f"❌ [OPTIONS ENGINE ERROR] {e}")
        return

    evaluator = AlphaEvaluator(
        db_path=ALPHA_RESEARCH_DB_PATH,
        logs_dir=ALPHA_RUNTIME_ARTIFACT_ROOT,
        asset_class=args.asset,
    )
    run_id = evaluator.log_option_backtest_result(
        factor_id=factor_id,
        result=result,
        factor_name=factor_name,
        factor_category=factor_category,
        factor_rationale=factor_rationale,
        factor_complexity=factor_complexity,
        factor_contract=contract.to_attrs(),
        option_chain_path=os.path.abspath(args.option_chain_file),
        underlying_path=os.path.abspath(underlying_path),
        initial_capital=capital_profile.initial_capital,
        capital_currency=capital_profile.currency,
        metadata={
            "market_vertical": args.asset,
            "dataset_id": os.path.splitext(os.path.basename(args.option_chain_file))[0],
            "data_vendor": "local_option_chain",
            "capital_profile": capital_profile.profile,
            "capital_source": capital_profile.source,
            "option_min_dte": args.option_min_dte,
            "option_max_dte": args.option_max_dte,
            "option_target_moneyness": args.option_target_moneyness,
            "option_contracts_per_signal": args.option_contracts_per_signal,
            "option_min_volume": args.option_min_volume,
            "option_min_open_interest": args.option_min_open_interest,
            "option_max_spread_pct": args.option_max_spread_pct,
            "option_allow_settlement_proxy": args.option_allow_settlement_proxy,
            "option_commission_per_contract": args.option_commission_per_contract,
            "universe_size": int(factor_df["underlying_symbol"].nunique())
            if "underlying_symbol" in factor_df.columns
            else 0,
        },
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Options Backtest Complete in {elapsed:.2f} seconds. run_id={run_id}")


def main():
    # 1. Setup the Terminal Command Arguments
    parser = argparse.ArgumentParser(description="Alpha Mine Backtest Engine")
    parser.add_argument("--factor", type=str, required=True, help="Name of the factor module (e.g., fac_003_MA_Crossover)")
    
    # 2. 🛠️ UPGRADE: Use the sophisticated Asset Taxonomy keys dynamically
    parser.add_argument(
        "--asset",
        type=str,
        default='FUTURES_CN',
        help=(
            "Asset taxonomy class or alias "
            f"(known: {', '.join(sorted(ASSET_TAXONOMY))})"
        ),
    )
    
    parser.add_argument("--data_file", type=str, default=str(DEFAULT_DATA_FILE), help="Path to the internal parquet file")
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Inclusive backtest start timestamp/date applied after factor data preparation.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Inclusive backtest end timestamp/date applied after factor data preparation.",
    )
    parser.add_argument("--split_mode", choices=["auto", "date", "ratio"], default="auto", help="Evaluation split policy.")
    parser.add_argument("--split_date", type=str, default="2023-01-01", help="Calendar split date for date/auto mode.")
    parser.add_argument("--validation_fraction", type=float, default=0.60, help="Chronological validation fraction for ratio fallback.")
    parser.add_argument("--purge_periods", type=int, default=None, help="Drop this many periods from the end of validation before holdout.")
    parser.add_argument("--embargo_periods", type=int, default=None, help="Drop this many periods from the start of holdout after validation.")
    parser.add_argument(
        "--purge_unit",
        choices=["auto", "days", "timestamps", "rows"],
        default=None,
        help="Unit for purge/embargo periods. Auto uses days for daily data and timestamps for intraday/tick data.",
    )
    parser.add_argument(
        "--execution_mode",
        choices=["auto", "risk_desk", "direct", "statarb"],
        default="auto",
        help="How factor scores become executable target weights.",
    )
    parser.add_argument("--max_gross_leverage", type=float, default=None, help="Override execution-mode gross leverage cap.")
    parser.add_argument("--max_weight_per_asset", type=float, default=None, help="Optional per-asset cap before gross scaling.")
    parser.add_argument(
        "--sizing_modules",
        type=str,
        default=None,
        help=(
            "Risk-desk allocation modules for heuristic factors. "
            "Examples: kelly,hrp, kelly, hrp, or none (default)."
        ),
    )
    parser.add_argument(
        "--kelly_fraction",
        type=float,
        default=None,
        help="Override fractional Kelly multiplier when the kelly sizing module is enabled.",
    )
    parser.add_argument(
        "--lot_mode",
        choices=["auto", "integer", "fractional"],
        default="auto",
        help="Contract sizing mode. Auto uses integer lots for futures and fractional sizing elsewhere.",
    )
    parser.add_argument(
        "--return_horizon",
        type=str,
        default="auto",
        help=(
            "Daily signal-to-mark return horizon. Examples: auto, "
            "next_open_to_next_close, next_open_to_next_open, "
            "close_to_next_close, close_to_next_open."
        ),
    )
    parser.add_argument(
        "--initial_capital",
        type=float,
        default=None,
        help="Override starting capital for execution simulation.",
    )
    parser.add_argument(
        "--capital_currency",
        type=str,
        default=None,
        help="Currency label for starting capital, e.g. CNY, USD, HKD.",
    )
    parser.add_argument(
        "--capital_profile",
        type=str,
        default=None,
        help="Named capital profile, e.g. small_personal_futures_cn or institutional_equity_us.",
    )
    parser.add_argument(
        "--min_trade_weight_delta",
        type=float,
        default=None,
        help="Execution dust tolerance. Defaults to 1e-8; factor thresholds should live inside factor params.",
    )
    parser.add_argument(
        "--strict_factor_contract",
        dest="strict_factor_contract",
        action="store_true",
        default=True,
        help="Require FACTOR_CONTRACT instead of allowing legacy inference. Enabled by default.",
    )
    parser.add_argument(
        "--allow_legacy_factor_contract",
        dest="strict_factor_contract",
        action="store_false",
        help="Allow legacy factor-contract inference for old research files.",
    )
    parser.add_argument(
        "--option_chain_file",
        type=str,
        default=None,
        help="Option-chain CSV/parquet for OPTIONS_US or OPTIONS_CN backtests.",
    )
    parser.add_argument(
        "--option_underlying_file",
        type=str,
        default=None,
        help="Underlying OHLC/settlement CSV/parquet used by the option factor.",
    )
    parser.add_argument("--option_min_dte", type=int, default=7, help="Minimum days to expiry for auto-selected options.")
    parser.add_argument("--option_max_dte", type=int, default=60, help="Maximum days to expiry for auto-selected options.")
    parser.add_argument("--option_contracts_per_signal", type=int, default=1, help="Contracts to trade per entry signal.")
    parser.add_argument("--option_target_moneyness", type=float, default=1.0, help="Target strike/spot ratio for auto-selection.")
    parser.add_argument("--option_min_volume", type=float, default=0.0, help="Minimum option-chain volume.")
    parser.add_argument("--option_min_open_interest", type=float, default=0.0, help="Minimum option-chain open interest.")
    parser.add_argument("--option_max_spread_pct", type=float, default=0.25, help="Maximum bid/ask spread as a fraction of mid.")
    parser.add_argument(
        "--option_allow_settlement_proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow mark/close settlement proxies when bid/ask is unavailable.",
    )
    parser.add_argument("--option_commission_per_contract", type=float, default=0.65, help="Commission per option contract.")
    parser.add_argument("--option_default_multiplier", type=float, default=None, help="Fallback contract multiplier.")
    
    args = parser.parse_args()
    args.asset = normalize_market_vertical(args.asset)
    if args.asset not in ASSET_TAXONOMY:
        parser.error(f"unsupported asset taxonomy class: {args.asset}")

    import pandas as pd
    from oqp.data import DataEngineFactory
    from oqp.research import attach_dataset_tradability_attrs, infer_dataset_tradability
    from oqp.research import (
        attach_factor_contract_attrs,
        resolve_factor_contract,
        validate_factor_market_compatibility,
    )
    from oqp.research.backtesting import AlphaEvaluator
    from oqp.research.backtesting import (
        ExecutionModeFactory,
        attach_capital_attrs,
        attach_trade_policy_attrs,
        resolve_execution_capital,
        resolve_execution_trade_policy,
    )
    from oqp.research.backtesting.return_horizons import (
        attach_return_horizon,
        normalize_return_horizon,
    )
    from oqp.research.factors import factor_search_roots, load_factor_module
    from oqp.research.parameter_schema import attach_factor_parameter_attrs

    try:
        args.return_horizon = normalize_return_horizon(args.return_horizon)
    except Exception as e:
        parser.error(str(e))

    def frame_date_window(frame: pd.DataFrame) -> tuple[str | None, str | None]:
        date_col = "date" if "date" in frame.columns else "datetime" if "datetime" in frame.columns else None
        if date_col is None:
            return None, None
        dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
        if dates.empty:
            return None, None
        return dates.min().isoformat(), dates.max().isoformat()

    def inclusive_end_boundary(value: str | None):
        if value is None or not str(value).strip():
            return None
        parsed = pd.to_datetime(value, errors="raise")
        raw = str(value).strip()
        if " " not in raw and "T" not in raw and len(raw) <= 10:
            return parsed + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return parsed

    def apply_backtest_date_filter(frame: pd.DataFrame) -> pd.DataFrame:
        if not args.start_date and not args.end_date:
            return frame
        if "date" not in frame.columns:
            raise ValueError("--start_date/--end_date require a date column after factor preparation.")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        mask = dates.notna()
        if args.start_date:
            mask &= dates.ge(pd.to_datetime(args.start_date, errors="raise"))
        end_boundary = inclusive_end_boundary(args.end_date)
        if end_boundary is not None:
            mask &= dates.le(end_boundary)
        return frame.loc[mask].copy()

    factor_module_name = args.factor
    # Strip .py if the user accidentally typed it in the terminal
    if factor_module_name.endswith('.py'):
        factor_module_name = factor_module_name[:-3]
        
    print(f"🚀 Booting Backtest Engine for: {factor_module_name}...")
    start_time = time.time()

    # 2. Dynamically Import the Factor Code
    try:
        factor_module = load_factor_module(factor_module_name)
    except ModuleNotFoundError:
        search_roots = "\n      - ".join(str(path) for path in factor_search_roots())
        print(
            f"❌ ERROR: Could not find factor module {factor_module_name}.py\n"
            f"   Searched:\n      - {search_roots}"
        )
        return

    factor_id, factor_name, factor_category, factor_rationale, factor_complexity, factor_geometry = get_factor_metadata(
        factor_module,
        factor_module_name,
    )
    try:
        supported_markets = validate_factor_market_compatibility(
            factor_module,
            args.asset,
            factor_id=factor_id,
        )
    except Exception as e:
        print(f"❌ [FACTOR MARKET ERROR] {e}")
        return
    print(
        "   -> 🧭 Factor Market Gate: "
        f"{'ALL' if '*' in supported_markets else ', '.join(supported_markets)}"
    )
    try:
        capital_profile = resolve_execution_capital(
            asset_class=args.asset,
            factor_module=factor_module,
            initial_capital=args.initial_capital,
            capital_currency=args.capital_currency,
            capital_profile=args.capital_profile,
        )
    except Exception as e:
        print(f"❌ [CAPITAL POLICY ERROR] {e}")
        return
    print(
        "   -> 💰 Capital Policy: "
        f"{capital_profile.initial_capital:,.0f} {capital_profile.currency} "
        f"({capital_profile.source}, profile={capital_profile.profile})"
    )
    try:
        trade_policy = resolve_execution_trade_policy(
            factor_module=factor_module,
            min_trade_weight_delta=args.min_trade_weight_delta,
        )
    except Exception as e:
        print(f"❌ [TRADE POLICY ERROR] {e}")
        return
    print(
        "   -> 🧹 Trade Policy: "
        f"min_trade_weight_delta={trade_policy.min_trade_weight_delta:g} "
        f"({trade_policy.source})"
    )

    if is_options_market(args.asset):
        run_options_backtest_route(
            args=args,
            factor_module=factor_module,
            factor_id=factor_id,
            factor_name=factor_name,
            factor_category=factor_category,
            factor_rationale=factor_rationale,
            factor_complexity=factor_complexity,
            capital_profile=capital_profile,
            trade_policy=trade_policy,
            start_time=start_time,
        )
        return

    # 3. Register Metadata to Database
    ALPHA_RESEARCH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ALPHA_RESEARCH_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS factors (
            factor_id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            economic_rationale TEXT,
            complexity_score INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id TEXT PRIMARY KEY,
            factor_id TEXT,
            round_number INTEGER,
            validation_ic REAL,
            holdout_ic REAL,
            crisis_ic REAL,
            turnover_rate REAL,
            annualized_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            total_trades INTEGER DEFAULT 0,
            asset_class TEXT,
            universe_size INTEGER,
            traded_tickers TEXT,
            returns_file_path TEXT,
            evaluation_geometry TEXT,
            ic_metric TEXT,
            validation_hit_rate REAL,
            holdout_hit_rate REAL,
            crisis_hit_rate REAL,
            split_mode TEXT,
            split_boundary TEXT,
            validation_rows INTEGER,
            holdout_rows INTEGER,
            crisis_rows INTEGER,
            avg_daily_cost_bps REAL,
            total_exchange_fees REAL,
            total_slippage_cost REAL,
            total_execution_cost REAL,
            factor_params TEXT,
            selected_product TEXT,
            selected_symbol TEXT,
            raw_event_count INTEGER,
            quality_event_count INTEGER,
            throttled_event_count INTEGER,
            active_tick_count INTEGER,
            stat_raw_p_value REAL,
            stat_metric_p_value REAL,
            stat_hit_rate_p_value REAL,
            stat_sharpe_p_value REAL,
            stat_adjusted_p_value REAL,
            stat_holm_p_value REAL,
            stat_fdr_q_value REAL,
            stat_trial_count INTEGER,
            stat_metric_observations INTEGER,
            stat_hit_rate_observations INTEGER,
            stat_test_method TEXT,
            stat_significance TEXT,
            stat_research_family TEXT,
            stat_trial_signature TEXT,
            execution_lot_mode TEXT,
            execution_lot_mode_requested TEXT,
            initial_capital REAL,
            capital_currency TEXT,
            capital_profile TEXT,
            capital_source TEXT,
            min_trade_weight_delta REAL,
            min_trade_weight_delta_source TEXT,
            lot_constrained_rate REAL,
            fee_constrained_rate REAL,
            avg_one_lot_weight REAL,
            avg_abs_rounding_error_weight REAL,
            avg_round_trip_fee_bps REAL,
            avg_round_trip_fee_ticks REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            factor_id TEXT,
            research_family TEXT,
            trial_signature TEXT,
            params_hash TEXT,
            asset_class TEXT,
            evaluation_geometry TEXT,
            metric_name TEXT,
            raw_p_value REAL,
            metric_p_value REAL,
            hit_rate_p_value REAL,
            sharpe_p_value REAL,
            bonferroni_p_value REAL,
            holm_p_value REAL,
            fdr_q_value REAL,
            trial_count_m INTEGER,
            significance TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnostics (
            run_id TEXT PRIMARY KEY,
            failure_code TEXT,
            suggested_action TEXT
        )
    ''')

    cursor.execute('''
        INSERT OR IGNORE INTO factors (factor_id, name, category, economic_rationale, complexity_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (factor_id, factor_name, factor_category, factor_rationale, factor_complexity))
    
    conn.commit()
    conn.close()

    # 4. Load the Data Lake dynamically via the Universal Data Engine
    print(f"   ⚡ Loading Universal Data Engine ({args.asset})...")
    if not os.path.exists(args.data_file):
        print(f"❌ ERROR: File {args.data_file} not found.")
        return
        
    try:
        # Split 'FUTURES_CN' to just 'FUTURES' so the DataEngineFactory doesn't break, 
        # but keep args.asset completely intact for the Evaluator!
        base_asset_type = args.asset.split('_')[0] 
        feed = DataEngineFactory.create_feed(base_asset_type, args.data_file)
        df = feed.load_data()
        crisis_period = feed.get_crisis_period()
    except Exception as e:
        print(f"❌ [DATA ENGINE ERROR] {e}")
        return
        
    # --- THE LONG FORMAT ENFORCER ---
    if 'ticker' not in df.columns and 'symbol' in df.columns:
        df['ticker'] = df['symbol'].astype(str)
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
    if 'close' not in df.columns and 'last_price' in df.columns:
        df['close'] = df['last_price']

    # Optional factor-owned universe adapter. Tick factors can narrow a
    # multi-contract parquet to their intended product/main contract without
    # hardcoding file names into the backtest runner.
    if hasattr(factor_module, "prepare_data"):
        try:
            print("   -> Applying factor data adapter...")
            df = factor_module.prepare_data(df)
        except Exception as e:
            print(f"❌ [FACTOR DATA ADAPTER ERROR] {e}")
            return

    if 'ticker' in df.columns and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        
        # --- NEW HFT FIX: Standardize tick data column names ---
        if 'close' not in df.columns and 'last_price' in df.columns:
            df = df.rename(columns={'last_price': 'close'})
    else:
        print("❌ ERROR: Data must have 'ticker' and 'date' columns (Long Format).")
        return

    prepared_start, prepared_end = frame_date_window(df)
    prepared_rows = int(len(df))
    if args.start_date or args.end_date:
        try:
            df = apply_backtest_date_filter(df)
        except Exception as e:
            print(f"❌ [DATE FILTER ERROR] {e}")
            return
        if df.empty:
            print(
                "❌ [DATE FILTER ERROR] No rows remain after "
                f"start_date={args.start_date!r}, end_date={args.end_date!r}."
            )
            return
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        filtered_start, filtered_end = frame_date_window(df)
        print(
            "   -> 🗓️ Backtest Window Filter: "
            f"{filtered_start} to {filtered_end} "
            f"({len(df):,}/{prepared_rows:,} prepared rows kept)"
        )
    else:
        filtered_start, filtered_end = prepared_start, prepared_end

    data_frequency = infer_data_frequency(df)
    try:
        df = attach_return_horizon(
            df,
            return_horizon=args.return_horizon,
            data_frequency=data_frequency,
        )
    except Exception as e:
        print(f"❌ [RETURN HORIZON ERROR] {e}")
        return
    print(
        "   -> ⏱️ Return Horizon: "
        f"{df.attrs.get('return_horizon')} "
        f"({df.attrs.get('return_horizon_description')})"
    )
    data_profile = infer_dataset_tradability(
        df,
        source_path=args.data_file,
        asset_class=args.asset,
        data_frequency=data_frequency,
    )
    vertical_attrs = {
        "market_vertical": args.asset,
        "source_path": os.path.abspath(args.data_file),
        "data_file": os.path.abspath(args.data_file),
        "dataset_id": os.path.splitext(os.path.basename(args.data_file))[0],
        "data_vendor": "local_parquet",
        "data_frequency": data_frequency,
        "prepared_data_start": prepared_start,
        "prepared_data_end": prepared_end,
        "prepared_data_rows": prepared_rows,
        "backtest_start": filtered_start,
        "backtest_end": filtered_end,
        "backtest_rows": int(len(df)),
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "dataset_role": data_profile.dataset_role,
        "data_tradability": data_profile.tradability,
        "data_price_source": data_profile.price_source,
        "data_roll_model": data_profile.roll_model,
        "data_liquidity_model": data_profile.liquidity_model,
        "data_execution_reality": data_profile.execution_reality,
        "min_daily_traded_value": ASSET_TAXONOMY.get(args.asset, {}).get("min_daily_traded_value", 0.0),
        "execution_lot_mode_requested": args.lot_mode,
        "initial_capital": capital_profile.initial_capital,
        "capital_currency": capital_profile.currency,
        "capital_profile": capital_profile.profile,
        "capital_source": capital_profile.source,
        "min_trade_weight_delta": trade_policy.min_trade_weight_delta,
        "min_trade_weight_delta_source": trade_policy.source,
        "return_horizon": df.attrs.get("return_horizon"),
        "return_horizon_description": df.attrs.get("return_horizon_description"),
        "benchmark_return_col": df.attrs.get("benchmark_return_col"),
        "execution_assumption": df.attrs.get("execution_assumption"),
    }
    apply_vertical_attrs(df, vertical_attrs)
    df = attach_capital_attrs(df, capital_profile)
    df = attach_trade_policy_attrs(df, trade_policy)
    df = attach_dataset_tradability_attrs(df, data_profile)
    print(
        "   -> 🧾 Data Contract: "
        f"role={data_profile.dataset_role}, "
        f"tradability={data_profile.tradability}, "
        f"price_source={data_profile.price_source}, "
        f"roll_model={data_profile.roll_model}"
    )
    for warning in data_profile.warnings[:3]:
        print(f"      ⚠️ {warning}")

    if df.empty:
        print("❌ Backtest halted due to data load failure.")
        return

    print(f"   ✅ Data Lake built successfully. Shape: {df.shape}")
    print(f"   🛡️ Crisis Period registered for Tail-Risk Stress Test: {crisis_period}")

    # 5. Apply the Math
    print("   -> Computing Factor Scores...")
    df = factor_module.compute(df)
    df = attach_factor_parameter_attrs(df, factor_module)
    apply_vertical_attrs(df, vertical_attrs)
    df = attach_capital_attrs(df, capital_profile)
    df = attach_trade_policy_attrs(df, trade_policy)

    try:
        contract = resolve_factor_contract(
            factor_module,
            df,
            factor_id=factor_id,
            requested_execution_mode=args.execution_mode,
            requested_return_assumption=(
                vertical_attrs["execution_assumption"]
                if args.return_horizon != "auto"
                else None
            ),
            default_return_assumption=vertical_attrs["execution_assumption"],
            market_vertical=args.asset,
            strict=args.strict_factor_contract,
        )
    except Exception as e:
        print(f"❌ [FACTOR CONTRACT ERROR] {e}")
        return
    df = attach_factor_contract_attrs(df, contract)
    factor_geometry = contract.evaluation_geometry
    if contract.contract_source != "explicit":
        print("   ⚠️ Factor Contract: inferred legacy contract. Add FACTOR_CONTRACT for strict mode.")
        for warning in contract.warnings[:4]:
            print(f"      - {warning}")
    print(
        "   -> 📜 Factor Contract: "
        f"geometry={contract.evaluation_geometry}, "
        f"execution_mode={contract.execution_mode}, "
        f"alpha={contract.alpha_signal_col}, "
        f"weight={contract.execution_weight_col}, "
        f"lag={contract.execution_lag}, "
        f"return={contract.return_assumption}"
    )

    # =========================================================
    # 🧭 EXECUTION MODE: optional risk desk or factor-owned weights
    # =========================================================
    execution_mode = contract.execution_mode
    execution_config = build_execution_mode_config(factor_module, args, contract.execution_weight_col)
    print(f"   -> 🧭 Execution Mode: {execution_mode}")
    try:
        execution_result = ExecutionModeFactory.create(execution_mode, execution_config).apply(df)
    except Exception as e:
        print(f"❌ [EXECUTION MODE ERROR] {e}")
        return
    df = execution_result.df
    print(f"   -> {execution_result.detail} Source column: {execution_result.source_col}")
    apply_vertical_attrs(df, vertical_attrs)
    df = attach_capital_attrs(df, capital_profile)
    df = attach_trade_policy_attrs(df, trade_policy)
    df = attach_factor_contract_attrs(df, contract)
    # =========================================================

    # 6. Pass to the Evaluator
    print("   -> Passing to Prime Broker Execution Desk & Database Logger...")
    
    # INJECT THE ASSET CLASS HERE
    evaluator = AlphaEvaluator(
        db_path=ALPHA_RESEARCH_DB_PATH,
        logs_dir=ALPHA_RUNTIME_ARTIFACT_ROOT,
        asset_class=args.asset,
    )
    
    # SURGICAL INJECTION: Passing the dynamic crisis period
    evaluator.run_evaluation(
        factor_id,
        df,
        crisis_period=crisis_period,
        split_date=args.split_date,
        split_mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        purge_periods=args.purge_periods,
        embargo_periods=args.embargo_periods,
        purge_unit=args.purge_unit,
        factor_category=factor_category,
        strategy_geometry=factor_geometry,
        alpha_signal_col=contract.alpha_signal_col,
        lot_mode=args.lot_mode,
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Engine Execution Complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
