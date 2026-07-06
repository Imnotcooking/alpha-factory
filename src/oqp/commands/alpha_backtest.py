from __future__ import annotations

import os
import argparse
import sqlite3
import time
from oqp.contracts.market_vertical import ASSET_TAXONOMY
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


def main():
    # 1. Setup the Terminal Command Arguments
    parser = argparse.ArgumentParser(description="Alpha Mine Backtest Engine")
    parser.add_argument("--factor", type=str, required=True, help="Name of the factor module (e.g., fac_003_MA_Crossover)")
    
    # 2. 🛠️ UPGRADE: Use the sophisticated Asset Taxonomy keys dynamically
    parser.add_argument("--asset", type=str, choices=list(ASSET_TAXONOMY.keys()), default='FUTURES_CN', help="Asset taxonomy class (e.g., FUTURES_CN, EQUITY_US)")
    
    parser.add_argument("--data_file", type=str, default=str(DEFAULT_DATA_FILE), help="Path to the internal parquet file")
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
    parser.add_argument("--max_weight_per_asset", type=float, default=None, help="Optional per-asset cap for direct/statarb modes.")
    parser.add_argument(
        "--lot_mode",
        choices=["auto", "integer", "fractional"],
        default="auto",
        help="Contract sizing mode. Auto uses integer lots for futures and fractional sizing elsewhere.",
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
        action="store_true",
        help="Require FACTOR_CONTRACT instead of allowing legacy inference.",
    )
    
    args = parser.parse_args()

    import pandas as pd
    from oqp.data import DataEngineFactory
    from oqp.research import attach_dataset_tradability_attrs, infer_dataset_tradability
    from oqp.research import attach_factor_contract_attrs, resolve_factor_contract
    from oqp.research.backtesting import AlphaEvaluator
    from oqp.research.backtesting import (
        ExecutionModeFactory,
        attach_capital_attrs,
        attach_trade_policy_attrs,
        resolve_execution_capital,
        resolve_execution_trade_policy,
    )
    from oqp.research.factors import factor_search_roots, load_factor_module

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
            
        # Automatically calculate executable next-session forward return.
        # The signal is known after today's close, so the tradable entry anchor
        # is tomorrow's open, not today's close.
        if 'forward_return' not in df.columns and 'close' in df.columns:
            is_intraday = bool(df['date'].dt.normalize().ne(df['date']).any())
            if is_intraday:
                next_close = df.groupby('ticker')['close'].shift(-1)
                df['forward_return'] = next_close / df['close'] - 1
            elif 'open' in df.columns:
                next_open = df.groupby('ticker')['open'].shift(-1)
                next_close = df.groupby('ticker')['close'].shift(-1)
                df['forward_return'] = next_close / next_open - 1
            else:
                print("⚠️ WARNING: 'open' column missing. Falling back to close-to-close forward_return.")
                df['forward_return'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1
    else:
        print("❌ ERROR: Data must have 'ticker' and 'date' columns (Long Format).")
        return

    data_frequency = infer_data_frequency(df)
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
        "dataset_role": data_profile.dataset_role,
        "data_tradability": data_profile.tradability,
        "data_price_source": data_profile.price_source,
        "data_roll_model": data_profile.roll_model,
        "data_liquidity_model": data_profile.liquidity_model,
        "data_execution_reality": data_profile.execution_reality,
        "execution_lot_mode_requested": args.lot_mode,
        "initial_capital": capital_profile.initial_capital,
        "capital_currency": capital_profile.currency,
        "capital_profile": capital_profile.profile,
        "capital_source": capital_profile.source,
        "min_trade_weight_delta": trade_policy.min_trade_weight_delta,
        "min_trade_weight_delta_source": trade_policy.source,
        "execution_assumption": (
            "bar_signal_next_bar"
            if data_frequency in {"intraday", "tick"}
            else "close_signal_next_open_to_close"
            if "open" in df.columns
            else "close_to_close_fallback"
        ),
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
    apply_vertical_attrs(df, vertical_attrs)
    df = attach_capital_attrs(df, capital_profile)
    df = attach_trade_policy_attrs(df, trade_policy)

    try:
        contract = resolve_factor_contract(
            factor_module,
            df,
            factor_id=factor_id,
            requested_execution_mode=args.execution_mode,
            default_return_assumption=vertical_attrs["execution_assumption"],
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
