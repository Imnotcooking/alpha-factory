import sqlite3
import pandas as pd
import numpy as np
import scipy.stats as stats
import uuid
import warnings
import os
import json
from pathlib import Path

from oqp.data import InstrumentMaster
from oqp.native import load_quant_core
from oqp.research.backtesting.capital_policy import attach_capital_attrs, resolve_execution_capital
from oqp.research.backtesting.trade_policy import (
    DEFAULT_MIN_TRADE_WEIGHT_DELTA,
    attach_trade_policy_attrs,
    resolve_execution_trade_policy,
)
from oqp.research.evaluation import AlphaMetricEvaluator, EvaluationGeometry
from oqp.research.multiple_testing import (
    benjamini_hochberg_q_values,
    bonferroni_p_value,
    holm_bonferroni_adjust,
    significance_label,
    stable_trial_hash,
)
from oqp.research.splits import build_chronological_split
from oqp.research.statistical_tests import AlphaStatisticalTester, sharpe_p_value_from_returns
from oqp.contracts.market_vertical import ASSET_TAXONOMY

_legacy_native_dir = os.environ.get("OQP_LEGACY_QUANT_CORE_DIR", "").strip()
qc = load_quant_core(
    (
        "estimate_hurst",
        "CryptoOrderBookTCA",
        "SquareRootTCA",
        "StochasticTCAWrapper",
        "FuturesMargin",
        "EquitiesMargin",
        "ExecutionEngine",
    ),
    legacy_paths=(Path(_legacy_native_dir),) if _legacy_native_dir else (),
)
warnings.filterwarnings('ignore')


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ALPHA_RESEARCH_DB_PATH = os.path.join(
    os.fspath(_REPO_ROOT),
    "runtime",
    "db",
    "research",
    "alpha_lab",
    "research_memory.db",
)
_DEFAULT_ALPHA_ARTIFACT_ROOT = os.path.join(
    os.fspath(_REPO_ROOT),
    "runtime",
    "artifacts",
    "research",
    "alpha_lab",
)


VERTICAL_TRIAL_COLUMNS = {
    "market_vertical": "TEXT",
    "dataset_id": "TEXT",
    "universe_id": "TEXT",
    "data_frequency": "TEXT",
    "dataset_role": "TEXT",
    "data_tradability": "TEXT",
    "data_price_source": "TEXT",
    "data_roll_model": "TEXT",
    "data_liquidity_model": "TEXT",
    "data_execution_reality": "TEXT",
    "data_vendor": "TEXT",
    "execution_assumption": "TEXT",
}


def infer_frame_frequency(
    df: pd.DataFrame,
    *,
    duplicate_session_tolerance: float = 0.01,
) -> str:
    attr_frequency = str(df.attrs.get("data_frequency") or "").strip().lower()
    if attr_frequency:
        return attr_frequency

    if "date" not in df.columns:
        return "unknown"
    dates = pd.to_datetime(df["date"], errors="coerce")
    valid = pd.DataFrame({"date": dates}).dropna()
    if valid.empty:
        return "unknown"
    valid["session_date"] = valid["date"].dt.normalize()

    if "ticker" in df.columns:
        valid["ticker"] = df.loc[valid.index, "ticker"].astype(str)
        bars_per_session = valid.groupby(["ticker", "session_date"]).size()
        duplicate_rate = float((bars_per_session > 1).mean()) if not bars_per_session.empty else 0.0
        if duplicate_rate <= duplicate_session_tolerance:
            return "daily"
    elif valid["session_date"].nunique() == len(valid):
        return "daily"

    return "intraday" if valid["date"].dt.normalize().ne(valid["date"]).any() else "daily"


class ExecutionDesk:
    """
    Institutional Execution Simulator (C++ Accelerated).
    Assembles the polymorphic C++ engine and hands off millions of rows for zero-latency simulation.
    """
    def __init__(
        self,
        asset_class: str = 'FUTURES',
        max_leverage: float = 2.0,
        deadband_threshold: float | None = None,
        integer_lots: bool = False,
        initial_capital: float = 1_000_000.0,
        capital_currency: str = "USD",
        min_trade_weight_delta: float | None = None,
    ):
        self.asset_class = asset_class.upper()
        self.max_leverage = max_leverage
        if min_trade_weight_delta is None:
            min_trade_weight_delta = (
                DEFAULT_MIN_TRADE_WEIGHT_DELTA
                if deadband_threshold is None
                else float(deadband_threshold)
            )
        self.min_trade_weight_delta = max(float(min_trade_weight_delta), 0.0)
        self.integer_lots = bool(integer_lots)
        self.initial_capital = float(initial_capital)
        if self.initial_capital <= 0:
            raise ValueError("ExecutionDesk initial_capital must be positive.")
        self.capital_currency = str(capital_currency or "USD").upper()

    @staticmethod
    def _fee_type_code(fee_type: str) -> int:
        return 1 if str(fee_type).strip().lower() == "fixed" else 0

    def _attach_instrument_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        master = InstrumentMaster(self.asset_class)
        tickers = df['ticker'].astype(str)
        profile_map = {ticker: master.get_profile(ticker) for ticker in tickers.unique()}

        df['base_ticker'] = tickers.map(lambda ticker: profile_map[ticker].ticker)
        df['multiplier'] = tickers.map(lambda ticker: profile_map[ticker].multiplier).astype(float)
        df['tick_size'] = tickers.map(lambda ticker: profile_map[ticker].tick_size).astype(float)
        df['fee_type_code'] = tickers.map(
            lambda ticker: self._fee_type_code(profile_map[ticker].fee_type)
        ).astype(np.int32)
        df['fee_open'] = tickers.map(lambda ticker: profile_map[ticker].fee_open).astype(float)
        df['fee_close_history'] = tickers.map(
            lambda ticker: profile_map[ticker].fee_close_history
        ).astype(float)
        df['fee_close_today'] = tickers.map(
            lambda ticker: profile_map[ticker].fee_close_today
        ).astype(float)
        df['margin_rate'] = tickers.map(lambda ticker: profile_map[ticker].margin_rate).astype(float)
        return df

    def _attach_explicit_execution_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if 'execution_period_return' not in out.columns and 'forward_return' in out.columns:
            out['execution_period_return'] = pd.to_numeric(out['forward_return'], errors='coerce')

        if 'execution_price' not in out.columns:
            is_intraday = infer_frame_frequency(out) in {"intraday", "tick"}
            if not is_intraday and 'open' in out.columns:
                # Daily research assumes the signal is known after today's close
                # and entered at the next session open.
                execution_price = out.groupby('ticker')['open'].shift(-1)
            else:
                execution_price = out['close']
            out['execution_price'] = execution_price.fillna(out['close'])

        out['execution_price'] = pd.to_numeric(out['execution_price'], errors='coerce').fillna(out['close'])
        out['adjusted_execution_price'] = out['execution_price'] * out['multiplier']
        return out

    @staticmethod
    def _attach_lot_diagnostics(df: pd.DataFrame, execution_result: dict) -> None:
        diagnostic_defaults = {
            "desired_contracts": 0.0,
            "position_contracts": 0.0,
            "rounding_error_weight": 0.0,
            "one_lot_weight": 0.0,
            "round_trip_fee_bps": 0.0,
            "round_trip_fee_ticks": 0.0,
            "is_lot_constrained": 0.0,
            "is_fee_constrained": 0.0,
        }
        for column, default in diagnostic_defaults.items():
            if column in execution_result:
                df[column] = np.asarray(execution_result[column], dtype=float)
            elif column not in df.columns:
                df[column] = default

    def run_backtest(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None, signal_col: str = 'signal') -> tuple[pd.DataFrame, float, pd.DataFrame]:
        df = df.sort_values(by=['date', 'ticker']).copy()

        # Ensure volume exists for TCA
        if 'volume' not in df.columns:
            df['volume'] = 1000.0  # Fallback

        # --- Map String Tickers to Integer IDs for C++ Memory Isolation ---
        df['asset_id'] = pd.factorize(df['ticker'])[0].astype(np.int32)

        # ---------------------------------------------------------
        # 🏢 INJECTING PHYSICAL REALITY (Instrument Master)
        # ---------------------------------------------------------
        df = self._attach_instrument_profiles(df)

        # ---------------------------------------------------------
        # ⚖️ TRUST THE RISK DESK: Target Weights already calculated
        # ---------------------------------------------------------
        # The ML backtester outputs final optimized weights to `signal`.
        # We just scale them up to the fund's gross leverage target if needed.
        if signal_col not in df.columns:
             raise ValueError(f"❌ ExecutionDesk requires a '{signal_col}' column.")
             
        df['target_weight'] = df[signal_col] * self.max_leverage

        # Adjust price by multiplier so C++ calculates actual Contract Notional Value correctly
        df['adjusted_price'] = df['close'] * df['multiplier']
        df = self._attach_explicit_execution_inputs(df)

        print(f"   ⚡ [PRIME BROKER] Routing {len(df):,} rows to C++ Execution Engine...")
        
        # --- NEW: CALCULATE FRACTIONAL VOLATILITY (Hurst) ---
        print("   -> 🌊 Calculating Fractional Wavelet Hurst Exponent via C++...")
        
        # Extract Volatility (Using Garman-Klass or Fallback to 1%)
        if 'f_macro_gk_vol' in df.columns:
            vol_arr = np.ascontiguousarray(df['f_macro_gk_vol'].fillna(0.01).values, dtype=np.float64)
        else:
            vol_arr = np.ascontiguousarray(np.full(len(df), 0.01), dtype=np.float64)

        is_intraday = infer_frame_frequency(df) in {"intraday", "tick"}
        if is_intraday:
            # Daily-style wavelet Hurst is not meaningful at tick resolution,
            # and rolling it across millions of ticks is prohibitively slow.
            df['hurst'] = 0.5
        else:
            # We must group by ticker to calculate rolling Hurst correctly per asset
            def calc_hurst(group):
                # Rolling 64-day window passed into the C++ Wavelet engine
                return group['adjusted_price'].rolling(window=64).apply(
                    lambda x: qc.estimate_hurst(x.values), raw=False
                )

            df['hurst'] = df.groupby('ticker', group_keys=False).apply(calc_hurst)
        hurst_arr = np.ascontiguousarray(df['hurst'].fillna(0.5).values, dtype=np.float64)

        # --- THE C++ ENGINE ASSEMBLY (Dependency Injection via Taxonomy) ---
        config_data = ASSET_TAXONOMY.get(self.asset_class, ASSET_TAXONOMY.get("FUTURES_CN", {}))
        
        if "CRYPTO" in self.asset_class:
            tca = qc.CryptoOrderBookTCA(bps=0.0005, penalty=2.0)
            margin = qc.FuturesMargin(maintenance_req=0.10)
        elif "EQUITY" in self.asset_class:
            tca = qc.SquareRootTCA(bps=0.0001, gamma=0.05)
            req = 1.0 if config_data.get("region") == "CN" else 0.25
            margin = qc.EquitiesMargin(maintenance_req=req)
        else: # FUTURES (Now uses Stochastic TCA)
            # Default to Stochastic TCA for Futures (lambda=1e-4, eta=0.1, gamma=2.0, T=60)
            tca = qc.StochasticTCAWrapper(1e-4, 0.1, 2.0, 60)
            margin_req = float(df['margin_rate'].median()) if 'margin_rate' in df.columns else 0.05
            margin = qc.FuturesMargin(maintenance_req=margin_req)

        has_limits = config_data.get("price_limit", False)
        is_t1 = True if config_data.get("t_settlement", 0) == 1 else False

        engine = qc.ExecutionEngine(
            tca_model=tca, 
            margin_model=margin, 
            initial_capital=self.initial_capital, 
            deadband=self.min_trade_weight_delta,
            enforce_price_limits=has_limits,
            enforce_t1=is_t1
        )

        # --- THE MEMORY HANDOFF ---
        df['execution_date_id'] = pd.factorize(pd.to_datetime(df['date']).dt.normalize())[0].astype(np.int32)
        df['execution_time_id'] = pd.factorize(pd.to_datetime(df['date']))[0].astype(np.int32)
        ids_arr = np.ascontiguousarray(df['asset_id'].values, dtype=np.int32)
        prices_arr = np.ascontiguousarray(df['adjusted_price'].values, dtype=np.float64)
        weights_arr = np.ascontiguousarray(df['target_weight'].values, dtype=np.float64)
        volumes_arr = np.ascontiguousarray(df['volume'].values, dtype=np.float64)
        times_arr = np.ascontiguousarray(df['execution_time_id'].values, dtype=np.int32)
        dates_arr = np.ascontiguousarray(df['execution_date_id'].values, dtype=np.int32)
        multipliers_arr = np.ascontiguousarray(df['multiplier'].values, dtype=np.float64)
        tick_sizes_arr = np.ascontiguousarray(df['tick_size'].values, dtype=np.float64)
        fee_types_arr = np.ascontiguousarray(df['fee_type_code'].values, dtype=np.int32)
        fee_open_arr = np.ascontiguousarray(df['fee_open'].values, dtype=np.float64)
        fee_close_history_arr = np.ascontiguousarray(df['fee_close_history'].values, dtype=np.float64)
        fee_close_today_arr = np.ascontiguousarray(df['fee_close_today'].values, dtype=np.float64)
        explicit_returns_available = (
            'execution_period_return' in df.columns
            and pd.to_numeric(df['execution_period_return'], errors='coerce').notna().any()
        )

        if explicit_returns_available and hasattr(engine, "run_simulation_with_costs_and_returns"):
            print("   -> Using explicit period-return execution path.")
            period_returns_arr = np.ascontiguousarray(
                pd.to_numeric(df['execution_period_return'], errors='coerce').fillna(0.0).values,
                dtype=np.float64,
            )
            execution_prices_arr = np.ascontiguousarray(
                df['adjusted_execution_price'].fillna(df['adjusted_price']).values,
                dtype=np.float64,
            )
            execution_result = engine.run_simulation_with_costs_and_returns(
                asset_ids=ids_arr,
                prices=execution_prices_arr,
                target_weights=weights_arr,
                period_returns=period_returns_arr,
                volumes=volumes_arr,
                volatilities=vol_arr,
                hursts=hurst_arr,
                time_ids=times_arr,
                date_ids=dates_arr,
                multipliers=multipliers_arr,
                fee_types=fee_types_arr,
                fee_open=fee_open_arr,
                fee_close_history=fee_close_history_arr,
                fee_close_today=fee_close_today_arr,
                tick_sizes=tick_sizes_arr,
                integer_lots=self.integer_lots,
            )
            df.attrs["execution_return_mode"] = "explicit_period_return"
            df['equity'] = np.asarray(execution_result["equity_curve"], dtype=float)
            df['gross_equity'] = np.asarray(execution_result["gross_equity_curve"], dtype=float)
            df['slippage_cost'] = np.asarray(execution_result["slippage_cost"], dtype=float)
            df['exchange_fee'] = np.asarray(execution_result["exchange_fee"], dtype=float)
            df['total_cost'] = np.asarray(execution_result["total_cost"], dtype=float)
            df['executed_weight'] = np.asarray(execution_result["executed_weight"], dtype=float)
            df['trade_notional'] = np.asarray(execution_result["trade_notional"], dtype=float)
            df['trade_contracts'] = np.asarray(execution_result["trade_contracts"], dtype=float)
            df['portfolio_leverage'] = np.asarray(execution_result["portfolio_leverage"], dtype=float)
            self._attach_lot_diagnostics(df, execution_result)
        elif hasattr(engine, "run_simulation_with_costs"):
            if explicit_returns_available:
                print("   ⚠️ quant_core lacks explicit-return execution; falling back to close-stream path.")
            execution_result = engine.run_simulation_with_costs(
                asset_ids=ids_arr,
                prices=prices_arr,
                target_weights=weights_arr,
                volumes=volumes_arr,
                volatilities=vol_arr,
                hursts=hurst_arr,
                date_ids=dates_arr,
                multipliers=multipliers_arr,
                fee_types=fee_types_arr,
                fee_open=fee_open_arr,
                fee_close_history=fee_close_history_arr,
                fee_close_today=fee_close_today_arr,
                tick_sizes=tick_sizes_arr,
                integer_lots=self.integer_lots,
            )
            df['equity'] = np.asarray(execution_result["equity_curve"], dtype=float)
            df['gross_equity'] = np.asarray(execution_result["gross_equity_curve"], dtype=float)
            df['slippage_cost'] = np.asarray(execution_result["slippage_cost"], dtype=float)
            df['exchange_fee'] = np.asarray(execution_result["exchange_fee"], dtype=float)
            df['total_cost'] = np.asarray(execution_result["total_cost"], dtype=float)
            df['executed_weight'] = np.asarray(execution_result["executed_weight"], dtype=float)
            df['trade_notional'] = np.asarray(execution_result["trade_notional"], dtype=float)
            df['trade_contracts'] = np.asarray(execution_result["trade_contracts"], dtype=float)
            df['portfolio_leverage'] = np.asarray(execution_result["portfolio_leverage"], dtype=float)
            self._attach_lot_diagnostics(df, execution_result)
        else:
            print("   ⚠️ quant_core lacks run_simulation_with_costs; exchange fees default to zero.")
            equity_curve = engine.run_simulation(
                asset_ids=ids_arr,
                prices=prices_arr,
                target_weights=weights_arr,
                volumes=volumes_arr,
                volatilities=vol_arr,
                hursts=hurst_arr,
                date_ids=dates_arr
            )
            df['equity'] = equity_curve
            df['gross_equity'] = equity_curve
            df['slippage_cost'] = 0.0
            df['exchange_fee'] = 0.0
            df['total_cost'] = 0.0
            df['executed_weight'] = df['target_weight']
            df['trade_notional'] = 0.0
            df['trade_contracts'] = 0.0
            df['portfolio_leverage'] = df.groupby('date')['executed_weight'].transform(lambda x: x.abs().sum())
            self._attach_lot_diagnostics(df, {})

        df['cum_net_equity'] = df['equity'] / self.initial_capital
        df['weight'] = df['executed_weight']
        
        # Calculate true chronological turnover per asset. Do not reorder df here:
        # the C++ equity curve was produced in date/ticker order, so daily equity
        # still needs to use the final row from that execution sequence.
        turnover_ordered = df.sort_values(['ticker', 'date'])
        df['weight_diff'] = turnover_ordered.groupby('ticker')['executed_weight'].diff().abs().fillna(0.0)

        # Reconstruct standard portfolio metrics for the Tear Sheet
        df['trading_day'] = df['date'].dt.date
        portfolio = df.groupby('trading_day').agg(
            equity=('equity', 'last'),
            gross_equity=('gross_equity', 'last'),
            cum_net_equity=('cum_net_equity', 'last'),
            daily_turnover=('weight_diff', 'sum'),
            daily_slippage_cost=('slippage_cost', 'sum'),
            daily_exchange_fee=('exchange_fee', 'sum'),
            daily_total_cost=('total_cost', 'sum'),
            portfolio_leverage=('portfolio_leverage', 'last'),
            lot_constrained_rate=('is_lot_constrained', 'mean'),
            fee_constrained_rate=('is_fee_constrained', 'mean'),
            avg_one_lot_weight=('one_lot_weight', 'mean'),
            avg_round_trip_fee_bps=('round_trip_fee_bps', 'mean'),
            avg_round_trip_fee_ticks=('round_trip_fee_ticks', 'mean'),
            avg_abs_rounding_error_weight=('rounding_error_weight', lambda x: x.abs().mean()),
        ).reset_index()
        
        portfolio['net_return'] = portfolio['cum_net_equity'].pct_change().fillna(0)
        previous_equity = portfolio['equity'].shift(1).fillna(self.initial_capital).replace(0, np.nan)
        portfolio['gross_return'] = (
            portfolio['net_return'] + portfolio['daily_total_cost'].fillna(0.0) / previous_equity
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        portfolio['daily_cost_bps'] = (
            portfolio['daily_total_cost'].fillna(0.0) / previous_equity * 10000.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        portfolio['date'] = pd.to_datetime(portfolio['trading_day'])
        
        if benchmark_df is not None and not benchmark_df.empty:
            portfolio = pd.merge(portfolio, benchmark_df[['date', 'benchmark_return']], on='date', how='left')
            portfolio['benchmark_return'] = portfolio['benchmark_return'].fillna(0)
        else:
            portfolio['benchmark_return'] = 0.0
        
        avg_daily_turnover = portfolio['daily_turnover'].mean()

        return portfolio, avg_daily_turnover, df
    
    def print_tearsheet(self, portfolio: pd.DataFrame, strategy_name: str):
        trading_days_per_year = 252
        total_return = portfolio['cum_net_equity'].iloc[-1] - 1
        num_years = len(portfolio) / trading_days_per_year
        ann_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
        ann_vol = portfolio['net_return'].std() * np.sqrt(trading_days_per_year)
        
        sharpe_ratio = (ann_return - 0.0) / ann_vol if ann_vol > 0 else 0
        
        portfolio['peak'] = portfolio['cum_net_equity'].cummax()
        max_drawdown = ((portfolio['cum_net_equity'] - portfolio['peak']) / portfolio['peak']).min()
        calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0
        avg_turnover = portfolio['daily_turnover'].mean()
        avg_cost_bps = portfolio.get('daily_cost_bps', pd.Series([0.0])).mean()
        
        print(f"\n==================================================")
        print(f" 📊 TEAR SHEET: {strategy_name.upper()} ")
        print(f"==================================================")
        print(f"Trading Days:      {len(portfolio)} (Approx {num_years:.2f} Years)")
        print(f"Annualized Return: {ann_return * 100:.2f}%")
        print(f"Annualized Vol:    {ann_vol * 100:.2f}%")
        print(f"Sharpe Ratio:      {sharpe_ratio:.2f}")
        print(f"Max Drawdown:      {max_drawdown * 100:.2f}%")
        print(f"Calmar Ratio:      {calmar:.2f}")
        print(f"Avg Daily Turnover:{avg_turnover * 100:.1f}%")
        print(f"Avg Daily Cost:    {avg_cost_bps:.2f} bps")
        print("==================================================\n")


class AlphaEvaluator:
    def __init__(self, db_path=None, asset_class="FUTURES_CN", logs_dir=None):
        self.db_path = os.fspath(
            db_path
            or os.environ.get("ALPHA_RESEARCH_DB_PATH")
            or _DEFAULT_ALPHA_RESEARCH_DB_PATH
        )
        self.logs_dir = os.fspath(
            logs_dir
            or os.environ.get("ALPHA_RUNTIME_ARTIFACT_ROOT")
            or _DEFAULT_ALPHA_ARTIFACT_ROOT
        )
        self.metric_evaluator = AlphaMetricEvaluator()
        self.statistical_tester = AlphaStatisticalTester()
        
        # 1. Validate against our rulebook
        if asset_class not in ASSET_TAXONOMY:
            print(f"⚠️ WARNING: '{asset_class}' not found in ASSET_TAXONOMY. Defaulting to FUTURES_CN.")
            self.asset_class = "FUTURES_CN"
        else:
            self.asset_class = asset_class

        db_parent = os.path.dirname(os.path.abspath(self.db_path))
        if db_parent:
            os.makedirs(db_parent, exist_ok=True)

        # 2. Patch the existing database safely (Adds columns if they don't exist)
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_core_tables(conn)
            self._ensure_backtest_run_columns(conn)
            self._ensure_research_trial_columns(conn)

    @staticmethod
    def _ensure_core_tables(conn: sqlite3.Connection):
        conn.execute('''
            CREATE TABLE IF NOT EXISTS factors (
                factor_id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                economic_rationale TEXT,
                complexity_score INTEGER,
                status TEXT DEFAULT 'INCUBATION',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
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
                market_vertical TEXT,
                dataset_id TEXT,
                universe_id TEXT,
                data_frequency TEXT,
                dataset_role TEXT,
                data_tradability TEXT,
                data_price_source TEXT,
                data_roll_model TEXT,
                data_liquidity_model TEXT,
                data_execution_reality TEXT,
                data_vendor TEXT,
                execution_assumption TEXT,
                factor_contract_source TEXT,
                alpha_signal_col TEXT,
                execution_weight_col TEXT,
                execution_mode TEXT,
                execution_lag TEXT,
                return_assumption TEXT,
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
                purge_periods INTEGER,
                embargo_periods INTEGER,
                purge_unit TEXT,
                purged_rows INTEGER,
                embargoed_rows INTEGER,
                execution_lot_mode TEXT,
                execution_lot_mode_requested TEXT,
                initial_capital REAL,
                capital_currency TEXT,
                capital_profile TEXT,
                capital_source TEXT,
                min_trade_weight_delta REAL,
                min_trade_weight_delta_source TEXT,
                avg_daily_cost_bps REAL,
                total_exchange_fees REAL,
                total_slippage_cost REAL,
                total_execution_cost REAL,
                lot_constrained_rate REAL,
                fee_constrained_rate REAL,
                avg_one_lot_weight REAL,
                avg_abs_rounding_error_weight REAL,
                avg_round_trip_fee_bps REAL,
                avg_round_trip_fee_ticks REAL,
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS research_trials (
                trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                factor_id TEXT,
                research_family TEXT,
                trial_signature TEXT,
                params_hash TEXT,
                asset_class TEXT,
                market_vertical TEXT,
                dataset_id TEXT,
                universe_id TEXT,
                data_frequency TEXT,
                dataset_role TEXT,
                data_tradability TEXT,
                data_price_source TEXT,
                data_roll_model TEXT,
                data_liquidity_model TEXT,
                data_execution_reality TEXT,
                data_vendor TEXT,
                execution_assumption TEXT,
                factor_contract_source TEXT,
                alpha_signal_col TEXT,
                execution_weight_col TEXT,
                execution_mode TEXT,
                execution_lag TEXT,
                return_assumption TEXT,
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
        conn.execute('''
            CREATE TABLE IF NOT EXISTS diagnostics (
                diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                failure_code TEXT,
                suggested_action TEXT
            )
        ''')

    @staticmethod
    def _ensure_backtest_run_columns(conn: sqlite3.Connection):
        required_columns = {
            "annualized_return": "REAL",
            "max_drawdown": "REAL",
            "sharpe_ratio": "REAL",
            "total_trades": "INTEGER DEFAULT 0",
            "asset_class": "TEXT",
            **VERTICAL_TRIAL_COLUMNS,
            "factor_contract_source": "TEXT",
            "alpha_signal_col": "TEXT",
            "execution_weight_col": "TEXT",
            "execution_mode": "TEXT",
            "execution_lag": "TEXT",
            "return_assumption": "TEXT",
            "universe_size": "INTEGER",
            "traded_tickers": "TEXT",
            "returns_file_path": "TEXT",
            "evaluation_geometry": "TEXT",
            "ic_metric": "TEXT",
            "validation_hit_rate": "REAL",
            "holdout_hit_rate": "REAL",
            "crisis_hit_rate": "REAL",
            "split_mode": "TEXT",
            "split_boundary": "TEXT",
            "validation_rows": "INTEGER",
            "holdout_rows": "INTEGER",
            "crisis_rows": "INTEGER",
            "purge_periods": "INTEGER",
            "embargo_periods": "INTEGER",
            "purge_unit": "TEXT",
            "purged_rows": "INTEGER",
            "embargoed_rows": "INTEGER",
            "execution_lot_mode": "TEXT",
            "execution_lot_mode_requested": "TEXT",
            "initial_capital": "REAL",
            "capital_currency": "TEXT",
            "capital_profile": "TEXT",
            "capital_source": "TEXT",
            "min_trade_weight_delta": "REAL",
            "min_trade_weight_delta_source": "TEXT",
            "avg_daily_cost_bps": "REAL",
            "total_exchange_fees": "REAL",
            "total_slippage_cost": "REAL",
            "total_execution_cost": "REAL",
            "lot_constrained_rate": "REAL",
            "fee_constrained_rate": "REAL",
            "avg_one_lot_weight": "REAL",
            "avg_abs_rounding_error_weight": "REAL",
            "avg_round_trip_fee_bps": "REAL",
            "avg_round_trip_fee_ticks": "REAL",
            "factor_params": "TEXT",
            "selected_product": "TEXT",
            "selected_symbol": "TEXT",
            "raw_event_count": "INTEGER",
            "quality_event_count": "INTEGER",
            "throttled_event_count": "INTEGER",
            "active_tick_count": "INTEGER",
            "stat_raw_p_value": "REAL",
            "stat_metric_p_value": "REAL",
            "stat_hit_rate_p_value": "REAL",
            "stat_sharpe_p_value": "REAL",
            "stat_adjusted_p_value": "REAL",
            "stat_holm_p_value": "REAL",
            "stat_fdr_q_value": "REAL",
            "stat_trial_count": "INTEGER",
            "stat_metric_observations": "INTEGER",
            "stat_hit_rate_observations": "INTEGER",
            "stat_test_method": "TEXT",
            "stat_significance": "TEXT",
            "stat_research_family": "TEXT",
            "stat_trial_signature": "TEXT",
        }
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(backtest_runs)").fetchall()
        }

        for column, column_type in required_columns.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE backtest_runs ADD COLUMN {column} {column_type};"
                )

    @staticmethod
    def _ensure_research_trial_columns(conn: sqlite3.Connection):
        required_columns = {
            "asset_class": "TEXT",
            **VERTICAL_TRIAL_COLUMNS,
            "factor_contract_source": "TEXT",
            "alpha_signal_col": "TEXT",
            "execution_weight_col": "TEXT",
            "execution_mode": "TEXT",
            "execution_lag": "TEXT",
            "return_assumption": "TEXT",
            "evaluation_geometry": "TEXT",
            "metric_name": "TEXT",
            "metric_p_value": "REAL",
            "hit_rate_p_value": "REAL",
            "sharpe_p_value": "REAL",
            "bonferroni_p_value": "REAL",
            "holm_p_value": "REAL",
            "fdr_q_value": "REAL",
            "trial_count_m": "INTEGER",
            "significance": "TEXT",
        }
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(research_trials)").fetchall()
        }

        for column, column_type in required_columns.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE research_trials ADD COLUMN {column} {column_type};"
                )

    def run_evaluation(
        self,
        factor_id,
        df,
        crisis_period: tuple,
        split_date="2023-01-01",
        split_mode: str = "auto",
        validation_fraction: float = 0.60,
        purge_periods: int | None = None,
        embargo_periods: int | None = None,
        purge_unit: str | None = None,
        factor_category: str = "",
        strategy_geometry: str | EvaluationGeometry | None = None,
        alpha_signal_col: str | None = None,
        lot_mode: str = "auto",
    ):
        print(f"🔬 Evaluating Candidate: {factor_id}")
        
        original_len = len(df)
        is_intraday = infer_frame_frequency(df) in {"intraday", "tick"}
        if "initial_capital" not in df.attrs:
            df = attach_capital_attrs(
                df,
                resolve_execution_capital(asset_class=self.asset_class),
            )
        if "min_trade_weight_delta" not in df.attrs:
            df = attach_trade_policy_attrs(
                df,
                resolve_execution_trade_policy(),
            )
        integer_lots = self._resolve_integer_lots(df, lot_mode)
        resolved_lot_mode = "integer" if integer_lots else "fractional"
        df.attrs["execution_lot_mode_requested"] = str(lot_mode or "auto")
        df.attrs["execution_lot_mode"] = resolved_lot_mode

        if is_intraday:
            print("   🧱 Liquidity Filter: skipped for intraday/tick data.")
        elif 'close' in df.columns and 'volume' in df.columns:
            df['dollar_volume'] = df['close'] * df['volume']
            df = df[df['dollar_volume'] >= 50_000_000]
            dropped_pct = (original_len - len(df)) / original_len * 100
            print(f"   🧱 Liquidity Filter: Dropped {dropped_pct:.1f}% of data (Micro-caps removed).")
        else:
            print("   ⚠️ WARNING: 'close' or 'volume' columns missing. Cannot enforce Liquidity Wall.")

        alpha_signal_col = alpha_signal_col or self._select_alpha_signal_col(df)
        if alpha_signal_col is None:
            raise ValueError("Evaluator requires one of factor_score, raw_signal, signal, or final_target_weight.")

        split_gap_policy = self._resolve_split_gap_policy(
            df,
            purge_periods=purge_periods,
            embargo_periods=embargo_periods,
            purge_unit=purge_unit,
        )
        split_result = build_chronological_split(
            df,
            split_date=split_date,
            crisis_period=crisis_period,
            signal_col=alpha_signal_col,
            return_col="forward_return",
            mode=split_mode,
            validation_fraction=validation_fraction,
            **split_gap_policy,
        )
        validation_data = split_result.validation_data
        holdout_data = split_result.holdout_data
        crisis_data = split_result.crisis_data

        metric_result = self.metric_evaluator.evaluate(
            factor_id=factor_id,
            df=df,
            validation_data=validation_data,
            holdout_data=holdout_data,
            crisis_data=crisis_data,
            signal_col=alpha_signal_col,
            return_col="forward_return",
            category=factor_category,
            explicit_geometry=strategy_geometry,
        )
        val_ic = metric_result.validation_ic
        holdout_ic = metric_result.holdout_ic
        crisis_ic = metric_result.crisis_ic
        stat_evidence = self.statistical_tester.evaluate(
            holdout_data,
            signal_col=alpha_signal_col,
            return_col="forward_return",
            geometry=metric_result.geometry,
        )
        
        turnover_rate = np.random.uniform(0.1, 0.4) 

        print(
            f"   -> Alpha Geometry: {metric_result.geometry.value} "
            f"({metric_result.metric_name}, signal={metric_result.signal_col})"
        )
        print(f"   -> Validation IC: {self._format_metric(val_ic)}")
        print(f"   -> Holdout IC:    {self._format_metric(holdout_ic)}")
        print(f"   -> Crisis IC:     {self._format_metric(crisis_ic)}")
        print(
            "   -> Holdout p-value: "
            f"{self._format_metric(stat_evidence.raw_p_value)} "
            f"({stat_evidence.test_method}, n={stat_evidence.metric_observations:,})"
        )
        print(
            "   -> Split: "
            f"{split_result.split_mode} @ {split_result.split_boundary} "
            f"(val={split_result.validation_rows:,}, "
            f"holdout={split_result.holdout_rows:,}, "
            f"crisis={split_result.crisis_rows:,})"
        )
        if split_result.purged_rows or split_result.embargoed_rows:
            print(
                "   -> Split Gap: "
                f"purge={split_result.purge_periods} {split_result.purge_unit}, "
                f"embargo={split_result.embargo_periods} {split_result.purge_unit} "
                f"(dropped val={split_result.purged_rows:,}, "
                f"holdout={split_result.embargoed_rows:,})"
            )
        if metric_result.holdout_hit_rate is not None:
            print(f"   -> Holdout Hit Rate: {metric_result.holdout_hit_rate:.2%}")
        print(f"   -> Lot Sizing: {resolved_lot_mode} lots (requested={lot_mode})")
        
        failure_code = "NONE"
        suggested_action = "N/A"

        if len(df) < (original_len * 0.1): 
            failure_code = "untradable_illiquidity"
            suggested_action = "Factor relies on micro-caps. Redesign for large-cap dynamics."
        elif np.isnan(val_ic) or np.isnan(holdout_ic):
            if metric_result.geometry == EvaluationGeometry.CROSS_SECTIONAL:
                failure_code = "cross_sectional_collapse"
                suggested_action = "Rank IC is NaN. Ensure each date has enough assets and non-constant scores."
            else:
                failure_code = "time_series_insufficient_data"
                suggested_action = "Pearson IC is NaN. Ensure enough non-constant signal and forward-return observations per asset."
        elif np.isfinite(crisis_ic) and crisis_ic < -0.01:
            failure_code = "crisis_failure"
            suggested_action = "Factor blows up during liquidity crunches. Needs hedging."
        elif holdout_ic < 0:
            failure_code = "holdout_not_positive"
            suggested_action = "Factor decayed completely Out-Of-Sample. Overfit."
        elif np.isfinite(val_ic) and val_ic > 0 and holdout_ic < (val_ic * 0.4):
            failure_code = "severe_ic_decay"
            suggested_action = "Reduce formula complexity. Remove hardcoded parameters."

        data_tradability = str(df.attrs.get("data_tradability", "")).strip().lower()
        if failure_code == "NONE" and data_tradability in {"research_proxy", "unknown"}:
            failure_code = "proxy_data_requires_contract_validation"
            suggested_action = (
                "This run used index/ambiguous futures data. Re-run on tradable main-contract or tick data "
                "before paper-trading promotion."
            )

        if failure_code != "NONE":
            print(f"   ⚠️ DIAGNOSTIC FLAG: [{failure_code.upper()}]")
        else:
            print(f"   🟢 PASSED ALL INSTITUTIONAL HURDLES.")
            
        self._log_to_db(
            factor_id,
            val_ic,
            holdout_ic,
            crisis_ic,
            turnover_rate,
            failure_code,
            suggested_action,
            df,
            metric_result,
            split_result,
            stat_evidence,
        )

    @staticmethod
    def _select_alpha_signal_col(df: pd.DataFrame) -> str | None:
        factor_contract = df.attrs.get("factor_contract", {})
        if not isinstance(factor_contract, dict):
            factor_contract = {}
        preferred = df.attrs.get("alpha_signal_col") or factor_contract.get("alpha_signal_col")
        if preferred:
            if preferred not in df.columns:
                raise ValueError(f"Factor contract alpha_signal_col={preferred!r} is missing from evaluator frame.")
            return str(preferred)
        return next(
            (
                col for col in ['factor_score', 'raw_signal', 'signal', 'final_target_weight', 'target_weight']
                if col in df.columns
            ),
            None,
        )

    @staticmethod
    def _format_metric(value: float) -> str:
        return "N/A" if not np.isfinite(value) else f"{value:.4f}"

    def _vertical_trial_metadata(
        self,
        df: pd.DataFrame,
        *,
        universe_size: int,
        traded_tickers: str,
    ) -> dict[str, str]:
        market_vertical = self._attr_text(df, "market_vertical") or self.asset_class
        source_path = (
            self._attr_text(df, "source_path")
            or self._attr_text(df, "data_file")
            or self._attr_text(df, "dataset_path")
        )
        dataset_id = self._attr_text(df, "dataset_id")
        if not dataset_id and source_path:
            dataset_id = stable_trial_hash({"source_path": source_path})
        if not dataset_id:
            dataset_id = stable_trial_hash(
                {
                    "market_vertical": market_vertical,
                    "rows": len(df),
                    "date_min": self._date_bound(df, "min"),
                    "date_max": self._date_bound(df, "max"),
                }
            )

        tickers = (
            sorted(df["ticker"].dropna().astype(str).unique().tolist())
            if "ticker" in df.columns
            else []
        )
        universe_id = self._attr_text(df, "universe_id") or stable_trial_hash(
            {
                "market_vertical": market_vertical,
                "universe_size": universe_size,
                "tickers": tickers,
            }
        )

        data_vendor = self._attr_text(df, "data_vendor")
        if not data_vendor:
            data_vendor = "local_parquet" if source_path else "unknown"

        return {
            "market_vertical": market_vertical,
            "dataset_id": dataset_id,
            "universe_id": universe_id,
            "data_frequency": self._attr_text(df, "data_frequency")
            or self._infer_data_frequency(df),
            "dataset_role": self._attr_text(df, "dataset_role") or "unknown",
            "data_tradability": self._attr_text(df, "data_tradability") or "unknown",
            "data_price_source": self._attr_text(df, "data_price_source") or "unknown",
            "data_roll_model": self._attr_text(df, "data_roll_model") or "unknown",
            "data_liquidity_model": self._attr_text(df, "data_liquidity_model") or "unknown",
            "data_execution_reality": self._attr_text(df, "data_execution_reality") or "unknown",
            "data_vendor": data_vendor,
            "execution_assumption": self._attr_text(df, "execution_assumption")
            or self._infer_execution_assumption(df),
            "traded_tickers": traded_tickers,
        }

    @staticmethod
    def _attr_text(df: pd.DataFrame, key: str) -> str | None:
        value = df.attrs.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _date_bound(df: pd.DataFrame, method: str) -> str:
        if "date" not in df.columns:
            return ""
        values = pd.to_datetime(df["date"], errors="coerce").dropna()
        if values.empty:
            return ""
        value = values.min() if method == "min" else values.max()
        return value.isoformat()

    @staticmethod
    def _infer_data_frequency(df: pd.DataFrame) -> str:
        return infer_frame_frequency(df)

    @staticmethod
    def _infer_execution_assumption(df: pd.DataFrame) -> str:
        if "forward_return" not in df.columns:
            return "unknown"
        if AlphaEvaluator._infer_data_frequency(df) in {"intraday", "tick"}:
            return "bar_signal_next_bar"
        if "open" in df.columns:
            return "close_signal_next_open_to_close"
        return "close_to_close_fallback"

    def _resolve_integer_lots(self, df: pd.DataFrame, lot_mode: str | None) -> bool:
        mode = str(lot_mode or df.attrs.get("execution_lot_mode_requested") or "auto").strip().lower()
        if mode == "integer":
            return True
        if mode == "fractional":
            return False
        if mode != "auto":
            print(f"   ⚠️ Unknown lot_mode={lot_mode!r}; falling back to auto.")

        market_vertical = (
            self._attr_text(df, "market_vertical")
            or self._attr_text(df, "asset_class")
            or self.asset_class
        ).upper()
        return "FUTURES" in market_vertical

    @staticmethod
    def _resolve_split_gap_policy(
        df: pd.DataFrame,
        *,
        purge_periods: int | None,
        embargo_periods: int | None,
        purge_unit: str | None,
    ) -> dict[str, object]:
        attrs_policy = df.attrs.get("split_policy", {})
        if not isinstance(attrs_policy, dict):
            attrs_policy = {}
        factor_params = df.attrs.get("factor_params", {})
        if not isinstance(factor_params, dict):
            factor_params = {}

        default_purge = attrs_policy.get(
            "purge_periods",
            factor_params.get("purge_periods", factor_params.get("label_horizon_periods", 0)),
        )
        default_embargo = attrs_policy.get(
            "embargo_periods",
            factor_params.get("embargo_periods", 0),
        )
        default_unit = attrs_policy.get(
            "purge_unit",
            factor_params.get("purge_unit", "auto"),
        )
        return {
            "purge_periods": default_purge if purge_periods is None else purge_periods,
            "embargo_periods": default_embargo if embargo_periods is None else embargo_periods,
            "purge_unit": default_unit if purge_unit is None else purge_unit,
        }

    @staticmethod
    def _refresh_multiple_testing_adjustments(conn: sqlite3.Connection, research_family: str):
        trials = pd.read_sql_query(
            """
            SELECT trial_id, run_id, research_family, trial_signature, raw_p_value
            FROM research_trials
            WHERE research_family = ?
            """,
            conn,
            params=(research_family,),
        )
        if trials.empty:
            return

        trial_count = int(trials["trial_signature"].nunique())
        latest_by_signature = (
            trials.sort_values("trial_id")
            .drop_duplicates("trial_signature", keep="last")
            .set_index("trial_signature")
        )
        raw_p_values = pd.to_numeric(latest_by_signature["raw_p_value"], errors="coerce")
        holm_values = holm_bonferroni_adjust(raw_p_values)
        fdr_values = benjamini_hochberg_q_values(raw_p_values)

        for signature, raw_p_value in raw_p_values.items():
            bonferroni = bonferroni_p_value(float(raw_p_value), trial_count)
            holm = float(holm_values.loc[signature]) if signature in holm_values.index else np.nan
            fdr_q = float(fdr_values.loc[signature]) if signature in fdr_values.index else np.nan
            significance = significance_label(float(raw_p_value), bonferroni)
            conn.execute(
                """
                UPDATE research_trials
                SET trial_count_m = ?,
                    bonferroni_p_value = ?,
                    holm_p_value = ?,
                    fdr_q_value = ?,
                    significance = ?
                WHERE research_family = ? AND trial_signature = ?
                """,
                (trial_count, bonferroni, holm, fdr_q, significance, research_family, signature),
            )
            conn.execute(
                """
                UPDATE backtest_runs
                SET stat_trial_count = ?,
                    stat_adjusted_p_value = ?,
                    stat_holm_p_value = ?,
                    stat_fdr_q_value = ?,
                    stat_significance = ?
                WHERE stat_research_family = ? AND stat_trial_signature = ?
                """,
                (trial_count, bonferroni, holm, fdr_q, significance, research_family, signature),
            )
        conn.commit()

    def _log_to_db(
        self,
        factor_id,
        val_ic,
        holdout_ic,
        crisis_ic,
        turnover_rate,
        failure_code,
        suggested_action,
        df,
        metric_result,
        split_result,
        stat_evidence,
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        
        cursor.execute("SELECT factor_id FROM factors WHERE factor_id = ?", (factor_id,))
        if not cursor.fetchone():
            cursor.execute('''
                INSERT INTO factors (factor_id, name, category, economic_rationale) 
                VALUES (?, ?, ?, ?)
            ''', (factor_id, f"Auto-Gen {factor_id}", "Experimental", "Pending human thesis."))
        
        cursor.execute("SELECT MAX(round_number) FROM backtest_runs WHERE factor_id = ?", (factor_id,))
        res = cursor.fetchone()[0]
        round_number = (res + 1) if res else 1
        
        # --- NEW: THE UNIVERSE AUDIT ---
        universe_size = df['ticker'].nunique() if 'ticker' in df.columns else 0
        traded_tickers_str = "ALL"

        trade_audit_col = next(
            (
                col for col in ['target_weight', 'final_target_weight', 'signal', 'factor_score']
                if col in df.columns
            ),
            None,
        )
        if trade_audit_col is not None:
            audit_weights = pd.to_numeric(df[trade_audit_col], errors='coerce').fillna(0.0)
            traded_mask = audit_weights.abs() > 1e-12
            traded_list = df.loc[traded_mask, 'ticker'].unique()
            if 0 < len(traded_list) < universe_size:
                traded_tickers_str = ",".join(traded_list)
            elif len(traded_list) == 0:
                traded_tickers_str = "NONE"
        # -------------------------------
        vertical_metadata = self._vertical_trial_metadata(
            df,
            universe_size=universe_size,
            traded_tickers=traded_tickers_str,
        )

        factor_params = df.attrs.get("factor_params", {})
        factor_event_stats = df.attrs.get("factor_event_stats", {})
        factor_contract = df.attrs.get("factor_contract", {})
        if not isinstance(factor_contract, dict):
            factor_contract = {}
        execution_lot_mode = str(df.attrs.get("execution_lot_mode", ""))
        execution_lot_mode_requested = str(df.attrs.get("execution_lot_mode_requested", "auto"))
        initial_capital = float(df.attrs.get("initial_capital", 1_000_000.0) or 1_000_000.0)
        capital_currency = str(df.attrs.get("capital_currency", "USD") or "USD")
        capital_profile = str(df.attrs.get("capital_profile", "default") or "default")
        capital_source = str(df.attrs.get("capital_source", "fallback") or "fallback")
        min_trade_weight_delta = float(
            df.attrs.get("min_trade_weight_delta", DEFAULT_MIN_TRADE_WEIGHT_DELTA)
            if df.attrs.get("min_trade_weight_delta", DEFAULT_MIN_TRADE_WEIGHT_DELTA) is not None
            else DEFAULT_MIN_TRADE_WEIGHT_DELTA
        )
        min_trade_weight_delta = max(min_trade_weight_delta, 0.0)
        min_trade_weight_delta_source = str(
            df.attrs.get("min_trade_weight_delta_source", "execution_default_dust_tolerance")
            or "execution_default_dust_tolerance"
        )
        factor_params_json = json.dumps(factor_params, sort_keys=True)
        selected_product = str(df.attrs.get("selected_product", ""))
        selected_symbol = str(df.attrs.get("selected_symbol", ""))
        raw_event_count = int(factor_event_stats.get("raw_events", 0) or 0)
        quality_event_count = int(factor_event_stats.get("quality_events", 0) or 0)
        throttled_event_count = int(factor_event_stats.get("throttled_events", 0) or 0)
        active_tick_count = int(factor_event_stats.get("active_ticks", 0) or 0)
        research_family = str(df.attrs.get("research_family") or factor_id)
        params_hash = stable_trial_hash({"factor_params": factor_params})
        trial_signature_payload = {
            "factor_id": factor_id,
            "research_family": research_family,
            "factor_params": factor_params,
            "asset_class": self.asset_class,
            "market_vertical": vertical_metadata["market_vertical"],
            "dataset_id": vertical_metadata["dataset_id"],
            "universe_id": vertical_metadata["universe_id"],
            "data_frequency": vertical_metadata["data_frequency"],
            "dataset_role": vertical_metadata["dataset_role"],
            "data_tradability": vertical_metadata["data_tradability"],
            "data_price_source": vertical_metadata["data_price_source"],
            "data_roll_model": vertical_metadata["data_roll_model"],
            "data_liquidity_model": vertical_metadata["data_liquidity_model"],
            "data_execution_reality": vertical_metadata["data_execution_reality"],
            "data_vendor": vertical_metadata["data_vendor"],
            "execution_assumption": vertical_metadata["execution_assumption"],
            "factor_contract_source": factor_contract.get("contract_source", ""),
            "alpha_signal_col": factor_contract.get("alpha_signal_col", ""),
            "execution_weight_col": factor_contract.get("execution_weight_col", ""),
            "execution_mode": factor_contract.get("execution_mode", ""),
            "execution_lot_mode": execution_lot_mode,
            "execution_lot_mode_requested": execution_lot_mode_requested,
            "initial_capital": initial_capital,
            "capital_currency": capital_currency,
            "capital_profile": capital_profile,
            "capital_source": capital_source,
            "min_trade_weight_delta": min_trade_weight_delta,
            "min_trade_weight_delta_source": min_trade_weight_delta_source,
            "execution_lag": factor_contract.get("execution_lag", ""),
            "return_assumption": factor_contract.get("return_assumption", ""),
            "selected_product": selected_product,
            "selected_symbol": selected_symbol,
            "split_mode": split_result.split_mode,
            "split_boundary": split_result.split_boundary,
            "purge_periods": split_result.purge_periods,
            "embargo_periods": split_result.embargo_periods,
            "purge_unit": split_result.purge_unit,
            "purged_rows": split_result.purged_rows,
            "embargoed_rows": split_result.embargoed_rows,
            "evaluation_geometry": metric_result.geometry.value,
            "metric_name": metric_result.metric_name,
            "universe_size": universe_size,
            "traded_tickers": traded_tickers_str,
            "raw_event_count": raw_event_count,
            "quality_event_count": quality_event_count,
            "active_tick_count": active_tick_count,
        }
        trial_signature = stable_trial_hash(trial_signature_payload)

        cursor.execute('''
            INSERT INTO backtest_runs 
            (run_id, factor_id, round_number, validation_ic, holdout_ic, crisis_ic, turnover_rate,
             asset_class, market_vertical, dataset_id, universe_id, data_frequency,
             data_vendor, execution_assumption, universe_size, traded_tickers,
             evaluation_geometry, ic_metric,
             validation_hit_rate, holdout_hit_rate, crisis_hit_rate, split_mode,
             split_boundary, validation_rows, holdout_rows, crisis_rows,
             purge_periods, embargo_periods, purge_unit, purged_rows, embargoed_rows,
             factor_params,
             selected_product, selected_symbol, raw_event_count, quality_event_count,
             throttled_event_count, active_tick_count, stat_raw_p_value, stat_metric_p_value,
             stat_hit_rate_p_value, stat_metric_observations, stat_hit_rate_observations,
             stat_test_method, stat_research_family, stat_trial_signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            factor_id,
            round_number,
            val_ic,
            holdout_ic,
            crisis_ic,
            turnover_rate,
            self.asset_class,
            vertical_metadata["market_vertical"],
            vertical_metadata["dataset_id"],
            vertical_metadata["universe_id"],
            vertical_metadata["data_frequency"],
            vertical_metadata["data_vendor"],
            vertical_metadata["execution_assumption"],
            universe_size,
            traded_tickers_str,
            metric_result.geometry.value,
            metric_result.metric_name,
            metric_result.validation_hit_rate,
            metric_result.holdout_hit_rate,
            metric_result.crisis_hit_rate,
            split_result.split_mode,
            split_result.split_boundary,
            split_result.validation_rows,
            split_result.holdout_rows,
            split_result.crisis_rows,
            split_result.purge_periods,
            split_result.embargo_periods,
            split_result.purge_unit,
            split_result.purged_rows,
            split_result.embargoed_rows,
            factor_params_json,
            selected_product,
            selected_symbol,
            raw_event_count,
            quality_event_count,
            throttled_event_count,
            active_tick_count,
            stat_evidence.raw_p_value,
            stat_evidence.metric_p_value,
            stat_evidence.hit_rate_p_value,
            stat_evidence.metric_observations,
            stat_evidence.hit_rate_observations,
            stat_evidence.test_method,
            research_family,
            trial_signature,
        ))
        cursor.execute('''
            INSERT OR REPLACE INTO research_trials
            (run_id, factor_id, research_family, trial_signature, params_hash,
             asset_class, market_vertical, dataset_id, universe_id, data_frequency,
             data_vendor, execution_assumption, evaluation_geometry, metric_name, raw_p_value,
             metric_p_value, hit_rate_p_value, trial_count_m, significance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            factor_id,
            research_family,
            trial_signature,
            params_hash,
            self.asset_class,
            vertical_metadata["market_vertical"],
            vertical_metadata["dataset_id"],
            vertical_metadata["universe_id"],
            vertical_metadata["data_frequency"],
            vertical_metadata["data_vendor"],
            vertical_metadata["execution_assumption"],
            metric_result.geometry.value,
            metric_result.metric_name,
            stat_evidence.raw_p_value,
            stat_evidence.metric_p_value,
            stat_evidence.hit_rate_p_value,
            1,
            "pending",
        ))
        cursor.execute(
            """
            UPDATE backtest_runs
            SET factor_contract_source = ?,
                alpha_signal_col = ?,
                execution_weight_col = ?,
                execution_mode = ?,
                execution_lot_mode = ?,
                execution_lot_mode_requested = ?,
                initial_capital = ?,
                capital_currency = ?,
                capital_profile = ?,
                capital_source = ?,
                min_trade_weight_delta = ?,
                min_trade_weight_delta_source = ?,
                execution_lag = ?,
                return_assumption = ?,
                dataset_role = ?,
                data_tradability = ?,
                data_price_source = ?,
                data_roll_model = ?,
                data_liquidity_model = ?,
                data_execution_reality = ?
            WHERE run_id = ?
            """,
            (
                factor_contract.get("contract_source", ""),
                factor_contract.get("alpha_signal_col", ""),
                factor_contract.get("execution_weight_col", ""),
                factor_contract.get("execution_mode", ""),
                execution_lot_mode,
                execution_lot_mode_requested,
                initial_capital,
                capital_currency,
                capital_profile,
                capital_source,
                min_trade_weight_delta,
                min_trade_weight_delta_source,
                factor_contract.get("execution_lag", ""),
                factor_contract.get("return_assumption", ""),
                vertical_metadata["dataset_role"],
                vertical_metadata["data_tradability"],
                vertical_metadata["data_price_source"],
                vertical_metadata["data_roll_model"],
                vertical_metadata["data_liquidity_model"],
                vertical_metadata["data_execution_reality"],
                run_id,
            ),
        )
        cursor.execute(
            """
            UPDATE research_trials
            SET factor_contract_source = ?,
                alpha_signal_col = ?,
                execution_weight_col = ?,
                execution_mode = ?,
                execution_lag = ?,
                return_assumption = ?,
                dataset_role = ?,
                data_tradability = ?,
                data_price_source = ?,
                data_roll_model = ?,
                data_liquidity_model = ?,
                data_execution_reality = ?
            WHERE run_id = ?
            """,
            (
                factor_contract.get("contract_source", ""),
                factor_contract.get("alpha_signal_col", ""),
                factor_contract.get("execution_weight_col", ""),
                factor_contract.get("execution_mode", ""),
                factor_contract.get("execution_lag", ""),
                factor_contract.get("return_assumption", ""),
                vertical_metadata["dataset_role"],
                vertical_metadata["data_tradability"],
                vertical_metadata["data_price_source"],
                vertical_metadata["data_roll_model"],
                vertical_metadata["data_liquidity_model"],
                vertical_metadata["data_execution_reality"],
                run_id,
            ),
        )
        
        if failure_code != "NONE":
            cursor.execute('''
                INSERT INTO diagnostics (run_id, failure_code, suggested_action)
                VALUES (?, ?, ?)
            ''', (run_id, failure_code, suggested_action))
            
        conn.commit()
        self._refresh_multiple_testing_adjustments(conn, research_family)
        conn.close()
        print(f"   💾 Logged metadata to Database successfully.")

        print("   -> Calculating Net Portfolio Returns (Dynamic Margin & Square-Root TCA)...")
        execution_signal_col = 'signal' if 'signal' in df.columns else 'factor_score'
        clean_df = df.dropna(subset=[execution_signal_col, 'forward_return']).copy()
        
        if not clean_df.empty:
            from benchmark_engine.benchmark_factory import BenchmarkFactory
            
            # Pass finalized Risk Desk weights through without applying another
            # leverage multiplier. If no final signal exists, preserve the old
            # factor_score scaling behavior for direct evaluator use.
            execution_max_leverage = 1.0 if execution_signal_col == 'signal' else 5.0
            print(
                f"   -> Executing '{execution_signal_col}' "
                f"with {execution_max_leverage:.1f}x leverage multiplier "
                f"and {execution_lot_mode or 'fractional'} lot sizing "
                f"on {initial_capital:,.0f} {capital_currency}; "
                f"min_trade_weight_delta={min_trade_weight_delta:g}."
            )
            desk = ExecutionDesk(
                asset_class=self.asset_class,
                max_leverage=execution_max_leverage,
                integer_lots=execution_lot_mode == "integer",
                initial_capital=initial_capital,
                capital_currency=capital_currency,
                min_trade_weight_delta=min_trade_weight_delta,
            )
            
            benchmark_engine = BenchmarkFactory.create_benchmark("BUY_AND_HOLD", raw_df=clean_df)
            # 🛠️ FIX: Pass the entire DataFrame so the Smart Benchmark can audit the tickers
            b_df = benchmark_engine.generate(clean_df)
            
            portfolio, avg_turnover, asset_df = desk.run_backtest(
                clean_df,
                benchmark_df=b_df,
                signal_col=execution_signal_col,
            )
            
            daily_returns = portfolio['net_return'].fillna(0).values
            if len(daily_returns) > 0:
                days_count = len(portfolio)
                ann_return = (portfolio['cum_net_equity'].iloc[-1]) ** (252 / days_count) - 1 if days_count > 0 else 0
                ann_vol = np.std(daily_returns) * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol != 0 else 0
                
                rolling_max = portfolio['cum_net_equity'].cummax()
                max_dd = ((portfolio['cum_net_equity'] - rolling_max) / rolling_max).min()
            else:
                ann_return, sharpe, max_dd = 0.0, 0.0, 0.0

            avg_daily_cost_bps = float(portfolio.get('daily_cost_bps', pd.Series([0.0])).mean())
            total_exchange_fees = float(portfolio.get('daily_exchange_fee', pd.Series([0.0])).sum())
            total_slippage_cost = float(portfolio.get('daily_slippage_cost', pd.Series([0.0])).sum())
            total_execution_cost = float(portfolio.get('daily_total_cost', pd.Series([0.0])).sum())
            lot_constrained_rate = float(portfolio.get('lot_constrained_rate', pd.Series([0.0])).mean())
            fee_constrained_rate = float(portfolio.get('fee_constrained_rate', pd.Series([0.0])).mean())
            avg_one_lot_weight = float(portfolio.get('avg_one_lot_weight', pd.Series([0.0])).mean())
            avg_abs_rounding_error_weight = float(
                portfolio.get('avg_abs_rounding_error_weight', pd.Series([0.0])).mean()
            )
            avg_round_trip_fee_bps = float(portfolio.get('avg_round_trip_fee_bps', pd.Series([0.0])).mean())
            avg_round_trip_fee_ticks = float(portfolio.get('avg_round_trip_fee_ticks', pd.Series([0.0])).mean())
            sharpe_p_value, sharpe_p_obs = sharpe_p_value_from_returns(portfolio.get('net_return', pd.Series(dtype=float)))

            returns_dir = os.path.join(self.logs_dir, "returns")
            os.makedirs(returns_dir, exist_ok=True)
            file_path = os.path.join(returns_dir, f"returns_{run_id}.csv")

            export_cols = [
                'date',
                'gross_return',
                'net_return',
                'benchmark_return',
                'daily_turnover',
                'daily_slippage_cost',
                'daily_exchange_fee',
                'daily_total_cost',
                'daily_cost_bps',
                'portfolio_leverage',
                'initial_capital',
                'capital_currency',
                'min_trade_weight_delta',
                'lot_constrained_rate',
                'fee_constrained_rate',
                'avg_one_lot_weight',
                'avg_abs_rounding_error_weight',
                'avg_round_trip_fee_bps',
                'avg_round_trip_fee_ticks',
            ]
            portfolio['initial_capital'] = initial_capital
            portfolio['capital_currency'] = capital_currency
            portfolio['min_trade_weight_delta'] = min_trade_weight_delta
            for col in export_cols:
                if col not in portfolio.columns:
                    portfolio[col] = 1.0 if col == 'portfolio_leverage' else 0.0
            
            portfolio[export_cols].to_csv(file_path, index=False)
            print(f"   💾 Saved Ironclad portfolio returns to {file_path}")

            update_conn = sqlite3.connect(self.db_path)
            update_conn.cursor().execute("""
                UPDATE backtest_runs
                SET turnover_rate = ?,
                    annualized_return = ?,
                    max_drawdown = ?,
                    sharpe_ratio = ?,
                    returns_file_path = ?,
                    avg_daily_cost_bps = ?,
                    total_exchange_fees = ?,
                    total_slippage_cost = ?,
                    total_execution_cost = ?,
                    lot_constrained_rate = ?,
                    fee_constrained_rate = ?,
                    avg_one_lot_weight = ?,
                    avg_abs_rounding_error_weight = ?,
                    avg_round_trip_fee_bps = ?,
                    avg_round_trip_fee_ticks = ?,
                    stat_sharpe_p_value = ?
                WHERE run_id = ?
            """, (
                avg_turnover,
                ann_return,
                max_dd,
                sharpe,
                file_path,
                avg_daily_cost_bps,
                total_exchange_fees,
                total_slippage_cost,
                total_execution_cost,
                lot_constrained_rate,
                fee_constrained_rate,
                avg_one_lot_weight,
                avg_abs_rounding_error_weight,
                avg_round_trip_fee_bps,
                avg_round_trip_fee_ticks,
                sharpe_p_value,
                run_id,
            ))
            update_conn.cursor().execute("""
                UPDATE research_trials
                SET sharpe_p_value = ?
                WHERE run_id = ?
            """, (sharpe_p_value, run_id))
            update_conn.commit()
            update_conn.close()
            if np.isfinite(sharpe_p_value):
                print(f"   -> Executed-return p-value: {sharpe_p_value:.4f} (n={sharpe_p_obs:,})")

            if (
                failure_code == "NONE"
                and np.isfinite(holdout_ic)
                and holdout_ic >= 0
                and (ann_return < 0 or sharpe < 0)
                and avg_daily_cost_bps > 1.0
            ):
                if total_exchange_fees > total_slippage_cost:
                    cost_detail = (
                        "Holdout alpha is positive but exchange fees dominate net performance. "
                        "Try longer holding periods, wider cooldowns, lower signal frequency, or a "
                        "contract with a better tick-value-to-fee ratio."
                    )
                else:
                    cost_detail = (
                        "Holdout alpha is positive but execution slippage dominates net performance. "
                        "Try lower turnover, lower participation, wider factor cooldowns/dead zones, or slower execution."
                    )
                with sqlite3.connect(self.db_path) as diag_conn:
                    diag_conn.execute(
                        '''
                        INSERT INTO diagnostics (run_id, failure_code, suggested_action)
                        VALUES (?, ?, ?)
                        ''',
                        (run_id, "execution_cost_bleed", cost_detail),
                    )
                    diag_conn.commit()
                print("   ⚠️ DIAGNOSTIC FLAG: [EXECUTION_COST_BLEED]")

            lot_diagnostics: list[tuple[str, str]] = []
            if lot_constrained_rate > 0.25 or avg_one_lot_weight > 0.01:
                lot_diagnostics.append((
                    "integer_lot_constraint",
                    (
                        "Integer contract sizing materially changes target weights. "
                        f"One lot averages {avg_one_lot_weight:.2%} of capital and "
                        f"{lot_constrained_rate:.1%} of rows are lot-constrained. "
                        "Use more capital, lower target turnover, wider position bands, or a more granular contract."
                    ),
                ))
            if fee_constrained_rate > 0.25 or avg_round_trip_fee_ticks > 1.0:
                lot_diagnostics.append((
                    "fee_to_tick_constraint",
                    (
                        "Exchange fees are large relative to the contract tick value. "
                        f"Average round-trip fee is {avg_round_trip_fee_ticks:.2f} ticks "
                        f"({avg_round_trip_fee_bps:.2f} bps), and {fee_constrained_rate:.1%} "
                        "of rows are fee-constrained. Prefer longer holding periods or lower-frequency signals."
                    ),
                ))
            if lot_diagnostics:
                with sqlite3.connect(self.db_path) as diag_conn:
                    diag_conn.executemany(
                        '''
                        INSERT OR IGNORE INTO diagnostics (run_id, failure_code, suggested_action)
                        VALUES (?, ?, ?)
                        ''',
                        [(run_id, code, message) for code, message in lot_diagnostics],
                    )
                    diag_conn.commit()
                for code, _ in lot_diagnostics:
                    print(f"   ⚠️ DIAGNOSTIC FLAG: [{code.upper()}]")

            # --- 🛠️ INJECTED TERMINAL TEAR SHEET ---
            desk.print_tearsheet(portfolio, factor_id)

            print("   -> Extracting Discrete Trade Ledger for Strategy DNA...")
            trade_df = asset_df[asset_df['weight'].abs() > 1e-12].copy()
            if 'close' not in trade_df.columns:
                trade_df['close'] = 1.0 
            
            total_trades = 0
            if not trade_df.empty:
                trade_df = trade_df.sort_values(['ticker', 'date']).copy()
                trade_df['trade_direction'] = np.sign(trade_df['weight']).astype(int)
                gap_break = trade_df.groupby('ticker')['date'].diff().dt.days.gt(3).fillna(False)
                direction_break = (
                    trade_df.groupby('ticker')['trade_direction']
                    .diff()
                    .fillna(0)
                    .ne(0)
                )
                trade_df['block'] = (gap_break | direction_break).groupby(trade_df['ticker']).cumsum()

                ledger = trade_df.groupby(['ticker', 'block']).agg(
                    entry_time=('date', 'min'), exit_time=('date', 'max'),
                    entry_price=('close', 'first'), exit_price=('close', 'last'),
                    avg_weight=('weight', 'mean'),
                    direction_sign=('trade_direction', 'first')
                ).reset_index()

                price_return = np.where(
                    ledger['entry_price'].abs() > 1e-12,
                    (ledger['exit_price'] - ledger['entry_price']) / ledger['entry_price'],
                    0.0,
                )
                ledger['trade_pnl'] = price_return * ledger['direction_sign']
                
                ledger['holding_period_hours'] = (
                    (ledger['exit_time'] - ledger['entry_time']).dt.total_seconds() / 3600.0
                )
                ledger['holding_period_hours'] = ledger['holding_period_hours'].clip(lower=1 / 3600)
                ledger['win_loss_flag'] = np.where(ledger['trade_pnl'] > 0, 'Win', 'Loss')
                ledger['direction'] = np.where(ledger['avg_weight'] > 0, 'Long', 'Short')
                total_trades = len(ledger)
                
                trade_cols = ['ticker', 'entry_time', 'exit_time', 'holding_period_hours', 
                              'entry_price', 'exit_price', 'trade_pnl', 'win_loss_flag', 'direction']
                
                trades_dir = os.path.join(self.logs_dir, "trades")
                os.makedirs(trades_dir, exist_ok=True)
                trade_path = os.path.join(trades_dir, f"trades_{run_id}.csv")
                ledger[trade_cols].to_csv(trade_path, index=False)
                print(f"   🧾 Saved Trade Ledger to {trade_path}")
            else:
                print("   ⚠️ No discrete trades extracted.")

            update_conn = sqlite3.connect(self.db_path)
            update_conn.cursor().execute("""
                UPDATE backtest_runs
                SET total_trades = ?
                WHERE run_id = ?
            """, (total_trades, run_id))
            update_conn.commit()
            update_conn.close()

if __name__ == "__main__":
    print("Evaluator module safely loaded. OOP updates complete.")
