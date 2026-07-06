from __future__ import annotations

import argparse
import os
import sqlite3
import time

from oqp.research_runtime import alpha_research_runtime_paths


VALID_ASSETS = ["EQUITY_US", "EQUITY_CN", "FUTURES_US", "FUTURES_CN"]
ALPHA_RUNTIME_PATHS = alpha_research_runtime_paths()
ALPHA_RUNTIME_ARTIFACT_ROOT = ALPHA_RUNTIME_PATHS.artifact_root
ALPHA_RESEARCH_DB_PATH = ALPHA_RUNTIME_PATHS.db_path
FEATURE_MATRIX_PATH = str(ALPHA_RUNTIME_PATHS.feature_matrix_path)
MODEL_OUTPUT_PATH = str(ALPHA_RUNTIME_PATHS.xgboost_model_output_path)
IMPORTANCE_OUTPUT_PATH = str(ALPHA_RUNTIME_PATHS.xgboost_feature_importance_path)


def infer_data_frequency(df: pd.DataFrame) -> str:
    from oqp.research.backtesting import infer_frame_frequency

    return infer_frame_frequency(df)


def apply_vertical_attrs(df: pd.DataFrame, attrs: dict[str, str]) -> pd.DataFrame:
    for key, value in attrs.items():
        if value and key not in df.attrs:
            df.attrs[key] = value
    return df


def _run_retraining():
    from oqp.research.ml import XGBoostTrainingEngine

    print("   -> [ML ENGINE] Retrain flag detected. Training XGBoost brain...")
    trainer = XGBoostTrainingEngine(
        data_path=FEATURE_MATRIX_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
        importance_output_path=IMPORTANCE_OUTPUT_PATH,
    )
    trainer.run()


def _save_feature_importance_if_available(factor_module, factor_id: str):
    if not hasattr(factor_module, "get_feature_importance"):
        return

    try:
        imp_df = factor_module.get_feature_importance()
    except Exception:
        return
    if imp_df is None or imp_df.empty:
        return

    ALPHA_RESEARCH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ALPHA_RESEARCH_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id FROM backtest_runs
        WHERE factor_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (factor_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return

    run_id = row[0]
    out_dir = os.path.join(os.fspath(ALPHA_RUNTIME_ARTIFACT_ROOT), "feature_importance")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"feature_importance_{run_id}.csv")
    imp_df.to_csv(out_path, index=False)
    print(f"   💾 Saved feature importance -> {out_path}")


def _load_feature_matrix(path: str) -> pd.DataFrame:
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Missing required matrix: {path}\n"
            "   Please run `python feature_engineering.py` first."
        )
    df = pd.read_parquet(path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("❌ [DATA ERROR] ML feature matrix must contain 'date' and 'ticker' columns.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if "forward_return" not in df.columns and "close" in df.columns:
        df["forward_return"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    return df


def main():
    parser = argparse.ArgumentParser(description="Institutional ML Backtest Wrapper")
    parser.add_argument("--asset", required=True, choices=VALID_ASSETS, help="Asset taxonomy class")
    parser.add_argument("--factor", required=True, help="Factor module name, e.g. fac_054_XGBoost_Alpha")
    parser.add_argument("--retrain", action="store_true", help="Retrain XGBoost model before backtest")
    args = parser.parse_args()

    from oqp.data import DataEngineFactory
    from oqp.research.backtesting import AlphaEvaluator, PortfolioOptimizer
    from oqp.research.factors import factor_search_roots, load_factor_module

    factor_module_name = args.factor[:-3] if args.factor.endswith(".py") else args.factor

    print(f"🚀 Booting ML Backtest Engine for: {factor_module_name}")
    start = time.time()

    if args.retrain:
        _run_retraining()

    print(f"   ⚡ Loading ML Feature Matrix: {FEATURE_MATRIX_PATH}")
    df = _load_feature_matrix(FEATURE_MATRIX_PATH)
    data_frequency = infer_data_frequency(df)
    vertical_attrs = {
        "market_vertical": args.asset,
        "source_path": os.path.abspath(FEATURE_MATRIX_PATH),
        "data_file": os.path.abspath(FEATURE_MATRIX_PATH),
        "dataset_id": os.path.splitext(os.path.basename(FEATURE_MATRIX_PATH))[0],
        "data_vendor": "local_feature_matrix",
        "data_frequency": data_frequency,
        "execution_assumption": "close_to_close_fallback",
    }
    apply_vertical_attrs(df, vertical_attrs)
    print(f"   ✅ Matrix loaded successfully. Shape: {df.shape}")

    try:
        factor_module = load_factor_module(factor_module_name)
    except ModuleNotFoundError as exc:
        search_roots = "\n      - ".join(str(path) for path in factor_search_roots())
        raise ModuleNotFoundError(
            f"❌ Could not import factor module {factor_module_name}.py\n"
            f"   Searched:\n      - {search_roots}"
        ) from exc

    print("   -> Computing ML factor scores...")
    df = factor_module.compute(df)
    apply_vertical_attrs(df, vertical_attrs)

    # =========================================================
    # 🏛️ INSTITUTIONAL RISK DESK: PORTFOLIO OPTIMIZATION
    # =========================================================
    print("   -> ⚖️ Routing to Risk Desk (Kelly Sizing & HRP)...")
    optimizer = PortfolioOptimizer(kelly_fraction=0.5, max_weight=0.05)

    # 1. Kelly Volatility Scaling (Scales down bets on highly volatile assets)
    # Prefer the factor's tradable signal, then fall back to raw model output.
    alpha_signal_col = next(
        (col for col in ["signal", "factor_score", "raw_signal"] if col in df.columns),
        None,
    )
    if alpha_signal_col is None:
        raise ValueError("❌ Factor module must output signal, factor_score, or raw_signal.")
    print(f"   -> Using '{alpha_signal_col}' as Risk Desk alpha input.")
    df["risk_signal"] = df[alpha_signal_col].fillna(0.0)
    df = optimizer.kelly.compute_weights(df, signal_col="risk_signal")

    # 2. Hierarchical Risk Parity (HRP) Correlation Budgeting
    print("   -> 🌳 Building HRP Correlation Tree...")
    if 'ret_1d' not in df.columns:
        df['ret_1d'] = df.groupby('ticker')['close'].pct_change()
    
    # Create a wide matrix of returns for the correlation matrix
    returns_wide = df.pivot(index='date', columns='ticker', values='ret_1d').fillna(0)

    # For maximum backtest speed, we compute the structural HRP budget globally across the history.
    # (In live production, you would run this on a rolling 60-day window rebalanced monthly).
    global_hrp_budgets = optimizer.hrp.compute_weights(returns_wide)
    df['hrp_budget'] = df['ticker'].map(global_hrp_budgets).fillna(0.0)

    # 3. Synthesize: Kelly Conviction * HRP Risk Budget
    df['synthesized_weight'] = df['kelly_weight'] * df['hrp_budget']

    # 4. Enforce the Strict Casino Cap (Max 5% per asset, Max 100% Gross Leverage)
    df = optimizer.cap.enforce(df, "synthesized_weight")

    # 5. Route the finalized weights to the Backtester instead of the raw AI score
    df["signal"] = df["final_target_weight"]
    apply_vertical_attrs(df, vertical_attrs)
    # =========================================================

    print("   -> Routing to Evaluator...")
    evaluator = AlphaEvaluator(
        db_path=ALPHA_RESEARCH_DB_PATH,
        logs_dir=ALPHA_RUNTIME_ARTIFACT_ROOT,
        asset_class=args.asset,
    )

    # Keep compatibility with current evaluator signature (requires crisis_period).
    try:
        base_asset_type = args.asset.split("_")[0]
        feed = DataEngineFactory.create_feed(base_asset_type, FEATURE_MATRIX_PATH)
        crisis_period = feed.get_crisis_period()
    except Exception:
        crisis_period = ("2024-01-01", "2024-02-28")

    evaluator.run_evaluation(
        factor_module.FACTOR_ID,
        df,
        crisis_period=crisis_period,
        factor_category=getattr(factor_module, "CATEGORY", ""),
        strategy_geometry=getattr(
            factor_module,
            "EVALUATION_GEOMETRY",
            getattr(factor_module, "STRATEGY_GEOMETRY", None),
        ),
    )
    _save_feature_importance_if_available(factor_module, factor_module.FACTOR_ID)

    elapsed = time.time() - start
    print(f"✅ ML Backtest completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
