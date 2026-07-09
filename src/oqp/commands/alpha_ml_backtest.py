from __future__ import annotations

import argparse
import os
import sqlite3
import time

from oqp.research_runtime import alpha_research_runtime_paths
from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical


VALID_ASSETS = sorted(ASSET_TAXONOMY)
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


def build_execution_mode_config(factor_module, args):
    from oqp.research.backtesting import ExecutionModeConfig

    raw_config = getattr(factor_module, "EXECUTION_MODE_CONFIG", {}) or {}
    if not isinstance(raw_config, dict):
        print("⚠️ WARNING: EXECUTION_MODE_CONFIG must be a dict. Ignoring it.")
        raw_config = {}

    config_data = dict(raw_config)
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
    parser.add_argument(
        "--asset",
        required=True,
        help=f"Asset taxonomy class or alias (known: {', '.join(VALID_ASSETS)})",
    )
    parser.add_argument("--factor", required=True, help="Factor module name, e.g. fac_054_XGBoost_Alpha")
    parser.add_argument("--retrain", action="store_true", help="Retrain XGBoost model before backtest")
    parser.add_argument(
        "--execution_mode",
        choices=["auto", "risk_desk", "direct", "statarb"],
        default="auto",
        help="How ML factor scores become executable target weights.",
    )
    parser.add_argument("--max_gross_leverage", type=float, default=None, help="Override execution-mode gross leverage cap.")
    parser.add_argument("--max_weight_per_asset", type=float, default=None, help="Optional per-asset cap before gross scaling.")
    parser.add_argument(
        "--sizing_modules",
        type=str,
        default=None,
        help="Risk-desk allocation modules. Examples: kelly,hrp (default), kelly, hrp, or none.",
    )
    parser.add_argument(
        "--kelly_fraction",
        type=float,
        default=None,
        help="Override fractional Kelly multiplier when the kelly sizing module is enabled.",
    )
    args = parser.parse_args()
    args.asset = normalize_market_vertical(args.asset)
    if args.asset not in ASSET_TAXONOMY:
        parser.error(f"unsupported asset taxonomy class: {args.asset}")

    from oqp.data import DataEngineFactory
    from oqp.research import validate_factor_market_compatibility
    from oqp.research.backtesting import AlphaEvaluator, ExecutionModeFactory
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
    try:
        supported_markets = validate_factor_market_compatibility(
            factor_module,
            args.asset,
            factor_id=getattr(factor_module, "FACTOR_ID", factor_module_name),
        )
    except Exception as e:
        print(f"❌ [FACTOR MARKET ERROR] {e}")
        return
    print(
        "   -> 🧭 Factor Market Gate: "
        f"{'ALL' if '*' in supported_markets else ', '.join(supported_markets)}"
    )

    print("   -> Computing ML factor scores...")
    df = factor_module.compute(df)
    apply_vertical_attrs(df, vertical_attrs)

    execution_mode = (
        args.execution_mode
        if args.execution_mode != "auto"
        else getattr(factor_module, "EXECUTION_MODE", None)
        or df.attrs.get("execution_mode")
        or "risk_desk"
    )
    execution_config = build_execution_mode_config(factor_module, args)
    print(
        "   -> ⚖️ Execution Mode: "
        f"{execution_mode} ({execution_config.sizing_modules or 'no sizing modules'})"
    )
    execution_result = ExecutionModeFactory.create(execution_mode, execution_config).apply(df)
    print(f"   -> {execution_result.detail}")
    print(f"   -> Using '{execution_result.source_col}' as execution input.")
    df = execution_result.df
    apply_vertical_attrs(df, vertical_attrs)

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
