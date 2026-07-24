from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

from oqp.research_runtime import alpha_research_runtime_paths
from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical


VALID_ASSETS = sorted(ASSET_TAXONOMY)
ALPHA_RUNTIME_PATHS = alpha_research_runtime_paths()
ALPHA_RUNTIME_ARTIFACT_ROOT = ALPHA_RUNTIME_PATHS.artifact_root
ALPHA_RESEARCH_DB_PATH = ALPHA_RUNTIME_PATHS.db_path
FEATURE_MATRIX_PATH = str(ALPHA_RUNTIME_PATHS.feature_matrix_path)
MODEL_OUTPUT_ROOT = ALPHA_RUNTIME_ARTIFACT_ROOT / "models"
IMPORTANCE_OUTPUT_ROOT = ALPHA_RUNTIME_ARTIFACT_ROOT / "feature_importance"
PREDICTION_OUTPUT_ROOT = ALPHA_RUNTIME_ARTIFACT_ROOT / "predictions"


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


def _model_artifact_paths(
    model_type: str,
    factor_id: str,
) -> tuple[Path, Path, Path]:
    extension = ".json" if model_type == "xgboost" else ".txt"
    stem = f"{factor_id}_{model_type}"
    return (
        MODEL_OUTPUT_ROOT / f"{stem}{extension}",
        IMPORTANCE_OUTPUT_ROOT / f"{stem}.csv",
        PREDICTION_OUTPUT_ROOT / f"{stem}.parquet",
    )


def _validation_config_from_args(args):
    from oqp.research.ml import ValidationConfig

    has_override = any(
        value is not None
        for value in (
            args.validation_mode,
            args.split_date,
            args.min_train_days,
            args.test_window_days,
            args.purge_gap_days,
        )
    )
    if not has_override:
        return None

    mode = args.validation_mode or (
        "fixed_date" if args.split_date else "walk_forward"
    )
    return ValidationConfig(
        mode=mode,
        split_date=(args.split_date or "2024-01-01") if mode == "fixed_date" else None,
        min_train_days=args.min_train_days or 756,
        test_window_days=args.test_window_days or 60,
        purge_gap_days=args.purge_gap_days if args.purge_gap_days is not None else 2,
    )


def _run_retraining(factor_module, args):
    from oqp.research.ml import MLModelFactory, require_model_runtime

    model_type = MLModelFactory.normalize_model_type(args.model)
    runtime_status = require_model_runtime(model_type)
    print(f"   -> [ML ENGINE] {runtime_status.detail} ({model_type})")
    factor_id = getattr(factor_module, "FACTOR_ID", args.factor)
    model_name = f"{factor_id}_{model_type}"
    model_path, importance_path, predictions_path = _model_artifact_paths(
        model_type,
        factor_id,
    )
    kwargs = {
        "model_name": model_name,
        "factor_id": factor_id,
        "asset_class": args.asset,
        "model_output_path": model_path,
        "importance_output_path": importance_path,
        "predictions_output_path": predictions_path,
        "registry_db_path": ALPHA_RESEARCH_DB_PATH,
    }
    target_col = args.target_column or getattr(factor_module, "TARGET_COLUMN", None)
    if target_col:
        kwargs["target_col"] = target_col
    validation_config = _validation_config_from_args(args)
    if validation_config is not None:
        kwargs["validation_config"] = validation_config
    feature_exclusions = getattr(factor_module, "FEATURE_EXCLUSIONS", ()) or ()
    if feature_exclusions:
        kwargs["exclude_features"] = tuple(feature_exclusions)

    print(
        "   -> [ML ENGINE] Retraining "
        f"{model_type} under the shared experiment contract..."
    )
    trainer = MLModelFactory.create_model(
        model_type,
        args.feature_matrix,
        **kwargs,
    )
    result = trainer.run()
    print(
        "   -> [ML ENGINE] Experiment "
        f"{result.experiment_id} registered with "
        f"{result.metrics.get('fold_count', 0)} validation fold(s)."
    )
    return result


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


def _load_feature_matrix(path: str, asset_class: str | None = None) -> pd.DataFrame:
    import pandas as pd

    from oqp.research.ml import (
        infer_feature_matrix_asset_class,
        scope_feature_matrix,
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Missing required matrix: {path}\n"
            "   Please run `python feature_engineering.py` first."
        )
    df = pd.read_parquet(path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("❌ [DATA ERROR] ML feature matrix must contain 'date' and 'ticker' columns.")
    df["date"] = pd.to_datetime(df["date"])
    if asset_class:
        default_asset_class = infer_feature_matrix_asset_class(path)
        df = scope_feature_matrix(
            df,
            asset_class,
            default_asset_class=default_asset_class,
        )
        if df.empty:
            raise ValueError(
                f"❌ [DATA ERROR] Feature matrix has no rows for {asset_class}."
            )
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if "forward_return" not in df.columns and "close" in df.columns:
        df["forward_return"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    return df


def _attach_oos_predictions(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only purged OOS predictions to the historical factor matrix."""

    required = {"date", "ticker", "target", "prediction", "fold"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"ML experiment predictions are missing columns: {missing}")

    prediction_frame = predictions[
        ["date", "ticker", "target", "prediction", "fold"]
    ].copy()
    prediction_frame["date"] = pd.to_datetime(
        prediction_frame["date"],
        errors="coerce",
    )
    prediction_frame["ticker"] = prediction_frame["ticker"].astype(str)
    if prediction_frame.duplicated(["date", "ticker"]).any():
        raise ValueError("ML experiment predictions contain duplicate date/ticker rows.")
    prediction_frame = prediction_frame.rename(
        columns={
            "target": "ml_target",
            "prediction": "ml_prediction",
            "fold": "ml_validation_fold",
        }
    )

    attrs = dict(df.attrs)
    merged = df.merge(
        prediction_frame,
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )
    merged.attrs.update(attrs)
    merged.attrs["ml_oos_prediction_rows"] = int(
        merged["ml_prediction"].notna().sum()
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Institutional ML Backtest Wrapper")
    parser.add_argument(
        "--asset",
        required=True,
        help=f"Asset taxonomy class or alias (known: {', '.join(VALID_ASSETS)})",
    )
    parser.add_argument("--factor", required=True, help="Factor module name, e.g. fac_054_XGBoost_Alpha")
    parser.add_argument(
        "--model",
        choices=["lightgbm", "xgboost"],
        default=None,
        help="Model adapter override. Defaults to factor MODEL_TYPE, then xgboost.",
    )
    parser.add_argument(
        "--feature-matrix",
        default=FEATURE_MATRIX_PATH,
        help="Taxonomy-aware feature matrix used for training and backtesting.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Training target override. Defaults to factor metadata or model config.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train the selected model and register a reproducible experiment first.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["walk_forward", "fixed_date"],
        default=None,
        help="Optional override for the model adapter's validation policy.",
    )
    parser.add_argument("--split-date", default=None, help="Start date for fixed-date validation.")
    parser.add_argument("--min-train-days", type=int, default=None)
    parser.add_argument("--test-window-days", type=int, default=None)
    parser.add_argument("--purge-gap-days", type=int, default=None)
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
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.asset = normalize_market_vertical(args.asset)
    if args.asset not in ASSET_TAXONOMY:
        parser.error(f"unsupported asset taxonomy class: {args.asset}")
    # Stabilize native model libraries before importing factor/model code.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    from oqp.data import DataEngineFactory
    from oqp.research import validate_factor_market_compatibility
    from oqp.research.backtesting import AlphaEvaluator, ExecutionModeFactory
    from oqp.research.factors import factor_search_roots, load_factor_module
    from oqp.research.parameter_schema import attach_factor_parameter_attrs

    factor_module_name = args.factor[:-3] if args.factor.endswith(".py") else args.factor

    print(f"🚀 Booting ML Backtest Engine for: {factor_module_name}")
    start = time.time()

    try:
        factor_module = load_factor_module(factor_module_name)
    except ModuleNotFoundError as exc:
        search_roots = "\n      - ".join(str(path) for path in factor_search_roots())
        raise ModuleNotFoundError(
            f"❌ Could not import factor module {factor_module_name}.py\n"
            f"   Searched:\n      - {search_roots}"
        ) from exc

    args.model = (
        args.model
        or getattr(factor_module, "MODEL_TYPE", None)
        or "xgboost"
    )
    from oqp.research.ml import MLModelFactory

    args.model = MLModelFactory.normalize_model_type(args.model)
    architecture_status = getattr(
        factor_module,
        "ML_ARCHITECTURE_STATUS",
        "promoted",
    )
    if architecture_status != "promoted":
        message = (
            f"{factor_module_name} uses {architecture_status}. Its feature/target "
            "builder must be extracted before it can use the shared trainer."
        )
        if args.retrain:
            parser.error(message)
        print(f"   -> [ML ENGINE] Legacy factor warning: {message}")
    try:
        supported_markets = validate_factor_market_compatibility(
            factor_module,
            args.asset,
            factor_id=getattr(factor_module, "FACTOR_ID", factor_module_name),
        )
    except Exception as exc:
        print(f"❌ [FACTOR MARKET ERROR] {exc}")
        return
    print(
        "   -> 🧭 Factor Market Gate: "
        f"{'ALL' if '*' in supported_markets else ', '.join(supported_markets)}"
    )

    experiment = _run_retraining(factor_module, args) if args.retrain else None
    experiment_record = None
    if experiment is None:
        from oqp.research.ml import latest_ml_experiment

        experiment_record = latest_ml_experiment(
            ALPHA_RESEARCH_DB_PATH,
            model_type=args.model,
            factor_id=getattr(factor_module, "FACTOR_ID", factor_module_name),
        )

    print(f"   ⚡ Loading ML Feature Matrix: {args.feature_matrix}")
    df = _load_feature_matrix(args.feature_matrix, args.asset)
    data_frequency = infer_data_frequency(df)
    vertical_attrs = {
        "market_vertical": args.asset,
        "source_path": os.path.abspath(args.feature_matrix),
        "data_file": os.path.abspath(args.feature_matrix),
        "dataset_id": os.path.splitext(os.path.basename(args.feature_matrix))[0],
        "data_vendor": "local_feature_matrix",
        "data_frequency": data_frequency,
        "execution_assumption": "close_to_close_fallback",
        "ml_model_type": args.model,
    }
    if experiment is not None:
        vertical_attrs.update(
            {
                "ml_experiment_id": experiment.experiment_id,
                "ml_model_name": experiment.model_name,
                "ml_model_path": experiment.artifact_path or "",
                "ml_feature_importance_path": experiment.importance_path or "",
                "ml_predictions_path": experiment.predictions_path or "",
                "ml_target_column": experiment.target_col,
            }
        )
    elif experiment_record is not None:
        vertical_attrs.update(
            {
                "ml_experiment_id": experiment_record["experiment_id"],
                "ml_model_name": experiment_record["model_name"],
                "ml_model_path": experiment_record.get("artifact_path") or "",
                "ml_feature_importance_path": experiment_record.get("importance_path") or "",
                "ml_predictions_path": experiment_record.get("predictions_path") or "",
                "ml_target_column": experiment_record["target_col"],
            }
        )
    apply_vertical_attrs(df, vertical_attrs)
    prediction_frame = experiment.predictions if experiment is not None else None
    if prediction_frame is None and experiment_record is not None:
        stored_predictions = experiment_record.get("predictions_path")
        if stored_predictions and Path(stored_predictions).exists():
            prediction_frame = pd.read_parquet(stored_predictions)
    if prediction_frame is not None:
        df = _attach_oos_predictions(df, prediction_frame)
    elif getattr(factor_module, "REQUIRES_OOS_PREDICTIONS", False):
        raise FileNotFoundError(
            "This ML factor requires registered out-of-sample predictions. "
            "Run the command again with --retrain."
        )
    print(f"   ✅ Matrix loaded successfully. Shape: {df.shape}")
    print("   -> Computing ML factor scores...")
    df = factor_module.compute(df)
    df = attach_factor_parameter_attrs(df, factor_module)
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
        feed = DataEngineFactory.create_feed(base_asset_type, args.feature_matrix)
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
