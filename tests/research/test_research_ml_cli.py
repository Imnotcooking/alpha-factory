from __future__ import annotations

import subprocess
from unittest.mock import patch

import pandas as pd

from oqp.commands.alpha_ml_backtest import (
    _attach_oos_predictions,
    _model_artifact_paths,
    _validation_config_from_args,
    build_parser,
)
from oqp.research.ml.tree_based.runtime import probe_model_runtime


def test_ml_cli_exposes_canonical_model_and_matrix_controls() -> None:
    args = build_parser().parse_args(
        [
            "--asset",
            "FUTURES_CN",
            "--factor",
            "fac_demo",
            "--model",
            "lightgbm",
            "--feature-matrix",
            "runtime/data/feature_store/demo.parquet",
            "--target-column",
            "target_4d_rank",
            "--retrain",
        ]
    )

    assert args.model == "lightgbm"
    assert args.feature_matrix.endswith("demo.parquet")
    assert args.target_column == "target_4d_rank"
    assert args.retrain is True


def test_ml_cli_builds_explicit_walk_forward_policy() -> None:
    args = build_parser().parse_args(
        [
            "--asset",
            "FUTURES_CN",
            "--factor",
            "fac_demo",
            "--validation-mode",
            "walk_forward",
            "--min-train-days",
            "500",
            "--test-window-days",
            "40",
            "--purge-gap-days",
            "4",
        ]
    )

    config = _validation_config_from_args(args)

    assert config is not None
    assert config.mode == "walk_forward"
    assert config.min_train_days == 500
    assert config.test_window_days == 40
    assert config.purge_gap_days == 4


def test_ml_artifact_paths_are_factor_and_model_specific() -> None:
    model_path, importance_path, predictions_path = _model_artifact_paths(
        "xgboost",
        "fac_054",
    )

    assert model_path.name == "fac_054_xgboost.json"
    assert importance_path.name == "fac_054_xgboost.csv"
    assert predictions_path.name == "fac_054_xgboost.parquet"


def test_oos_predictions_attach_without_filling_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "ticker": ["a", "a"],
            "f_signal": [1.0, 2.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"]),
            "ticker": ["a"],
            "target": [0.4],
            "prediction": [0.3],
            "fold": [1],
        }
    )

    result = _attach_oos_predictions(frame, predictions)

    assert result["ml_prediction"].isna().tolist() == [True, False]
    assert result.attrs["ml_oos_prediction_rows"] == 1


def test_native_runtime_probe_contains_adapter_crashes() -> None:
    failed = subprocess.CompletedProcess(
        args=["python", "-c", "probe"],
        returncode=139,
        stdout="",
        stderr="",
    )
    with patch("subprocess.run", return_value=failed):
        status = probe_model_runtime("lightgbm")

    assert status.available is False
    assert status.returncode == 139
    assert "segmentation fault" in status.detail
