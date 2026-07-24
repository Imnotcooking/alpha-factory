from __future__ import annotations

import os

import pandas as pd

from oqp.research.tick_pulse import (
    TickXGBoostConfig,
    build_tick_ml_study_params,
    ensure_tick_ml_tables,
    make_tick_ml_study_key,
    score_probability_threshold,
)


def test_tick_xgboost_config_uses_promoted_oqp_config() -> None:
    config = TickXGBoostConfig(
        horizon_ticks=30,
        min_success_ticks=1.0,
        hypothesis="relative_velocity_fade",
    )

    assert config.hyperparams()["n_estimators"] == 180
    assert config.hyperparams()["max_depth"] == 3


def test_tick_ml_study_key_is_deterministic_and_tables_initialize(tmp_path) -> None:
    data_path = tmp_path / "ticks.parquet"
    data_path.write_bytes(b"not-a-real-parquet-for-keying")
    db_path = tmp_path / "research_memory.db"

    params = build_tick_ml_study_params(
        file_path=str(data_path),
        project_root=str(tmp_path),
        file_mtime=os.path.getmtime(data_path),
        product="au",
        symbol="au2608",
        window=120,
        horizon_ticks=30,
        hypothesis="relative_velocity_fade",
        min_success_ticks=1.0,
        max_rows=1_000,
        test_fraction=0.30,
    )

    ensure_tick_ml_tables(str(db_path))

    assert params["file_path"] == "ticks.parquet"
    assert make_tick_ml_study_key(params) == make_tick_ml_study_key(dict(reversed(params.items())))
    assert db_path.exists()


def test_score_probability_threshold_reports_signal_slice() -> None:
    predictions = pd.DataFrame(
        {
            "ml_probability": [0.1, 0.7, 0.8],
            "target": [0, 1, 0],
        }
    )

    result = score_probability_threshold(predictions, probability_threshold=0.6)

    assert result["rows"] == 3
    assert result["signal_count"] == 2
    assert result["signal_accuracy"] == 0.5
