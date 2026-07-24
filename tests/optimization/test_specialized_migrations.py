from __future__ import annotations

import pandas as pd
import pytest

from oqp.optimization import require_dataset_fingerprint
from oqp.research.tick_pulse.calibration import (
    TickPulseCalibrationConfig,
    TickPulseHeuristicOptimizer,
)
from oqp.research.tick_pulse.xgboost_calibration import (
    TickXGBoostBayesianOptimizer,
)
from oqp.research.tick_pulse.xgboost_model import TickXGBoostConfig


def test_optimization_rejects_unfingerprinted_data() -> None:
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        require_dataset_fingerprint(pd.DataFrame({"value": [1.0]}))


def test_tick_pulse_split_is_frozen_from_input_calendar() -> None:
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    features = pd.DataFrame(
        {
            "symbol": ["A"] * len(dates),
            "datetime": dates,
            "mid_price": range(100, 110),
            "tick_size_est": [1.0] * len(dates),
        }
    )
    optimizer = TickPulseHeuristicOptimizer(
        TickPulseCalibrationConfig(holdout_fraction=0.30, n_trials=2)
    )
    prepared = optimizer._prepare_features(features)
    optimizer._freeze_temporal_split(prepared)

    assert len(optimizer._calibration_dates) == 7
    assert len(optimizer._holdout_dates) == 3
    assert optimizer._frozen_split_info["holdout_start"] == "2025-01-08"


def test_specialized_optimizers_use_component_schemas() -> None:
    heuristic = TickPulseHeuristicOptimizer(
        TickPulseCalibrationConfig(n_trials=2)
    )
    xgboost = TickXGBoostBayesianOptimizer(
        TickXGBoostConfig(
            horizon_ticks=30,
            min_success_ticks=1.0,
            hypothesis="relative_velocity_fade",
        ),
        n_trials=2,
    )

    assert heuristic._parameter_schema().component_type == "factor"
    assert len(heuristic._parameter_schema().tunable_names) == 6
    assert xgboost._parameter_schema().component_type == "model"
    assert len(xgboost._parameter_schema().tunable_names) == 9
