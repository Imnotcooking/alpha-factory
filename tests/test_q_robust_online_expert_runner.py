from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_FILE = (
    REPO_ROOT
    / "notebooks/Phase_7_Research_Projects/"
    "07_04_daily_volatility_router_cn_futures_replication_private/"
    "experiments/13_q_robust_causal_online_expert_search/run_search.py"
)


def _load_round_four():
    module_name = "qrobust_online_expert_runner_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROUND_FOUR = _load_round_four()


def _base_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=3)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "symbol": "A1",
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": 2_000.0,
            "open_interest": 200.0,
            "open_oi": 200.0,
            "sector": "Test",
            "open_to_next_open_return": [0.01, 0.02, np.nan],
        }
    )


def test_round_four_factor_input_exposes_only_dedicated_feedback() -> None:
    observed = ROUND_FOUR._factor_input_with_roll_clean_feedback(
        _base_panel()
    )

    assert "feedback_open_to_next_open_return" in observed
    assert "open_to_next_open_return" not in observed
    assert observed["feedback_open_to_next_open_return"].iloc[:2].tolist() == [
        0.01,
        0.02,
    ]
    assert (
        observed.attrs["feedback_stream"]
        == "roll_clean_open_to_next_open_return_for_matured_labels_only"
    )


def test_round_four_frozen_protocol_rejects_purified_component_sources() -> None:
    protocol = ROUND_FOUR.RUNNER.load_protocol()
    with pytest.raises(ValueError, match="hash changed"):
        ROUND_FOUR.RUNNER.validate_component_registry(protocol)

    assert protocol["status"] == "locked_before_first_pair_trial"
    assert set(protocol["factor_family"]["component_source_sha256"]) == set(
        ROUND_FOUR._COMPONENT_FACTOR_IDS
    )
