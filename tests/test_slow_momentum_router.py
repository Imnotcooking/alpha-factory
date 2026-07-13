from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/research/analyze_cn_futures_slow_momentum_fast_reversal_router.py"


def load_module():
    spec = importlib.util.spec_from_file_location("slow_momentum_router_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_night_and_weekend_fragments_map_to_the_same_trade_day() -> None:
    module = load_module()
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2026-07-10 21:00:00",
                "2026-07-11 01:00:00",
                "2026-07-13 09:00:00",
            ]
        )
    )

    result = module.trading_day(timestamps)

    assert result.eq(pd.Timestamp("2026-07-13")).all()


def test_metrics_report_arithmetic_return_and_mean_t_stat() -> None:
    module = load_module()
    result = module.metrics(pd.Series([0.01, -0.005, 0.015, 0.0]))

    assert result["days"] == 4
    assert result["annual_return_arithmetic"] == 1.26
    assert result["mean_t_stat"] > 0.0
