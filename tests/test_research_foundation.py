from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oqp.research import (
    AlphaMetricEvaluator,
    EvaluationGeometry,
    build_chronological_split,
    infer_dataset_tradability,
    resolve_factor_contract,
    stable_trial_hash,
)
from oqp.research.factor_presets import CROSS_SECTIONAL_DAILY_NEXT_OPEN
from oqp.research.statistics import AlphaStatisticalTester, bonferroni_p_value


class ResearchFoundationTests(unittest.TestCase):
    def test_factor_contract_resolves_explicit_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
                "ticker": ["AAA", "BBB"],
                "factor_score": [0.3, -0.2],
                "forward_return": [0.01, -0.01],
            }
        )
        module = SimpleNamespace(FACTOR_CONTRACT=CROSS_SECTIONAL_DAILY_NEXT_OPEN)

        contract = resolve_factor_contract(module, frame, factor_id="fac_demo", strict=True)

        self.assertEqual(contract.evaluation_geometry, "cross_sectional")
        self.assertEqual(contract.execution_lag, "next_open")
        self.assertEqual(contract.alpha_signal_col, "factor_score")

    def test_dataset_policy_labels_tick_contract_data_as_executable(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-06-01 09:30:00"]),
                "symbol": ["au2608"],
                "last_price": [790.0],
                "bid_price_1": [789.98],
                "ask_price_1": [790.02],
                "bid_volume_1": [10],
                "ask_volume_1": [8],
            }
        )

        profile = infer_dataset_tradability(
            frame,
            source_path="data_cache/8contract_au_raw_tick.parquet",
            asset_class="FUTURES_CN",
            data_frequency="tick",
        )

        self.assertEqual(profile.dataset_role, "contract_tick")
        self.assertEqual(profile.tradability, "executable")

    def test_split_policy_applies_purge_gap(self) -> None:
        dates = pd.date_range("2026-01-01", periods=12)
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAA"] * len(dates),
                "factor_score": np.arange(len(dates), dtype=float),
                "forward_return": np.arange(len(dates), dtype=float) * 0.01,
            }
        )

        split = build_chronological_split(
            frame,
            mode="ratio",
            validation_fraction=0.5,
            purge_periods=1,
            purge_unit="days",
        )

        self.assertEqual(split.split_mode, "ratio_days_purged")
        self.assertGreater(split.purged_rows, 0)
        self.assertGreater(split.holdout_rows, 0)

    def test_metrics_and_statistics_are_importable_from_oqp_research(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-02"] * 4),
                "ticker": ["A", "B", "C", "D"] * 2,
                "factor_score": [1, 2, 3, 4, 4, 3, 2, 1],
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01],
            }
        )

        metric = AlphaMetricEvaluator().evaluate(
            factor_id="fac_metric",
            df=frame,
            validation_data=frame,
            holdout_data=frame,
            crisis_data=frame.iloc[0:0],
            signal_col="factor_score",
            explicit_geometry="cross_sectional",
        )
        evidence = AlphaStatisticalTester().evaluate(
            frame,
            signal_col="factor_score",
            return_col="forward_return",
            geometry=EvaluationGeometry.CROSS_SECTIONAL,
        )

        self.assertEqual(metric.validation_ic, 1.0)
        self.assertEqual(evidence.test_method, "daily_rank_ic_ttest_greater")
        self.assertEqual(bonferroni_p_value(0.01, 3), 0.03)
        self.assertEqual(len(stable_trial_hash({"factor": "demo"})), 24)


if __name__ == "__main__":
    unittest.main()
