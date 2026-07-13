from __future__ import annotations

from datetime import date
import json
from math import log
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from oqp.research.daily_regimes.features import (
    LEGACY_EPSILON,
    NUMERICAL_EPSILON,
    build_features,
    feature_set_request,
    wavelet_hurst,
)
from oqp.research.daily_regimes.preprocessing import (
    PreprocessingConfig,
    PreprocessingFitContext,
    fit_preprocessor,
)
from oqp.research.daily_regimes.stage4_fixtures import (
    make_stage4_synthetic_fixture,
)


def _path_from_movements(start: float, movements: list[float]) -> list[float]:
    values = [float(start)]
    for movement in movements:
        values.append(values[-1] + float(movement))
    return values


def _sparse_panel(
    continuous_index: list[float],
    *,
    sequence_ids: list[int] | None = None,
    start: str = "2024-01-02",
    product: str = "SYN_A",
) -> pd.DataFrame:
    count = len(continuous_index)
    dates = pd.bdate_range(start, periods=count)
    close = np.linspace(100.25, 100.25 + count - 1, count)
    return pd.DataFrame(
        {
            "product": product,
            "trading_date": dates,
            "contract": ["SYN_A2406"] * count,
            "source_row_id": [f"sparse:{index}" for index in range(count)],
            "sequence_id": sequence_ids or [1] * count,
            "roll_flag": [False] * count,
            "chain_reset_flag": [index == 0 for index in range(count)],
            "open": close - 0.25,
            "high": close + 1.25,
            "low": close - 1.00,
            "close": close,
            "previous_same_contract_close": close - 0.75,
            "same_contract_log_return": np.linspace(0.001, 0.003, count),
            "turnover": np.linspace(1_000.0, 2_000.0, count),
            "continuous_index": np.asarray(continuous_index, dtype=float),
        }
    )


def _h7_panel() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dates = pd.bdate_range("2024-01-02", periods=64)
    movements = np.tile(
        np.asarray([2.0, -1.0, 3.0, -2.0, 0.5, -0.25, -1.5, 0.75]),
        8,
    )
    price_a = 100.0 + np.cumsum(movements)
    price_b = 150.0 + np.arange(64, dtype=float) * 0.25
    simple_a = np.resize(np.asarray([-0.02, 0.01, 0.03, -0.01, 0.04]), 64)
    simple_b = np.full(64, -0.02)

    frames: list[pd.DataFrame] = []
    for product, prices, returns in (
        ("SYN_A", price_a, simple_a),
        ("SYN_B", price_b, simple_b),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "product": product,
                    "trading_date": dates,
                    "contract": f"{product}2406",
                    "source_row_id": [
                        f"h7:{product}:{index}" for index in range(len(dates))
                    ],
                    "sequence_id": 1,
                    "roll_flag": False,
                    "chain_reset_flag": [index == 0 for index in range(len(dates))],
                    "open": prices - 0.25,
                    "high": prices + 2.0,
                    "low": prices - 1.0,
                    "close": prices,
                    "volume": 100.0 + np.arange(64, dtype=float),
                    "open_interest": 1_000.0 + 10.0 * np.arange(64),
                    "same_contract_log_return": np.log1p(returns),
                    "continuous_index": prices,
                    "sector": "synthetic_metals",
                }
            )
        )
    return pd.concat(frames, ignore_index=True), price_a, simple_a


def _fit_context(
    *,
    fold_id: str = "fold_01",
    start: date = date(2024, 1, 2),
    end: date = date(2024, 1, 31),
) -> PreprocessingFitContext:
    return PreprocessingFitContext(
        fold_id=fold_id,
        training_start=start,
        training_end=end,
        seed=4204,
    )


def _two_product_training_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=3)
    return pd.DataFrame(
        {
            "product": ["A"] * 3 + ["B"] * 3,
            "trading_date": list(dates) * 2,
            "log_amihud": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )


class Stage4SparseFeatureFormulaTests(unittest.TestCase):
    def test_gk_gap_amihud_and_ker_match_hand_calculations(self) -> None:
        # Twenty moves of +2, -1 have displacement 10 and path length 30.
        continuous = _path_from_movements(100.0, [2.0, -1.0] * 10)
        panel = _sparse_panel(continuous)
        target = panel.index[-1]
        panel.loc[target, ["open", "high", "low", "close"]] = [
            105.0,
            109.0,
            103.0,
            107.0,
        ]
        panel.loc[target, "previous_same_contract_close"] = 102.0
        panel.loc[target, "same_contract_log_return"] = -0.025
        panel.loc[target, "turnover"] = 250_000.0
        original = panel.copy(deep=True)

        result = build_features(panel, request=feature_set_request("M3"))
        row = result.frame.iloc[-1]

        gap = log(105.0 / 102.0)
        intraday = 0.5 * log(109.0 / 103.0) ** 2
        intraday -= (2.0 * log(2.0) - 1.0) * log(107.0 / 105.0) ** 2
        expected_variance = gap**2 + intraday
        expected_amihud = 0.025 / 250_000.0
        self.assertAlmostEqual(row["gk_gap_variance"], expected_variance, places=14)
        self.assertAlmostEqual(
            row["log_gk_gap_variance"], log(expected_variance), places=14
        )
        self.assertAlmostEqual(row["amihud"], expected_amihud, places=18)
        self.assertAlmostEqual(
            row["log_amihud"], log(expected_amihud + NUMERICAL_EPSILON), places=14
        )
        self.assertAlmostEqual(row["ker_20d"], 10.0 / 30.0, places=14)
        assert_frame_equal(panel, original)

    def test_roll_day_gap_uses_previous_close_of_the_selected_contract(self) -> None:
        panel = _sparse_panel([100.0, 101.0])
        panel.loc[0, "close"] = 100.0  # Previous row belongs to the old contract.
        panel.loc[1, "contract"] = "SYN_A2409"
        panel.loc[1, "roll_flag"] = True
        panel.loc[1, ["open", "high", "low", "close"]] = [
            120.0,
            122.0,
            119.0,
            121.0,
        ]
        panel.loc[1, "previous_same_contract_close"] = 118.0

        row = build_features(panel, request=feature_set_request("M2")).frame.iloc[1]

        expected = log(120.0 / 118.0) ** 2
        expected += 0.5 * log(122.0 / 119.0) ** 2
        expected -= (2.0 * log(2.0) - 1.0) * log(121.0 / 120.0) ** 2
        cross_contract = log(120.0 / 100.0) ** 2
        cross_contract += 0.5 * log(122.0 / 119.0) ** 2
        cross_contract -= (2.0 * log(2.0) - 1.0) * log(121.0 / 120.0) ** 2
        self.assertAlmostEqual(row["gk_gap_variance"], expected, places=14)
        self.assertNotAlmostEqual(row["gk_gap_variance"], cross_contract, places=8)

    def test_ker_never_crosses_sequence_reset_and_flags_zero_path(self) -> None:
        first = [100.0] * 21
        second = list(np.arange(200.0, 221.0))
        panel = _sparse_panel(
            first + second,
            sequence_ids=[1] * len(first) + [2] * len(second),
        )
        panel.loc[panel.index[len(first)], "chain_reset_flag"] = True

        result = build_features(panel, request=feature_set_request("M3")).frame
        sequence_one_end = result.loc[result["sequence_id"].eq(1)].iloc[-1]
        sequence_two = result.loc[result["sequence_id"].eq(2)].reset_index(drop=True)

        self.assertTrue(pd.isna(sequence_one_end["ker_20d"]))
        self.assertTrue(bool(sequence_one_end["quality_ker_zero_path"]))
        self.assertTrue(sequence_two.loc[:19, "ker_20d"].isna().all())
        self.assertAlmostEqual(sequence_two.loc[20, "ker_20d"], 1.0, places=14)
        self.assertFalse(bool(sequence_two.loc[20, "quality_ker_zero_path"]))

    def test_zero_turnover_is_missing_and_explicitly_flagged(self) -> None:
        panel = _sparse_panel([100.0, 101.0])
        panel.loc[1, "turnover"] = 0.0

        row = build_features(panel, request=feature_set_request("M2")).frame.iloc[1]

        self.assertTrue(pd.isna(row["amihud"]))
        self.assertTrue(pd.isna(row["log_amihud"]))
        self.assertTrue(bool(row["quality_nonpositive_turnover"]))

    def test_future_append_and_future_perturbation_cannot_rewrite_history(self) -> None:
        panel = _sparse_panel(list(np.linspace(100.0, 135.0, 36)))
        cutoff = panel.loc[27, "trading_date"]
        baseline = build_features(panel, request=feature_set_request("M3")).frame

        perturbed_panel = panel.copy(deep=True)
        future = perturbed_panel["trading_date"].gt(cutoff)
        perturbed_panel.loc[future, "continuous_index"] += 50_000.0
        perturbed_panel.loc[future, "open"] *= 1.7
        perturbed_panel.loc[future, "high"] *= 1.7
        perturbed_panel.loc[future, "low"] *= 1.7
        perturbed_panel.loc[future, "close"] *= 1.7
        perturbed = build_features(
            perturbed_panel, request=feature_set_request("M3")
        ).frame
        assert_frame_equal(
            baseline.loc[baseline["trading_date"].le(cutoff)].reset_index(drop=True),
            perturbed.loc[perturbed["trading_date"].le(cutoff)].reset_index(drop=True),
        )

        future_panel = _sparse_panel(
            list(np.linspace(136.0, 140.0, 5)), start="2024-03-01"
        )
        future_panel["source_row_id"] = [f"appended:{i}" for i in range(5)]
        extended = pd.concat([panel, future_panel], ignore_index=True)
        extended = extended.sort_values("trading_date", kind="mergesort").reset_index(
            drop=True
        )
        appended = build_features(extended, request=feature_set_request("M3")).frame
        original_end = panel["trading_date"].max()
        assert_frame_equal(
            baseline,
            appended.loc[appended["trading_date"].le(original_end)].reset_index(
                drop=True
            ),
        )


class Stage4H7FormulaTests(unittest.TestCase):
    def test_h7_formulas_match_fixed_numeric_oracles_including_hurst(self) -> None:
        panel, prices, simple_returns = _h7_panel()
        original = panel.copy(deep=True)

        result = build_features(panel, request=feature_set_request("H7")).frame
        product_a = result.loc[result["product"].eq("SYN_A")].reset_index(drop=True)
        target = product_a.iloc[-1]

        expected_clv = -1.0 / (3.0 + LEGACY_EPSILON)
        expected_volume_climax = 163.0 / (
            np.mean(np.arange(144.0, 164.0)) + LEGACY_EPSILON
        )
        final_returns = simple_returns[-5:]
        centered = final_returns - final_returns.mean()
        expected_skew = np.mean(centered**3) / np.mean(centered**2) ** 1.5
        expected_value = 1.0 - prices[-1] / np.mean(prices[-60:])
        expected_oi_growth = 1_630.0 / 1_530.0
        b_return = -0.02
        cross_section = np.asarray([simple_returns[-1], b_return])
        expected_momentum = (simple_returns[-1] - cross_section.mean()) / (
            cross_section.std(ddof=1) + LEGACY_EPSILON
        )

        self.assertAlmostEqual(target["f_clv"], expected_clv, places=14)
        self.assertAlmostEqual(
            target["f_vol_climax"], expected_volume_climax, places=14
        )
        self.assertAlmostEqual(target["f_ret_skew_5d"], expected_skew, places=14)
        self.assertAlmostEqual(target["f_value_60d"], expected_value, places=14)
        self.assertAlmostEqual(target["f_mom_z_sector"], expected_momentum, places=14)
        self.assertAlmostEqual(target["f_oi_growth_10d"], expected_oi_growth, places=14)
        self.assertAlmostEqual(target["f_macro_hurst"], 0.47190056951224846, places=14)
        self.assertAlmostEqual(wavelet_hurst(prices), 0.47190056951224846, places=14)
        self.assertEqual(product_a.loc[:62, "f_macro_hurst"].unique().tolist(), [0.5])
        assert_frame_equal(panel, original)

    def test_stage4_fixture_is_deterministic_long_and_non_evidentiary(self) -> None:
        first = make_stage4_synthetic_fixture()
        second = make_stage4_synthetic_fixture()

        assert_frame_equal(first.panel, second.panel)
        assert_frame_equal(first.source_contract_rows, second.source_contract_rows)
        self.assertEqual(first.panel["product"].nunique(), 4)
        self.assertGreaterEqual(
            first.panel.groupby("product").size().min(),
            64,
        )
        self.assertTrue(first.metadata["synthetic"])
        self.assertFalse(first.metadata["scientific_evidence"])
        self.assertFalse(first.metadata["paper_eligible"])


class Stage4FoldLocalPreprocessingTests(unittest.TestCase):
    def test_fit_enforces_declared_training_dates(self) -> None:
        training = _two_product_training_frame()
        training.loc[0, "trading_date"] = pd.Timestamp("2024-02-01")

        with self.assertRaisesRegex(
            ValueError, "outside the declared training interval"
        ):
            fit_preprocessor(
                training,
                feature_columns=("log_amihud",),
                config=PreprocessingConfig(),
                context=_fit_context(),
            )

    def test_state_is_deterministic_and_independent_of_training_row_order(self) -> None:
        training = _two_product_training_frame()
        original = training.copy(deep=True)
        shuffled = training.sample(frac=1.0, random_state=91)

        first = fit_preprocessor(
            training,
            feature_columns=("log_amihud",),
            config=PreprocessingConfig(),
            context=_fit_context(),
        )
        second = fit_preprocessor(
            shuffled,
            feature_columns=("log_amihud",),
            config=PreprocessingConfig(),
            context=_fit_context(),
        )

        self.assertEqual(first.training_rows_hash, second.training_rows_hash)
        self.assertEqual(first.state_hash, second.state_hash)
        self.assertEqual(first.state_dict(), second.state_dict())
        assert_frame_equal(training, original)

    def test_product_relative_z_scores_use_only_each_training_product(self) -> None:
        training = _two_product_training_frame()
        fitted = fit_preprocessor(
            training,
            feature_columns=("log_amihud",),
            config=PreprocessingConfig(),
            context=_fit_context(),
        )

        transformed = fitted.transform(training).frame
        for _, group in transformed.groupby("product"):
            self.assertAlmostEqual(group["log_amihud_z"].mean(), 0.0, places=14)
            self.assertAlmostEqual(group["log_amihud_z"].std(ddof=0), 1.0, places=14)
        expected = np.asarray([-1.224744871391589, 0.0, 1.224744871391589])
        np.testing.assert_allclose(
            transformed.loc[transformed["product"].eq("A"), "log_amihud_z"],
            expected,
            rtol=0.0,
            atol=1e-14,
        )

    def test_validation_transform_cannot_change_fitted_state_or_input(self) -> None:
        training = _two_product_training_frame()
        fitted = fit_preprocessor(
            training,
            feature_columns=("log_amihud",),
            config=PreprocessingConfig(),
            context=_fit_context(),
        )
        validation = pd.DataFrame(
            {
                "product": ["A", "B"],
                "trading_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
                "log_amihud": [1e12, -1e12],
            }
        )
        validation_original = validation.copy(deep=True)
        before = json.dumps(
            fitted.state_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

        fitted.transform(validation)

        after = json.dumps(
            fitted.state_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.assertEqual(after, before)
        assert_frame_equal(validation, validation_original)

    def test_missing_values_and_unseen_products_remain_missing(self) -> None:
        fitted = fit_preprocessor(
            _two_product_training_frame(),
            feature_columns=("log_amihud",),
            config=PreprocessingConfig(),
            context=_fit_context(),
        )
        validation = pd.DataFrame(
            {
                "product": ["A", "C"],
                "trading_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
                "log_amihud": [np.nan, 2.0],
            }
        )
        original = validation.copy(deep=True)

        result = fitted.transform(validation)

        self.assertTrue(result.frame["log_amihud_z"].isna().all())
        self.assertEqual(result.diagnostics["unseen_products"], ["C"])
        self.assertEqual(result.diagnostics["incomplete_rows"], 2)
        assert_frame_equal(validation, original)

    def test_training_missing_values_are_rejected(self) -> None:
        training = _two_product_training_frame()
        training.loc[0, "log_amihud"] = np.nan

        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            fit_preprocessor(
                training,
                feature_columns=("log_amihud",),
                config=PreprocessingConfig(),
                context=_fit_context(),
            )

    def test_clipping_bounds_are_learned_from_training_rows_only(self) -> None:
        training = pd.DataFrame(
            {
                "product": ["A"] * 4,
                "trading_date": pd.bdate_range("2024-01-02", periods=4),
                "x": [0.0, 1.0, 2.0, 100.0],
            }
        )
        fitted = fit_preprocessor(
            training,
            feature_columns=("x",),
            config=PreprocessingConfig(
                standardize=False,
                clip_quantiles=(0.25, 0.75),
            ),
            context=_fit_context(),
        )
        validation = pd.DataFrame(
            {
                "product": ["A", "A"],
                "trading_date": pd.to_datetime(["2024-02-01", "2024-02-02"]),
                "x": [-100.0, 1_000.0],
            }
        )

        transformed = fitted.transform(validation).frame

        np.testing.assert_allclose(transformed["x"], [0.75, 26.5])
        state = fitted.state_dict()["groups"]["A"]["x"]
        self.assertEqual(state["lower"], 0.75)
        self.assertEqual(state["upper"], 26.5)

    def test_pca_state_has_deterministic_signs_and_json_serialization(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        training = pd.DataFrame(
            {
                "product": ["A"] * 5 + ["B"] * 5,
                "trading_date": list(dates) * 2,
                "f1": [0, 1, 2, 3, 4, 5, 7, 6, 9, 8],
                "f2": [0, 1, 4, 2, 3, 3, 8, 5, 4, 9],
                "f3": [2, 0, 1, 4, 3, 9, 5, 8, 6, 7],
            }
        )
        config = PreprocessingConfig(pca_components=2)
        first = fit_preprocessor(
            training,
            feature_columns=("f1", "f2", "f3"),
            config=config,
            context=_fit_context(),
        )
        second = fit_preprocessor(
            training.sample(frac=1.0, random_state=7),
            feature_columns=("f1", "f2", "f3"),
            config=config,
            context=_fit_context(),
        )

        for component in first.pca_components or ():
            pivot = int(np.argmax(np.abs(component)))
            self.assertGreater(component[pivot], 0.0)
        self.assertEqual(first.state_hash, second.state_hash)
        self.assertEqual(first.state_dict(), second.state_dict())
        encoded = json.dumps(
            first.state_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.assertEqual(json.loads(encoded), first.state_dict())
        transformed = first.transform(training)
        self.assertEqual(transformed.feature_columns, ("hpca_1", "hpca_2"))
        self.assertTrue(
            np.isfinite(transformed.frame[["hpca_1", "hpca_2"]]).all(axis=None)
        )


if __name__ == "__main__":
    unittest.main()
