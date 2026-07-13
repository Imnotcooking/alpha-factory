from __future__ import annotations

from math import exp, log
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from oqp.research.daily_regimes.continuous_series import (
    ContinuousSeriesConfig,
    build_continuous_series,
    validate_continuous_series_result,
)
from oqp.research.daily_regimes.contracts import (
    ContractViolation,
    RawDailyBarContract,
)
from oqp.research.daily_regimes.stage3_fixtures import (
    make_invalid_stage3_frames,
    make_stage3_adversarial_fixture,
)


def _bar(
    trading_date: str,
    contract: str,
    *,
    close: float,
    open_interest: int,
    volume: int,
    product: str = "SYN_A",
    listing_date: str = "2024-01-01",
    last_trade_date: str = "2024-12-31",
    limit_lock_flag: bool = False,
    stale_bar_flag: bool = False,
) -> dict[str, object]:
    open_price = close - 0.5
    return {
        "product": product,
        "contract": contract,
        "exchange": "SYN",
        "trading_date": pd.Timestamp(trading_date),
        "open": open_price,
        "high": close + 1.0,
        "low": open_price - 1.0,
        "close": close,
        "settlement": close - 0.1,
        "volume": volume,
        "turnover": close * volume * 10.0,
        "open_interest": open_interest,
        "multiplier": 10.0,
        "tick_size": 0.01,
        "roll_flag": False,
        "limit_lock_flag": limit_lock_flag,
        "stale_bar_flag": stale_bar_flag,
        "zero_volume_flag": volume == 0,
        "zero_open_interest_flag": open_interest == 0,
        "source_row_id": f"{contract}:{trading_date}",
        "listing_date": pd.Timestamp(listing_date),
        "last_trade_date": pd.Timestamp(last_trade_date),
    }


def _bars(rows: list[dict[str, object]]) -> pd.DataFrame:
    return (
        pd.DataFrame(rows)
        .sort_values(["product", "contract", "trading_date"], kind="mergesort")
        .reset_index(drop=True)
    )


def _selected_contract_by_date(ledger: pd.DataFrame) -> dict[object, str | None]:
    return {
        pd.Timestamp(row.trading_date).date(): (
            None if pd.isna(row.selected_contract) else str(row.selected_contract)
        )
        for row in ledger.itertuples(index=False)
    }


class RawDailyBarStage3ContractTests(unittest.TestCase):
    def test_clean_rows_validate_without_mutation(self) -> None:
        bars = _bars(
            [
                _bar("2024-01-02", "SYN_A2403", close=100.0, open_interest=80, volume=50),
                _bar("2024-01-03", "SYN_A2403", close=101.0, open_interest=90, volume=60),
            ]
        )
        original = bars.copy(deep=True)

        report = RawDailyBarContract(require_sorted=True).validate(bars)

        self.assertEqual(report.row_count, 2)
        self.assertEqual(report.entity_count, 1)
        self.assertEqual(report.warnings, ())
        assert_frame_equal(bars, original)

    def test_invalid_ohlc_is_rejected(self) -> None:
        bars = _bars(
            [_bar("2024-01-02", "SYN_A2403", close=100.0, open_interest=80, volume=50)]
        )
        bars.loc[0, "high"] = bars.loc[0, "close"] - 1.0

        with self.assertRaisesRegex(ContractViolation, "high"):
            RawDailyBarContract().validate(bars)

    def test_all_adversarial_invalid_frames_are_rejected(self) -> None:
        for scenario, frame in make_invalid_stage3_frames().items():
            with self.subTest(scenario=scenario):
                with self.assertRaises(ContractViolation):
                    RawDailyBarContract().validate(frame)

    def test_duplicate_contract_day_is_rejected(self) -> None:
        row = _bar(
            "2024-01-02",
            "SYN_A2403",
            close=100.0,
            open_interest=80,
            volume=50,
        )
        bars = _bars([row, dict(row)])

        with self.assertRaisesRegex(ContractViolation, "duplicate"):
            RawDailyBarContract().validate(bars)

    def test_contract_lifecycle_dates_are_mandatory(self) -> None:
        bars = _bars(
            [
                _bar(
                    "2024-01-02",
                    "SYN_A2403",
                    close=100.0,
                    open_interest=80,
                    volume=50,
                )
            ]
        ).drop(columns=["listing_date", "last_trade_date"])

        with self.assertRaisesRegex(ContractViolation, "listing_date"):
            RawDailyBarContract().validate(bars)

    def test_zero_liquidity_and_market_constraints_are_explicit_warnings(self) -> None:
        bars = _bars(
            [
                _bar(
                    "2024-01-02",
                    "SYN_A2403",
                    close=100.0,
                    open_interest=0,
                    volume=0,
                    limit_lock_flag=True,
                    stale_bar_flag=True,
                )
            ]
        )

        report = RawDailyBarContract().validate(bars)

        self.assertEqual(
            set(report.warnings),
            {
                "zero_volume_rows_present",
                "zero_open_interest_rows_present",
                "limit_locked_rows_present",
                "stale_rows_present",
            },
        )

    def test_adversarial_fixture_is_synthetic_and_non_evidentiary(self) -> None:
        fixture = make_stage3_adversarial_fixture()

        RawDailyBarContract(require_sorted=True).validate(fixture.contract_rows)
        self.assertTrue(fixture.metadata["synthetic"])
        self.assertFalse(fixture.metadata["scientific_evidence"])
        self.assertFalse(fixture.metadata["paper_eligible"])
        self.assertTrue(fixture.metadata["contains_cross_contract_basis"])
        self.assertIn(None, fixture.expected_selection.values())


class PointInTimeContinuousSeriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = make_stage3_adversarial_fixture()
        self.config = ContinuousSeriesConfig()

    def test_clean_selection_matches_the_frozen_adversarial_oracle(self) -> None:
        original = self.fixture.contract_rows.copy(deep=True)

        result = build_continuous_series(
            self.fixture.contract_rows,
            config=self.config,
        )

        self.assertEqual(
            _selected_contract_by_date(result.roll_ledger),
            dict(self.fixture.expected_selection),
        )
        validate_continuous_series_result(result)
        assert_frame_equal(self.fixture.contract_rows, original)

    def test_every_selection_uses_exactly_the_previous_available_product_date(self) -> None:
        result = build_continuous_series(self.fixture.contract_rows, config=self.config)
        raw = self.fixture.contract_rows

        for row in result.roll_ledger.itertuples(index=False):
            self.assertLess(row.decision_date, row.trading_date)
            if row.selection_status != "selected":
                continue
            decision_row = raw.loc[
                raw["trading_date"].eq(row.decision_date)
                & raw["contract"].eq(row.selected_contract)
            ].iloc[0]
            self.assertEqual(row.selection_open_interest, decision_row["open_interest"])
            self.assertEqual(row.selection_volume, decision_row["volume"])

        monday = result.roll_ledger.loc[
            result.roll_ledger["trading_date"].eq(pd.Timestamp("2020-01-06"))
        ].iloc[0]
        self.assertEqual(monday["decision_date"], pd.Timestamp("2020-01-03"))

    def test_perturbing_future_liquidity_cannot_change_past_outputs(self) -> None:
        cutoff = pd.Timestamp("2020-01-09")
        baseline = build_continuous_series(self.fixture.contract_rows, config=self.config)
        perturbed_rows = self.fixture.contract_rows.copy(deep=True)
        future = perturbed_rows["trading_date"].gt(cutoff)
        perturbed_rows.loc[future, "open_interest"] = (
            perturbed_rows.loc[future, "open_interest"] + 100_000
        )
        perturbed_rows.loc[future, "volume"] = (
            perturbed_rows.loc[future, "volume"] + 100_000
        )

        perturbed = build_continuous_series(perturbed_rows, config=self.config)

        assert_frame_equal(
            baseline.panel.loc[baseline.panel["trading_date"].le(cutoff)].reset_index(
                drop=True
            ),
            perturbed.panel.loc[
                perturbed.panel["trading_date"].le(cutoff)
            ].reset_index(drop=True),
        )
        assert_frame_equal(
            baseline.roll_ledger.loc[
                baseline.roll_ledger["trading_date"].le(cutoff)
            ].reset_index(drop=True),
            perturbed.roll_ledger.loc[
                perturbed.roll_ledger["trading_date"].le(cutoff)
            ].reset_index(drop=True),
        )

    def test_appending_a_future_date_never_rewrites_existing_history(self) -> None:
        baseline = build_continuous_series(self.fixture.contract_rows, config=self.config)
        original_end = self.fixture.contract_rows["trading_date"].max()
        appended = self.fixture.contract_rows.loc[
            self.fixture.contract_rows["contract"].eq("SYN_RB2003")
        ].sort_values("trading_date").iloc[-1].copy()
        appended["trading_date"] = original_end + pd.offsets.BDay(1)
        appended["source_row_id"] = "SYN_RB:20200115:SYN_RB2003:future"
        appended["open"] = float(appended["open"]) + 0.5
        appended["high"] = float(appended["high"]) + 0.5
        appended["low"] = float(appended["low"]) + 0.5
        appended["close"] = float(appended["close"]) + 0.5
        appended["settlement"] = float(appended["settlement"]) + 0.5
        extended_rows = pd.concat(
            [self.fixture.contract_rows, appended.to_frame().T],
            ignore_index=True,
        )
        for column in ("trading_date", "listing_date", "last_trade_date"):
            extended_rows[column] = pd.to_datetime(extended_rows[column])

        extended = build_continuous_series(extended_rows, config=self.config)

        assert_frame_equal(
            baseline.panel,
            extended.panel.loc[
                extended.panel["trading_date"].le(original_end)
            ].reset_index(drop=True),
            check_dtype=False,
        )
        assert_frame_equal(
            baseline.roll_ledger,
            extended.roll_ledger.loc[
                extended.roll_ledger["trading_date"].le(original_end)
            ].reset_index(drop=True),
            check_dtype=False,
        )

    def test_roll_returns_use_same_contract_close_not_the_basis_jump(self) -> None:
        result = build_continuous_series(self.fixture.contract_rows, config=self.config)
        roll_rows = result.panel.loc[result.panel["roll_flag"]]

        self.assertEqual(len(roll_rows), 2)
        self.assertTrue(
            (
                roll_rows["diagnostic_cross_contract_log_return"].abs()
                > roll_rows["same_contract_log_return"].abs() * 50
            ).all()
        )
        for row in roll_rows.itertuples(index=False):
            expected = log(row.close / row.previous_same_contract_close)
            self.assertAlmostEqual(row.same_contract_log_return, expected, places=12)

    def test_chained_index_uses_same_contract_return_and_resets_after_gap(self) -> None:
        result = build_continuous_series(self.fixture.contract_rows, config=self.config)

        for _, sequence in result.panel.groupby("sequence_id", sort=False):
            ordered = sequence.sort_values("trading_date").reset_index(drop=True)
            self.assertTrue(bool(ordered.loc[0, "chain_reset_flag"]))
            self.assertAlmostEqual(
                ordered.loc[0, "continuous_index"],
                self.config.continuous_index_base,
            )
            for index in range(1, len(ordered)):
                self.assertFalse(bool(ordered.loc[index, "chain_reset_flag"]))
                expected = ordered.loc[index - 1, "continuous_index"] * exp(
                    ordered.loc[index, "same_contract_log_return"]
                )
                self.assertAlmostEqual(
                    ordered.loc[index, "continuous_index"], expected, places=10
                )

    def test_ties_expiry_and_new_listing_follow_frozen_ordering(self) -> None:
        result = build_continuous_series(self.fixture.contract_rows, config=self.config)
        selected = result.roll_ledger.set_index("trading_date")["selected_contract"]

        # On the 2020-01-08 decision date, 2002 and the newly eligible 2003
        # tie on both liquidity fields; earliest last-trade date selects 2002.
        self.assertEqual(selected.loc[pd.Timestamp("2020-01-09")], "SYN_RB2002")
        # The expired 2001 is ineligible and 2003 wins once its lagged liquidity leads.
        self.assertEqual(selected.loc[pd.Timestamp("2020-01-10")], "SYN_RB2003")
        january_seventh = selected.loc[pd.Timestamp("2020-01-07")]
        self.assertNotEqual(january_seventh, "SYN_RB2003")

    def test_zero_liquidity_limit_lock_and_missing_session_are_not_filled(self) -> None:
        result = build_continuous_series(self.fixture.contract_rows, config=self.config)
        ledger = result.roll_ledger.set_index("trading_date")

        no_selection = ledger.loc[pd.Timestamp("2020-01-13")]
        self.assertEqual(no_selection["selection_status"], "no_eligible_contract")
        self.assertTrue(pd.isna(no_selection["selected_contract"]))
        self.assertNotIn(pd.Timestamp("2020-01-13"), set(result.panel["trading_date"]))
        # A limit-locked 2002 has greater OI on 2020-01-13 but cannot be used
        # for the 2020-01-14 decision; the eligible 2003 is selected instead.
        self.assertEqual(
            ledger.loc[pd.Timestamp("2020-01-14"), "selected_contract"],
            "SYN_RB2003",
        )
        resumed = result.panel.loc[
            result.panel["trading_date"].eq(pd.Timestamp("2020-01-14"))
        ].iloc[0]
        self.assertTrue(bool(resumed["chain_reset_flag"]))
        self.assertNotIn(5, set(result.roll_ledger["trading_date"].dt.dayofweek))
        self.assertNotIn(6, set(result.roll_ledger["trading_date"].dt.dayofweek))

    def test_missing_effective_bar_is_ledgered_without_backward_fill(self) -> None:
        rows = _bars(
            [
                _bar("2024-01-02", "SYN_A2403", close=100.0, open_interest=100, volume=100),
                _bar("2024-01-02", "SYN_A2406", close=150.0, open_interest=50, volume=50),
                _bar("2024-01-03", "SYN_A2406", close=151.0, open_interest=60, volume=60),
            ]
        )

        result = build_continuous_series(rows, config=self.config)

        self.assertTrue(result.panel.empty)
        self.assertEqual(len(result.roll_ledger), 1)
        failure = result.roll_ledger.iloc[0]
        self.assertEqual(failure["selection_status"], "missing_effective_bar")
        self.assertEqual(failure["selected_contract"], "SYN_A2403")
        self.assertTrue(pd.isna(failure["selected_source_row_id"]))

    def test_panel_and_roll_ledger_reconstruct_raw_provenance(self) -> None:
        result = build_continuous_series(self.fixture.contract_rows, config=self.config)
        selected_ledger = result.roll_ledger.loc[
            result.roll_ledger["selection_status"].eq("selected")
        ]
        merged = result.panel.merge(
            selected_ledger,
            on=["product", "trading_date", "decision_date", "selected_contract"],
            suffixes=("_panel", "_ledger"),
            validate="one_to_one",
        )

        self.assertTrue(
            merged["source_row_id"].eq(merged["selected_source_row_id"]).all()
        )
        raw_ids = set(self.fixture.contract_rows["source_row_id"])
        self.assertTrue(set(merged["source_row_id"]).issubset(raw_ids))
        self.assertTrue(merged["roll_flag_panel"].eq(merged["roll_flag_ledger"]).all())
        self.assertEqual(result.diagnostics["ledger_rows"], len(result.roll_ledger))
        self.assertEqual(result.diagnostics["unselected_dates"], 1)
        self.assertFalse(result.diagnostics["scientific_evidence"])


if __name__ == "__main__":
    unittest.main()
