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
    validate_factor_market_compatibility,
)
from oqp.research.factor_presets import CROSS_SECTIONAL_DAILY_NEXT_OPEN
from oqp.research.factors import PRIVATE_FACTOR_ROOT, load_factor_module
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
        self.assertEqual(contract.supported_markets, ("FUTURES_CN",))

    def test_factor_contract_rejects_unsupported_market_vertical(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
                "ticker": ["AAA", "BBB"],
                "factor_score": [0.3, -0.2],
                "forward_return": [0.01, -0.01],
            }
        )
        module = SimpleNamespace(
            FACTOR_CONTRACT={
                **CROSS_SECTIONAL_DAILY_NEXT_OPEN,
                "supported_markets": ["FUTURES_CN"],
            }
        )

        with self.assertRaisesRegex(ValueError, "not declared for OPTIONS_US"):
            resolve_factor_contract(
                module,
                frame,
                factor_id="fac_futures_only",
                market_vertical="US options",
                strict=True,
            )

    def test_factor_market_compatibility_accepts_metadata_aliases(self) -> None:
        module = SimpleNamespace(
            FACTOR_METADATA={
                "supported_markets": ["US equities", "Chinese options"],
            }
        )

        supported = validate_factor_market_compatibility(
            module,
            "usa equity",
            factor_id="fac_alias_demo",
        )

        self.assertEqual(supported, ("EQUITY_US", "OPTIONS_CN"))

    def test_fac_044_declares_cn_futures_tick_contract(self) -> None:
        factor_path = (
            PRIVATE_FACTOR_ROOT
            / "tick_pulse"
            / "fac_044_Relative_Velocity_Fade.py"
        )
        if not factor_path.exists():
            self.skipTest("private fac_044 factor is not present in this checkout")
        module = load_factor_module("fac_044_Relative_Velocity_Fade")
        frame = _synthetic_tick_frame()

        result = module.compute(
            frame,
            window=20,
            fast_window=3,
            slow_window=60,
            percentile=0.80,
            min_fast_move_ticks=0.5,
            min_threshold_ratio=1.0,
            hold_ticks=5,
            cooldown_ticks=10,
        )
        contract = resolve_factor_contract(
            module,
            result,
            factor_id=module.FACTOR_ID,
            market_vertical="FUTURES_CN",
            strict=True,
        )

        self.assertEqual(module.FACTOR_METADATA["native_market"], "FUTURES_CN")
        self.assertEqual(module.FACTOR_CONTRACT["supported_markets"], ["FUTURES_CN"])
        self.assertEqual(result.attrs["factor_id"], "fac_044")
        self.assertEqual(result.attrs["factor_metadata"]["supported_markets"], ["FUTURES_CN"])
        self.assertEqual(contract.supported_markets, ("FUTURES_CN",))

    def test_fac_091_builds_risk_balanced_cn_futures_pair_weights(self) -> None:
        factor_path = (
            PRIVATE_FACTOR_ROOT
            / "daily_signals"
            / "fac_091_Intraday_Precious_Metals_Relative_Value_Fade_Futures_CN.py"
        )
        if not factor_path.exists():
            self.skipTest("private fac_091 factor is not present in this checkout")
        module = load_factor_module(
            "fac_091_Intraday_Precious_Metals_Relative_Value_Fade_Futures_CN"
        )
        frame = _synthetic_precious_pair_frame()

        result = module.compute(
            frame,
            fast_window=5,
            vol_window=60,
            spread_window=120,
            regime_window=480,
            min_leg_correlation=-1.0,
            entry_z=1.5,
            reentry_z=1.0,
            pending_minutes=20,
            hold_minutes=15,
            cooldown_minutes=15,
            target_gross=0.5,
        )
        contract = resolve_factor_contract(
            module,
            result,
            factor_id=module.FACTOR_ID,
            market_vertical="FUTURES_CN",
            strict=True,
        )
        gross_by_minute = result.groupby("date")["signal"].apply(lambda s: s.abs().sum())

        self.assertEqual(contract.execution_weight_col, "signal")
        self.assertEqual(contract.execution_lag, "next_bar")
        self.assertEqual(contract.supported_markets, ("FUTURES_CN",))
        self.assertTrue({"factor_score", "signal"}.issubset(result.columns))
        self.assertLessEqual(float(gross_by_minute.max()), 0.5 + 1e-12)
        self.assertEqual(result.attrs["factor_id"], module.FACTOR_ID)

    def test_fac_093_combines_broad_and_pair_mean_reversion_sleeves(self) -> None:
        factor_path = (
            PRIVATE_FACTOR_ROOT
            / "daily_signals"
            / "fac_093_Intraday_Multi_Sleeve_Mean_Reversion_Futures_CN.py"
        )
        if not factor_path.exists():
            self.skipTest("private fac_093 factor is not present in this checkout")
        module = load_factor_module("fac_093_Intraday_Multi_Sleeve_Mean_Reversion_Futures_CN")
        frame = _synthetic_precious_pair_frame()
        frame["open"] = frame["close"]
        frame["high"] = frame["close"] + 0.1
        frame["low"] = frame["close"] - 0.1
        frame["volume"] = 100.0

        result = module.compute(frame)
        contract = resolve_factor_contract(
            module,
            result,
            factor_id=module.FACTOR_ID,
            market_vertical="FUTURES_CN",
            strict=True,
        )
        gross_by_minute = result.groupby("date")["signal"].apply(lambda s: s.abs().sum())

        self.assertEqual(contract.execution_weight_col, "signal")
        self.assertEqual(contract.execution_lag, "next_bar")
        self.assertTrue(
            {"broad_signal", "pair_signal", "factor_score", "signal"}.issubset(result.columns)
        )
        self.assertLessEqual(float(gross_by_minute.max()), 4.0 + 1e-12)
        self.assertEqual(result.attrs["factor_id"], module.FACTOR_ID)

    def test_fac_094_uses_fixed_weights_for_accepted_intraday_breakouts(self) -> None:
        factor_path = (
            PRIVATE_FACTOR_ROOT
            / "daily_signals"
            / "fac_094_Intraday_Fixed_Weight_Breadth_Breakout_Futures_CN.py"
        )
        if not factor_path.exists():
            self.skipTest("private fac_094 factor is not present in this checkout")
        module = load_factor_module("fac_094_Intraday_Fixed_Weight_Breadth_Breakout_Futures_CN")

        result = module.compute(
            _synthetic_intraday_breakout_frame(),
            min_volume_ratio=0.75,
            min_ker=0.30,
            min_trend_atr=0.35,
            min_edge_cost_ratio=0.0,
        )
        contract = resolve_factor_contract(
            module,
            result,
            factor_id=module.FACTOR_ID,
            market_vertical="FUTURES_CN",
            strict=True,
        )
        active = result.loc[result["signal"].ne(0.0)]
        gross_by_minute = result.groupby("date")["signal"].apply(lambda s: s.abs().sum())

        self.assertEqual(contract.execution_weight_col, "signal")
        self.assertGreater(len(active), 0)
        self.assertEqual(set(active["signal"].abs().round(10)), {0.2})
        self.assertLessEqual(float(gross_by_minute.max()), 2.0 + 1e-12)
        self.assertTrue(result.loc[result["signal"].eq(0.0), "factor_score"].eq(0.0).all())
        self.assertEqual(result.attrs["factor_id"], module.FACTOR_ID)

    def test_fac_095_routes_lagged_volatility_quartiles_to_the_next_day(self) -> None:
        factor_path = (
            PRIVATE_FACTOR_ROOT
            / "daily_signals"
            / "fac_095_Intraday_Volatility_Quartile_Router_Futures_CN.py"
        )
        if not factor_path.exists():
            self.skipTest("private fac_095 factor is not present in this checkout")
        module = load_factor_module("fac_095_Intraday_Volatility_Quartile_Router_Futures_CN")
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        volatility = pd.Series([1.0, 2.0, 3.0, 4.0, 0.5, 2.5], index=dates)

        regimes = module.assign_next_period_quartiles(volatility, lookback=4)
        first_ready = regimes.loc[
            regimes["market_volatility_quartile"].notna()
        ].iloc[0]

        self.assertEqual(first_ready["volatility_observation_day"], dates[3])
        self.assertEqual(first_ready["router_calendar_day"], dates[4])
        self.assertEqual(first_ready["market_volatility_quartile"], 4.0)
        self.assertAlmostEqual(first_ready["market_volatility_q75"], 3.25)

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


def _synthetic_tick_frame(rows: int = 180) -> pd.DataFrame:
    base_time = pd.Timestamp("2026-06-01 09:00:00")
    records = []
    volume = 100
    for i in range(rows):
        volume += 1 + (i % 4)
        mid = 900.0 + 0.05 * np.sin(i / 4.0)
        if i in {70, 110, 145}:
            mid += 1.5
        if i in {72, 112, 147}:
            mid -= 1.0
        records.append(
            {
                "symbol": "au2608",
                "datetime": base_time + pd.Timedelta(milliseconds=500 * i),
                "last_price": mid,
                "volume": volume,
                "bid_price_1": mid - 0.1,
                "bid_volume_1": 10 + (i % 5),
                "ask_price_1": mid + 0.1,
                "ask_volume_1": 11 + (i % 7),
                "oi": 1000.0 + i,
            }
        )
    return pd.DataFrame(records)


def _synthetic_precious_pair_frame(rows: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2026-01-05 09:00:00", periods=rows, freq="min")
    common = np.cumsum(0.02 * np.sin(np.arange(rows) / 19.0))
    au = 800.0 + common
    ag = 10_000.0 + 12.0 * common
    au[620:626] += np.linspace(0.0, 4.0, 6)
    au[626:632] += np.linspace(3.0, 0.0, 6)
    records = []
    for ticker, prices in (("KQ.m@SHFE.au", au), ("KQ.m@SHFE.ag", ag)):
        for date, close in zip(dates, prices, strict=True):
            records.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": float(close),
                    "month_change": 0,
                }
            )
    return pd.DataFrame(records)


def _synthetic_intraday_breakout_frame(rows: int = 300) -> pd.DataFrame:
    dates = pd.date_range("2026-06-01 09:00:00", periods=rows, freq="min")
    records = []
    instruments = (
        ("KQ.m@SHFE.au", 800.0, 0.10),
        ("KQ.m@SHFE.ag", 10_000.0, 1.20),
        ("KQ.m@SHFE.cu", 80_000.0, 6.00),
        ("KQ.m@SHFE.al", 20_000.0, 2.00),
    )
    for ticker, base_price, trend_step in instruments:
        prices = base_price + 0.02 * np.sin(np.arange(rows) / 8.0)
        prices[130:] += trend_step * np.arange(rows - 130)
        for index, (date, close) in enumerate(zip(dates, prices, strict=True)):
            records.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": float(close - 0.01),
                    "high": float(close + max(0.03, trend_step * 0.20)),
                    "low": float(close - max(0.03, trend_step * 0.20)),
                    "close": float(close),
                    "volume": 250.0 if index >= 120 else 100.0,
                    "open_interest": 10_000.0,
                    "month_change": 0.0,
                }
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    unittest.main()
