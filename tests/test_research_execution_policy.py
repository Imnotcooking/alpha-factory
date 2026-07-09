from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

from oqp.research.backtesting import (
    DEFAULT_MIN_TRADE_WEIGHT_DELTA,
    DirectExecutionMode,
    ExecutionModeConfig,
    ExecutionModeFactory,
    StatArbExecutionMode,
    attach_capital_attrs,
    attach_trade_policy_attrs,
    resolve_execution_capital,
    resolve_execution_trade_policy,
)


class ResearchExecutionPolicyTests(unittest.TestCase):
    def test_asset_class_defaults_resolve_amount_and_currency(self) -> None:
        futures = resolve_execution_capital(asset_class="FUTURES_CN")
        equities = resolve_execution_capital(asset_class="EQUITY_US")
        cn_options = resolve_execution_capital(asset_class="Chinese options")
        us_options = resolve_execution_capital(asset_class="OPTIONS_US")
        crypto = resolve_execution_capital(asset_class="CRYPTO_PERP")

        self.assertEqual(futures.initial_capital, 1_000_000.0)
        self.assertEqual(futures.currency, "CNY")
        self.assertEqual(equities.initial_capital, 1_000_000.0)
        self.assertEqual(equities.currency, "USD")
        self.assertEqual(us_options.initial_capital, 100_000.0)
        self.assertEqual(us_options.currency, "USD")
        self.assertEqual(cn_options.initial_capital, 200_000.0)
        self.assertEqual(cn_options.currency, "CNY")
        self.assertEqual(crypto.initial_capital, 100_000.0)
        self.assertEqual(crypto.currency, "USD")

    def test_cli_initial_capital_overrides_factor_profiles(self) -> None:
        factor = SimpleNamespace(
            FACTOR_CONTRACT={
                "capital_profile": "small_personal_futures_cn",
                "capital_currency": "CNY",
            }
        )

        profile = resolve_execution_capital(
            asset_class="FUTURES_CN",
            factor_module=factor,
            initial_capital=350_000,
        )

        self.assertEqual(profile.initial_capital, 350_000.0)
        self.assertEqual(profile.currency, "CNY")
        self.assertEqual(profile.source, "cli_initial_capital")

    def test_factor_contract_capital_profile_is_used_when_cli_is_silent(self) -> None:
        factor = SimpleNamespace(
            FACTOR_CONTRACT={"capital_profile": "small_personal_futures_cn"}
        )

        profile = resolve_execution_capital(
            asset_class="FUTURES_CN",
            factor_module=factor,
        )

        self.assertEqual(profile.initial_capital, 200_000.0)
        self.assertEqual(profile.currency, "CNY")
        self.assertEqual(profile.profile, "small_personal_futures_cn")
        self.assertEqual(profile.source, "factor_contract_capital_profile")

    def test_capital_and_trade_policy_attrs_are_attached(self) -> None:
        frame = pd.DataFrame({"x": [1]})
        capital = resolve_execution_capital(
            asset_class="EQUITY_US",
            capital_profile="small_personal_equity_us",
        )
        policy = resolve_execution_trade_policy(min_trade_weight_delta=0.001)

        attach_capital_attrs(frame, capital)
        returned = attach_trade_policy_attrs(frame, policy)

        self.assertIs(returned, frame)
        self.assertEqual(frame.attrs["initial_capital"], 100_000.0)
        self.assertEqual(frame.attrs["capital_currency"], "USD")
        self.assertEqual(frame.attrs["min_trade_weight_delta"], 0.001)

    def test_trade_policy_precedence_and_validation(self) -> None:
        module = ModuleType("factor_fixture")
        module.FACTOR_CONTRACT = {"min_trade_weight_delta": "0.00025"}

        default_policy = resolve_execution_trade_policy()
        factor_policy = resolve_execution_trade_policy(factor_module=module)
        cli_policy = resolve_execution_trade_policy(
            factor_module=module,
            min_trade_weight_delta=0.0,
        )

        self.assertEqual(default_policy.min_trade_weight_delta, DEFAULT_MIN_TRADE_WEIGHT_DELTA)
        self.assertEqual(factor_policy.min_trade_weight_delta, 0.00025)
        self.assertEqual(cli_policy.min_trade_weight_delta, 0.0)
        with self.assertRaises(ValueError):
            resolve_execution_trade_policy(min_trade_weight_delta=-0.01)

    def test_statarb_execution_preserves_leg_ratios_under_gross_cap(self) -> None:
        result = StatArbExecutionMode(
            ExecutionModeConfig(source_col="target_weight", max_gross_leverage=0.30)
        ).apply(self._frame())

        day = result.df[result.df["date"].eq(pd.Timestamp("2026-01-01"))].set_index(
            "ticker"
        )

        self.assertEqual(result.mode, "statarb")
        self.assertEqual(result.source_col, "target_weight")
        self.assertTrue(np.isclose(day["signal"].abs().sum(), 0.30))
        self.assertTrue(np.isclose(day.loc["a", "signal"] / day.loc["b", "signal"], -2.0))
        self.assertNotIn("kelly_weight", result.df.columns)
        self.assertNotIn("hrp_budget", result.df.columns)

    def test_direct_neutralization_removes_daily_net_exposure(self) -> None:
        frame = self._frame().drop(columns=["target_weight"]).rename(
            columns={"factor_score": "desired_weight"}
        )
        result = ExecutionModeFactory.create(
            "direct",
            ExecutionModeConfig(
                source_col="desired_weight",
                neutralize=True,
                max_gross_leverage=1.0,
            ),
        ).apply(frame)

        self.assertIsInstance(result.df.attrs["execution_mode"], str)
        self.assertIsInstance(ExecutionModeFactory.create("factor_owned"), DirectExecutionMode)
        self.assertTrue(np.allclose(result.df.groupby("date")["signal"].sum().values, 0.0))

    def test_risk_desk_default_uses_raw_signal_and_honors_caps(self) -> None:
        result = ExecutionModeFactory.create(
            "risk_desk",
            ExecutionModeConfig(max_gross_leverage=0.30, max_weight_per_asset=0.20),
        ).apply(self._frame())

        daily_gross = result.df.groupby("date")["signal"].apply(lambda values: values.abs().sum())

        self.assertEqual(result.mode, "risk_desk")
        self.assertEqual(result.detail, "Raw factor signal + portfolio cap.")
        self.assertNotIn("kelly_weight", result.df.columns)
        self.assertNotIn("hrp_budget", result.df.columns)
        self.assertLessEqual(float(daily_gross.max()), 0.30 + 1e-12)
        self.assertLessEqual(float(result.df["signal"].abs().max()), 0.20 + 1e-12)

    def test_risk_desk_can_opt_into_kelly_hrp_and_honor_caps(self) -> None:
        result = ExecutionModeFactory.create(
            "risk_desk",
            ExecutionModeConfig(
                sizing_modules=("kelly", "hrp"),
                max_gross_leverage=0.30,
                max_weight_per_asset=0.20,
            ),
        ).apply(self._frame())

        daily_gross = result.df.groupby("date")["signal"].apply(lambda values: values.abs().sum())

        self.assertEqual(result.detail, "Kelly sizing + HRP budgets + portfolio cap.")
        self.assertIn("kelly_weight", result.df.columns)
        self.assertIn("hrp_budget", result.df.columns)
        self.assertLessEqual(float(daily_gross.max()), 0.30 + 1e-12)
        self.assertLessEqual(float(result.df["signal"].abs().max()), 0.20 + 1e-12)

    def test_risk_desk_can_disable_allocator_modules(self) -> None:
        result = ExecutionModeFactory.create(
            "risk_desk",
            ExecutionModeConfig(
                sizing_modules="none",
                max_gross_leverage=0.30,
                max_weight_per_asset=0.20,
            ),
        ).apply(self._frame())

        daily_gross = result.df.groupby("date")["signal"].apply(lambda values: values.abs().sum())

        self.assertEqual(result.detail, "Raw factor signal + portfolio cap.")
        self.assertNotIn("kelly_weight", result.df.columns)
        self.assertNotIn("hrp_budget", result.df.columns)
        self.assertLessEqual(float(daily_gross.max()), 0.30 + 1e-12)
        self.assertLessEqual(float(result.df["signal"].abs().max()), 0.20 + 1e-12)

    def test_risk_desk_can_run_hrp_without_kelly(self) -> None:
        result = ExecutionModeFactory.create(
            "risk_desk",
            ExecutionModeConfig(
                sizing_modules=("hrp",),
                max_gross_leverage=0.30,
                max_weight_per_asset=0.20,
            ),
        ).apply(self._frame())

        self.assertEqual(result.detail, "HRP risk budgets + portfolio cap.")
        self.assertNotIn("kelly_weight", result.df.columns)
        self.assertIn("hrp_budget", result.df.columns)
        self.assertLessEqual(float(result.df["signal"].abs().max()), 0.20 + 1e-12)

    def test_sizing_module_config_accepts_aliases_and_rejects_unknowns(self) -> None:
        self.assertEqual(
            ExecutionModeConfig(sizing_modules="kelly+risk_parity").sizing_modules,
            ("kelly", "hrp"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown risk_desk sizing module"):
            ExecutionModeConfig(sizing_modules="mvo")

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
                ),
                "ticker": ["a", "b", "a", "b"],
                "close": [100.0, 200.0, 101.0, 198.0],
                "target_weight": [0.40, -0.20, 0.10, -0.10],
                "factor_score": [9.0, 9.0, 9.0, 9.0],
            }
        )


if __name__ == "__main__":
    unittest.main()
