from __future__ import annotations

import unittest

import pandas as pd

from oqp.risk import (
    average_true_range,
    black_scholes_greeks,
    concentration_table,
    enrich_position_risk,
    hedge_diagnosis,
    inverse_hedge_plan,
    summarize_portfolio_risk,
)


class PortfolioRiskTests(unittest.TestCase):
    def test_enrich_positions_calculates_option_delta_exposure(self) -> None:
        positions = pd.DataFrame(
            [
                {
                    "broker": "IBKR",
                    "ticker": "AAPL",
                    "asset_type": "Equity",
                    "shares": 10,
                    "avg_cost": 100,
                    "current_price": 120,
                    "unrealized_pnl": 0,
                    "currency": "USD",
                    "delta": 0,
                    "gamma": 0,
                },
                {
                    "broker": "IBKR",
                    "ticker": "SPY PUT",
                    "asset_type": "Option",
                    "shares": 1,
                    "avg_cost": 2,
                    "current_price": 3,
                    "unrealized_pnl": 0,
                    "currency": "USD",
                    "delta": -0.4,
                    "gamma": 0.01,
                },
            ]
        )

        enriched = enrich_position_risk(positions)

        self.assertEqual(enriched.loc[0, "signed_delta_exposure"], 1200)
        self.assertEqual(enriched.loc[1, "market_value"], 300)
        self.assertEqual(enriched.loc[1, "signed_delta_exposure"], -120)

    def test_summarize_portfolio_risk_uses_nav_drawdown_and_var(self) -> None:
        positions = enrich_position_risk(
            pd.DataFrame(
                [
                    {
                        "broker": "IBKR",
                        "ticker": "AAPL",
                        "asset_type": "Equity",
                        "shares": 10,
                        "avg_cost": 100,
                        "current_price": 120,
                        "unrealized_pnl": 200,
                        "currency": "USD",
                        "delta": 1,
                        "gamma": 0,
                    }
                ]
            )
        )
        nav = pd.DataFrame(
            [
                {"total_net_worth": 1000, "total_cash": 100, "portfolio_beta": 0.8, "daily_pnl": 0, "drawdown": 0, "drawdown_pct": 0},
                {"total_net_worth": 900, "total_cash": 80, "portfolio_beta": 0.7, "daily_pnl": -100, "drawdown": -100, "drawdown_pct": -0.1},
                {"total_net_worth": 1100, "total_cash": 120, "portfolio_beta": 0.75, "daily_pnl": 200, "drawdown": 0, "drawdown_pct": 0},
            ]
        )

        summary = summarize_portfolio_risk(positions, nav)

        self.assertEqual(summary.latest_nav, 1100)
        self.assertEqual(summary.beta_adjusted_exposure, 825)
        self.assertEqual(summary.max_drawdown, -100)
        self.assertGreater(summary.one_day_var_95, 0)

    def test_concentration_table_weights(self) -> None:
        positions = enrich_position_risk(
            pd.DataFrame(
                [
                    {"broker": "A", "ticker": "AAPL", "asset_type": "Equity", "shares": 2, "avg_cost": 1, "current_price": 50, "unrealized_pnl": 0, "currency": "USD", "delta": 1},
                    {"broker": "A", "ticker": "MSFT", "asset_type": "Equity", "shares": 1, "avg_cost": 1, "current_price": 100, "unrealized_pnl": 0, "currency": "USD", "delta": 1},
                ]
            )
        )

        table = concentration_table(positions)

        self.assertAlmostEqual(table["Weight %"].sum(), 1.0)

    def test_inverse_hedge_plan(self) -> None:
        plan = inverse_hedge_plan(
            portfolio_value=100_000,
            portfolio_beta=0.8,
            hedge_asset="SQQQ",
            hedge_price=20,
            budget=10_000,
            leverage=-3,
            atr=1,
        )

        self.assertEqual(plan.shares_to_buy, 500)
        self.assertEqual(plan.current_delta, 80_000)
        self.assertEqual(plan.hedge_delta, -30_000)
        self.assertEqual(plan.stop_loss, 18.5)
        self.assertEqual(plan.take_profit, 23)

    def test_black_scholes_put_delta_and_diagnosis(self) -> None:
        greeks = black_scholes_greeks(spot=100, strike=95, time_to_expiry=30 / 365, rate=0.04, volatility=0.2, option_type="put")
        diagnosis = hedge_diagnosis(
            beta_adjusted_exposure=50_000,
            net_contract_delta=greeks["Delta"],
            underlying_price=100,
            contracts=0,
        )

        self.assertLess(greeks["Delta"], 0)
        self.assertEqual(diagnosis.verdict, "unhedged")
        self.assertGreaterEqual(diagnosis.additional_contracts_needed, 1)

    def test_average_true_range(self) -> None:
        history = pd.DataFrame(
            {
                "High": [11, 12, 13, 14, 15],
                "Low": [9, 10, 11, 12, 13],
                "Close": [10, 11, 12, 13, 14],
            }
        )

        self.assertGreater(average_true_range(history, window=2), 0)


if __name__ == "__main__":
    unittest.main()
