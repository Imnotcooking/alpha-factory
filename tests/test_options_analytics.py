from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from oqp.options import (
    black_scholes_greeks,
    black_scholes_price,
    choose_expiration,
    historical_holding_returns,
    historical_odds,
    normalize_option_chain,
    scan_backspreads,
    scan_cash_secured_puts,
    scan_call_butterflies,
    scan_calendar_spreads,
    scan_iron_condors,
    scan_long_options,
    scan_ratio_spreads,
    scan_vertical_spreads,
    score_option_strategies,
    simulate_single_option,
    solve_implied_volatility,
    volatility_snapshot,
)


class OptionsAnalyticsTests(unittest.TestCase):
    def option_chain(self, strikes: list[float], mids: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strike": strike,
                    "bid": max(mid - 0.05, 0.01),
                    "ask": mid + 0.05,
                    "lastPrice": mid,
                    "impliedVolatility": 0.25,
                }
                for strike, mid in zip(strikes, mids, strict=True)
            ]
        )

    def test_black_scholes_price_and_implied_vol_round_trip(self) -> None:
        price = black_scholes_price(100, 100, 30 / 365, 0.04, 0.25, "call")
        implied = solve_implied_volatility(price, 100, 100, 30 / 365, 0.04, "call")

        self.assertGreater(price, 0)
        self.assertAlmostEqual(implied, 0.25, places=2)

    def test_put_delta_is_negative(self) -> None:
        greeks = black_scholes_greeks(100, 95, 45 / 365, 0.04, 0.30, "put")

        self.assertLess(greeks["delta"], 0)
        self.assertGreater(greeks["vega"], 0)

    def test_choose_expiration(self) -> None:
        expiry = choose_expiration(["2026-07-01", "2026-08-15"], 30, today=date(2026, 6, 25))

        self.assertEqual(expiry, "2026-08-15")

    def test_normalize_option_chain_uses_mid(self) -> None:
        chain = pd.DataFrame(
            [
                {"strike": 100, "bid": 1.0, "ask": 1.4, "lastPrice": 1.1},
                {"strike": 105, "bid": 0.0, "ask": 0.0, "lastPrice": 0.8},
            ]
        )

        normalized = normalize_option_chain(chain, "call")

        self.assertEqual(normalized["mid"].round(2).tolist(), [1.2, 0.8])

    def test_scan_long_options_and_short_puts(self) -> None:
        calls = pd.DataFrame(
            [
                {"strike": 100, "bid": 2.0, "ask": 2.2, "lastPrice": 2.1},
                {"strike": 130, "bid": 0.4, "ask": 0.5, "lastPrice": 0.45},
            ]
        )
        puts = pd.DataFrame(
            [
                {"strike": 95, "bid": 1.5, "ask": 1.7, "lastPrice": 1.6},
                {"strike": 80, "bid": 0.6, "ask": 0.7, "lastPrice": 0.65},
            ]
        )

        long_calls = scan_long_options(
            calls,
            spot=100,
            expiry="2026-08-15",
            option_type="call",
            budget=500,
            days_to_hold=30,
            forecast_vol=0.25,
            today=date(2026, 6, 25),
        )
        short_puts = scan_cash_secured_puts(
            puts,
            spot=100,
            expiry="2026-08-15",
            max_collateral=10_000,
            days_to_hold=30,
            forecast_vol=0.25,
            today=date(2026, 6, 25),
        )

        self.assertFalse(long_calls.empty)
        self.assertFalse(short_puts.empty)

    def test_volatility_snapshot_and_historical_odds(self) -> None:
        history = pd.DataFrame(
            {
                "Close": [100 + i for i in range(80)],
                "High": [101 + i for i in range(80)],
                "Low": [99 + i for i in range(80)],
            }
        )
        snapshot = volatility_snapshot(history)
        returns = historical_holding_returns(history, days_to_hold=30)

        self.assertGreater(snapshot.forecast_vol, 0)
        self.assertGreaterEqual(historical_odds(returns, 0.01, "up"), 0)

    def test_simulate_single_option(self) -> None:
        simulation = simulate_single_option(
            spot=100,
            strike=100,
            premium=3,
            option_type="call",
            side="long",
            days_to_hold=30,
            volatility=0.25,
            simulations=1000,
        )

        self.assertEqual(len(simulation.profits), 1000)

    def test_strategy_router_favors_short_vol_when_iv_is_rich_and_market_is_choppy(self) -> None:
        ranked = score_option_strategies(
            spot=100,
            moving_average_20=100,
            rolling_std_20=2,
            rsi_14=50,
            market_iv=0.40,
            forecast_vol=0.20,
            target_beta=0.5,
        )

        self.assertFalse(ranked.empty)
        self.assertIn(ranked.iloc[0]["Strategy"], {"Short Straddle / Strangle", "Iron Condors"})
        self.assertGreater(ranked.iloc[0]["Score"], 80)

    def test_strategy_router_can_surface_capitulation_leaps(self) -> None:
        ranked = score_option_strategies(
            spot=90,
            moving_average_20=100,
            rolling_std_20=4,
            rsi_14=15,
            market_iv=0.30,
            forecast_vol=0.35,
            target_beta=0.5,
        )

        self.assertEqual(ranked.iloc[0]["Strategy"], "Deep Value LEAPS")

    def test_vertical_and_calendar_scanners_return_candidates(self) -> None:
        calls = self.option_chain([90, 95, 100, 105, 110, 115], [12, 8, 5, 3, 1.5, 0.8])
        far_calls = self.option_chain([90, 95, 100, 105, 110, 115], [14, 10, 7, 4.5, 2.7, 1.5])
        puts = self.option_chain([85, 90, 95, 100, 105, 110], [0.9, 1.5, 2.6, 4.5, 7.0, 10.5])

        bull_calls = scan_vertical_spreads(
            calls,
            spot=100,
            expiry="2026-08-15",
            spread_type="bull_call",
            budget=600,
            days_to_hold=30,
            forecast_vol=0.25,
            simulations=300,
        )
        bear_puts = scan_vertical_spreads(
            puts,
            spot=100,
            expiry="2026-08-15",
            spread_type="bear_put",
            budget=600,
            days_to_hold=30,
            forecast_vol=0.25,
            simulations=300,
        )
        calendars = scan_calendar_spreads(
            calls,
            far_calls,
            spot=100,
            near_expiry="2026-08-15",
            far_expiry="2026-09-18",
            budget=500,
            forecast_vol=0.25,
            today=date(2026, 6, 25),
            simulations=300,
        )

        self.assertEqual(bull_calls.iloc[0]["Strategy"], "Bull Call Spread")
        self.assertEqual(bear_puts.iloc[0]["Strategy"], "Bear Put Spread")
        self.assertEqual(calendars.iloc[0]["Strategy"], "Calendar Spread")

    def test_condor_butterfly_ratio_and_backspread_scanners_return_candidates(self) -> None:
        calls = self.option_chain(
            [85, 90, 95, 100, 105, 110, 115, 120],
            [16, 12, 8, 5, 2.8, 1.5, 0.8, 0.4],
        )
        puts = self.option_chain(
            [80, 85, 90, 95, 100, 105, 110, 115],
            [0.4, 0.8, 1.5, 2.8, 5, 8, 12, 16],
        )

        condors = scan_iron_condors(
            calls,
            puts,
            spot=100,
            expiry="2026-08-15",
            max_risk=1000,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=300,
        )
        butterflies = scan_call_butterflies(
            calls,
            spot=100,
            expiry="2026-08-15",
            budget=600,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=300,
        )
        call_ratios = scan_ratio_spreads(
            calls,
            spot=100,
            expiry="2026-08-15",
            option_type="call",
            budget=600,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=300,
        )
        put_ratios = scan_ratio_spreads(
            puts,
            spot=100,
            expiry="2026-08-15",
            option_type="put",
            budget=600,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=300,
        )
        call_backspreads = scan_backspreads(
            calls,
            spot=100,
            expiry="2026-08-15",
            option_type="call",
            budget=600,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=300,
        )
        put_backspreads = scan_backspreads(
            puts,
            spot=100,
            expiry="2026-08-15",
            option_type="put",
            budget=600,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=300,
        )

        self.assertEqual(condors.iloc[0]["Strategy"], "Iron Condor")
        self.assertEqual(butterflies.iloc[0]["Strategy"], "Call Butterfly")
        self.assertEqual(call_ratios.iloc[0]["Strategy"], "Call Ratio Spread")
        self.assertEqual(put_ratios.iloc[0]["Strategy"], "Put Ratio Spread")
        self.assertEqual(call_backspreads.iloc[0]["Strategy"], "Call Backspread")
        self.assertEqual(put_backspreads.iloc[0]["Strategy"], "Put Backspread")


if __name__ == "__main__":
    unittest.main()
