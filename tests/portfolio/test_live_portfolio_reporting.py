from __future__ import annotations

import json
import unittest

import pandas as pd

from oqp.market import historical_volatility_frame, normalize_price_history
from oqp.options import option_leg_report, recognize_option_spreads, underlying_exposure_report
from oqp.portfolio import (
    asset_sleeve_mix,
    concentration_curve_frame,
    concentration_diagnostics_frame,
    currency_exposure_frame,
    enriched_live_holdings,
    position_risk_frame,
    sector_exposure_frame,
)


class LivePortfolioReportingTests(unittest.TestCase):
    def test_historical_volatility_frame_normalizes_common_columns(self) -> None:
        history = pd.DataFrame(
            {
                "Ticker": ["SPY"] * 25,
                "Date": pd.date_range("2026-01-01", periods=25, freq="D"),
                "Close": [100 + i for i in range(25)],
            }
        )

        normalized = normalize_price_history(history)
        hv = historical_volatility_frame(history, windows=(5, 20))

        self.assertEqual(list(normalized.columns), ["symbol", "date", "close"])
        self.assertEqual(hv.iloc[0]["symbol"], "SPY")
        self.assertGreaterEqual(float(hv.iloc[0]["hv_5d"]), 0.0)
        self.assertGreaterEqual(float(hv.iloc[0]["hv_20d"]), 0.0)

    def test_option_spread_recognition_detects_vertical_spread(self) -> None:
        positions = pd.DataFrame(
            [
                {
                    "symbol": "SPY260117C00500000",
                    "asset_class": "option",
                    "quantity": 1,
                    "market_price": 12.0,
                    "market_value": 1200.0,
                    "unrealized_pnl": 100.0,
                    "multiplier": 100,
                    "metadata_json": json.dumps({"delta": 0.55, "gamma": 0.02}),
                },
                {
                    "symbol": "SPY260117C00510000",
                    "asset_class": "option",
                    "quantity": -1,
                    "market_price": 7.0,
                    "market_value": -700.0,
                    "unrealized_pnl": -50.0,
                    "multiplier": 100,
                    "metadata_json": json.dumps({"delta": -0.35, "gamma": -0.01}),
                },
            ]
        )
        hv = pd.DataFrame([{"symbol": "SPY", "hv_5d": 0.12, "hv_20d": 0.18}])

        legs = option_leg_report(positions, hv)
        spreads = recognize_option_spreads(positions, hv)
        exposure = underlying_exposure_report(positions, hv)

        self.assertEqual(len(legs), 2)
        self.assertEqual(spreads.iloc[0]["Structure"], "call vertical spread")
        self.assertEqual(spreads.iloc[0]["Legs"], 2)
        self.assertEqual(exposure.iloc[0]["Underlying"], "SPY")
        self.assertEqual(exposure.iloc[0]["HV 20D"], 0.18)
        self.assertAlmostEqual(float(exposure.iloc[0]["Option Cost Basis"]), 450.0)

    def test_enriched_live_holdings_adds_hv_and_spread_columns(self) -> None:
        positions = pd.DataFrame(
            [
                {
                    "symbol": "SPY260117P00450000",
                    "asset_class": "option",
                    "quantity": 1,
                    "average_cost": 4.5,
                    "market_price": 5.0,
                    "market_value": 500.0,
                    "unrealized_pnl": 50.0,
                    "currency": "USD",
                    "as_of": "2026-06-29T12:00:00+00:00",
                    "metadata_json": json.dumps({"iv": 0.25, "delta": -0.3}),
                }
            ]
        )
        hv = pd.DataFrame([{"symbol": "SPY", "hv_5d": 0.11, "hv_20d": 0.20}])

        holdings = enriched_live_holdings(positions, hv)

        self.assertEqual(holdings.iloc[0]["Underlying"], "SPY")
        self.assertEqual(holdings.iloc[0]["HV 5D"], 0.11)
        self.assertEqual(holdings.iloc[0]["IV / 20D HV"], 1.25)
        self.assertIn("Spread Group", holdings.columns)

    def test_enriched_live_holdings_keeps_native_price_with_usd_value(self) -> None:
        positions = pd.DataFrame(
            [
                {
                    "symbol": "0700.HK 2027-03-30 480C",
                    "broker": "unified",
                    "asset_class": "option",
                    "quantity": 1,
                    "average_cost": 4.80,
                    "market_price": 4.80,
                    "market_value": 480.00,
                    "unrealized_pnl": 0.0,
                    "currency": "USD",
                    "multiplier": 100,
                    "metadata_json": json.dumps(
                        {
                            "source_broker": "external_manual",
                            "native_currency": "HKD",
                            "local_current_price": 37.3,
                            "underlying": "0700.HK",
                        }
                    ),
                }
            ]
        )

        holdings = enriched_live_holdings(positions)

        self.assertEqual(holdings.iloc[0]["Broker"], "external_manual")
        self.assertEqual(holdings.iloc[0]["Currency"], "USD")
        self.assertEqual(holdings.iloc[0]["Native Currency"], "HKD")
        self.assertEqual(holdings.iloc[0]["Market Price"], 37.3)
        self.assertEqual(holdings.iloc[0]["Market Value"], 480.0)

    def test_asset_sleeve_mix_groups_cash_equities_and_options(self) -> None:
        holdings = pd.DataFrame(
            [
                {"Symbol": "SPY", "Asset Class": "etf", "Market Value": 5000.0, "DTE": pd.NA},
                {"Symbol": "XEON", "Asset Class": "etf", "Market Value": 2000.0, "DTE": pd.NA},
                {"Symbol": "JNJ", "Asset Class": "equity", "Market Value": 3000.0, "DTE": pd.NA},
                {"Symbol": "AMZN", "Asset Class": "equity", "Market Value": 7000.0, "DTE": pd.NA},
                {"Symbol": "SPY260117C00500000", "Asset Class": "option", "Market Value": -400.0, "DTE": 120},
            ]
        )

        mix = asset_sleeve_mix(holdings, cash=1000.0)
        values = dict(zip(mix["Sleeve"], mix["Exposure Value"], strict=False))

        self.assertEqual(values["Cash"], 3000.0)
        self.assertEqual(values["Equity (index)"], 5000.0)
        self.assertEqual(values["Equity (defensive)"], 3000.0)
        self.assertEqual(values["Equity (aggressive)"], 7000.0)
        self.assertEqual(values["Options"], 400.0)
        self.assertEqual(mix.loc[mix["Sleeve"].eq("Options"), "Symbols"].iloc[0], "SPY")
        self.assertAlmostEqual(float(mix["Weight"].sum()), 1.0)

    def test_position_risk_frame_includes_cash_weight_and_pnl_percent(self) -> None:
        holdings = pd.DataFrame(
            [
                {
                    "Symbol": "AMZN",
                    "Asset Class": "equity",
                    "Market Value": 1200.0,
                    "Unrealized P&L": 200.0,
                    "DTE": pd.NA,
                }
            ]
        )

        risk = position_risk_frame(holdings, nav=2000.0, cash=800.0)
        by_symbol = {row["Symbol"]: row for row in risk.to_dict("records")}

        self.assertEqual(by_symbol["Cash"]["Sleeve"], "Cash")
        self.assertEqual(by_symbol["Cash"]["Weight"], 0.4)
        self.assertEqual(by_symbol["AMZN"]["Sleeve"], "Equity (aggressive)")
        self.assertEqual(by_symbol["AMZN"]["Weight"], 0.6)
        self.assertAlmostEqual(by_symbol["AMZN"]["Unrealized P&L %"], 0.2)

    def test_sector_exposure_frame_groups_cash_and_known_sectors(self) -> None:
        holdings = pd.DataFrame(
            [
                {
                    "Symbol": "NVO",
                    "Asset Class": "equity",
                    "Market Value": 4800.0,
                    "Unrealized P&L": 300.0,
                    "Underlying": "NVO",
                },
                {
                    "Symbol": "GOOGL",
                    "Asset Class": "equity",
                    "Market Value": 3200.0,
                    "Unrealized P&L": 120.0,
                    "Underlying": "GOOGL",
                },
                {
                    "Symbol": "XEON",
                    "Asset Class": "etf",
                    "Market Value": 2000.0,
                    "Unrealized P&L": 0.0,
                    "Underlying": "XEON",
                },
            ]
        )

        sector = sector_exposure_frame(holdings, cash=1000.0)
        values = dict(zip(sector["Sector"], sector["Exposure Value"], strict=False))

        self.assertEqual(values["Cash"], 3000.0)
        self.assertEqual(values["Health Care"], 4800.0)
        self.assertEqual(values["Communication Services"], 3200.0)
        self.assertAlmostEqual(float(sector["Weight"].sum()), 1.0)

    def test_concentration_frames_use_position_weights(self) -> None:
        holdings = pd.DataFrame(
            [
                {"Symbol": "AMZN", "Asset Class": "equity", "Market Value": 1200.0, "Unrealized P&L": 100.0},
                {"Symbol": "NVO", "Asset Class": "equity", "Market Value": 800.0, "Unrealized P&L": 50.0},
            ]
        )
        risk = position_risk_frame(holdings, nav=2500.0, cash=500.0)

        diagnostics = concentration_diagnostics_frame(risk)
        curve = concentration_curve_frame(risk)
        metric_values = dict(zip(diagnostics["Metric"], diagnostics["Value"], strict=False))

        self.assertAlmostEqual(metric_values["Largest exposure"], 0.48)
        self.assertAlmostEqual(metric_values["Top 3 weight"], 1.0)
        self.assertEqual(curve.iloc[0]["Symbol"], "AMZN")
        self.assertAlmostEqual(curve.iloc[-1]["Cumulative Weight"], 1.0)

    def test_currency_exposure_frame_adds_cash_to_base_currency(self) -> None:
        holdings = pd.DataFrame(
            [
                {"Symbol": "AAPL", "Currency": "USD", "Market Value": 1500.0},
                {"Symbol": "TSM", "Currency": "USD", "Market Value": 500.0},
            ]
        )

        currency = currency_exposure_frame(holdings, cash=250.0, cash_currency="USD")

        self.assertEqual(currency.iloc[0]["Currency"], "USD")
        self.assertEqual(currency.iloc[0]["Exposure Value"], 2250.0)
        self.assertAlmostEqual(currency.iloc[0]["Weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
