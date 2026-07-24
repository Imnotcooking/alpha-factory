from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    load_historical_nav,
    load_portfolio_nav_job_settings,
    market_tickers_for_positions,
    update_portfolio_nav,
    write_live_positions_frame,
)


class PortfolioNavJobTests(unittest.TestCase):
    def test_updates_historical_nav_from_latest_positions_and_manual_inputs(self) -> None:
        captured: dict[str, object] = {}

        def fake_market_data_provider(tickers: list[str], period: str) -> pd.DataFrame:
            captured["tickers"] = tickers
            captured["period"] = period
            return pd.DataFrame(
                {
                    "AAPL": [100.0, 110.0, 120.0],
                    "QQQ": [100.0, 105.0, 110.0],
                    "EURUSD=X": [1.2, 1.2, 1.2],
                    "GBPUSD=X": [1.3, 1.3, 1.3],
                    "CNYUSD=X": [0.14, 0.14, 0.14],
                    "HKDUSD=X": [0.128, 0.128, 0.128],
                    "GC=F": [1900.0, 1950.0, 2000.0],
                    "SPY": [100.0, 103.0, 106.0],
                },
                index=pd.to_datetime(["2026-06-22", "2026-06-23", "2026-06-24"]),
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "portfolio.sqlite3"
            defaults_path = tmp_path / "defaults.json"
            ibkr_metrics_path = tmp_path / "ibkr_metrics.json"
            defaults_path.write_text(
                json.dumps(
                    {
                        "t212_cash_eur": 100.0,
                        "futu_cash_usd": 50.0,
                        "asset_preferences": {
                            "AAPL": {
                                "Category": "Core Compounding",
                                "Target_Weight_%": 42.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            ibkr_metrics_path.write_text(
                json.dumps({"Available_Cash_USD": 25.0}),
                encoding="utf-8",
            )

            positions = pd.DataFrame(
                [
                    {
                        "Broker": "Futubull",
                        "Ticker": "AAPL",
                        "AssetType": "Equity",
                        "Shares": 10,
                        "AvgPrice": 100.0,
                        "Broker_Price": 0.0,
                        "Broker_PnL": 0.0,
                        "Currency": "USD",
                    }
                ]
            )
            write_live_positions_frame(
                db_path,
                positions,
                snapshot_date="2026-06-24",
            )

            result = update_portfolio_nav(
                db_path=db_path,
                snapshot_date="2026-06-24",
                period="3mo",
                benchmark="QQQ",
                defaults_path=defaults_path,
                ibkr_metrics_path=ibkr_metrics_path,
                market_data_provider=fake_market_data_provider,
            )
            nav = load_historical_nav(db_path)

        self.assertEqual(result.status, "updated")
        self.assertEqual(result.position_rows, 1)
        self.assertEqual(captured["period"], "3mo")
        self.assertIn("AAPL", captured["tickers"])
        self.assertIn("QQQ", captured["tickers"])
        self.assertIn("EURUSD=X", captured["tickers"])
        self.assertIn("GC=F", captured["tickers"])
        self.assertEqual(len(nav), 1)
        self.assertAlmostEqual(float(nav.iloc[0]["total_net_worth"]), 1395.0)
        self.assertAlmostEqual(float(nav.iloc[0]["total_cash"]), 195.0)
        self.assertAlmostEqual(result.total_net_worth, 1395.0)
        self.assertAlmostEqual(result.total_cash, 195.0)

    def test_dry_run_values_without_writing_nav(self) -> None:
        def fake_market_data_provider(tickers: list[str], period: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "MSFT": [50.0, 55.0],
                    "QQQ": [100.0, 102.0],
                    "EURUSD=X": [1.0, 1.0],
                    "GBPUSD=X": [1.0, 1.0],
                    "CNYUSD=X": [0.14, 0.14],
                    "HKDUSD=X": [0.128, 0.128],
                    "GC=F": [2000.0, 2000.0],
                },
                index=pd.to_datetime(["2026-06-23", "2026-06-24"]),
            )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "portfolio.sqlite3"
            write_live_positions_frame(
                db_path,
                pd.DataFrame(
                    [
                        {
                            "Broker": "IBKR Live",
                            "Ticker": "MSFT",
                            "AssetType": "Equity",
                            "Shares": 2,
                            "AvgPrice": 50.0,
                            "Broker_Price": 55.0,
                            "Broker_PnL": 10.0,
                            "Currency": "USD",
                        }
                    ]
                ),
                snapshot_date="2026-06-24",
            )

            result = update_portfolio_nav(
                db_path=db_path,
                snapshot_date="2026-06-24",
                defaults_path=Path(tmp) / "missing_defaults.json",
                ibkr_metrics_path=Path(tmp) / "missing_ibkr_metrics.json",
                market_data_provider=fake_market_data_provider,
                dry_run=True,
            )
            nav = load_historical_nav(db_path)

        self.assertEqual(result.status, "dry_run")
        self.assertAlmostEqual(result.total_net_worth, 110.0)
        self.assertTrue(nav.empty)

    def test_no_positions_short_circuits_without_fetching_market_data(self) -> None:
        def unexpected_provider(tickers: list[str], period: str) -> pd.DataFrame:
            raise AssertionError("market data should not be fetched without positions")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = update_portfolio_nav(
                db_path=tmp_path / "empty.sqlite3",
                snapshot_date="2026-06-24",
                defaults_path=tmp_path / "missing_defaults.json",
                ibkr_metrics_path=tmp_path / "missing_ibkr_metrics.json",
                market_data_provider=unexpected_provider,
            )

        self.assertEqual(result.status, "no_positions")
        self.assertEqual(result.position_rows, 0)

    def test_updates_nav_from_manual_inputs_without_live_positions(self) -> None:
        captured: dict[str, object] = {}

        def fake_market_data_provider(tickers: list[str], period: str) -> pd.DataFrame:
            captured["tickers"] = tickers
            return pd.DataFrame(
                {
                    "QQQ": [100.0, 101.0],
                    "EURUSD=X": [1.2, 1.2],
                    "GBPUSD=X": [1.3, 1.3],
                    "CNYUSD=X": [0.14, 0.14],
                    "HKDUSD=X": [0.128, 0.128],
                    "GC=F": [2000.0, 2000.0],
                },
                index=pd.to_datetime(["2026-06-23", "2026-06-24"]),
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "cash_only.sqlite3"
            defaults_path = tmp_path / "defaults.json"
            ibkr_metrics_path = tmp_path / "ibkr_metrics.json"
            defaults_path.write_text(
                json.dumps({"t212_cash_eur": 100.0, "futu_cash_usd": 50.0}),
                encoding="utf-8",
            )
            ibkr_metrics_path.write_text(
                json.dumps({"Available_Cash_USD": 25.0}),
                encoding="utf-8",
            )

            result = update_portfolio_nav(
                db_path=db_path,
                snapshot_date="2026-06-24",
                defaults_path=defaults_path,
                ibkr_metrics_path=ibkr_metrics_path,
                market_data_provider=fake_market_data_provider,
            )
            nav = load_historical_nav(db_path)

        self.assertEqual(result.status, "updated")
        self.assertEqual(result.position_rows, 0)
        self.assertIn("manual portfolio inputs only", result.message)
        self.assertIn("QQQ", captured["tickers"])
        self.assertIn("EURUSD=X", captured["tickers"])
        self.assertAlmostEqual(result.total_net_worth, 195.0)
        self.assertAlmostEqual(result.total_cash, 195.0)
        self.assertEqual(len(nav), 1)
        self.assertAlmostEqual(float(nav.iloc[0]["total_net_worth"]), 195.0)

    def test_market_ticker_builder_skips_options_and_adds_macro_fx(self) -> None:
        positions = pd.DataFrame(
            {
                "ticker": [
                    "BRK.B",
                    "QQQ260331P570000",
                    "AAPL",
                    "VWCE",
                    "XEON",
                ]
            }
        )

        tickers = market_tickers_for_positions(positions, extra_tickers=("IWM",))

        self.assertIn("BRK-B", tickers)
        self.assertIn("AAPL", tickers)
        self.assertIn("VWCE.DE", tickers)
        self.assertIn("XEON.DE", tickers)
        self.assertIn("IWM", tickers)
        self.assertIn("EURUSD=X", tickers)
        self.assertIn("GC=F", tickers)
        self.assertNotIn("QQQ260331P570000", tickers)

    def test_loads_manual_inputs_from_non_secret_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            defaults_path = tmp_path / "defaults.json"
            metrics_path = tmp_path / "ibkr_metrics.json"
            defaults_path.write_text(
                json.dumps(
                    {
                        "t212_cash_eur": "12.5",
                        "futu_cash_usd": 20,
                        "cny_mutual_fund": 1000,
                        "cny_mutual_fund_pnl": 50,
                        "cny_gold_grams": 2,
                        "cny_gold_cost": 900,
                        "asset_preferences": {"AAPL": {"Category": "Core"}},
                    }
                ),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps({"Available_Cash_USD": "33.5"}),
                encoding="utf-8",
            )

            settings = load_portfolio_nav_job_settings(
                defaults_path=defaults_path,
                ibkr_metrics_path=metrics_path,
            )

        self.assertEqual(settings.manual_inputs.t212_cash_eur, 12.5)
        self.assertEqual(settings.manual_inputs.futu_cash_usd, 20.0)
        self.assertEqual(settings.manual_inputs.ibkr_cash_usd, 33.5)
        self.assertEqual(settings.asset_preferences["AAPL"]["Category"], "Core")


if __name__ == "__main__":
    unittest.main()
