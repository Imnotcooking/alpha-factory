from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    PortfolioPositionSnapshot,
    futubull_option_to_occ,
    parse_futubull_csv,
    parse_trading212_csv,
    position_snapshots_to_live_positions_frame,
)


class PortfolioBrokerImportTests(unittest.TestCase):
    def test_translates_futubull_option_to_occ(self) -> None:
        occ_ticker, underlying = futubull_option_to_occ("AAPL 240119 150.00C")

        self.assertEqual(occ_ticker, "O:AAPL240119C00150000")
        self.assertEqual(underlying, "AAPL")

    def test_parses_futubull_holdings_and_injects_option_greeks(self) -> None:
        csv_data = io.StringIO(
            "\n".join(
                [
                    "Symbol,Quantity,Average Cost,Current price,Currency",
                    "AAPL,10,150.50,155.25,USD",
                    "AAPL 240119 150.00C,2,5.25,6.50,USD",
                    "AAPL 240119 150.00C/AAPL 240119 155.00C,1,1.00,1.25,USD",
                ]
            )
        )
        seen: list[tuple[str, str]] = []

        def greeks_provider(occ_ticker: str, underlying: str) -> tuple[float, float]:
            seen.append((occ_ticker, underlying))
            return 0.42, 0.03

        parsed = parse_futubull_csv(csv_data, greeks_provider=greeks_provider)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed["Ticker"].tolist(), ["AAPL", "AAPL 240119 150.00C"])
        self.assertEqual(parsed["AssetType"].tolist(), ["Equity", "Option"])
        self.assertEqual(parsed["Multiplier"].tolist(), [1.0, 100.0])
        self.assertEqual(parsed["delta"].tolist(), [1.0, 0.42])
        self.assertEqual(parsed["gamma"].tolist(), [0.0, 0.03])
        self.assertEqual(seen, [("O:AAPL240119C00150000", "AAPL")])

    def test_reconstructs_trading212_positions_and_banked_profit(self) -> None:
        csv_data = io.StringIO(
            "\n".join(
                [
                    "Action,Time,Ticker,No. of shares,Total,Currency (Total),Result",
                    "Market buy,2026-01-01,VWCE,10,1000,EUR,0",
                    "Market buy,2026-01-02,VWCE,5,600,EUR,0",
                    "Market sell,2026-01-03,VWCE,3,390,EUR,20",
                    "Dividend (Dividend),2026-01-04,VWCE,0,5,EUR,0",
                    "Interest on cash,2026-01-05,,0,2,EUR,0",
                ]
            )
        )

        result = parse_trading212_csv(csv_data)

        self.assertEqual(result.banked_profit, 27.0)
        self.assertEqual(len(result.positions), 1)
        row = result.positions.iloc[0]
        self.assertEqual(row["Ticker"], "VWCE.DE")
        self.assertEqual(row["Broker"], "Trading212")
        self.assertEqual(row["Shares"], 12.0)
        self.assertAlmostEqual(row["AvgPrice"], 106.6667, places=4)
        self.assertEqual(row["Currency"], "EUR")
        self.assertEqual(row["Broker_Price"], 0.0)

    def test_position_snapshot_exports_live_position_schema(self) -> None:
        frame = position_snapshots_to_live_positions_frame(
            [
                PortfolioPositionSnapshot(
                    broker="Futubull",
                    ticker="AAPL",
                    shares=2,
                    avg_price=150,
                    broker_price=155,
                    currency="USD",
                )
            ],
            "2026-06-24",
        )

        self.assertEqual(
            frame.iloc[0].to_dict(),
            {
                "date": "2026-06-24",
                "broker": "Futubull",
                "ticker": "AAPL",
                "asset_type": "Equity",
                "shares": 2,
                "avg_cost": 150,
                "current_price": 155,
                "unrealized_pnl": 0.0,
                "currency": "USD",
                "delta": 1.0,
                "gamma": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
