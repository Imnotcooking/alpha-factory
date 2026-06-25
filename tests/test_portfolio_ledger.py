from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    LIVE_POSITION_COLUMNS,
    compute_nav_drawdowns,
    ensure_portfolio_ledger_schema,
    load_historical_nav,
    load_latest_live_positions,
    normalize_live_positions_frame,
    write_historical_nav,
    write_live_positions_frame,
)


class PortfolioLedgerTests(unittest.TestCase):
    def test_ensures_portfolio_ledger_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "portfolio.sqlite3"
            ensure_portfolio_ledger_schema(db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

        self.assertIn("historical_nav", tables)
        self.assertIn("live_positions", tables)

    def test_normalizes_legacy_position_frame_to_live_positions_contract(self) -> None:
        legacy = pd.DataFrame(
            [
                {
                    "Broker": "Futubull",
                    "Ticker": "AAPL",
                    "AssetType": "Equity",
                    "Shares": "10",
                    "AvgPrice": "150.5",
                    "Broker_Price": "155.25",
                    "Currency": "USD",
                }
            ]
        )

        normalized = normalize_live_positions_frame(
            legacy,
            snapshot_date="2026-06-24",
        )

        self.assertEqual(normalized.columns.tolist(), LIVE_POSITION_COLUMNS)
        row = normalized.iloc[0].to_dict()
        self.assertEqual(row["date"], "2026-06-24")
        self.assertEqual(row["broker"], "Futubull")
        self.assertEqual(row["ticker"], "AAPL")
        self.assertEqual(row["unrealized_pnl"], 0.0)
        self.assertEqual(row["delta"], 1.0)
        self.assertEqual(row["gamma"], 0.0)

    def test_writes_and_loads_latest_live_positions_idempotently(self) -> None:
        first = pd.DataFrame(
            [
                {
                    "Broker": "Futubull",
                    "Ticker": "AAPL",
                    "AssetType": "Equity",
                    "Shares": 10,
                    "AvgPrice": 150.0,
                    "Broker_Price": 155.0,
                    "Broker_PnL": 50.0,
                    "Currency": "USD",
                }
            ]
        )
        replacement = first.copy()
        replacement.loc[0, "Shares"] = 12
        later = first.copy()
        later.loc[0, "Ticker"] = "MSFT"

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "portfolio.sqlite3"
            self.assertEqual(
                write_live_positions_frame(
                    db_path,
                    first,
                    snapshot_date="2026-06-24",
                ),
                1,
            )
            self.assertEqual(
                write_live_positions_frame(
                    db_path,
                    replacement,
                    snapshot_date="2026-06-24",
                ),
                1,
            )
            write_live_positions_frame(
                db_path,
                later,
                snapshot_date="2026-06-25",
            )
            latest = load_latest_live_positions(db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                count_2026_06_24 = conn.execute(
                    "SELECT COUNT(*) FROM live_positions WHERE date = ?",
                    ("2026-06-24",),
                ).fetchone()[0]

        self.assertEqual(count_2026_06_24, 1)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest.iloc[0]["date"], "2026-06-25")
        self.assertEqual(latest.iloc[0]["ticker"], "MSFT")

    def test_writes_historical_nav_and_computes_drawdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "portfolio.sqlite3"
            write_historical_nav(
                db_path,
                snapshot_date="2026-06-24",
                total_net_worth=100_000,
                total_cash=5_000,
                portfolio_beta=0.8,
            )
            write_historical_nav(
                db_path,
                snapshot_date="2026-06-25",
                total_net_worth=95_000,
                total_cash=4_500,
                portfolio_beta=0.7,
            )
            write_historical_nav(
                db_path,
                snapshot_date="2026-06-25",
                total_net_worth=96_000,
                total_cash=4_750,
                portfolio_beta=0.75,
            )
            write_historical_nav(
                db_path,
                snapshot_date="2026-06-26",
                total_net_worth=110_000,
                total_cash=6_000,
                portfolio_beta=0.9,
            )
            nav = load_historical_nav(db_path)
            drawdowns = compute_nav_drawdowns(nav)

        self.assertEqual(len(nav), 3)
        self.assertEqual(nav["date"].tolist(), ["2026-06-24", "2026-06-25", "2026-06-26"])
        self.assertEqual(nav["daily_pnl"].tolist(), [0.0, -4000.0, 14000.0])
        self.assertEqual(drawdowns["drawdown"].tolist(), [0.0, -4000.0, 0.0])
        self.assertEqual(drawdowns["drawdown_pct"].round(2).tolist(), [0.0, -0.04, 0.0])


if __name__ == "__main__":
    unittest.main()
