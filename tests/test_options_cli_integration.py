from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from oqp.commands import alpha_backtest
from oqp.research import factors as factor_utils


class OptionsCliIntegrationTests(unittest.TestCase):
    def test_run_backtest_cli_routes_options_to_event_driven_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factor_root = root / "factors"
            factor_dir = factor_root / "options_tmp"
            factor_dir.mkdir(parents=True)
            factor_file = factor_dir / "fac_tmp_options_cli.py"
            factor_file.write_text(
                textwrap.dedent(
                    """
                    from oqp.research.factor_presets import OPTIONS_DAILY_DIRECTIONAL

                    FACTOR_ID = "fac_tmp_options_cli"
                    NAME_EN = "Temporary Options CLI Factor"
                    CATEGORY = "Options"
                    ECONOMIC_RATIONALE_EN = "Test-only directional option signal."
                    COMPLEXITY = 1
                    FACTOR_CONTRACT = OPTIONS_DAILY_DIRECTIONAL

                    def compute(data):
                        out = data.copy()
                        out["factor_score"] = [1.0, 0.0]
                        return out
                    """
                ),
                encoding="utf-8",
            )

            underlying_file = root / "underlying.csv"
            pd.DataFrame(
                {
                    "date": ["2026-01-02", "2026-01-03"],
                    "underlying_symbol": ["AAPL", "AAPL"],
                    "close": [100.0, 103.0],
                }
            ).to_csv(underlying_file, index=False)

            chain_file = root / "chain.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-01-02",
                        "option_symbol": "AAPL260116C00100000",
                        "underlying_symbol": "AAPL",
                        "expiry": "2026-01-16",
                        "right": "call",
                        "strike": 100.0,
                        "bid": 4.8,
                        "ask": 5.2,
                        "close": 5.0,
                        "volume": 100,
                        "open_interest": 500,
                    },
                    {
                        "date": "2026-01-03",
                        "option_symbol": "AAPL260116C00100000",
                        "underlying_symbol": "AAPL",
                        "expiry": "2026-01-16",
                        "right": "call",
                        "strike": 100.0,
                        "bid": 5.8,
                        "ask": 6.2,
                        "close": 6.0,
                        "volume": 100,
                        "open_interest": 500,
                    },
                ]
            ).to_csv(chain_file, index=False)

            db_path = root / "db" / "research_memory.db"
            artifact_root = root / "artifacts"
            argv = [
                "run_backtest.py",
                "--factor",
                "fac_tmp_options_cli",
                "--asset",
                "OPTIONS_US",
                "--option_chain_file",
                str(chain_file),
                "--option_underlying_file",
                str(underlying_file),
                "--initial_capital",
                "10000",
                "--option_min_dte",
                "1",
                "--option_max_dte",
                "30",
            ]

            with mock.patch.object(factor_utils, "PRIVATE_FACTOR_ROOT", factor_root), mock.patch.object(
                alpha_backtest,
                "ALPHA_RESEARCH_DB_PATH",
                db_path,
            ), mock.patch.object(
                alpha_backtest,
                "ALPHA_RUNTIME_ARTIFACT_ROOT",
                artifact_root,
            ), mock.patch.object(sys, "argv", argv):
                alpha_backtest.main()

            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    """
                    SELECT run_id, asset_class, execution_mode, total_trades, returns_file_path
                    FROM backtest_runs
                    WHERE factor_id = ?
                    """,
                    ("fac_tmp_options_cli",),
                ).fetchone()

            self.assertIsNotNone(row)
            run_id, asset_class, execution_mode, total_trades, returns_file_path = row
            self.assertTrue(str(run_id).startswith("run_"))
            self.assertEqual(asset_class, "OPTIONS_US")
            self.assertEqual(execution_mode, "event_driven_options")
            self.assertEqual(total_trades, 2)
            self.assertTrue(Path(returns_file_path).exists())
            self.assertTrue((artifact_root / "trades" / f"trades_{run_id}.csv").exists())
            manifest_path = artifact_root / "assumptions" / f"assumptions_{run_id}.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["signal_and_execution_mode"]["execution_mode"], "event_driven_options")
            self.assertEqual(manifest["option_contract_selection"]["min_dte"], 1)
            self.assertEqual(manifest["option_contract_selection"]["max_dte"], 30)
            self.assertEqual(manifest["liquidity_policy"]["max_spread_pct"], 0.25)
            self.assertEqual(manifest["realized_summary"]["trades_file_path"], str(artifact_root / "trades" / f"trades_{run_id}.csv"))


if __name__ == "__main__":
    unittest.main()
