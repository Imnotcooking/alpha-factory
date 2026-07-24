from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.execution import (  # noqa: E402
    SignalProposalConfig,
    build_trade_proposal_from_signal_rows,
    load_research_signal_rows,
    load_trade_proposal_artifacts,
    write_research_signal_proposal,
)
from oqp.execution.research_bridge import main as bridge_main  # noqa: E402


class ResearchSignalBridgeTests(unittest.TestCase):
    def test_builds_trade_proposal_from_signal_rows(self) -> None:
        rows = [
            {
                "symbol": "spy",
                "asset_class": "etf",
                "direction": "buy",
                "strength": 0.7,
                "strategy_id": "demo",
                "reference_price": 500,
                "quantity": 2,
            },
            {
                "symbol": "qqq",
                "direction": "flat",
                "strength": 0.9,
                "strategy_id": "demo",
            },
        ]

        result = build_trade_proposal_from_signal_rows(
            rows,
            proposal_id="paper-bridge-test",
            config=SignalProposalConfig(min_strength=0.5),
        )

        self.assertEqual(result.proposal.proposal_id, "paper-bridge-test")
        self.assertTrue(result.proposal.paper_only)
        self.assertEqual(len(result.proposal.intents), 1)
        self.assertEqual(result.proposal.intents[0].instrument.symbol, "SPY")
        self.assertEqual(result.proposal.intents[0].estimated_notional, 1000)
        self.assertEqual(len(result.issues), 1)

    def test_loads_csv_aliases_and_negative_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["ticker", "score", "close", "shares", "factor"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ticker": "AAPL",
                        "score": "-0.42",
                        "close": "200",
                        "shares": "3",
                        "factor": "csv_demo",
                    }
                )

            rows = load_research_signal_rows(path)
            result = build_trade_proposal_from_signal_rows(rows)

        intent = result.proposal.intents[0]
        self.assertEqual(intent.instrument.symbol, "AAPL")
        self.assertEqual(intent.side.value, "sell")
        self.assertEqual(intent.confidence, 0.42)
        self.assertEqual(intent.estimated_notional, 600)

    def test_writes_output_consumable_by_proposal_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_path = root / "signals.json"
            signal_path.write_text(
                json.dumps(
                    {
                        "signals": [
                            {
                                "symbol": "MSFT",
                                "direction": 1,
                                "strength": 0.5,
                                "reference_price": 400,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = write_research_signal_proposal(
                signal_path,
                output_dir=root / "proposals",
                proposal_id="paper-written-test",
            )
            loaded = load_trade_proposal_artifacts(root / "proposals")

        self.assertIsNotNone(result.written_path)
        self.assertEqual(len(loaded.loaded), 1)
        self.assertEqual(loaded.loaded[0].proposal.proposal_id, "paper-written-test")

    def test_no_valid_intents_does_not_write_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_path = root / "signals.json"
            signal_path.write_text(
                json.dumps({"signals": [{"symbol": "MSFT", "direction": "flat"}]}),
                encoding="utf-8",
            )
            result = write_research_signal_proposal(
                signal_path,
                output_dir=root / "proposals",
                proposal_id="paper-empty-test",
            )

        self.assertIsNone(result.written_path)
        self.assertEqual(len(result.proposal.intents), 0)
        self.assertEqual(len(result.issues), 1)

    def test_cli_writes_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_path = root / "signals.json"
            output_dir = root / "proposals"
            signal_path.write_text(
                json.dumps(
                    {
                        "signals": [
                            {
                                "symbol": "SPY",
                                "direction": "buy",
                                "strength": 0.5,
                                "reference_price": 500,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = bridge_main(
                    [
                        str(signal_path),
                        "--output-dir",
                        str(output_dir),
                        "--proposal-id",
                        "paper-cli-test",
                    ]
                )
            loaded = load_trade_proposal_artifacts(output_dir)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(loaded.loaded), 1)
        self.assertEqual(loaded.loaded[0].proposal.proposal_id, "paper-cli-test")


if __name__ == "__main__":
    unittest.main()
