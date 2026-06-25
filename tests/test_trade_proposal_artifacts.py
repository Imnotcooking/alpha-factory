from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.domain import AssetClass, Instrument, OrderSide, OrderType  # noqa: E402
from oqp.execution import (  # noqa: E402
    OrderIntent,
    TradeProposal,
    load_trade_proposal_artifacts,
    write_trade_proposal_artifact,
)


class TradeProposalArtifactTests(unittest.TestCase):
    def test_loads_single_proposal_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "proposal.json").write_text(
                json.dumps(
                    {
                        "proposal_id": "paper-test-001",
                        "source": "unit_test",
                        "paper_only": True,
                        "intents": [
                            {
                                "instrument": {
                                    "symbol": "spy",
                                    "asset_class": "etf",
                                    "currency": "usd",
                                },
                                "side": "buy",
                                "quantity": 2,
                                "order_type": "limit",
                                "limit_price": 500,
                                "reference_price": 501,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = load_trade_proposal_artifacts(directory)

        self.assertEqual(len(result.loaded), 1)
        self.assertEqual(len(result.issues), 0)
        proposal = result.loaded[0].proposal
        self.assertEqual(proposal.proposal_id, "paper-test-001")
        self.assertEqual(proposal.intents[0].instrument.symbol, "SPY")
        self.assertEqual(proposal.estimated_notional, 1002)

    def test_collects_invalid_artifact_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "bad.json").write_text("not-json", encoding="utf-8")
            (directory / "good.json").write_text(
                json.dumps(
                    {
                        "proposal_id": "paper-test-002",
                        "source": "unit_test",
                        "intents": [],
                    }
                ),
                encoding="utf-8",
            )

            result = load_trade_proposal_artifacts(directory)

        self.assertEqual(len(result.loaded), 1)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.loaded[0].proposal.proposal_id, "paper-test-002")

    def test_writes_trade_proposal_artifact(self) -> None:
        proposal = TradeProposal(
            proposal_id="paper-test-003",
            source="unit_test",
            intents=(
                OrderIntent(
                    instrument=Instrument("AAPL", AssetClass.EQUITY),
                    side=OrderSide.SELL,
                    quantity=3,
                    order_type=OrderType.MARKET,
                    reference_price=200,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = write_trade_proposal_artifact(proposal, directory)
            result = load_trade_proposal_artifacts(directory)

        self.assertEqual(path.name, "paper-test-003.json")
        self.assertEqual(len(result.loaded), 1)
        self.assertEqual(result.loaded[0].proposal.estimated_notional, 600)


if __name__ == "__main__":
    unittest.main()
