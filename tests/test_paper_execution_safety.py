from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from oqp.brokers import BrokerEnvironment, get_broker_profile_config
from oqp.config import load_settings
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType
from oqp.execution import OrderIntent, TradeProposal
from oqp.paper_trading import (
    PaperExecutionPolicy,
    review_paper_execution_proposal,
)


def settings_from_lines(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(path)
    return tmp, settings


def proposal(
    *,
    proposal_id: str = "proposal-001",
    symbol: str = "SPY",
    asset_class: AssetClass = AssetClass.ETF,
    order_type: OrderType = OrderType.LIMIT,
    quantity: float = 2,
    limit_price: float | None = 500,
    reference_price: float | None = 500,
    paper_only: bool = True,
) -> TradeProposal:
    return TradeProposal(
        proposal_id=proposal_id,
        source="unit_test",
        paper_only=paper_only,
        intents=(
            OrderIntent(
                instrument=Instrument(symbol, asset_class),
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                reference_price=reference_price,
            ),
        ),
    )


class PaperExecutionSafetyTests(unittest.TestCase):
    def test_default_policy_blocks_paper_trading_switch(self) -> None:
        tmp, settings = settings_from_lines(["ALLOW_LIVE_TRADING=false"])
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal(),
            settings=settings,
            broker_config=broker_config,
        )

        self.assertFalse(review.passed)
        self.assertEqual(review.decision.value, "blocked")
        self.assertIn("Paper trading switch", review.message)

    def test_ready_when_switch_caps_and_allowlists_pass(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_ALLOWED_SYMBOLS=SPY,QQQ",
                "PAPER_ALLOWED_ASSET_CLASSES=etf",
                "PAPER_MAX_ORDER_NOTIONAL=2000",
                "PAPER_MAX_DAILY_NOTIONAL=5000",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal(),
            settings=settings,
            broker_config=broker_config,
            daily_notional_used=1000,
        )

        self.assertTrue(review.passed, review.to_dict())
        self.assertEqual(review.estimated_notional, 1000)

    def test_blocks_live_broker_profile(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "IBKR_LIVE_MONITOR_ENABLED=true",
                "ALLOW_LIVE_TRADING=false",
            ]
        )
        self.addCleanup(tmp.cleanup)
        live_config = get_broker_profile_config("ibkr_live_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal(),
            settings=settings,
            broker_config=live_config,
        )

        self.assertFalse(review.passed)
        self.assertEqual(live_config.environment, BrokerEnvironment.LIVE)
        self.assertIn("Paper broker profile", review.message)

    def test_blocks_market_orders_by_default(self) -> None:
        tmp, settings = settings_from_lines(
            ["ALLOW_PAPER_TRADING=true", "ALLOW_LIVE_TRADING=false"]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal(
                order_type=OrderType.MARKET,
                limit_price=None,
                reference_price=500,
            ),
            settings=settings,
            broker_config=broker_config,
        )

        self.assertFalse(review.passed)
        self.assertIn("market order", review.message.lower())

    def test_can_override_policy_for_review_unit_tests(self) -> None:
        tmp, settings = settings_from_lines(["ALLOW_LIVE_TRADING=false"])
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal(order_type=OrderType.MARKET, limit_price=None, reference_price=500),
            settings=settings,
            broker_config=broker_config,
            policy=PaperExecutionPolicy(
                allow_paper_trading=True,
                allow_market_orders=True,
                allowed_asset_classes=("etf",),
            ),
        )

        self.assertTrue(review.passed, review.to_dict())


if __name__ == "__main__":
    unittest.main()
