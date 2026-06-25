from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.brokers import get_broker_profile_config  # noqa: E402
from oqp.config import load_settings  # noqa: E402
from oqp.domain import AssetClass, OrderSide  # noqa: E402
from oqp.execution import (  # noqa: E402
    build_option_trade_proposal_from_candidate,
    load_trade_proposal_artifacts,
    write_option_trade_proposal_from_candidate,
)
from oqp.options import scan_iron_condors  # noqa: E402
from oqp.paper_trading import review_paper_execution_proposal  # noqa: E402


def option_chain(strikes: list[float], mids: list[float]) -> pd.DataFrame:
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


def settings_from_lines(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(path)
    return tmp, settings


class OptionsProposalBridgeTests(unittest.TestCase):
    def option_proposal(self):
        calls = option_chain([95, 100, 105, 110, 115], [8, 5, 2.8, 1.5, 0.8])
        puts = option_chain([85, 90, 95, 100, 105], [0.8, 1.5, 2.8, 5, 8])
        candidate = scan_iron_condors(
            calls,
            puts,
            spot=100,
            expiry="2026-08-15",
            max_risk=1000,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=250,
        ).iloc[0].to_dict()
        return build_option_trade_proposal_from_candidate(
            candidate,
            underlying="QQQ",
            contracts=1,
            proposal_id="options-policy-test",
        )

    def test_option_policy_settings_parse_from_env_file(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "PAPER_OPTIONS_ENABLED=true",
                "PAPER_OPTION_ALLOWED_UNDERLYINGS=SPY,QQQ",
                "PAPER_OPTION_ALLOWED_STRATEGIES=iron_condor,vertical",
                "PAPER_OPTION_MAX_CONTRACTS=2",
                "PAPER_OPTION_MAX_PREMIUM=250",
                "PAPER_OPTION_MAX_DEFINED_RISK=750",
                "PAPER_OPTION_MAX_SPREAD_WIDTH=5",
            ]
        )
        self.addCleanup(tmp.cleanup)

        self.assertTrue(settings.paper_options_enabled)
        self.assertEqual(settings.paper_option_allowed_underlyings, ("SPY", "QQQ"))
        self.assertEqual(settings.paper_option_allowed_strategies, ("iron_condor", "vertical"))
        self.assertEqual(settings.paper_option_max_contracts, 2)
        self.assertEqual(settings.paper_option_max_premium, 250)
        self.assertEqual(settings.paper_option_max_defined_risk, 750)
        self.assertEqual(settings.paper_option_max_spread_width, 5)

    def test_builds_option_trade_proposal_from_scanner_candidate(self) -> None:
        calls = option_chain([95, 100, 105, 110, 115], [8, 5, 2.8, 1.5, 0.8])
        puts = option_chain([85, 90, 95, 100, 105], [0.8, 1.5, 2.8, 5, 8])
        candidate = scan_iron_condors(
            calls,
            puts,
            spot=100,
            expiry="2026-08-15",
            max_risk=1000,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=250,
        ).iloc[0].to_dict()

        proposal = build_option_trade_proposal_from_candidate(
            candidate,
            underlying="QQQ",
            contracts=1,
            proposal_id="options-bridge-test",
        )

        self.assertEqual(proposal.source, "options_desk")
        self.assertTrue(proposal.paper_only)
        self.assertEqual(len(proposal.intents), 4)
        self.assertTrue(all(intent.instrument.asset_class == AssetClass.OPTION for intent in proposal.intents))
        self.assertEqual(proposal.intents[0].side, OrderSide.BUY)
        self.assertEqual(proposal.intents[1].side, OrderSide.SELL)
        self.assertGreater(proposal.estimated_notional or 0, 0)

    def test_writes_artifact_consumable_by_proposal_loader(self) -> None:
        calls = option_chain([95, 100, 105, 110, 115], [8, 5, 2.8, 1.5, 0.8])
        puts = option_chain([85, 90, 95, 100, 105], [0.8, 1.5, 2.8, 5, 8])
        candidate = scan_iron_condors(
            calls,
            puts,
            spot=100,
            expiry="2026-08-15",
            max_risk=1000,
            days_to_hold=30,
            forecast_vol=0.22,
            simulations=250,
        ).iloc[0].to_dict()

        with tempfile.TemporaryDirectory() as tmp:
            result = write_option_trade_proposal_from_candidate(
                candidate,
                underlying="QQQ",
                output_dir=Path(tmp),
                proposal_id="options-written-test",
            )
            loaded = load_trade_proposal_artifacts(Path(tmp))

        self.assertEqual(result.written_path.name, "options-written-test.json")
        self.assertEqual(len(loaded.loaded), 1)
        self.assertEqual(loaded.loaded[0].proposal.proposal_id, "options-written-test")
        self.assertEqual(len(loaded.loaded[0].proposal.intents), 4)

    def test_current_paper_policy_blocks_options_when_disabled(self) -> None:
        proposal = self.option_proposal()
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_ALLOWED_ASSET_CLASSES=equity,etf",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal,
            settings=settings,
            broker_config=broker_config,
        )

        self.assertFalse(review.passed)
        self.assertIn("Options enabled", review.message)

    def test_configured_option_policy_can_pass_defined_risk_spread(self) -> None:
        proposal = self.option_proposal()
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_OPTIONS_ENABLED=true",
                "PAPER_OPTION_ALLOWED_UNDERLYINGS=QQQ,SPY",
                "PAPER_OPTION_ALLOWED_STRATEGIES=iron_condor",
                "PAPER_OPTION_MAX_CONTRACTS=1",
                "PAPER_OPTION_MAX_PREMIUM=500",
                "PAPER_OPTION_MAX_DEFINED_RISK=1000",
                "PAPER_OPTION_MAX_SPREAD_WIDTH=10",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal,
            settings=settings,
            broker_config=broker_config,
        )

        self.assertTrue(review.passed, review.to_dict())

    def test_option_policy_blocks_unlisted_underlying_and_strategy(self) -> None:
        proposal = self.option_proposal()
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_OPTIONS_ENABLED=true",
                "PAPER_OPTION_ALLOWED_UNDERLYINGS=SPY",
                "PAPER_OPTION_ALLOWED_STRATEGIES=long_call",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal,
            settings=settings,
            broker_config=broker_config,
        )

        self.assertFalse(review.passed)
        self.assertIn("Option underlying allowlist", review.message)
        self.assertIn("Option strategy allowlist", review.message)

    def test_option_policy_blocks_risk_limits(self) -> None:
        proposal = self.option_proposal()
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_OPTIONS_ENABLED=true",
                "PAPER_OPTION_ALLOWED_UNDERLYINGS=QQQ",
                "PAPER_OPTION_ALLOWED_STRATEGIES=iron_condor",
                "PAPER_OPTION_MAX_CONTRACTS=0.5",
                "PAPER_OPTION_MAX_DEFINED_RISK=100",
                "PAPER_OPTION_MAX_SPREAD_WIDTH=1",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        review = review_paper_execution_proposal(
            proposal,
            settings=settings,
            broker_config=broker_config,
        )

        self.assertFalse(review.passed)
        self.assertIn("Option intent 1 max contracts", review.message)
        self.assertIn("Option max defined risk", review.message)
        self.assertIn("Option max spread width", review.message)


if __name__ == "__main__":
    unittest.main()
