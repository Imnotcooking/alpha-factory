from __future__ import annotations

import unittest

import pandas as pd

from oqp.intelligence import (
    AllocationAdvisoryEngine,
    BaseEngine,
    EngineContext,
    EngineRegistry,
    EngineStatus,
    IntelligenceCoordinator,
    PortfolioManagerEngine,
    RegimeSnapshotEngine,
    RiskControlRoomEngine,
    default_intelligence_registry,
)
from oqp.intelligence.allocation_engine import (
    apply_weight_constraints,
    hrp_weights,
    kelly_weights,
    portfolio_volatility,
    scale_to_vol_target,
)
from oqp.intelligence.regime_engine import MarketHMM


class PassingEngine(BaseEngine):
    engine_id = "passing"
    engine_name = "Passing Engine"

    def run(self, context: EngineContext):
        return self.result(
            status=EngineStatus.PASS,
            summary=f"ran for {context.environment}",
        )


class FailingEngine(BaseEngine):
    engine_id = "failing"
    engine_name = "Failing Engine"

    def run(self, context: EngineContext):
        raise RuntimeError("boom")


class IntelligenceTests(unittest.TestCase):
    def test_registry_and_coordinator_run_engines(self) -> None:
        registry = EngineRegistry()
        registry.register_factory(PassingEngine.engine_id, PassingEngine)
        results = IntelligenceCoordinator(registry).run(EngineContext(environment="test"))

        self.assertEqual(tuple(results), ("passing",))
        self.assertEqual(results["passing"].status, EngineStatus.PASS)
        self.assertIn("test", results["passing"].summary)

    def test_coordinator_captures_engine_failure(self) -> None:
        registry = EngineRegistry()
        registry.register_factory(FailingEngine.engine_id, FailingEngine)
        results = IntelligenceCoordinator(registry).run(EngineContext())

        self.assertEqual(results["failing"].status, EngineStatus.FAIL)
        self.assertIn("boom", results["failing"].summary)

    def test_default_registry_contains_risk_control_room(self) -> None:
        self.assertEqual(
            default_intelligence_registry().engine_ids(),
            (
                "allocation_advisory",
                "portfolio_manager",
                "regime_snapshot",
                "risk_control_room",
            ),
        )

    def test_portfolio_manager_skips_until_approved_strategies_exist(self) -> None:
        result = PortfolioManagerEngine().run(EngineContext())

        self.assertEqual(result.status, EngineStatus.SKIPPED)
        self.assertFalse(result.frame("requirements").empty)
        self.assertEqual(result.signals["runtime_role"], "post_approval_command_center")

    def test_portfolio_manager_marks_active_paper_strategy_triggerable(self) -> None:
        result = PortfolioManagerEngine().run(
            EngineContext(
                settings={
                    "allow_paper_trading": True,
                    "allow_live_trading": False,
                },
                approved_strategies=pd.DataFrame(
                    [
                        {
                            "strategy_id": "fac_public_001",
                            "market_vertical": "us_equities",
                            "target_environment": "paper",
                            "status": "paper_approved",
                        }
                    ]
                ),
                strategy_signals=pd.DataFrame(
                    [
                        {
                            "strategy_id": "fac_public_001",
                            "active": True,
                            "strength": 0.7,
                        }
                    ]
                ),
            )
        )

        self.assertEqual(result.status, EngineStatus.PASS)
        decisions = result.frame("runtime_decisions")
        self.assertEqual(decisions.iloc[0]["Runtime Decision"], "triggerable")
        self.assertEqual(result.signals["triggerable_strategy_ids"], ["fac_public_001"])

    def test_risk_control_room_flags_concentration(self) -> None:
        context = EngineContext(
            live_summary={
                "source": "account_ledger",
                "nav": 1_000.0,
                "cash": 20.0,
                "daily_pnl": -5.0,
                "position_count": 1,
                "as_of": "2026-06-29T12:00:00+00:00",
                "performance": {
                    "gross_exposure": 800.0,
                    "gross_exposure_pct": 0.8,
                    "max_drawdown_pct": -0.05,
                },
            },
            paper_summary={
                "source": "account_ledger",
                "nav": 100_000.0,
                "cash": 80_000.0,
                "daily_pnl": 100.0,
                "position_count": 1,
                "as_of": "2026-06-29T12:00:00+00:00",
                "performance": {
                    "gross_exposure": 20_000.0,
                    "gross_exposure_pct": 0.2,
                    "max_drawdown_pct": -0.02,
                },
            },
            live_positions=pd.DataFrame(
                [
                    {
                        "symbol": "SPY",
                        "asset_class": "etf",
                        "market_value": 800.0,
                        "unrealized_pnl": 10.0,
                    }
                ]
            ),
            paper_positions=pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "asset_class": "equity",
                        "market_value": 20_000.0,
                        "unrealized_pnl": 50.0,
                    }
                ]
            ),
        )

        result = RiskControlRoomEngine().run(context)

        self.assertEqual(result.status, EngineStatus.WARN)
        self.assertEqual(len(result.frame("summary")), 2)
        self.assertFalse(result.frame("concentration").empty)
        self.assertIn(
            "Position concentration",
            set(result.frame("risk_flags")["Check"]),
        )

    def test_market_hmm_prepare_emissions_matches_alpha_lab_shape(self) -> None:
        model = MarketHMM(n_components=2)
        emissions = model._prepare_emissions(
            pd.DataFrame(
                {
                    "returns": [0.01, None, -0.02],
                    "volatility": [0.10, 0.20, None],
                }
            )
        )

        self.assertEqual(emissions.shape, (3, 2))
        self.assertEqual(float(emissions[1, 0]), 0.0)
        self.assertEqual(float(emissions[2, 1]), 0.0)

    def test_regime_snapshot_engine_reads_nav_return_vol_emissions(self) -> None:
        nav_history = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=30, freq="D"),
                "net_liquidation": [100 + i + (5 if i > 20 else 0) for i in range(30)],
                "daily_return": [0.0] + [0.01] * 20 + [-0.03, 0.02, -0.04, 0.03, -0.02, 0.01, -0.01, 0.02, -0.03],
            }
        )
        result = RegimeSnapshotEngine().run(
            EngineContext(live_nav_history=nav_history, paper_nav_history=nav_history)
        )

        self.assertIn(result.status, {EngineStatus.PASS, EngineStatus.WARN})
        self.assertEqual(len(result.frame("regimes")), 2)
        self.assertIn("State", result.frame("regimes").columns)

    def test_allocation_helpers_produce_constrained_weights(self) -> None:
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.01, 0.03, 0.01],
                "B": [0.005, 0.004, 0.006, 0.003, 0.005],
                "C": [-0.01, 0.01, 0.02, -0.02, 0.01],
            }
        )
        expected = pd.Series({"A": 0.02, "B": 0.005, "C": 0.01})
        cov = returns.cov()

        hrp = hrp_weights(returns)
        kelly = kelly_weights(expected, cov, max_abs_weight=0.5, max_gross=1.0)
        constrained = apply_weight_constraints(kelly, max_abs_weight=0.4, max_gross=0.8)
        scaled = scale_to_vol_target(constrained, cov, target_volatility=0.10)

        self.assertAlmostEqual(float(hrp.sum()), 1.0)
        self.assertLessEqual(float(constrained.abs().sum()), 0.8 + 1e-9)
        self.assertGreaterEqual(portfolio_volatility(scaled, cov), 0.0)

    def test_allocation_advisory_skips_until_research_inputs_exist(self) -> None:
        result = AllocationAdvisoryEngine().run(EngineContext())

        self.assertEqual(result.status, EngineStatus.SKIPPED)
        self.assertFalse(result.frame("requirements").empty)

    def test_allocation_advisory_runs_when_returns_exist(self) -> None:
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.01, 0.03, 0.01],
                "B": [0.005, 0.004, 0.006, 0.003, 0.005],
                "C": [-0.01, 0.01, 0.02, -0.02, 0.01],
            }
        )
        result = AllocationAdvisoryEngine().run(
            EngineContext(metadata={"allocation_returns": returns})
        )

        self.assertEqual(result.status, EngineStatus.PASS)
        self.assertIn("target_weights", result.signals)


if __name__ == "__main__":
    unittest.main()
