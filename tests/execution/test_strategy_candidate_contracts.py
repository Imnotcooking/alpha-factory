from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.contracts import (  # noqa: E402
    CandidateIntakeState,
    CandidateMetrics,
    CandidateStatus,
    MarketVertical,
    StrategyCandidate,
    candidate_from_backtest_row,
    load_latest_candidate_from_research_db,
    load_strategy_candidate_artifacts,
    write_strategy_candidate_artifact,
)


class StrategyCandidateContractTests(unittest.TestCase):
    def test_writes_and_loads_strategy_candidate_artifact(self) -> None:
        candidate = StrategyCandidate(
            candidate_id="candidate-run-123",
            strategy_id="fac_demo",
            source="unit_test",
            promotion_status=CandidateStatus.PAPER_CANDIDATE,
            native_market_vertical="FUTURES_CN",
            target_market_vertical="FUTURES_CN",
            research_run_id="run_123",
            metrics=CandidateMetrics(holdout_ic=0.02, sharpe_ratio=1.1),
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = write_strategy_candidate_artifact(candidate, directory)
            result = load_strategy_candidate_artifacts(directory)

        self.assertEqual(path.name, "candidate-run-123.json")
        self.assertEqual(len(result.loaded), 1)
        loaded = result.loaded[0].candidate
        self.assertEqual(loaded.candidate_id, "candidate-run-123")
        self.assertTrue(loaded.can_enter_paper_queue)
        self.assertEqual(loaded.intake_state, CandidateIntakeState.PAPER_QUEUE_ELIGIBLE)
        self.assertEqual(loaded.intake_state_label, "Paper Queue Eligible")
        self.assertEqual(loaded.market_scoped_status, "FUTURES_CN:paper_candidate")
        self.assertEqual(loaded.metrics.sharpe_ratio, 1.1)

    def test_cross_market_target_cannot_enter_paper_queue_without_own_test(self) -> None:
        candidate = StrategyCandidate(
            candidate_id="candidate-cross-market",
            strategy_id="fac_cn_only",
            source="unit_test",
            promotion_status=CandidateStatus.PAPER_CANDIDATE,
            native_market_vertical="FUTURES_CN",
            tested_market_vertical="FUTURES_CN",
            target_market_vertical="EQUITY_US",
            intended_market_verticals=("FUTURES_CN",),
        )

        self.assertEqual(candidate.market_scoped_status, "EQUITY_US:paper_candidate")
        self.assertTrue(candidate.is_cross_market_translation)
        self.assertFalse(candidate.can_enter_paper_queue)
        self.assertEqual(candidate.intake_state, CandidateIntakeState.NEEDS_REVIEW)

    def test_collects_invalid_candidate_artifact_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "bad.json").write_text("not-json", encoding="utf-8")
            (directory / "good.json").write_text(
                json.dumps(
                    {
                        "candidate_id": "candidate-good",
                        "strategy_id": "fac_good",
                        "source": "unit_test",
                    }
                ),
                encoding="utf-8",
            )
            result = load_strategy_candidate_artifacts(directory)

        self.assertEqual(len(result.loaded), 1)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.loaded[0].candidate.strategy_id, "fac_good")

    def test_maps_backtest_row_to_research_only_candidate(self) -> None:
        candidate = candidate_from_backtest_row(
            {
                "run_id": "run_abc",
                "factor_id": "fac_abc",
                "asset_class": "FUTURES_CN",
                "market_vertical": "FUTURES_CN",
                "dataset_id": "dataset_daily",
                "universe_id": "universe_1",
                "data_frequency": "daily",
                "execution_assumption": "close_signal_next_open_to_close",
                "evaluation_geometry": "cross_sectional",
                "ic_metric": "rank_ic_spearman",
                "holdout_ic": 0.012,
                "sharpe_ratio": -0.43,
                "stat_metric_p_value": 0.08,
            },
            diagnostics={
                "failure_code": "holdout_not_positive",
                "suggested_action": "review",
            },
        )

        self.assertEqual(candidate.candidate_id, "candidate-run_abc")
        self.assertEqual(candidate.promotion_status, CandidateStatus.RESEARCH_ONLY)
        self.assertEqual(candidate.native_market_vertical, "FUTURES_CN")
        self.assertEqual(candidate.tested_market_vertical, "FUTURES_CN")
        self.assertEqual(candidate.metrics.holdout_ic, 0.012)
        self.assertFalse(candidate.can_enter_paper_queue)
        self.assertEqual(candidate.intake_state, CandidateIntakeState.RESEARCH_SNAPSHOT)
        self.assertIn("holdout_not_positive", candidate.notes or "")

    def test_market_vertical_normalizes_asset_taxonomy_aliases(self) -> None:
        candidate = StrategyCandidate(
            candidate_id="candidate-hk",
            strategy_id="fac_hk",
            source="unit_test",
            native_market_vertical="hk equities",
            target_market_vertical=MarketVertical.EQUITY_HK,
        )

        self.assertEqual(candidate.native_market_vertical, "EQUITY_HK")
        self.assertEqual(candidate.target_market_vertical, "EQUITY_HK")

    def test_loads_latest_candidate_from_research_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "research_memory.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE backtest_runs (
                        run_id TEXT,
                        factor_id TEXT,
                        market_vertical TEXT,
                        holdout_ic REAL,
                        sharpe_ratio REAL,
                        timestamp TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO backtest_runs
                    VALUES ('run_old', 'fac_demo', 'FUTURES_CN', -0.01, -1.0, '2026-01-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO backtest_runs
                    VALUES ('run_new', 'fac_demo', 'FUTURES_CN', 0.02, 0.5, '2026-01-02')
                    """
                )

            candidate = load_latest_candidate_from_research_db(
                db_path,
                factor_id="fac_demo",
            )

        self.assertEqual(candidate.research_run_id, "run_new")
        self.assertEqual(candidate.metrics.holdout_ic, 0.02)


if __name__ == "__main__":
    unittest.main()
