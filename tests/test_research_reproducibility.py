from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from oqp.research import (
    EvidenceTicketRecord,
    ModelArtifactStore,
    build_data_fingerprint,
    get_evidence_ticket,
    latest_model_artifact,
    list_evidence_tickets,
    record_from_artifact,
    record_evidence_ticket,
    record_research_trial,
    register_model_artifact,
    update_evidence_ticket_status,
)


class ResearchReproducibilityTests(unittest.TestCase):
    def test_trial_ledger_records_vertical_metadata_and_adjustments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "research.db"
            first = record_research_trial(
                str(db_path),
                factor_id="fac_demo",
                research_family="family_demo",
                trial_signature_payload={"factor": "fac_demo", "window": 20},
                run_id="run_first",
                experiment_source="unit_test",
                asset_class="FUTURES_CN",
                vertical_metadata={
                    "market_vertical": "FUTURES_CN",
                    "dataset_id": "fixture_tick",
                    "data_frequency": "tick",
                    "data_tradability": "executable",
                },
                metric_name="rank_ic",
                metric_value=0.12,
                raw_p_value=0.02,
                sample_size=120,
                metadata={"note": "first pass"},
            )
            second = record_research_trial(
                str(db_path),
                factor_id="fac_demo",
                research_family="family_demo",
                trial_signature_payload={"factor": "fac_demo", "window": 40},
                run_id="run_second",
                experiment_source="unit_test",
                asset_class="FUTURES_CN",
                vertical_metadata={"market_vertical": "FUTURES_CN"},
                metric_name="rank_ic",
                metric_value=0.08,
                raw_p_value=0.04,
                sample_size=120,
            )

            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT run_id, trial_count_m, bonferroni_p_value, fdr_q_value,
                           significance, market_vertical, dataset_id, metadata_json
                    FROM research_trials
                    ORDER BY raw_p_value
                    """
                ).fetchall()

        self.assertEqual(first.trial_count, 1)
        self.assertEqual(second.trial_count, 2)
        self.assertEqual([row[1] for row in rows], [2, 2])
        self.assertAlmostEqual(rows[0][2], 0.04)
        self.assertAlmostEqual(rows[1][2], 0.08)
        self.assertAlmostEqual(rows[0][3], 0.04)
        self.assertAlmostEqual(rows[1][3], 0.04)
        self.assertEqual(rows[0][4], "survives_multiple_testing")
        self.assertEqual(rows[1][4], "fails_multiple_testing")
        self.assertEqual(rows[0][5], "FUTURES_CN")
        self.assertEqual(rows[0][6], "fixture_tick")
        self.assertEqual(json.loads(rows[0][7]), {"note": "first pass"})

    def test_evidence_ticket_records_analyst_workflow_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "research.db"
            trial = record_research_trial(
                str(db_path),
                factor_id="tick_pulse_relative_velocity",
                research_family="tick_pulse_lab",
                trial_signature_payload={"symbol": "au2608", "rule": "relative_velocity"},
                run_id="tick_trial_001",
                experiment_source="tick_pulse_lab",
                metric_name="event_accuracy_vs_base_rate",
                metric_value=0.07,
                raw_p_value=0.01,
                sample_size=240,
            )
            ticket = record_evidence_ticket(
                str(db_path),
                title="Relative velocity pulse has follow-through",
                source_page="03_Tick_Event_Study",
                evidence_type="microstructure_hypothesis",
                stage="hypothesis_tested",
                status="open",
                decision="promote_to_validation",
                thesis="Fast relative velocity pulses beat the base continuation rate.",
                factor_id=trial.factor_id,
                research_family=trial.research_family,
                run_id=trial.run_id,
                trial_signature=trial.trial_signature,
                metric_name="lift",
                metric_value=0.07,
                confidence_score=0.82,
                priority=1,
                metrics={"base_rate": 0.42, "event_rate": 0.49, "events": 240},
                artifacts=[{"kind": "cache", "path": "runtime/db/research/alpha_lab/research_memory.db"}],
                context={"market_vertical": "FUTURES_CN", "data_frequency": "tick"},
                metadata={"next_page": "08_Factor_Review"},
            )
            record_evidence_ticket(
                str(db_path),
                ticket_id=ticket.ticket_id,
                title=ticket.title,
                source_page=ticket.source_page,
                evidence_type=ticket.evidence_type,
                stage=ticket.stage,
                status="ready_for_review",
                decision=ticket.decision,
                factor_id=trial.factor_id,
                research_family=trial.research_family,
                run_id=trial.run_id,
                trial_signature=trial.trial_signature,
                metric_name=ticket.metric_name,
                metric_value=ticket.metric_value,
                confidence_score=ticket.confidence_score,
                metrics={"base_rate": 0.42, "event_rate": 0.49, "events": 240},
            )

            tickets = list_evidence_tickets(str(db_path), status="ready_for_review")
            fetched = get_evidence_ticket(str(db_path), ticket.ticket_id)

        self.assertIsInstance(ticket, EvidenceTicketRecord)
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets.iloc[0]["ticket_id"], ticket.ticket_id)
        self.assertEqual(tickets.iloc[0]["metrics"]["events"], 240)
        self.assertEqual(tickets.iloc[0]["artifacts"], [])
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched["status"], "ready_for_review")
        self.assertEqual(fetched["trial_signature"], trial.trial_signature)

    def test_evidence_ticket_review_update_preserves_evidence_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "research.db"
            ticket = record_evidence_ticket(
                str(db_path),
                title="Spread dislocation survives initial review",
                source_page="04_Arbitrage_Lab",
                evidence_type="state_space_hypothesis",
                stage="hypothesis_tested",
                status="ready_for_review",
                decision="promote_to_validation",
                thesis="Dual Kalman spread dislocations mean revert after extreme z-score states.",
                factor_id="dual_kalman_spread_dislocation",
                research_family="adaptive_relationship_lab",
                run_id="relationship_run_001",
                trial_signature="trial_signature_001",
                metric_name="mean_reversion_score",
                metric_value=0.31,
                confidence_score=0.74,
                priority=2,
                metrics={"half_life": 4.5, "z_score": 2.2},
                artifacts=[{"kind": "model", "path": "runtime/artifacts/relationship/model.json"}],
                context={"market_vertical": "FUTURES_CN", "pair": "au-ag"},
                metadata={"next_page": "08_Factor_Review"},
            )

            reviewed = update_evidence_ticket_status(
                str(db_path),
                ticket.ticket_id,
                status="reviewed",
                reviewer_note="Evidence is coherent enough for validation planning.",
                metadata_patch={"review_source_page": "08_Factor_Review"},
                reviewer="unit_test",
            )

        self.assertEqual(reviewed["status"], "reviewed")
        self.assertEqual(reviewed["decision"], "promote_to_validation")
        self.assertEqual(reviewed["metrics"], {"half_life": 4.5, "z_score": 2.2})
        self.assertEqual(
            reviewed["artifacts"],
            [{"kind": "model", "path": "runtime/artifacts/relationship/model.json"}],
        )
        self.assertEqual(reviewed["context"], {"market_vertical": "FUTURES_CN", "pair": "au-ag"})
        self.assertEqual(reviewed["metadata"]["next_page"], "08_Factor_Review")
        self.assertEqual(reviewed["metadata"]["review_source_page"], "08_Factor_Review")
        self.assertEqual(reviewed["metadata"]["last_review_status"], "reviewed")
        self.assertEqual(reviewed["metadata"]["last_reviewer"], "unit_test")
        self.assertEqual(len(reviewed["metadata"]["review_notes"]), 1)
        self.assertEqual(reviewed["metadata"]["review_notes"][0]["status"], "reviewed")
        self.assertIn("coherent enough", reviewed["metadata"]["review_notes"][0]["note"])

    def test_model_artifact_store_and_registry_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            source_dir = workspace_root / "scratch"
            data_dir = workspace_root / "data"
            source_dir.mkdir()
            data_dir.mkdir()
            source_artifact = source_dir / "model.txt"
            source_data = data_dir / "features.csv"
            source_artifact.write_text("trained-weights-v1\n", encoding="utf-8")
            source_data.write_text("feature_a,feature_b,target\n1,2,0\n", encoding="utf-8")

            store = ModelArtifactStore(root_dir="model_artifacts", workspace_root=workspace_root)
            stored = store.archive_file(
                source_artifact,
                model_name="Demo Model",
                artifact_id="demo_artifact_v1",
                filename="model.bin",
            )
            data_fingerprint = build_data_fingerprint(
                source_data,
                workspace_root=workspace_root,
            )
            record = record_from_artifact(
                artifact_id=stored.artifact_id,
                model_name="Demo Model",
                model_type="fixture",
                artifact_path=workspace_root / stored.path,
                artifact_format="txt",
                artifact_sha256=stored.sha256,
                artifact_size_bytes=stored.size_bytes,
                factor_id="fac_demo",
                legacy_path=source_artifact,
                source_module="tests.test_research_reproducibility",
                data_fingerprint=data_fingerprint,
                feature_cols=["feature_a", "feature_b"],
                target_col="target",
                split_policy={"mode": "walk_forward", "embargo_days": 1},
                metrics={"validation_ic": 0.12},
                hyperparams={"depth": 2},
                metadata={"purpose": "fixture"},
                workspace_root=workspace_root,
            )
            db_path = workspace_root / "registry.db"
            register_model_artifact(record, db_path=db_path)
            latest = latest_model_artifact("Demo Model", db_path=db_path)

            self.assertTrue((workspace_root / stored.path).exists())
            self.assertEqual(len(stored.sha256), 64)
            self.assertIsNotNone(data_fingerprint)
            assert data_fingerprint is not None
            self.assertEqual(data_fingerprint.path, "data/features.csv")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest["artifact_id"], "demo_artifact_v1")
            self.assertEqual(latest["artifact_path"], stored.path)
            self.assertEqual(latest["legacy_path"], "scratch/model.txt")
            self.assertEqual(latest["data_path"], "data/features.csv")
            self.assertEqual(latest["feature_count"], 2)
            self.assertEqual(json.loads(latest["metrics_json"]), {"validation_ic": 0.12})

    def test_promoted_reproducibility_objects_are_public_research_surface(self) -> None:
        self.assertEqual(record_research_trial.__module__, "oqp.research.trials")
        self.assertEqual(record_evidence_ticket.__module__, "oqp.research.trials")
        self.assertEqual(update_evidence_ticket_status.__module__, "oqp.research.trials")
        self.assertEqual(EvidenceTicketRecord.__module__, "oqp.research.trials")
        self.assertEqual(ModelArtifactStore.__module__, "oqp.research.artifacts")


if __name__ == "__main__":
    unittest.main()
