from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from oqp.research.daily_regimes.artifacts import build_run_id
from oqp.research.daily_regimes.config import load_preregistration_config
from oqp.research.daily_regimes.pipeline import PipelineStage
from oqp.research.daily_regimes.smoke import (
    SMOKE_RUN_LABEL,
    SmokeConfigError,
    combined_execution_hash,
    load_smoke_config,
    run_smoke,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (
    REPO_ROOT
    / "notebooks"
    / "Phase_7_Research_Projects"
    / "07_01_daily_latent_regimes_cn_futures"
)
SMOKE_CONFIG = PROJECT_ROOT / "config" / "smoke.yaml"
PREREGISTRATION = PROJECT_ROOT / "config" / "preregistration.yaml"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "research" / "run_daily_regime_smoke.py"


class DailyRegimeSmokeRunnerTests(unittest.TestCase):
    def test_repository_smoke_config_loads_strictly(self) -> None:
        config = load_smoke_config(SMOKE_CONFIG)

        self.assertEqual(config.project_id, "paper_01_daily_latent_regimes_cn_futures")
        self.assertEqual(config.repetitions, 2)
        self.assertEqual(config.fixture.products, ("SYN_A", "SYN_B"))
        self.assertEqual(len(config.config_hash), 64)

    def test_unknown_keys_and_permission_like_mode_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _write_test_config(Path(temp_dir))
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload["mode"] = "validation"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(SmokeConfigError, "Unknown keys"):
                load_smoke_config(config_path)

    def test_preregistration_path_cannot_escape_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _write_test_config(Path(temp_dir))
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload["preregistration_config"] = "../preregistration.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(SmokeConfigError, "without '\\.\\.'"):
                load_smoke_config(config_path)

    def test_execution_identity_changes_with_fixture_geometry(self) -> None:
        config = load_smoke_config(SMOKE_CONFIG)
        preregistration = load_preregistration_config(PREREGISTRATION)
        changed = replace(
            config,
            fixture=replace(config.fixture, periods=config.fixture.periods + 1),
        )

        first_hash = combined_execution_hash(config, preregistration)
        second_hash = combined_execution_hash(changed, preregistration)
        first_run_id = build_run_id(
            config.project_id,
            first_hash,
            config.seed,
            label=SMOKE_RUN_LABEL,
        )
        second_run_id = build_run_id(
            changed.project_id,
            second_hash,
            changed.seed,
            label=SMOKE_RUN_LABEL,
        )

        self.assertNotEqual(first_hash, second_hash)
        self.assertNotEqual(first_run_id, second_run_id)

    def test_two_attempt_gate_is_exact_quarantined_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_test_config(root / "config")
            output_root = root / "output"

            result = run_smoke(config_path, output_root=output_root)

            self.assertTrue(result.reproducibility_passed)
            self.assertEqual(len(result.attempts), 2)
            self.assertEqual(
                result.attempts[0].computational_digest,
                result.attempts[1].computational_digest,
            )
            self.assertEqual(
                dict(result.attempts[0].computational_hashes),
                dict(result.attempts[1].computational_hashes),
            )
            expected_artifacts = {
                "config/preregistration.resolved.json",
                "config/smoke.resolved.json",
                "fixtures/contract_rows.csv",
                "fixtures/dominant_mapping.csv",
                "fixtures/smoke_panel.csv",
                "quarantine/synthetic_truth.csv",
                "reports/capabilities.json",
                "reports/contract_validation.json",
                "reports/pipeline_ledger.json",
                "reports/seed.json",
                "reports/smoke_summary.json",
                "reports/stage3_contract_validation.json",
                "reports/stage3_summary.json",
                "reports/stage4_feature_summary.json",
                "reports/stage4_formula_registry.json",
                "reports/stage4_preprocessing_hpca3.json",
                "reports/stage4_preprocessing_m3.json",
                "reports/stage4_preprocessing_summary.json",
                "stage3/contract_rows.csv",
                "stage3/continuous_daily_panel.csv",
                "stage3/roll_ledger.csv",
                "stage4/feature_fixture_panel.csv",
                "stage4/features_hpca3_transformed.csv",
                "stage4/features_m3_transformed.csv",
                "stage4/features_raw.csv",
            }
            self.assertEqual(
                set(result.attempts[0].computational_hashes), expected_artifacts
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertFalse(report["scientific_evidence"])
            self.assertFalse(report["paper_eligible"])
            self.assertIn("manifests", report["comparison_scope"])

            for attempt in result.attempts:
                manifest = attempt.manifest.manifest
                self.assertEqual(manifest.status, "complete")
                self.assertEqual(manifest.mode, "synthetic")
                self.assertFalse(manifest.metadata["scientific_evidence"])
                self.assertFalse(manifest.metadata["paper_eligible"])
                self.assertTrue(
                    all(
                        not capability["paper_eligible"]
                        for capability in manifest.capabilities
                    )
                )
                self.assertTrue(
                    all(
                        not Path(record.path).is_absolute()
                        for record in manifest.artifacts
                    )
                )
                self.assertEqual(
                    next(
                        record.kind
                        for record in manifest.artifacts
                        if record.name == "synthetic_truth.csv"
                    ),
                    "quarantined_truth",
                )
                self.assertTrue(
                    all(
                        not artifact.scientific_evidence
                        for artifact in attempt.pipeline.artifacts
                    )
                )
                self.assertNotIn(
                    PipelineStage.REPORTING,
                    tuple(record.stage for record in attempt.pipeline.records),
                )
                continuous_record = next(
                    record
                    for record in attempt.pipeline.records
                    if record.stage is PipelineStage.CONTINUOUS_SERIES
                )
                self.assertEqual(continuous_record.status.value, "succeeded")
                feature_record = next(
                    record
                    for record in attempt.pipeline.records
                    if record.stage is PipelineStage.FEATURES
                )
                preprocessing_record = next(
                    record
                    for record in attempt.pipeline.records
                    if record.stage is PipelineStage.PREPROCESSING
                )
                self.assertEqual(feature_record.status.value, "succeeded")
                self.assertEqual(preprocessing_record.status.value, "succeeded")

                panel_columns = pd.read_csv(
                    attempt.run_dir / "fixtures" / "smoke_panel.csv", nrows=0
                ).columns
                truth_columns = pd.read_csv(
                    attempt.run_dir / "quarantine" / "synthetic_truth.csv", nrows=0
                ).columns
                self.assertNotIn("hidden_state", panel_columns)
                self.assertIn("hidden_state", truth_columns)
                summary = json.loads(
                    (attempt.run_dir / "reports" / "smoke_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(summary["stage4_causal_features_verified"])
                self.assertTrue(summary["stage4_fold_local_preprocessing_verified"])

            with self.assertRaises(FileExistsError):
                run_smoke(config_path, output_root=output_root)

    def test_thin_runner_script_completes_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_test_config(root / "config")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_SCRIPT),
                    "--config",
                    str(config_path),
                    "--output-root",
                    str(root / "cli_output"),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout.strip())
            self.assertTrue(payload["reproducibility_passed"])
            self.assertEqual(payload["attempt_count"], 2)


def _write_test_config(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PREREGISTRATION, directory / "preregistration.yaml")
    payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["fixture"]["periods"] = 8
    payload["fixture"]["regime_block_periods"] = 2
    config_path = directory / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


if __name__ == "__main__":
    unittest.main()
