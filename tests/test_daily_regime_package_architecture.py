from __future__ import annotations

import importlib
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from oqp.research.daily_regimes import (
    PIPELINE_STAGE_ORDER,
    ArtifactReference,
    DailyRegimeConfig,
    DailyRegimeConfigError,
    PipelineExecutionError,
    PipelineGuards,
    PipelineMode,
    PipelineRequest,
    PipelineStage,
    ProbabilitySemantics,
    ProbabilityUse,
    ProspectiveProbabilityError,
    RawDailyBarContract,
    RunArtifactWriter,
    StageOutput,
    StageStatus,
    StateProbabilityBatch,
    StateProbabilityContract,
    StateProbabilityRecord,
    daily_regime_runtime_paths,
    default_capability_registry,
    execute_pipeline,
    load_daily_regime_config,
    make_clean_synthetic_fixture,
    restart_seeds,
)
from oqp.research.daily_regimes.capabilities import CapabilityStage
from oqp.research.daily_regimes.targets import (
    TargetBuildRequest,
    TargetFitResult,
    TargetFitScope,
    TargetKind,
    TargetSpec,
    TargetTrainingContext,
)
from oqp.research.daily_regimes.vqvae import (
    VQInputColumn,
    VQInputRole,
    VQObservationRef,
    VQSampleRole,
    VQWindow,
    VQVAEConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    REPO_ROOT
    / "notebooks"
    / "Phase_7_Research_Projects"
    / "07_01_daily_latent_regimes_cn_futures"
    / "config"
    / "preregistration.yaml"
)

ARCHITECTURE_MODULES = (
    "artifacts",
    "baselines",
    "capabilities",
    "config",
    "continuous_series",
    "contracts",
    "diagnostics",
    "evaluation",
    "features",
    "filtering",
    "folds",
    "hmm",
    "pipeline",
    "preprocessing",
    "reporting",
    "risk_throttle",
    "runtime_paths",
    "seeding",
    "smoke",
    "state_alignment",
    "stage3_fixtures",
    "stage4_fixtures",
    "synthetic",
    "targets",
    "vqvae",
)


class _ConfigComponent:
    component_id = "architecture_config"
    stage = PipelineStage.CONFIG
    owner_stage = 2

    def run(self, context):
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            updates={"resolved_config": {"run_id": context.request.run_id}},
        )


class _LatentScaffoldComponent:
    component_id = "architecture_hmm_scaffold"
    stage = PipelineStage.HMM_MODELS
    owner_stage = 7

    def run(self, context):
        del context
        return StageOutput.skipped_scaffold("HMM implementation belongs to Stage 7.")


class _SyntheticEvidenceComponent:
    component_id = "invalid_synthetic_evidence"
    stage = PipelineStage.MANIFEST
    owner_stage = 2

    def run(self, context):
        del context
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            artifacts=(
                ArtifactReference(
                    kind="model_evidence",
                    relative_path="validation/claim.json",
                    scientific_evidence=True,
                ),
            ),
        )


class _ReportingProbeComponent:
    component_id = "invalid_synthetic_reporting"
    stage = PipelineStage.REPORTING
    owner_stage = 13

    def __init__(self) -> None:
        self.ran = False

    def run(self, context):
        del context
        self.ran = True
        return StageOutput(status=StageStatus.SUCCEEDED)


class DailyRegimePackageArchitectureTests(unittest.TestCase):
    def test_all_architecture_modules_import(self) -> None:
        imported = [
            importlib.import_module(f"oqp.research.daily_regimes.{name}")
            for name in ARCHITECTURE_MODULES
        ]

        self.assertEqual(len(imported), len(ARCHITECTURE_MODULES))

    def test_preregistration_loads_with_stable_hash(self) -> None:
        first = load_daily_regime_config(PREREGISTRATION)
        second = load_daily_regime_config(PREREGISTRATION)

        self.assertEqual(first.project.id, "paper_01_daily_latent_regimes_cn_futures")
        self.assertEqual(first.data_construction.selector, "lagged_liquidity")
        self.assertEqual(first.data_construction.decision_lag_periods, 1)
        self.assertEqual(first.data_construction.primary_metric, "open_interest")
        self.assertEqual(
            first.data_construction.return_convention,
            "selected_contract_same_contract_close",
        )
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(len(first.config_hash), 64)

    def test_unknown_nested_configuration_key_is_rejected(self) -> None:
        payload = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
        payload["models"]["restartz"] = 20

        with self.assertRaisesRegex(DailyRegimeConfigError, "restartz"):
            DailyRegimeConfig.from_mapping(payload)

    def test_capability_registry_is_conservative(self) -> None:
        registry = default_capability_registry()

        self.assertEqual(
            registry.require("configuration").stage,
            CapabilityStage.SYNTHETIC_VERIFIED,
        )
        self.assertEqual(
            registry.require("pipeline_orchestration").stage,
            CapabilityStage.SYNTHETIC_VERIFIED,
        )
        self.assertEqual(
            registry.require("smoke_runner").stage,
            CapabilityStage.SYNTHETIC_VERIFIED,
        )
        self.assertEqual(
            registry.require("continuous_series").stage,
            CapabilityStage.SYNTHETIC_VERIFIED,
        )
        self.assertEqual(
            registry.require("features").stage,
            CapabilityStage.SYNTHETIC_VERIFIED,
        )
        self.assertEqual(
            registry.require("preprocessing").stage,
            CapabilityStage.SYNTHETIC_VERIFIED,
        )
        self.assertEqual(
            registry.require("hmm_models").stage,
            CapabilityStage.DECLARED,
        )
        self.assertEqual(registry.paper_eligible_capabilities(), ())

    def test_clean_synthetic_fixture_is_deterministic_and_quarantined(self) -> None:
        first = make_clean_synthetic_fixture()
        second = make_clean_synthetic_fixture()

        pd.testing.assert_frame_equal(first.contract_rows, second.contract_rows)
        pd.testing.assert_frame_equal(first.smoke_panel, second.smoke_panel)
        pd.testing.assert_frame_equal(first.truth, second.truth)
        self.assertEqual(len(first.contract_rows), 240)
        self.assertNotIn("hidden_state", first.smoke_panel.columns)
        self.assertIn("hidden_state", first.truth.columns)
        self.assertFalse(first.metadata["scientific_evidence"])

    def test_raw_contract_accepts_clean_fixture(self) -> None:
        fixture = make_clean_synthetic_fixture()

        report = RawDailyBarContract(require_sorted=True).validate(
            fixture.contract_rows
        )

        self.assertEqual(report.row_count, 240)
        self.assertEqual(report.entity_count, 2)
        self.assertTrue(report.is_sorted)

    def test_runtime_paths_resolve_without_creating_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = daily_regime_runtime_paths(
                artifact_root=root / "artifacts",
                data_root=root / "data",
            )

            self.assertFalse(paths.artifact_root.exists())
            self.assertEqual(
                paths.mode_dir("synthetic_smoke"),
                paths.synthetic_dir,
            )
            paths.ensure_directories(include_holdout=False)
            self.assertTrue(paths.synthetic_dir.is_dir())
            self.assertFalse(paths.holdout_dir.exists())

    def test_restart_seeds_are_reproducible_and_unique(self) -> None:
        first = restart_seeds(42, 20)
        second = restart_seeds(42, 20)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_synthetic_pipeline_guards_reject_privileged_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Synthetic smoke runs must forbid"):
            PipelineRequest(
                run_id="invalid_smoke",
                mode=PipelineMode.SYNTHETIC_SMOKE,
                seed=42,
                artifact_root=Path("/tmp/invalid-smoke"),
                guards=PipelineGuards(market_data_allowed=True),
            )

    def test_pipeline_records_later_scaffold_without_faking_success(self) -> None:
        request = PipelineRequest(
            run_id="architecture_smoke",
            mode=PipelineMode.SYNTHETIC_SMOKE,
            seed=42,
            artifact_root=Path("/tmp/architecture-smoke"),
        )

        result = execute_pipeline(
            request,
            (_ConfigComponent(), _LatentScaffoldComponent()),
        )

        self.assertEqual(result.records[0].status, StageStatus.SUCCEEDED)
        self.assertEqual(result.records[1].status, StageStatus.SKIPPED_SCAFFOLD)
        self.assertEqual(result.scaffolded_stages, (PipelineStage.HMM_MODELS,))

    def test_pipeline_has_explicit_fold_model_and_diagnostic_stages(self) -> None:
        order = {stage: index for index, stage in enumerate(PIPELINE_STAGE_ORDER)}

        self.assertLess(order[PipelineStage.FOLDS], order[PipelineStage.TARGETS])
        self.assertLess(order[PipelineStage.HMM_MODELS], order[PipelineStage.VQVAE])
        self.assertLess(order[PipelineStage.VQVAE], order[PipelineStage.DIAGNOSTICS])
        self.assertEqual(
            len(
                {
                    PipelineStage.HMM_MODELS,
                    PipelineStage.VQVAE,
                    PipelineStage.DIAGNOSTICS,
                }
            ),
            3,
        )

    def test_synthetic_pipeline_rejects_scientific_evidence(self) -> None:
        request = PipelineRequest(
            run_id="invalid_synthetic_evidence",
            mode=PipelineMode.SYNTHETIC_SMOKE,
            seed=42,
            artifact_root=Path("/tmp/invalid-synthetic-evidence"),
        )

        with self.assertRaisesRegex(PipelineExecutionError, "cannot emit"):
            execute_pipeline(request, (_SyntheticEvidenceComponent(),))

    def test_synthetic_reporting_is_rejected_before_component_runs(self) -> None:
        request = PipelineRequest(
            run_id="invalid_synthetic_reporting",
            mode=PipelineMode.SYNTHETIC_SMOKE,
            seed=42,
            artifact_root=Path("/tmp/invalid-synthetic-reporting"),
        )
        component = _ReportingProbeComponent()

        with self.assertRaisesRegex(PipelineExecutionError, "manuscript_writes"):
            execute_pipeline(request, (component,))
        self.assertFalse(component.ran)

    def test_tail_target_requires_fold_local_fitted_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "training-fold-only"):
            TargetSpec(
                name="next_day_tail_loss_event",
                kind=TargetKind.BINARY,
                horizon_periods=1,
            )

        spec = TargetSpec(
            name="next_day_tail_loss_event",
            kind=TargetKind.BINARY,
            horizon_periods=1,
            fit_scope=TargetFitScope.TRAINING_FOLD_ONLY,
        )
        request = TargetBuildRequest(specs=(spec,))
        context = TargetTrainingContext(
            fold_id="fold_000",
            training_start=date(2000, 1, 3),
            training_end=date(2000, 6, 30),
            training_rows_hash="a" * 64,
            training_row_count=120,
        )

        self.assertTrue(request.requires_training_fit)
        with self.assertRaisesRegex(ValueError, "requires fitted parameters"):
            TargetFitResult(
                specs=(spec,),
                context=context,
                builder_id="target_builder_v1",
                parameter_hash="b" * 64,
            )

    def test_artifact_writer_creates_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = RunArtifactWriter(
                temp_dir,
                run_id="architecture_smoke",
                mode="synthetic_smoke",
                workspace_root=temp_dir,
            )
            payload_record = writer.write_json("reports/payload.json", {"ok": True})
            written = writer.finalize(
                project_id="paper_01_daily_latent_regimes_cn_futures",
                config_hash="a" * 64,
                seed=42,
                capabilities=default_capability_registry().capabilities,
                created_at="2000-01-01T00:00:00+00:00",
            )

            self.assertEqual(len(payload_record.sha256), 64)
            self.assertEqual(len(written.record.sha256), 64)
            self.assertEqual(written.manifest.status, "complete")
            self.assertTrue(Path(temp_dir, written.record.path).is_file())

    def test_prospective_contract_rejects_smoothed_probabilities(self) -> None:
        record = StateProbabilityRecord(
            product_id="SYN_A",
            sequence_id="SYN_A:0",
            row_id="SYN_A:20000103",
            trading_date=date(2000, 1, 3),
            information_date=date(2000, 1, 4),
            model_id="descriptive_only",
            refit_id="fold_000",
            state_labels=("state_0", "state_1"),
            probabilities=(0.4, 0.6),
            semantics=ProbabilitySemantics.SMOOTHED,
            forecast_horizon_periods=0,
        )
        batch = StateProbabilityBatch(
            model_id="descriptive_only",
            feature_set_id="SMOKE3",
            fold_id="fold_000",
            records=(record,),
        )

        with self.assertRaises(ProspectiveProbabilityError):
            batch.require_eligible_for(ProbabilityUse.RISK_DECISION)

    def test_public_dataframe_contract_rejects_smoothed_probabilities(self) -> None:
        frame = pd.DataFrame(
            {
                "product": ["SYN_A"],
                "trading_date": [date(2000, 1, 3)],
                "information_date": [date(2000, 1, 4)],
                "fold_id": ["fold_000"],
                "model_id": ["descriptive_only"],
                "probability_semantics": [ProbabilitySemantics.SMOOTHED.value],
                "forecast_horizon_periods": [0],
                "p_state_0": [0.4],
                "p_state_1": [0.6],
            }
        )

        with self.assertRaises(ProspectiveProbabilityError):
            StateProbabilityContract().validate(frame)

        report = StateProbabilityContract(
            intended_use=ProbabilityUse.DESCRIPTION
        ).validate(frame)
        self.assertEqual(report.row_count, 1)

    def test_vq_contract_rejects_target_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "FEATURE role"):
            VQVAEConfig(
                input_columns=(VQInputColumn("next_day_return", VQInputRole.TARGET),)
            )

    def test_vq_window_rejects_cross_fold_observations(self) -> None:
        observations = tuple(
            VQObservationRef(
                row_id=f"SYN_A:{index}",
                product_id="SYN_A",
                sequence_id="SYN_A:0",
                fold_id=fold_id,
                trading_date=date(2000, 1, 3) + timedelta(days=index),
                sample_role=VQSampleRole.TRAIN,
            )
            for index, fold_id in enumerate(("fold_000", "fold_001"))
        )

        with self.assertRaisesRegex(ValueError, "fold boundaries"):
            VQWindow(
                observations=observations,
                values=((0.1,), (0.2,)),
                input_columns=("log_gk_gap_variance_z",),
            )


if __name__ == "__main__":
    unittest.main()
