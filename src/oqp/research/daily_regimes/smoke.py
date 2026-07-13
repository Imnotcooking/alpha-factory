"""Offline, deterministic cumulative smoke gate for the daily-regime paper.

The runner exercises configuration, contracts, orchestration, artifact writing,
Stage 3 continuous construction and Stage 4 feature/preprocessing algorithms
are exercised using synthetic data only.  No smoke output is scientific
evidence for a paper claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from uuid import uuid4

import yaml  # type: ignore[import-untyped]
import pandas as pd

from oqp.research.daily_regimes.artifacts import (
    RunArtifactWriter,
    WrittenManifest,
    build_run_id,
)
from oqp.research.daily_regimes.capabilities import default_capability_registry
from oqp.research.daily_regimes.config import (
    DailyRegimeConfig,
    load_preregistration_config,
)
from oqp.research.daily_regimes.continuous_series import (
    ContinuousSeriesConfig,
    ContinuousSeriesResult,
    build_continuous_series,
)
from oqp.research.daily_regimes.contracts import RawDailyBarContract
from oqp.research.daily_regimes.features import (
    H7_COLUMNS,
    M3_RAW_COLUMNS,
    FeatureBuildResult,
    build_features,
    feature_formula_registry,
    feature_set_request,
)
from oqp.research.daily_regimes.pipeline import (
    ArtifactReference,
    PipelineContext,
    PipelineMode,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    StageOutput,
    StageRecord,
    StageStatus,
    execute_pipeline,
)
from oqp.research.daily_regimes.runtime_paths import daily_regime_runtime_paths
from oqp.research.daily_regimes.preprocessing import (
    FittedFoldLocalPreprocessor,
    PreprocessingConfig,
    PreprocessingFitContext,
    PreprocessingResult,
    fit_preprocessor,
)
from oqp.research.daily_regimes.seeding import normalize_seed, seed_everything
from oqp.research.daily_regimes.stage3_fixtures import (
    Stage3AdversarialFixture,
    make_stage3_adversarial_fixture,
)
from oqp.research.daily_regimes.stage4_fixtures import (
    Stage4SyntheticFixture,
    make_stage4_synthetic_fixture,
)
from oqp.research.daily_regimes.synthetic import (
    CleanSyntheticFixtureFactory,
    SyntheticFixture,
    SyntheticFixtureConfig,
)


SMOKE_SCHEMA_VERSION = "daily_regime_smoke_v1"
SMOKE_IMPLEMENTATION_VERSION = "daily_regime_cumulative_smoke_runner_v5"
SMOKE_MODE = PipelineMode.SYNTHETIC_SMOKE
SMOKE_ATTEMPTS = 2
SMOKE_RUN_LABEL = "cumulative_smoke"


class SmokeConfigError(ValueError):
    """Raised when the dedicated smoke configuration is invalid."""


class SmokeReproducibilityError(RuntimeError):
    """Raised when the two independent attempts disagree."""


@dataclass(frozen=True, slots=True)
class SmokeFixtureSpec:
    products: tuple[str, ...]
    periods: int
    start_date: date
    state_count: int
    regime_block_periods: int

    @classmethod
    def from_mapping(cls, value: Any) -> "SmokeFixtureSpec":
        data = _strict_mapping(
            value,
            {
                "products",
                "periods",
                "start_date",
                "state_count",
                "regime_block_periods",
            },
            "fixture",
        )
        products_value = _required(data, "products", "fixture")
        if isinstance(products_value, (str, bytes)) or not isinstance(
            products_value, Sequence
        ):
            raise SmokeConfigError("fixture.products must be a sequence of strings.")
        products = tuple(
            _nonempty_text(item, f"fixture.products[{index}]")
            for index, item in enumerate(products_value)
        )
        start_date_value = _nonempty_text(
            _required(data, "start_date", "fixture"), "fixture.start_date"
        )
        try:
            parsed_start_date = date.fromisoformat(start_date_value)
        except ValueError as exc:
            raise SmokeConfigError("fixture.start_date must use YYYY-MM-DD.") from exc
        spec = cls(
            products=products,
            periods=_positive_int(
                _required(data, "periods", "fixture"), "fixture.periods"
            ),
            start_date=parsed_start_date,
            state_count=_positive_int(
                _required(data, "state_count", "fixture"), "fixture.state_count"
            ),
            regime_block_periods=_positive_int(
                _required(data, "regime_block_periods", "fixture"),
                "fixture.regime_block_periods",
            ),
        )
        # Reuse the fixture's stronger product/state/geometry validation.
        spec.to_fixture_config(seed=0)
        return spec

    def to_fixture_config(self, *, seed: int) -> SyntheticFixtureConfig:
        return SyntheticFixtureConfig(
            seed=seed,
            products=self.products,
            periods=self.periods,
            start_date=self.start_date,
            state_count=self.state_count,
            regime_block_periods=self.regime_block_periods,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "products": list(self.products),
            "periods": self.periods,
            "start_date": self.start_date.isoformat(),
            "state_count": self.state_count,
            "regime_block_periods": self.regime_block_periods,
        }


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    schema_version: str
    project_id: str
    preregistration_config: str
    seed: int
    repetitions: int
    fixture: SmokeFixtureSpec

    @classmethod
    def from_mapping(cls, value: Any) -> "SmokeConfig":
        data = _strict_mapping(
            value,
            {
                "schema_version",
                "project_id",
                "preregistration_config",
                "seed",
                "repetitions",
                "fixture",
            },
            "root",
        )
        schema_version = _nonempty_text(
            _required(data, "schema_version", "root"), "schema_version"
        )
        if schema_version != SMOKE_SCHEMA_VERSION:
            raise SmokeConfigError(
                f"schema_version must equal {SMOKE_SCHEMA_VERSION!r}."
            )
        preregistration_config = _nonempty_text(
            _required(data, "preregistration_config", "root"),
            "preregistration_config",
        )
        preregistration_path = PurePosixPath(preregistration_config)
        if preregistration_path.is_absolute() or ".." in preregistration_path.parts:
            raise SmokeConfigError(
                "preregistration_config must be a relative path without '..'."
            )
        repetitions = _positive_int(
            _required(data, "repetitions", "root"), "repetitions"
        )
        if repetitions != SMOKE_ATTEMPTS:
            raise SmokeConfigError(
                f"The Stage 2 gate requires exactly {SMOKE_ATTEMPTS} repetitions."
            )
        seed_value = _required(data, "seed", "root")
        try:
            seed = normalize_seed(seed_value)
        except (TypeError, ValueError) as exc:
            raise SmokeConfigError(str(exc)) from exc
        return cls(
            schema_version=schema_version,
            project_id=_nonempty_text(
                _required(data, "project_id", "root"), "project_id"
            ),
            preregistration_config=preregistration_config,
            seed=seed,
            repetitions=repetitions,
            fixture=SmokeFixtureSpec.from_mapping(_required(data, "fixture", "root")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "preregistration_config": self.preregistration_config,
            "seed": self.seed,
            "repetitions": self.repetitions,
            "fixture": self.fixture.to_dict(),
        }

    @property
    def config_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class SmokeAttemptResult:
    attempt: int
    run_dir: Path
    manifest_path: Path
    manifest: WrittenManifest
    computational_hashes: Mapping[str, str]
    computational_digest: str
    pipeline: PipelineResult


@dataclass(frozen=True, slots=True)
class SmokeRunResult:
    config_path: Path
    preregistration_path: Path
    output_root: Path
    run_id: str
    execution_hash: str
    attempts: tuple[SmokeAttemptResult, ...]
    reproducibility_passed: bool
    reproducibility_digest: str
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "implementation_version": SMOKE_IMPLEMENTATION_VERSION,
            "mode": SMOKE_MODE.value,
            "run_id": self.run_id,
            "execution_hash": self.execution_hash,
            "attempt_count": len(self.attempts),
            "reproducibility_passed": self.reproducibility_passed,
            "reproducibility_digest": self.reproducibility_digest,
            "report_path": self.report_path.as_posix(),
        }


@dataclass(frozen=True, slots=True)
class _ConfigComponent:
    execution_hash: str
    smoke_config_hash: str
    preregistration_hash: str

    component_id = "stage2_smoke_config"
    stage = PipelineStage.CONFIG
    owner_stage = 2

    def run(self, context: PipelineContext) -> StageOutput:
        del context
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            updates={
                "execution_hash": self.execution_hash,
                "smoke_config_hash": self.smoke_config_hash,
                "preregistration_hash": self.preregistration_hash,
            },
            artifacts=(
                _non_evidence("resolved_config", "config/smoke.resolved.json"),
                _non_evidence(
                    "resolved_config", "config/preregistration.resolved.json"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _SyntheticInputComponent:
    fixture: SyntheticFixture

    component_id = "stage2_clean_synthetic_fixture"
    stage = PipelineStage.SYNTHETIC_INPUT
    owner_stage = 2

    def run(self, context: PipelineContext) -> StageOutput:
        del context
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            updates={
                "fixture_id": self.fixture.fixture_id,
                "synthetic_row_count": len(self.fixture.contract_rows),
            },
            diagnostics={
                "contains_market_data": False,
                "scientific_evidence": False,
            },
            artifacts=(
                _non_evidence("synthetic_fixture", "fixtures/contract_rows.csv"),
                _non_evidence("synthetic_fixture", "fixtures/dominant_mapping.csv"),
                _non_evidence("synthetic_fixture", "fixtures/smoke_panel.csv"),
                _non_evidence("quarantined_truth", "quarantine/synthetic_truth.csv"),
                _non_evidence(
                    "contract_validation", "reports/contract_validation.json"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _ContinuousSeriesComponent:
    fixture: Stage3AdversarialFixture
    result: ContinuousSeriesResult

    component_id = "stage3_point_in_time_continuous_series"
    stage = PipelineStage.CONTINUOUS_SERIES
    owner_stage = 3

    def run(self, context: PipelineContext) -> StageOutput:
        del context
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            updates={
                "stage3_fixture_id": self.fixture.metadata["fixture_id"],
                "continuous_series_row_count": len(self.result.panel),
                "roll_ledger_row_count": len(self.result.roll_ledger),
            },
            diagnostics=dict(self.result.diagnostics),
            artifacts=(
                _non_evidence("stage3_synthetic_fixture", "stage3/contract_rows.csv"),
                _non_evidence(
                    "continuous_daily_panel", "stage3/continuous_daily_panel.csv"
                ),
                _non_evidence("roll_ledger", "stage3/roll_ledger.csv"),
                _non_evidence(
                    "stage3_contract_validation",
                    "reports/stage3_contract_validation.json",
                ),
                _non_evidence("stage3_summary", "reports/stage3_summary.json"),
            ),
        )


@dataclass(frozen=True, slots=True)
class _FeatureComponent:
    fixture: Stage4SyntheticFixture
    result: FeatureBuildResult

    component_id = "stage4_causal_feature_reconstruction"
    stage = PipelineStage.FEATURES
    owner_stage = 4

    def run(self, context: PipelineContext) -> StageOutput:
        del context
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            updates={
                "stage4_fixture_id": self.fixture.metadata["fixture_id"],
                "stage4_feature_row_count": len(self.result.frame),
                "stage4_feature_builder_id": self.result.builder_id,
            },
            diagnostics=dict(self.result.diagnostics),
            artifacts=(
                _non_evidence(
                    "stage4_synthetic_fixture", "stage4/feature_fixture_panel.csv"
                ),
                _non_evidence("stage4_raw_features", "stage4/features_raw.csv"),
                _non_evidence(
                    "stage4_formula_registry", "reports/stage4_formula_registry.json"
                ),
                _non_evidence(
                    "stage4_feature_summary", "reports/stage4_feature_summary.json"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _PreprocessingComponent:
    m3_result: PreprocessingResult
    hpca3_result: PreprocessingResult
    m3_transformer: FittedFoldLocalPreprocessor
    hpca3_transformer: FittedFoldLocalPreprocessor

    component_id = "stage4_fold_local_preprocessing"
    stage = PipelineStage.PREPROCESSING
    owner_stage = 4

    def run(self, context: PipelineContext) -> StageOutput:
        del context
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            updates={
                "stage4_m3_transformer_id": self.m3_transformer.transformer_id,
                "stage4_hpca3_transformer_id": self.hpca3_transformer.transformer_id,
            },
            diagnostics={
                "fit_scope": "training_fold_only",
                "m3_training_row_count": self.m3_transformer.training_row_count,
                "hpca3_training_row_count": self.hpca3_transformer.training_row_count,
                "scientific_evidence": False,
            },
            artifacts=(
                _non_evidence(
                    "stage4_transformed_features", "stage4/features_m3_transformed.csv"
                ),
                _non_evidence(
                    "stage4_transformed_features",
                    "stage4/features_hpca3_transformed.csv",
                ),
                _non_evidence(
                    "stage4_preprocessor_state", "reports/stage4_preprocessing_m3.json"
                ),
                _non_evidence(
                    "stage4_preprocessor_state",
                    "reports/stage4_preprocessing_hpca3.json",
                ),
                _non_evidence(
                    "stage4_preprocessing_summary",
                    "reports/stage4_preprocessing_summary.json",
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _ScaffoldComponent:
    component_id: str
    stage: PipelineStage
    owner_stage: int

    def run(self, context: PipelineContext) -> StageOutput:
        del context
        return StageOutput.skipped_scaffold(
            f"{self.stage.value} is declared only; implementation belongs to "
            f"research-plan Stage {self.owner_stage}."
        )


class _ManifestComponent:
    component_id = "stage2_smoke_manifest"
    stage = PipelineStage.MANIFEST
    owner_stage = 2

    def run(self, context: PipelineContext) -> StageOutput:
        del context
        return StageOutput(
            status=StageStatus.SUCCEEDED,
            artifacts=(
                _non_evidence("seed_report", "reports/seed.json"),
                _non_evidence("capability_registry", "reports/capabilities.json"),
                _non_evidence("stage_ledger", "reports/pipeline_ledger.json"),
                _non_evidence("smoke_summary", "reports/smoke_summary.json"),
                _non_evidence("manifest", "manifest.json"),
            ),
        )


def load_smoke_config(path: str | Path) -> SmokeConfig:
    """Load the dedicated smoke YAML and reject unknown keys."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Smoke config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SmokeConfigError(f"Invalid smoke YAML in {config_path}: {exc}") from exc
    return SmokeConfig.from_mapping(payload)


def combined_execution_hash(
    smoke_config: SmokeConfig,
    preregistration: DailyRegimeConfig,
) -> str:
    """Bind run identity to both software-fixture and research contracts."""

    return _canonical_hash(
        {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "implementation_version": SMOKE_IMPLEMENTATION_VERSION,
            "smoke_config": smoke_config.to_dict(),
            "preregistration_hash": preregistration.config_hash,
        }
    )


def run_smoke(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    overwrite: bool = False,
) -> SmokeRunResult:
    """Run two independent synthetic attempts and require exact payload hashes."""

    source_path = Path(config_path).expanduser().resolve()
    smoke_config = load_smoke_config(source_path)
    preregistration_path = _resolve_preregistration_path(source_path, smoke_config)
    preregistration = load_preregistration_config(preregistration_path)
    if preregistration.project.id != smoke_config.project_id:
        raise SmokeConfigError(
            "Smoke project_id does not match the preregistration project.id."
        )

    execution_hash = combined_execution_hash(smoke_config, preregistration)
    run_id = build_run_id(
        smoke_config.project_id,
        execution_hash,
        smoke_config.seed,
        label=SMOKE_RUN_LABEL,
    )
    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (daily_regime_runtime_paths().artifact_root / "smoke_gate").resolve()
    )
    gate_dir = resolved_output_root / run_id
    if gate_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Smoke run already exists and overwrite is disabled: {gate_dir}"
        )
    gate_dir.mkdir(parents=True, exist_ok=True)

    attempts = tuple(
        _run_attempt(
            attempt=index,
            gate_dir=gate_dir,
            run_id=run_id,
            smoke_config=smoke_config,
            preregistration=preregistration,
            smoke_config_path=source_path,
            preregistration_path=preregistration_path,
            execution_hash=execution_hash,
            overwrite=overwrite,
        )
        for index in range(1, smoke_config.repetitions + 1)
    )
    reference_hashes = dict(attempts[0].computational_hashes)
    reproducibility_passed = all(
        dict(attempt.computational_hashes) == reference_hashes
        for attempt in attempts[1:]
    )
    reproducibility_digest = _canonical_hash(reference_hashes)
    report = {
        "schema_version": "daily_regime_smoke_reproducibility_v1",
        "implementation_version": SMOKE_IMPLEMENTATION_VERSION,
        "mode": SMOKE_MODE.value,
        "run_id": run_id,
        "execution_hash": execution_hash,
        "scientific_evidence": False,
        "paper_eligible": False,
        "comparison_scope": (
            "registered computational payloads only; manifests, timestamps, input "
            "paths, mtimes, and environment-dependent fields are excluded"
        ),
        "attempts": [
            {
                "attempt": attempt.attempt,
                "computational_digest": attempt.computational_digest,
            }
            for attempt in attempts
        ],
        "compared_artifact_hashes": reference_hashes,
        "passed": reproducibility_passed,
    }
    report_path = gate_dir / "reproducibility.json"
    _atomic_write_json(report_path, report, overwrite=overwrite)
    if not reproducibility_passed:
        raise SmokeReproducibilityError(
            f"Synthetic attempts disagreed; inspect {report_path}"
        )
    return SmokeRunResult(
        config_path=source_path,
        preregistration_path=preregistration_path,
        output_root=resolved_output_root,
        run_id=run_id,
        execution_hash=execution_hash,
        attempts=attempts,
        reproducibility_passed=True,
        reproducibility_digest=reproducibility_digest,
        report_path=report_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to strict smoke.yaml.")
    parser.add_argument(
        "--output-root",
        help=(
            "Artifact root. Defaults to "
            "runtime/artifacts/research/daily_latent_regimes/smoke_gate."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace files for the same deterministic run ID.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_smoke(
            arguments.config,
            output_root=arguments.output_root,
            overwrite=arguments.overwrite,
        )
    except (SmokeConfigError, FileNotFoundError) as exc:
        parser.exit(2, f"configuration error: {exc}\n")
    except SmokeReproducibilityError as exc:
        parser.exit(3, f"reproducibility failure: {exc}\n")
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"smoke run failed: {exc}\n")
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


def _run_attempt(
    *,
    attempt: int,
    gate_dir: Path,
    run_id: str,
    smoke_config: SmokeConfig,
    preregistration: DailyRegimeConfig,
    smoke_config_path: Path,
    preregistration_path: Path,
    execution_hash: str,
    overwrite: bool,
) -> SmokeAttemptResult:
    attempt_root = gate_dir / f"attempt_{attempt:02d}"
    seed_report = seed_everything(smoke_config.seed, deterministic_torch=False)
    fixture = CleanSyntheticFixtureFactory().generate(
        smoke_config.fixture.to_fixture_config(seed=smoke_config.seed)
    )
    contract_report = RawDailyBarContract(require_sorted=True).validate(
        fixture.contract_rows
    )
    stage3_fixture = make_stage3_adversarial_fixture()
    stage3_contract_report = RawDailyBarContract(require_sorted=True).validate(
        stage3_fixture.contract_rows
    )
    construction = preregistration.data_construction
    continuous_result = build_continuous_series(
        stage3_fixture.contract_rows,
        config=ContinuousSeriesConfig(
            decision_lag_periods=construction.decision_lag_periods,
            primary_metric=construction.primary_metric,
            secondary_metric=construction.secondary_metric,
            minimum_volume=construction.minimum_volume,
            minimum_open_interest=construction.minimum_open_interest,
            tie_breakers=construction.tie_breakers,
            exclude_limit_locked=construction.exclude_limit_locked,
            exclude_stale_bars=construction.exclude_stale_bars,
            return_convention=construction.return_convention,
            adjustment_convention=construction.adjustment_convention,
            continuous_index_base=construction.continuous_index_base,
            missing_policy=construction.missing_policy,
        ),
    )
    stage4_fixture = make_stage4_synthetic_fixture(seed=smoke_config.seed)
    stage4_features = build_features(
        stage4_fixture.panel,
        request=feature_set_request("H7_M3"),
    )
    stage4_dates = tuple(
        sorted(pd.to_datetime(stage4_features.frame["trading_date"]).unique())
    )
    if len(stage4_dates) < 30:
        raise RuntimeError(
            "Stage 4 smoke fixture is too short for a training/evaluation split."
        )
    stage4_training_end = pd.Timestamp(stage4_dates[-21])
    stage4_training_start = pd.Timestamp(stage4_dates[0])

    m3_training = stage4_features.frame.loc[
        stage4_features.frame["trading_date"].le(stage4_training_end)
    ].dropna(subset=list(M3_RAW_COLUMNS))
    m3_transformer = fit_preprocessor(
        m3_training,
        feature_columns=M3_RAW_COLUMNS,
        config=PreprocessingConfig(clip_quantiles=(0.01, 0.99)),
        context=PreprocessingFitContext(
            fold_id="stage4_smoke_fold",
            training_start=stage4_training_start.date(),
            training_end=stage4_training_end.date(),
            seed=smoke_config.seed,
        ),
    )
    m3_transformed = m3_transformer.transform(stage4_features.frame)

    h7_training = stage4_features.frame.loc[
        stage4_features.frame["trading_date"].le(stage4_training_end)
    ].dropna(subset=list(H7_COLUMNS))
    hpca3_transformer = fit_preprocessor(
        h7_training,
        feature_columns=H7_COLUMNS,
        config=PreprocessingConfig(
            clip_quantiles=(0.01, 0.99),
            pca_components=3,
        ),
        context=PreprocessingFitContext(
            fold_id="stage4_smoke_fold",
            training_start=stage4_training_start.date(),
            training_end=stage4_training_end.date(),
            seed=smoke_config.seed,
        ),
    )
    hpca3_transformed = hpca3_transformer.transform(stage4_features.frame)
    pipeline = execute_pipeline(
        PipelineRequest(
            run_id=run_id,
            mode=SMOKE_MODE,
            seed=smoke_config.seed,
            artifact_root=attempt_root,
        ),
        _smoke_components(
            fixture=fixture,
            stage3_fixture=stage3_fixture,
            continuous_result=continuous_result,
            stage4_fixture=stage4_fixture,
            stage4_features=stage4_features,
            m3_transformed=m3_transformed,
            hpca3_transformed=hpca3_transformed,
            m3_transformer=m3_transformer,
            hpca3_transformer=hpca3_transformer,
            execution_hash=execution_hash,
            smoke_config_hash=smoke_config.config_hash,
            preregistration_hash=preregistration.config_hash,
        ),
    )
    if any(artifact.scientific_evidence for artifact in pipeline.artifacts):
        raise RuntimeError("Synthetic smoke pipeline emitted scientific evidence.")

    writer = RunArtifactWriter(
        attempt_root,
        run_id=run_id,
        mode=SMOKE_MODE.value,
        workspace_root=attempt_root,
        overwrite=overwrite,
    )
    writer.write_json(
        "config/smoke.resolved.json",
        smoke_config.to_dict(),
        kind="resolved_config",
    )
    writer.write_json(
        "config/preregistration.resolved.json",
        preregistration.to_dict(),
        kind="resolved_config",
    )
    writer.write_frame(
        "fixtures/contract_rows.csv",
        fixture.contract_rows,
        kind="synthetic_fixture",
    )
    writer.write_frame(
        "fixtures/dominant_mapping.csv",
        fixture.dominant_mapping,
        kind="synthetic_fixture",
    )
    writer.write_frame(
        "fixtures/smoke_panel.csv",
        fixture.smoke_panel,
        kind="synthetic_fixture",
    )
    writer.write_frame(
        "quarantine/synthetic_truth.csv",
        fixture.truth,
        kind="quarantined_truth",
    )
    writer.write_json(
        "reports/contract_validation.json",
        asdict(contract_report),
        kind="contract_validation",
    )
    writer.write_frame(
        "stage3/contract_rows.csv",
        stage3_fixture.contract_rows,
        kind="stage3_synthetic_fixture",
    )
    writer.write_frame(
        "stage3/continuous_daily_panel.csv",
        continuous_result.panel,
        kind="continuous_daily_panel",
    )
    writer.write_frame(
        "stage3/roll_ledger.csv",
        continuous_result.roll_ledger,
        kind="roll_ledger",
    )
    writer.write_json(
        "reports/stage3_contract_validation.json",
        asdict(stage3_contract_report),
        kind="stage3_contract_validation",
    )
    writer.write_json(
        "reports/stage3_summary.json",
        {
            "schema_version": "daily_regime_stage3_summary_v1",
            "fixture_id": stage3_fixture.metadata["fixture_id"],
            "builder_id": continuous_result.builder_id,
            "data_construction": asdict(construction),
            "diagnostics": dict(continuous_result.diagnostics),
            "causal_selection": True,
            "same_contract_returns": True,
            "non_revising_chained_index": True,
            "missing_policy": "flag_no_backfill",
            "scientific_evidence": False,
            "paper_eligible": False,
        },
        kind="stage3_summary",
    )
    writer.write_frame(
        "stage4/feature_fixture_panel.csv",
        stage4_fixture.panel,
        kind="stage4_synthetic_fixture",
    )
    writer.write_frame(
        "stage4/features_raw.csv",
        stage4_features.frame,
        kind="stage4_raw_features",
    )
    writer.write_json(
        "reports/stage4_formula_registry.json",
        feature_formula_registry(),
        kind="stage4_formula_registry",
    )
    writer.write_json(
        "reports/stage4_feature_summary.json",
        {
            "schema_version": "daily_regime_stage4_feature_summary_v1",
            "fixture_id": stage4_fixture.metadata["fixture_id"],
            "builder_id": stage4_features.builder_id,
            "feature_set_id": stage4_features.feature_set_id,
            "feature_columns": list(stage4_features.feature_columns),
            "diagnostics": dict(stage4_features.diagnostics),
            "causal_information_dates": True,
            "sequence_safe_rolling_windows": True,
            "scientific_evidence": False,
            "paper_eligible": False,
        },
        kind="stage4_feature_summary",
    )
    writer.write_frame(
        "stage4/features_m3_transformed.csv",
        m3_transformed.frame,
        kind="stage4_transformed_features",
    )
    writer.write_frame(
        "stage4/features_hpca3_transformed.csv",
        hpca3_transformed.frame,
        kind="stage4_transformed_features",
    )
    writer.write_json(
        "reports/stage4_preprocessing_m3.json",
        m3_transformer.state_dict(),
        kind="stage4_preprocessor_state",
    )
    writer.write_json(
        "reports/stage4_preprocessing_hpca3.json",
        hpca3_transformer.state_dict(),
        kind="stage4_preprocessor_state",
    )
    writer.write_json(
        "reports/stage4_preprocessing_summary.json",
        {
            "schema_version": "daily_regime_stage4_preprocessing_summary_v1",
            "fit_scope": "training_fold_only",
            "m3_transformer_id": m3_transformer.transformer_id,
            "hpca3_transformer_id": hpca3_transformer.transformer_id,
            "m3_diagnostics": dict(m3_transformed.diagnostics),
            "hpca3_diagnostics": dict(hpca3_transformed.diagnostics),
            "scientific_evidence": False,
            "paper_eligible": False,
        },
        kind="stage4_preprocessing_summary",
    )
    writer.write_json(
        "reports/seed.json",
        {
            "seed": seed_report.seed,
            "python_seeded": seed_report.python_seeded,
            "numpy_seeded": seed_report.numpy_seeded,
            "deterministic_torch_requested": seed_report.deterministic_torch_requested,
            "python_hash_seed": seed_report.python_hash_seed,
        },
        kind="seed_report",
    )
    registry = default_capability_registry()
    writer.write_json(
        "reports/capabilities.json",
        registry.to_dict(),
        kind="capability_registry",
    )
    writer.write_json(
        "reports/pipeline_ledger.json",
        [_stage_record_payload(record) for record in pipeline.records],
        kind="stage_ledger",
    )
    writer.write_json(
        "reports/smoke_summary.json",
        {
            "schema_version": "daily_regime_smoke_summary_v1",
            "implementation_version": SMOKE_IMPLEMENTATION_VERSION,
            "mode": SMOKE_MODE.value,
            "run_id": run_id,
            "execution_hash": execution_hash,
            "fixture_id": fixture.fixture_id,
            "row_count": len(fixture.contract_rows),
            "product_count": len(smoke_config.fixture.products),
            "scientific_evidence": False,
            "contains_market_data": False,
            "hidden_truth_quarantined": True,
            "stage3_point_in_time_construction_verified": True,
            "stage3_fixture_id": stage3_fixture.metadata["fixture_id"],
            "stage4_causal_features_verified": True,
            "stage4_fold_local_preprocessing_verified": True,
            "stage4_fixture_id": stage4_fixture.metadata["fixture_id"],
            "permissions": {
                "market_data": False,
                "network": False,
                "holdout": False,
                "manuscript_writes": False,
            },
            "scaffolded_stages": [stage.value for stage in pipeline.scaffolded_stages],
        },
        kind="smoke_summary",
    )
    computational_hashes = _registered_hashes(writer)
    computational_digest = _canonical_hash(computational_hashes)
    manifest = writer.finalize(
        project_id=smoke_config.project_id,
        config_hash=execution_hash,
        seed=smoke_config.seed,
        input_paths=(smoke_config_path, preregistration_path),
        capabilities=registry.capabilities,
        metadata={
            "attempt": attempt,
            "fixture_id": fixture.fixture_id,
            "stage3_fixture_id": stage3_fixture.metadata["fixture_id"],
            "stage4_fixture_id": stage4_fixture.metadata["fixture_id"],
            "stage4_m3_transformer_id": m3_transformer.transformer_id,
            "stage4_hpca3_transformer_id": hpca3_transformer.transformer_id,
            "scientific_evidence": False,
            "paper_eligible": False,
            "computational_digest": computational_digest,
            "manifest_excluded_from_reproducibility_digest": True,
        },
    )
    return SmokeAttemptResult(
        attempt=attempt,
        run_dir=writer.run_dir,
        manifest_path=writer.run_dir / "manifest.json",
        manifest=manifest,
        computational_hashes=computational_hashes,
        computational_digest=computational_digest,
        pipeline=pipeline,
    )


def _smoke_components(
    *,
    fixture: SyntheticFixture,
    stage3_fixture: Stage3AdversarialFixture,
    continuous_result: ContinuousSeriesResult,
    stage4_fixture: Stage4SyntheticFixture,
    stage4_features: FeatureBuildResult,
    m3_transformed: PreprocessingResult,
    hpca3_transformed: PreprocessingResult,
    m3_transformer: FittedFoldLocalPreprocessor,
    hpca3_transformer: FittedFoldLocalPreprocessor,
    execution_hash: str,
    smoke_config_hash: str,
    preregistration_hash: str,
) -> tuple[Any, ...]:
    owner_stages = {
        PipelineStage.FOLDS: 5,
        PipelineStage.TARGETS: 5,
        PipelineStage.BASELINES: 6,
        PipelineStage.HMM_MODELS: 7,
        PipelineStage.FILTERING: 7,
        PipelineStage.ALIGNMENT: 7,
        PipelineStage.VQVAE: 9,
        PipelineStage.DIAGNOSTICS: 8,
        PipelineStage.EVALUATION: 8,
        PipelineStage.RISK_THROTTLE: 10,
    }
    scaffolds = {
        stage: _ScaffoldComponent(
            component_id=f"declared_{stage.value}",
            stage=stage,
            owner_stage=owner_stage,
        )
        for stage, owner_stage in owner_stages.items()
    }
    return (
        _ConfigComponent(
            execution_hash=execution_hash,
            smoke_config_hash=smoke_config_hash,
            preregistration_hash=preregistration_hash,
        ),
        _SyntheticInputComponent(fixture=fixture),
        _ContinuousSeriesComponent(
            fixture=stage3_fixture,
            result=continuous_result,
        ),
        _FeatureComponent(fixture=stage4_fixture, result=stage4_features),
        scaffolds[PipelineStage.FOLDS],
        scaffolds[PipelineStage.TARGETS],
        _PreprocessingComponent(
            m3_result=m3_transformed,
            hpca3_result=hpca3_transformed,
            m3_transformer=m3_transformer,
            hpca3_transformer=hpca3_transformer,
        ),
        *(
            scaffolds[stage]
            for stage in owner_stages
            if stage not in {PipelineStage.FOLDS, PipelineStage.TARGETS}
        ),
        _ManifestComponent(),
    )


def _resolve_preregistration_path(
    smoke_path: Path,
    config: SmokeConfig,
) -> Path:
    candidate = (smoke_path.parent / config.preregistration_config).resolve()
    config_directory = smoke_path.parent.resolve()
    try:
        candidate.relative_to(config_directory)
    except ValueError as exc:
        raise SmokeConfigError(
            "preregistration_config must remain inside the smoke config directory."
        ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Preregistration config does not exist: {candidate}")
    return candidate


def _registered_hashes(writer: RunArtifactWriter) -> dict[str, str]:
    prefix = PurePosixPath(writer.mode) / writer.run_id
    hashes: dict[str, str] = {}
    for record in writer.records:
        path = PurePosixPath(record.path)
        try:
            relative = path.relative_to(prefix).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"Artifact path is not portable/run-relative: {record.path}"
            ) from exc
        hashes[relative] = record.sha256
    return dict(sorted(hashes.items()))


def _stage_record_payload(record: StageRecord) -> dict[str, Any]:
    return {
        "stage": record.stage.value,
        "component_id": record.component_id,
        "owner_stage": record.owner_stage,
        "status": record.status.value,
        "diagnostic_keys": list(record.diagnostic_keys),
        "artifact_paths": list(record.artifact_paths),
        "note": record.note,
    }


def _non_evidence(kind: str, relative_path: str) -> ArtifactReference:
    return ArtifactReference(
        kind=kind,
        relative_path=relative_path,
        scientific_evidence=False,
    )


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, Path)):
        return value.isoformat() if isinstance(value, date) else value.as_posix()
    if isinstance(value, Enum):
        return value.value
    return value


def _atomic_write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Artifact exists and overwrite is disabled: {path}")
    content = (
        json.dumps(
            _jsonable(payload),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _strict_mapping(
    value: Any,
    allowed: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeConfigError(f"{context} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise SmokeConfigError(f"{context} keys must be strings.")
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise SmokeConfigError(f"Unknown keys in {context}: {unknown}")
    return dict(value)


def _required(data: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise SmokeConfigError(f"Missing required key {context}.{key}.")
    return data[key]


def _nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeConfigError(f"{context} must be a non-empty string.")
    return value.strip()


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SmokeConfigError(f"{context} must be a positive integer.")
    return value


__all__ = [
    "SMOKE_ATTEMPTS",
    "SMOKE_IMPLEMENTATION_VERSION",
    "SMOKE_MODE",
    "SMOKE_SCHEMA_VERSION",
    "SmokeAttemptResult",
    "SmokeConfig",
    "SmokeConfigError",
    "SmokeFixtureSpec",
    "SmokeReproducibilityError",
    "SmokeRunResult",
    "build_arg_parser",
    "combined_execution_hash",
    "load_smoke_config",
    "main",
    "run_smoke",
]
