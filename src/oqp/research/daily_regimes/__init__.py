"""Failure-aware daily latent-regime research architecture.

The package is intentionally conservative: Stage 2 provides typed contracts and
orchestration, while Stages 3 and 4 add point-in-time series and causal feature
construction.  Synthetic verification never makes a capability eligible for
paper evidence; later validation and freeze gates remain mandatory.
"""

from oqp.research.daily_regimes.artifacts import (
    ArtifactRecord,
    RunArtifactWriter,
    RunManifest,
    WrittenManifest,
    build_run_id,
)
from oqp.research.daily_regimes.capabilities import (
    CapabilityRegistry,
    CapabilityStage,
    ResearchCapability,
    default_capability_registry,
)
from oqp.research.daily_regimes.config import (
    DataConstructionConfig,
    DailyRegimeConfig,
    DailyRegimeConfigError,
    load_daily_regime_config,
    load_preregistration_config,
)
from oqp.research.daily_regimes.continuous_series import (
    ContinuousSeriesConfig,
    ContinuousSeriesResult,
    LaggedLiquidityContinuousSeriesBuilder,
    build_continuous_series,
    validate_continuous_series_result,
)
from oqp.research.daily_regimes.contracts import (
    ContractViolation,
    FeatureFrameContract,
    FoldSpec,
    ProbabilitySemantics,
    ProbabilityUse,
    ProspectiveProbabilityError,
    RawDailyBarContract,
    StateProbabilityContract,
)
from oqp.research.daily_regimes.filtering import (
    StateProbabilityBatch,
    StateProbabilityRecord,
)
from oqp.research.daily_regimes.features import (
    DailyRegimeFeatureBuilder,
    FeatureBuildResult,
    FeatureSetRequest,
    FeatureSpec,
    build_features,
    feature_formula_registry,
    feature_set_request,
    validate_feature_result,
    wavelet_hurst,
)
from oqp.research.daily_regimes.pipeline import (
    PIPELINE_STAGE_ORDER,
    ArtifactReference,
    PipelineExecutionError,
    PipelineGuards,
    PipelineMode,
    PipelinePermission,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    StageOutput,
    StageRecord,
    StageStatus,
    execute_pipeline,
)
from oqp.research.daily_regimes.runtime_paths import (
    DailyRegimeRuntimePaths,
    daily_regime_runtime_paths,
)
from oqp.research.daily_regimes.preprocessing import (
    FittedFoldLocalPreprocessor,
    FoldLocalPreprocessor,
    PreprocessingConfig,
    PreprocessingFitContext,
    PreprocessingResult,
    fit_preprocessor,
    validate_preprocessing_result,
)
from oqp.research.daily_regimes.seeding import (
    SeedReport,
    derive_seed,
    restart_seeds,
    seed_everything,
)
from oqp.research.daily_regimes.smoke import (
    SMOKE_ATTEMPTS,
    SMOKE_IMPLEMENTATION_VERSION,
    SMOKE_SCHEMA_VERSION,
    SmokeAttemptResult,
    SmokeConfig,
    SmokeConfigError,
    SmokeFixtureSpec,
    SmokeReproducibilityError,
    SmokeRunResult,
    combined_execution_hash,
    load_smoke_config,
    run_smoke,
)
from oqp.research.daily_regimes.synthetic import (
    SyntheticFixture,
    SyntheticFixtureConfig,
    make_clean_synthetic_fixture,
)
from oqp.research.daily_regimes.stage3_fixtures import (
    Stage3AdversarialFixture,
    make_invalid_stage3_frames,
    make_stage3_adversarial_fixture,
)
from oqp.research.daily_regimes.stage4_fixtures import (
    Stage4SyntheticFixture,
    make_stage4_synthetic_fixture,
)


PACKAGE_STAGE = 2


__all__ = [
    "ArtifactRecord",
    "ArtifactReference",
    "CapabilityRegistry",
    "CapabilityStage",
    "ContractViolation",
    "ContinuousSeriesConfig",
    "ContinuousSeriesResult",
    "DailyRegimeConfig",
    "DailyRegimeConfigError",
    "DailyRegimeRuntimePaths",
    "DataConstructionConfig",
    "DailyRegimeFeatureBuilder",
    "FeatureBuildResult",
    "FeatureSetRequest",
    "FeatureSpec",
    "FeatureFrameContract",
    "FoldSpec",
    "FoldLocalPreprocessor",
    "FittedFoldLocalPreprocessor",
    "LaggedLiquidityContinuousSeriesBuilder",
    "PACKAGE_STAGE",
    "PIPELINE_STAGE_ORDER",
    "PipelineExecutionError",
    "PipelineGuards",
    "PipelineMode",
    "PipelinePermission",
    "PipelineRequest",
    "PipelineResult",
    "PipelineStage",
    "PreprocessingConfig",
    "PreprocessingFitContext",
    "PreprocessingResult",
    "ProbabilitySemantics",
    "ProbabilityUse",
    "ProspectiveProbabilityError",
    "RawDailyBarContract",
    "ResearchCapability",
    "RunArtifactWriter",
    "RunManifest",
    "SeedReport",
    "SMOKE_ATTEMPTS",
    "SMOKE_IMPLEMENTATION_VERSION",
    "SMOKE_SCHEMA_VERSION",
    "SmokeAttemptResult",
    "SmokeConfig",
    "SmokeConfigError",
    "SmokeFixtureSpec",
    "SmokeReproducibilityError",
    "SmokeRunResult",
    "StageOutput",
    "StageRecord",
    "StageStatus",
    "StateProbabilityContract",
    "StateProbabilityBatch",
    "StateProbabilityRecord",
    "Stage3AdversarialFixture",
    "Stage4SyntheticFixture",
    "SyntheticFixture",
    "SyntheticFixtureConfig",
    "WrittenManifest",
    "build_run_id",
    "build_continuous_series",
    "build_features",
    "combined_execution_hash",
    "daily_regime_runtime_paths",
    "default_capability_registry",
    "derive_seed",
    "execute_pipeline",
    "feature_formula_registry",
    "feature_set_request",
    "fit_preprocessor",
    "load_daily_regime_config",
    "load_preregistration_config",
    "load_smoke_config",
    "make_clean_synthetic_fixture",
    "make_invalid_stage3_frames",
    "make_stage3_adversarial_fixture",
    "make_stage4_synthetic_fixture",
    "restart_seeds",
    "run_smoke",
    "seed_everything",
    "validate_continuous_series_result",
    "validate_feature_result",
    "validate_preprocessing_result",
    "wavelet_hurst",
]
