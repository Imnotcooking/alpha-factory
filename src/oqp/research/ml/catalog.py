"""Dependency-light catalog of research model implementations and designs.

This is an inventory, not an artifact registry.  A catalog entry means the
implementation exists; it does not claim that a model has been fitted,
validated, or promoted.  Fitted instances continue to live in the model
artifact registry and completed experiments in their corresponding ledgers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


MODEL_CATALOG_VERSION = "oqp_research_model_catalog_v4"


class ModelCategory(str, Enum):
    """Economic/research role of an ML implementation."""

    SUPERVISED_PREDICTOR = "supervised_predictor"
    REGIME_ESTIMATOR = "regime_estimator"
    LATENT_REPRESENTATION = "latent_representation"
    STATE_SPACE_ESTIMATOR = "state_space_estimator"
    STUDY_CONTROL = "study_control"


class LearningParadigm(str, Enum):
    """What information supplies the model's learning signal."""

    SUPERVISED = "supervised"
    UNSUPERVISED_SEQUENTIAL = "unsupervised_sequential"
    UNSUPERVISED_IID = "unsupervised_iid"
    SELF_SUPERVISED = "self_supervised"
    ONLINE_SEQUENTIAL = "online_sequential"


class ImplementationScope(str, Enum):
    """How broadly the implementation is intended to be reused."""

    SUPERVISED_ADAPTER = "supervised_adapter"
    SHARED_CORE = "shared_core"
    STUDY_ONLY = "study_only"


@dataclass(frozen=True, slots=True)
class ResearchModelDescriptor:
    """One implemented model family, without implying a fitted artifact."""

    model_key: str
    display_name: str
    category: ModelCategory
    learning_paradigm: LearningParadigm
    task: str
    input_geometry: str
    output_contract: str
    requires_target: bool
    implementation_path: str
    artifact_contract: str
    scope: ImplementationScope

    def __post_init__(self) -> None:
        _identifier(self.model_key, "model_key")
        _identifier(self.display_name, "display_name")
        _identifier(self.task, "task")
        _identifier(self.input_geometry, "input_geometry")
        _identifier(self.output_contract, "output_contract")
        _identifier(self.artifact_contract, "artifact_contract")
        if not isinstance(self.category, ModelCategory):
            raise TypeError("category must be a ModelCategory")
        if not isinstance(self.learning_paradigm, LearningParadigm):
            raise TypeError("learning_paradigm must be a LearningParadigm")
        if not isinstance(self.scope, ImplementationScope):
            raise TypeError("scope must be an ImplementationScope")
        if type(self.requires_target) is not bool:
            raise TypeError("requires_target must be a bool")
        if (
            type(self.implementation_path) is not str
            or self.implementation_path.count(":") != 1
            or any(not part for part in self.implementation_path.split(":"))
        ):
            raise ValueError(
                "implementation_path must use the 'module:object' convention"
            )


@dataclass(frozen=True, slots=True)
class ResearchExperimentDesign:
    """Registered comparison design, distinct from an executed experiment."""

    experiment_key: str
    display_name: str
    design_status: str
    empirical_status: str
    primary_metric: str
    primary_state_count: int
    model_keys: tuple[str, ...]
    comparisons: tuple[str, ...]
    implementation_path: str

    def __post_init__(self) -> None:
        for name in (
            "experiment_key",
            "display_name",
            "design_status",
            "empirical_status",
            "primary_metric",
        ):
            _identifier(getattr(self, name), name)
        if type(self.primary_state_count) is not int or self.primary_state_count < 2:
            raise ValueError("primary_state_count must be an integer of at least two")
        if (
            type(self.model_keys) is not tuple
            or len(self.model_keys) < 2
            or len(set(self.model_keys)) != len(self.model_keys)
            or any(type(value) is not str or not value for value in self.model_keys)
        ):
            raise ValueError("model_keys must contain unique model identifiers")
        if (
            type(self.comparisons) is not tuple
            or not self.comparisons
            or any(type(value) is not str or not value for value in self.comparisons)
        ):
            raise ValueError("comparisons must contain explicit comparison labels")
        if (
            type(self.implementation_path) is not str
            or self.implementation_path.count(":") != 1
            or any(not part for part in self.implementation_path.split(":"))
        ):
            raise ValueError(
                "implementation_path must use the 'module:object' convention"
            )


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


_MODELS = (
    ResearchModelDescriptor(
        model_key="lightgbm_regressor",
        display_name="LightGBM",
        category=ModelCategory.SUPERVISED_PREDICTOR,
        learning_paradigm=LearningParadigm.SUPERVISED,
        task="Cross-sectional target prediction",
        input_geometry="Labelled feature rows with time-aware folds",
        output_contract="Continuous predictions and feature importance",
        requires_target=True,
        implementation_path=(
            "oqp.research.ml.tree_based.lightgbm:LightGBMRegressorTrainer"
        ),
        artifact_contract="Native booster plus governed predictions and importance",
        scope=ImplementationScope.SUPERVISED_ADAPTER,
    ),
    ResearchModelDescriptor(
        model_key="xgboost_regressor",
        display_name="XGBoost",
        category=ModelCategory.SUPERVISED_PREDICTOR,
        learning_paradigm=LearningParadigm.SUPERVISED,
        task="Cross-sectional target prediction",
        input_geometry="Labelled feature rows with time-aware folds",
        output_contract="Continuous predictions and feature importance",
        requires_target=True,
        implementation_path=(
            "oqp.research.ml.tree_based.xgboost:XGBoostRegressorTrainer"
        ),
        artifact_contract="Native booster plus governed predictions and importance",
        scope=ImplementationScope.SUPERVISED_ADAPTER,
    ),
    ResearchModelDescriptor(
        model_key="gaussian_hmm",
        display_name="Gaussian HMM",
        category=ModelCategory.REGIME_ESTIMATOR,
        learning_paradigm=LearningParadigm.UNSUPERVISED_SEQUENTIAL,
        task="Causal latent-regime inference",
        input_geometry="Ordered entity sequences with explicit reset boundaries",
        output_contract="Filtered state probabilities and predictive density",
        requires_target=False,
        implementation_path="oqp.research.ml.regimes.gaussian_hmm:GaussianHMM",
        artifact_contract="Immutable FittedDiagonalHMM JSON",
        scope=ImplementationScope.SHARED_CORE,
    ),
    ResearchModelDescriptor(
        model_key="gmm_hmm",
        display_name="GMM-HMM",
        category=ModelCategory.REGIME_ESTIMATOR,
        learning_paradigm=LearningParadigm.UNSUPERVISED_SEQUENTIAL,
        task="Causal latent-regime inference with mixture emissions",
        input_geometry="Ordered entity sequences with explicit reset boundaries",
        output_contract="Filtered state probabilities and predictive density",
        requires_target=False,
        implementation_path="oqp.research.ml.regimes.gmm_hmm:GMMHMM",
        artifact_contract="Immutable FittedDiagonalHMM JSON",
        scope=ImplementationScope.SHARED_CORE,
    ),
    ResearchModelDescriptor(
        model_key="student_t_hmm",
        display_name="Student-t HMM",
        category=ModelCategory.REGIME_ESTIMATOR,
        learning_paradigm=LearningParadigm.UNSUPERVISED_SEQUENTIAL,
        task="Causal heavy-tailed latent-regime inference",
        input_geometry="Ordered entity sequences with explicit reset boundaries",
        output_contract="Filtered state probabilities and predictive density",
        requires_target=False,
        implementation_path="oqp.research.ml.regimes.student_t_hmm:StudentTHMM",
        artifact_contract="Immutable FittedDiagonalHMM JSON",
        scope=ImplementationScope.SHARED_CORE,
    ),
    ResearchModelDescriptor(
        model_key="vqvae",
        display_name="MLP VQ-VAE core",
        category=ModelCategory.LATENT_REPRESENTATION,
        learning_paradigm=LearningParadigm.SELF_SUPERVISED,
        task="Discrete nonlinear representation learning",
        input_geometry="Prepared finite feature matrix; windowing is adapter-owned",
        output_contract="Discrete codes, latent vectors, and reconstructions",
        requires_target=False,
        implementation_path="oqp.research.ml.latent.vqvae.model:VQVAETrainer",
        artifact_contract="Immutable tensor bundle with manifest hashes",
        scope=ImplementationScope.SHARED_CORE,
    ),
    ResearchModelDescriptor(
        model_key="dual_kalman_regression",
        display_name="Dual Kalman Regression",
        category=ModelCategory.STATE_SPACE_ESTIMATOR,
        learning_paradigm=LearningParadigm.ONLINE_SEQUENTIAL,
        task="Adaptive linear relationship estimation",
        input_geometry="Ordered observations with explicit entity reset boundaries",
        output_contract="Time-varying coefficients, innovations, and state uncertainty",
        requires_target=False,
        implementation_path=(
            "oqp.research.ml.state_space.dual_kalman_regression:"
            "DualKalmanRegression"
        ),
        artifact_contract="Workflow-owned Parquet features and JSON metadata",
        scope=ImplementationScope.SHARED_CORE,
    ),
    ResearchModelDescriptor(
        model_key="iid_gaussian_mixture_control",
        display_name="IID Gaussian Mixture",
        category=ModelCategory.STUDY_CONTROL,
        learning_paradigm=LearningParadigm.UNSUPERVISED_IID,
        task="Non-Markov latent clustering control",
        input_geometry="Independent prepared feature rows",
        output_contract="Component probabilities and row log density",
        requires_target=False,
        implementation_path=(
            "departments.research.workflows.hmm_complexity.estimators:"
            "IIDGaussianMixtureTrainer"
        ),
        artifact_contract="Authenticated HMM-complexity workflow-run record",
        scope=ImplementationScope.STUDY_ONLY,
    ),
)


_EXPERIMENTS = (
    ResearchExperimentDesign(
        experiment_key="daily_cn_futures_hmm_complexity_v1",
        display_name="Daily CN Futures HMM Complexity",
        design_status="workflow_ready_real_adapter_verified",
        empirical_status="fresh_complete_m2_panel_not_available",
        primary_metric="Causal one-step log predictive density",
        primary_state_count=2,
        model_keys=(
            "iid_gaussian_mixture_control",
            "gaussian_hmm",
            "gmm_hmm",
            "student_t_hmm",
        ),
        comparisons=(
            "Markov dependence: Gaussian HMM minus IID mixture",
            "Mixture emissions: GMM-HMM minus Gaussian HMM",
            "Heavy tails: Student-t HMM minus Gaussian HMM",
        ),
        implementation_path=(
            "departments.research.workflows.hmm_complexity.workflow:"
            "run_hmm_complexity_study"
        ),
    ),
)


_MODEL_KEY_ALIASES = {
    # v1 catalog key retained for callers; the reusable core itself does not
    # own temporal-window construction.
    "temporal_vqvae": "vqvae",
}


def research_model_catalog() -> tuple[ResearchModelDescriptor, ...]:
    """Return the immutable implementation catalog in presentation order."""

    return _MODELS


def research_experiment_catalog() -> tuple[ResearchExperimentDesign, ...]:
    """Return registered designs without implying they have been executed."""

    return _EXPERIMENTS


def model_descriptor(model_key: str) -> ResearchModelDescriptor:
    """Look up one implementation by stable key."""

    model_key = _MODEL_KEY_ALIASES.get(model_key, model_key)
    for item in _MODELS:
        if item.model_key == model_key:
            return item
    raise KeyError(f"unknown research model: {model_key}")


def _validate_catalog() -> None:
    model_keys = tuple(item.model_key for item in _MODELS)
    if len(model_keys) != len(set(model_keys)):
        raise RuntimeError("research model catalog contains duplicate keys")
    known = set(model_keys)
    experiment_keys = tuple(item.experiment_key for item in _EXPERIMENTS)
    if len(experiment_keys) != len(set(experiment_keys)):
        raise RuntimeError("research experiment catalog contains duplicate keys")
    for experiment in _EXPERIMENTS:
        missing = set(experiment.model_keys).difference(known)
        if missing:
            raise RuntimeError(
                f"experiment {experiment.experiment_key!r} references "
                f"unknown models: {sorted(missing)}"
            )


_validate_catalog()


__all__ = [
    "ImplementationScope",
    "LearningParadigm",
    "MODEL_CATALOG_VERSION",
    "ModelCategory",
    "ResearchExperimentDesign",
    "ResearchModelDescriptor",
    "model_descriptor",
    "research_experiment_catalog",
    "research_model_catalog",
]
