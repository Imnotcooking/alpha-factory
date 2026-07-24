"""Purpose-separated optimization architecture for research and allocation."""

from oqp.optimization.constraints import (
    ConstraintEvaluation,
    evaluate_constraints,
    hard_constraints_satisfied,
    optuna_constraint_values,
)
from oqp.optimization.contracts import (
    ConstraintSpec,
    FrozenResearchInputs,
    ObjectiveSpec,
    OptimizationCandidate,
    OptimizationDirection,
    OptimizationEvaluationContext,
    OptimizationPurpose,
    OptimizationStudyResult,
    OptimizationStudySpec,
    SearchBudget,
    TrialEvaluation,
    stable_optimization_hash,
)
from oqp.optimization.parameter_spaces import (
    ComponentParameterSchema,
    ComponentParameterSpec,
    build_component_parameter_schema,
    resolve_component_parameter_schema,
    resolve_component_parameter_values,
    suggest_component_parameters,
)
from oqp.optimization.study_runner import (
    CandidateEvaluator,
    OptimizationComparisonResult,
    OptimizationStudyRunner,
)
from oqp.optimization.study_store import OptimizationStudyStore
from oqp.optimization.research_inputs import require_dataset_fingerprint
from oqp.optimization.method_registry import (
    OptimizationMethodProfile,
    OptimizationMethodRegistry,
    OptimizationPurposeProfile,
)


__all__ = [
    "CandidateEvaluator",
    "ComponentParameterSchema",
    "ComponentParameterSpec",
    "ConstraintEvaluation",
    "ConstraintSpec",
    "FrozenResearchInputs",
    "ObjectiveSpec",
    "OptimizationCandidate",
    "OptimizationComparisonResult",
    "OptimizationDirection",
    "OptimizationEvaluationContext",
    "OptimizationPurpose",
    "OptimizationStudyResult",
    "OptimizationStudyRunner",
    "OptimizationStudySpec",
    "OptimizationStudyStore",
    "OptimizationMethodProfile",
    "OptimizationMethodRegistry",
    "OptimizationPurposeProfile",
    "SearchBudget",
    "TrialEvaluation",
    "build_component_parameter_schema",
    "evaluate_constraints",
    "hard_constraints_satisfied",
    "optuna_constraint_values",
    "resolve_component_parameter_schema",
    "resolve_component_parameter_values",
    "require_dataset_fingerprint",
    "stable_optimization_hash",
    "suggest_component_parameters",
]
