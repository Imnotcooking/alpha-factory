"""Compatibility import for the canonical ML model catalog.

New code should import :mod:`oqp.research.ml.catalog`.  This module remains so
existing dashboards and notebooks keep resolving the identical objects.
"""

from oqp.research.ml.catalog import (
    ImplementationScope,
    LearningParadigm,
    MODEL_CATALOG_VERSION,
    ModelCategory,
    ResearchExperimentDesign,
    ResearchModelDescriptor,
    model_descriptor,
    research_experiment_catalog,
    research_model_catalog,
)


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
