"""Statistical evidence and multiple-testing helpers for research workflows."""

from oqp.research.multiple_testing import (
    MultipleTestingAdjustment,
    benjamini_hochberg_q_values,
    bonferroni_p_value,
    holm_bonferroni_adjust,
    significance_label,
    stable_trial_hash,
)
from oqp.research.statistical_tests import (
    AlphaStatisticalTester,
    StatisticalEvidence,
    sharpe_p_value_from_returns,
)

__all__ = [
    "AlphaStatisticalTester",
    "MultipleTestingAdjustment",
    "StatisticalEvidence",
    "benjamini_hochberg_q_values",
    "bonferroni_p_value",
    "holm_bonferroni_adjust",
    "sharpe_p_value_from_returns",
    "significance_label",
    "stable_trial_hash",
]
