"""Pure evaluation of approved risk-limit definitions."""

from __future__ import annotations

import math
from collections.abc import Mapping

from oqp.risk.contracts import (
    LimitEvaluationState,
    RiskCalculationStatus,
    RiskEnforcementMode,
    RiskLimitCatalog,
    RiskLimitDefinition,
    RiskLimitDirection,
    RiskLimitEvaluation,
)
from oqp.risk.limit_catalog import load_risk_limit_catalog


def evaluate_risk_limits(
    values: Mapping[str, float | int | None],
    *,
    catalog: RiskLimitCatalog | None = None,
) -> tuple[RiskLimitEvaluation, ...]:
    """Evaluate values without persisting state or performing side effects."""

    active_catalog = catalog or load_risk_limit_catalog()
    return tuple(
        evaluate_risk_limit(control, values.get(control.control_id))
        for control in active_catalog.controls
    )


def evaluate_risk_limit(
    control: RiskLimitDefinition,
    value: float | int | None,
) -> RiskLimitEvaluation:
    """Evaluate one value against one validated control definition."""

    common = {
        "control_id": control.control_id,
        "enforcement_mode": control.enforcement_mode,
        "warning_threshold": control.warning_threshold,
        "hard_threshold": control.hard_threshold,
    }
    if control.calculation_status is RiskCalculationStatus.PLANNED:
        return RiskLimitEvaluation(
            **common,
            state=LimitEvaluationState.PLANNED,
            value=None,
            message="Calculation is planned and cannot be evaluated.",
        )

    numeric_value = _finite_value(value)
    if numeric_value is None:
        return RiskLimitEvaluation(
            **common,
            state=LimitEvaluationState.UNAVAILABLE,
            value=None,
            message="Metric is missing or non-finite.",
        )

    if control.enforcement_mode is RiskEnforcementMode.OBSERVE:
        return RiskLimitEvaluation(
            **common,
            state=LimitEvaluationState.OBSERVED,
            value=numeric_value,
            message="Metric observed; no approved threshold is enforced.",
        )

    if (
        control.enforcement_mode is RiskEnforcementMode.BLOCK
        and control.hard_threshold is not None
        and _threshold_reached(control, numeric_value, control.hard_threshold)
    ):
        return RiskLimitEvaluation(
            **common,
            state=LimitEvaluationState.BREACH,
            value=numeric_value,
            message="Approved hard threshold reached.",
        )

    if control.warning_threshold is not None and _threshold_reached(
        control, numeric_value, control.warning_threshold
    ):
        return RiskLimitEvaluation(
            **common,
            state=LimitEvaluationState.WARNING,
            value=numeric_value,
            message="Approved warning threshold reached.",
        )

    return RiskLimitEvaluation(
        **common,
        state=LimitEvaluationState.PASS,
        value=numeric_value,
        message="Metric is within approved thresholds.",
    )


def _threshold_reached(
    control: RiskLimitDefinition,
    value: float,
    threshold: float,
) -> bool:
    if control.direction is RiskLimitDirection.MAX:
        return value >= threshold
    return value <= threshold


def _finite_value(value: float | int | None) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
