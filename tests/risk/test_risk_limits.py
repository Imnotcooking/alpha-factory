from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from oqp.risk.contracts import (
    LimitEvaluationState,
    RiskCalculationStatus,
    RiskEnforcementMode,
    RiskLimitCatalog,
    RiskLimitDefinition,
    RiskLimitDirection,
)
from oqp.risk.limits import evaluate_risk_limit, evaluate_risk_limits


def test_observe_mode_reports_value_without_claiming_pass() -> None:
    result = evaluate_risk_limit(_control(), 0.42)

    assert result.state is LimitEvaluationState.OBSERVED
    assert result.value == pytest.approx(0.42)
    assert result.blocks_action is False


def test_planned_and_unavailable_controls_are_explicit() -> None:
    planned = evaluate_risk_limit(
        _control(calculation_status=RiskCalculationStatus.PLANNED), 0.2
    )
    unavailable = evaluate_risk_limit(
        _control(
            enforcement_mode=RiskEnforcementMode.BLOCK,
            hard_threshold=0.5,
        ),
        None,
    )

    assert planned.state is LimitEvaluationState.PLANNED
    assert unavailable.state is LimitEvaluationState.UNAVAILABLE
    assert unavailable.blocks_action is True


def test_max_control_warns_and_blocks_at_approved_thresholds() -> None:
    control = _control(
        enforcement_mode=RiskEnforcementMode.BLOCK,
        warning_threshold=0.4,
        hard_threshold=0.5,
    )

    assert evaluate_risk_limit(control, 0.39).state is LimitEvaluationState.PASS
    assert evaluate_risk_limit(control, 0.4).state is LimitEvaluationState.WARNING
    breached = evaluate_risk_limit(control, 0.5)
    assert breached.state is LimitEvaluationState.BREACH
    assert breached.blocks_action is True


def test_min_control_uses_lower_values_as_worse() -> None:
    control = _control(
        direction=RiskLimitDirection.MIN,
        enforcement_mode=RiskEnforcementMode.BLOCK,
        warning_threshold=0.3,
        hard_threshold=0.2,
    )

    assert evaluate_risk_limit(control, 0.31).state is LimitEvaluationState.PASS
    assert evaluate_risk_limit(control, 0.3).state is LimitEvaluationState.WARNING
    assert evaluate_risk_limit(control, 0.2).state is LimitEvaluationState.BREACH


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_or_nonfinite_values_are_unavailable(value: object) -> None:
    result = evaluate_risk_limit(_control(), value)  # type: ignore[arg-type]

    assert result.state is LimitEvaluationState.UNAVAILABLE
    assert result.value is None


def test_catalog_evaluation_preserves_catalog_order() -> None:
    catalog = RiskLimitCatalog(
        schema_version="test",
        catalog_owner="risk",
        reporting_currency="USD",
        controls=(_control(control_id="first"), _control(control_id="second")),
        source_path=Path("test.yaml"),
    )

    results = evaluate_risk_limits({"second": 0.2}, catalog=catalog)

    assert [result.control_id for result in results] == ["first", "second"]
    assert results[0].state is LimitEvaluationState.UNAVAILABLE
    assert results[1].state is LimitEvaluationState.OBSERVED


def _control(
    *,
    control_id: str = "test_control",
    direction: RiskLimitDirection = RiskLimitDirection.MAX,
    calculation_status: RiskCalculationStatus = RiskCalculationStatus.ACTIVE,
    enforcement_mode: RiskEnforcementMode = RiskEnforcementMode.OBSERVE,
    warning_threshold: float | None = None,
    hard_threshold: float | None = None,
) -> RiskLimitDefinition:
    return RiskLimitDefinition(
        control_id=control_id,
        category="test",
        scope="portfolio",
        unit="ratio",
        direction=direction,
        calculation_status=calculation_status,
        enforcement_mode=enforcement_mode,
        warning_threshold=warning_threshold,
        hard_threshold=hard_threshold,
        metric_source_path=PurePosixPath("src/oqp/risk/portfolio.py"),
        metric_source_symbol="summarize_portfolio_risk",
        owner="risk",
        description="Test control.",
    )
