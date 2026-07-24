from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from oqp.risk.contracts import RiskCalculationStatus, RiskEnforcementMode
from oqp.risk.limit_catalog import (
    RISK_LIMIT_CATALOG_SCHEMA_VERSION,
    RiskLimitCatalogError,
    load_risk_limit_catalog,
)


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
CATALOG_PATH = REPO_ROOT / "departments" / "risk" / "limit_catalog.yaml"


def test_committed_risk_limit_catalog_documents_current_control_surface() -> None:
    catalog = load_risk_limit_catalog(CATALOG_PATH)
    by_id = {control.control_id: control for control in catalog.controls}

    assert catalog.schema_version == RISK_LIMIT_CATALOG_SCHEMA_VERSION
    assert catalog.catalog_owner == "risk"
    assert catalog.reporting_currency == "USD"
    assert set(by_id) == {
        "portfolio_gross_exposure_nav",
        "largest_position_weight",
        "top5_position_weight",
        "largest_sector_weight",
        "largest_currency_weight",
        "daily_loss_nav",
        "drawdown_nav",
        "historical_var_95_nav",
        "historical_cvar_95_nav",
        "margin_utilization",
        "option_delta_abs_nav",
        "option_gamma_1pct_nav",
        "option_vega_1vol_nav",
        "option_model_gap_count",
        "critical_reconciliation_break_count",
        "post_trade_gross_exposure_nav",
    }
    assert all(
        control.enforcement_mode is RiskEnforcementMode.OBSERVE
        for control in catalog.controls
    )
    assert {
        control.control_id
        for control in catalog.controls
        if control.calculation_status is RiskCalculationStatus.PLANNED
    } == {
        "margin_utilization",
        "critical_reconciliation_break_count",
        "post_trade_gross_exposure_nav",
    }
    assert catalog.entry("option_model_gap_count") == by_id["option_model_gap_count"]
    assert all(
        (REPO_ROOT / control.metric_source_path).is_file()
        for control in catalog.controls
    )
    for control in catalog.controls:
        module = ast.parse(
            (REPO_ROOT / control.metric_source_path).read_text(encoding="utf-8")
        )
        declared_symbols = {
            node.name
            for node in module.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert control.metric_source_symbol in declared_symbols


def test_risk_limit_catalog_rejects_duplicate_control_ids(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    payload["controls"].append(dict(payload["controls"][0]))

    with pytest.raises(RiskLimitCatalogError, match="Duplicate risk control id"):
        load_risk_limit_catalog(_write_catalog(tmp_path, payload))


def test_risk_limit_catalog_rejects_parent_traversal(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    payload["controls"][0]["metric_source"] = "src/../unsafe.py:calculate"

    with pytest.raises(RiskLimitCatalogError, match="cannot contain"):
        load_risk_limit_catalog(_write_catalog(tmp_path, payload))


def test_risk_limit_catalog_requires_threshold_for_enforcement(
    tmp_path: Path,
) -> None:
    payload = _minimal_catalog()
    payload["controls"][0]["enforcement_mode"] = "block"

    with pytest.raises(RiskLimitCatalogError, match="hard_threshold is required"):
        load_risk_limit_catalog(_write_catalog(tmp_path, payload))


def test_risk_limit_catalog_validates_threshold_order(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    payload["controls"][0].update(
        {
            "enforcement_mode": "block",
            "warning_threshold": 0.8,
            "hard_threshold": 0.7,
        }
    )

    with pytest.raises(RiskLimitCatalogError, match="reached before hard"):
        load_risk_limit_catalog(_write_catalog(tmp_path, payload))


def _minimal_catalog() -> dict:
    return {
        "schema_version": RISK_LIMIT_CATALOG_SCHEMA_VERSION,
        "catalog_owner": "risk",
        "reporting_currency": "USD",
        "controls": [
            {
                "id": "example_limit",
                "category": "exposure",
                "scope": "portfolio",
                "unit": "ratio",
                "direction": "max",
                "calculation_status": "active",
                "enforcement_mode": "observe",
                "warning_threshold": None,
                "hard_threshold": None,
                "metric_source": "src/oqp/risk/portfolio.py:summarize_portfolio_risk",
                "owner": "risk",
                "description": "Example control.",
            }
        ],
    }


def _write_catalog(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "limit_catalog.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
