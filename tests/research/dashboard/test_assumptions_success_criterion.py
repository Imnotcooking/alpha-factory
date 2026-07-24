from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.research_dashboard.views.assumptions_view import AssumptionsView


def test_assumptions_extracts_frozen_success_criterion() -> None:
    manifest = {
        "success_criterion": {
            "status": "incomplete",
            "profile_id": "router_incremental_net_value_v1",
            "profile_fingerprint": "abc123",
            "definition": {
                "research_object": "router",
                "decision_sample": "development_validation",
                "economic_question": "Does routing add value?",
                "primary": {
                    "metric": "validation_net_mean_return",
                    "comparator_metric": (
                        "best_alternative_validation_net_mean_return"
                    ),
                    "minimum_improvement": 0.0,
                    "absolute_floor": 0.0,
                },
                "gates": [
                    {
                        "name": "incremental_hac_evidence",
                        "metric": "validation_increment_hac_t",
                        "operator": ">=",
                        "threshold": 1.645,
                    }
                ],
            },
            "evaluation": {
                "decision": "incomplete",
                "missing_metrics": ["validation_increment_hac_t"],
            },
        }
    }

    status = AssumptionsView._success_criterion_status(manifest)

    assert status is not None
    assert status["status"] == "incomplete"
    assert status["research_object"] == "router"
    assert status["primary_metric"] == "validation_net_mean_return"
    assert status["missing_metrics"] == ["validation_increment_hac_t"]


def test_legacy_manifest_without_criterion_returns_none() -> None:
    assert AssumptionsView._success_criterion_status({}) is None
