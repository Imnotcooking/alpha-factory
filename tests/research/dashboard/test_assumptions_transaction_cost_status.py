from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.research_dashboard.views.assumptions_view import AssumptionsView


def test_assumptions_cost_status_is_scoped_to_the_selected_run_profile() -> None:
    manifest = {
        "asset_class": "EQUITY_US",
        "costs_and_slippage": {
            "profile_id": "ibkr_pro_fixed_us_equity_v1",
            "profile_fingerprint": "abc123",
            "profile_status": "verified",
            "profile_completeness": "broker_and_regulatory_schedule",
            "use_case": "exploratory_gross",
            "research_net_ready": False,
            "production_ready": False,
            "engine_support": "estimator_only",
            "gross_only": True,
            "profile_assumptions": {
                "profile_id": "ibkr_pro_fixed_us_equity_v1",
                "market_vertical": "EQUITY_US",
                "limitations": ["Exact order-bound engine adapter is pending."],
            },
        },
    }

    status = AssumptionsView._transaction_cost_status(manifest)

    assert status is not None
    assert status["market_vertical"] == "EQUITY_US"
    assert status["profile_id"] == "ibkr_pro_fixed_us_equity_v1"
    assert status["gross_only"] is True
    assert status["limitations"] == ["Exact order-bound engine adapter is pending."]
    assert "FUTURES_CN" not in str(status)


def test_legacy_manifest_uses_only_its_markets_current_default_and_marks_it_unfrozen() -> None:
    manifest = {
        "asset_class": "FUTURES_CN",
        "costs_and_slippage": {
            "summary": "Legacy reconstructed costs",
            "total_execution_cost": 10.0,
        },
    }

    status = AssumptionsView._transaction_cost_status(manifest)

    assert status is not None
    assert status["market_vertical"] == "FUTURES_CN"
    assert status["profile_id"] == "cn_futures_broker_v1"
    assert status["frozen_with_run"] is False
    assert "EQUITY_US" not in str(status)
