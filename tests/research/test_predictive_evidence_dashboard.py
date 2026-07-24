from __future__ import annotations

import pandas as pd

from apps.research_dashboard.views.predictive_evidence_panel import (
    load_predictive_evidence_snapshot,
)
from oqp.research.predictive_evidence import (
    PredictiveEvidenceConfig,
    build_predictive_evidence,
    write_predictive_evidence_bundle,
)


def test_dashboard_loads_only_complete_market_specific_bundle(tmp_path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    products = ("A", "B", "C", "D")
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": product,
                "factor_score": product_index + date_index * 0.1,
                "forward_return": product_index * 0.01 + date_index * 0.001,
                "research_split": (
                    "validation" if date_index < 5 else "holdout"
                ),
            }
            for date_index, date in enumerate(dates)
            for product_index, product in enumerate(products)
        ]
    )
    frame.attrs["causal_return_alignment_verified"] = True
    bundle = build_predictive_evidence(
        frame,
        PredictiveEvidenceConfig(
            factor_id="fac_dashboard_test",
            rolling_window=3,
            rolling_min_periods=2,
            minimum_cross_section=3,
            minimum_product_observations=3,
            execution_lag="next_open",
            return_assumption="next_open_to_close",
        ),
    )
    artifact_root = tmp_path / "research"
    write_predictive_evidence_bundle(
        bundle,
        artifact_root
        / "predictive_evidence"
        / "fac_dashboard_test"
        / "FUTURES_CN",
    )

    snapshot = load_predictive_evidence_snapshot(
        str(artifact_root), "fac_dashboard_test", "FUTURES_CN"
    )

    assert snapshot is not None
    assert snapshot["summary"]["factor_id"] == "fac_dashboard_test"
    assert set(snapshot["split_summary"]["research_split"]) == {
        "full",
        "validation",
        "holdout",
    }
    assert (
        load_predictive_evidence_snapshot(
            str(artifact_root), "fac_dashboard_test", "EQUITY_US"
        )
        is None
    )
