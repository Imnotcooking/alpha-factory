from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oqp.research.predictive_evidence import (
    CausalAlignmentError,
    PredictiveEvidenceConfig,
    build_predictive_evidence,
    load_predictive_evidence_bundle,
    write_predictive_evidence_bundle,
)


def _panel(*, return_sign: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(71)
    dates = pd.bdate_range("2024-01-02", periods=18)
    products = [f"P{index:02d}" for index in range(10)]
    rows = []
    for date_index, date in enumerate(dates):
        for product_index, product in enumerate(products):
            signal = np.sin(date_index / 3.0 + product_index * 0.47)
            signal += 0.08 * (product_index - 4.5)
            forward_return = return_sign * 0.02 * signal + rng.normal(0.0, 0.003)
            rows.append(
                {
                    "date": date,
                    "ticker": product,
                    "factor_score": signal,
                    "forward_return": forward_return,
                    "research_split": (
                        "validation" if date_index < 12 else "holdout"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["causal_return_alignment_verified"] = True
    frame.attrs["dataset_manifest_fingerprint"] = "dataset-sha"
    frame.attrs["factor_definition_fingerprint"] = "factor-sha"
    return frame


def _config(**overrides) -> PredictiveEvidenceConfig:
    values = {
        "factor_id": "fac_test",
        "rolling_window": 5,
        "rolling_min_periods": 3,
        "minimum_cross_section": 8,
        "minimum_product_observations": 5,
        "execution_lag": "next_open",
        "return_assumption": "next_open_to_close",
    }
    values.update(overrides)
    return PredictiveEvidenceConfig(**values)


def test_builds_cross_sectional_and_product_predictive_evidence() -> None:
    bundle = build_predictive_evidence(_panel(), _config())

    full = bundle.split_summary.set_index("research_split").loc["full"]
    assert set(bundle.split_summary["research_split"]) == {
        "full",
        "validation",
        "holdout",
    }
    assert full["mean_pearson_ic"] > 0.9
    assert full["mean_rank_ic"] > 0.9
    assert full["pearson_ic_hit_rate"] == pytest.approx(1.0)
    assert full["rank_ic_hit_rate"] == pytest.approx(1.0)
    assert full["date_count"] == 18
    assert full["product_count"] == 10
    assert full["joint_coverage"] == pytest.approx(1.0)

    assert bundle.period_ic["rolling_rank_ic"].notna().sum() == 16
    assert bundle.period_ic["cumulative_rank_ic"].iloc[-1] > 0
    full_products = bundle.product_ic.query("research_split == 'full'")
    assert len(full_products) == 10
    assert (full_products["oriented_rank_ic"] > 0).all()
    assert set(bundle.concentration["dimension"]) == {"year", "product"}
    assert bundle.summary["causal_alignment_verified"] is True
    assert "transaction costs" in bundle.summary["interpretation_boundary"]


def test_expected_sign_orients_reversal_evidence_without_changing_raw_ic() -> None:
    bundle = build_predictive_evidence(
        _panel(return_sign=-1),
        _config(expected_sign=-1),
    )

    assert bundle.period_ic["pearson_ic"].mean() < -0.9
    assert bundle.period_ic["oriented_pearson_ic"].mean() > 0.9
    full = bundle.split_summary.set_index("research_split").loc["full"]
    assert full["pearson_ic_hit_rate"] == pytest.approx(1.0)
    assert full["mean_pearson_ic"] < -0.9
    assert full["oriented_mean_pearson_ic"] > 0.9


def test_reports_signal_and_forward_return_coverage_separately() -> None:
    frame = _panel()
    frame.loc[frame.index[:9], "factor_score"] = np.nan
    frame.loc[frame.index[9:15], "forward_return"] = np.nan

    bundle = build_predictive_evidence(frame, _config())
    full = bundle.split_summary.set_index("research_split").loc["full"]

    assert full["signal_coverage"] == pytest.approx(171 / 180)
    assert full["forward_return_coverage"] == pytest.approx(174 / 180)
    assert full["joint_coverage"] == pytest.approx(165 / 180)
    assert full["active_signal_coverage"] <= full["signal_coverage"]


def test_rejects_unverified_or_noncausal_forward_returns() -> None:
    frame = _panel()
    frame.attrs.clear()
    with pytest.raises(CausalAlignmentError, match="causal alignment"):
        build_predictive_evidence(frame, _config())

    frame["signal_available_at"] = frame["date"] + pd.Timedelta(hours=15)
    frame["return_starts_at"] = frame["signal_available_at"]
    frame["return_ends_at"] = frame["date"] + pd.Timedelta(days=1, hours=15)
    timestamp_config = _config(
        signal_available_col="signal_available_at",
        return_start_col="return_starts_at",
        return_end_col="return_ends_at",
    )
    with pytest.raises(CausalAlignmentError, match="must start after"):
        build_predictive_evidence(frame, timestamp_config)


def test_requires_execution_assumptions_for_causal_claims() -> None:
    with pytest.raises(ValueError, match="execution_lag"):
        PredictiveEvidenceConfig(
            factor_id="fac_test",
            return_assumption="next_open_to_close",
        )
    with pytest.raises(ValueError, match="return_assumption"):
        PredictiveEvidenceConfig(
            factor_id="fac_test",
            execution_lag="next_open",
        )


def test_bundle_round_trip(tmp_path) -> None:
    original = build_predictive_evidence(_panel(), _config())
    output_dir = write_predictive_evidence_bundle(original, tmp_path / "evidence")
    restored = load_predictive_evidence_bundle(output_dir)

    assert restored.config == original.config
    assert restored.manifest["config_fingerprint"] == original.config.fingerprint
    assert restored.summary["factor_id"] == "fac_test"
    pd.testing.assert_frame_equal(restored.period_ic, original.period_ic)
    assert set(path.name for path in output_dir.iterdir()) == {
        "concentration.csv",
        "manifest.json",
        "period_ic.parquet",
        "product_ic.csv",
        "split_summary.csv",
        "summary.json",
        "yearly_summary.csv",
    }
