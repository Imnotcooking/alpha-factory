from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oqp.research.sleeves import (
    SleeveAlignmentError,
    SleeveConstructionConfig,
    build_sleeve_evidence,
    build_sleeve_targets,
    load_sleeve_evidence_bundle,
    write_sleeve_evidence_bundle,
)


def _panel(*, dates: int = 4, products: int = 10) -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.bdate_range("2025-01-02", periods=dates)):
        for product_index in range(products):
            rows.append(
                {
                    "date": date,
                    "ticker": f"P{product_index:02d}",
                    "sector": "A" if product_index < products / 2 else "B",
                    "alpha_score": float(product_index),
                    "research_split": "validation" if date_index < dates - 1 else "holdout",
                    "forward_return": (product_index - (products - 1) / 2.0) / 1_000.0,
                    "next_symbol": f"P{product_index:02d}01",
                    "next_actual_open": 100.0,
                    "next_multiplier": 10.0,
                    "next_tick_size": 1.0,
                    "next_fee_type": "fixed",
                    "next_fee_open": 2.0,
                    "next_fee_close_today": 3.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["causal_return_alignment_verified"] = True
    frame.attrs["input_data_fingerprint"] = "data-sha"
    frame.attrs["factor_definition_fingerprint"] = "factor-sha"
    return frame


def _config(**overrides) -> SleeveConstructionConfig:
    values = {
        "sleeve_id": "slv_test",
        "factor_id": "fac_test",
        "market_vertical": "FUTURES_CN",
        "max_weight_per_contract": 0.30,
    }
    values.update(overrides)
    return SleeveConstructionConfig(**values)


def test_fixed_top_bottom_sleeve_is_market_neutral() -> None:
    result = build_sleeve_targets(_panel(), _config())
    first = result.positions.loc[result.positions["date"].eq(pd.Timestamp("2025-01-02"))]

    assert first["target_weight"].gt(0).sum() == 2
    assert first["target_weight"].lt(0).sum() == 2
    assert first["target_weight"].abs().sum() == pytest.approx(1.0)
    assert first["target_weight"].sum() == pytest.approx(0.0)
    assert set(first.loc[first["target_weight"].gt(0), "ticker"]) == {"P08", "P09"}
    assert set(first.loc[first["target_weight"].lt(0), "ticker"]) == {"P00", "P01"}


def test_bearish_orientation_reverses_the_ranking() -> None:
    result = build_sleeve_targets(
        _panel(),
        _config(signal_orientation="higher_is_bearish"),
    )
    first = result.positions.loc[result.positions["date"].eq(pd.Timestamp("2025-01-02"))]
    assert set(first.loc[first["target_weight"].gt(0), "ticker"]) == {"P00", "P01"}
    assert set(first.loc[first["target_weight"].lt(0), "ticker"]) == {"P08", "P09"}


def test_time_series_sign_accepts_one_common_direction() -> None:
    frame = _panel(dates=1, products=5)
    frame["alpha_score"] = 1.0
    result = build_sleeve_targets(
        frame,
        _config(
            construction_geometry="time_series",
            expression="directional",
            construction="time_series_sign",
            normalization="equal_weight",
            zero_signal_policy="neutral",
            minimum_cross_section=3,
            minimum_distinct_signals=1,
            max_weight_per_contract=None,
            winsor_lower_quantile=None,
            winsor_upper_quantile=None,
        ),
    )

    assert result.positions["target_weight"].tolist() == pytest.approx([0.2] * 5)
    assert result.positions["target_weight"].abs().sum() == pytest.approx(1.0)
    assert result.positions["selection_side"].eq("long").all()


def test_cross_sectional_construction_still_requires_two_distinct_signals() -> None:
    with pytest.raises(
        ValueError,
        match="cross-sectional minimum_distinct_signals must be at least 2",
    ):
        _config(minimum_distinct_signals=1)


def test_continuous_zscore_sleeve_is_neutral_and_scales_score_magnitude() -> None:
    result = build_sleeve_targets(
        _panel(dates=1, products=5),
        _config(
            construction="continuous_zscore",
            normalization="zscore_weight",
            minimum_cross_section=3,
            max_weight_per_contract=None,
            winsor_lower_quantile=None,
            winsor_upper_quantile=None,
        ),
    )
    positions = result.positions.set_index("ticker")

    assert positions["target_weight"].abs().sum() == pytest.approx(1.0)
    assert positions["target_weight"].sum() == pytest.approx(0.0)
    assert positions.loc["P04", "target_weight"] == pytest.approx(1.0 / 3.0)
    assert positions.loc["P03", "target_weight"] == pytest.approx(1.0 / 6.0)
    assert positions.loc["P02", "target_weight"] == pytest.approx(0.0)
    assert positions["cross_sectional_zscore"].mean() == pytest.approx(0.0)
    assert positions["cross_sectional_zscore"].std(ddof=0) == pytest.approx(1.0)


def test_proportional_score_preserves_magnitude_and_does_not_refill_cap() -> None:
    frame = _panel(dates=1, products=5)
    frame["alpha_score"] = [-4.0, -1.0, 0.0, 1.0, 4.0]
    result = build_sleeve_targets(
        frame,
        _config(
            construction="proportional_score",
            normalization="absolute_score_to_gross",
            expression="directional",
            zero_signal_policy="neutral",
            minimum_cross_section=3,
            max_weight_per_contract=0.12,
            winsor_lower_quantile=None,
            winsor_upper_quantile=None,
        ),
    )
    positions = result.positions.set_index("ticker")

    assert positions["target_weight"].tolist() == pytest.approx(
        [-0.12, -0.10, 0.0, 0.10, 0.12]
    )
    assert positions["target_weight"].abs().sum() == pytest.approx(0.44)
    assert positions["target_weight"].sum() == pytest.approx(0.0)
    assert positions["contract_cap_bound"].sum() == 2
    assert positions["winsorized_signal"].dropna().tolist() == pytest.approx(
        [-4.0, -1.0, 1.0, 4.0]
    )


def test_proportional_score_respects_bearish_orientation() -> None:
    frame = _panel(dates=1, products=5)
    frame["alpha_score"] = [-4.0, -1.0, 0.0, 1.0, 4.0]
    common = {
        "construction": "proportional_score",
        "normalization": "absolute_score_to_gross",
        "expression": "directional",
        "zero_signal_policy": "neutral",
        "minimum_cross_section": 3,
        "max_weight_per_contract": None,
        "winsor_lower_quantile": None,
        "winsor_upper_quantile": None,
    }
    bullish = build_sleeve_targets(
        frame,
        _config(**common),
    ).positions.set_index("ticker")["target_weight"]
    bearish = build_sleeve_targets(
        frame,
        _config(signal_orientation="higher_is_bearish", **common),
    ).positions.set_index("ticker")["target_weight"]

    assert bearish.tolist() == pytest.approx((-bullish).tolist())


def test_time_series_proportional_score_preserves_magnitude_without_refilling_cap() -> None:
    frame = _panel(dates=1, products=5)
    frame["alpha_score"] = [-4.0, -1.0, 0.0, 1.0, 4.0]
    result = build_sleeve_targets(
        frame,
        _config(
            construction_geometry="time_series",
            construction="proportional_score",
            normalization="absolute_score_to_gross",
            expression="directional",
            return_assumption="close_signal_next_open_to_next_open",
            zero_signal_policy="neutral",
            minimum_cross_section=3,
            max_weight_per_contract=0.12,
            winsor_lower_quantile=None,
            winsor_upper_quantile=None,
        ),
    )
    positions = result.positions.set_index("ticker")

    assert positions["target_weight"].tolist() == pytest.approx(
        [-0.12, -0.10, 0.0, 0.10, 0.12]
    )
    assert positions["target_weight"].abs().sum() == pytest.approx(0.44)
    assert positions["contract_cap_bound"].sum() == 2
    assert positions["winsorized_signal"].dropna().tolist() == pytest.approx(
        [-4.0, -1.0, 1.0, 4.0]
    )


def test_proportional_score_contract_rejects_incompatible_semantics() -> None:
    with pytest.raises(
        ValueError,
        match="proportional_score requires absolute_score_to_gross",
    ):
        _config(
            construction="proportional_score",
            normalization="equal_weight",
            expression="directional",
        )
    with pytest.raises(
        ValueError,
        match="proportional_score requires directional expression",
    ):
        _config(
            construction="proportional_score",
            normalization="absolute_score_to_gross",
            expression="long_short",
        )


def test_contract_and_sector_caps_reduce_realized_gross_without_breaking_neutrality() -> None:
    contract_capped = build_sleeve_targets(
        _panel(), _config(max_weight_per_contract=0.05)
    ).daily_summary
    assert contract_capped["gross_exposure"].iloc[0] == pytest.approx(0.20)
    assert contract_capped["net_exposure"].iloc[0] == pytest.approx(0.0)

    sector_capped = build_sleeve_targets(
        _panel(products=20),
        _config(max_weight_per_contract=0.30, max_sector_gross=0.40),
    ).daily_summary
    assert sector_capped["gross_exposure"].iloc[0] == pytest.approx(0.40)
    assert sector_capped["net_exposure"].iloc[0] == pytest.approx(0.0)
    assert sector_capped["sector_cap_count"].iloc[0] == 8


def test_rebalance_and_one_period_holding_do_not_forward_fill() -> None:
    result = build_sleeve_targets(
        _panel(),
        _config(rebalance_every_n_periods=2, holding_periods=1),
    )
    gross = result.daily_summary["gross_exposure"].tolist()
    assert gross == pytest.approx([1.0, 0.0, 1.0, 0.0])


def test_sparse_event_zeros_are_not_forced_into_quantile_positions() -> None:
    frame = _panel(dates=1)
    frame["alpha_score"] = 0.0
    frame.loc[frame["ticker"].isin(["P00", "P01"]), "alpha_score"] = -1.0
    frame.loc[frame["ticker"].isin(["P08", "P09"]), "alpha_score"] = 1.0
    result = build_sleeve_targets(
        frame,
        _config(zero_signal_policy="neutral", minimum_cross_section=5),
    )
    assert result.positions["target_weight"].eq(0.0).all()

    enough = pd.concat([frame, frame.assign(
        ticker=frame["ticker"].map(lambda value: f"X{value}"),
        next_symbol=frame["next_symbol"].map(lambda value: f"X{value}"),
    )], ignore_index=True)
    active = build_sleeve_targets(
        enough,
        _config(zero_signal_policy="neutral", minimum_cross_section=5),
    ).positions
    assert active.loc[active["target_weight"].ne(0.0), "alpha_score"].ne(0.0).all()


def test_requires_causal_attestation_and_blocks_optimization() -> None:
    frame = _panel()
    frame.attrs.clear()
    with pytest.raises(SleeveAlignmentError, match="causal"):
        build_sleeve_targets(frame, _config())
    with pytest.raises(ValueError, match="cannot permit optimization"):
        _config(optimization_permitted=True)
    with pytest.raises(ValueError, match="execution delay"):
        _config(signal_timing="decision_time", execution_delay_periods=0)


def test_intraday_execution_charges_both_sides_and_whole_contracts() -> None:
    construction = build_sleeve_targets(_panel(), _config())
    bundle = build_sleeve_evidence(
        construction,
        capital=1_000_000.0,
        slippage_ticks_per_side=0.5,
    )
    first = bundle.daily_returns.iloc[0]

    assert first["turnover"] == pytest.approx(2.0)
    assert first["exchange_fee_return"] == pytest.approx(0.005)
    assert first["slippage_return"] == pytest.approx(0.01)
    assert first["cost_return"] == pytest.approx(0.015)
    assert first["executed_gross"] == pytest.approx(1.0)
    assert first["net_return"] == pytest.approx(first["gross_return"] - 0.015)
    assert np.equal(np.modf(bundle.positions["contracts"])[0], 0.0).all()


def test_evidence_bundle_round_trip(tmp_path) -> None:
    construction = build_sleeve_targets(_panel(), _config())
    original = build_sleeve_evidence(
        construction,
        capital=10_000_000.0,
        slippage_ticks_per_side=0.5,
    )
    output = write_sleeve_evidence_bundle(original, tmp_path / "sleeve")
    restored = load_sleeve_evidence_bundle(output)

    assert restored.config == original.config
    assert restored.manifest["config_fingerprint"] == original.config.fingerprint
    assert restored.summary["factor_id"] == "fac_test"
    pd.testing.assert_frame_equal(restored.daily_returns, original.daily_returns)
    assert {path.name for path in output.iterdir()} == {
        "daily_returns.parquet",
        "manifest.json",
        "positions.parquet",
        "product_summary.csv",
        "sector_summary.csv",
        "split_summary.csv",
        "summary.json",
        "yearly_summary.csv",
    }
