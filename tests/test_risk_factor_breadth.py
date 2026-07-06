import numpy as np
import pandas as pd

from oqp.risk import (
    RiskBreadthConfig,
    compute_breadth_metrics,
    compute_rolling_breadth,
    extract_base_symbol,
    infer_component_labels,
    map_chinese_futures_sector,
    run_covariance_pca,
)


def _synthetic_returns(days: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=days, freq="B")
    x = np.linspace(0, 8, days)
    common = np.sin(x) * 0.01
    rates = np.cos(x / 2) * 0.004
    return pd.DataFrame(
        {
            "铜(cu)[指数]": common + np.linspace(0, 0.002, days),
            "铝(al)[指数]": common * 0.8 + np.linspace(0, 0.001, days),
            "原油(sc)[指数]": -common * 0.5 + 0.002 * np.sin(x * 2),
            "10年期国债(T)[指数]": rates,
            "黄金(au)[指数]": 0.003 * np.cos(x * 1.5),
        },
        index=idx,
    )


def test_extract_base_symbol_from_chinese_labels_and_contracts():
    assert extract_base_symbol("铜(cu)[指数]") == "cu"
    assert extract_base_symbol("10年期国债(T)[指数]") == "T"
    assert extract_base_symbol("PTA(TA)[主连]") == "TA"
    assert extract_base_symbol("au2608") == "au"
    assert extract_base_symbol("rb2405") == "rb"


def test_sector_mapping_uses_promoted_instrument_master_and_unknown_fallback():
    assert map_chinese_futures_sector("铜(cu)[指数]") == "有色"
    assert map_chinese_futures_sector("黄金(au)[指数]") == "贵金属"
    assert map_chinese_futures_sector("not_a_contract") == "Unknown"


def test_breadth_metrics_on_synthetic_eigenvalues():
    metrics = compute_breadth_metrics(np.array([4.0, 1.0, 0.0]), naive_breadth=10)

    assert metrics["br_threshold"] == 2
    assert metrics["naive_breadth"] == 10
    assert np.isclose(metrics["breadth_haircut"], 0.2)
    assert 1.0 < metrics["effective_rank"] < 3.0
    assert 1.0 < metrics["participation_ratio"] < 3.0


def test_covariance_pca_shapes_and_monotonic_cumulative_variance():
    returns = _synthetic_returns()
    result = run_covariance_pca(
        returns,
        config=RiskBreadthConfig(min_observations=10, min_history_pct=0.5, max_components=4),
    )

    spectrum = result["spectrum"]
    loadings = result["asset_loadings"]

    assert len(spectrum) == returns.shape[1]
    assert (spectrum["explained_variance_ratio"] >= 0).all()
    assert spectrum["cumulative_variance"].is_monotonic_increasing
    assert np.isclose(spectrum["cumulative_variance"].iloc[-1], 1.0)
    assert {"ticker", "sector", "component", "loading", "abs_loading"}.issubset(loadings.columns)
    assert loadings["component"].nunique() == 4
    assert not result["component_labels"].empty
    assert "label_en" in result["component_labels"].columns


def test_component_label_infers_industrial_factor():
    sector_abs = pd.DataFrame(
        {
            "component": ["PC1", "PC1", "PC1"],
            "component_idx": [1, 1, 1],
            "sector": ["化工", "黑色", "能源"],
            "abs_loading_share": [0.4, 0.3, 0.2],
        }
    )
    asset_loadings = pd.DataFrame(
        {
            "component": ["PC1", "PC1", "PC1"],
            "base_symbol": ["TA", "i", "sc"],
            "abs_loading": [0.4, 0.3, 0.2],
        }
    )

    labels = infer_component_labels(sector_abs, asset_loadings)

    assert labels.loc[0, "label_en"] == "Cyclical industrial commodity beta"


def test_rolling_breadth_skips_windows_with_too_few_assets():
    returns = _synthetic_returns(days=40).iloc[:, :3]
    rolling, skipped = compute_rolling_breadth(
        returns,
        config=RiskBreadthConfig(
            min_observations=5,
            min_history_pct=0.5,
            rolling_window=10,
            rolling_step=10,
            rolling_min_assets=5,
        ),
    )

    assert rolling.empty
    assert skipped > 0
