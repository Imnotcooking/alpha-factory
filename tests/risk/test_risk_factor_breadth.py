import numpy as np
import pandas as pd

from oqp.risk import (
    RiskBreadthConfig,
    classify_breadth_regimes,
    compute_breadth_metrics,
    compute_component_stability,
    compute_log_return_matrix,
    compute_risk_factor_breadth,
    compute_rolling_breadth,
    extract_base_symbol,
    infer_component_labels,
    load_daily_market_data,
    map_chinese_futures_sector,
    run_covariance_pca,
    summarize_breadth_regime_periods,
    translate_sector_label,
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
    assert extract_base_symbol("KQ.m@CZCE.FG") == "FG"
    assert extract_base_symbol("KQ.i@DCE.i") == "i"
    assert extract_base_symbol("KQ.m@CFFEX.IC") == "IC"


def test_sector_mapping_uses_promoted_instrument_master_and_unknown_fallback():
    assert map_chinese_futures_sector("铜(cu)[指数]") == "有色"
    assert map_chinese_futures_sector("黄金(au)[指数]") == "贵金属"
    assert map_chinese_futures_sector("KQ.m@CZCE.FG") == "建材"
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


def test_component_label_translates_shipping_and_new_energy_sectors():
    assert translate_sector_label("航运", "en") == "Shipping"
    assert translate_sector_label("新能源", "en") == "New energy"

    sector_abs = pd.DataFrame(
        {
            "component": ["PC1", "PC1"],
            "component_idx": [1, 1],
            "sector": ["航运", "化工"],
            "abs_loading_share": [0.55, 0.45],
        }
    )
    asset_loadings = pd.DataFrame(
        {
            "component": ["PC1", "PC1"],
            "base_symbol": ["ec", "TA"],
            "sector": ["航运", "化工"],
            "loading": [0.7, -0.4],
            "abs_loading": [0.7, 0.4],
        }
    )

    labels = infer_component_labels(sector_abs, asset_loadings)

    assert labels.loc[0, "label_en"] == "Shipping vs Chemicals spread"
    assert "航运" not in labels.loc[0, "label_en"]
    assert "Shipping" in labels.loc[0, "positive_basket_en"]
    assert "Chemicals" in labels.loc[0, "negative_basket_en"]


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


def test_breadth_regime_periods_classify_and_collapse_consecutive_windows():
    rolling = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="ME"),
            "br95": [10, 11, 18, 19, 30, 31],
            "effective_rank": [5, 6, 9, 10, 15, 16],
            "participation_ratio": [3, 4, 5, 6, 8, 9],
            "breadth_haircut": [0.20, 0.22, 0.48, 0.50, 0.80, 0.82],
        }
    )

    classified = classify_breadth_regimes(rolling)
    periods = summarize_breadth_regime_periods(classified)

    assert classified["breadth_regime"].tolist() == ["Low", "Low", "Normal", "Normal", "High", "High"]
    assert periods["breadth_regime"].tolist() == ["Low", "Normal", "High"]
    assert periods["windows"].tolist() == [2, 2, 2]
    assert np.isclose(periods.loc[periods["breadth_regime"].eq("High"), "avg_breadth_haircut"].iloc[0], 0.81)


def test_component_stability_tracks_rolling_pca_similarity():
    returns = _synthetic_returns(days=80)
    config = RiskBreadthConfig(
        min_observations=10,
        min_history_pct=0.5,
        rolling_window=30,
        rolling_step=20,
        rolling_min_assets=2,
        stability_components=2,
        stability_max_windows=3,
    )
    baseline = run_covariance_pca(returns, config=config)

    stability, skipped = compute_component_stability(
        returns,
        baseline_loadings=baseline["asset_loadings"],
        baseline_labels=baseline["component_labels"],
        config=config,
    )

    assert skipped >= 0
    assert not stability.empty
    assert set(stability["component"]).issubset({"PC1", "PC2"})
    assert stability["loading_similarity"].between(0, 1).all()
    assert {"label_match", "dominant_sector_match", "label_confidence"}.issubset(
        stability.columns
    )
    assert {"window_dominant_sector_en", "window_dominant_sector_zh"}.issubset(
        stability.columns
    )


def test_log_return_matrix_can_use_brownian_bridge_risk_imputation():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-03", "2026-01-01", "2026-01-02", "2026-01-03"]
            ),
            "ticker": ["A", "A", "B", "B", "B"],
            "close": [100.0, 121.0, 200.0, 202.0, 204.0],
        }
    )

    ffill_returns = compute_log_return_matrix(raw, max_stale_bars=1)
    bridge_returns = compute_log_return_matrix(
        raw,
        max_stale_bars=1,
        risk_imputation="brownian_bridge",
        bridge_max_gap_bars=2,
    )

    assert np.isclose(ffill_returns.loc[pd.Timestamp("2026-01-02"), "A"], 0.0)
    assert bridge_returns.loc[pd.Timestamp("2026-01-02"), "A"] > 0.0
    assert bridge_returns.loc[pd.Timestamp("2026-01-03"), "A"] > 0.0


def test_load_daily_market_data_normalizes_equity_like_parquet(tmp_path):
    path = tmp_path / "equity_cn_sample.parquet"
    raw = pd.DataFrame(
        {
            "symbol": ["SSE.600000", "SSE.600000", "SZSE.000001"],
            "name": ["浦发银行", "浦发银行", "平安银行"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
            "close": [10.0, 10.5, 12.0],
            "volume": [1000, 1100, 900],
        }
    )
    raw.to_parquet(path)

    market = load_daily_market_data(path, asset_class="EQUITY_CN")

    assert list(market.columns) == ["date", "ticker", "close", "volume", "name", "asset_class"]
    assert market["asset_class"].unique().tolist() == ["EQUITY_CN"]
    assert market["ticker"].tolist() == ["SSE.600000", "SSE.600000", "SZSE.000001"]


def test_risk_breadth_accepts_equity_like_panel_and_caps_assets(tmp_path):
    path = tmp_path / "equity_panel.parquet"
    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    rows = []
    for asset_idx, ticker in enumerate(["SSE.600000", "SSE.600004", "SZSE.000001"]):
        for pos, date in enumerate(dates):
            rows.append(
                {
                    "symbol": ticker,
                    "date": date,
                    "close": 10.0 + asset_idx + pos * (0.1 + asset_idx * 0.02),
                    "volume": 1000 + asset_idx * 100,
                    "sector": "Bank" if asset_idx != 1 else "Airport",
                }
            )
    pd.DataFrame(rows).to_parquet(path)

    result = compute_risk_factor_breadth(
        path,
        RiskBreadthConfig(
            asset_class="EQUITY_CN",
            min_observations=4,
            min_history_pct=0.3,
            rolling_window=6,
            rolling_step=3,
            rolling_min_assets=2,
            max_assets=2,
        ),
    )

    assert result["asset_class"] == "EQUITY_CN"
    assert result["source_assets"] == 3
    assert result["selected_assets"] == 2
    assert result["metrics"]["valid_assets"] == 2
    assert not result["spectrum"].empty
    assert "component_stability" in result
