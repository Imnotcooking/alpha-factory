from __future__ import annotations

import importlib.util
import importlib
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

from oqp.research.tick_pulse import (
    CACHE_SCHEMA_VERSION,
    PULSE_CACHE_TYPE,
    build_directionless_pulse_frame,
    build_pulse_zone_summary,
    compute_main_contract_file_sweep,
    evaluate_microstructure_hypothesis,
    make_cache_key,
    make_hypothesis_seed_id,
)
from oqp.research.tick_pulse import engine as package_tick_engine


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResearchDashboardPreflightTests(unittest.TestCase):
    def test_research_dashboard_pages_compile(self) -> None:
        sidebar_pages = sorted((RESEARCH_APP / "pages").glob("*.py"))
        self.assertEqual(6, len(sidebar_pages))

        page_paths = [
            RESEARCH_APP / "Homepage.py",
            RESEARCH_APP / "app.py",
            *sidebar_pages,
            RESEARCH_APP / "discovery" / "pattern_page.py",
            RESEARCH_APP / "discovery" / "relationship_page.py",
        ]

        for path in page_paths:
            with self.subTest(path=path.name):
                compile(path.read_text(), str(path), "exec")

    def test_runtime_paths_are_named_and_resolvable(self) -> None:
        config = _load_module(RESEARCH_APP / "config.py", "_research_dashboard_config_preflight")

        self.assertEqual(
            Path(config.ALPHA_RUNTIME_DATA_ROOT),
            REPO_ROOT / "runtime" / "data",
        )
        self.assertEqual(
            Path(config.RESEARCH_ARTIFACT_ROOT),
            REPO_ROOT / "runtime" / "artifacts" / "research",
        )
        self.assertEqual(Path(config.LOGS_DIR), Path(config.RESEARCH_ARTIFACT_ROOT))
        self.assertTrue(Path(config.DB_PATH).parent.exists())
        self.assertTrue((REPO_ROOT / "runtime" / "logs").exists())

    def test_runtime_estimator_summarizes_asset_routes(self) -> None:
        estimator = _load_module(RESEARCH_APP / "runtime_estimator.py", "_research_runtime_estimator")
        runs = pd.DataFrame(
            {
                "asset_class": ["EQUITY_CN", "OPTIONS_US"],
                "validation_rows": [4_000_000, 100_000],
                "holdout_rows": [4_000_000, 50_000],
                "data_frequency": ["daily", "daily"],
                "execution_mode": ["direct", "event_driven_options"],
                "timestamp": ["2026-07-08", "2026-07-08"],
            }
        )

        frame = estimator.runtime_estimate_frame(runs)

        self.assertEqual(set(frame["Asset"]), {"EQUITY_CN", "OPTIONS_US"})
        self.assertIn("8,000,000", frame.loc[frame["Asset"].eq("EQUITY_CN"), "Rows"].iloc[0])
        self.assertIn("Event-driven", frame.loc[frame["Asset"].eq("OPTIONS_US"), "Notes"].iloc[0])

    def test_tick_pulse_backend_helpers_are_package_owned(self) -> None:
        self.assertTrue(CACHE_SCHEMA_VERSION.startswith("tick_pulse_research_cache_"))
        self.assertEqual(PULSE_CACHE_TYPE, "directionless_pulse_discovery_v1")
        self.assertEqual(make_cache_key({"b": 2, "a": 1}), make_cache_key({"a": 1, "b": 2}))
        self.assertEqual(
            make_hypothesis_seed_id({"symbol": "au2608", "rule": {"x": 1}}),
            make_hypothesis_seed_id({"rule": {"x": 1}, "symbol": "au2608"}),
        )
        self.assertTrue(callable(build_directionless_pulse_frame))
        self.assertTrue(callable(build_pulse_zone_summary))
        self.assertTrue(callable(evaluate_microstructure_hypothesis))
        self.assertTrue(callable(compute_main_contract_file_sweep))

    def test_app_tick_pulse_wrappers_delegate_to_package_helpers(self) -> None:
        wrapper_dir = RESEARCH_APP / "tick_pulse_lab"
        cache_wrapper = _load_module(wrapper_dir / "research_cache.py", "_tick_cache_wrapper")
        seed_wrapper = _load_module(wrapper_dir / "hypothesis_seeds.py", "_tick_seed_wrapper")
        pulse_wrapper = _load_module(wrapper_dir / "pulse_discovery.py", "_pulse_discovery_wrapper")
        engine_wrapper = _load_module(wrapper_dir / "engine.py", "_tick_engine_wrapper")

        self.assertEqual(cache_wrapper.CACHE_SCHEMA_VERSION, CACHE_SCHEMA_VERSION)
        self.assertIs(cache_wrapper.make_cache_key, make_cache_key)
        self.assertIs(seed_wrapper.make_hypothesis_seed_id, make_hypothesis_seed_id)
        self.assertIs(pulse_wrapper.build_directionless_pulse_frame, build_directionless_pulse_frame)
        self.assertIs(engine_wrapper.evaluate_microstructure_hypothesis, evaluate_microstructure_hypothesis)
        self.assertIs(engine_wrapper._build_research_sweep, package_tick_engine._build_research_sweep)
        self.assertIs(engine_wrapper._evaluate_horizon_summary, package_tick_engine._evaluate_horizon_summary)
        self.assertIs(
            engine_wrapper._is_relative_velocity_fade_hypothesis,
            package_tick_engine._is_relative_velocity_fade_hypothesis,
        )

    def test_tick_pulse_evidence_ticket_payload_is_reproducible(self) -> None:
        app_path = str(RESEARCH_APP)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)
        dashboard = importlib.import_module("tick_pulse_lab.dashboard")

        features = pd.DataFrame(
            {
                "expected_direction": ["Up", "Up", "Down", "Down"],
                "future_move_ticks": [2.0, 0.5, -2.0, None],
            }
        )
        candidates = features.loc[[0, 2]].copy()
        candidates["is_correct"] = [True, True]

        kwargs = {
            "selected_file": "runtime/data/futures_cn/tick/demo_tick_all_data.parquet",
            "file_path": str(REPO_ROOT / "runtime" / "data" / "futures_cn" / "tick" / "demo_tick_all_data.parquet"),
            "selected_product": "au",
            "selected_symbol": "au2608",
            "selected_hypothesis": "relative_velocity",
            "hypothesis_label": "Relative tick velocity",
            "window": 120,
            "horizon_ticks": 120,
            "min_success_ticks": 1.0,
            "threshold_mode": "adaptive",
            "thresholds": {"rtv_min_fast_move_ticks": 1.5},
            "features": features,
            "candidates": candidates,
            "backend": "python",
            "selected_seed": None,
        }
        first = dashboard._build_tick_pulse_evidence_ticket_payload(**kwargs)
        second = dashboard._build_tick_pulse_evidence_ticket_payload(**kwargs)

        self.assertEqual(first["ticket_id"], second["ticket_id"])
        self.assertTrue(str(first["ticket_id"]).startswith("ticket_"))
        self.assertEqual(first["source_page"], "02_Discovery_Lab")
        self.assertEqual(first["research_family"], "tick_pulse_lab")
        self.assertEqual(first["factor_id"], "tick_pulse_relative_velocity")
        self.assertEqual(first["decision"], "promote_to_validation")
        self.assertEqual(first["status"], "ready_for_review")
        self.assertEqual(first["metrics"]["events"], 2)
        self.assertAlmostEqual(first["metrics"]["accuracy"], 1.0)
        self.assertAlmostEqual(first["metrics"]["base_rate"], 2 / 3)
        self.assertEqual(first["context"]["symbol"], "au2608")
        self.assertEqual(first["context"]["thresholds"], {"rtv_min_fast_move_ticks": 1.5})

    def test_factor_promotion_run_history_uses_truthful_legacy_fallbacks(self) -> None:
        app_path = str(RESEARCH_APP)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)
        view_module = importlib.import_module("views.factor_promotion_view")
        view = view_module.FactorPromotionView

        legacy_row = pd.Series(
            {
                "market_vertical": None,
                "asset_class": "FUTURES_CN",
                "universe_id": None,
                "traded_tickers": "ALL",
                "universe_size": 69,
            }
        )
        selected_row = legacy_row.copy()
        selected_row["traded_tickers"] = "a,b,c,d,e,f"

        self.assertEqual(view._run_market_label(legacy_row), "FUTURES_CN")
        self.assertEqual(
            view._run_universe_label(legacy_row, "assets", "traded"),
            "ALL (69 assets)",
        )
        self.assertEqual(
            view._run_universe_label(selected_row, "assets", "traded"),
            "6 traded (69 assets)",
        )

        coverage = pd.DataFrame(
            {
                "dataset_id": [None, ""],
                "total_trades": [0, 0],
            }
        )
        self.assertFalse(view._series_has_values(coverage, "dataset_id"))
        self.assertTrue(view._series_has_values(coverage, "total_trades"))

    def test_streamlit_nav_widgets_do_not_mix_default_with_seeded_state(self) -> None:
        pulse_page = (RESEARCH_APP / "discovery" / "pattern_page.py").read_text()
        tick_page = (RESEARCH_APP / "tick_pulse_lab" / "dashboard.py").read_text()

        self.assertNotIn('default=st.session_state["pulse_discovery_nav"]', pulse_page)
        self.assertNotIn('default=st.session_state["tick_pulse_nav"]', tick_page)
        self.assertNotIn('default=st.session_state["tick_scope_hypothesis_source"]', tick_page)
        self.assertNotIn("default=threshold_mode", tick_page)

    def test_discovery_lab_shell_preserves_three_workflow_boundaries(self) -> None:
        shell = (RESEARCH_APP / "pages" / "02_Discovery_Lab.py").read_text()

        self.assertIn('options=["pattern", "event_study", "relationships"]', shell)
        self.assertIn('"relationship_page.py"', shell)
        self.assertIn('"pattern_page.py"', shell)
        self.assertIn('"_OQP_EMBEDDED_STREAMLIT_PAGE": True', shell)
        self.assertIn('init_globals={', shell)
        self.assertFalse((RESEARCH_APP / "pages" / "02_Pattern_Lab.py").exists())
        self.assertFalse((RESEARCH_APP / "pages" / "04_Arbitrage_Lab.py").exists())

    def test_intraday_pattern_lab_clock_edge_summary_buckets_sessions(self) -> None:
        original_flag = os.environ.get("OQP_EMBEDDED_STREAMLIT_PAGE")
        os.environ["OQP_EMBEDDED_STREAMLIT_PAGE"] = "1"
        try:
            page = _load_module(
                RESEARCH_APP / "discovery" / "pattern_page.py",
                "_intraday_pattern_lab_preflight",
            )
        finally:
            if original_flag is None:
                os.environ.pop("OQP_EMBEDDED_STREAMLIT_PAGE", None)
            else:
                os.environ["OQP_EMBEDDED_STREAMLIT_PAGE"] = original_flag

        times = (
            list(pd.date_range("2026-01-05 09:00", periods=70, freq="min"))
            + list(pd.date_range("2026-01-05 14:00", periods=70, freq="min"))
            + list(pd.date_range("2026-01-05 22:00", periods=70, freq="min"))
        )
        moves = [0.1] * 70 + [0.4] * 70 + [-0.2] * 70
        prices = []
        price = 100.0
        for move in moves:
            price += move
            prices.append(price)
        scope = pd.DataFrame(
            {
                "datetime": times,
                "symbol": "au2608",
                "last_price": prices,
                "tick_size_est": 1.0,
            }
        )

        summary = page._clock_edge_summary(scope, min_bars=20)
        by_session = summary.groupby("session_key")["avg_move_ticks"].mean()

        self.assertIn("morning", by_session)
        self.assertIn("afternoon", by_session)
        self.assertIn("night", by_session)
        self.assertGreater(by_session["afternoon"], by_session["morning"])
        self.assertLess(by_session["night"], 0)

    def test_homepage_ml_tab_is_artifact_gated(self) -> None:
        homepage = (RESEARCH_APP / "Homepage.py").read_text()

        self.assertIn("ml_view.should_render_tab", homepage)
        self.assertIn('if "ml" in tabs:', homepage)
        self.assertNotIn("with tabs[4]", homepage)
        self.assertNotIn("with tabs[5]", homepage)

    def test_homepage_analysis_tabs_follow_research_workflow_order(self) -> None:
        homepage = (RESEARCH_APP / "Homepage.py").read_text()
        expected_order = (
            '("tearsheet", t["tab_tearsheet"])',
            '("dna", t["tab_dna"])',
            '("pareto", t["tab_pareto"])',
            '("correlation", t["tab_correlation"])',
            'tab_specs.append(("assumptions", t.get("tab_assumptions", "Assumptions")))',
        )

        positions = [homepage.index(item) for item in expected_order]
        self.assertEqual(positions, sorted(positions))

    def test_homepage_run_context_card_precedes_tabs(self) -> None:
        homepage = (RESEARCH_APP / "Homepage.py").read_text()

        context_index = homepage.index("_render_run_context_card(asset_class, universe_label, context_scope, t)")
        tabs_index = homepage.index("tabs = dict(zip")
        self.assertLess(context_index, tabs_index)
        self.assertIn("show_test_scope=False", homepage)
        self.assertNotIn('st.info(f"**🏛️ Asset Class:', homepage)
        self.assertIn('card = (', homepage)
        self.assertNotIn('f"""\n        <style>', homepage)

    def test_homepage_tearsheet_hover_shows_exact_dates(self) -> None:
        tearsheet = (RESEARCH_APP / "views" / "tearsheet_view.py").read_text()

        self.assertIn('days = pd.to_datetime(df[\'date\'], errors="coerce")', tearsheet)
        self.assertIn("Date: %{x|%Y-%m-%d}", tearsheet)
        self.assertIn('fig.update_xaxes(hoverformat="%Y-%m-%d")', tearsheet)

    def test_tearsheet_benchmark_title_and_guide_are_compact(self) -> None:
        app_path = str(RESEARCH_APP)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)
        view_module = importlib.import_module("views.tearsheet_view")
        view = view_module.TearSheetView(None)

        returns = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                "benchmark_return": [0.001, -0.002],
                "benchmark_return_passive_active_universe": [0.003, 0.001],
                "benchmark_return_nanhua": [0.002, 0.0],
                "benchmark_return_same_horizon": [0.0005, -0.001],
            }
        )
        manifest = {
            "benchmark": {
                "benchmark_label": "Futures active-universe equal-weight same-horizon control",
                "benchmark_column": "benchmark_return",
                "benchmark_role": "same_horizon_universe_control",
                "return_mode": "same_horizon",
                "secondary_benchmarks": [
                    {
                        "benchmark_label": "Dynamic equal-weight active futures universe",
                        "benchmark_column": "benchmark_return_passive_active_universe",
                        "benchmark_role": "active_universe_futures_basket",
                    },
                    {
                        "benchmark_label": "Nanhua commodity index buy-and-hold",
                        "benchmark_column": "benchmark_return_nanhua",
                        "benchmark_role": "broad_commodity_index",
                    },
                ],
                "same_horizon_controls": [
                    {
                        "benchmark_label": "Futures active-universe equal-weight same-horizon control",
                        "benchmark_column": "benchmark_return_same_horizon",
                        "benchmark_role": "same_horizon_universe_control",
                        "return_mode": "same_horizon",
                    }
                ],
            }
        }

        self.assertEqual(view._benchmark_title("run_1", {"benchmark_eq": "#### 🟡 Benchmark"}), "#### 🟡 Benchmark")
        entries = view._benchmark_guide_entries(manifest, returns)

        self.assertEqual([entry["slot_key"] for entry in entries], ["primary", "secondary", "secondary", "control"])
        self.assertEqual(
            [entry["column"] for entry in entries],
            [
                "benchmark_return",
                "benchmark_return_passive_active_universe",
                "benchmark_return_nanhua",
                "benchmark_return_same_horizon",
            ],
        )
        self.assertEqual(entries[0]["return_mode"], "same_horizon")

        from oqp.ui.translations import RESEARCH_TEXT

        self.assertTrue(RESEARCH_TEXT["en"]["strategy_net"].startswith("#### "))
        self.assertTrue(RESEARCH_TEXT["en"]["benchmark_eq"].startswith("#### "))

    def test_tearsheet_metric_labels_cover_performance_and_sample_quality(self) -> None:
        app_path = str(RESEARCH_APP)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)
        view_module = importlib.import_module("views.tearsheet_view")
        view = view_module.TearSheetView

        qualities = [
            view._classify_annual_return(0.0327, 0.0279),
            view._classify_volatility(0.0260, 0.0789),
            view._classify_sharpe(1.26),
            view._classify_drawdown(-0.0233),
            view._classify_calmar(1.40),
            view._classify_turnover(0.674),
            view._classify_holdout_ic(0.0009),
            view._classify_trade_count(313),
            view._classify_record_count(978),
            view._classify_signal_rows(11_642),
        ]

        self.assertEqual(
            [quality.label_key for quality in qualities],
            [
                "ahead",
                "lower_risk",
                "good",
                "controlled",
                "good",
                "very_high",
                "weak",
                "broad",
                "recorded",
                "broad",
            ],
        )

        from oqp.ui.translations import RESEARCH_TEXT

        for language in ("en", "zh"):
            labels = RESEARCH_TEXT[language]["tearsheet_quality_labels"]
            help_text = RESEARCH_TEXT[language]["tearsheet_quality_help"]
            for quality in qualities:
                self.assertIn(quality.label_key, labels)
                self.assertIn(quality.help_key, help_text)

    def test_sidebar_global_controls_do_not_render_taxonomy_toggle(self) -> None:
        ui_state = (RESEARCH_APP / "ui_state.py").read_text()

        self.assertNotIn("_render_asset_taxonomy_sidebar", ui_state)
        self.assertNotIn("st.sidebar.expander", ui_state)
        self.assertNotIn("dashboard_taxonomy_frame", ui_state)

    def test_arbitrage_radar_uses_full_universe_without_extra_filters(self) -> None:
        arbitrage_page = (RESEARCH_APP / "discovery" / "relationship_page.py").read_text()

        self.assertIn("full_universe_note", arbitrage_page)
        self.assertNotIn("_filtered_candidates", arbitrage_page)
        self.assertNotIn("Filters (default: full universe)", arbitrage_page)
        self.assertNotIn("default=type_options", arbitrage_page)
        self.assertNotIn("default=sector_options[:8]", arbitrage_page)
        self.assertIn('t["workspace_mode"]', arbitrage_page)
        self.assertIn("dynamic_relationship", arbitrage_page)
        self.assertIn('t["map_view"]', arbitrage_page)
        self.assertIn("schema", arbitrage_page)
        self.assertNotIn('"Opportunity family"', arbitrage_page)
        self.assertNotIn('"No cross-product candidates passed the current filters."', arbitrage_page)
        self.assertNotIn('metric("Rows"', arbitrage_page)

    def test_regime_top_controls_are_aligned_selectboxes(self) -> None:
        regime_page = (RESEARCH_APP / "pages" / "05_Regime_Analysis.py").read_text()

        self.assertIn(
            'selector_left, selector_right = st.columns(2, vertical_alignment="bottom")',
            regime_page,
        )
        self.assertIn('return st.selectbox(\n        lt["asset_class"]', regime_page)
        self.assertIn('selected_ticker = st.selectbox(lt["asset"]', regime_page)
        self.assertNotIn("st.radio(", regime_page)
        self.assertNotIn("render_page_workflow()", regime_page)
        self.assertNotIn("_render_lane_summary", regime_page)
        self.assertNotIn("scope_help_title", regime_page)
        self.assertNotIn("taxonomy_help_title", regime_page)

    def test_risk_breadth_top_section_stays_compact(self) -> None:
        risk_page = (RESEARCH_APP / "pages" / "06_Market_Breadth_Lab.py").read_text()

        self.assertIn("label = path.name", risk_page)
        self.assertIn("load_asset_taxonomy", risk_page)
        self.assertIn("is_vectorizable_asset_class", risk_page)
        self.assertIn("discover_asset_class_files", risk_page)
        self.assertIn('InstrumentMaster("FUTURES_CN").get_sector_map()', risk_page)
        self.assertIn('asset_class=asset_class', risk_page)
        self.assertIn('max_assets=max_assets', risk_page)
        self.assertIn("pc1_top_sectors", risk_page)
        self.assertIn('st.markdown(f"### {t[\'cards\']}")', risk_page)
        self.assertNotIn('st.info(t["feature_link"])', risk_page)
        self.assertNotIn('st.expander(t["manual_title"]', risk_page)
        self.assertNotIn('st.expander(t["metric_toggle"]', risk_page)
        self.assertNotIn('st.expander(t["pc_label_toggle"]', risk_page)
        self.assertNotIn('st.metric(\n            t["pc_label_prefix"]', risk_page)

    def test_ml_hub_scope_and_lifecycle_navigation(self) -> None:
        feature_page = (RESEARCH_APP / "pages" / "07_ML_Hub.py").read_text()
        manual_block = 'with st.expander(t["manual_title"], expanded=False):\n    st.markdown(t["manual"])'

        metric_index = feature_page.index('metric_cols[3].metric(t["dates"], date_label)')
        manual_index = feature_page.index(manual_block)
        tabs_index = feature_page.index("tabs = st.tabs(")

        self.assertGreater(manual_index, metric_index)
        self.assertLess(manual_index, tabs_index)
        self.assertIn("infer_feature_matrix_asset_class", feature_page)
        self.assertIn("observed_feature_matrix_asset_classes", feature_page)
        self.assertIn("scope_feature_matrix", feature_page)
        self.assertIn('key=f"feature_review_asset_class_{lang.lower()}"', feature_page)
        self.assertIn(
            "metric_cols = st.columns([0.18, 0.18, 0.16, 0.48])",
            feature_page,
        )
        for translation_key in (
            't["tab_data"]',
            't["tab_evidence"]',
            't["tab_models"]',
            't["tab_explainability"]',
            't["tab_registry"]',
        ):
            self.assertIn(translation_key, feature_page)
        self.assertNotIn('t["tab_promotion"]', feature_page)
        self.assertIn("list_ml_experiments", feature_page)
        self.assertIn("list_model_artifacts", feature_page)
        self.assertIn("probe_model_runtime", feature_page)
        self.assertIn("render_model_catalog_panel", feature_page)
        self.assertNotIn('st.checkbox(t["include_prob"]', feature_page)
        self.assertIn("include_prob_features=False", feature_page)
        data_tab_index = feature_page.index('t["tab_data"]', tabs_index)
        model_tab_index = feature_page.index('t["tab_models"]', tabs_index)
        evidence_tab_index = feature_page.index('t["tab_evidence"]', tabs_index)
        self.assertLess(data_tab_index, model_tab_index)
        self.assertLess(model_tab_index, evidence_tab_index)
        self.assertIn("scripts/research/run_ml_backtest.py", feature_page)

    def test_strategy_comparison_correlation_matrix_uses_compact_labels(self) -> None:
        logic = (RESEARCH_APP / "executive_dashboard_logic.py").read_text()

        self.assertIn('table_left, table_right = st.columns([0.5, 0.5])', logic)
        self.assertIn("def _short_profile_labels", logic)
        self.assertIn('label_by_profile[profile.label] = f"P{idx} {group}"', logic)
        self.assertIn("go.Heatmap", logic)
        self.assertIn("texttemplate", logic)
        self.assertIn("customdata=hover_names", logic)
        self.assertIn("hovertemplate", logic)
        self.assertNotIn("px.imshow", logic)

    def test_data_health_header_and_overview_stay_compact(self) -> None:
        health_view = (RESEARCH_APP / "views" / "system_health_view.py").read_text()

        self.assertIn("def _render_page_header", health_view)
        self.assertIn('st.columns([0.76, 0.24], vertical_alignment="center")', health_view)
        self.assertIn(
            'st.button(copy["refresh"], use_container_width=True)',
            health_view,
        )
        self.assertNotIn('with st.expander("How to use / 使用说明"', health_view)
        self.assertNotIn('cols[4].metric(copy["latest_run"]', health_view)
        self.assertIn("def _render_latest_run", health_view)
        self.assertIn("def _render_api_readiness", health_view)
        self.assertIn("def _api_provider_snapshot", health_view)
        self.assertIn("def _render_data_folder_overview", health_view)
        self.assertIn("def _render_freshness_overview", health_view)
        self.assertIn("def _folder_freshness_summary", health_view)
        self.assertIn("def _folder_risk_imputation_summary", health_view)
        self.assertIn("def _cap_risk_lens_frame", health_view)
        self.assertIn("def _read_quality_parquet_sample", health_view)
        self.assertIn("def _runtime_data_folder_snapshot", health_view)
        self.assertIn('"api_readiness": api_readiness_df', health_view)
        self.assertIn('"api_readiness": "API Readiness"', health_view)
        self.assertIn('"freshness_overview": "Market Data Freshness"', health_view)
        self.assertIn('"risk_lens_title": "Risk Imputation Lens"', health_view)
        self.assertIn('"provider_api": "Provider / API"', health_view)
        self.assertIn('"data_folder_overview": "Runtime Data Folder Coverage"', health_view)
        self.assertIn('"provider": "FMP"', health_view)
        self.assertIn('"provider": "Massive"', health_view)
        self.assertIn('"provider": "Yahoo Finance"', health_view)
        self.assertIn('display_cols = [\n            "provider",\n            "asset_class",\n            "status"', health_view)
        self.assertIn('copy["latest_update"]: st.column_config.TextColumn(width="medium")', health_view)
        self.assertIn('copy["date_range"]: st.column_config.TextColumn(width="large")', health_view)
        self.assertIn("def _whole_number_label", health_view)
        self.assertIn("compare_risk_imputation_views", health_view)
        self.assertIn('"bridge_rv_median"', health_view)
        self.assertIn('"ffill_zero_return_pct"', health_view)
        self.assertNotIn('copy["runtime"],', health_view)
        self.assertIn('asset_class="EQUITY_CN"', health_view)
        self.assertIn('asset_class="OPTIONS_CN"', health_view)

    def test_assumptions_view_surfaces_data_health_contract(self) -> None:
        assumptions_view = (RESEARCH_APP / "views" / "assumptions_view.py").read_text()

        self.assertIn("def _render_data_health_assumptions", assumptions_view)
        self.assertIn("SystemHealthView._load_snapshot", assumptions_view)
        self.assertIn("assumptions_data_health", assumptions_view)
        self.assertIn('"fill_policy"', assumptions_view)

    def test_research_dashboard_uses_supported_streamlit_width_api(self) -> None:
        for path in RESEARCH_APP.rglob("*.py"):
            with self.subTest(path=path.relative_to(RESEARCH_APP)):
                self.assertNotIn('width="stretch"', path.read_text())

    def test_remaining_research_page_dependencies_import(self) -> None:
        app_path = str(RESEARCH_APP)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)

        modules = [
            "arbitrage_lab.charts",
            "arbitrage_lab.components",
            "arbitrage_lab.text",
            "executive_dashboard_logic",
            "views.factor_promotion_view",
            "views.system_health_view",
            "oqp.research.ml",
            "oqp.research.ml.latent",
            "oqp.research.ml.state_space",
            "departments.research.workflows.latent",
            "departments.research.workflows.statistical_arbitrage",
            "apps.research_dashboard.services.latent_artifacts",
            "oqp.risk.factor_breadth",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_arbitrage_candidate_table_translates_display_layer(self) -> None:
        app_path = str(RESEARCH_APP)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)
        components = importlib.import_module("arbitrage_lab.components")

        raw = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "candidate_id": "au2608 ~ ag2608",
                    "arbitrage_type": "cross_product",
                    "sector_pair": "贵金属 / 贵金属",
                    "opportunity_score": 71.2,
                    "latest_z": 2.1,
                    "correlation": 0.82,
                    "half_life": 8.4,
                    "beta_drift": 0.12,
                    "round_turn_cost_bps": 2.5,
                    "interpretation": "Worth reviewing: some evidence, needs drilldown and cost checks",
                }
            ]
        )

        display = components.candidate_display_table(raw, lang="ZH")

        self.assertIn("类型代码", display.columns)
        self.assertIn("类型", display.columns)
        self.assertIn("解读", display.columns)
        self.assertEqual(display.iloc[0]["类型代码"], "cross_product")
        self.assertEqual(display.iloc[0]["类型"], "跨品种")
        self.assertIn("值得复核", display.iloc[0]["解读"])


if __name__ == "__main__":
    unittest.main()
