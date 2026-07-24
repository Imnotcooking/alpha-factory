import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from config import ALPHA_RUNTIME_DATA_ROOT, BASE_DIR, TEXT, get_plotly_template

from oqp.research.oracle_evaluator import OracleEvaluator


STATE_LABELS = {
    0: "Trend",
    1: "Chop",
    2: "Panic",
}

AI_REGIME_LABELS = {
    0: "Quiet/Trend",
    1: "Chop",
    2: "Panic",
}

REGIME_SOLID_COLORS = {
    0: "#00C853",
    1: "#FFAB00",
    2: "#D50000",
}

REGIME_HEATMAP_SCALE = [
    [0.0, "#00C853"],
    [0.3332, "#00C853"],
    [0.3333, "#FFAB00"],
    [0.6665, "#FFAB00"],
    [0.6666, "#D50000"],
    [1.0, "#D50000"],
]


@st.cache_data(show_spinner=False)
def _load_oracle_result(
    matrix_path: str,
    probs_path: str,
    matrix_mtime: float,
    probs_mtime: float,
    tickers: tuple[str, ...],
):
    evaluator = OracleEvaluator(matrix_path=matrix_path, probs_path=probs_path)
    return evaluator.evaluate(tickers=tickers or None)


class OracleView:
    def __init__(self, data_manager):
        self.dm = data_manager

    @staticmethod
    def is_supported(run_metadata: pd.Series) -> bool:
        signature = " ".join(
            str(run_metadata.get(col, ""))
            for col in ["factor_id", "name", "category"]
        ).lower()
        return any(
            keyword in signature
            for keyword in ["gmm", "hmm", "regime", "fac_056", "fac_057", "ensemble"]
        )

    def render(self, run_metadata: pd.Series, lang: str = "EN", theme_mode: str = "DARK"):
        t = TEXT[lang] if lang in TEXT else TEXT["EN"]
        st.markdown(f"### {t.get('oracle_title', 'Oracle Regime Validation')}")
        st.caption(
            t.get(
                "oracle_caption",
                "Compares GMM/HMM regime probabilities against hindsight trend/chop/panic labels.",
            )
        )
        self._render_manual(t)

        if not self.is_supported(run_metadata):
            st.info(t.get("oracle_not_applicable", "Oracle validation is only shown for regime-aware factors."))
            return

        matrix_path = os.path.join(ALPHA_RUNTIME_DATA_ROOT, "feature_store", "ML_Feature_Matrix.parquet")
        probs_path = os.path.join(ALPHA_RUNTIME_DATA_ROOT, "regime", "GMM_Rolling_Probabilities.parquet")
        missing_paths = [path for path in [matrix_path, probs_path] if not os.path.exists(path)]
        if missing_paths:
            st.warning(
                t.get("oracle_missing", "Oracle inputs are missing.")
                + "\n\n"
                + "\n".join(f"- `{path}`" for path in missing_paths)
            )
            return

        traded_tickers = self._extract_traded_tickers(run_metadata)
        scope_label = t.get("oracle_scope_all", "All GMM assets")
        tickers: tuple[str, ...] = ()
        if traded_tickers:
            scope_label = st.radio(
                t.get("oracle_scope", "Evaluation universe"),
                [
                    t.get("oracle_scope_run", "Selected run traded universe"),
                    t.get("oracle_scope_all", "All GMM assets"),
                ],
                horizontal=True,
            )
            if scope_label == t.get("oracle_scope_run", "Selected run traded universe"):
                tickers = tuple(sorted(traded_tickers))

        try:
            with st.spinner(t.get("oracle_loading", "Running oracle regime validation...")):
                result = _load_oracle_result(
                    matrix_path,
                    probs_path,
                    os.path.getmtime(matrix_path),
                    os.path.getmtime(probs_path),
                    tickers,
                )
        except Exception as exc:
            st.error(f"{t.get('oracle_error', 'Oracle validation failed')}: {exc}")
            return

        self._render_metrics(result, t)
        self._render_help("oracle_metrics_help_title", "oracle_metrics_help", t)
        self._render_confusion_and_report(result, theme_mode, t)
        self._render_help("oracle_confusion_help_title", "oracle_confusion_help", t)
        self._render_probability_diagnostics(result.eval_df, theme_mode, t)
        self._render_help("oracle_probability_help_title", "oracle_probability_help", t)
        self._render_asset_drilldown(result.eval_df, theme_mode, t)
        self._render_help("oracle_asset_help_title", "oracle_asset_help", t)

        state_map_text = ", ".join(
            f"raw {raw_state} -> {label}" for raw_state, label in sorted(result.state_map.items())
        )
        st.info(f"{t.get('oracle_state_map', 'State alignment')}: {state_map_text}")

    @staticmethod
    def _render_manual(t: dict):
        with st.expander(t.get("oracle_manual_title", "How to use this page"), expanded=False):
            st.markdown(t.get("oracle_manual", ""))

    @staticmethod
    def _render_help(title_key: str, body_key: str, t: dict):
        body = t.get(body_key, "")
        if not body:
            return
        with st.expander(t.get(title_key, "How to read this"), expanded=False):
            st.markdown(body)

    @staticmethod
    def _extract_traded_tickers(run_metadata: pd.Series) -> list[str]:
        raw = str(run_metadata.get("traded_tickers", "")).strip()
        if not raw or raw.upper() in {"ALL", "NONE", "NAN", "<NA>"}:
            return []
        return [ticker.strip() for ticker in raw.split(",") if ticker.strip()]

    @staticmethod
    def _render_metrics(result, t: dict):
        eval_df = result.eval_df
        oracle_panic_rate = (eval_df["Oracle_State"] == 2).mean()
        ai_panic_rate = (eval_df["AI_Predicted_State"] == 2).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t.get("oracle_accuracy", "Oracle Accuracy"), f"{result.accuracy * 100:.1f}%")
        c2.metric(
            t.get("oracle_panic_auc", "Panic ROC-AUC"),
            "N/A" if result.panic_auc is None else f"{result.panic_auc:.3f}",
        )
        c3.metric(t.get("oracle_samples", "Labeled Samples"), f"{len(eval_df):,}")
        c4.metric(
            t.get("oracle_panic_rate", "Oracle Panic Rate"),
            f"{oracle_panic_rate * 100:.1f}% / AI {ai_panic_rate * 100:.1f}%",
        )

    @staticmethod
    def _render_confusion_and_report(result, theme_mode: str, t: dict):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            value_mode = st.radio(
                t.get("oracle_confusion_mode", "Confusion matrix values"),
                [
                    t.get("oracle_confusion_counts", "Counts"),
                    t.get("oracle_confusion_row_pct", "Row %"),
                ],
                horizontal=True,
            )
            display_matrix = result.confusion.astype(float)
            if value_mode == t.get("oracle_confusion_row_pct", "Row %"):
                row_sums = display_matrix.sum(axis=1).replace(0, np.nan)
                display_matrix = display_matrix.div(row_sums, axis=0).mul(100).fillna(0.0)
                text_matrix = display_matrix.round(1).astype(str) + "%"
                title = t.get("oracle_confusion_title_pct", "Confusion Matrix (Row %)")
            else:
                text_matrix = result.confusion.astype(int).astype(str)
                title = t.get("oracle_confusion_title_counts", "Confusion Matrix (Counts)")

            fig = go.Figure(
                data=go.Heatmap(
                    z=display_matrix.values,
                    x=result.confusion.columns,
                    y=result.confusion.index,
                    text=text_matrix.values,
                    texttemplate="%{text}",
                    hovertemplate="Oracle=%{y}<br>AI=%{x}<br>Value=%{text}<extra></extra>",
                    colorscale="Viridis",
                )
            )
            fig.update_layout(
                template=get_plotly_template(theme_mode),
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                title=title,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            report_df = result.report.copy()
            numeric_cols = report_df.select_dtypes(include="number").columns
            report_df[numeric_cols] = report_df[numeric_cols].round(3)
            st.markdown(f"#### {t.get('oracle_classification_report', 'Classification Report')}")
            st.dataframe(report_df, use_container_width=True)

    @staticmethod
    def _render_probability_diagnostics(eval_df: pd.DataFrame, theme_mode: str, t: dict):
        plot_df = eval_df.copy()
        plot_df["Oracle Label"] = plot_df["Oracle_State"].map(STATE_LABELS)
        plot_df["AI Label"] = plot_df["AI_Predicted_State"].map(STATE_LABELS)

        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                plot_df,
                x="Oracle Label",
                y="prob_Panic",
                color="Oracle Label",
                category_orders={"Oracle Label": ["Trend", "Chop", "Panic"]},
                title=t.get("oracle_panic_box", "AI Panic Probability by Oracle State"),
            )
            fig_box.update_layout(
                template=get_plotly_template(theme_mode),
                height=360,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            state_counts = (
                plot_df.groupby(["Oracle Label", "AI Label"])
                .size()
                .reset_index(name="count")
            )
            fig_bar = px.bar(
                state_counts,
                x="Oracle Label",
                y="count",
                color="AI Label",
                category_orders={"Oracle Label": ["Trend", "Chop", "Panic"]},
                title=t.get("oracle_state_mix", "Predicted State Mix by Oracle Label"),
            )
            fig_bar.update_layout(
                template=get_plotly_template(theme_mode),
                height=360,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    @staticmethod
    def _render_asset_drilldown(eval_df: pd.DataFrame, theme_mode: str, t: dict):
        tickers = sorted(eval_df["ticker"].dropna().unique())
        if not tickers:
            return

        selected_ticker = st.selectbox(
            t.get("oracle_select_asset", "Inspect asset"),
            tickers,
            index=0,
        )
        asset_df = eval_df[eval_df["ticker"] == selected_ticker].sort_values("date").copy()
        asset_df["date"] = pd.to_datetime(asset_df["date"])
        asset_df["close"] = pd.to_numeric(asset_df["close"], errors="coerce")
        asset_df = asset_df.dropna(subset=["date", "close", "AI_Predicted_State"])
        if asset_df.empty:
            st.warning(t.get("oracle_asset_no_data", "No usable price/regime rows for this asset."))
            return

        asset_accuracy = (asset_df["Oracle_State"] == asset_df["AI_Predicted_State"]).mean()
        asset_panic = (asset_df["Oracle_State"] == 2).astype(int)
        asset_panic_auc = None
        if asset_panic.nunique() >= 2:
            asset_panic_auc = roc_auc_score(asset_panic, asset_df["prob_Panic"])
        oracle_panic_rate = asset_panic.mean()
        ai_panic_rate = (asset_df["AI_Predicted_State"] == 2).mean()

        st.caption(t.get("oracle_asset_scope_note", "These asset metrics update when you switch the dropdown; the top metrics stay at the selected universe scope."))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t.get("oracle_asset_accuracy", "Asset Accuracy"), f"{asset_accuracy * 100:.1f}%")
        c2.metric(
            t.get("oracle_asset_panic_auc", "Asset Panic ROC-AUC"),
            "N/A" if asset_panic_auc is None else f"{asset_panic_auc:.3f}",
        )
        c3.metric(t.get("oracle_asset_samples", "Asset Samples"), f"{len(asset_df):,}")
        c4.metric(
            t.get("oracle_asset_panic_rate", "Asset Panic Rate"),
            f"{oracle_panic_rate * 100:.1f}% / AI {ai_panic_rate * 100:.1f}%",
        )

        asset_df["AI_Regime"] = asset_df["AI_Predicted_State"].map(AI_REGIME_LABELS)
        asset_df["Oracle_Label"] = asset_df["Oracle_State"].map(STATE_LABELS)

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=asset_df["date"],
                y=[0.5],
                z=[asset_df["AI_Predicted_State"].astype(float).to_numpy()],
                colorscale=REGIME_HEATMAP_SCALE,
                zmin=0,
                zmax=2,
                opacity=0.24,
                showscale=False,
                hoverinfo="skip",
                yaxis="y2",
                name="AI regime background",
            )
        )

        price_line_color = "#F5F5F5" if str(theme_mode).upper() == "DARK" else "#263238"

        for state, label in AI_REGIME_LABELS.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=REGIME_SOLID_COLORS[state]),
                    name=f"AI {label} region",
                    showlegend=True,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=asset_df["date"],
                y=asset_df["close"],
                mode="lines",
                name="Close",
                line=dict(color=price_line_color, width=2.0),
                customdata=asset_df[
                    ["AI_Regime", "Oracle_Label", "prob_Quiet", "prob_Chop", "prob_Panic"]
                ],
                hovertemplate=(
                    "Date=%{x}<br>"
                    "Close=%{y:.3f}<br>"
                    "AI regime=%{customdata[0]}<br>"
                    "Oracle=%{customdata[1]}<br>"
                    "Quiet=%{customdata[2]:.1%}<br>"
                    "Chop=%{customdata[3]:.1%}<br>"
                    "Panic=%{customdata[4]:.1%}<extra></extra>"
                ),
            )
        )

        panic_days = asset_df[asset_df["Oracle_State"] == 2]
        if not panic_days.empty:
            fig.add_trace(
                go.Scatter(
                    x=panic_days["date"],
                    y=panic_days["close"],
                    mode="markers",
                    name="Oracle Panic",
                    marker=dict(color="#FF1744", size=6, symbol="x"),
                ),
            )

        fig.update_layout(
            template=get_plotly_template(theme_mode),
            height=460,
            margin=dict(l=10, r=10, t=40, b=10),
            title=t.get("oracle_asset_drilldown", "Asset Probability Drilldown"),
            yaxis2=dict(
                overlaying="y",
                range=[0, 1],
                visible=False,
                showgrid=False,
                zeroline=False,
            ),
        )
        fig.update_yaxes(title_text="Close")
        st.plotly_chart(fig, use_container_width=True)
