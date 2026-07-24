from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


COPY = {
    "EN": {
        "title": "Predictive Evidence (Phase 2)",
        "boundary": "IC measures prediction before transaction costs. It does not prove that a portfolio translation is profitable.",
        "alignment": "Executable-return alignment",
        "mean_ic": "Mean IC",
        "mean_rank_ic": "Mean Rank IC",
        "pearson_icir": "Pearson ICIR",
        "rank_icir": "Rank ICIR",
        "hit_rate": "Rank IC hit rate",
        "sample": "Dates",
        "sample_note": "Full sample: {dates} dates and {products} products.",
        "split_title": "Validation and holdout",
        "time_tab": "Time profile",
        "product_tab": "Products",
        "coverage_tab": "Coverage and concentration",
        "rolling": "Rolling IC",
        "cumulative": "Cumulative IC",
        "pearson": "Pearson IC",
        "rank": "Rank IC",
        "product_distribution": "Product-level Rank IC distribution",
        "year_profile": "Evidence by year",
        "concentration": "Concentration diagnostics",
        "ic_display": "IC values are displayed as correlation percentages; ICIR is unannualized mean divided by sample standard deviation.",
    },
    "ZH": {
        "title": "预测证据（第二阶段）",
        "boundary": "IC 衡量交易成本前的预测关系；它不能证明组合转换后能够盈利。",
        "alignment": "可执行收益对齐",
        "mean_ic": "平均 IC",
        "mean_rank_ic": "平均 Rank IC",
        "pearson_icir": "Pearson ICIR",
        "rank_icir": "Rank ICIR",
        "hit_rate": "Rank IC 命中率",
        "sample": "日期数",
        "sample_note": "全样本：{dates} 个日期、{products} 个品种。",
        "split_title": "验证集与留出集",
        "time_tab": "时间路径",
        "product_tab": "品种分布",
        "coverage_tab": "覆盖率与集中度",
        "rolling": "滚动 IC",
        "cumulative": "累计 IC",
        "pearson": "Pearson IC",
        "rank": "Rank IC",
        "product_distribution": "品种层面 Rank IC 分布",
        "year_profile": "年度证据",
        "concentration": "集中度诊断",
        "ic_display": "IC 以相关系数百分比显示；ICIR 为未经年化的 IC 均值除以样本标准差。",
    },
}


@st.cache_data(show_spinner=False)
def load_predictive_evidence_snapshot(
    artifact_root: str,
    factor_id: str,
    market_vertical: str,
) -> dict[str, Any] | None:
    root = Path(artifact_root).expanduser().resolve()
    source = (root / "predictive_evidence" / factor_id / market_vertical).resolve()
    try:
        source.relative_to(root)
    except ValueError:
        return None
    required = {
        "summary.json",
        "manifest.json",
        "period_ic.parquet",
        "split_summary.csv",
        "product_ic.csv",
        "yearly_summary.csv",
        "concentration.csv",
    }
    if not source.is_dir() or not required.issubset(
        path.name for path in source.iterdir()
    ):
        return None
    period_ic = pd.read_parquet(source / "period_ic.parquet")
    if "date" in period_ic.columns:
        period_ic["date"] = pd.to_datetime(period_ic["date"], errors="coerce")
    return {
        "source": str(source),
        "summary": json.loads((source / "summary.json").read_text(encoding="utf-8")),
        "manifest": json.loads((source / "manifest.json").read_text(encoding="utf-8")),
        "period_ic": period_ic,
        "split_summary": pd.read_csv(source / "split_summary.csv"),
        "product_ic": pd.read_csv(source / "product_ic.csv"),
        "yearly_summary": pd.read_csv(source / "yearly_summary.csv"),
        "concentration": pd.read_csv(source / "concentration.csv"),
    }


def render_predictive_evidence_panel(
    artifact_root: str | Path,
    factor_id: str,
    market_vertical: str,
    *,
    lang: str = "EN",
    plotly_template: str = "plotly_white",
) -> bool:
    snapshot = load_predictive_evidence_snapshot(
        str(artifact_root), factor_id, market_vertical
    )
    if snapshot is None:
        return False

    copy = COPY.get(lang, COPY["EN"])
    summary = snapshot["summary"]
    split = snapshot["split_summary"]
    full_rows = split.loc[split["research_split"].eq("full")]
    if full_rows.empty:
        return False
    full = full_rows.iloc[0]

    st.markdown(f"#### {copy['title']}")
    st.caption(copy["boundary"])
    alignment = snapshot["manifest"].get("causal_alignment") or {}
    st.caption(
        f"{copy['alignment']}: {alignment.get('execution_lag', '')} | "
        f"{alignment.get('return_assumption', '')}"
    )

    cards = st.columns(6)
    cards[0].metric(copy["mean_ic"], _format_ic(full["mean_pearson_ic"]))
    cards[1].metric(copy["mean_rank_ic"], _format_ic(full["mean_rank_ic"]))
    cards[2].metric(copy["pearson_icir"], _format_ratio(full["pearson_icir"]))
    cards[3].metric(copy["rank_icir"], _format_ratio(full["rank_icir"]))
    cards[4].metric(copy["hit_rate"], _format_percent(full["rank_ic_hit_rate"]))
    cards[5].metric(
        copy["sample"],
        f"{int(full['date_count']):,}",
    )
    st.caption(
        copy["ic_display"]
        + " "
        + copy["sample_note"].format(
            dates=f"{int(full['date_count']):,}",
            products=f"{int(full['product_count']):,}",
        )
    )

    st.markdown(f"**{copy['split_title']}**")
    split_display = split[
        [
            "research_split",
            "mean_pearson_ic",
            "mean_rank_ic",
            "pearson_icir",
            "rank_icir",
            "pearson_ic_hit_rate",
            "rank_ic_hit_rate",
            "date_count",
            "product_count",
        ]
    ].copy()
    split_display = split_display.rename(
        columns={
            "research_split": "Split",
            "mean_pearson_ic": "IC",
            "mean_rank_ic": "Rank IC",
            "pearson_icir": "P ICIR",
            "rank_icir": "R ICIR",
            "pearson_ic_hit_rate": "P hit",
            "rank_ic_hit_rate": "R hit",
            "date_count": "Dates",
            "product_count": "Products",
        }
    )
    split_display["Split"] = pd.Categorical(
        split_display["Split"],
        categories=["validation", "holdout", "full"],
        ordered=True,
    )
    split_display = split_display.sort_values("Split")
    st.dataframe(
        split_display.style.format(
            {
                "IC": "{:.2%}",
                "Rank IC": "{:.2%}",
                "P ICIR": "{:.3f}",
                "R ICIR": "{:.3f}",
                "P hit": "{:.1%}",
                "R hit": "{:.1%}",
                "Dates": "{:,.0f}",
                "Products": "{:,.0f}",
            },
            na_rep="",
        ),
        width="stretch",
        hide_index=True,
    )

    time_tab, product_tab, coverage_tab = st.tabs(
        [copy["time_tab"], copy["product_tab"], copy["coverage_tab"]]
    )
    with time_tab:
        _render_time_profile(snapshot["period_ic"], copy, plotly_template)
        st.markdown(f"**{copy['year_profile']}**")
        yearly = snapshot["yearly_summary"].copy()
        st.dataframe(
            yearly.style.format(
                {
                    "mean_pearson_ic": "{:.2%}",
                    "mean_rank_ic": "{:.2%}",
                    "pearson_icir": "{:.3f}",
                    "rank_icir": "{:.3f}",
                    "pearson_ic_hit_rate": "{:.1%}",
                    "rank_ic_hit_rate": "{:.1%}",
                    "mean_joint_coverage": "{:.1%}",
                },
                na_rep="",
            ),
            width="stretch",
            hide_index=True,
        )
    with product_tab:
        _render_product_profile(snapshot["product_ic"], copy, plotly_template)
    with coverage_tab:
        _render_coverage(split, snapshot["concentration"], copy)
    return True


def _render_time_profile(period_ic: pd.DataFrame, copy: dict, template: str) -> None:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(copy["rolling"], copy["cumulative"]),
    )
    colors = {"pearson": "#2563eb", "rank": "#dc2626"}
    for key, label in (("pearson", copy["pearson"]), ("rank", copy["rank"])):
        figure.add_trace(
            go.Scatter(
                x=period_ic["date"],
                y=period_ic[f"rolling_{key}_ic"],
                name=label,
                line={"color": colors[key], "width": 2},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=period_ic["date"],
                y=period_ic[f"cumulative_{key}_ic"],
                name=label,
                line={"color": colors[key], "width": 2},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8", row=1, col=1)
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8", row=2, col=1)
    figure.update_yaxes(tickformat=".1%", row=1, col=1)
    figure.update_layout(
        template=template,
        height=540,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_product_profile(product_ic: pd.DataFrame, copy: dict, template: str) -> None:
    products = product_ic.loc[product_ic["research_split"].eq("full")].copy()
    values = pd.to_numeric(products["oriented_rank_ic"], errors="coerce").dropna()
    figure = go.Figure(
        go.Histogram(x=values, marker_color="#0f766e", nbinsx=20)
    )
    figure.add_vline(x=0, line_width=1, line_color="#64748b")
    figure.update_layout(
        template=template,
        title=copy["product_distribution"],
        height=340,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis={"tickformat": ".1%", "title": copy["rank"]},
        yaxis={"title": "Products"},
        bargap=0.08,
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    columns = [
        "product",
        "oriented_pearson_ic",
        "oriented_rank_ic",
        "valid_pairs",
        "signal_coverage",
        "forward_return_coverage",
    ]
    product_display = products[columns].rename(
        columns={
            "product": "Product",
            "oriented_pearson_ic": "Oriented IC",
            "oriented_rank_ic": "Oriented Rank IC",
            "valid_pairs": "Pairs",
            "signal_coverage": "Signal coverage",
            "forward_return_coverage": "Return coverage",
        }
    )
    st.dataframe(
        product_display
        .sort_values("Oriented Rank IC", ascending=False)
        .style.format(
            {
                "Oriented IC": "{:.2%}",
                "Oriented Rank IC": "{:.2%}",
                "Pairs": "{:,.0f}",
                "Signal coverage": "{:.1%}",
                "Return coverage": "{:.1%}",
            },
            na_rep="",
        ),
        width="stretch",
        hide_index=True,
        height=420,
    )


def _render_coverage(
    split_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    copy: dict,
) -> None:
    coverage_columns = [
        "research_split",
        "row_count",
        "date_count",
        "product_count",
        "signal_coverage",
        "active_signal_coverage",
        "forward_return_coverage",
        "joint_coverage",
    ]
    coverage_display = split_summary[coverage_columns].rename(
        columns={
            "research_split": "Split",
            "row_count": "Rows",
            "date_count": "Dates",
            "product_count": "Products",
            "signal_coverage": "Signal",
            "active_signal_coverage": "Active signal",
            "forward_return_coverage": "Return",
            "joint_coverage": "Joint",
        }
    )
    st.dataframe(
        coverage_display.style.format(
            {
                "Rows": "{:,.0f}",
                "Dates": "{:,.0f}",
                "Products": "{:,.0f}",
                "Signal": "{:.1%}",
                "Active signal": "{:.1%}",
                "Return": "{:.1%}",
                "Joint": "{:.1%}",
            },
            na_rep="",
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown(f"**{copy['concentration']}**")
    concentration_display = concentration.sort_values(
        ["dimension", "absolute_rank_ic_mass_share"],
        ascending=[True, False],
    )
    st.dataframe(
        concentration_display.style.format(
            {
                "oriented_rank_ic": "{:.2%}",
                "evidence_count": "{:,.0f}",
                "signed_rank_ic_mass": "{:.3f}",
                "absolute_rank_ic_mass_share": "{:.1%}",
                "observation_share": "{:.1%}",
            },
            na_rep="",
        ),
        width="stretch",
        hide_index=True,
        height=420,
    )


def _format_ic(value: Any) -> str:
    return f"{float(value):.2%}" if pd.notna(value) else "n/a"


def _format_ratio(value: Any) -> str:
    return f"{float(value):.3f}" if pd.notna(value) else "n/a"


def _format_percent(value: Any) -> str:
    return f"{float(value):.1%}" if pd.notna(value) else "n/a"


__all__ = [
    "load_predictive_evidence_snapshot",
    "render_predictive_evidence_panel",
]
