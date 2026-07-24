from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEFAULT_SLEEVE_ID = "slv_001_Cross_Sectional_Quintile_Long_Short"
LEGACY_QUINTILE_SLEEVE_ID = "slv_004_Default_Cross_Sectional_Long_Short"


COPY = {
    "EN": {
        "title": "Conditional Behaviour (Phase 5)",
        "boundary": (
            "Observable conditions describe where a frozen sleeve behaves "
            "differently. This is exploratory evidence, not a router backtest."
        ),
        "condition": "Observable condition",
        "scope": "Scope",
        "timing": "Timing",
        "timing_value": "Known after close t; sleeve return begins at open t+1",
        "market": "One market condition is assigned to the whole sleeve date.",
        "contract": (
            "Each contract contribution is assigned using that contract's own "
            "condition, then reconciled to total sleeve P&L."
        ),
        "validation": "Validation",
        "holdout": "Holdout",
        "full": "Full sample",
        "chart": "Annualised net mean with 95% HAC interval",
        "table": "Bucket evidence",
        "small_sample": (
            "At least one bucket has fewer than 30 dates or active dates. Treat "
            "its annualised return and Sharpe as unstable."
        ),
        "confidence": (
            "Intervals use Newey-West standard errors with five lags and are not "
            "adjusted for searching across conditions or buckets."
        ),
    },
    "ZH": {
        "title": "条件行为（第五阶段）",
        "boundary": (
            "可观察条件只用于描述冻结策略腿在什么环境下表现不同。这里是探索性证据，"
            "不是路由回测。"
        ),
        "condition": "可观察条件",
        "scope": "作用范围",
        "timing": "时间口径",
        "timing_value": "条件在 t 日收盘后已知；策略收益从 t+1 日开盘开始",
        "market": "同一天的完整策略腿使用一个市场条件。",
        "contract": "每个合约贡献按自身条件归类，再核对汇总结果与策略腿总盈亏一致。",
        "validation": "验证集",
        "holdout": "留出集",
        "full": "全样本",
        "chart": "年化净收益均值与 95% HAC 区间",
        "table": "分组证据",
        "small_sample": (
            "至少一个分组少于 30 个观测日或活跃日，其年化收益与夏普不稳定。"
        ),
        "confidence": (
            "区间使用五阶 Newey-West 标准误，尚未对多个条件及分组搜索进行校正。"
        ),
    },
}


@st.cache_data(show_spinner=False)
def load_conditional_behaviour_snapshot(
    artifact_root: str,
    factor_id: str,
    market_vertical: str,
    sleeve_id: str = DEFAULT_SLEEVE_ID,
) -> dict[str, Any] | None:
    root = Path(artifact_root).expanduser().resolve()
    candidate_ids = [sleeve_id]
    if sleeve_id == DEFAULT_SLEEVE_ID:
        candidate_ids.append(LEGACY_QUINTILE_SLEEVE_ID)
    source = None
    for candidate_id in candidate_ids:
        candidate = (
            root
            / "conditional_behaviour"
            / factor_id
            / market_vertical
            / candidate_id
        ).resolve()
        if candidate.is_relative_to(root) and candidate.is_dir():
            source = candidate
            break
    if source is None:
        return None
    required = {
        "summary.json",
        "manifest.json",
        "definitions.json",
        "bucket_metrics.csv",
    }
    if not required.issubset(path.name for path in source.iterdir()):
        return None
    return {
        "summary": json.loads((source / "summary.json").read_text(encoding="utf-8")),
        "manifest": json.loads((source / "manifest.json").read_text(encoding="utf-8")),
        "definitions": json.loads(
            (source / "definitions.json").read_text(encoding="utf-8")
        ),
        "metrics": pd.read_csv(source / "bucket_metrics.csv"),
    }


def render_conditional_behaviour_panel(
    artifact_root: str | Path,
    factor_id: str,
    market_vertical: str,
    *,
    lang: str = "EN",
    plotly_template: str = "plotly_white",
) -> bool:
    snapshot = load_conditional_behaviour_snapshot(
        str(artifact_root), factor_id, market_vertical
    )
    if snapshot is None:
        return False
    copy = COPY.get(lang, COPY["EN"])
    definitions = {
        item["condition_id"]: item for item in snapshot["definitions"]
    }
    condition_ids = list(definitions)
    if not condition_ids:
        return False

    st.markdown(f"#### {copy['title']}")
    st.info(copy["boundary"])
    condition_id = st.selectbox(
        copy["condition"],
        condition_ids,
        format_func=lambda value: definitions[value]["display_name"],
        key=f"phase5_condition_{factor_id}_{market_vertical}",
    )
    definition = definitions[condition_id]
    st.caption(definition["formula"])
    meta = st.columns(2)
    meta[0].caption(copy["scope"])
    meta[0].markdown(f"**{str(definition['scope']).title()}**")
    meta[1].caption(copy["timing"])
    meta[1].markdown(f"**{copy['timing_value']}**")
    st.caption(copy[str(definition["scope"])])

    metrics = snapshot["metrics"].loc[
        snapshot["metrics"]["condition_id"].eq(condition_id)
    ].copy()
    tabs = st.tabs([copy["validation"], copy["holdout"], copy["full"]])
    for tab, split in zip(tabs, ("validation", "holdout", "full"), strict=True):
        with tab:
            sample = metrics.loc[metrics["research_split"].eq(split)].sort_values(
                "bucket_order"
            )
            if sample.empty:
                st.caption("No eligible observations.")
                continue
            _render_interval_chart(sample, copy, plotly_template)
            st.markdown(f"**{copy['table']}**")
            _render_metrics_table(sample, condition_id)
            if sample["date_count"].lt(30).any() or sample[
                "active_date_count"
            ].lt(30).any():
                st.warning(copy["small_sample"])
    st.caption(copy["confidence"])
    return True


def _render_interval_chart(
    sample: pd.DataFrame,
    copy: dict[str, str],
    template: str,
) -> None:
    mean = pd.to_numeric(sample["net_annualized_mean"], errors="coerce")
    lower = pd.to_numeric(
        sample["net_annualized_mean_ci_lower"], errors="coerce"
    )
    upper = pd.to_numeric(
        sample["net_annualized_mean_ci_upper"], errors="coerce"
    )
    colors = ["#0f766e" if value >= 0.0 else "#b91c1c" for value in mean]
    figure = go.Figure(
        go.Bar(
            x=sample["bucket"],
            y=mean,
            marker_color=colors,
            error_y={
                "type": "data",
                "array": (upper - mean).clip(lower=0.0),
                "arrayminus": (mean - lower).clip(lower=0.0),
                "color": "#111827",
                "thickness": 1.2,
            },
            customdata=sample[["date_count", "active_date_count"]],
            hovertemplate=(
                "%{x}<br>Net mean: %{y:.2%}<br>Dates: %{customdata[0]:,.0f}"
                "<br>Active dates: %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0.0, line_width=1, line_color="#64748b")
    figure.update_layout(
        template=template,
        title=copy["chart"],
        height=390,
        margin={"l": 20, "r": 20, "t": 55, "b": 35},
        yaxis={"tickformat": ".1%"},
        xaxis={"title": None},
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_metrics_table(sample: pd.DataFrame, condition_id: str) -> None:
    display = sample[
        [
            "bucket",
            "mean_condition_value",
            "date_count",
            "active_date_count",
            "net_annualized_mean",
            "net_annualized_mean_ci_lower",
            "net_annualized_mean_ci_upper",
            "net_sharpe",
            "annualized_turnover",
            "annualized_cost",
        ]
    ].copy()
    if condition_id == "shock_age":
        display["Condition mean"] = display["mean_condition_value"].map(
            lambda value: f"{value:.1f} sessions" if pd.notna(value) else "n/a"
        )
    else:
        display["Condition mean"] = display["mean_condition_value"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "n/a"
        )
    display["95% HAC interval"] = display.apply(
        lambda row: (
            f"[{row['net_annualized_mean_ci_lower']:.1%}, "
            f"{row['net_annualized_mean_ci_upper']:.1%}]"
        ),
        axis=1,
    )
    display = display[
        [
            "bucket",
            "Condition mean",
            "date_count",
            "active_date_count",
            "net_annualized_mean",
            "95% HAC interval",
            "net_sharpe",
            "annualized_turnover",
            "annualized_cost",
        ]
    ].rename(
        columns={
            "bucket": "Bucket",
            "date_count": "Dates",
            "active_date_count": "Active dates",
            "net_annualized_mean": "Net mean",
            "net_sharpe": "Net Sharpe",
            "annualized_turnover": "Turnover (x)",
            "annualized_cost": "Cost",
        }
    )
    st.dataframe(
        display.style.format(
            {
                "Dates": "{:,.0f}",
                "Active dates": "{:,.0f}",
                "Net mean": "{:.2%}",
                "Net Sharpe": "{:.2f}",
                "Turnover (x)": "{:.1f}",
                "Cost": "{:.2%}",
            },
            na_rep="",
        ),
        width="stretch",
        hide_index=True,
    )


__all__ = [
    "load_conditional_behaviour_snapshot",
    "render_conditional_behaviour_panel",
]
