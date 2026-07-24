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
        "title": "Standalone Sleeve Test (Phase 4)",
        "boundary": "The frozen sleeve is tested without routing or optimisation. Concentration and extreme-event results are diagnostics, not retrospective gates.",
        "blocked_validation": "Blocked before router research: one or more validation gates failed.",
        "blocked_holdout_confirmation": "Validation passed, but the frozen holdout confirmation failed. The sleeve remains blocked from router research.",
        "eligible_for_router_research": "Standalone validation and holdout confirmation passed. The sleeve may enter router research; this is not production approval.",
        "validation_sharpe": "Validation net Sharpe",
        "validation_return": "Validation net mean",
        "break_even": "Gross edge / cost",
        "active_days": "Validation active days",
        "holdout_sharpe": "Holdout net Sharpe",
        "hit_rate": "Full active-day hit rate",
        "gates": "Frozen gates",
        "standalone": "Standalone evidence",
        "concentration": "Contribution and concentration",
        "events": "Extreme events",
        "sample": "Sample comparison",
        "contribution": "Net contribution by year",
        "product_contribution": "Largest product contributions",
        "effective_products": "Effective products",
        "largest_position": "Largest position share",
        "top_product": "Top product contribution",
        "top_year": "Top year contribution",
        "event_path": "Mean sleeve return around broad market shocks",
        "event_windows": "Before, during and after broad shocks",
        "event_note": "Shock threshold: {threshold:.2%}, frozen at validation percentile {quantile:.1%}. Events: validation {validation}, holdout {holdout}.",
    },
    "ZH": {
        "title": "策略腿独立经济性检验（第四阶段）",
        "boundary": "冻结策略腿在没有路由与参数优化的条件下独立检验。集中度与极端事件结果仅作诊断，不作为事后新增门槛。",
        "blocked_validation": "暂不进入路由研究：一个或多个验证集门槛未通过。",
        "blocked_holdout_confirmation": "验证集通过，但冻结留出集确认失败；该策略腿仍不能进入路由研究。",
        "eligible_for_router_research": "独立验证及留出集确认均通过，可以进入路由研究；这不代表获准生产。",
        "validation_sharpe": "验证集净夏普",
        "validation_return": "验证集净年化均值",
        "break_even": "毛收益／成本",
        "active_days": "验证集活跃天数",
        "holdout_sharpe": "留出集净夏普",
        "hit_rate": "全样本活跃日胜率",
        "gates": "冻结门槛",
        "standalone": "独立经济性",
        "concentration": "贡献与集中度",
        "events": "极端事件",
        "sample": "样本对比",
        "contribution": "年度净贡献",
        "product_contribution": "主要品种贡献",
        "effective_products": "有效品种数",
        "largest_position": "最大单品种权重",
        "top_product": "最大品种贡献占比",
        "top_year": "最大年度贡献占比",
        "event_path": "广泛市场冲击前后的平均策略腿收益",
        "event_windows": "广泛冲击之前、当日及之后",
        "event_note": "冲击阈值：{threshold:.2%}，按验证集 {quantile:.1%} 分位冻结。事件数：验证集 {validation}，留出集 {holdout}。",
    },
}


@st.cache_data(show_spinner=False)
def load_standalone_sleeve_test_snapshot(
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
            / "standalone_sleeve_tests"
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
        "split_metrics.csv",
        "daily_diagnostics.parquet",
        "product_contribution.csv",
        "yearly_contribution.csv",
        "extreme_event_study.csv",
        "extreme_window_summary.csv",
        "gate_evaluation.csv",
    }
    if not required.issubset(path.name for path in source.iterdir()):
        return None
    daily = pd.read_parquet(source / "daily_diagnostics.parquet")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    return {
        "summary": json.loads((source / "summary.json").read_text(encoding="utf-8")),
        "manifest": json.loads((source / "manifest.json").read_text(encoding="utf-8")),
        "split": pd.read_csv(source / "split_metrics.csv"),
        "daily": daily,
        "products": pd.read_csv(source / "product_contribution.csv"),
        "years": pd.read_csv(source / "yearly_contribution.csv"),
        "event_study": pd.read_csv(source / "extreme_event_study.csv"),
        "event_windows": pd.read_csv(source / "extreme_window_summary.csv"),
        "gates": pd.read_csv(source / "gate_evaluation.csv"),
    }


def render_standalone_sleeve_test_panel(
    artifact_root: str | Path,
    factor_id: str,
    market_vertical: str,
    *,
    lang: str = "EN",
    plotly_template: str = "plotly_white",
) -> bool:
    snapshot = load_standalone_sleeve_test_snapshot(
        str(artifact_root), factor_id, market_vertical
    )
    if snapshot is None:
        return False
    copy = COPY.get(lang, COPY["EN"])
    summary = snapshot["summary"]
    split = snapshot["split"].set_index("research_split")
    if not {"full", "validation", "holdout"}.issubset(split.index):
        return False
    full = split.loc["full"]
    validation = split.loc["validation"]
    holdout = split.loc["holdout"]

    st.markdown(f"#### {copy['title']}")
    st.caption(copy["boundary"])
    validation_cards = st.columns(3)
    validation_cards[0].metric(
        copy["validation_sharpe"], _ratio(validation["net_sharpe"])
    )
    validation_cards[1].metric(
        copy["validation_return"], _percent(validation["net_annualized_mean"])
    )
    validation_cards[2].metric(
        copy["break_even"],
        f"{float(validation['break_even_cost_multiple']):.2f}x",
    )
    confirmation_cards = st.columns(3)
    confirmation_cards[0].metric(
        copy["active_days"], f"{int(validation['active_date_count']):,}"
    )
    confirmation_cards[1].metric(
        copy["holdout_sharpe"], _ratio(holdout["net_sharpe"])
    )
    confirmation_cards[2].metric(
        copy["hit_rate"], _percent(full["active_net_hit_rate"])
    )

    status = str(summary.get("standalone_status") or "blocked_validation")
    if status == "eligible_for_router_research":
        st.success(copy[status])
    else:
        st.error(copy.get(status, copy["blocked_validation"]))

    st.markdown(f"**{copy['gates']}**")
    _render_gates(snapshot["gates"])

    standalone, concentration, events = st.tabs(
        [copy["standalone"], copy["concentration"], copy["events"]]
    )
    with standalone:
        st.markdown(f"**{copy['sample']}**")
        _render_split(snapshot["split"])
    with concentration:
        metrics = summary.get("concentration") or {}
        cols = st.columns(4)
        cols[0].metric(copy["effective_products"], _ratio(full["mean_effective_products"]))
        cols[1].metric(copy["largest_position"], _percent(full["mean_largest_position_share"]))
        cols[2].metric(copy["top_product"], _percent(metrics.get("top_product_absolute_net_contribution_share")))
        cols[3].metric(copy["top_year"], _percent(metrics.get("top_year_absolute_net_contribution_share")))
        _render_contributions(snapshot["years"], snapshot["products"], copy, plotly_template)
    with events:
        event = summary.get("extreme_event") or {}
        st.caption(
            copy["event_note"].format(
                threshold=float(event.get("validation_threshold") or 0.0),
                quantile=float(event.get("quantile") or 0.0),
                validation=int(event.get("validation_event_count") or 0),
                holdout=int(event.get("holdout_event_count") or 0),
            )
        )
        _render_event_study(snapshot["event_study"], copy, plotly_template)
        st.markdown(f"**{copy['event_windows']}**")
        _render_event_windows(snapshot["event_windows"])
    return True


def _render_gates(gates: pd.DataFrame) -> None:
    display = gates.copy()
    display["Result"] = display["passed"].map({True: "Pass", False: "Fail"})
    display = display[["gate", "metric", "value", "operator", "threshold", "Result"]].rename(
        columns={"gate": "Gate", "metric": "Metric", "value": "Observed", "operator": "Rule", "threshold": "Threshold"}
    )
    st.dataframe(
        display.style.format({"Observed": "{:.3f}", "Threshold": "{:.3f}"}, na_rep=""),
        use_container_width=True,
        hide_index=True,
    )


def _render_split(split: pd.DataFrame) -> None:
    display = split[
        [
            "research_split", "net_annualized_mean", "net_annualized_volatility",
            "net_sharpe", "maximum_drawdown", "annualized_turnover",
            "annualized_exchange_fees", "annualized_slippage", "active_net_hit_rate",
            "position_net_hit_rate", "active_date_count", "net_annualized_mean_ex_extremes",
        ]
    ].copy()
    display["research_split"] = pd.Categorical(
        display["research_split"], ["validation", "holdout", "full"], ordered=True
    )
    display = display.sort_values("research_split").rename(
        columns={
            "research_split": "Split", "net_annualized_mean": "Net mean",
            "net_annualized_volatility": "Net volatility", "net_sharpe": "Net Sharpe",
            "maximum_drawdown": "MDD", "annualized_turnover": "Turnover (x)",
            "annualized_exchange_fees": "Fees", "annualized_slippage": "Slippage",
            "active_net_hit_rate": "Active-day hit", "position_net_hit_rate": "Position hit",
            "active_date_count": "Active days", "net_annualized_mean_ex_extremes": "Net mean ex shocks",
        }
    )
    st.dataframe(
        display.style.format(
            {"Net mean": "{:.2%}", "Net volatility": "{:.2%}", "Net Sharpe": "{:.2f}",
             "MDD": "{:.2%}", "Turnover (x)": "{:.1f}", "Fees": "{:.2%}",
             "Slippage": "{:.2%}", "Active-day hit": "{:.1%}", "Position hit": "{:.1%}",
             "Active days": "{:,.0f}", "Net mean ex shocks": "{:.2%}"}, na_rep=""
        ),
        use_container_width=True, hide_index=True,
    )


def _render_contributions(
    years: pd.DataFrame,
    products: pd.DataFrame,
    copy: dict,
    template: str,
) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{copy['contribution']}**")
        figure = go.Figure()
        figure.add_trace(go.Bar(x=years["year"], y=years["gross_contribution"], name="Gross", marker_color="#2563eb"))
        figure.add_trace(go.Bar(x=years["year"], y=-years["cost_return"], name="Cost", marker_color="#d97706"))
        figure.add_trace(go.Bar(x=years["year"], y=years["net_contribution"], name="Net", marker_color="#b91c1c"))
        figure.update_layout(template=template, barmode="group", height=340, margin={"l": 10, "r": 10, "t": 15, "b": 20}, yaxis={"tickformat": ".1%"}, legend={"orientation": "h", "y": 1.1})
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown(f"**{copy['product_contribution']}**")
        full = products.loc[products["research_split"].eq("full")].copy()
        full = full.sort_values("absolute_net_contribution_share", ascending=False).head(12).sort_values("net_contribution")
        colors = ["#b91c1c" if value < 0 else "#0f766e" for value in full["net_contribution"]]
        figure = go.Figure(go.Bar(x=full["net_contribution"], y=full["ticker"], orientation="h", marker_color=colors))
        figure.add_vline(x=0, line_width=1, line_color="#94a3b8")
        figure.update_layout(template=template, height=340, margin={"l": 10, "r": 10, "t": 15, "b": 20}, xaxis={"tickformat": ".1%"})
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _render_event_study(study: pd.DataFrame, copy: dict, template: str) -> None:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=study["relative_day"], y=study["mean_gross_return"], name="Gross", mode="lines+markers", line={"color": "#2563eb", "width": 2}))
    figure.add_trace(go.Scatter(x=study["relative_day"], y=study["mean_net_return"], name="Net", mode="lines+markers", line={"color": "#b91c1c", "width": 2}))
    figure.add_vline(x=0, line_width=1, line_color="#111827")
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8")
    figure.update_layout(template=template, title=copy["event_path"], height=360, margin={"l": 20, "r": 20, "t": 55, "b": 20}, yaxis={"tickformat": ".2%"}, xaxis={"dtick": 1}, legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _render_event_windows(windows: pd.DataFrame) -> None:
    full = windows.loc[windows["research_split"].eq("full")].copy()
    display = full[["window", "date_count", "net_annualized_mean", "net_sharpe", "active_net_hit_rate", "annualized_cost", "annualized_turnover"]].rename(
        columns={"window": "Window", "date_count": "Days", "net_annualized_mean": "Net mean", "net_sharpe": "Net Sharpe", "active_net_hit_rate": "Hit rate", "annualized_cost": "Cost", "annualized_turnover": "Turnover (x)"}
    )
    st.dataframe(
        display.style.format({"Days": "{:,.0f}", "Net mean": "{:.2%}", "Net Sharpe": "{:.2f}", "Hit rate": "{:.1%}", "Cost": "{:.2%}", "Turnover (x)": "{:.1f}"}, na_rep=""),
        use_container_width=True, hide_index=True,
    )


def _percent(value: Any) -> str:
    return f"{float(value):.1%}" if value is not None and pd.notna(value) else "n/a"


def _ratio(value: Any) -> str:
    return f"{float(value):.2f}" if value is not None and pd.notna(value) else "n/a"


__all__ = [
    "load_standalone_sleeve_test_snapshot",
    "render_standalone_sleeve_test_panel",
]
