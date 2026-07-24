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
        "title": "Fixed Sleeve Translation (Phase 3)",
        "boundary": "One frozen construction is applied without parameter optimisation. Returns below use whole contracts and include instrument fees plus 0.5 tick slippage on entry and exit.",
        "gross_mean": "Gross annual mean",
        "cost": "Annual cost",
        "net_mean": "Net annual mean",
        "sharpe": "Net Sharpe",
        "drawdown": "Maximum drawdown",
        "turnover": "Annual turnover",
        "cost_failure": "The factor has positive gross economics, but this sleeve loses them after execution costs.",
        "gross_failure": "The sleeve is negative before costs; execution is not the primary failure.",
        "net_positive": "This fixed sleeve remains positive after the modelled execution costs; it still requires later validation gates.",
        "holdout_failure": "The full-sample net result is positive, but the holdout net result is not. The sleeve has not demonstrated stable tradability.",
        "performance": "Performance",
        "contributors": "Contributors",
        "construction": "Construction",
        "path": "Gross, cost and net path",
        "yearly": "Annual decomposition",
        "products": "Product contribution",
        "sectors": "Sector contribution",
        "split": "Sample comparison",
        "decision_date": "Decision date",
        "cumulative_gross": "Cumulative gross",
        "cumulative_net": "Cumulative net",
        "cumulative_cost": "Cumulative cost",
    },
    "ZH": {
        "title": "固定策略腿转换（第三阶段）",
        "boundary": "本阶段只使用一个冻结的默认构造，不进行参数优化。以下收益按整数合约执行，并计入品种手续费及开仓、平仓各 0.5 跳滑点。",
        "gross_mean": "毛年化均值",
        "cost": "年化成本",
        "net_mean": "净年化均值",
        "sharpe": "净夏普",
        "drawdown": "最大回撤",
        "turnover": "年化换手",
        "cost_failure": "因子具有正的成本前收益，但该策略腿的执行成本将其完全抵消。",
        "gross_failure": "该策略腿在成本前已经为负，执行成本不是主要失败原因。",
        "net_positive": "该固定策略腿在当前成本模型后仍为正，但仍需通过后续验证门槛。",
        "holdout_failure": "全样本净收益为正，但留出集净收益不为正；该策略腿尚未证明具有稳定可交易性。",
        "performance": "收益表现",
        "contributors": "贡献来源",
        "construction": "构造口径",
        "path": "毛收益、成本与净收益路径",
        "yearly": "年度分解",
        "products": "品种贡献",
        "sectors": "板块贡献",
        "split": "样本对比",
        "decision_date": "决策日期",
        "cumulative_gross": "累计毛收益",
        "cumulative_net": "累计净收益",
        "cumulative_cost": "累计成本",
    },
}


@st.cache_data(show_spinner=False)
def load_sleeve_evidence_snapshot(
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
            / "sleeve_construction"
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
        "daily_returns.parquet",
        "split_summary.csv",
        "yearly_summary.csv",
        "product_summary.csv",
        "sector_summary.csv",
    }
    if not required.issubset(path.name for path in source.iterdir()):
        return None
    daily = pd.read_parquet(source / "daily_returns.parquet")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    return {
        "source": str(source),
        "summary": json.loads((source / "summary.json").read_text(encoding="utf-8")),
        "manifest": json.loads((source / "manifest.json").read_text(encoding="utf-8")),
        "daily": daily,
        "split": pd.read_csv(source / "split_summary.csv"),
        "yearly": pd.read_csv(source / "yearly_summary.csv"),
        "products": pd.read_csv(source / "product_summary.csv"),
        "sectors": pd.read_csv(source / "sector_summary.csv"),
    }


def render_sleeve_evidence_panel(
    artifact_root: str | Path,
    factor_id: str,
    market_vertical: str,
    *,
    lang: str = "EN",
    plotly_template: str = "plotly_white",
) -> bool:
    snapshot = load_sleeve_evidence_snapshot(
        str(artifact_root), factor_id, market_vertical
    )
    if snapshot is None:
        return False
    copy = COPY.get(lang, COPY["EN"])
    split = snapshot["split"]
    full_rows = split.loc[split["research_split"].eq("full")]
    if full_rows.empty:
        return False
    full = full_rows.iloc[0]

    st.markdown(f"#### {copy['title']}")
    st.caption(copy["boundary"])
    metrics = st.columns(6)
    metrics[0].metric(copy["gross_mean"], _percent(full["gross_annualized_mean"]))
    metrics[1].metric(copy["cost"], _percent(full["annualized_cost"]))
    metrics[2].metric(copy["net_mean"], _percent(full["net_annualized_mean"]))
    metrics[3].metric(copy["sharpe"], _ratio(full["net_sharpe"]))
    metrics[4].metric(copy["drawdown"], _percent(full["maximum_drawdown"]))
    metrics[5].metric(copy["turnover"], f"{float(full['annualized_turnover']):.1f}x")

    holdout_rows = split.loc[split["research_split"].eq("holdout")]
    holdout_net = (
        float(holdout_rows.iloc[0]["net_annualized_mean"])
        if not holdout_rows.empty
        else float("nan")
    )
    if float(full["net_annualized_mean"]) > 0 and holdout_net <= 0:
        st.warning(copy["holdout_failure"])
    elif float(full["net_annualized_mean"]) > 0:
        st.success(copy["net_positive"])
    elif float(full["gross_annualized_mean"]) > 0:
        st.warning(copy["cost_failure"])
    else:
        st.error(copy["gross_failure"])

    performance, contributors, construction = st.tabs(
        [copy["performance"], copy["contributors"], copy["construction"]]
    )
    with performance:
        _render_path(snapshot["daily"], copy, plotly_template)
        st.markdown(f"**{copy['split']}**")
        _render_split(split)
        st.markdown(f"**{copy['yearly']}**")
        _render_yearly(snapshot["yearly"], plotly_template)
    with contributors:
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{copy['products']}**")
            _render_member_chart(
                _contribution_extremes(snapshot["products"], 6),
                "ticker",
                plotly_template,
            )
        with right:
            st.markdown(f"**{copy['sectors']}**")
            _render_member_chart(snapshot["sectors"], "sector", plotly_template)
    with construction:
        _render_config(snapshot["manifest"].get("config") or {}, lang)
    return True


def _render_path(daily: pd.DataFrame, copy: dict, template: str) -> None:
    cumulative_cost = daily["cost_return"].fillna(0.0).cumsum()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["cumulative_gross_return"],
            name=copy["cumulative_gross"], line={"color": "#2563eb", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["cumulative_net_return"],
            name=copy["cumulative_net"], line={"color": "#b91c1c", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=daily["date"], y=-cumulative_cost,
            name=copy["cumulative_cost"], line={"color": "#d97706", "width": 1.5, "dash": "dot"},
        )
    )
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8")
    figure.update_layout(
        template=template,
        title=copy["path"],
        height=390,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        yaxis={"tickformat": ".0%"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_split(split: pd.DataFrame) -> None:
    display = split[
        [
            "research_split", "gross_annualized_mean", "annualized_cost",
            "net_annualized_mean", "gross_sharpe", "net_sharpe",
            "maximum_drawdown", "annualized_turnover", "mean_executed_gross",
        ]
    ].copy()
    display["research_split"] = pd.Categorical(
        display["research_split"], ["validation", "holdout", "full"], ordered=True
    )
    display = display.sort_values("research_split").rename(
        columns={
            "research_split": "Split", "gross_annualized_mean": "Gross mean",
            "annualized_cost": "Cost", "net_annualized_mean": "Net mean",
            "gross_sharpe": "Gross Sharpe", "net_sharpe": "Net Sharpe",
            "maximum_drawdown": "MDD", "annualized_turnover": "Turnover (x)",
            "mean_executed_gross": "Mean gross",
        }
    )
    st.dataframe(
        display.style.format(
            {"Gross mean": "{:.2%}", "Cost": "{:.2%}", "Net mean": "{:.2%}",
             "Gross Sharpe": "{:.2f}", "Net Sharpe": "{:.2f}", "MDD": "{:.2%}",
             "Turnover (x)": "{:.1f}", "Mean gross": "{:.1%}"},
            na_rep="",
        ),
        width="stretch", hide_index=True,
    )


def _render_yearly(yearly: pd.DataFrame, template: str) -> None:
    figure = go.Figure()
    for column, name, color in (
        ("gross_annualized_mean", "Gross", "#2563eb"),
        ("annualized_cost", "Cost", "#d97706"),
        ("net_annualized_mean", "Net", "#b91c1c"),
    ):
        figure.add_trace(go.Bar(x=yearly["year"], y=yearly[column], name=name, marker_color=color))
    figure.update_layout(
        template=template, barmode="group", height=340,
        margin={"l": 20, "r": 20, "t": 20, "b": 20}, yaxis={"tickformat": ".0%"},
        legend={"orientation": "h", "y": 1.1, "x": 0},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_member_chart(frame: pd.DataFrame, member_col: str, template: str) -> None:
    ordered = frame.sort_values("net_contribution")
    colors = ["#b91c1c" if value < 0 else "#0f766e" for value in ordered["net_contribution"]]
    figure = go.Figure(
        go.Bar(
            x=ordered["net_contribution"], y=ordered[member_col], orientation="h",
            marker_color=colors,
            customdata=ordered[["gross_contribution", "cost_return", "active_days"]],
            hovertemplate=("%{y}<br>Net %{x:.2%}<br>Gross %{customdata[0]:.2%}"
                           "<br>Cost %{customdata[1]:.2%}<br>Active days %{customdata[2]:,.0f}<extra></extra>"),
        )
    )
    figure.add_vline(x=0, line_width=1, line_color="#94a3b8")
    figure.update_layout(
        template=template, height=max(300, 26 * len(ordered)),
        margin={"l": 10, "r": 10, "t": 10, "b": 20}, xaxis={"tickformat": ".1%"},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _contribution_extremes(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = frame.sort_values("net_contribution")
    return pd.concat([ordered.head(count), ordered.tail(count)]).drop_duplicates()


def _render_config(config: dict[str, Any], lang: str) -> None:
    labels = {
        "construction_geometry": ("Geometry", "构造维度"),
        "expression": ("Expression", "多空表达"),
        "construction": ("Selection", "选取方法"),
        "normalization": ("Normalization", "权重归一化"),
        "winsor_lower_quantile": ("Winsor lower", "缩尾下界"),
        "winsor_upper_quantile": ("Winsor upper", "缩尾上界"),
        "long_fraction": ("Long fraction", "多头比例"),
        "short_fraction": ("Short fraction", "空头比例"),
        "rebalance_every_n_periods": ("Rebalance interval", "调仓间隔"),
        "holding_periods": ("Holding periods", "持有期"),
        "max_weight_per_contract": ("Contract cap", "单品种上限"),
        "max_sector_gross": ("Sector cap", "板块上限"),
        "target_gross_exposure": ("Target gross", "目标毛敞口"),
        "target_net_exposure": ("Target net", "目标净敞口"),
        "execution_delay_periods": ("Additional delay", "额外执行延迟"),
        "missing_signal_policy": ("Missing score", "缺失信号"),
        "zero_signal_policy": ("Zero score", "零信号"),
        "minimum_cross_section": ("Minimum products", "最少品种数"),
        "minimum_distinct_signals": ("Minimum score levels", "最少信号层级"),
        "optimization_permitted": ("Optimisation", "允许优化"),
    }
    index = 0 if lang == "EN" else 1
    rows = [
        {"Setting" if lang == "EN" else "设置": labels[key][index],
         "Value" if lang == "EN" else "取值": value}
        for key, value in config.items() if key in labels
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if config.get("sector_cap_reason"):
        st.caption(str(config["sector_cap_reason"]))


def _percent(value: Any) -> str:
    return f"{float(value):.1%}" if pd.notna(value) else "n/a"


def _ratio(value: Any) -> str:
    return f"{float(value):.2f}" if pd.notna(value) else "n/a"


__all__ = ["load_sleeve_evidence_snapshot", "render_sleeve_evidence_panel"]
