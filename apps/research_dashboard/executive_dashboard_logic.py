from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import get_plotly_template


@dataclass
class StrategyProfile:
    label: str
    group: str
    row: pd.Series
    returns: pd.DataFrame
    metrics: dict


class ExecutiveDashboardView:
    """Reusable strategy comparison view for research reviews."""

    COLORS = {
        "Heuristic": "#1f77b4",
        "Raw ML": "#ff7f0e",
        "Gatekept / Regime": "#2ca02c",
        "Additional": "#9467bd",
        "Benchmark": "#7f8c8d",
    }

    COPY = {
        "EN": {
            "title": "Strategy Comparison",
            "subtitle": "Compare heuristic, raw ML, and gatekept/regime-aware strategies using completed backtest runs.",
            "manual": """
Use this page as a reusable manager dashboard.

1. Pick one representative run for each strategy profile: heuristic baseline, raw ML, and gatekept or regime-aware ML.
2. Read the metric cards first. A better strategy should improve Sharpe and drawdown, not just annual return.
3. Use the equity and drawdown charts to see path quality. Smooth survival beats one lucky spike.
4. Check return correlation. Low correlation means the strategies may diversify each other; high correlation means they are probably the same risk in different clothes.
5. Use the trade x-ray last to inspect which assets created or destroyed PnL.
""",
            "heuristic": "Heuristic / rules profile",
            "raw_ml": "Raw ML profile",
            "gate": "Gatekept / regime-aware profile",
            "additional": "Additional comparison runs",
            "metrics": "Profile Metrics",
            "equity": "Cumulative Equity",
            "drawdown": "Drawdown",
            "correlation": "Return Correlation",
            "improvement": "Improvement vs Heuristic Baseline",
            "xray": "Trade X-Ray",
            "xray_select": "Inspect selected run",
            "asset_pnl": "Asset PnL Contribution",
            "holding": "Holding Period Distribution",
            "no_returns": "No return series found for the selected profile.",
            "no_trades": "No discrete trade ledger found for this run.",
            "no_runs": "No completed backtest runs found. Run the evaluator first.",
            "ann_return": "Ann. Return",
            "ann_vol": "Ann. Vol",
            "sharpe": "Sharpe",
            "max_dd": "Max DD",
            "calmar": "Calmar",
            "turnover": "Avg Turnover",
            "holdout_ic": "Holdout IC",
            "trades": "Trades",
        },
        "ZH": {
            "title": "策略对比",
            "subtitle": "用已完成的回测，手动对比启发式策略、纯 ML 策略、以及带状态/风控门控的 ML 策略。",
            "manual": """
这个页面现在是通用的经理汇报看板。

1. 先为每类策略选择一个代表运行：启发式基线、纯 ML、以及带状态/门控的 ML。
2. 先看顶部指标。更好的策略不应该只提高年化收益，还应该改善夏普和回撤。
3. 再看净值和回撤曲线。稳定存活比单次暴涨更重要。
4. 检查收益相关性。低相关说明策略可能互补；高相关说明它们可能只是同一种风险的不同包装。
5. 最后用交易 X-Ray 查看哪些资产贡献或拖累了收益。
""",
            "heuristic": "启发式 / 规则策略",
            "raw_ml": "纯 ML 策略",
            "gate": "门控 / 状态感知 ML 策略",
            "additional": "额外对比运行",
            "metrics": "策略指标",
            "equity": "累计净值",
            "drawdown": "历史回撤",
            "correlation": "收益相关性",
            "improvement": "相对启发式基线的改善",
            "xray": "交易 X-Ray",
            "xray_select": "选择要检查的运行",
            "asset_pnl": "资产 PnL 贡献",
            "holding": "持仓时间分布",
            "no_returns": "所选策略没有收益序列。",
            "no_trades": "该运行没有离散交易记录。",
            "no_runs": "未找到已完成回测。请先运行 evaluator。",
            "ann_return": "年化收益",
            "ann_vol": "年化波动",
            "sharpe": "夏普",
            "max_dd": "最大回撤",
            "calmar": "卡玛",
            "turnover": "平均换手",
            "holdout_ic": "样本外 IC",
            "trades": "交易数",
        },
    }

    def __init__(self, data_manager):
        self.dm = data_manager

    def render(self, lang: str = "EN", theme_mode: str = "LIGHT"):
        copy = self.COPY.get(lang, self.COPY["EN"])
        tpl = get_plotly_template(theme_mode)
        runs_df = self._prepare_runs(self.dm.get_all_runs())

        st.title(copy["title"])
        st.caption(copy["subtitle"])
        with st.expander("How to use / 使用说明", expanded=False):
            st.markdown(copy["manual"])

        if runs_df.empty:
            st.warning(copy["no_runs"])
            return

        label_map = self._build_label_map(runs_df)
        labels = list(label_map.keys())

        selector_cols = st.columns([0.31, 0.31, 0.31])
        heuristic_label = self._select_profile(
            selector_cols[0],
            copy["heuristic"],
            labels,
            runs_df,
            label_map,
            key="exec_profile_heuristic",
            patterns=("fac_039", "fac_040", "fac_041", "fac_042", "sma", "bollinger", "heuristic", "rule"),
        )
        raw_ml_label = self._select_profile(
            selector_cols[1],
            copy["raw_ml"],
            labels,
            runs_df,
            label_map,
            key="exec_profile_raw_ml",
            patterns=("fac_054", "xgboost", "raw ml", "machine learning", "ml alpha"),
        )
        gate_label = self._select_profile(
            selector_cols[2],
            copy["gate"],
            labels,
            runs_df,
            label_map,
            key="exec_profile_gate",
            patterns=("fac_056", "fac_057", "fac_043", "gmm", "hmm", "ensemble", "gate", "regime", "router"),
        )

        selected_primary = [heuristic_label, raw_ml_label, gate_label]
        additional_labels = st.multiselect(
            copy["additional"],
            labels,
            default=[],
            key="exec_profile_additional",
        )

        profiles = self._load_profiles(
            [
                ("Heuristic", heuristic_label),
                ("Raw ML", raw_ml_label),
                ("Gatekept / Regime", gate_label),
                *[("Additional", label) for label in additional_labels],
            ],
            label_map,
        )

        profiles = self._dedupe_profiles(profiles)
        if not profiles:
            st.info(copy["no_returns"])
            return

        st.markdown("---")
        self._render_metric_cards(profiles, copy)
        st.markdown("---")

        chart_left, chart_right = st.columns([0.58, 0.42])
        with chart_left:
            self._render_equity_chart(profiles, tpl, copy)
        with chart_right:
            self._render_drawdown_chart(profiles, tpl, copy)

        table_left, table_right = st.columns([0.5, 0.5])
        with table_left:
            self._render_metric_table(profiles, copy)
        with table_right:
            self._render_correlation(profiles, tpl, copy)

        st.markdown("---")
        self._render_improvement(profiles, copy)
        st.markdown("---")
        self._render_trade_xray(profiles, tpl, copy)

    @staticmethod
    def _prepare_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
        if runs_df.empty:
            return runs_df
        out = runs_df.copy()
        out["timestamp"] = pd.to_datetime(out.get("timestamp"), errors="coerce")
        out["factor_id"] = out.get("factor_id", "").astype(str)
        out["name"] = out.get("name", "").astype(str)
        return out.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)

    @staticmethod
    def _run_label(row: pd.Series) -> str:
        timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
        date_label = timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else "no-date"
        name = str(row.get("name", "Unknown"))
        factor_id = str(row.get("factor_id", ""))
        run_id = str(row.get("run_id", ""))
        sharpe = pd.to_numeric(row.get("sharpe_ratio"), errors="coerce")
        ic = pd.to_numeric(row.get("holdout_ic"), errors="coerce")
        return f"{date_label} | {name} | {factor_id} | Sharpe {sharpe:.2f} | IC {ic:.4f} | {run_id[:8]}"

    def _build_label_map(self, runs_df: pd.DataFrame) -> dict[str, pd.Series]:
        label_map: dict[str, pd.Series] = {}
        for _, row in runs_df.iterrows():
            label = self._run_label(row)
            label_map[label] = row
        return label_map

    def _select_profile(
        self,
        container,
        title: str,
        labels: list[str],
        runs_df: pd.DataFrame,
        label_map: dict[str, pd.Series],
        key: str,
        patterns: tuple[str, ...],
    ) -> str | None:
        if not labels:
            return None
        default_label = self._default_label(runs_df, label_map, patterns) or labels[0]
        default_index = labels.index(default_label) if default_label in labels else 0
        with container:
            return st.selectbox(title, labels, index=default_index, key=key)

    def _default_label(
        self,
        runs_df: pd.DataFrame,
        label_map: dict[str, pd.Series],
        patterns: tuple[str, ...],
    ) -> str | None:
        haystack = (
            runs_df["factor_id"].fillna("").astype(str)
            + " "
            + runs_df["name"].fillna("").astype(str)
        ).str.lower()
        for pattern in patterns:
            matches = runs_df[haystack.str.contains(pattern.lower(), regex=False)]
            if not matches.empty:
                row_id = str(matches.iloc[0].get("run_id"))
                for label, row in label_map.items():
                    if str(row.get("run_id")) == row_id:
                        return label
        return None

    def _load_profiles(
        self,
        selections: list[tuple[str, str | None]],
        label_map: dict[str, pd.Series],
    ) -> list[StrategyProfile]:
        profiles = []
        for group, label in selections:
            if not label or label not in label_map:
                continue
            row = label_map[label]
            run_id = str(row.get("run_id", ""))
            returns = self.dm.get_run_returns(run_id, returns_path=row.get("returns_file_path"))
            if returns.empty:
                continue
            metrics = self._compute_metrics(returns, row)
            short_name = self._short_profile_name(row, group)
            profiles.append(
                StrategyProfile(
                    label=short_name,
                    group=group,
                    row=row,
                    returns=returns,
                    metrics=metrics,
                )
            )
        return profiles

    @staticmethod
    def _dedupe_profiles(profiles: list[StrategyProfile]) -> list[StrategyProfile]:
        seen = set()
        out = []
        for profile in profiles:
            run_id = str(profile.row.get("run_id", ""))
            if run_id in seen:
                continue
            seen.add(run_id)
            out.append(profile)
        return out

    @staticmethod
    def _short_profile_name(row: pd.Series, group: str) -> str:
        name = str(row.get("name", "Unknown"))
        factor_id = str(row.get("factor_id", ""))
        if factor_id and factor_id.lower() not in name.lower():
            return f"{group}: {name} ({factor_id})"
        return f"{group}: {name}"

    @staticmethod
    def _compute_metrics(returns: pd.DataFrame, row: pd.Series) -> dict:
        ret = pd.to_numeric(returns.get("net_return", 0.0), errors="coerce").fillna(0.0)
        eq = (1.0 + ret).cumprod()
        days = max(len(ret), 1)
        ann_return = float(eq.iloc[-1] ** (252 / days) - 1.0) if not eq.empty else np.nan
        ann_vol = float(ret.std(ddof=1) * np.sqrt(252)) if len(ret) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0
        drawdown = eq / eq.cummax() - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
        calmar = ann_return / abs(max_dd) if pd.notna(max_dd) and abs(max_dd) > 1e-12 else 0.0
        turnover = pd.to_numeric(
            returns.get("daily_turnover", returns.get("turnover", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        holdout_ic = pd.to_numeric(row.get("holdout_ic"), errors="coerce")
        total_trades = pd.to_numeric(row.get("total_trades"), errors="coerce")
        return {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": float(sharpe),
            "max_dd": max_dd,
            "calmar": float(calmar),
            "avg_turnover": float(turnover.mean()),
            "holdout_ic": float(holdout_ic) if pd.notna(holdout_ic) else np.nan,
            "total_trades": int(total_trades) if pd.notna(total_trades) else 0,
        }

    def _profile_curve(self, profile: StrategyProfile) -> pd.DataFrame:
        df = profile.returns.copy()
        df["date"] = pd.to_datetime(df["date"])
        ret = pd.to_numeric(df.get("net_return", 0.0), errors="coerce").fillna(0.0)
        eq = (1.0 + ret).cumprod()
        dd = eq / eq.cummax() - 1.0
        out = pd.DataFrame(
            {
                "date": df["date"],
                "profile": profile.label,
                "group": profile.group,
                "net_return": ret,
                "equity": eq,
                "drawdown": dd,
            }
        )
        if "benchmark_return" in df.columns:
            bench = pd.to_numeric(df["benchmark_return"], errors="coerce").fillna(0.0)
            out["benchmark_equity"] = (1.0 + bench).cumprod()
        return out

    def _render_metric_cards(self, profiles: list[StrategyProfile], copy: dict):
        st.markdown(f"### {copy['metrics']}")
        cols = st.columns(min(len(profiles), 4))
        for idx, profile in enumerate(profiles[:4]):
            metrics = profile.metrics
            with cols[idx % len(cols)]:
                st.markdown(f"**{profile.label}**")
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c1.metric(copy["ann_return"], self._pct(metrics["ann_return"]))
                c2.metric(copy["sharpe"], self._num(metrics["sharpe"]))
                c3.metric(copy["max_dd"], self._pct(metrics["max_dd"]))
                c4.metric(copy["turnover"], self._pct(metrics["avg_turnover"]))

    def _render_equity_chart(self, profiles: list[StrategyProfile], tpl: str, copy: dict):
        st.markdown(f"### {copy['equity']}")
        fig = go.Figure()
        for profile in profiles:
            curve = self._profile_curve(profile)
            fig.add_trace(
                go.Scatter(
                    x=curve["date"],
                    y=curve["equity"],
                    mode="lines",
                    name=profile.label,
                    line=dict(color=self.COLORS.get(profile.group), width=2),
                )
            )
        first_curve = self._profile_curve(profiles[0])
        if "benchmark_equity" in first_curve.columns:
            fig.add_trace(
                go.Scatter(
                    x=first_curve["date"],
                    y=first_curve["benchmark_equity"],
                    mode="lines",
                    name="Benchmark",
                    line=dict(color=self.COLORS["Benchmark"], width=1.4, dash="dash"),
                )
            )
        fig.update_layout(
            template=tpl,
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis_title="Equity",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_drawdown_chart(self, profiles: list[StrategyProfile], tpl: str, copy: dict):
        st.markdown(f"### {copy['drawdown']}")
        fig = go.Figure()
        for profile in profiles:
            curve = self._profile_curve(profile)
            fig.add_trace(
                go.Scatter(
                    x=curve["date"],
                    y=curve["drawdown"],
                    mode="lines",
                    name=profile.label,
                    line=dict(color=self.COLORS.get(profile.group), width=2),
                )
            )
        fig.update_layout(
            template=tpl,
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_tickformat=".0%",
            yaxis_title="Drawdown",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_metric_table(self, profiles: list[StrategyProfile], copy: dict):
        st.markdown(f"### {copy['metrics']}")
        rows = []
        for profile in profiles:
            row = {
                "profile": profile.label,
                "group": profile.group,
                copy["ann_return"]: profile.metrics["ann_return"],
                copy["ann_vol"]: profile.metrics["ann_vol"],
                copy["sharpe"]: profile.metrics["sharpe"],
                copy["max_dd"]: profile.metrics["max_dd"],
                copy["calmar"]: profile.metrics["calmar"],
                copy["turnover"]: profile.metrics["avg_turnover"],
                copy["holdout_ic"]: profile.metrics["holdout_ic"],
                copy["trades"]: profile.metrics["total_trades"],
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        pct_cols = [copy["ann_return"], copy["ann_vol"], copy["max_dd"], copy["turnover"]]
        fmt = {col: "{:.2%}" for col in pct_cols if col in df.columns}
        fmt.update({copy["sharpe"]: "{:.2f}", copy["calmar"]: "{:.2f}", copy["holdout_ic"]: "{:.4f}"})
        st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)

    @staticmethod
    def _short_profile_labels(profiles: list[StrategyProfile]) -> dict[str, str]:
        label_by_profile = {}
        group_names = {
            "Gatekept / Regime": "Gate",
            "Additional": "Add",
        }
        for idx, profile in enumerate(profiles, start=1):
            group = group_names.get(profile.group, profile.group or "Profile")
            label_by_profile[profile.label] = f"P{idx} {group}"
        return label_by_profile

    def _render_correlation(self, profiles: list[StrategyProfile], tpl: str, copy: dict):
        st.markdown(f"### {copy['correlation']}")
        ret_data = {}
        for profile in profiles:
            curve = self._profile_curve(profile)
            ret_data[profile.label] = curve.set_index("date")["net_return"]
        corr_df = pd.DataFrame(ret_data).dropna()
        if corr_df.shape[1] < 2 or corr_df.empty:
            st.info("Need at least two overlapping return series.")
            return
        corr = corr_df.corr()
        short_labels = self._short_profile_labels(profiles)
        axis_labels = [short_labels.get(label, str(label)) for label in corr.columns]
        z_values = corr.to_numpy()
        text_values = [[f"{value:.2f}" for value in row] for row in z_values]
        hover_names = [[[str(y_name), str(x_name)] for x_name in corr.columns] for y_name in corr.index]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=axis_labels,
                y=axis_labels,
                text=text_values,
                texttemplate="%{text}",
                textfont=dict(size=13),
                customdata=hover_names,
                hovertemplate=(
                    "Row: %{customdata[0]}<br>"
                    "Column: %{customdata[1]}<br>"
                    "Correlation: %{z:.2f}<extra></extra>"
                ),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(thickness=12, len=0.72, tickformat=".1f"),
            )
        )
        fig.update_layout(
            template=tpl,
            height=min(620, max(380, 85 * len(axis_labels) + 150)),
            margin=dict(l=8, r=8, t=20, b=8),
            xaxis=dict(side="top", tickangle=0, constrain="domain"),
            yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_improvement(self, profiles: list[StrategyProfile], copy: dict):
        baseline = next((p for p in profiles if p.group == "Heuristic"), profiles[0])
        rows = []
        for profile in profiles:
            if profile is baseline:
                continue
            rows.append(
                {
                    "profile": profile.label,
                    "delta_sharpe": profile.metrics["sharpe"] - baseline.metrics["sharpe"],
                    "delta_ann_return": profile.metrics["ann_return"] - baseline.metrics["ann_return"],
                    "drawdown_reduction": abs(baseline.metrics["max_dd"]) - abs(profile.metrics["max_dd"]),
                    "turnover_delta": profile.metrics["avg_turnover"] - baseline.metrics["avg_turnover"],
                }
            )
        st.markdown(f"### {copy['improvement']}")
        if not rows:
            st.info("Select at least two different profiles.")
            return
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format(
                {
                    "delta_sharpe": "{:.2f}",
                    "delta_ann_return": "{:.2%}",
                    "drawdown_reduction": "{:.2%}",
                    "turnover_delta": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    def _render_trade_xray(self, profiles: list[StrategyProfile], tpl: str, copy: dict):
        st.markdown(f"### {copy['xray']}")
        label_to_profile = {p.label: p for p in profiles}
        selected = st.selectbox(copy["xray_select"], list(label_to_profile), key="exec_xray_profile")
        profile = label_to_profile[selected]
        trades = self.dm.get_trade_ledger(str(profile.row.get("run_id", "")))
        if trades.empty:
            st.info(copy["no_trades"])
            return

        left, right = st.columns(2)
        with left:
            st.markdown(f"#### {copy['asset_pnl']}")
            pnl_col = "trade_pnl" if "trade_pnl" in trades.columns else None
            if pnl_col is None or "ticker" not in trades.columns:
                st.info(copy["no_trades"])
            else:
                pnl = trades.groupby("ticker")[pnl_col].sum().reset_index()
                pnl = pnl.sort_values(pnl_col, ascending=False).head(20)
                fig = px.bar(
                    pnl,
                    x="ticker",
                    y=pnl_col,
                    color=pnl_col,
                    color_continuous_scale="RdYlGn",
                    template=tpl,
                )
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown(f"#### {copy['holding']}")
            hold_col = "holding_period_hours" if "holding_period_hours" in trades.columns else None
            if hold_col is None:
                st.info(copy["no_trades"])
            else:
                fig = px.histogram(
                    trades,
                    x=hold_col,
                    nbins=60,
                    template=tpl,
                    color_discrete_sequence=[self.COLORS.get(profile.group, "#9467bd")],
                )
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Trades")
                st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _pct(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value) * 100:.1f}%"

    @staticmethod
    def _num(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.2f}"
