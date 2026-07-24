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

    SERIES_COLORS = ("#2563eb", "#f97316", "#059669", "#9333ea", "#dc2626", "#0891b2")
    BENCHMARK_COLOR = "#6b7280"

    COPY = {
        "EN": {
            "title": "Strategy Run Comparison",
            "subtitle": (
                "Compare completed, registered strategy backtests. Factor tests, sleeve tests, "
                "smoke runs, and unfinished ideas are excluded."
            ),
            "manual": """
This page compares developed strategy runs, not factors.

1. Select one completed strategy run as the baseline and one or more completed runs to compare.
2. Metrics are recomputed over the exact dates shared by every selected run.
3. A run may represent a different strategy or a frozen version of the same strategy.
4. Read return, Sharpe, drawdown, turnover, and costs together. A higher return bought with much more risk or trading is not a clean improvement.
5. Use the equity, drawdown, correlation, and trade views to inspect path and implementation differences.
""",
            "market": "Market",
            "frequency": "Frequency",
            "core_type": "Core type",
            "all": "All",
            "baseline": "Baseline strategy run",
            "comparisons": "Comparison strategy runs",
            "strategies": "Developed strategies",
            "runs": "Completed runs",
            "selected": "Selected runs",
            "common_dates": "Common dates",
            "metrics": "Strategy Metrics",
            "equity": "Cumulative Equity",
            "drawdown": "Drawdown",
            "correlation": "Return Correlation",
            "improvement": "Improvement vs Selected Baseline",
            "xray": "Trade X-Ray",
            "xray_select": "Inspect selected run",
            "asset_pnl": "Asset PnL Contribution",
            "holding": "Holding Period Distribution",
            "no_returns": "No readable return series found for the selected strategy runs.",
            "no_trades": "No discrete trade ledger found for this run.",
            "no_runs": "No completed developed-strategy runs found.",
            "no_filtered_runs": "No developed-strategy runs match these filters.",
            "total_return": "Total Return",
            "ann_return": "Ann. Return",
            "ann_vol": "Ann. Vol",
            "sharpe": "Sharpe",
            "max_dd": "Max DD",
            "calmar": "Calmar",
            "turnover": "Ann. Turnover",
            "cost": "Total Cost",
            "hit_rate": "Hit Rate",
            "observations": "Observations",
            "trades": "Trades",
        },
        "ZH": {
            "title": "策略运行对比",
            "subtitle": "仅比较已完成并登记的策略回测；不纳入因子测试、袖套测试、冒烟测试或未完成想法。",
            "manual": """
本页比较的是已开发策略的回测运行，不是因子。

1. 选择一个已完成策略运行作为基线，再选择一个或多个运行进行比较。
2. 所有指标按所选运行共同拥有的日期重新计算，确保口径一致。
3. 对比项可以是不同策略，也可以是同一策略的不同冻结版本。
4. 收益、夏普、回撤、换手与成本必须一起看；用更多风险和交易换来的高收益不一定更好。
5. 最后检查净值、回撤、相关性和交易明细，理解差异来自哪里。
""",
            "market": "市场",
            "frequency": "频率",
            "core_type": "核心类型",
            "all": "全部",
            "baseline": "基准策略运行",
            "comparisons": "对比策略运行",
            "strategies": "已开发策略",
            "runs": "已完成运行",
            "selected": "已选运行",
            "common_dates": "共同日期",
            "metrics": "策略指标",
            "equity": "累计净值",
            "drawdown": "历史回撤",
            "correlation": "收益相关性",
            "improvement": "相对所选基线的改善",
            "xray": "交易 X-Ray",
            "xray_select": "选择要检查的运行",
            "asset_pnl": "资产 PnL 贡献",
            "holding": "持仓时间分布",
            "no_returns": "所选策略运行没有可读取的收益序列。",
            "no_trades": "该运行没有离散交易记录。",
            "no_runs": "未找到已完成的正式策略回测。",
            "no_filtered_runs": "没有正式策略运行符合当前筛选条件。",
            "total_return": "累计收益",
            "ann_return": "年化收益",
            "ann_vol": "年化波动",
            "sharpe": "夏普",
            "max_dd": "最大回撤",
            "calmar": "卡玛",
            "turnover": "年化换手",
            "cost": "累计成本",
            "hit_rate": "正收益占比",
            "observations": "观测数",
            "trades": "交易数",
        },
    }

    def __init__(self, data_manager):
        self.dm = data_manager

    def render(
        self,
        lang: str = "EN",
        theme_mode: str = "LIGHT",
        *,
        embedded: bool = False,
    ):
        copy = self.COPY.get(lang, self.COPY["EN"])
        tpl = get_plotly_template(theme_mode)
        runs_df = self._prepare_runs(self.dm.get_completed_strategy_runs())

        if embedded:
            st.subheader(copy["title"])
        else:
            st.title(copy["title"])
        st.caption(copy["subtitle"])
        with st.expander("How to use / 使用说明", expanded=False):
            st.markdown(copy["manual"])

        if runs_df.empty:
            st.warning(copy["no_runs"])
            return

        runs_df = self._render_run_filters(runs_df, copy)
        if runs_df.empty:
            st.info(copy["no_filtered_runs"])
            return

        label_map = self._build_label_map(runs_df)
        labels = list(label_map)
        baseline_label = st.selectbox(
            copy["baseline"],
            labels,
            index=0,
            key="strategy_comparison_baseline",
        )
        comparison_options = [label for label in labels if label != baseline_label]
        default_comparisons = comparison_options[:1]
        comparison_labels = st.multiselect(
            copy["comparisons"],
            comparison_options,
            default=default_comparisons,
            key="strategy_comparison_runs",
        )
        selected_labels = [baseline_label, *comparison_labels]
        profiles = self._load_profiles(
            [
                ("Baseline" if index == 0 else "Comparison", label)
                for index, label in enumerate(selected_labels)
            ],
            label_map,
        )
        profiles = self._align_profiles(self._dedupe_profiles(profiles))
        if not profiles:
            st.info(copy["no_returns"])
            return

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            copy["strategies"],
            runs_df["strategy_id"].nunique(),
        )
        metric_cols[1].metric(copy["runs"], len(runs_df))
        metric_cols[2].metric(copy["selected"], len(profiles))
        metric_cols[3].metric(
            copy["common_dates"],
            min((len(profile.returns) for profile in profiles), default=0),
        )
        self._render_comparability_note(profiles)
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
        out["strategy_id"] = out.get("strategy_id", "").fillna("").astype(str)
        out["strategy_id"] = out["strategy_id"].mask(
            out["strategy_id"].eq(""),
            out["factor_id"],
        )
        out["name"] = out.get("name", "").astype(str)
        for column in ("market_vertical", "data_frequency", "strategy_core_type"):
            if column not in out.columns:
                out[column] = ""
            out[column] = out[column].fillna("").astype(str)
        out["strategy_core_type"] = out["strategy_core_type"].replace("", "Unrecorded")
        return out.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)

    @staticmethod
    def _run_label(row: pd.Series) -> str:
        timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
        date_label = timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else "no-date"
        name = str(row.get("name", "Unknown"))
        strategy_id = str(row.get("strategy_id", row.get("factor_id", "")))
        run_id = str(row.get("run_id", ""))
        sharpe = pd.to_numeric(row.get("sharpe_ratio"), errors="coerce")
        sharpe_label = f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"
        meaningful_name = (
            name
            if name
            and name.lower() not in {"unknown", "nan", "none"}
            and not name.lower().startswith("auto-gen")
            and strategy_id.lower() not in name.lower()
            else ""
        )
        identity = (
            f"{strategy_id} | {meaningful_name}"
            if meaningful_name
            else strategy_id
        )
        return (
            f"{date_label} | {identity} | "
            f"Sharpe {sharpe_label} | {run_id[:8]}"
        )

    def _build_label_map(self, runs_df: pd.DataFrame) -> dict[str, pd.Series]:
        label_map: dict[str, pd.Series] = {}
        for _, row in runs_df.iterrows():
            label = self._run_label(row)
            label_map[label] = row
        return label_map

    @staticmethod
    def _filter_options(series: pd.Series) -> list[str]:
        values = sorted(
            value
            for value in series.fillna("").astype(str).unique().tolist()
            if value and value.lower() not in {"nan", "none"}
        )
        return ["All", *values]

    def _render_run_filters(self, runs_df: pd.DataFrame, copy: dict) -> pd.DataFrame:
        cols = st.columns(3)
        market = cols[0].selectbox(
            copy["market"],
            self._filter_options(runs_df["market_vertical"]),
            key="strategy_comparison_market",
        )
        frequency = cols[1].selectbox(
            copy["frequency"],
            self._filter_options(runs_df["data_frequency"]),
            key="strategy_comparison_frequency",
        )
        core_type = cols[2].selectbox(
            copy["core_type"],
            self._filter_options(runs_df["strategy_core_type"]),
            key="strategy_comparison_core_type",
        )
        filtered = runs_df.copy()
        for column, value in (
            ("market_vertical", market),
            ("data_frequency", frequency),
            ("strategy_core_type", core_type),
        ):
            if value != "All":
                filtered = filtered.loc[filtered[column].eq(value)]
        return filtered.reset_index(drop=True)

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
        strategy_id = str(row.get("strategy_id", row.get("factor_id", "Unknown")))
        run_id = str(row.get("run_id", ""))
        timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
        date_label = timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else "no-date"
        return f"{group}: {strategy_id} [{date_label}, {run_id[:8]}]"

    @staticmethod
    def _compute_metrics(returns: pd.DataFrame, row: pd.Series) -> dict:
        def numeric_column(
            primary: str,
            *,
            fallback: str | None = None,
            default: float = 0.0,
        ) -> pd.Series:
            if primary in returns.columns:
                values = returns[primary]
            elif fallback and fallback in returns.columns:
                values = returns[fallback]
            else:
                values = pd.Series(default, index=returns.index, dtype=float)
            return pd.to_numeric(values, errors="coerce")

        ret = numeric_column("net_return").fillna(0.0)
        eq = (1.0 + ret).cumprod()
        days = max(len(ret), 1)
        total_return = float(eq.iloc[-1] - 1.0) if not eq.empty else np.nan
        ann_return = float(eq.iloc[-1] ** (252 / days) - 1.0) if not eq.empty else np.nan
        ann_vol = float(ret.std(ddof=1) * np.sqrt(252)) if len(ret) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0
        drawdown = eq / eq.cummax() - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
        calmar = ann_return / abs(max_dd) if pd.notna(max_dd) and abs(max_dd) > 1e-12 else 0.0
        turnover = numeric_column("daily_turnover", fallback="turnover").fillna(0.0)
        total_cost = numeric_column("daily_total_cost").fillna(0.0)
        daily_cost_bps = numeric_column("daily_cost_bps", default=np.nan)
        if daily_cost_bps.notna().any():
            total_cost_rate = float(daily_cost_bps.fillna(0.0).sum() / 10_000.0)
        else:
            initial_capital = numeric_column("initial_capital", default=np.nan)
            capital = initial_capital.dropna()
            total_cost_rate = (
                float(total_cost.sum() / capital.iloc[0])
                if not capital.empty and capital.iloc[0] > 0
                else np.nan
            )
        total_trades = pd.to_numeric(row.get("total_trades"), errors="coerce")
        active_returns = ret.loc[ret.ne(0.0)]
        hit_rate = float(active_returns.gt(0.0).mean()) if not active_returns.empty else np.nan
        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": float(sharpe),
            "max_dd": max_dd,
            "calmar": float(calmar),
            "ann_turnover": float(turnover.mean() * 252.0),
            "total_cost": total_cost_rate,
            "hit_rate": hit_rate,
            "observations": int(len(ret)),
            "total_trades": int(total_trades) if pd.notna(total_trades) else 0,
        }

    def _align_profiles(
        self,
        profiles: list[StrategyProfile],
    ) -> list[StrategyProfile]:
        if len(profiles) < 2:
            return profiles
        date_sets = []
        normalized = []
        for profile in profiles:
            returns = profile.returns.copy()
            returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
            returns = returns.dropna(subset=["date"]).sort_values("date")
            normalized.append((profile, returns))
            date_sets.append(set(returns["date"]))
        common_dates = set.intersection(*date_sets) if date_sets else set()
        if not common_dates:
            return []
        aligned = []
        for profile, returns in normalized:
            common = returns.loc[returns["date"].isin(common_dates)].copy()
            aligned.append(
                StrategyProfile(
                    label=profile.label,
                    group=profile.group,
                    row=profile.row,
                    returns=common,
                    metrics=self._compute_metrics(common, profile.row),
                )
            )
        return aligned

    @staticmethod
    def _render_comparability_note(profiles: list[StrategyProfile]):
        if not profiles:
            return
        fields = {
            "market": {str(profile.row.get("market_vertical", "")) for profile in profiles},
            "frequency": {str(profile.row.get("data_frequency", "")) for profile in profiles},
            "dataset": {str(profile.row.get("dataset_id", "")) for profile in profiles},
        }
        mismatches = [name for name, values in fields.items() if len(values - {"", "nan"}) > 1]
        dates = pd.to_datetime(profiles[0].returns["date"], errors="coerce").dropna()
        if mismatches:
            st.warning(
                "Selected runs differ in "
                + ", ".join(mismatches)
                + ". Curves share dates, but the economic comparison is not fully like-for-like."
            )
        if not dates.empty:
            st.caption(
                f"Metrics use {len(dates):,} common observations from "
                f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}."
            )

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
                c4.metric(copy["turnover"], self._pct(metrics["ann_turnover"]))

    def _render_equity_chart(self, profiles: list[StrategyProfile], tpl: str, copy: dict):
        st.markdown(f"### {copy['equity']}")
        fig = go.Figure()
        for index, profile in enumerate(profiles):
            curve = self._profile_curve(profile)
            fig.add_trace(
                go.Scatter(
                    x=curve["date"],
                    y=curve["equity"],
                    mode="lines",
                    name=profile.label,
                    line=dict(
                        color=self.SERIES_COLORS[index % len(self.SERIES_COLORS)],
                        width=2,
                    ),
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
                    line=dict(color=self.BENCHMARK_COLOR, width=1.4, dash="dash"),
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
        for index, profile in enumerate(profiles):
            curve = self._profile_curve(profile)
            fig.add_trace(
                go.Scatter(
                    x=curve["date"],
                    y=curve["drawdown"],
                    mode="lines",
                    name=profile.label,
                    line=dict(
                        color=self.SERIES_COLORS[index % len(self.SERIES_COLORS)],
                        width=2,
                    ),
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
                "strategy_run": profile.label,
                "role": profile.group,
                copy["total_return"]: profile.metrics["total_return"],
                copy["ann_return"]: profile.metrics["ann_return"],
                copy["ann_vol"]: profile.metrics["ann_vol"],
                copy["sharpe"]: profile.metrics["sharpe"],
                copy["max_dd"]: profile.metrics["max_dd"],
                copy["calmar"]: profile.metrics["calmar"],
                copy["turnover"]: profile.metrics["ann_turnover"],
                copy["cost"]: profile.metrics["total_cost"],
                copy["hit_rate"]: profile.metrics["hit_rate"],
                copy["observations"]: profile.metrics["observations"],
                copy["trades"]: profile.metrics["total_trades"],
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        pct_cols = [
            copy["total_return"],
            copy["ann_return"],
            copy["ann_vol"],
            copy["max_dd"],
            copy["turnover"],
            copy["cost"],
            copy["hit_rate"],
        ]
        fmt = {col: "{:.2%}" for col in pct_cols if col in df.columns}
        fmt.update({copy["sharpe"]: "{:.2f}", copy["calmar"]: "{:.2f}"})
        st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)

    @staticmethod
    def _short_profile_labels(profiles: list[StrategyProfile]) -> dict[str, str]:
        label_by_profile = {}
        for idx, profile in enumerate(profiles, start=1):
            group = profile.group or "Strategy"
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
        baseline = next((p for p in profiles if p.group == "Baseline"), profiles[0])
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
                    "turnover_delta": profile.metrics["ann_turnover"] - baseline.metrics["ann_turnover"],
                    "cost_delta": profile.metrics["total_cost"] - baseline.metrics["total_cost"],
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
                    "cost_delta": "{:.2%}",
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
                    color_discrete_sequence=[self.SERIES_COLORS[0]],
                )
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Trades")
                st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _pct(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value) * 100:.1f}%"

    @staticmethod
    def _num(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.2f}"
