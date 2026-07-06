from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import TEXT, get_plotly_template


DNA_COPY = {
    "EN": {
        "title": "Strategy DNA",
        "caption": "Universal trade diagnostics. This tab no longer switches into ML/GMM-specific views; every factor is inspected through realized trades and portfolio returns.",
        "detected": "Detected factor",
        "run_id": "Run ID",
        "total_trades": "Trades",
        "win_rate": "Win Rate",
        "profit_factor": "Profit Factor",
        "median_hold": "Median Hold",
        "asset_pnl": "Asset PnL Contribution",
        "pnl_dist": "Trade PnL Distribution",
        "holding_pain": "PnL vs Holding Time",
        "trade_equity": "Cumulative Trade PnL",
        "portfolio_fallback": "Portfolio-Level DNA",
        "fallback_caption": "No discrete trade ledger was found, so this run is inspected through daily portfolio returns instead.",
        "daily_return_dist": "Daily Net Return Distribution",
        "equity_drawdown": "Equity and Drawdown",
        "turnover_return": "Turnover vs Daily Return",
        "no_data": "No trade ledger or return series found for this run.",
        "manual_title": "How to read Strategy DNA",
        "manual": """
This tab asks one question: where did the strategy actually make or lose money?

- Asset PnL Contribution shows which contracts contributed most. If one asset dominates, the factor may be less diversified than it looks.
- Trade PnL Distribution shows skew and tails. A good strategy should not rely on rare giant wins while bleeding many small losses.
- PnL vs Holding Time shows whether bad trades linger. Large negative points at long holding periods often reveal stubborn exits.
- Cumulative Trade PnL shows whether edge arrived steadily or from a few isolated trades.

If no trade ledger exists, the page falls back to portfolio-level returns so the tab still gives useful diagnostics.
""",
    },
    "ZH": {
        "title": "策略 DNA",
        "caption": "通用交易诊断。本标签页不再按 ML/GMM 因子切换特殊视图，而是统一从真实交易与组合收益出发检查策略。",
        "detected": "检测到因子",
        "run_id": "运行 ID",
        "total_trades": "交易笔数",
        "win_rate": "胜率",
        "profit_factor": "盈亏因子",
        "median_hold": "中位持仓",
        "asset_pnl": "资产 PnL 贡献",
        "pnl_dist": "单笔交易 PnL 分布",
        "holding_pain": "PnL vs 持仓时间",
        "trade_equity": "累计交易 PnL",
        "portfolio_fallback": "组合级 DNA",
        "fallback_caption": "未找到离散交易记录，因此改用每日组合收益检查该运行。",
        "daily_return_dist": "日度净收益分布",
        "equity_drawdown": "净值与回撤",
        "turnover_return": "换手率 vs 日收益",
        "no_data": "该运行没有交易记录或收益序列。",
        "manual_title": "如何阅读策略 DNA",
        "manual": """
这个标签页只问一个问题：策略到底在哪里赚钱、在哪里亏钱？

- 资产 PnL 贡献：显示哪些合约贡献最大。如果收益集中在少数资产，说明策略可能并没有看起来那么分散。
- 单笔交易 PnL 分布：观察偏度与尾部。好的策略不应该依赖极少数大赚，同时长期承受大量小亏。
- PnL vs 持仓时间：观察坏交易是否拖太久。右侧长持仓区域的大亏点，通常说明退出机制太顽固。
- 累计交易 PnL：观察收益是稳定累积，还是来自少数孤立交易。

如果没有交易记录，本页会自动退回到组合级收益分析。
""",
    },
}


class DNAView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render(self, run_id: str, lang: str = "EN", theme_mode: str = "DARK"):
        t = TEXT[lang] if lang in TEXT else TEXT["EN"]
        copy = DNA_COPY.get(lang, DNA_COPY["EN"])
        template = get_plotly_template(theme_mode)

        st.markdown(f"### {t.get('tab_dna', copy['title'])}")
        st.caption(copy["caption"])
        with st.expander(copy["manual_title"], expanded=False):
            st.markdown(copy["manual"])

        self._render_metadata(run_id, copy)

        trades = self.dm.get_trade_ledger(run_id)
        if trades.empty:
            self._render_portfolio_fallback(run_id, template, copy)
            return

        trades = self._clean_trades(trades)
        if trades.empty:
            self._render_portfolio_fallback(run_id, template, copy)
            return

        self._render_trade_metrics(trades, copy)
        self._render_trade_charts(trades, template, copy)

    def _render_metadata(self, run_id: str, copy: dict):
        runs_df = self.dm.get_all_runs()
        run_meta = runs_df[runs_df["run_id"] == run_id] if not runs_df.empty else pd.DataFrame()
        if run_meta.empty:
            st.write(f"**{copy['run_id']}:** `{run_id}`")
            return

        row = run_meta.iloc[0]
        factor_id = str(row.get("factor_id", "Unknown"))
        name = str(row.get("name", "Unknown"))
        st.write(
            f"**{copy['detected']}:** `{factor_id}` | `{name}` &nbsp;&nbsp; "
            f"**{copy['run_id']}:** `{run_id}`"
        )

    @staticmethod
    def _clean_trades(trades: pd.DataFrame) -> pd.DataFrame:
        out = trades.copy()
        for col in ["trade_pnl", "holding_period_hours", "entry_price", "exit_price"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "trade_pnl" not in out.columns:
            return pd.DataFrame()
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["trade_pnl"])
        out["trade_pnl_pct"] = out["trade_pnl"] * 100.0
        if "win_loss_flag" not in out.columns:
            out["win_loss_flag"] = np.where(out["trade_pnl"] >= 0, "Win", "Loss")
        if "direction" not in out.columns:
            out["direction"] = "Unknown"
        if "exit_time" in out.columns:
            out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
        if "entry_time" in out.columns:
            out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
        return out

    @staticmethod
    def _trade_stats(trades: pd.DataFrame) -> dict:
        wins = trades[trades["trade_pnl"] > 0]["trade_pnl"]
        losses = trades[trades["trade_pnl"] < 0]["trade_pnl"]
        gross_win = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_win / gross_loss if gross_loss > 1e-12 else np.nan
        hold = pd.to_numeric(trades.get("holding_period_hours"), errors="coerce")
        return {
            "total_trades": int(len(trades)),
            "win_rate": float((trades["trade_pnl"] > 0).mean()) if len(trades) else np.nan,
            "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
            "median_hold": float(hold.median()) if hold.notna().any() else np.nan,
        }

    def _render_trade_metrics(self, trades: pd.DataFrame, copy: dict):
        stats = self._trade_stats(trades)
        cols = st.columns(4)
        cols[0].metric(copy["total_trades"], f"{stats['total_trades']:,}")
        cols[1].metric(copy["win_rate"], self._pct(stats["win_rate"]))
        cols[2].metric(copy["profit_factor"], self._num(stats["profit_factor"]))
        cols[3].metric(
            copy["median_hold"],
            "N/A" if pd.isna(stats["median_hold"]) else f"{stats['median_hold']:.0f}h",
        )

    def _render_trade_charts(self, trades: pd.DataFrame, template: str, copy: dict):
        top_left, top_right = st.columns([0.52, 0.48])
        bottom_left, bottom_right = st.columns([0.52, 0.48])

        with top_left:
            st.markdown(f"#### {copy['asset_pnl']}")
            if "ticker" not in trades.columns:
                st.info("Ticker column missing.")
            else:
                asset = trades.groupby("ticker", dropna=False)["trade_pnl"].sum().reset_index()
                asset["trade_pnl_pct"] = asset["trade_pnl"] * 100.0
                asset = asset.reindex(asset["trade_pnl"].abs().sort_values(ascending=False).index).head(20)
                asset = asset.sort_values("trade_pnl_pct")
                fig = px.bar(
                    asset,
                    x="trade_pnl_pct",
                    y="ticker",
                    orientation="h",
                    color="trade_pnl_pct",
                    color_continuous_scale="RdYlGn",
                    template=template,
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")

        with top_right:
            st.markdown(f"#### {copy['pnl_dist']}")
            fig = px.histogram(
                trades,
                x="trade_pnl_pct",
                color="win_loss_flag",
                barmode="overlay",
                color_discrete_map={"Win": "#2ca02c", "Loss": "#d62728"},
                template=template,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Trade PnL (%)")
            st.plotly_chart(fig, width="stretch")

        with bottom_left:
            st.markdown(f"#### {copy['holding_pain']}")
            if "holding_period_hours" not in trades.columns or trades["holding_period_hours"].dropna().empty:
                st.info("Holding period column missing.")
            else:
                fig = px.scatter(
                    trades,
                    x="holding_period_hours",
                    y="trade_pnl_pct",
                    color="direction",
                    size=trades["trade_pnl_pct"].abs().clip(lower=0.1),
                    hover_data=[col for col in ["ticker", "entry_time", "exit_time", "win_loss_flag"] if col in trades.columns],
                    template=template,
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Trade PnL (%)")
                st.plotly_chart(fig, width="stretch")

        with bottom_right:
            st.markdown(f"#### {copy['trade_equity']}")
            if "exit_time" in trades.columns and trades["exit_time"].notna().any():
                curve = trades.sort_values("exit_time").copy()
                x_axis = curve["exit_time"]
            else:
                curve = trades.reset_index(drop=True).copy()
                x_axis = curve.index + 1
            curve["cum_trade_pnl_pct"] = curve["trade_pnl_pct"].cumsum()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=curve["cum_trade_pnl_pct"],
                    mode="lines",
                    name="Cumulative Trade PnL",
                    line=dict(color="#40C4FF", width=2.5),
                )
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                template=template,
                height=430,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Cumulative PnL (%)",
            )
            st.plotly_chart(fig, width="stretch")

    def _render_portfolio_fallback(self, run_id: str, template: str, copy: dict):
        returns = self.dm.get_run_returns(run_id)
        if returns.empty:
            st.warning(copy["no_data"])
            return

        st.markdown(f"### {copy['portfolio_fallback']}")
        st.caption(copy["fallback_caption"])

        df = returns.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["net_return"] = pd.to_numeric(df.get("net_return", 0.0), errors="coerce").fillna(0.0)
        df["equity"] = (1.0 + df["net_return"]).cumprod()
        df["drawdown"] = df["equity"] / df["equity"].cummax() - 1.0
        turnover_col = "daily_turnover" if "daily_turnover" in df.columns else "turnover"
        if turnover_col not in df.columns:
            df[turnover_col] = 0.0
        df[turnover_col] = pd.to_numeric(df[turnover_col], errors="coerce").fillna(0.0)
        df["net_return_pct"] = df["net_return"] * 100.0

        left, right = st.columns(2)
        with left:
            st.markdown(f"#### {copy['equity_drawdown']}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["equity"], mode="lines", name="Equity"))
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["drawdown"],
                    mode="lines",
                    name="Drawdown",
                    yaxis="y2",
                    line=dict(color="#d62728"),
                )
            )
            fig.update_layout(
                template=template,
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(title="Equity"),
                yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%"),
            )
            st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown(f"#### {copy['daily_return_dist']}")
            fig = px.histogram(df, x="net_return_pct", nbins=60, template=template)
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Daily Net Return (%)")
            st.plotly_chart(fig, width="stretch")

        st.markdown(f"#### {copy['turnover_return']}")
        fig = px.scatter(
            df,
            x=turnover_col,
            y="net_return_pct",
            color="drawdown",
            color_continuous_scale="RdYlGn",
            template=template,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Daily Net Return (%)")
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def _pct(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value) * 100:.1f}%"

    @staticmethod
    def _num(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.2f}"
