from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import BASE_DIR, TEXT, get_plotly_template


DNA_COPY = {
    "EN": {
        "title": "Strategy DNA",
        "caption": "Universal trade diagnostics. This tab no longer switches into ML/GMM-specific views; every factor is inspected through realized trades and portfolio returns.",
        "detected": "Detected factor",
        "run_id": "Run ID",
        "asset_pnl": "Top Winners and Losers",
        "pnl_dist": "Trade PnL Distribution",
        "holding_pain": "PnL vs Holding Time",
        "edge_stability": "Edge Stability Heatmap",
        "edge_caption": "Color shows average trade PnL by entry period and holding bucket. Hover reveals trade count, win rate, and total PnL.",
        "edge_color": "Avg PnL %",
        "hover_company": "Company",
        "hover_chinese_name": "Chinese Name",
        "hover_trades": "Trades",
        "hover_trade_pnl": "Trade PnL",
        "hover_side": "Side",
        "hover_result": "Result",
        "hover_entry": "Entry",
        "hover_exit": "Exit",
        "hover_hold_hours": "Hold Hours",
        "side_long": "Long",
        "side_short": "Short",
        "side_unknown": "Unknown",
        "result_win": "Win",
        "result_loss": "Loss",
        "result_flat": "Flat",
        "holding_x": "Holding Period (hours)",
        "holding_y": "Trade PnL (%)",
        "dist_context": "Flat trades: {flat:,} ({flat_pct:.1f}%) | shown non-flat trades: {shown:,} | tails outside view: {left_tail:,} left / {right_tail:,} right",
        "dist_x": "Trade PnL (%)",
        "dist_y": "Trades (% of shown)",
        "hover_bin": "Bin",
        "hover_share": "Share",
        "hover_avg_pnl": "Avg PnL",
        "hover_sum_pnl": "Sum PnL",
        "hover_period": "Entry Period",
        "hover_hold_bucket": "Holding Bucket",
        "hover_win_rate": "Win Rate",
        "portfolio_fallback": "Portfolio-Level DNA",
        "fallback_caption": "No discrete trade ledger was found, so this run is inspected through daily portfolio returns instead.",
        "daily_return_dist": "Daily Net Return Distribution",
        "equity_drawdown": "Equity and Drawdown",
        "turnover_return": "Turnover vs Daily Return",
        "no_data": "No trade ledger or return series found for this run.",
        "manual_title": "How to read Strategy DNA",
        "manual": """
This tab asks one question: where did the strategy actually make or lose money?

- Top Winners and Losers shows the five best and five worst tickers by summed trade PnL %. If one asset dominates, the factor may be less diversified than it looks.
- Trade PnL Distribution shows skew and tails. A good strategy should not rely on rare giant wins while bleeding many small losses.
- PnL vs Holding Time shows whether bad trades linger. Large negative points at long holding periods often reveal stubborn exits.
- Edge Stability Heatmap shows whether the factor works consistently across entry periods and holding horizons.

If no trade ledger exists, the page falls back to portfolio-level returns so the tab still gives useful diagnostics.
""",
    },
    "ZH": {
        "title": "策略 DNA",
        "caption": "通用交易诊断。本标签页不再按 ML/GMM 因子切换特殊视图，而是统一从真实交易与组合收益出发检查策略。",
        "detected": "检测到因子",
        "run_id": "运行 ID",
        "asset_pnl": "盈利/亏损前五资产",
        "pnl_dist": "单笔交易 PnL 分布",
        "holding_pain": "PnL vs 持仓时间",
        "edge_stability": "边际稳定性热力图",
        "edge_caption": "颜色表示不同入场周期与持仓区间下的平均单笔交易 PnL。悬停可查看交易数、胜率和累计 PnL。",
        "edge_color": "平均 PnL %",
        "hover_company": "公司",
        "hover_chinese_name": "中文名",
        "hover_trades": "交易数",
        "hover_trade_pnl": "交易 PnL",
        "hover_side": "方向",
        "hover_result": "结果",
        "hover_entry": "入场",
        "hover_exit": "出场",
        "hover_hold_hours": "持仓小时",
        "side_long": "多头",
        "side_short": "空头",
        "side_unknown": "未知",
        "result_win": "盈利",
        "result_loss": "亏损",
        "result_flat": "持平",
        "holding_x": "持仓时间（小时）",
        "holding_y": "单笔交易 PnL (%)",
        "dist_context": "持平交易: {flat:,} ({flat_pct:.1f}%) | 图中非持平交易: {shown:,} | 图外尾部: 左侧 {left_tail:,} / 右侧 {right_tail:,}",
        "dist_x": "单笔交易 PnL (%)",
        "dist_y": "交易占比（图中样本）",
        "hover_bin": "区间",
        "hover_share": "占比",
        "hover_avg_pnl": "平均 PnL",
        "hover_sum_pnl": "累计 PnL",
        "hover_period": "入场周期",
        "hover_hold_bucket": "持仓区间",
        "hover_win_rate": "胜率",
        "portfolio_fallback": "组合级 DNA",
        "fallback_caption": "未找到离散交易记录，因此改用每日组合收益检查该运行。",
        "daily_return_dist": "日度净收益分布",
        "equity_drawdown": "净值与回撤",
        "turnover_return": "换手率 vs 日收益",
        "no_data": "该运行没有交易记录或收益序列。",
        "manual_title": "如何阅读策略 DNA",
        "manual": """
这个标签页只问一个问题：策略到底在哪里赚钱、在哪里亏钱？

- 盈利/亏损前五资产：按累计单笔交易 PnL % 展示最赚钱和最亏钱的五个标的。如果收益集中在少数资产，说明策略可能并没有看起来那么分散。
- 单笔交易 PnL 分布：观察偏度与尾部。好的策略不应该依赖极少数大赚，同时长期承受大量小亏。
- PnL vs 持仓时间：观察坏交易是否拖太久。右侧长持仓区域的大亏点，通常说明退出机制太顽固。
- 边际稳定性热力图：观察因子在不同入场周期与持仓区间中是否仍然稳定有效。

如果没有交易记录，本页会自动退回到组合级收益分析。
""",
    },
}


CN_FUTURES_PRODUCT_NAMES_ZH = {
    "AP": "苹果",
    "CF": "棉花",
    "CJ": "红枣",
    "CY": "棉纱",
    "FG": "玻璃",
    "MA": "甲醇",
    "OI": "菜籽油",
    "PF": "短纤",
    "PK": "花生",
    "PL": "瓶片",
    "PR": "瓶片",
    "PX": "对二甲苯",
    "RM": "菜籽粕",
    "SA": "纯碱",
    "SF": "硅铁",
    "SH": "烧碱",
    "SM": "锰硅",
    "SR": "白糖",
    "TA": "PTA",
    "UR": "尿素",
    "a": "豆一",
    "b": "豆二",
    "c": "玉米",
    "cs": "玉米淀粉",
    "eb": "苯乙烯",
    "eg": "乙二醇",
    "i": "铁矿石",
    "j": "焦炭",
    "jd": "鸡蛋",
    "jm": "焦煤",
    "l": "聚乙烯",
    "lg": "原木",
    "lh": "生猪",
    "m": "豆粕",
    "p": "棕榈油",
    "pg": "液化石油气",
    "pp": "聚丙烯",
    "rr": "粳米",
    "v": "PVC",
    "y": "豆油",
    "bz": "纯苯",
    "ad": "铸造铝合金",
    "ag": "白银",
    "al": "铝",
    "ao": "氧化铝",
    "au": "黄金",
    "bc": "国际铜",
    "br": "丁二烯橡胶",
    "bu": "沥青",
    "cu": "铜",
    "fu": "燃料油",
    "hc": "热卷",
    "lu": "低硫燃料油",
    "ni": "镍",
    "nr": "20号胶",
    "op": "纸浆",
    "pb": "铅",
    "rb": "螺纹钢",
    "ru": "橡胶",
    "sc": "原油",
    "sn": "锡",
    "sp": "纸浆",
    "ss": "不锈钢",
    "zn": "锌",
    "ec": "集运欧线",
    "si": "工业硅",
    "ps": "多晶硅",
    "lc": "碳酸锂",
    "pd": "钯",
    "pt": "铂",
    "IC": "中证500股指",
    "IF": "沪深300股指",
    "IH": "上证50股指",
    "IM": "中证1000股指",
    "T": "10年期国债",
    "TF": "5年期国债",
    "TL": "30年期国债",
    "TS": "2年期国债",
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

        self._render_trade_metrics(trades, t)
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
        if "ticker" in out.columns:
            out["ticker"] = out["ticker"].astype(str)
        return out

    @staticmethod
    def _trade_stats(trades: pd.DataFrame) -> dict:
        wins = trades[trades["trade_pnl"] > 0]["trade_pnl"]
        losses = trades[trades["trade_pnl"] < 0]["trade_pnl"]
        gross_win = wins.sum()
        gross_loss = abs(losses.sum())
        avg_win = wins.mean() if not wins.empty else np.nan
        avg_loss = losses.mean() if not losses.empty else np.nan
        avg_loss_abs = abs(avg_loss) if pd.notna(avg_loss) else np.nan
        profit_factor = gross_win / gross_loss if gross_loss > 1e-12 else np.nan
        payoff_ratio = avg_win / avg_loss_abs if pd.notna(avg_win) and avg_loss_abs > 1e-12 else np.nan
        hold = pd.to_numeric(trades.get("holding_period_hours"), errors="coerce")
        concentration = DNAView._profit_concentration(trades)
        return {
            "total_trades": int(len(trades)),
            "win_rate": float((trades["trade_pnl"] > 0).mean()) if len(trades) else np.nan,
            "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
            "median_hold": float(hold.median()) if hold.notna().any() else np.nan,
            "payoff_ratio": float(payoff_ratio) if pd.notna(payoff_ratio) else np.nan,
            "avg_win": float(avg_win) if pd.notna(avg_win) else np.nan,
            "avg_loss": float(avg_loss) if pd.notna(avg_loss) else np.nan,
            "expectancy": float(trades["trade_pnl"].mean()) if len(trades) else np.nan,
            **concentration,
        }

    @staticmethod
    def _profit_concentration(trades: pd.DataFrame, threshold: float = 0.80) -> dict:
        if "ticker" not in trades.columns or "trade_pnl" not in trades.columns:
            return {
                "profit_concentration_count": np.nan,
                "profit_concentration_total": 0,
                "profit_concentration_share": np.nan,
            }

        asset_pnl = (
            trades.assign(trade_pnl=pd.to_numeric(trades["trade_pnl"], errors="coerce"))
            .dropna(subset=["trade_pnl"])
            .groupby("ticker", dropna=False)["trade_pnl"]
            .sum()
        )
        positive = asset_pnl[asset_pnl > 0].sort_values(ascending=False)
        total_profit = float(positive.sum())
        if positive.empty or total_profit <= 1e-12:
            return {
                "profit_concentration_count": np.nan,
                "profit_concentration_total": int(len(positive)),
                "profit_concentration_share": np.nan,
            }

        cumulative_share = positive.cumsum() / total_profit
        count = int((cumulative_share < threshold).sum() + 1)
        count = min(count, int(len(positive)))
        share = float(positive.head(count).sum() / total_profit)
        return {
            "profit_concentration_count": count,
            "profit_concentration_total": int(len(positive)),
            "profit_concentration_share": share,
        }

    def _render_trade_metrics(self, trades: pd.DataFrame, t: dict):
        stats = self._trade_stats(trades)
        primary = st.columns(4)
        primary[0].metric(t.get("dna_total_trades", "Trades"), f"{stats['total_trades']:,}")
        primary[1].metric(t.get("dna_win_rate", "Win Rate"), self._pct(stats["win_rate"]))
        primary[2].metric(t.get("dna_profit_factor", "Profit Factor"), self._num(stats["profit_factor"]))
        primary[3].metric(
            t.get("dna_median_hold", "Median Hold"),
            "N/A" if pd.isna(stats["median_hold"]) else f"{stats['median_hold']:.0f}h",
        )

        secondary = st.columns(4)
        secondary[0].metric(t.get("dna_payoff_ratio", "Payoff Ratio"), self._multiple(stats["payoff_ratio"]))
        secondary[1].metric(t.get("dna_avg_win", "Avg Win"), self._signed_pct(stats["avg_win"]))
        secondary[2].metric(t.get("dna_avg_loss", "Avg Loss"), self._signed_pct(stats["avg_loss"]))
        secondary[3].metric(
            t.get("dna_profit_concentration", "80% Profit Tickers"),
            self._count_fraction(
                stats["profit_concentration_count"],
                stats["profit_concentration_total"],
            ),
        )

    def _render_trade_charts(self, trades: pd.DataFrame, template: str, copy: dict):
        top_left, top_right = st.columns([0.52, 0.48])
        bottom_left, bottom_right = st.columns([0.52, 0.48])

        with top_left:
            st.markdown(f"#### {copy['asset_pnl']}")
            if "ticker" not in trades.columns:
                st.info("Ticker column missing.")
            else:
                asset = self._asset_winner_loser_frame(trades)
                fig = px.bar(
                    asset,
                    x="trade_pnl_pct",
                    y="display_label",
                    orientation="h",
                    color="side",
                    color_discrete_map={"Winner": "#2ca02c", "Loser": "#d62728"},
                    custom_data=[
                        "ticker",
                        "company_name",
                        "asset_name_zh",
                        "trade_count",
                        "trade_pnl_pct",
                    ],
                    template=template,
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{copy['hover_company']}: %{{customdata[1]}}<br>"
                        f"{copy['hover_chinese_name']}: %{{customdata[2]}}<br>"
                        f"{copy['hover_trades']}: %{{customdata[3]:,}}<br>"
                        f"{copy['hover_trade_pnl']}: %{{customdata[4]:+.2f}}%"
                        "<extra></extra>"
                    )
                )
                fig.update_layout(
                    height=430,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Summed Trade PnL (%)",
                    yaxis_title="",
                    legend_title_text="",
                )
                st.plotly_chart(fig, use_container_width=True)

        with top_right:
            st.markdown(f"#### {copy['pnl_dist']}")
            dist, dist_summary = self._trade_pnl_distribution_frame(trades)
            if dist.empty:
                st.info("No non-flat trade PnL values to chart.")
            else:
                st.caption(copy["dist_context"].format(**dist_summary))
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=dist["bin_mid"],
                        y=dist["share_pct"],
                        width=dist["bin_width"] * 0.92,
                        marker_color=np.where(dist["bin_mid"] >= 0, "#2ca02c", "#d62728"),
                        customdata=dist[
                            ["range_label", "count", "share_pct", "avg_pnl", "sum_pnl"]
                        ].to_numpy(),
                        hovertemplate=(
                            f"{copy['hover_bin']}: %{{customdata[0]}}<br>"
                            f"{copy['hover_trades']}: %{{customdata[1]:,}}<br>"
                            f"{copy['hover_share']}: %{{customdata[2]:.2f}}%<br>"
                            f"{copy['hover_avg_pnl']}: %{{customdata[3]:+.2f}}%<br>"
                            f"{copy['hover_sum_pnl']}: %{{customdata[4]:+.2f}}%"
                            "<extra></extra>"
                        ),
                    )
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    template=template,
                    height=430,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title=copy["dist_x"],
                    yaxis_title=copy["dist_y"],
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        with bottom_left:
            st.markdown(f"#### {copy['holding_pain']}")
            if "holding_period_hours" not in trades.columns or trades["holding_period_hours"].dropna().empty:
                st.info("Holding period column missing.")
            else:
                holding = self._holding_pain_frame(trades)
                if holding.empty:
                    st.info("No valid holding-period trades to chart.")
                else:
                    side_labels = {
                        "Long": copy["side_long"],
                        "Short": copy["side_short"],
                        "Unknown": copy["side_unknown"],
                    }
                    result_labels = {
                        "Win": copy["result_win"],
                        "Loss": copy["result_loss"],
                        "Flat": copy["result_flat"],
                    }
                    holding["side_label"] = holding["side_key"].map(side_labels).fillna(copy["side_unknown"])
                    holding["result_label"] = holding["result_key"].map(result_labels).fillna(copy["result_flat"])

                    fig = go.Figure()
                    colors = {"Long": "#2563EB", "Short": "#F97316", "Unknown": "#64748B"}
                    for side_key in ["Long", "Short", "Unknown"]:
                        side_frame = holding[holding["side_key"].eq(side_key)]
                        if side_frame.empty:
                            continue
                        fig.add_trace(
                            go.Scattergl(
                                x=side_frame["holding_period_hours"],
                                y=side_frame["trade_pnl_pct"],
                                mode="markers",
                                name=side_labels[side_key],
                                marker=dict(
                                    color=colors[side_key],
                                    size=side_frame["marker_size"],
                                    opacity=0.68,
                                    line=dict(width=0.4, color="rgba(15, 23, 42, 0.38)"),
                                ),
                                customdata=side_frame[
                                    [
                                        "ticker",
                                        "company_name",
                                        "side_label",
                                        "result_label",
                                        "entry_label",
                                        "exit_label",
                                        "holding_period_hours",
                                        "trade_pnl_pct",
                                    ]
                                ].to_numpy(),
                                hovertemplate=(
                                    "<b>%{customdata[0]}</b><br>"
                                    f"{copy['hover_company']}: %{{customdata[1]}}<br>"
                                    f"{copy['hover_side']}: %{{customdata[2]}}<br>"
                                    f"{copy['hover_result']}: %{{customdata[3]}}<br>"
                                    f"{copy['hover_entry']}: %{{customdata[4]}}<br>"
                                    f"{copy['hover_exit']}: %{{customdata[5]}}<br>"
                                    f"{copy['hover_hold_hours']}: %{{customdata[6]:,.1f}}h<br>"
                                    f"{copy['hover_trade_pnl']}: %{{customdata[7]:+.2f}}%"
                                    "<extra></extra>"
                                ),
                            )
                        )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        template=template,
                        height=430,
                        margin=dict(l=10, r=10, t=28, b=10),
                        xaxis_title=copy["holding_x"],
                        yaxis_title=copy["holding_y"],
                        legend_title_text=copy["hover_side"],
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with bottom_right:
            st.markdown(f"#### {copy['edge_stability']}")
            edge, edge_meta = self._edge_stability_frame(trades)
            if edge.empty:
                st.info("Entry time or holding-period data is missing.")
            else:
                st.caption(copy["edge_caption"])
                periods = edge_meta["periods"]
                buckets = edge_meta["buckets"]
                avg_matrix = edge.pivot(
                    index="hold_bucket",
                    columns="entry_period",
                    values="avg_pnl",
                ).reindex(index=buckets, columns=periods)
                count_matrix = edge.pivot(
                    index="hold_bucket",
                    columns="entry_period",
                    values="trade_count",
                ).reindex(index=buckets, columns=periods)
                win_matrix = edge.pivot(
                    index="hold_bucket",
                    columns="entry_period",
                    values="win_rate",
                ).reindex(index=buckets, columns=periods)
                sum_matrix = edge.pivot(
                    index="hold_bucket",
                    columns="entry_period",
                    values="sum_pnl",
                ).reindex(index=buckets, columns=periods)

                customdata = np.dstack(
                    [
                        count_matrix.fillna(0).to_numpy(dtype=float),
                        win_matrix.to_numpy(dtype=float),
                        avg_matrix.to_numpy(dtype=float),
                        sum_matrix.to_numpy(dtype=float),
                    ]
                )
                z_values = avg_matrix.to_numpy(dtype=float)
                finite_z = z_values[np.isfinite(z_values)]
                z_cap = float(np.nanquantile(np.abs(finite_z), 0.95)) if finite_z.size else 1.0
                z_cap = max(z_cap, 0.1)

                fig = go.Figure(
                    go.Heatmap(
                        x=periods,
                        y=buckets,
                        z=z_values,
                        zmin=-z_cap,
                        zmax=z_cap,
                        zmid=0,
                        colorscale="RdYlGn",
                        xgap=2,
                        ygap=2,
                        customdata=customdata,
                        hoverongaps=False,
                        colorbar=dict(title=copy["edge_color"]),
                        hovertemplate=(
                            f"{copy['hover_period']}: %{{x}}<br>"
                            f"{copy['hover_hold_bucket']}: %{{y}}<br>"
                            f"{copy['hover_trades']}: %{{customdata[0]:,.0f}}<br>"
                            f"{copy['hover_win_rate']}: %{{customdata[1]:.1f}}%<br>"
                            f"{copy['hover_avg_pnl']}: %{{customdata[2]:+.2f}}%<br>"
                            f"{copy['hover_sum_pnl']}: %{{customdata[3]:+.2f}}%"
                            "<extra></extra>"
                        ),
                    )
                )
                fig.update_layout(
                    template=template,
                    height=430,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="",
                    yaxis_title="",
                )
                fig.update_xaxes(tickangle=-45)
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

    def _asset_winner_loser_frame(self, trades: pd.DataFrame) -> pd.DataFrame:
        asset = (
            trades.groupby("ticker", dropna=False)
            .agg(trade_pnl=("trade_pnl", "sum"), trade_count=("trade_pnl", "size"))
            .reset_index()
        )
        asset["ticker"] = asset["ticker"].astype(str)
        asset["trade_pnl_pct"] = asset["trade_pnl"] * 100.0

        name_map = self._trade_name_lookup(trades)
        cn_name_map = self._trade_chinese_name_lookup(trades)
        name_map.update(
            {
                key: value
                for key, value in self._cn_equity_name_lookup(str(BASE_DIR)).items()
                if key not in name_map
            }
        )
        cn_name_map.update(
            {
                key: value
                for key, value in self._cn_equity_name_lookup(str(BASE_DIR)).items()
                if key not in cn_name_map
            }
        )
        asset["company_name"] = asset["ticker"].map(name_map).fillna("N/A")
        asset["asset_name_zh"] = (
            asset["ticker"]
            .map(cn_name_map)
            .fillna(asset["ticker"].map(self._cn_futures_display_name))
            .fillna("N/A")
        )
        asset["display_label"] = asset["ticker"]

        winners = asset[asset["trade_pnl_pct"] > 0].nlargest(5, "trade_pnl_pct")
        losers = asset[asset["trade_pnl_pct"] < 0].nsmallest(5, "trade_pnl_pct")
        out = pd.concat([losers, winners], ignore_index=True)
        if out.empty:
            out = asset.reindex(asset["trade_pnl_pct"].abs().sort_values(ascending=False).index).head(10)
        out["side"] = np.where(out["trade_pnl_pct"] >= 0, "Winner", "Loser")
        return out.sort_values("trade_pnl_pct", ascending=True).reset_index(drop=True)

    def _holding_pain_frame(self, trades: pd.DataFrame) -> pd.DataFrame:
        out = trades.copy()
        if "ticker" not in out.columns:
            out["ticker"] = "N/A"
        out["ticker"] = out["ticker"].astype(str)
        out["holding_period_hours"] = pd.to_numeric(out.get("holding_period_hours"), errors="coerce")
        out["trade_pnl_pct"] = pd.to_numeric(out.get("trade_pnl_pct"), errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["holding_period_hours", "trade_pnl_pct"]
        )
        if out.empty:
            return out

        name_map = self._trade_name_lookup(out)
        name_map.update(
            {
                key: value
                for key, value in self._cn_equity_name_lookup(str(BASE_DIR)).items()
                if key not in name_map
            }
        )
        out["company_name"] = out["ticker"].map(name_map).fillna("N/A")

        direction = out.get("direction", pd.Series("Unknown", index=out.index))
        out["side_key"] = direction.map(self._trade_side_key)
        out["result_key"] = out["trade_pnl_pct"].map(self._trade_result_key)

        abs_pnl = out["trade_pnl_pct"].abs()
        cap = float(abs_pnl.quantile(0.95)) if abs_pnl.notna().any() else 0.0
        cap = max(cap, 0.1)
        scaled = np.sqrt(abs_pnl.clip(lower=0.0, upper=cap) / cap)
        out["marker_size"] = (5.0 + scaled * 13.0).fillna(5.0)

        entry_time = out.get("entry_time", pd.Series(pd.NA, index=out.index))
        exit_time = out.get("exit_time", pd.Series(pd.NA, index=out.index))
        entry_price = out.get("entry_price", pd.Series(pd.NA, index=out.index))
        exit_price = out.get("exit_price", pd.Series(pd.NA, index=out.index))
        out["entry_label"] = [
            self._time_price_label(time_value, price_value)
            for time_value, price_value in zip(entry_time, entry_price)
        ]
        out["exit_label"] = [
            self._time_price_label(time_value, price_value)
            for time_value, price_value in zip(exit_time, exit_price)
        ]
        return out

    @staticmethod
    def _edge_stability_frame(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        out = trades.copy()
        if "trade_pnl_pct" not in out.columns and "trade_pnl" in out.columns:
            out["trade_pnl_pct"] = pd.to_numeric(out["trade_pnl"], errors="coerce") * 100.0
        out["trade_pnl_pct"] = pd.to_numeric(out.get("trade_pnl_pct"), errors="coerce")
        out["holding_period_hours"] = pd.to_numeric(out.get("holding_period_hours"), errors="coerce")

        entry_time = (
            pd.to_datetime(out["entry_time"], errors="coerce")
            if "entry_time" in out.columns
            else pd.Series(pd.NaT, index=out.index)
        )
        exit_time = (
            pd.to_datetime(out["exit_time"], errors="coerce")
            if "exit_time" in out.columns
            else pd.Series(pd.NaT, index=out.index)
        )
        out["_entry_period_time"] = entry_time.fillna(exit_time)
        out = out.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["_entry_period_time", "holding_period_hours", "trade_pnl_pct"]
        )
        if out.empty:
            return pd.DataFrame(), {"periods": [], "buckets": [], "period_level": "month"}

        month_count = int(out["_entry_period_time"].dt.to_period("M").nunique())
        freq = "Q" if month_count > 36 else "M"
        period_level = "quarter" if freq == "Q" else "month"
        out["_period"] = out["_entry_period_time"].dt.to_period(freq)
        period_index = pd.PeriodIndex(out["_period"].dropna().unique()).sort_values()
        period_labels = [str(period) for period in period_index]

        bucket_labels = ["<1d", "1-3d", "3-7d", "7-14d", ">14d"]
        out["hold_bucket"] = pd.cut(
            out["holding_period_hours"],
            bins=[-np.inf, 24.0, 72.0, 168.0, 336.0, np.inf],
            labels=bucket_labels,
            right=True,
        )
        out = out.dropna(subset=["hold_bucket"]).copy()
        if out.empty:
            return pd.DataFrame(), {
                "periods": period_labels,
                "buckets": bucket_labels,
                "period_level": period_level,
            }

        out["entry_period"] = out["_period"].astype(str)
        out["hold_bucket"] = out["hold_bucket"].astype(str)
        grouped = (
            out.groupby(["hold_bucket", "entry_period"], observed=True)
            .agg(
                trade_count=("trade_pnl_pct", "size"),
                avg_pnl=("trade_pnl_pct", "mean"),
                sum_pnl=("trade_pnl_pct", "sum"),
                win_rate=("trade_pnl_pct", lambda series: float((series > 0).mean()) * 100.0),
            )
            .reset_index()
        )
        return grouped, {
            "periods": period_labels,
            "buckets": bucket_labels,
            "period_level": period_level,
        }

    @staticmethod
    def _trade_side_key(value) -> str:
        if pd.isna(value):
            return "Unknown"
        text = str(value).strip().lower()
        if text in {"1", "1.0", "+1", "l", "long", "buy", "b"} or "long" in text:
            return "Long"
        if text in {"-1", "-1.0", "s", "short", "sell"} or "short" in text:
            return "Short"
        return "Unknown"

    @staticmethod
    def _trade_result_key(value) -> str:
        value = pd.to_numeric(value, errors="coerce")
        if pd.isna(value) or abs(float(value)) <= 1e-12:
            return "Flat"
        return "Win" if float(value) > 0 else "Loss"

    @staticmethod
    def _time_price_label(time_value, price_value) -> str:
        ts = pd.to_datetime(time_value, errors="coerce")
        if pd.isna(ts):
            time_label = "N/A"
        elif ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            time_label = ts.strftime("%Y-%m-%d")
        else:
            time_label = ts.strftime("%Y-%m-%d %H:%M")

        price = pd.to_numeric(price_value, errors="coerce")
        price_label = "N/A" if pd.isna(price) else f"{float(price):,.2f}"

        if time_label == "N/A" and price_label == "N/A":
            return "N/A"
        if price_label == "N/A":
            return time_label
        if time_label == "N/A":
            return price_label
        return f"{time_label} @ {price_label}"

    @staticmethod
    def _trade_pnl_distribution_frame(
        trades: pd.DataFrame,
        *,
        bins: int = 48,
        flat_threshold: float = 1e-9,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
    ) -> tuple[pd.DataFrame, dict]:
        values = pd.to_numeric(trades.get("trade_pnl_pct"), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        total = int(len(values))
        if values.empty:
            return pd.DataFrame(), {
                "flat": 0,
                "flat_pct": 0.0,
                "shown": 0,
                "left_tail": 0,
                "right_tail": 0,
            }

        flat_mask = values.abs() <= flat_threshold
        flat_count = int(flat_mask.sum())
        nonflat = values[~flat_mask]
        if nonflat.empty:
            return pd.DataFrame(), {
                "flat": flat_count,
                "flat_pct": flat_count / max(total, 1) * 100.0,
                "shown": 0,
                "left_tail": 0,
                "right_tail": 0,
            }

        lower = float(nonflat.quantile(lower_q))
        upper = float(nonflat.quantile(upper_q))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            lower = float(nonflat.min())
            upper = float(nonflat.max())
        if lower >= upper:
            pad = max(abs(lower) * 0.1, 0.01)
            lower -= pad
            upper += pad

        core = nonflat[(nonflat >= lower) & (nonflat <= upper)]
        if core.empty:
            core = nonflat.copy()
            lower = float(core.min())
            upper = float(core.max())

        left_tail = int((nonflat < lower).sum())
        right_tail = int((nonflat > upper).sum())
        bin_count = max(8, min(int(bins), max(8, int(core.nunique()))))
        edges = np.linspace(lower, upper, bin_count + 1)
        if len(np.unique(edges)) < 2:
            edges = np.array([lower - 0.01, upper + 0.01])

        frame = pd.DataFrame({"trade_pnl_pct": core})
        frame["bin"] = pd.cut(frame["trade_pnl_pct"], bins=edges, include_lowest=True)
        grouped = (
            frame.dropna(subset=["bin"])
            .groupby("bin", observed=False)
            .agg(
                count=("trade_pnl_pct", "size"),
                avg_pnl=("trade_pnl_pct", "mean"),
                sum_pnl=("trade_pnl_pct", "sum"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["count"] > 0].copy()
        if grouped.empty:
            return pd.DataFrame(), {
                "flat": flat_count,
                "flat_pct": flat_count / max(total, 1) * 100.0,
                "shown": 0,
                "left_tail": left_tail,
                "right_tail": right_tail,
            }

        grouped["bin_left"] = [float(interval.left) for interval in grouped["bin"]]
        grouped["bin_right"] = [float(interval.right) for interval in grouped["bin"]]
        grouped["bin_mid"] = (grouped["bin_left"] + grouped["bin_right"]) / 2.0
        grouped["bin_width"] = grouped["bin_right"] - grouped["bin_left"]
        grouped["share_pct"] = grouped["count"] / max(int(grouped["count"].sum()), 1) * 100.0
        grouped["range_label"] = grouped.apply(
            lambda row: f"{row['bin_left']:.2f}% to {row['bin_right']:.2f}%",
            axis=1,
        )

        summary = {
            "flat": flat_count,
            "flat_pct": flat_count / max(total, 1) * 100.0,
            "shown": int(grouped["count"].sum()),
            "left_tail": left_tail,
            "right_tail": right_tail,
        }
        return grouped, summary

    @staticmethod
    def _trade_name_lookup(trades: pd.DataFrame) -> dict[str, str]:
        if "ticker" not in trades.columns:
            return {}
        name_cols = [
            "company_name_zh",
            "name_zh",
            "stock_name_zh",
            "company_name",
            "stock_name",
            "asset_name",
            "instrument_name",
            "name",
        ]
        available = [col for col in name_cols if col in trades.columns]
        if not available:
            return {}

        out: dict[str, str] = {}
        keyed = trades.copy()
        keyed["ticker"] = keyed["ticker"].astype(str)
        for col in available:
            values = keyed[["ticker", col]].dropna()
            if values.empty:
                continue
            values[col] = values[col].astype(str).str.strip()
            values = values[values[col].ne("")]
            for ticker, name in values.drop_duplicates("ticker").itertuples(index=False):
                out.setdefault(str(ticker), str(name))
        return out

    @staticmethod
    def _trade_chinese_name_lookup(trades: pd.DataFrame) -> dict[str, str]:
        if "ticker" not in trades.columns:
            return {}
        primary_cols = [
            "company_name_zh",
            "name_zh",
            "stock_name_zh",
            "asset_name_zh",
            "instrument_name_zh",
            "chinese_name",
        ]
        fallback_cols = ["asset_name", "instrument_name", "name"]
        available = [col for col in primary_cols if col in trades.columns]
        fallback_available = [col for col in fallback_cols if col in trades.columns]
        if not available and not fallback_available:
            return {}

        out: dict[str, str] = {}
        keyed = trades.copy()
        keyed["ticker"] = keyed["ticker"].astype(str)

        for col in available:
            values = keyed[["ticker", col]].dropna()
            if values.empty:
                continue
            values[col] = values[col].astype(str).str.strip()
            values = values[values[col].ne("")]
            for ticker, name in values.drop_duplicates("ticker").itertuples(index=False):
                out.setdefault(str(ticker), str(name))

        for col in fallback_available:
            values = keyed[["ticker", col]].dropna()
            if values.empty:
                continue
            values[col] = values[col].astype(str).str.strip()
            values = values[values[col].map(DNAView._contains_cjk)]
            for ticker, name in values.drop_duplicates("ticker").itertuples(index=False):
                out.setdefault(str(ticker), str(name))
        return out

    @staticmethod
    def _contains_cjk(value: object) -> bool:
        text = str(value or "")
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    @staticmethod
    def _cn_futures_base_symbol(ticker: object) -> str:
        text = str(ticker or "").strip()
        if not text:
            return ""
        if "@" in text:
            text = text.rsplit("@", 1)[-1]
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text.strip()

    @staticmethod
    def _cn_futures_display_name(ticker: object) -> str | None:
        base = DNAView._cn_futures_base_symbol(ticker)
        if not base:
            return None
        name = (
            CN_FUTURES_PRODUCT_NAMES_ZH.get(base)
            or CN_FUTURES_PRODUCT_NAMES_ZH.get(base.upper())
            or CN_FUTURES_PRODUCT_NAMES_ZH.get(base.lower())
        )
        if not name:
            return None
        return f"{name} ({base})"

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _cn_equity_name_lookup(base_dir: str) -> dict[str, str]:
        root = Path(base_dir) / "runtime" / "data" / "equity_cn" / "daily"
        if not root.exists():
            return {}
        files = sorted(
            root.glob("*.parquet"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        for path in files:
            try:
                frame = pd.read_parquet(path, columns=["symbol", "name"])
            except Exception:
                continue
            if frame.empty or not {"symbol", "name"}.issubset(frame.columns):
                continue
            frame = frame.dropna(subset=["symbol", "name"]).copy()
            frame["symbol"] = frame["symbol"].astype(str).str.strip()
            frame["name"] = frame["name"].astype(str).str.strip()
            frame = frame[frame["symbol"].ne("") & frame["name"].ne("")]
            if frame.empty:
                continue
            return dict(
                frame.drop_duplicates("symbol", keep="last")[["symbol", "name"]].itertuples(
                    index=False,
                    name=None,
                )
            )
        return {}

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
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown(f"#### {copy['daily_return_dist']}")
            fig = px.histogram(df, x="net_return_pct", nbins=60, template=template)
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Daily Net Return (%)")
            st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _pct(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value) * 100:.1f}%"

    @staticmethod
    def _num(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.2f}"

    @staticmethod
    def _multiple(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.2f}x"

    @staticmethod
    def _signed_pct(value) -> str:
        return "N/A" if pd.isna(value) else f"{float(value) * 100:+.2f}%"

    @staticmethod
    def _count_fraction(count, total) -> str:
        if pd.isna(count) or int(total or 0) <= 0:
            return "N/A"
        return f"{int(count)} / {int(total)}"
