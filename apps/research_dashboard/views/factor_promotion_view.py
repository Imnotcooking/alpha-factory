from __future__ import annotations

import ast
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import BASE_DIR, DB_PATH, LOGS_DIR, get_plotly_template

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from oqp.config import load_settings  # noqa: E402
from oqp.contracts import (  # noqa: E402
    load_strategy_candidate_artifacts,
    strategy_candidate_directory,
    write_candidate_from_research_db,
)
from oqp.research.factors import iter_factor_files  # noqa: E402
from oqp.research import list_evidence_tickets, update_evidence_ticket_status  # noqa: E402


STAGE_ORDER = [
    "Idea",
    "Translation Required",
    "Hypothesis Tested",
    "Calibrated",
    "Backtested",
    "Validation Candidate",
    "Paper-Trading Candidate",
    "Retired / Repair",
]

STAGE_COLORS = {
    "Idea": "#94a3b8",
    "Translation Required": "#a855f7",
    "Hypothesis Tested": "#38bdf8",
    "Calibrated": "#818cf8",
    "Backtested": "#f59e0b",
    "Validation Candidate": "#22c55e",
    "Paper-Trading Candidate": "#14b8a6",
    "Retired / Repair": "#ef4444",
}

BOARD_PRIORITY = {
    "Paper-Trading Candidate": 6,
    "Validation Candidate": 5,
    "Backtested": 4,
    "Calibrated": 3,
    "Hypothesis Tested": 2,
    "Translation Required": 1,
    "Idea": 1,
    "Retired / Repair": 0,
}

LEGACY_STAGE_MAP = {
    "OOS Candidate": "Paper-Trading Candidate",
    "Watchlist": "Paper-Trading Candidate",
}

LEGACY_NATIVE_MARKET = "FUTURES_CN"
UNSPECIFIED_MARKET = "UNSPECIFIED"


COPY = {
    "EN": {
        "title": "Factor Review",
        "subtitle": "A research review board for evidence, blockers, tickets, and candidate exports.",
        "main_tab": "Review Board",
        "drilldown_tab": "Factor Drilldown",
        "tickets_tab": "Evidence Tickets",
        "stage": "Stage",
        "category": "Category",
        "market": "Market Vertical",
        "native_market": "Native Market",
        "tested_markets": "Tested Markets",
        "vertical_status": "Vertical Status",
        "refresh": "Refresh evidence",
        "summary": "Review Summary",
        "funnel": "Lifecycle Funnel",
        "table": "Review Board",
        "drilldown": "Factor Drilldown",
        "factors": "Factors",
        "validation_candidates": "Validation Candidates",
        "paper_candidates": "Paper-Trading Candidates",
        "backtested": "Backtested",
        "avg_score": "Avg Review Score",
        "factor": "Factor",
        "score": "Review Score",
        "runs": "Runs",
        "best_sharpe": "Best Sharpe",
        "best_ic": "Best Holdout IC",
        "raw_p": "Raw p",
        "adjusted_p": "Bonf. p",
        "fdr_q": "FDR q",
        "trial_count": "Trials m",
        "evidence_tickets": "Evidence Tickets",
        "ready_tickets": "Ready Tickets",
        "reviewed_tickets": "Reviewed Tickets",
        "latest_ticket_status": "Latest Ticket",
        "best_return": "Best Ann. Return",
        "max_dd": "Best Run Max DD",
        "trades": "Trades",
        "last_run": "Last Run",
        "blockers": "Blockers",
        "next": "Suggested Next Step",
        "evidence": "Evidence Checklist",
        "runs_title": "Run History",
        "gate_title": "Promotion Gates",
        "artifact_title": "Artifacts",
        "exported_title": "Exported Candidate Snapshot Status",
        "exported_empty": "No research candidate snapshot has been exported for this factor/market row yet.",
        "exported_issues": "Candidate artifact issues",
        "recent_exports_title": "Recent Candidate Snapshots",
        "recent_exports_desc": "Latest research snapshots for the currently visible factors and market verticals. Only Paper Queue Eligible rows can feed paper-trading proposals.",
        "recent_exports_empty": "No exported candidate snapshots match the current board filters.",
        "export_title": "Export Research Candidate Snapshot",
        "export_desc": "Write the selected research run into the shared candidate contract. This makes it visible to dashboards, but it only enters the paper queue when status and market guardrails pass.",
        "export_run": "Research run",
        "export_status": "Promotion status",
        "export_target": "Target market vertical",
        "export_button": "Export research snapshot",
        "export_update_button": "Update research snapshot",
        "export_no_runs": "No backtest run exists for this market vertical yet.",
        "export_success": "Exported snapshot {candidate_id} to {path}",
        "export_warning": "Paper-candidate status should only be used after this market-specific row passes the paper-trading gate.",
        "export_error": "Candidate export failed",
        "no_data": "No factors found. Add factor files or run a backtest first.",
        "select_factor": "Inspect factor",
        "all": "All",
        "source_exists": "Source file exists",
        "metadata_exists": "Database metadata exists",
        "rationale_exists": "Economic rationale exists",
        "has_backtest": "Backtest run exists",
        "has_returns": "Return log exists",
        "has_trades": "Trade ledger exists",
        "has_importance": "Feature importance exists",
        "has_tick_ml": "Tick ML study exists",
        "has_market_metadata": "Suitability metadata exists",
        "market_tested": "This market was tested",
        "criteria": "Gate definitions and scoring",
        "no_blockers": "No automatic blockers detected.",
        "tickets_title": "Evidence Ticket Inbox",
        "tickets_desc": "Review inbox for evidence saved by discovery and validation pages before promotion review.",
        "tickets_empty": "No evidence tickets have been saved yet. Start from Tick Event Study and click Save Evidence Ticket after a hypothesis produces events.",
        "tickets_total": "Tickets",
        "tickets_ready": "Ready for Review",
        "tickets_promote": "Promote Decisions",
        "tickets_open": "Open",
        "ticket_source": "Source Page",
        "ticket_type": "Evidence Type",
        "ticket_status": "Status",
        "ticket_decision": "Decision",
        "ticket_family": "Research Family",
        "ticket_metric": "Metric",
        "ticket_confidence": "Confidence",
        "ticket_updated": "Updated",
        "ticket_select": "Inspect ticket",
        "ticket_thesis": "Thesis",
        "ticket_metrics": "Metrics",
        "ticket_context": "Context",
        "ticket_artifacts": "Linked Artifacts",
        "ticket_metadata": "Metadata",
        "ticket_review_title": "Review Action",
        "ticket_review_note": "Reviewer note",
        "ticket_mark_reviewed": "Mark Reviewed",
        "ticket_needs_more_evidence": "Needs More Evidence",
        "ticket_archive": "Archive",
        "ticket_action_success": "Updated ticket {ticket_id} to {status}.",
        "ticket_action_error": "Could not update ticket",
        "criteria_text": """
**Avg Review Score:** the mean `Review Score` of the currently visible factors. This is a research hygiene gauge, not a trading performance metric.

**Review Score:** `evidence points - blocker penalties`, clipped to 0-100. Evidence points: source +8, DB metadata +7, rationale +7, first test +10, repeat/calibration +10, return log +13, trades +8, positive holdout IC +10, holdout IC >= 1% +7, Sharpe > 0 and <= 10 +10, positive annual return +5, drawdown better than -35% +5, survives multiple-testing penalty +8.

**Automatic Blockers:** machine-detected reasons a factor should not advance yet. Examples: no test evidence, missing return log, no trades, non-positive holdout IC, non-positive Sharpe, unrealistic Sharpe, deep drawdown, weak p-value after multiple-testing correction, latest diagnostic failure codes, or index/proxy data that still needs tradable-contract validation. Ordinary blockers subtract 7 points; serious blockers subtract 18 points.

**Research Penalty:** `Raw p` is the holdout alpha p-value. `Trials m` is the count of distinct parameter/data signatures tried inside the same research family. `Bonf. p = min(raw p * m, 1)`. A factor should survive the Bonferroni p-value before being trusted as a paper-trading candidate.

**Idea:** a factor file or metadata row exists, but no test evidence yet.

**Hypothesis Tested:** at least one backtest or tick ML study exists.

**Calibrated:** repeated rounds exist, or tick ML/calibration evidence exists.

**Backtested:** executable returns were saved and the strategy produced trades.

**Validation Candidate:** first-pass out-of-sample evidence is acceptable: return log exists, trades exist, holdout IC > 0, Sharpe > 0, max drawdown better than -35%, and no serious diagnostic blocker. The summary card counts how many visible factors currently pass this softer validation gate.

**Paper-Trading Candidate:** stronger deployment-simulation gate: holdout IC >= 1%, Sharpe between 1 and 10, annual return > 0, max drawdown better than -25%, at least 10 trades, no serious blockers, and if statistical evidence exists, Bonferroni-adjusted p <= 0.05. This is the bucket for simulated paper trading, not live deployment.

**Market Vertical:** promotion is now evaluated per tested or declared market. A Chinese futures candidate does not automatically become a US equity candidate. Untested cross-market rows are marked `Translation Required` until they have their own vertical backtest evidence.

**Retired / Repair:** evidence exists, but diagnostics or repeated weak OOS metrics say the factor needs redesign.
""",
    },
    "ZH": {
        "title": "因子审查",
        "subtitle": "集中审查因子证据、阻碍项、证据票据与候选快照。",
        "main_tab": "审查看板",
        "drilldown_tab": "因子明细",
        "tickets_tab": "证据票据",
        "stage": "阶段",
        "category": "类别",
        "market": "市场垂直",
        "native_market": "原生市场",
        "tested_markets": "已测试市场",
        "vertical_status": "垂直状态",
        "refresh": "刷新证据",
        "summary": "审查总览",
        "funnel": "生命周期漏斗",
        "table": "审查看板",
        "drilldown": "因子明细",
        "factors": "因子数",
        "validation_candidates": "验证候选",
        "paper_candidates": "模拟盘候选",
        "backtested": "已回测",
        "avg_score": "平均审查分数",
        "factor": "因子",
        "score": "审查分数",
        "runs": "回测次数",
        "best_sharpe": "最佳夏普",
        "best_ic": "最佳样本外 IC",
        "raw_p": "原始 p 值",
        "adjusted_p": "Bonf. p",
        "fdr_q": "FDR q",
        "trial_count": "试验数 m",
        "evidence_tickets": "证据票据",
        "ready_tickets": "待审票据",
        "reviewed_tickets": "已审票据",
        "latest_ticket_status": "最新票据",
        "best_return": "最佳年化收益",
        "max_dd": "最佳回撤",
        "trades": "交易数",
        "last_run": "最近运行",
        "blockers": "阻碍项",
        "next": "建议下一步",
        "evidence": "证据清单",
        "runs_title": "运行历史",
        "gate_title": "晋级门槛",
        "artifact_title": "产物证据",
        "exported_title": "已导出的候选快照状态",
        "exported_empty": "该因子/市场行还没有导出研究候选快照。",
        "exported_issues": "候选产物问题",
        "recent_exports_title": "最近候选快照",
        "recent_exports_desc": "当前筛选范围内因子和市场垂直对应的最新研究快照。只有 Paper Queue Eligible 的行才可以进入模拟盘提案流程。",
        "recent_exports_empty": "当前看板筛选范围内没有匹配的候选快照。",
        "export_title": "导出研究候选快照",
        "export_desc": "把选中的研究运行写入共享候选合约，让各个看板可见；只有状态和市场保护条件都通过时，才会进入模拟盘队列。",
        "export_run": "研究运行",
        "export_status": "晋级状态",
        "export_target": "目标市场垂直",
        "export_button": "导出研究快照",
        "export_update_button": "更新研究快照",
        "export_no_runs": "该市场垂直还没有回测运行。",
        "export_success": "已导出快照 {candidate_id} 到 {path}",
        "export_warning": "只有当该市场行通过模拟盘门槛后，才应使用 paper-candidate 状态。",
        "export_error": "候选导出失败",
        "no_data": "未找到因子。请先添加因子文件或运行回测。",
        "select_factor": "查看因子",
        "all": "全部",
        "source_exists": "源码文件存在",
        "metadata_exists": "数据库元数据存在",
        "rationale_exists": "经济逻辑存在",
        "has_backtest": "已有回测运行",
        "has_returns": "已有收益日志",
        "has_trades": "已有交易明细",
        "has_importance": "已有特征重要性",
        "has_tick_ml": "已有 Tick ML 研究",
        "has_market_metadata": "已有适用市场元数据",
        "market_tested": "该市场已测试",
        "criteria": "门槛定义与打分逻辑",
        "no_blockers": "未发现自动 blocker。",
        "tickets_title": "证据票据收件箱",
        "tickets_desc": "集中审阅由发现和验证页面保存的证据票据，在因子晋级前完成状态确认。",
        "tickets_empty": "还没有保存证据票据。可以先到 Tick Event Study，在假设产生事件后点击保存证据票据。",
        "tickets_total": "票据数",
        "tickets_ready": "待审阅",
        "tickets_promote": "晋级决策",
        "tickets_open": "打开中",
        "ticket_source": "来源页面",
        "ticket_type": "证据类型",
        "ticket_status": "状态",
        "ticket_decision": "决策",
        "ticket_family": "研究家族",
        "ticket_metric": "指标",
        "ticket_confidence": "置信度",
        "ticket_updated": "更新时间",
        "ticket_select": "查看票据",
        "ticket_thesis": "研究假设",
        "ticket_metrics": "指标",
        "ticket_context": "上下文",
        "ticket_artifacts": "关联产物",
        "ticket_metadata": "元数据",
        "ticket_review_title": "审阅操作",
        "ticket_review_note": "审阅备注",
        "ticket_mark_reviewed": "标记已审阅",
        "ticket_needs_more_evidence": "需要更多证据",
        "ticket_archive": "归档",
        "ticket_action_success": "已将票据 {ticket_id} 更新为 {status}。",
        "ticket_action_error": "无法更新票据",
        "criteria_text": """
**平均审查分数：** 当前可见因子的 `审查分数` 均值。它衡量研究池证据完整度/研究质量，不是交易收益指标。

**审查分数：** `证据加分 - blocker 扣分`，最终限制在 0-100。证据加分：源码 +8、数据库元数据 +7、经济逻辑 +7、首次测试 +10、重复测试/校准 +10、收益日志 +13、有交易 +8、holdout IC 为正 +10、holdout IC >= 1% +7、Sharpe > 0 且 <= 10 +10、年化收益为正 +5、回撤好于 -35% +5、通过多重检验惩罚 +8。

**自动 blocker：** 机器识别出的暂时不能晋级原因，例如没有测试证据、没有收益日志、没有交易、样本外 IC 非正、Sharpe 非正、Sharpe 不现实、回撤过深、多重检验修正后 p 值太弱、最近诊断失败，或当前回测仍然基于指数/代理数据、尚未用可交易合约验证。普通 blocker 扣 7 分；严重 blocker 扣 18 分。

**研究惩罚：** `原始 p 值` 是 holdout 上的 alpha 证据；`试验数 m` 是同一研究家族里尝试过的不同参数/数据签名数量；`Bonf. p = min(原始 p 值 * m, 1)`。如果一个因子想进入模拟盘候选，最好先通过 Bonferroni 修正后的 p <= 0.05。

**Idea / 想法：** 有因子源码或数据库元数据，但还没有测试证据。

**Hypothesis Tested / 已验证假设：** 至少有一次回测或 tick ML 研究。

**Calibrated / 已校准：** 有多轮实验，或存在 tick ML / 参数校准证据。

**Backtested / 已回测：** 已保存可执行收益曲线，并且策略产生了交易。

**Validation Candidate / 验证候选：** 初步样本外证据可接受：有收益日志、有交易、holdout IC > 0、Sharpe > 0、最大回撤好于 -35%，且没有严重诊断 blocker。顶部指标卡统计当前可见因子里有多少通过了这个较宽松的验证门槛。

**Paper-Trading Candidate / 模拟盘候选：** 更严格的模拟部署门槛：holdout IC >= 1%、Sharpe 在 1 到 10 之间、年化收益 > 0、最大回撤好于 -25%、至少 10 笔交易、没有严重 blocker；如果已有统计证据，则 Bonferroni 修正 p <= 0.05。这个类别适合进入模拟盘观察，不等于实盘上线。

**市场垂直：** 现在按已测试或已声明的市场分别评估晋级状态。中国期货上的候选因子不会自动变成美股候选。跨市场但尚未验证的行会标记为 `Translation Required`，直到该市场有自己的垂直回测证据。

**Retired / Repair / 修复或退役：** 已有证据，但诊断或多轮弱样本外表现说明该因子需要重构。
""",
    },
}


@dataclass
class FactorEvidence:
    evidence_key: str
    factor_id: str
    name: str
    category: str
    economic_rationale: str
    complexity_score: float | None
    market_vertical: str
    native_market: str
    tested_markets: tuple[str, ...]
    suitable_markets: tuple[str, ...]
    experimental_markets: tuple[str, ...]
    unsupported_markets: tuple[str, ...]
    required_fields: tuple[str, ...]
    suitability_declared: bool
    source_path: str | None
    source_exists: bool
    metadata_exists: bool
    runs: pd.DataFrame
    tick_ml_rows: pd.DataFrame
    feature_importance_exists: bool
    return_logs: int
    trade_logs: int


class FactorPromotionView:
    """Automatic research lifecycle board for factors and tick hypotheses."""

    def __init__(self, db_path: str = DB_PATH, base_dir: str = BASE_DIR):
        self.db_path = db_path
        self.base_dir = Path(base_dir)
        self.factors_dir = self.base_dir / "factors"
        self.logs_dir = Path(LOGS_DIR)

    def render(self, lang: str = "EN", theme_mode: str = "LIGHT") -> None:
        copy = COPY.get(lang, COPY["EN"])
        tpl = get_plotly_template(theme_mode)

        title_col, refresh_col = st.columns([0.78, 0.22], vertical_alignment="top")
        with title_col:
            st.title(copy["title"])
            st.caption(copy["subtitle"])
        with refresh_col:
            st.write("")
            if st.button(copy["refresh"], use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        board, detail = self._load_board()
        board = self._normalize_board_stages(board)
        if board.empty:
            st.warning(copy["no_data"])
            return

        filtered = self._render_filters(board, copy)
        if filtered.empty:
            st.info("No factors match the current filters.")
            return

        board_tab, drilldown_tab, tickets_tab = st.tabs(
            [copy["main_tab"], copy["drilldown_tab"], copy["tickets_tab"]]
        )
        with board_tab:
            self._render_summary(filtered, tpl, copy)
            st.markdown("---")
            self._render_recent_candidate_exports(filtered, copy)
            st.markdown("---")
            self._render_board(filtered, copy)
        with drilldown_tab:
            self._render_drilldown(filtered, detail, tpl, copy)
        with tickets_tab:
            self._render_evidence_ticket_inbox(copy)

    def _render_filters(self, board: pd.DataFrame, copy: dict) -> pd.DataFrame:
        cols = st.columns(3)
        stage_options = [copy["all"], *[stage for stage in STAGE_ORDER if stage in set(board["stage"])]]
        category_options = [copy["all"], *sorted(board["category"].dropna().astype(str).unique())]
        market_options = [copy["all"], *sorted(board["market_vertical"].dropna().astype(str).unique())]
        selected_stage = cols[0].selectbox(copy["stage"], stage_options)
        selected_category = cols[1].selectbox(copy["category"], category_options)
        selected_market = cols[2].selectbox(copy["market"], market_options)

        filtered = board.copy()
        if selected_stage != copy["all"]:
            filtered = filtered[filtered["stage"] == selected_stage]
        if selected_category != copy["all"]:
            filtered = filtered[filtered["category"] == selected_category]
        if selected_market != copy["all"]:
            filtered = filtered[filtered["market_vertical"] == selected_market]
        return filtered

    def _render_summary(self, board: pd.DataFrame, tpl: str, copy: dict) -> None:
        st.markdown(f"### {copy['summary']}")
        cards = st.columns(5)
        cards[0].metric(copy["factors"], f"{len(board):,}")
        cards[1].metric(copy["paper_candidates"], f"{(board['stage'] == 'Paper-Trading Candidate').sum():,}")
        cards[2].metric(copy["validation_candidates"], f"{(board['stage'] == 'Validation Candidate').sum():,}")
        cards[3].metric(copy["backtested"], f"{board['has_returns'].sum():,}")
        cards[4].metric(copy["avg_score"], f"{board['promotion_score'].mean():.0f}/100")

        counts = (
            board.assign(stage_order=board["stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER)}))
            .groupby(["stage", "stage_order"], as_index=False)
            .agg(count=("factor_id", "count"))
            .sort_values("stage_order")
        )
        fig = px.bar(
            counts,
            x="stage",
            y="count",
            color="stage",
            color_discrete_map=STAGE_COLORS,
            template=tpl,
            title=copy["funnel"],
        )
        fig.update_layout(height=330, showlegend=False, margin=dict(l=10, r=10, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"#### {copy['criteria']}")
        st.markdown(copy["criteria_text"])

    def _render_board(self, board: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['table']}")
        show_cols = [
            "stage",
            "promotion_score",
            "market_vertical",
            "native_market",
            "vertical_status",
            "factor_id",
            "name",
            "category",
            "evidence_ticket_count",
            "ready_ticket_count",
            "reviewed_ticket_count",
            "latest_ticket_status",
            "run_count",
            "best_holdout_ic",
            "best_raw_p_value",
            "best_adjusted_p_value",
            "best_fdr_q_value",
            "stat_trial_count",
            "best_sharpe",
            "best_annualized_return",
            "best_max_drawdown",
            "total_trades",
            "last_run_at",
            "blockers",
            "suggested_next_step",
        ]
        display = board[show_cols].rename(
            columns={
                "stage": copy["stage"],
                "promotion_score": copy["score"],
                "market_vertical": copy["market"],
                "native_market": copy["native_market"],
                "vertical_status": copy["vertical_status"],
                "factor_id": "factor_id",
                "name": copy["factor"],
                "category": copy["category"],
                "evidence_ticket_count": copy["evidence_tickets"],
                "ready_ticket_count": copy["ready_tickets"],
                "reviewed_ticket_count": copy["reviewed_tickets"],
                "latest_ticket_status": copy["latest_ticket_status"],
                "run_count": copy["runs"],
                "best_holdout_ic": copy["best_ic"],
                "best_raw_p_value": copy["raw_p"],
                "best_adjusted_p_value": copy["adjusted_p"],
                "best_fdr_q_value": copy["fdr_q"],
                "stat_trial_count": copy["trial_count"],
                "best_sharpe": copy["best_sharpe"],
                "best_annualized_return": copy["best_return"],
                "best_max_drawdown": copy["max_dd"],
                "total_trades": copy["trades"],
                "last_run_at": copy["last_run"],
                "blockers": copy["blockers"],
                "suggested_next_step": copy["next"],
            }
        )
        styled = display.style.format(
            {
                copy["score"]: "{:.0f}",
                copy["evidence_tickets"]: "{:.0f}",
                copy["ready_tickets"]: "{:.0f}",
                copy["reviewed_tickets"]: "{:.0f}",
                copy["best_ic"]: "{:.4f}",
                copy["raw_p"]: "{:.4f}",
                copy["adjusted_p"]: "{:.4f}",
                copy["fdr_q"]: "{:.4f}",
                copy["trial_count"]: "{:.0f}",
                copy["best_sharpe"]: "{:.2f}",
                copy["best_return"]: "{:.1%}",
                copy["max_dd"]: "{:.1%}",
                copy["trades"]: "{:.0f}",
            },
            na_rep="",
        ).apply(self._style_stage, subset=[copy["stage"]])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=520)

    def _render_recent_candidate_exports(self, board: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['recent_exports_title']}")
        st.caption(copy["recent_exports_desc"])
        exports, issues = self._recent_candidate_export_summary(board)
        if exports.empty:
            st.info(copy["recent_exports_empty"])
        else:
            st.dataframe(exports, use_container_width=True, hide_index=True)

        if issues:
            with st.expander(copy["exported_issues"], expanded=False):
                issue_df = pd.DataFrame(
                    [
                        {
                            "artifact": self._display_repo_path(issue.path),
                            "issue": issue.message,
                        }
                        for issue in issues
                    ]
                )
                st.dataframe(issue_df, use_container_width=True, hide_index=True)

    def _render_evidence_ticket_inbox(self, copy: dict) -> None:
        st.markdown(f"### {copy['tickets_title']}")
        st.caption(copy["tickets_desc"])
        flash = st.session_state.pop("evidence_ticket_flash", "")
        if flash:
            st.success(flash)

        tickets = self._load_evidence_tickets()
        if tickets.empty:
            st.info(copy["tickets_empty"])
            return

        tickets = self._prepare_ticket_frame(tickets)
        summary_cols = st.columns(4)
        summary_cols[0].metric(copy["tickets_total"], f"{len(tickets):,}")
        summary_cols[1].metric(copy["tickets_ready"], f"{(tickets['status'] == 'ready_for_review').sum():,}")
        summary_cols[2].metric(copy["tickets_promote"], f"{(tickets['decision'] == 'promote_to_validation').sum():,}")
        summary_cols[3].metric(copy["tickets_open"], f"{(tickets['status'] == 'open').sum():,}")

        filters = st.columns(4)
        all_label = copy["all"]
        selected_status = filters[0].selectbox(
            copy["ticket_status"],
            [all_label, *sorted(tickets["status"].dropna().astype(str).unique())],
            key="evidence-ticket-status-filter",
        )
        selected_source = filters[1].selectbox(
            copy["ticket_source"],
            [all_label, *sorted(tickets["source_page"].dropna().astype(str).unique())],
            key="evidence-ticket-source-filter",
        )
        selected_decision = filters[2].selectbox(
            copy["ticket_decision"],
            [all_label, *sorted(tickets["decision"].dropna().astype(str).unique())],
            key="evidence-ticket-decision-filter",
        )
        selected_family = filters[3].selectbox(
            copy["ticket_family"],
            [all_label, *sorted(tickets["research_family"].dropna().astype(str).unique())],
            key="evidence-ticket-family-filter",
        )

        visible = tickets.copy()
        if selected_status != all_label:
            visible = visible[visible["status"] == selected_status]
        if selected_source != all_label:
            visible = visible[visible["source_page"] == selected_source]
        if selected_decision != all_label:
            visible = visible[visible["decision"] == selected_decision]
        if selected_family != all_label:
            visible = visible[visible["research_family"] == selected_family]

        display_cols = [
            "status",
            "decision",
            "source_page",
            "evidence_type",
            "stage",
            "factor_id",
            "research_family",
            "metric_name",
            "metric_value",
            "confidence_score",
            "updated_at",
        ]
        display = visible[[col for col in display_cols if col in visible.columns]].rename(
            columns={
                "status": copy["ticket_status"],
                "decision": copy["ticket_decision"],
                "source_page": copy["ticket_source"],
                "evidence_type": copy["ticket_type"],
                "stage": copy["stage"],
                "factor_id": "factor_id",
                "research_family": "research_family",
                "metric_name": copy["ticket_metric"],
                "metric_value": "value",
                "confidence_score": copy["ticket_confidence"],
                "updated_at": copy["ticket_updated"],
            }
        )
        st.dataframe(
            display.style.format(
                {
                    "value": "{:.4f}",
                    copy["ticket_confidence"]: "{:.2f}",
                },
                na_rep="",
            ),
            use_container_width=True,
            hide_index=True,
            height=320,
        )
        if visible.empty:
            return

        labels = [
            f"{row.ticket_id} | {row.source_page} | {row.decision}"
            for row in visible.itertuples(index=False)
        ]
        selected = st.selectbox(copy["ticket_select"], labels)
        ticket_id = selected.split(" | ", 1)[0]
        ticket = visible[visible["ticket_id"] == ticket_id].iloc[0]

        st.markdown(f"#### {ticket['title']}")
        st.caption(
            f"{copy['ticket_status']}: `{ticket['status']}` | "
            f"{copy['ticket_decision']}: `{ticket['decision']}` | "
            f"{copy['ticket_source']}: `{ticket['source_page']}`"
        )
        self._render_ticket_review_actions(copy, ticket)
        with st.expander(copy["ticket_thesis"], expanded=True):
            st.write(ticket.get("thesis") or "")
        with st.expander(copy["ticket_metrics"], expanded=False):
            st.json(ticket.get("metrics") or {}, expanded=True)
        with st.expander(copy["ticket_context"], expanded=False):
            st.json(ticket.get("context") or {}, expanded=True)
        with st.expander(copy["ticket_artifacts"], expanded=False):
            artifacts = ticket.get("artifacts") or []
            if artifacts:
                st.dataframe(pd.DataFrame(artifacts), use_container_width=True, hide_index=True)
            else:
                st.write([])
        with st.expander(copy["ticket_metadata"], expanded=False):
            st.json(ticket.get("metadata") or {}, expanded=True)

    def _render_ticket_review_actions(self, copy: dict, ticket: pd.Series) -> None:
        ticket_id = str(ticket.get("ticket_id") or "")
        current_status = str(ticket.get("status") or "")
        st.markdown(f"#### {copy['ticket_review_title']}")
        note = st.text_area(
            copy["ticket_review_note"],
            key=f"evidence-ticket-review-note-{ticket_id}",
            height=84,
        )
        cols = st.columns(3)
        actions = [
            (
                cols[0],
                copy["ticket_mark_reviewed"],
                "reviewed",
                current_status not in {"ready_for_review", "needs_more_evidence"},
            ),
            (
                cols[1],
                copy["ticket_needs_more_evidence"],
                "needs_more_evidence",
                current_status not in {"open", "ready_for_review"},
            ),
            (
                cols[2],
                copy["ticket_archive"],
                "archived",
                current_status not in {"open", "ready_for_review", "needs_more_evidence", "reviewed"},
            ),
        ]
        for col, label, next_status, disabled in actions:
            if col.button(
                label,
                key=f"evidence-ticket-action-{next_status}-{ticket_id}",
                use_container_width=True,
                disabled=disabled,
            ):
                self._apply_ticket_review_action(copy, ticket_id, next_status, note)

    def _apply_ticket_review_action(
        self,
        copy: dict,
        ticket_id: str,
        status: str,
        reviewer_note: str,
    ) -> None:
        try:
            updated = update_evidence_ticket_status(
                self.db_path,
                ticket_id,
                status=status,
                reviewer_note=reviewer_note,
                metadata_patch={"review_source_page": "08_Factor_Review"},
                reviewer="factor_promotion_pipeline",
            )
        except Exception as exc:
            st.error(f"{copy['ticket_action_error']}: {exc}")
            return

        st.session_state["evidence_ticket_flash"] = copy["ticket_action_success"].format(
            ticket_id=ticket_id,
            status=updated.get("status", status),
        )
        st.cache_data.clear()
        st.rerun()

    def _render_drilldown(self, board: pd.DataFrame, detail: dict[str, FactorEvidence], tpl: str, copy: dict) -> None:
        st.markdown(f"### {copy['drilldown']}")
        labels = [
            f"{row.factor_id} | {row.market_vertical} | {row.stage} | Score {row.promotion_score:.0f}"
            for row in board.itertuples(index=False)
        ]
        label_to_key = dict(zip(labels, board["evidence_key"]))
        selected_label = st.selectbox(copy["select_factor"], labels)
        evidence_key = label_to_key[selected_label]
        row = board[board["evidence_key"] == evidence_key].iloc[0]
        evidence = detail[evidence_key]

        cols = st.columns(6)
        cols[0].metric(copy["stage"], row["stage"])
        cols[1].metric(copy["score"], f"{row['promotion_score']:.0f}/100")
        cols[2].metric(copy["best_sharpe"], self._fmt(row["best_sharpe"], "{:.2f}"))
        cols[3].metric(copy["best_ic"], self._fmt(row["best_holdout_ic"], "{:.4f}"))
        cols[4].metric(copy["adjusted_p"], self._fmt(row["best_adjusted_p_value"], "{:.4f}"))
        cols[5].metric(copy["trades"], f"{int(row['total_trades']):,}")

        st.markdown(f"**{row['name']}**")
        if evidence.economic_rationale:
            st.caption(evidence.economic_rationale)
        if row["blockers"]:
            st.warning(f"{copy['blockers']}: {row['blockers']}")
        else:
            st.success(copy["no_blockers"])
        st.info(f"{copy['next']}: {row['suggested_next_step']}")

        left, right = st.columns([0.46, 0.54])
        with left:
            self._render_evidence_checklist(evidence, copy)
        with right:
            self._render_artifacts(evidence, copy)

        candidate_exports, candidate_issues = self._candidate_exports_for_evidence(evidence)
        self._render_candidate_readback(candidate_exports, candidate_issues, copy)
        self._render_candidate_export(row, evidence, copy, candidate_exports)

        st.markdown(f"#### {copy['runs_title']}")
        self._render_run_history(evidence.runs, tpl)

    def _render_evidence_checklist(self, evidence: FactorEvidence, copy: dict) -> None:
        st.markdown(f"#### {copy['evidence']}")
        checks = [
            (copy["source_exists"], evidence.source_exists),
            (copy["metadata_exists"], evidence.metadata_exists),
            (copy["rationale_exists"], bool(evidence.economic_rationale)),
            (copy["has_market_metadata"], evidence.suitability_declared),
            (copy["market_tested"], not evidence.runs.empty),
            (copy["has_backtest"], not evidence.runs.empty),
            (copy["has_returns"], evidence.return_logs > 0),
            (copy["has_trades"], evidence.trade_logs > 0),
            (copy["has_importance"], evidence.feature_importance_exists),
            (copy["has_tick_ml"], not evidence.tick_ml_rows.empty),
        ]
        checklist = pd.DataFrame(
            {
                "Evidence": [label for label, _ in checks],
                "Status": ["PASS" if ok else "MISSING" for _, ok in checks],
            }
        )
        styled = checklist.style.map(
            lambda value: "background-color: #dcfce7; color: #166534"
            if value == "PASS"
            else "background-color: #fee2e2; color: #991b1b",
            subset=["Status"],
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    def _render_artifacts(self, evidence: FactorEvidence, copy: dict) -> None:
        st.markdown(f"#### {copy['artifact_title']}")
        artifacts = [
            ("source_path", evidence.source_path or ""),
            ("market_vertical", evidence.market_vertical),
            ("native_market", evidence.native_market),
            ("tested_markets", ", ".join(evidence.tested_markets)),
            ("suitable_markets", ", ".join(evidence.suitable_markets)),
            ("experimental_markets", ", ".join(evidence.experimental_markets)),
            ("unsupported_markets", ", ".join(evidence.unsupported_markets)),
            ("required_fields", ", ".join(evidence.required_fields)),
            ("return_logs", evidence.return_logs),
            ("trade_logs", evidence.trade_logs),
            ("tick_ml_studies", len(evidence.tick_ml_rows)),
            ("feature_importance", "yes" if evidence.feature_importance_exists else "no"),
        ]
        artifact_df = pd.DataFrame(artifacts, columns=["artifact", "value"])
        artifact_df["value"] = artifact_df["value"].astype(str)
        st.dataframe(artifact_df, use_container_width=True, hide_index=True)

        if not evidence.tick_ml_rows.empty:
            with st.expander("Tick ML studies / Tick ML 研究", expanded=False):
                cols = [
                    "updated_at",
                    "symbol",
                    "hypothesis",
                    "horizon_ticks",
                    "prediction_rows",
                    "accuracy_50",
                    "roc_auc",
                ]
                tick_df = evidence.tick_ml_rows.copy()
                for col in cols:
                    if col not in tick_df.columns:
                        tick_df[col] = np.nan
                st.dataframe(tick_df[cols], use_container_width=True, hide_index=True)

    def _render_candidate_readback(
        self,
        exports: pd.DataFrame,
        issues: tuple,
        copy: dict,
    ) -> None:
        st.markdown(f"#### {copy['exported_title']}")
        if exports.empty:
            st.info(copy["exported_empty"])
        else:
            st.dataframe(exports, use_container_width=True, hide_index=True)

        if issues:
            with st.expander(copy["exported_issues"], expanded=False):
                issue_df = pd.DataFrame(
                    [
                        {
                            "artifact": self._display_repo_path(issue.path),
                            "issue": issue.message,
                        }
                        for issue in issues
                    ]
                )
                st.dataframe(issue_df, use_container_width=True, hide_index=True)

    def _render_candidate_export(
        self,
        row: pd.Series,
        evidence: FactorEvidence,
        copy: dict,
        existing_exports: pd.DataFrame,
    ) -> None:
        st.markdown(f"#### {copy['export_title']}")
        st.caption(copy["export_desc"])

        if evidence.runs.empty or "run_id" not in evidence.runs.columns:
            st.info(copy["export_no_runs"])
            return

        runs = evidence.runs.copy()
        if "timestamp" in runs.columns:
            runs = runs.sort_values("timestamp", ascending=False)
        run_options = [
            run_id
            for run_id in runs["run_id"].dropna().astype(str).drop_duplicates().tolist()
            if run_id
        ]
        if not run_options:
            st.info(copy["export_no_runs"])
            return

        status_options = [
            "research_only",
            "market_review",
            "paper_candidate",
            "paper_running",
            "rejected",
            "out_of_scope",
        ]
        cols = st.columns(3)
        selected_run_id = cols[0].selectbox(
            copy["export_run"],
            run_options,
            key=f"candidate-export-run-{evidence.evidence_key}",
        )
        selected_status = cols[1].selectbox(
            copy["export_status"],
            status_options,
            index=0,
            key=f"candidate-export-status-{evidence.evidence_key}",
        )
        target_market = cols[2].text_input(
            copy["export_target"],
            value=str(row.get("market_vertical") or evidence.market_vertical),
            key=f"candidate-export-target-{evidence.evidence_key}",
        )
        selected_run_exported = (
            not existing_exports.empty
            and selected_run_id in set(existing_exports["research_run_id"].astype(str))
        )
        export_clicked = st.button(
            copy["export_update_button"] if selected_run_exported else copy["export_button"],
            key=f"candidate-export-button-{evidence.evidence_key}",
            use_container_width=True,
        )

        if selected_status == "paper_candidate" and row.get("stage") != "Paper-Trading Candidate":
            st.warning(copy["export_warning"])

        if not export_clicked:
            return

        try:
            candidate, path = self._export_strategy_candidate(
                selected_run_id,
                selected_status,
                target_market,
            )
        except Exception as exc:
            st.error(f"{copy['export_error']}: {exc}")
            return

        st.success(
            copy["export_success"].format(
                candidate_id=candidate.candidate_id,
                path=self._display_repo_path(path),
            )
        )

    def _candidate_exports_for_evidence(
        self,
        evidence: FactorEvidence,
        *,
        directory: Path | None = None,
    ) -> tuple[pd.DataFrame, tuple]:
        run_ids = set()
        if not evidence.runs.empty and "run_id" in evidence.runs.columns:
            run_ids = {
                str(run_id)
                for run_id in evidence.runs["run_id"].dropna().astype(str)
                if str(run_id)
            }
        return self._candidate_exports_for_run_ids(
            run_ids,
            factor_id=evidence.factor_id,
            market_vertical=evidence.market_vertical,
            directory=directory,
        )

    def _recent_candidate_export_summary(
        self,
        board: pd.DataFrame,
        *,
        directory: Path | None = None,
        max_rows: int = 25,
    ) -> tuple[pd.DataFrame, tuple]:
        target_dir = directory or strategy_candidate_directory(load_settings())
        result = load_strategy_candidate_artifacts(target_dir, max_files=500)
        if board.empty:
            return pd.DataFrame(), result.issues

        visible_factors = set(board["factor_id"].dropna().astype(str))
        visible_markets = set(board["market_vertical"].dropna().astype(str))
        stage_by_pair = {
            (str(row.factor_id), str(row.market_vertical)): str(row.stage)
            for row in board.itertuples(index=False)
        }

        rows = []
        for loaded in result.loaded:
            candidate = loaded.candidate
            factor_id = candidate.strategy_id
            target_market = candidate.target_market_vertical
            if factor_id not in visible_factors:
                continue
            if target_market not in visible_markets:
                continue

            trial = (
                candidate.metadata.get("trial_signature")
                or candidate.research_run_id
                or ""
            )
            rows.append(
                {
                    "factor_id": factor_id,
                    "trial": trial,
                    "research_run_id": candidate.research_run_id or "",
                    "intake_state": candidate.intake_state_label,
                    "market_status": candidate.market_scoped_status,
                    "board_stage": stage_by_pair.get((factor_id, target_market), ""),
                    "tested_market": candidate.tested_market_vertical,
                    "target_market": target_market,
                    "paper_queue_eligible": "yes"
                    if candidate.can_enter_paper_queue
                    else "no",
                    "created_at": candidate.created_at.isoformat(timespec="seconds"),
                    "artifact": self._display_repo_path(loaded.path),
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return out, result.issues

        out["_created_at_sort"] = pd.to_datetime(out["created_at"], errors="coerce")
        out = (
            out.sort_values(["_created_at_sort", "factor_id"], ascending=[False, True])
            .drop_duplicates(["factor_id", "research_run_id", "target_market"])
            .head(max_rows)
            .drop(columns=["_created_at_sort"])
        )
        return out, result.issues

    def _candidate_exports_for_run_ids(
        self,
        run_ids: set[str],
        *,
        factor_id: str | None = None,
        market_vertical: str | None = None,
        directory: Path | None = None,
    ) -> tuple[pd.DataFrame, tuple]:
        target_dir = directory or strategy_candidate_directory(load_settings())
        result = load_strategy_candidate_artifacts(target_dir, max_files=500)
        rows = []
        for loaded in result.loaded:
            candidate = loaded.candidate
            if run_ids and str(candidate.research_run_id) not in run_ids:
                continue
            if factor_id and candidate.strategy_id != factor_id:
                continue
            if market_vertical and candidate.target_market_vertical != market_vertical:
                continue
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "research_run_id": candidate.research_run_id or "",
                    "intake_state": candidate.intake_state_label,
                    "market_status": candidate.market_scoped_status,
                    "status": candidate.promotion_status.value,
                    "tested_market": candidate.tested_market_vertical,
                    "target_market": candidate.target_market_vertical,
                    "paper_queue_eligible": "yes"
                    if candidate.can_enter_paper_queue
                    else "no",
                    "created_at": candidate.created_at.isoformat(timespec="seconds"),
                    "artifact": self._display_repo_path(loaded.path),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["created_at", "candidate_id"], ascending=[False, True])
        return out, result.issues

    def _export_strategy_candidate(
        self,
        run_id: str,
        status: str,
        target_market_vertical: str,
        *,
        output_dir: Path | None = None,
    ):
        return write_candidate_from_research_db(
            Path(self.db_path),
            output_dir=output_dir,
            run_id=run_id,
            status=status,
            target_market_vertical=target_market_vertical,
            overwrite=True,
        )

    @staticmethod
    def _display_repo_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def _render_run_history(self, runs: pd.DataFrame, tpl: str) -> None:
        if runs.empty:
            st.info("No backtest runs yet.")
            return

        plot_df = runs.copy()
        plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
        plot_df["round_number"] = pd.to_numeric(plot_df["round_number"], errors="coerce")
        metric = "sharpe_ratio" if plot_df["sharpe_ratio"].notna().any() else "holdout_ic"
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"].fillna(plot_df["round_number"]),
                y=plot_df[metric],
                mode="lines+markers",
                name=metric,
                text=plot_df["run_id"],
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(template=tpl, height=310, margin=dict(l=10, r=10, t=20, b=20), yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

        show_cols = [
            "run_id",
            "round_number",
            "timestamp",
            "asset_class",
            "market_vertical",
            "dataset_id",
            "universe_id",
            "data_frequency",
            "data_vendor",
            "execution_assumption",
            "universe_size",
            "traded_tickers",
            "validation_ic",
            "holdout_ic",
            "crisis_ic",
            "annualized_return",
            "sharpe_ratio",
            "stat_raw_p_value",
            "stat_adjusted_p_value",
            "stat_fdr_q_value",
            "stat_trial_count",
            "stat_significance",
            "max_drawdown",
            "turnover_rate",
            "total_trades",
            "failure_code",
            "suggested_action",
        ]
        for col in show_cols:
            if col not in runs.columns:
                runs[col] = np.nan
        st.dataframe(
            runs[show_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    @st.cache_data(show_spinner=False)
    def _load_evidence_tickets(_self) -> pd.DataFrame:
        try:
            return list_evidence_tickets(_self.db_path, parse_json=True)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _prepare_ticket_frame(tickets: pd.DataFrame) -> pd.DataFrame:
        out = tickets.copy()
        defaults = {
            "ticket_id": "",
            "title": "",
            "source_page": "",
            "evidence_type": "",
            "stage": "",
            "status": "",
            "decision": "",
            "thesis": "",
            "factor_id": "",
            "research_family": "",
            "run_id": "",
            "trial_signature": "",
            "metric_name": "",
            "metric_value": np.nan,
            "confidence_score": np.nan,
            "metrics": {},
            "artifacts": [],
            "context": {},
            "metadata": {},
            "created_at": pd.NaT,
            "updated_at": pd.NaT,
        }
        for col, default in defaults.items():
            if col not in out.columns:
                if isinstance(default, dict):
                    out[col] = [dict(default) for _ in range(len(out))]
                elif isinstance(default, list):
                    out[col] = [list(default) for _ in range(len(out))]
                else:
                    out[col] = default
        for col in ["created_at", "updated_at"]:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        for col in ["metric_value", "confidence_score"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out["updated_at"] = out["updated_at"].fillna(out["created_at"])
        return out.sort_values(["updated_at", "ticket_id"], ascending=[False, True])

    @staticmethod
    def _empty_ticket_summary() -> dict:
        return {
            "evidence_ticket_count": 0,
            "ready_ticket_count": 0,
            "reviewed_ticket_count": 0,
            "open_ticket_count": 0,
            "archived_ticket_count": 0,
            "needs_more_evidence_ticket_count": 0,
            "latest_ticket_status": "",
            "latest_ticket_updated_at": pd.NaT,
        }

    @staticmethod
    def _ticket_summary_by_factor(tickets: pd.DataFrame) -> pd.DataFrame:
        columns = ["factor_id", *FactorPromotionView._empty_ticket_summary().keys()]
        if tickets.empty:
            return pd.DataFrame(columns=columns)

        normalized = FactorPromotionView._prepare_ticket_frame(tickets)
        if normalized.empty:
            return pd.DataFrame(columns=columns)

        normalized["factor_id"] = normalized["factor_id"].fillna("").astype(str).str.strip()
        normalized["status"] = normalized["status"].fillna("").astype(str).str.strip()
        normalized = normalized[normalized["factor_id"] != ""].copy()
        if normalized.empty:
            return pd.DataFrame(columns=columns)

        summary = normalized.groupby("factor_id", dropna=False).agg(
            evidence_ticket_count=("ticket_id", "nunique")
        )
        status_counts = pd.crosstab(normalized["factor_id"], normalized["status"])
        for status, column in (
            ("ready_for_review", "ready_ticket_count"),
            ("reviewed", "reviewed_ticket_count"),
            ("open", "open_ticket_count"),
            ("archived", "archived_ticket_count"),
            ("needs_more_evidence", "needs_more_evidence_ticket_count"),
        ):
            summary[column] = status_counts[status] if status in status_counts else 0

        latest = (
            normalized.sort_values(["updated_at", "ticket_id"], ascending=[False, True])
            .drop_duplicates("factor_id")
            .set_index("factor_id")
        )
        summary["latest_ticket_status"] = latest["status"]
        summary["latest_ticket_updated_at"] = latest["updated_at"]
        summary = summary.reset_index()
        for column in [
            "evidence_ticket_count",
            "ready_ticket_count",
            "reviewed_ticket_count",
            "open_ticket_count",
            "archived_ticket_count",
            "needs_more_evidence_ticket_count",
        ]:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)
        return summary[columns]

    @st.cache_data(show_spinner=False)
    def _load_board(_self) -> tuple[pd.DataFrame, dict[str, FactorEvidence]]:
        factors = _self._load_factor_inventory()
        runs = _self._load_runs()
        tick_ml = _self._load_tick_ml_studies()
        importance_index = _self._feature_importance_index()
        ticket_summary = _self._ticket_summary_by_factor(_self._load_evidence_tickets())
        ticket_by_factor = (
            ticket_summary.set_index("factor_id").to_dict("index")
            if not ticket_summary.empty
            else {}
        )
        evidence_by_factor = {}
        rows = []

        for factor_id, metadata in factors.items():
            all_factor_runs = runs[runs["factor_id"] == factor_id].copy() if not runs.empty else pd.DataFrame()
            tested_markets = _self._tested_markets(all_factor_runs)
            native_market = _self._native_market(metadata, tested_markets)
            market_rows = _self._market_rows(metadata, tested_markets, native_market)

            for market_vertical in market_rows:
                factor_runs = _self._filter_runs_for_market(all_factor_runs, market_vertical)
                factor_tick_ml = _self._match_tick_ml(factor_id, tick_ml)
                return_logs = _self._count_return_logs(factor_id, factor_runs)
                trade_logs = _self._count_trade_logs(factor_runs)
                feature_importance_exists = factor_id in importance_index or any(
                    str(run_id) in importance_index
                    for run_id in factor_runs.get("run_id", pd.Series(dtype=str)).astype(str)
                )
                evidence_key = f"{factor_id}::{market_vertical}"
                evidence = FactorEvidence(
                    evidence_key=evidence_key,
                    factor_id=factor_id,
                    name=metadata.get("name") or factor_id,
                    category=metadata.get("category") or "Uncategorized",
                    economic_rationale=metadata.get("economic_rationale") or "",
                    complexity_score=metadata.get("complexity_score"),
                    market_vertical=market_vertical,
                    native_market=native_market,
                    tested_markets=tested_markets,
                    suitable_markets=tuple(metadata.get("suitable_markets", ())),
                    experimental_markets=tuple(metadata.get("experimental_markets", ())),
                    unsupported_markets=tuple(metadata.get("unsupported_markets", ())),
                    required_fields=tuple(metadata.get("required_fields", ())),
                    suitability_declared=bool(metadata.get("suitability_declared")),
                    source_path=metadata.get("source_path"),
                    source_exists=bool(metadata.get("source_exists")),
                    metadata_exists=bool(metadata.get("metadata_exists")),
                    runs=factor_runs,
                    tick_ml_rows=factor_tick_ml,
                    feature_importance_exists=feature_importance_exists,
                    return_logs=return_logs,
                    trade_logs=trade_logs,
                )
                evidence_by_factor[evidence_key] = evidence
                row = _self._score_factor(evidence)
                row.update(_self._empty_ticket_summary())
                row.update(ticket_by_factor.get(factor_id, {}))
                rows.append(row)

        board = pd.DataFrame(rows)
        if not board.empty:
            board["stage_rank"] = board["stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER)}).fillna(0)
            board["board_priority"] = board["stage"].map(BOARD_PRIORITY).fillna(0)
            board = board.sort_values(
                ["board_priority", "promotion_score", "best_sharpe"],
                ascending=[False, False, False],
            )
        return board, evidence_by_factor

    def _normalize_board_stages(self, board: pd.DataFrame) -> pd.DataFrame:
        if board.empty or "stage" not in board.columns:
            return board

        out = board.copy()
        out["stage"] = out["stage"].replace(LEGACY_STAGE_MAP)
        out["stage_rank"] = out["stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER)}).fillna(0)
        out["board_priority"] = out["stage"].map(BOARD_PRIORITY).fillna(0)
        sort_cols = [col for col in ["board_priority", "promotion_score", "best_sharpe"] if col in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        return out

    def _load_factor_inventory(self) -> dict[str, dict]:
        inventory: dict[str, dict] = {}
        for path in iter_factor_files():
            meta = self._parse_factor_file(path)
            factor_id = meta.get("factor_id") or path.stem
            try:
                source_path = path.relative_to(REPO_ROOT)
            except ValueError:
                try:
                    source_path = path.relative_to(self.base_dir)
                except ValueError:
                    source_path = path
            meta.update(
                {
                    "factor_id": factor_id,
                    "source_path": str(source_path),
                    "source_exists": True,
                    "metadata_exists": False,
                }
            )
            inventory[factor_id] = meta

        db_factors = self._read_sql("SELECT * FROM factors")
        if not db_factors.empty:
            for _, row in db_factors.iterrows():
                factor_id = str(row.get("factor_id", "")).strip()
                if not factor_id:
                    continue
                current = inventory.get(factor_id, {})
                current.update(
                    {
                        "factor_id": factor_id,
                        "name": self._coalesce(row.get("name"), current.get("name"), factor_id),
                        "category": self._coalesce(row.get("category"), current.get("category"), "Uncategorized"),
                        "economic_rationale": self._coalesce(
                            row.get("economic_rationale"),
                            current.get("economic_rationale"),
                            "",
                        ),
                        "complexity_score": self._coalesce(row.get("complexity_score"), current.get("complexity_score")),
                        "metadata_exists": True,
                        "source_exists": bool(current.get("source_exists", False)),
                        "source_path": current.get("source_path"),
                    }
                )
                inventory[factor_id] = current
        return inventory

    def _parse_factor_file(self, path: Path) -> dict:
        meta = {
            "factor_id": path.stem,
            "name": path.stem,
            "category": "Uncategorized",
            "economic_rationale": "",
            "complexity_score": None,
            "native_market": None,
            "suitable_markets": (),
            "experimental_markets": (),
            "unsupported_markets": (),
            "required_fields": (),
            "suitability_declared": False,
        }
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            return meta

        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            constants[target.id] = ast.literal_eval(node.value)
                        except Exception:
                            pass

        meta["factor_id"] = str(constants.get("FACTOR_ID") or path.stem)
        meta["name"] = str(constants.get("NAME_EN") or constants.get("FACTOR_NAME") or constants.get("NAME") or meta["factor_id"])
        meta["category"] = str(constants.get("CATEGORY") or meta["category"])
        meta["economic_rationale"] = str(
            constants.get("ECONOMIC_RATIONALE_EN")
            or constants.get("ECONOMIC_RATIONALE")
            or constants.get("RATIONALE")
            or ""
        )
        complexity = constants.get("COMPLEXITY") or constants.get("COMPLEXITY_SCORE")
        meta["complexity_score"] = float(complexity) if isinstance(complexity, (int, float)) else None
        factor_metadata = constants.get("FACTOR_METADATA")
        if not isinstance(factor_metadata, dict):
            factor_metadata = {}
        meta["native_market"] = self._normalize_market(
            factor_metadata.get("native_market")
            or constants.get("NATIVE_MARKET")
        )
        meta["suitable_markets"] = self._normalize_market_list(
            factor_metadata.get("suitable_markets")
            or constants.get("SUITABLE_MARKETS")
        )
        meta["experimental_markets"] = self._normalize_market_list(
            factor_metadata.get("experimental_markets")
            or constants.get("EXPERIMENTAL_MARKETS")
        )
        meta["unsupported_markets"] = self._normalize_market_list(
            factor_metadata.get("unsupported_markets")
            or constants.get("UNSUPPORTED_MARKETS")
        )
        meta["required_fields"] = self._normalize_text_list(
            factor_metadata.get("required_fields")
            or constants.get("REQUIRED_FIELDS")
        )
        meta["suitability_declared"] = bool(
            factor_metadata
            or constants.get("NATIVE_MARKET")
            or constants.get("SUITABLE_MARKETS")
            or constants.get("EXPERIMENTAL_MARKETS")
            or constants.get("UNSUPPORTED_MARKETS")
        )
        return meta

    def _load_runs(self) -> pd.DataFrame:
        query = """
            SELECT
                r.*,
                d.failure_code,
                d.suggested_action
            FROM backtest_runs r
            LEFT JOIN diagnostics d ON r.run_id = d.run_id
        """
        runs = self._read_sql(query)
        if runs.empty:
            return runs
        numeric_cols = [
            "round_number",
            "validation_ic",
            "holdout_ic",
            "crisis_ic",
            "turnover_rate",
            "annualized_return",
            "max_drawdown",
            "sharpe_ratio",
            "total_trades",
            "universe_size",
            "stat_raw_p_value",
            "stat_metric_p_value",
            "stat_hit_rate_p_value",
            "stat_sharpe_p_value",
            "stat_adjusted_p_value",
            "stat_holm_p_value",
            "stat_fdr_q_value",
            "stat_trial_count",
        ]
        for col in numeric_cols:
            if col in runs.columns:
                runs[col] = pd.to_numeric(runs[col], errors="coerce")
        if "timestamp" in runs.columns:
            runs["timestamp"] = pd.to_datetime(runs["timestamp"], errors="coerce")
        else:
            runs["timestamp"] = pd.NaT
        return runs

    def _load_tick_ml_studies(self) -> pd.DataFrame:
        if not self._table_exists("tick_ml_studies"):
            return pd.DataFrame()
        studies = self._read_sql("SELECT * FROM tick_ml_studies")
        if studies.empty:
            return studies

        metrics = studies["metrics_json"].apply(self._parse_json_dict)
        metric_df = pd.json_normalize(metrics)
        out = pd.concat([studies.drop(columns=["metrics_json"]), metric_df], axis=1)
        out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")
        return out

    def _match_tick_ml(self, factor_id: str, tick_ml: pd.DataFrame) -> pd.DataFrame:
        if tick_ml.empty:
            return tick_ml
        factor_lower = factor_id.lower()
        if "fac_043" in factor_lower:
            return tick_ml.copy()
        if "tick" in factor_lower or "imbalance" in factor_lower or "breakdown" in factor_lower:
            return tick_ml.copy()
        return tick_ml.iloc[0:0].copy()

    def _feature_importance_index(self) -> set[str]:
        out = set()
        for directory in [self.logs_dir / "feature_importance", self.logs_dir / "returns" / "feature_importance"]:
            if not directory.exists():
                continue
            for path in directory.glob("feature_importance_*.csv"):
                key = path.stem.replace("feature_importance_", "")
                out.add(key)
        return out

    def _count_return_logs(self, factor_id: str, runs: pd.DataFrame) -> int:
        count = 0
        for _, run in runs.iterrows():
            path = self._resolve_returns_path(run.get("run_id"), run.get("returns_file_path"))
            if path and path.exists():
                count += 1
        if count:
            return count
        return len(list((self.logs_dir / "returns").glob(f"returns_{factor_id}*.csv")))

    def _count_trade_logs(self, runs: pd.DataFrame) -> int:
        count = 0
        trade_dir = self.logs_dir / "trades"
        for run_id in runs.get("run_id", pd.Series(dtype=str)).astype(str):
            if (trade_dir / f"trades_{run_id}.csv").exists():
                count += 1
        return count

    def _resolve_returns_path(self, run_id: str | None, returns_file_path: str | None) -> Path | None:
        if returns_file_path is not None and not pd.isna(returns_file_path) and str(returns_file_path).strip():
            path = Path(str(returns_file_path))
            if not path.is_absolute():
                path = self.base_dir / path
            return path
        if run_id is None or pd.isna(run_id):
            return None
        return self.logs_dir / "returns" / f"returns_{run_id}.csv"

    def _tested_markets(self, runs: pd.DataFrame) -> tuple[str, ...]:
        market_column = self._market_column(runs)
        if runs.empty or market_column is None:
            return ()
        values = [
            self._normalize_market(value)
            for value in runs[market_column].dropna().astype(str)
        ]
        return tuple(sorted({value for value in values if value}))

    def _native_market(self, metadata: dict, tested_markets: tuple[str, ...]) -> str:
        declared = self._normalize_market(metadata.get("native_market"))
        if declared:
            return declared
        if tested_markets:
            return tested_markets[0]
        return LEGACY_NATIVE_MARKET

    def _market_rows(
        self,
        metadata: dict,
        tested_markets: tuple[str, ...],
        native_market: str,
    ) -> tuple[str, ...]:
        declared = {
            native_market,
            *metadata.get("suitable_markets", ()),
            *metadata.get("experimental_markets", ()),
        }
        markets = {market for market in [*tested_markets, *declared] if market}
        if not markets:
            markets = {UNSPECIFIED_MARKET}
        return tuple(sorted(markets))

    def _filter_runs_for_market(self, runs: pd.DataFrame, market_vertical: str) -> pd.DataFrame:
        market_column = self._market_column(runs)
        if runs.empty or market_column is None:
            return runs.iloc[0:0].copy() if market_vertical != UNSPECIFIED_MARKET else runs.copy()
        normalized = runs[market_column].apply(self._normalize_market)
        return runs[normalized == market_vertical].copy()

    @staticmethod
    def _market_column(runs: pd.DataFrame) -> str | None:
        if "market_vertical" in runs.columns and runs["market_vertical"].notna().any():
            return "market_vertical"
        if "asset_class" in runs.columns:
            return "asset_class"
        return None

    def _vertical_status(self, evidence: FactorEvidence) -> str:
        market = evidence.market_vertical
        if market in evidence.unsupported_markets:
            return "unsupported"
        if not evidence.suitability_declared:
            return "legacy_inferred"
        if not evidence.runs.empty and market == evidence.native_market:
            return "native_validated"
        if not evidence.runs.empty:
            return "cross_market_validated"
        if market in evidence.experimental_markets:
            return "experimental_not_tested"
        if market in evidence.suitable_markets and market != evidence.native_market:
            return "declared_not_tested"
        if market == evidence.native_market:
            return "native_untested"
        return "not_declared"

    def _vertical_blockers(self, evidence: FactorEvidence) -> list[str]:
        blockers: list[str] = []
        market = evidence.market_vertical
        if market in evidence.unsupported_markets:
            blockers.append("market_unsupported")
        if market in evidence.experimental_markets and evidence.runs.empty:
            blockers.append("market_experimental")
        if evidence.runs.empty and market != evidence.native_market:
            blockers.append("market_not_validated")
        return blockers

    def _score_factor(self, evidence: FactorEvidence) -> dict:
        runs = evidence.runs
        run_count = len(runs)
        total_trades = int(pd.to_numeric(runs.get("total_trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        best_sharpe = self._max_numeric(runs, "sharpe_ratio")
        best_holdout = self._max_numeric(runs, "holdout_ic")
        best_validation = self._max_numeric(runs, "validation_ic")
        best_ann_return = self._max_numeric(runs, "annualized_return")
        best_max_dd = self._best_drawdown(runs)
        best_raw_p = self._min_numeric(runs, "stat_raw_p_value")
        best_adjusted_p = self._min_numeric(runs, "stat_adjusted_p_value")
        best_fdr_q = self._min_numeric(runs, "stat_fdr_q_value")
        stat_trial_count = self._max_numeric(runs, "stat_trial_count")
        last_run_at = runs["timestamp"].max() if not runs.empty and "timestamp" in runs.columns else pd.NaT
        failures = self._failure_codes(runs)
        vertical_status = self._vertical_status(evidence)

        score = 0
        blockers: list[str] = []
        score += 8 if evidence.source_exists else 0
        score += 7 if evidence.metadata_exists else 0
        score += 7 if evidence.economic_rationale else 0
        score += 10 if run_count > 0 or not evidence.tick_ml_rows.empty else 0
        score += 10 if run_count >= 2 or not evidence.tick_ml_rows.empty else 0
        score += 13 if evidence.return_logs > 0 else 0
        score += 8 if total_trades > 0 or evidence.trade_logs > 0 else 0
        score += 10 if pd.notna(best_holdout) and best_holdout > 0 else 0
        score += 7 if pd.notna(best_holdout) and best_holdout >= 0.01 else 0
        score += 10 if pd.notna(best_sharpe) and 0 < best_sharpe <= 10 else 0
        score += 5 if pd.notna(best_ann_return) and best_ann_return > 0 else 0
        score += 5 if pd.notna(best_max_dd) and best_max_dd > -0.35 else 0
        score += 8 if pd.notna(best_adjusted_p) and best_adjusted_p <= 0.05 else 0

        if run_count == 0 and evidence.tick_ml_rows.empty:
            blockers.append("no_test_evidence")
        if run_count > 0 and pd.isna(best_raw_p):
            blockers.append("missing_statistical_evidence")
        if pd.notna(best_raw_p) and best_raw_p > 0.05:
            blockers.append("raw_p_not_significant")
        if pd.notna(best_raw_p) and best_raw_p <= 0.05 and pd.notna(best_adjusted_p) and best_adjusted_p > 0.05:
            blockers.append("multiple_testing_not_significant")
        if run_count > 0 and evidence.return_logs == 0:
            blockers.append("missing_return_log")
        if evidence.return_logs > 0 and total_trades <= 0:
            blockers.append("no_discrete_trades")
        if pd.notna(best_holdout) and best_holdout <= 0:
            blockers.append("holdout_ic_not_positive")
        if pd.notna(best_sharpe) and best_sharpe <= 0:
            blockers.append("non_positive_sharpe")
        if pd.notna(best_sharpe) and best_sharpe > 10:
            blockers.append("sharpe_unrealistic")
        if pd.notna(best_max_dd) and best_max_dd <= -0.35:
            blockers.append("drawdown_too_deep")
        if failures:
            blockers.extend(sorted(failures))
        blockers.extend(self._vertical_blockers(evidence))

        blockers = list(dict.fromkeys(blockers))
        serious_failures = {
            "holdout_not_positive",
            "severe_ic_decay",
            "crisis_failure",
            "cross_sectional_collapse",
            "sharpe_unrealistic",
            "raw_p_not_significant",
            "multiple_testing_not_significant",
            "market_unsupported",
            "proxy_data_requires_contract_validation",
        }
        has_serious_failure = bool(serious_failures.intersection(blockers))
        score = max(score - self._blocker_penalty(blockers, serious_failures), 0)
        stage = self._infer_stage(
            run_count=run_count,
            tick_ml_count=len(evidence.tick_ml_rows),
            return_logs=evidence.return_logs,
            total_trades=total_trades,
            best_holdout=best_holdout,
            best_sharpe=best_sharpe,
            best_ann_return=best_ann_return,
            best_max_dd=best_max_dd,
            best_adjusted_p=best_adjusted_p,
            has_serious_failure=has_serious_failure,
            market_translation_required="market_not_validated" in blockers,
            market_unsupported="market_unsupported" in blockers,
        )
        next_step = self._suggest_next_step(stage, blockers)

        return {
            "factor_id": evidence.factor_id,
            "name": evidence.name,
            "category": evidence.category,
            "evidence_key": evidence.evidence_key,
            "market_vertical": evidence.market_vertical,
            "native_market": evidence.native_market,
            "tested_markets": ", ".join(evidence.tested_markets),
            "vertical_status": vertical_status,
            "stage": stage,
            "promotion_score": min(float(score), 100.0),
            "run_count": run_count,
            "tick_ml_count": len(evidence.tick_ml_rows),
            "best_validation_ic": best_validation,
            "best_holdout_ic": best_holdout,
            "best_raw_p_value": best_raw_p,
            "best_adjusted_p_value": best_adjusted_p,
            "best_fdr_q_value": best_fdr_q,
            "stat_trial_count": stat_trial_count,
            "best_sharpe": best_sharpe,
            "best_annualized_return": best_ann_return,
            "best_max_drawdown": best_max_dd,
            "total_trades": total_trades,
            "last_run_at": last_run_at,
            "has_returns": evidence.return_logs > 0,
            "has_trades": evidence.trade_logs > 0,
            "blockers": ", ".join(blockers),
            "suggested_next_step": next_step,
        }

    def _infer_stage(
        self,
        *,
        run_count: int,
        tick_ml_count: int,
        return_logs: int,
        total_trades: int,
        best_holdout: float,
        best_sharpe: float,
        best_ann_return: float,
        best_max_dd: float,
        best_adjusted_p: float,
        has_serious_failure: bool,
        market_translation_required: bool = False,
        market_unsupported: bool = False,
    ) -> str:
        tested = run_count > 0 or tick_ml_count > 0
        if market_unsupported:
            return "Retired / Repair"
        if market_translation_required and not tested:
            return "Translation Required"
        calibrated = run_count >= 2 or tick_ml_count > 0
        backtested = return_logs > 0
        statistical_ok = pd.isna(best_adjusted_p) or best_adjusted_p <= 0.05
        validation_ok = (
            backtested
            and total_trades > 0
            and pd.notna(best_holdout)
            and best_holdout > 0
            and pd.notna(best_sharpe)
            and best_sharpe > 0
            and (pd.isna(best_max_dd) or best_max_dd > -0.35)
            and statistical_ok
            and not has_serious_failure
        )
        paper_trading_ok = (
            validation_ok
            and best_holdout >= 0.01
            and 1.0 <= best_sharpe <= 10
            and pd.notna(best_ann_return)
            and best_ann_return > 0
            and (pd.isna(best_max_dd) or best_max_dd > -0.25)
            and total_trades >= 10
        )
        repair = tested and (
            has_serious_failure
            or (pd.notna(best_holdout) and best_holdout <= 0)
            or (backtested and pd.notna(best_sharpe) and best_sharpe <= 0)
        )
        if paper_trading_ok:
            return "Paper-Trading Candidate"
        if repair:
            return "Retired / Repair"
        if validation_ok:
            return "Validation Candidate"
        if backtested:
            return "Backtested"
        if calibrated:
            return "Calibrated"
        if tested:
            return "Hypothesis Tested"
        return "Idea"

    def _suggest_next_step(self, stage: str, blockers: list[str]) -> str:
        if "market_unsupported" in blockers:
            return "Do not route this factor to the selected market; its metadata marks the vertical unsupported."
        if "market_not_validated" in blockers:
            return "Run a dedicated vertical backtest before treating this market as paper-trading eligible."
        if "market_experimental" in blockers:
            return "Treat this as a translation experiment; require fresh OOS validation in this market."
        if "no_test_evidence" in blockers:
            return "Run initial hypothesis test or backtest."
        if "missing_statistical_evidence" in blockers:
            return "Re-run through the upgraded evaluator to compute p-values and research penalties."
        if "raw_p_not_significant" in blockers:
            return "Do not tune further yet; first improve the economic signal so raw holdout evidence is significant."
        if "multiple_testing_not_significant" in blockers:
            return "Freeze the current idea and test on fresh OOS data; current evidence does not survive trial-count correction."
        if "proxy_data_requires_contract_validation" in blockers:
            return "Re-run on tradable main-contract or tick data before treating this as paper-trading eligible."
        if "missing_return_log" in blockers:
            return "Re-run through evaluator so returns_file_path is saved."
        if "no_discrete_trades" in blockers:
            return "Inspect signal sizing; strategy may not be trading."
        if "holdout_ic_not_positive" in blockers:
            return "Redesign or invert the hypothesis before more optimization."
        if "drawdown_too_deep" in blockers:
            return "Add regime/risk filter before promotion."
        if "sharpe_unrealistic" in blockers:
            return "Audit return series and zero-volatility periods before trusting performance."
        if stage == "Paper-Trading Candidate":
            return "Prepare OOS/paper-trading protocol for this specific market vertical."
        if stage == "Validation Candidate":
            return "Compare against baseline and run stricter OOS split."
        if stage == "Backtested":
            return "Check OOS robustness, turnover, and parameter stability."
        if stage == "Calibrated":
            return "Run full evaluator/backtest with saved return log."
        if stage == "Hypothesis Tested":
            return "Calibrate parameters and repeat on another sample."
        return "Write thesis and run first test."

    @staticmethod
    def _blocker_penalty(blockers: list[str], serious_failures: set[str]) -> int:
        penalty = 0
        for blocker in blockers:
            penalty += 18 if blocker in serious_failures else 7
        return penalty

    def _failure_codes(self, runs: pd.DataFrame) -> set[str]:
        if runs.empty or "failure_code" not in runs.columns:
            return set()
        sorted_runs = runs.sort_values("timestamp", ascending=False) if "timestamp" in runs.columns else runs
        latest_run_id = sorted_runs["run_id"].iloc[0] if "run_id" in sorted_runs.columns else None
        if latest_run_id is not None:
            sorted_runs = sorted_runs[sorted_runs["run_id"] == latest_run_id]
        codes = {
            str(code).strip()
            for code in sorted_runs["failure_code"].dropna().unique()
            if str(code).strip() and str(code).upper() != "NONE"
        }
        return codes

    def _max_numeric(self, df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return np.nan
        values = pd.to_numeric(df[column], errors="coerce")
        return float(values.max()) if values.notna().any() else np.nan

    def _min_numeric(self, df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return np.nan
        values = pd.to_numeric(df[column], errors="coerce")
        return float(values.min()) if values.notna().any() else np.nan

    def _best_drawdown(self, df: pd.DataFrame) -> float:
        if df.empty or "max_drawdown" not in df.columns:
            return np.nan
        values = pd.to_numeric(df["max_drawdown"], errors="coerce")
        return float(values.max()) if values.notna().any() else np.nan

    def _read_sql(self, query: str) -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
            return df
        except Exception:
            return pd.DataFrame()

    def _table_exists(self, table_name: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                ).fetchone()
            return row is not None
        except Exception:
            return False

    @staticmethod
    def _coalesce(*values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    @staticmethod
    def _parse_json_dict(value) -> dict:
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            import json

            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _normalize_market(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return text.upper().replace("-", "_").replace(" ", "_")

    @classmethod
    def _normalize_market_list(cls, value) -> tuple[str, ...]:
        values = cls._normalize_text_list(value)
        return tuple(
            sorted(
                {
                    normalized
                    for item in values
                    if (normalized := cls._normalize_market(item))
                }
            )
        )

    @staticmethod
    def _normalize_text_list(value) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]
        out = []
        for item in raw_values:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)

    @staticmethod
    def _style_stage(values: pd.Series) -> list[str]:
        styles = []
        for value in values:
            color = STAGE_COLORS.get(str(value), "#e5e7eb")
            styles.append(f"background-color: {color}; color: white; font-weight: 700")
        return styles

    @staticmethod
    def _fmt(value, pattern: str) -> str:
        if pd.isna(value):
            return "N/A"
        return pattern.format(value)
