from __future__ import annotations

import ast
import sqlite3
import sys
from dataclasses import dataclass
from html import escape
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
from views.factor_library_view import (  # noqa: E402
    FactorLibraryView,
    load_factor_library_snapshot,
)
from views.sleeve_library_view import SleeveLibraryView  # noqa: E402
from views.predictive_evidence_panel import (  # noqa: E402
    load_predictive_evidence_snapshot,
    render_predictive_evidence_panel,
)
from views.sleeve_evidence_panel import render_sleeve_evidence_panel  # noqa: E402
from views.standalone_sleeve_test_panel import (  # noqa: E402
    render_standalone_sleeve_test_panel,
)
from views.conditional_behaviour_panel import (  # noqa: E402
    render_conditional_behaviour_panel,
)
from views.router_library_view import RouterLibraryView  # noqa: E402
from views.strategy_composition_panel import (  # noqa: E402
    render_strategy_composition_panel,
)
from views.optimization_workspace_panel import (  # noqa: E402
    render_optimization_workspace_panel,
)


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
        "title": "Research Review",
        "subtitle": "Review developed factors, validation evidence, promotion blockers, and completed backtest runs in one workspace.",
        "main_tab": "Review Board",
        "library_tab": "Factor Library",
        "sleeve_tab": "Sleeve Library",
        "router_tab": "Router Library",
        "construction_tab": "Strategy Construction",
        "optimization_tab": "Optimisation",
        "comparison_tab": "Run Comparison",
        "stage": "Stage",
        "category": "Category",
        "market": "Market Vertical",
        "native_market": "Native Market",
        "tested_markets": "Tested Markets",
        "vertical_status": "Vertical Status",
        "refresh": "Refresh evidence",
        "summary": "Review Summary",
        "funnel": "Lifecycle Funnel",
        "component_mix": "Research Components",
        "count": "Count",
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
        "quality_guide": "How to read IC and Sharpe labels",
        "quality_labels": {
            "not_available": "Not available",
            "negative": "Negative",
            "weak": "Weak",
            "marginal": "Marginal",
            "modest": "Modest",
            "good": "Good",
            "strong": "Strong",
            "high_audit": "High - audit",
            "extreme_audit": "Extreme - audit",
        },
        "quality_help": {
            "not_available": "No finite value is available for this metric.",
            "negative": "The estimate is non-positive and does not show positive predictive or risk-adjusted evidence.",
            "weak": "The estimate is positive but too small to treat as meaningful without stronger stability evidence.",
            "marginal": "The estimate is near a usable range but remains fragile after costs and sampling error.",
            "modest": "The IC is useful enough to investigate, but not strong enough to stand alone.",
            "good": "A credible research range if it is genuinely out of sample, stable, and measured after costs where applicable.",
            "strong": "Strong evidence, but verify stability across time, assets, and parameter choices.",
            "high_audit": "Unusually high. Audit annualization, costs, trial selection, overlap, and parameter stability before trusting it.",
            "extreme_audit": "Extreme value. Treat as an integrity warning and audit look-ahead bias, target leakage, curve fitting, omitted costs, and return construction.",
        },
        "quality_guide_text": """
**Holdout IC** (assuming the factor sign has already been aligned): `<= 0` negative; `(0, 0.01)` weak; `[0.01, 0.03)` modest; `[0.03, 0.05)` good; `[0.05, 0.10)` strong; `>= 0.10` integrity audit.

**Sharpe** (only meaningful on after-cost, untouched out-of-sample returns): `<= 0` negative; `(0, 0.5)` weak; `[0.5, 1.0)` marginal; `[1.0, 2.0)` good; `[2.0, 3.0)` strong; `[3.0, 5.0)` unusually high and requires audit; `>= 5.0` extreme integrity warning.

These are **diagnostic heuristics, not promotion gates**. Compare IC only across the same definition, forecast horizon, universe, and sampling method. Check Sharpe annualization, serial correlation, overlapping returns, turnover, and execution costs. This page shows the **best value across recorded runs**, so trial count and the full run distribution matter more than the maximum alone. A very high value does not prove leakage, but it should trigger checks for look-ahead bias, target leakage, curve fitting, cherry-picking, and omitted costs.
""",
        "raw_p": "Raw p",
        "adjusted_p": "Bonf. p",
        "adjusted_p_help": "Bonferroni-adjusted p-value: min(raw p x number of trials, 1). It controls false positives from testing many variants. Values <= 0.05 survive the correction; N/A means usable p-value or trial-count evidence has not been recorded.",
        "fdr_q": "FDR q",
        "trial_count": "Trials m",
        "best_return": "Best Ann. Return",
        "max_dd": "Best Run Max DD",
        "trades": "Trades",
        "last_run": "Last Run",
        "blockers": "Blockers",
        "next": "Suggested Next Step",
        "evidence": "Evidence Checklist",
        "runs_title": "Run History",
        "runs_empty": "No backtest runs yet.",
        "runs_desc": "Recorded run evidence. Columns with no usable values for the selected factor are omitted.",
        "run_provenance_title": "Data and Execution Provenance",
        "run_provenance_desc": "Dataset, vendor, frequency, and execution assumptions captured when each run was created.",
        "run_provenance_missing": "These {count} historical runs predate provenance capture. Re-run the factor with the current backtest CLI to record dataset, frequency, vendor, universe, and execution assumptions.",
        "run_not_recorded": "Not recorded",
        "run_record_complete": "Complete",
        "run_record_partial": "Partial",
        "run_record_legacy": "Legacy",
        "assets_suffix": "assets",
        "traded_suffix": "traded",
        "run_columns": {
            "run_id": "Run",
            "round_number": "Round",
            "timestamp": "Time",
            "market": "Market",
            "universe": "Universe",
            "validation_ic": "Validation IC",
            "holdout_ic": "Holdout IC",
            "crisis_ic": "Crisis IC",
            "annualized_return": "Ann. Return",
            "sharpe_ratio": "Sharpe",
            "max_drawdown": "Max Drawdown",
            "turnover_rate": "Turnover",
            "total_trades": "Trades",
            "stat_adjusted_p_value": "Bonf. p",
            "stat_significance": "Statistical Result",
            "failure_code": "Diagnostic",
            "record_status": "Record",
            "dataset_id": "Dataset",
            "universe_id": "Universe ID",
            "data_frequency": "Frequency",
            "data_vendor": "Vendor",
            "execution_assumption": "Execution Assumption",
        },
        "gate_title": "Promotion Gates",
        "artifact_title": "Artifacts",
        "exported_title": "Exported Candidate Snapshot Status",
        "exported_empty": "No research candidate snapshot has been exported for this factor/market row yet.",
        "exported_issues": "Candidate artifact issues",
        "candidate_handoff": "Paper-Trading Handoff",
        "candidate_handoff_desc": "Optional promotion control. Export a reviewed run only when it is ready to enter the shared candidate contract.",
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
        "has_predictive_evidence": "Phase 2 predictive evidence exists",
        "has_returns": "Return log exists",
        "has_trades": "Trade ledger exists",
        "has_importance": "Feature importance exists",
        "has_tick_ml": "Tick ML study exists",
        "has_market_metadata": "Suitability metadata exists",
        "market_tested": "This market was tested",
        "no_blockers": "No automatic blockers detected.",
    },
    "ZH": {
        "title": "研究评审",
        "subtitle": "在同一工作区评审已开发因子、验证证据、晋级阻碍项与已完成回测。",
        "main_tab": "审查看板",
        "library_tab": "因子库",
        "sleeve_tab": "策略腿库",
        "router_tab": "路由器库",
        "construction_tab": "策略构建",
        "optimization_tab": "优化",
        "comparison_tab": "运行对比",
        "stage": "阶段",
        "category": "类别",
        "market": "市场垂直",
        "native_market": "原生市场",
        "tested_markets": "已测试市场",
        "vertical_status": "垂直状态",
        "refresh": "刷新证据",
        "summary": "审查总览",
        "funnel": "生命周期漏斗",
        "component_mix": "研究组件分布",
        "count": "数量",
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
        "quality_guide": "如何解读 IC 与夏普标签",
        "quality_labels": {
            "not_available": "暂无数据",
            "negative": "负值",
            "weak": "偏弱",
            "marginal": "临界",
            "modest": "中等",
            "good": "良好",
            "strong": "较强",
            "high_audit": "偏高 - 需审计",
            "extreme_audit": "极高 - 需审计",
        },
        "quality_help": {
            "not_available": "该指标目前没有有限数值。",
            "negative": "该估计值非正，尚未显示正向预测能力或风险调整后证据。",
            "weak": "该估计值虽为正，但在没有更强稳定性证据前仍不足以视为有效。",
            "marginal": "该估计值接近可用区间，但考虑成本和抽样误差后仍较脆弱。",
            "modest": "IC 值值得继续研究，但不足以单独支撑晋级。",
            "good": "如果确为样本外、具有稳定性，并在适用时计入成本，则属于可信研究区间。",
            "strong": "证据较强，但仍需检查跨时间、跨资产和参数变化的稳定性。",
            "high_audit": "数值异常偏高。在采信前应审计年化方式、交易成本、试验筛选、收益重叠和参数稳定性。",
            "extreme_audit": "数值极高。应视为完整性警告，检查前视偏差、目标泄漏、曲线拟合、遗漏成本和收益构造。",
        },
        "quality_guide_text": """
**样本外 IC**（假设因子方向已经正确对齐）：`<= 0` 为负；`(0, 0.01)` 偏弱；`[0.01, 0.03)` 中等；`[0.03, 0.05)` 良好；`[0.05, 0.10)` 较强；`>= 0.10` 触发完整性审计。

**夏普比率**（只有在计入成本且未使用过的样本外收益上才有充分意义）：`<= 0` 为负；`(0, 0.5)` 偏弱；`[0.5, 1.0)` 临界；`[1.0, 2.0)` 良好；`[2.0, 3.0)` 较强；`[3.0, 5.0)` 异常偏高并需要审计；`>= 5.0` 属于极高完整性警告。

这些只是**诊断启发式标签，不是晋级门槛**。IC 只能在相同定义、预测周期、资产池和采样方法下比较。夏普需要检查年化方式、序列相关、重叠收益、换手率和执行成本。本页显示的是**所有已记录运行中的最佳值**，因此试验次数和完整运行分布比单个最大值更重要。极高数值并不自动证明存在泄漏，但应触发对前视偏差、目标泄漏、曲线拟合、结果挑选和遗漏成本的检查。
""",
        "raw_p": "原始 p 值",
        "adjusted_p": "Bonf. p",
        "adjusted_p_help": "Bonferroni 修正后的 p 值：min(原始 p 值 x 试验次数, 1)。它用于控制尝试多个版本带来的假阳性。数值 <= 0.05 表示通过修正；N/A 表示尚未记录可用的 p 值或试验次数证据。",
        "fdr_q": "FDR q",
        "trial_count": "试验数 m",
        "best_return": "最佳年化收益",
        "max_dd": "最佳回撤",
        "trades": "交易数",
        "last_run": "最近运行",
        "blockers": "阻碍项",
        "next": "建议下一步",
        "evidence": "证据清单",
        "runs_title": "运行历史",
        "runs_empty": "暂无回测运行。",
        "runs_desc": "已记录的运行证据。当前因子完全没有可用数值的列会自动隐藏。",
        "run_provenance_title": "数据与执行来源",
        "run_provenance_desc": "每次运行创建时记录的数据集、供应商、频率与执行假设。",
        "run_provenance_missing": "这 {count} 次历史运行早于来源信息采集。请使用当前回测 CLI 重新运行，以记录数据集、频率、供应商、资产池和执行假设。",
        "run_not_recorded": "未记录",
        "run_record_complete": "完整",
        "run_record_partial": "部分",
        "run_record_legacy": "历史记录",
        "assets_suffix": "个资产",
        "traded_suffix": "个有交易",
        "run_columns": {
            "run_id": "运行",
            "round_number": "轮次",
            "timestamp": "时间",
            "market": "市场",
            "universe": "资产池",
            "validation_ic": "验证 IC",
            "holdout_ic": "样本外 IC",
            "crisis_ic": "危机期 IC",
            "annualized_return": "年化收益",
            "sharpe_ratio": "夏普",
            "max_drawdown": "最大回撤",
            "turnover_rate": "换手率",
            "total_trades": "交易数",
            "stat_adjusted_p_value": "Bonf. p",
            "stat_significance": "统计结果",
            "failure_code": "诊断",
            "record_status": "记录状态",
            "dataset_id": "数据集",
            "universe_id": "资产池 ID",
            "data_frequency": "频率",
            "data_vendor": "供应商",
            "execution_assumption": "执行假设",
        },
        "gate_title": "晋级门槛",
        "artifact_title": "产物证据",
        "exported_title": "已导出的候选快照状态",
        "exported_empty": "该因子/市场行还没有导出研究候选快照。",
        "exported_issues": "候选产物问题",
        "candidate_handoff": "模拟交易交接",
        "candidate_handoff_desc": "可选晋级操作。只有在研究运行完成审查并准备进入共享候选合约时才需要导出。",
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
        "has_predictive_evidence": "已有第二阶段预测证据",
        "has_returns": "已有收益日志",
        "has_trades": "已有交易明细",
        "has_importance": "已有特征重要性",
        "has_tick_ml": "已有 Tick ML 研究",
        "has_market_metadata": "已有适用市场元数据",
        "market_tested": "该市场已测试",
        "no_blockers": "未发现自动 blocker。",
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
    predictive_evidence: dict | None


@dataclass(frozen=True)
class MetricQuality:
    label_key: str
    color: str


class FactorPromotionView:
    """Automatic research lifecycle board for factors and tick hypotheses."""

    def __init__(self, db_path: str = DB_PATH, base_dir: str = BASE_DIR):
        self.db_path = db_path
        self.base_dir = Path(base_dir)
        self.factors_dir = self.base_dir / "factors"
        self.logs_dir = Path(LOGS_DIR)

    def render(
        self,
        lang: str = "EN",
        theme_mode: str = "LIGHT",
        comparison_view=None,
    ) -> None:
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

        tab_labels = [
            copy["main_tab"],
            copy["library_tab"],
            copy["sleeve_tab"],
            copy["router_tab"],
            copy["construction_tab"],
            copy["optimization_tab"],
        ]
        if comparison_view is not None:
            tab_labels.append(copy["comparison_tab"])
        tabs = st.tabs(tab_labels)
        (
            board_tab,
            library_tab,
            sleeve_tab,
            router_tab,
            construction_tab,
            optimization_tab,
        ) = tabs[:6]
        with board_tab:
            self._render_summary(filtered, tpl, copy, theme_mode)
            st.markdown("---")
            self._render_board(filtered, copy)
            st.markdown("---")
            self._render_recent_candidate_exports(filtered, copy)
        with library_tab:
            FactorLibraryView(self.base_dir).render(
                lang=lang,
                theme_mode=theme_mode,
                review_factor_ids=board["factor_id"].dropna().astype(str),
                drilldown_renderer=lambda: self._render_drilldown(
                    filtered,
                    detail,
                    tpl,
                    copy,
                    lang,
                ),
            )
        with sleeve_tab:
            SleeveLibraryView(self.base_dir).render(lang=lang)
        with router_tab:
            RouterLibraryView(self.base_dir).render(lang=lang)
        with construction_tab:
            render_strategy_composition_panel(self.logs_dir, lang=lang)
        with optimization_tab:
            render_optimization_workspace_panel(self.logs_dir, lang=lang)
        if comparison_view is not None:
            with tabs[6]:
                comparison_view.render(
                    lang=lang,
                    theme_mode=theme_mode,
                    embedded=True,
                )

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

    def _render_summary(
        self,
        board: pd.DataFrame,
        tpl: str,
        copy: dict,
        theme_mode: str,
    ) -> None:
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
        funnel = px.bar(
            counts,
            x="stage",
            y="count",
            color="stage",
            color_discrete_map=STAGE_COLORS,
            template=tpl,
            title=copy["funnel"],
        )
        funnel.update_layout(
            height=350,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=20),
        )
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(funnel, use_container_width=True)
        with chart_cols[1]:
            snapshot = load_factor_library_snapshot(str(self.base_dir))
            FactorLibraryView._render_component_mix(
                snapshot.component_summary,
                theme_mode,
                copy,
            )

    def _render_board(self, board: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['table']}")
        board = board[
            board["factor_id"].astype(str).str.startswith("fac_")
        ].copy()
        show_cols = [
            "stage",
            "factor_id",
            "last_run_at",
            "promotion_score",
            "category",
            "market_vertical",
            "native_market",
            "vertical_status",
            "name",
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
        display[copy["last_run"]] = (
            pd.to_datetime(display[copy["last_run"]], errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M")
            .fillna("")
        )
        styled = display.style.format(
            {
                copy["score"]: "{:.0f}",
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

    def _render_drilldown(
        self,
        board: pd.DataFrame,
        detail: dict[str, FactorEvidence],
        tpl: str,
        copy: dict,
        lang: str,
    ) -> None:
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

        primary_cols = st.columns(3)
        with primary_cols[0]:
            stage = str(row["stage"])
            stage_color = STAGE_COLORS.get(stage, "#64748b")
            st.markdown(
                f"""
                <div style="min-height: 4.7rem; padding-top: 0.1rem;">
                    <div style="font-size: 0.875rem; line-height: 1.2; margin-bottom: 0.55rem;">
                        {escape(copy["stage"])}
                    </div>
                    <div title="{escape(stage)}" style="
                        display: inline-flex;
                        max-width: 100%;
                        padding: 0.4rem 0.55rem;
                        border: 1px solid {stage_color}66;
                        border-left: 4px solid {stage_color};
                        border-radius: 4px;
                        background: {stage_color}14;
                        font-size: 0.95rem !important;
                        font-weight: 700;
                        line-height: 1.2;
                        white-space: normal !important;
                        overflow-wrap: anywhere;
                    ">
                        {escape(stage)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        primary_cols[1].metric(copy["score"], f"{row['promotion_score']:.0f}/100")
        self._render_quality_metric(
            primary_cols[2],
            copy["best_sharpe"],
            self._fmt(row["best_sharpe"], "{:.2f}"),
            self._classify_sharpe(row["best_sharpe"]),
            copy,
        )
        secondary_cols = st.columns(3)
        self._render_quality_metric(
            secondary_cols[0],
            copy["best_ic"],
            self._fmt(row["best_holdout_ic"], "{:.4f}"),
            self._classify_holdout_ic(row["best_holdout_ic"]),
            copy,
        )
        secondary_cols[1].metric(
            copy["adjusted_p"],
            self._fmt(row["best_adjusted_p_value"], "{:.4f}"),
            help=copy["adjusted_p_help"],
        )
        secondary_cols[2].metric(copy["trades"], f"{int(row['total_trades']):,}")

        with st.expander(copy["quality_guide"]):
            st.markdown(copy["quality_guide_text"])

        st.markdown(f"**{row['name']}**")
        if evidence.economic_rationale:
            st.caption(evidence.economic_rationale)
        if row["blockers"]:
            st.warning(f"{copy['blockers']}: {row['blockers']}")
        else:
            st.success(copy["no_blockers"])
        st.info(f"{copy['next']}: {row['suggested_next_step']}")

        render_predictive_evidence_panel(
            self.logs_dir,
            evidence.factor_id,
            evidence.market_vertical,
            lang=lang,
            plotly_template=tpl,
        )
        render_sleeve_evidence_panel(
            self.logs_dir,
            evidence.factor_id,
            evidence.market_vertical,
            lang=lang,
            plotly_template=tpl,
        )
        render_standalone_sleeve_test_panel(
            self.logs_dir,
            evidence.factor_id,
            evidence.market_vertical,
            lang=lang,
            plotly_template=tpl,
        )
        render_conditional_behaviour_panel(
            self.logs_dir,
            evidence.factor_id,
            evidence.market_vertical,
            lang=lang,
            plotly_template=tpl,
        )

        st.markdown(f"#### {copy['runs_title']}")
        self._render_run_history(evidence.runs, tpl, copy)

        left, right = st.columns([0.46, 0.54])
        with left:
            self._render_evidence_checklist(evidence, copy)
        with right:
            self._render_artifacts(evidence, copy)

        candidate_exports, candidate_issues = self._candidate_exports_for_evidence(evidence)
        with st.expander(copy["candidate_handoff"], expanded=False):
            st.caption(copy["candidate_handoff_desc"])
            self._render_candidate_readback(candidate_exports, candidate_issues, copy)
            self._render_candidate_export(row, evidence, copy, candidate_exports)

    def _render_evidence_checklist(self, evidence: FactorEvidence, copy: dict) -> None:
        st.markdown(f"#### {copy['evidence']}")
        checks = [
            (copy["source_exists"], evidence.source_exists),
            (copy["metadata_exists"], evidence.metadata_exists),
            (copy["rationale_exists"], bool(evidence.economic_rationale)),
            (copy["has_market_metadata"], evidence.suitability_declared),
            (
                copy["market_tested"],
                not evidence.runs.empty or evidence.predictive_evidence is not None,
            ),
            (copy["has_backtest"], not evidence.runs.empty),
            (
                copy["has_predictive_evidence"],
                evidence.predictive_evidence is not None,
            ),
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

    def _render_run_history(self, runs: pd.DataFrame, tpl: str, copy: dict) -> None:
        if runs.empty:
            st.info(copy["runs_empty"])
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

        st.caption(copy["runs_desc"])
        history = runs.copy().sort_values("timestamp", ascending=False)
        history["market"] = history.apply(self._run_market_label, axis=1)
        history["universe"] = history.apply(
            lambda row: self._run_universe_label(
                row,
                copy["assets_suffix"],
                copy["traded_suffix"],
            ),
            axis=1,
        )

        core_cols = ["run_id", "round_number", "timestamp", "market"]
        optional_cols = [
            "universe",
            "validation_ic",
            "holdout_ic",
            "crisis_ic",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "turnover_rate",
            "total_trades",
            "stat_adjusted_p_value",
            "stat_significance",
            "failure_code",
        ]
        show_cols = core_cols + [
            col for col in optional_cols if self._series_has_values(history, col)
        ]
        percentage_cols = ["annualized_return", "max_drawdown", "turnover_rate"]
        for col in percentage_cols:
            if col in history.columns:
                history[col] = pd.to_numeric(history[col], errors="coerce") * 100.0
        if "total_trades" in history.columns:
            history["total_trades"] = pd.to_numeric(
                history["total_trades"], errors="coerce"
            ).astype("Int64")

        labels = copy["run_columns"]
        display = history[show_cols].rename(columns=labels)
        column_config = {
            labels["run_id"]: st.column_config.TextColumn(width="medium"),
            labels["round_number"]: st.column_config.NumberColumn(format="%d", width="small"),
            labels["timestamp"]: st.column_config.DatetimeColumn(
                format="YYYY-MM-DD HH:mm", width="medium"
            ),
            labels["market"]: st.column_config.TextColumn(width="medium"),
        }
        for col in ("validation_ic", "holdout_ic", "crisis_ic", "stat_adjusted_p_value"):
            if col in show_cols:
                column_config[labels[col]] = st.column_config.NumberColumn(format="%.4f")
        if "sharpe_ratio" in show_cols:
            column_config[labels["sharpe_ratio"]] = st.column_config.NumberColumn(format="%.2f")
        for col in percentage_cols:
            if col in show_cols:
                column_config[labels[col]] = st.column_config.NumberColumn(format="%.2f%%")
        if "total_trades" in show_cols:
            column_config[labels["total_trades"]] = st.column_config.NumberColumn(format="%d")
        if "failure_code" in show_cols:
            column_config[labels["failure_code"]] = st.column_config.TextColumn(width="medium")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

        provenance_fields = [
            "dataset_id",
            "universe_id",
            "data_frequency",
            "data_vendor",
            "execution_assumption",
        ]
        provenance_cols = [
            "run_id",
            "timestamp",
            "market",
            "record_status",
        ]
        available_provenance = [
            col for col in provenance_fields if self._series_has_values(history, col)
        ]
        with st.expander(copy["run_provenance_title"], expanded=False):
            st.caption(copy["run_provenance_desc"])
            if not available_provenance:
                st.info(copy["run_provenance_missing"].format(count=len(history)))
            else:
                provenance = history.copy()
                provenance["record_status"] = provenance.apply(
                    lambda row: self._run_provenance_status(row, provenance_fields, copy),
                    axis=1,
                )
                for col in available_provenance:
                    provenance[col] = provenance[col].apply(
                        lambda value: self._display_text(value) or copy["run_not_recorded"]
                    )
                provenance_display = provenance[
                    provenance_cols + available_provenance
                ].rename(columns=labels)
                provenance_column_config = {
                    labels["run_id"]: st.column_config.TextColumn(width="medium"),
                    labels["timestamp"]: st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm", width="medium"
                    ),
                }
                if "execution_assumption" in available_provenance:
                    provenance_column_config[
                        labels["execution_assumption"]
                    ] = st.column_config.TextColumn(width="large")
                st.dataframe(
                    provenance_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config=provenance_column_config,
                )

    @staticmethod
    def _display_text(value) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"", "none", "nan", "nat"} else text

    @classmethod
    def _series_has_values(cls, frame: pd.DataFrame, column: str) -> bool:
        if column not in frame.columns:
            return False
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            return bool(series.notna().any())
        return bool(series.map(cls._display_text).ne("").any())

    @classmethod
    def _run_market_label(cls, row: pd.Series) -> str:
        return cls._display_text(row.get("market_vertical")) or cls._display_text(
            row.get("asset_class")
        )

    @classmethod
    def _run_universe_label(
        cls,
        row: pd.Series,
        assets_suffix: str,
        traded_suffix: str,
    ) -> str:
        universe_id = cls._display_text(row.get("universe_id"))
        traded = cls._display_text(row.get("traded_tickers"))
        size = pd.to_numeric(pd.Series([row.get("universe_size")]), errors="coerce").iloc[0]
        size_label = f"{int(size):,} {assets_suffix}" if pd.notna(size) else ""

        if universe_id:
            return f"{universe_id} ({size_label})" if size_label else universe_id
        if traded.upper() == "ALL":
            return f"ALL ({size_label})" if size_label else "ALL"
        if traded:
            tickers = [ticker.strip() for ticker in traded.split(",") if ticker.strip()]
            if len(tickers) <= 4:
                base = ", ".join(tickers)
            else:
                base = f"{len(tickers):,} {traded_suffix}"
            return f"{base} ({size_label})" if size_label else base
        return size_label

    @classmethod
    def _run_provenance_status(
        cls,
        row: pd.Series,
        fields: list[str],
        copy: dict,
    ) -> str:
        populated = sum(bool(cls._display_text(row.get(field))) for field in fields)
        if populated == len(fields):
            return copy["run_record_complete"]
        if populated:
            return copy["run_record_partial"]
        return copy["run_record_legacy"]

    @st.cache_data(show_spinner=False)
    def _load_board(_self) -> tuple[pd.DataFrame, dict[str, FactorEvidence]]:
        factors = _self._load_factor_inventory()
        runs = _self._load_runs()
        tick_ml = _self._load_tick_ml_studies()
        importance_index = _self._feature_importance_index()
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
                    predictive_evidence=load_predictive_evidence_snapshot(
                        str(_self.logs_dir), factor_id, market_vertical
                    ),
                )
                evidence_by_factor[evidence_key] = evidence
                row = _self._score_factor(evidence)
                rows.append(row)

        board = pd.DataFrame(rows)
        if not board.empty:
            board["stage_rank"] = board["stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER)}).fillna(0)
            board["board_priority"] = board["stage"].map(BOARD_PRIORITY).fillna(0)
            board = _self._sort_board_by_recency(board)
        return board, evidence_by_factor

    @staticmethod
    def _sort_board_by_recency(board: pd.DataFrame) -> pd.DataFrame:
        if board.empty:
            return board

        out = board.copy()
        out["_last_run_sort"] = pd.to_datetime(
            out.get("last_run_at"),
            errors="coerce",
        )
        sort_cols = [
            col
            for col in [
                "_last_run_sort",
                "board_priority",
                "promotion_score",
                "best_sharpe",
                "factor_id",
            ]
            if col in out.columns
        ]
        ascending = [col == "factor_id" for col in sort_cols]
        return (
            out.sort_values(
                sort_cols,
                ascending=ascending,
                na_position="last",
                kind="stable",
            )
            .drop(columns=["_last_run_sort"])
            .reset_index(drop=True)
        )

    def _normalize_board_stages(self, board: pd.DataFrame) -> pd.DataFrame:
        if board.empty or "stage" not in board.columns:
            return board

        out = board.copy()
        out["stage"] = out["stage"].replace(LEGACY_STAGE_MAP)
        out["stage_rank"] = out["stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER)}).fillna(0)
        out["board_priority"] = out["stage"].map(BOARD_PRIORITY).fillna(0)
        return self._sort_board_by_recency(out)

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
        if evidence.predictive_evidence is not None:
            if market == evidence.native_market:
                return "native_predictive_evidence"
            return "cross_market_predictive_evidence"
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
        if (
            market in evidence.experimental_markets
            and evidence.runs.empty
            and evidence.predictive_evidence is None
        ):
            blockers.append("market_experimental")
        if (
            evidence.runs.empty
            and evidence.predictive_evidence is None
            and market != evidence.native_market
        ):
            blockers.append("market_not_validated")
        return blockers

    def _score_factor(self, evidence: FactorEvidence) -> dict:
        runs = evidence.runs
        run_count = len(runs)
        has_predictive_evidence = evidence.predictive_evidence is not None
        total_trades = int(pd.to_numeric(runs.get("total_trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        best_sharpe = self._max_numeric(runs, "sharpe_ratio")
        best_holdout = self._max_numeric(runs, "holdout_ic")
        best_validation = self._max_numeric(runs, "validation_ic")
        if has_predictive_evidence:
            predictive_split = evidence.predictive_evidence["split_summary"].set_index(
                "research_split"
            )
            if pd.isna(best_validation) and "validation" in predictive_split.index:
                best_validation = pd.to_numeric(
                    predictive_split.loc["validation", "mean_rank_ic"],
                    errors="coerce",
                )
            if pd.isna(best_holdout) and "holdout" in predictive_split.index:
                best_holdout = pd.to_numeric(
                    predictive_split.loc["holdout", "mean_rank_ic"],
                    errors="coerce",
                )
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
        score += 10 if run_count > 0 or not evidence.tick_ml_rows.empty or has_predictive_evidence else 0
        score += 10 if run_count >= 2 or not evidence.tick_ml_rows.empty else 0
        score += 13 if evidence.return_logs > 0 else 0
        score += 8 if total_trades > 0 or evidence.trade_logs > 0 else 0
        score += 10 if pd.notna(best_holdout) and best_holdout > 0 else 0
        score += 7 if pd.notna(best_holdout) and best_holdout >= 0.01 else 0
        score += 10 if pd.notna(best_sharpe) and 0 < best_sharpe <= 10 else 0
        score += 5 if pd.notna(best_ann_return) and best_ann_return > 0 else 0
        score += 5 if pd.notna(best_max_dd) and best_max_dd > -0.35 else 0
        score += 8 if pd.notna(best_adjusted_p) and best_adjusted_p <= 0.05 else 0

        if run_count == 0 and evidence.tick_ml_rows.empty and not has_predictive_evidence:
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
            run_count=run_count + int(has_predictive_evidence),
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
            "predictive_evidence_count": int(has_predictive_evidence),
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
            return "Review holdout stability and concentration, then test a frozen portfolio translation after costs."
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

    @staticmethod
    def _classify_holdout_ic(value) -> MetricQuality:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric) or not np.isfinite(numeric):
            return MetricQuality("not_available", "#64748b")
        if numeric <= 0:
            return MetricQuality("negative", "#ef4444")
        if numeric < 0.01:
            return MetricQuality("weak", "#ef4444")
        if numeric < 0.03:
            return MetricQuality("modest", "#f59e0b")
        if numeric < 0.05:
            return MetricQuality("good", "#22c55e")
        if numeric < 0.10:
            return MetricQuality("strong", "#14b8a6")
        return MetricQuality("extreme_audit", "#ef4444")

    @staticmethod
    def _classify_sharpe(value) -> MetricQuality:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric) or not np.isfinite(numeric):
            return MetricQuality("not_available", "#64748b")
        if numeric <= 0:
            return MetricQuality("negative", "#ef4444")
        if numeric < 0.5:
            return MetricQuality("weak", "#ef4444")
        if numeric < 1.0:
            return MetricQuality("marginal", "#f59e0b")
        if numeric < 2.0:
            return MetricQuality("good", "#22c55e")
        if numeric < 3.0:
            return MetricQuality("strong", "#14b8a6")
        if numeric < 5.0:
            return MetricQuality("high_audit", "#8b5cf6")
        return MetricQuality("extreme_audit", "#ef4444")

    @staticmethod
    def _render_quality_metric(
        column,
        metric_label: str,
        metric_value: str,
        quality: MetricQuality,
        copy: dict,
    ) -> None:
        label = copy["quality_labels"][quality.label_key]
        help_text = copy["quality_help"][quality.label_key]
        column.html(
            f"""
            <div style="min-height: 4.7rem; padding-top: 0.1rem;">
                <div style="font-size: 0.875rem; line-height: 1.2; margin-bottom: 0.55rem;">
                    {escape(metric_label)}
                </div>
                <div style="white-space: nowrap; overflow: visible; line-height: 1.1;">
                    <span style="display: inline-block; font-size: 1.75rem; line-height: 1.1; font-weight: 400; vertical-align: middle;">
                        {escape(metric_value)}
                    </span>
                    <span title="{escape(help_text)}" style="
                        display: inline-flex;
                        margin-left: 0.35rem;
                        padding: 0.18rem 0.3rem;
                        border: 1px solid {quality.color}66;
                        border-radius: 4px;
                        background: {quality.color}14;
                        font-size: 0.62rem;
                        font-weight: 700;
                        line-height: 1.15;
                        white-space: nowrap;
                        vertical-align: middle;
                    ">
                        {escape(label)}
                    </span>
                </div>
            </div>
            """
        )
