from __future__ import annotations

import importlib
import importlib.util
import html
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

_CONFIG_PATH = UI_DIR / "config.py"
_CONFIG_SPEC = importlib.util.spec_from_file_location("_ui_v2_config", _CONFIG_PATH)
if _CONFIG_SPEC is None or _CONFIG_SPEC.loader is None:
    raise ImportError(f"Unable to load UI config from {_CONFIG_PATH}")
_UI_CONFIG = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_UI_CONFIG)

BASE_DIR = _UI_CONFIG.BASE_DIR
DB_PATH = _UI_CONFIG.DB_PATH
LOGS_DIR = _UI_CONFIG.LOGS_DIR
ALPHA_RUNTIME_DATA_ROOT = _UI_CONFIG.ALPHA_RUNTIME_DATA_ROOT
ALPHA_RUNTIME_ARTIFACT_ROOT = _UI_CONFIG.ALPHA_RUNTIME_ARTIFACT_ROOT
get_plotly_template = _UI_CONFIG.get_plotly_template

try:
    from oqp.data.asset_taxonomy import load_asset_taxonomy

    ASSET_TAXONOMY = load_asset_taxonomy(BASE_DIR)
except Exception:
    ASSET_TAXONOMY = {}


STATUS_RANK = {"OK": 2, "WARN": 1, "FAIL": 0}
STATUS_COLORS = {"OK": "#16a34a", "WARN": "#f59e0b", "FAIL": "#dc2626"}


COPY = {
    "EN": {
        "title": "Data Health",
        "subtitle": "A read-only readiness check for market coverage, data files, model artifacts, return logs, and runtime infrastructure.",
        "overview": "Overview",
        "core": "Core Data",
        "tick": "Tick Data",
        "database": "Database & Returns",
        "latent": "Latent Artifacts",
        "models": "Model Registry",
        "infra": "C++ / ML Infra",
        "data_sources": "Data Sources",
        "research_artifacts": "Research Artifacts",
        "runtime": "Runtime",
        "data_sources_help_title": "How to read Data Sources",
        "data_sources_help": """
This tab answers: *what sample am I really researching?*

- Market Coverage shows which asset classes are active locally versus waiting on vendor/API wiring.
- Daily matrices feed most factor, regime, and relationship pages.
- Tick cache files feed the intraday/tick labs; the newest file and date range matter most.
- `WARN` often means stale or optional data. `FAIL` means a required file is missing, empty, or unreadable.
""",
        "artifacts_help_title": "How to read Research Artifacts",
        "artifacts_help": """
This tab answers: *can I reproduce the research evidence?*

- Database schema checks tell you whether runs, factors, diagnostics, and model records can be queried.
- Return-log linkage tells you whether equity curves can be reconstructed.
- Latent and model artifacts tell you whether saved ML/regime evidence still points to real files.
""",
        "runtime_help_title": "How to read Runtime",
        "runtime_help": """
This tab answers: *will the page run correctly and fast enough?*

- C++ failures usually affect tick acceleration or execution-engine speed.
- Missing optimization caches are often acceptable on a fresh machine.
- Runtime warnings do not automatically invalidate research, but they can make results slower or harder to reproduce.
""",
        "status_ok": "Valid research starting point: no failing health checks. You can use downstream pages, while still checking warnings for stale or optional inputs.",
        "status_warn": "Usable but not clean: warnings mean some inputs are stale, optional, or incomplete. Treat downstream signals as candidates for further research, not final evidence.",
        "status_fail": "Not valid enough yet: at least one required dependency is broken or missing. Fix failures before trusting affected downstream pages.",
        "refresh": "Refresh health snapshot",
        "readiness": "Readiness",
        "ok": "OK",
        "warn": "Warnings",
        "fail": "Failures",
        "latest_run": "Latest Run",
        "api_readiness": "API Readiness",
        "api_readiness_note": "US equities and US options are API-backed lanes. They are shown here because missing credentials can block the next research phase without affecting the current bundled Chinese futures dataset.",
        "checks": "Checks",
        "path": "Path",
        "status": "Status",
        "detail": "Detail",
        "modified": "Modified",
        "rows": "Rows",
        "cols": "Columns",
        "date_range": "Date Range",
        "assets": "Assets",
        "size": "Size",
        "market_coverage": "Market Coverage & Vendor Readiness",
        "asset_class": "Asset Class",
        "role": "Role",
        "description": "Description",
        "region": "Region",
        "settlement": "Settlement",
        "price_limit": "Price Limit",
        "vectorizable": "Vectorizable",
        "data_mode": "Data Mode",
        "provider": "Provider",
        "env_status": "Credential Status",
        "env_source": "Credential Source",
        "data_matrix": "Daily Data Matrices",
        "tick_files": "Tick Cache Files",
        "db_schema": "Database Schema",
        "returns_linkage": "Return Log Linkage",
        "return_issues_title": "Runs With Missing Return Logs",
        "return_issues_empty": "No missing return-log files detected.",
        "run_id": "Run ID",
        "resolved_path": "Expected Path",
        "stored_path": "Stored Path",
        "annualized_return": "Annualized Return",
        "sharpe_ratio": "Sharpe Ratio",
        "latent_title": "VQ-VAE / Latent Factor Artifacts",
        "model_title": "Registered Model Artifacts",
        "model_empty": "No model artifacts registered yet. Retrain an ML model to populate this table.",
        "model_manual": """
This table is the bridge between model files and reproducible research.

- **Artifact Path** is the versioned copy under `runtime/artifacts/research/model_artifacts/`.
- **Legacy Path** is the old path that current factors may still load, such as `ml_engine/xgb_base_model.json`.
- **Data Hash** tells you which training file produced the artifact.
- **Split Policy** tells you how the model separated train and validation data.
- **Metrics** stores the training/validation evidence available at save time.
""",
        "model_registered": "Registered Models",
        "model_missing": "Missing Artifact Files",
        "model_latest": "Latest Artifact",
        "model_families": "Model Families",
        "artifact_path": "Artifact Path",
        "artifact_exists": "Artifact Exists",
        "legacy_path": "Legacy Path",
        "legacy_exists": "Legacy Exists",
        "feature_count": "Features",
        "target_col": "Target",
        "model_type": "Model Type",
        "model_name": "Model",
        "artifact_format": "Format",
        "data_path": "Data Path",
        "factor_id": "Factor",
        "created_at": "Created",
        "data_hash": "Data Hash",
        "split_policy": "Split Policy",
        "metrics_json": "Metrics",
        "model_health_ok": "Model registry looks reproducible: registered artifact files are present. This supports further model research, but does not prove model quality.",
        "model_health_warn": "Model research is partly reproducible: versioned artifacts exist, but at least one legacy path or compatibility link needs attention.",
        "model_health_fail": "Model evidence is not fully reproducible yet: at least one registered model artifact points to a missing file.",
        "infra_title": "Runtime Infrastructure",
        "no_tick": "No tick parquet files found in runtime/data/alpha_lab/market_data/tick.",
    },
    "ZH": {
        "title": "数据健康检查",
        "subtitle": "只读检查市场覆盖、数据文件、模型产物、收益日志与运行基础设施。",
        "overview": "总览",
        "core": "核心数据",
        "tick": "Tick 数据",
        "database": "数据库与收益日志",
        "latent": "潜在因子产物",
        "models": "模型注册表",
        "infra": "C++ / ML 基础设施",
        "data_sources": "数据源",
        "research_artifacts": "研究产物",
        "runtime": "运行环境",
        "data_sources_help_title": "如何阅读数据源",
        "data_sources_help": """
这个标签回答：*我到底在研究哪段样本？*

- Market Coverage 显示哪些资产类别已经有本地数据，哪些还在等待供应商/API 接入。
- 日频矩阵服务于大多数因子、状态和关系页面。
- Tick 缓存服务于高频/逐笔研究；最新文件和日期范围最重要。
- `WARN` 常见于数据偏旧或可选数据缺失。`FAIL` 表示必要文件缺失、为空或不可读。
""",
        "artifacts_help_title": "如何阅读研究产物",
        "artifacts_help": """
这个标签回答：*研究证据是否能复现？*

- 数据库结构检查说明因子、运行记录、诊断和模型记录是否能被查询。
- 收益日志链接说明净值曲线是否能被恢复。
- 潜在因子和模型产物说明 ML / 状态识别证据是否仍然指向真实文件。
""",
        "runtime_help_title": "如何阅读运行环境",
        "runtime_help": """
这个标签回答：*页面是否能正确且足够快地运行？*

- C++ 失败通常影响 tick 加速或执行引擎速度。
- 新机器上缺少优化缓存通常可以接受。
- 运行环境警告不一定否定研究结论，但会让结果更慢或更难复现。
""",
        "status_ok": "有效的研究起点：没有失败项。可以使用下游页面，但仍要留意过期或可选输入的警告。",
        "status_warn": "可以使用但不干净：警告表示部分输入过期、可选或不完整。下游信号只能视为进一步研究候选，而不是最终证据。",
        "status_fail": "暂时不够有效：至少一个必要依赖损坏或缺失。先修复失败项，再信任受影响的下游页面。",
        "refresh": "刷新健康快照",
        "readiness": "就绪度",
        "ok": "正常",
        "warn": "警告",
        "fail": "失败",
        "latest_run": "最近运行",
        "api_readiness": "API 就绪度",
        "api_readiness_note": "美股与美股期权属于 API 数据线。这里单独展示，是因为缺少密钥会影响下一阶段研究，但不代表当前本地中国期货样本不可用。",
        "checks": "检查项",
        "path": "路径",
        "status": "状态",
        "detail": "说明",
        "modified": "修改时间",
        "rows": "行数",
        "cols": "列数",
        "date_range": "日期范围",
        "assets": "资产数",
        "size": "大小",
        "market_coverage": "市场覆盖与供应商就绪度",
        "asset_class": "资产类别",
        "role": "定位",
        "description": "说明",
        "region": "地区",
        "settlement": "交割/结算",
        "price_limit": "涨跌停",
        "vectorizable": "可向量化",
        "data_mode": "数据模式",
        "provider": "供应商",
        "env_status": "密钥状态",
        "env_source": "密钥来源",
        "data_matrix": "日频数据矩阵",
        "tick_files": "Tick 缓存文件",
        "db_schema": "数据库结构",
        "returns_linkage": "收益日志链接",
        "return_issues_title": "缺失收益日志的运行",
        "return_issues_empty": "未检测到缺失的收益日志文件。",
        "run_id": "运行 ID",
        "resolved_path": "预期路径",
        "stored_path": "数据库路径",
        "annualized_return": "年化收益",
        "sharpe_ratio": "夏普比率",
        "latent_title": "VQ-VAE / 潜在因子产物",
        "model_title": "已注册模型产物",
        "model_empty": "目前还没有注册的模型产物。重新训练一次 ML 模型后，这里会自动出现记录。",
        "model_manual": """
这个表是“模型文件”和“可复现实验记录”之间的桥。

- **Artifact Path / 归档路径**：`runtime/artifacts/research/model_artifacts/` 下的版本化模型副本。
- **Legacy Path / 旧路径**：当前因子仍可能读取的老路径，例如 `ml_engine/xgb_base_model.json`。
- **Data Hash / 数据哈希**：记录这个模型由哪个训练文件产生。
- **Split Policy / 切分规则**：记录训练集和验证集如何分开。
- **Metrics / 指标**：保存模型训练时可用的验证证据。
""",
        "model_registered": "已注册模型数",
        "model_missing": "缺失模型文件",
        "model_latest": "最新模型产物",
        "model_families": "模型类型数",
        "artifact_path": "归档路径",
        "artifact_exists": "归档文件存在",
        "legacy_path": "旧路径",
        "legacy_exists": "旧文件存在",
        "feature_count": "特征数",
        "target_col": "目标列",
        "model_type": "模型类型",
        "model_name": "模型",
        "artifact_format": "格式",
        "data_path": "数据路径",
        "factor_id": "因子",
        "created_at": "创建时间",
        "data_hash": "数据哈希",
        "split_policy": "切分规则",
        "metrics_json": "指标",
        "model_health_ok": "模型注册表具备可复现性：已注册模型文件存在。可以继续模型研究，但这并不证明模型质量。",
        "model_health_warn": "模型研究部分可复现：版本化产物存在，但至少一个旧路径或兼容链接需要关注。",
        "model_health_fail": "模型证据尚未完全可复现：至少一个已注册模型产物指向缺失文件。",
        "infra_title": "运行基础设施",
        "no_tick": "runtime/data/alpha_lab/market_data/tick 中未找到 tick parquet 文件。",
    },
}


@dataclass
class HealthCheck:
    area: str
    check: str
    status: str
    detail: str
    path: str = ""
    modified: str = ""


class SystemHealthView:
    """Read-only health board for the research platform's data/artifact layer."""

    CORE_PARQUETS = [
        ("ML Feature Matrix", "feature_store/ML_Feature_Matrix.parquet", True),
        ("GMM Rolling Probabilities", "regime/GMM_Rolling_Probabilities.parquet", True),
        ("ML Stacked Matrix", "feature_store/ML_Stacked_Matrix.parquet", False),
        ("Macro Index", "regime/Macro_Index_V2.parquet", False),
        ("Macro Regimes", "regime/Macro_Regimes.parquet", False),
    ]

    REQUIRED_DB_COLUMNS = {
        "factors": {"factor_id", "name", "category", "economic_rationale"},
        "backtest_runs": {
            "run_id",
            "factor_id",
            "round_number",
            "validation_ic",
            "holdout_ic",
            "crisis_ic",
            "turnover_rate",
            "annualized_return",
            "max_drawdown",
            "sharpe_ratio",
            "total_trades",
            "asset_class",
            "universe_size",
            "traded_tickers",
            "returns_file_path",
            "evaluation_geometry",
            "ic_metric",
            "validation_hit_rate",
            "holdout_hit_rate",
            "crisis_hit_rate",
            "timestamp",
        },
        "diagnostics": {"run_id", "failure_code", "suggested_action"},
        "tick_ml_studies": {"study_key", "updated_at", "symbol", "hypothesis", "metrics_json"},
        "model_artifacts": {
            "artifact_id",
            "model_name",
            "model_type",
            "artifact_format",
            "artifact_path",
            "feature_count",
            "target_col",
            "split_policy_json",
            "metrics_json",
            "created_at",
        },
    }

    EXPECTED_VQ_FILES = [
        "storm_temporal_vqvae_latents.parquet",
        "storm_temporal_vqvae_codebook.csv",
        "storm_temporal_vqvae_loss_history.csv",
        "storm_temporal_vqvae_usage.csv",
        "storm_temporal_vqvae_encoder.joblib",
    ]

    def __init__(self, base_dir: str = BASE_DIR, db_path: str = DB_PATH):
        self.base_dir = Path(base_dir)
        self.repo_root = self.base_dir if (self.base_dir / "runtime").exists() else self.base_dir.parent
        self.db_path = Path(db_path)
        self.logs_dir = Path(LOGS_DIR)
        self.runtime_data_root = Path(ALPHA_RUNTIME_DATA_ROOT)
        self.runtime_artifact_root = Path(ALPHA_RUNTIME_ARTIFACT_ROOT)
        self.data_cache = self.runtime_data_root / "market_data" / "tick"
        legacy_native_dir = os.environ.get("OQP_LEGACY_QUANT_CORE_DIR", "").strip()
        self.legacy_cpp_dir = Path(legacy_native_dir) if legacy_native_dir else None

    def render(self, lang: str = "EN", theme_mode: str = "LIGHT") -> None:
        copy = COPY.get(lang, COPY["EN"])
        tpl = get_plotly_template(theme_mode)

        self._render_page_header(copy)
        if st.session_state.pop("_data_health_refresh_requested", False):
            st.cache_data.clear()
            st.rerun()

        snapshot = self._load_snapshot(str(self.base_dir), str(self.db_path))
        checks = snapshot["checks"]
        market_df = snapshot["markets"]
        parquet_df = snapshot["parquets"]
        tick_df = snapshot["ticks"]
        db_df = snapshot["db"]
        returns_df = snapshot["returns"]
        return_issues_df = snapshot["return_issues"]
        latent_df = snapshot["latent"]
        model_df = snapshot["models"]
        infra_df = snapshot["infra"]

        tabs = st.tabs(
            [
                copy["overview"],
                copy["data_sources"],
                copy["research_artifacts"],
                copy["runtime"],
            ]
        )
        with tabs[0]:
            self._render_overview(checks, snapshot, tpl, copy)
        with tabs[1]:
            self._render_help_toggle(copy["data_sources_help_title"], copy["data_sources_help"])
            self._render_market_coverage(market_df, copy)
            st.divider()
            self._render_parquet_table(parquet_df, copy["data_matrix"], copy)
            st.divider()
            self._render_tick_table(tick_df, copy)
        with tabs[2]:
            self._render_help_toggle(copy["artifacts_help_title"], copy["artifacts_help"])
            self._render_database(db_df, returns_df, return_issues_df, copy)
            st.divider()
            self._render_latent(latent_df, copy)
            st.divider()
            self._render_models(model_df, copy)
        with tabs[3]:
            self._render_help_toggle(copy["runtime_help_title"], copy["runtime_help"])
            self._render_infra(infra_df, copy)

    @staticmethod
    def _render_page_header(copy: dict) -> None:
        title_col, action_col = st.columns([0.76, 0.24], vertical_alignment="center")
        with title_col:
            st.title(copy["title"])
            st.caption(copy["subtitle"])
        with action_col:
            st.write("")
            if st.button(copy["refresh"], width="stretch"):
                st.session_state["_data_health_refresh_requested"] = True

    @staticmethod
    def _render_help_toggle(title: str, body: str, expanded: bool = False) -> None:
        with st.expander(title, expanded=expanded):
            st.markdown(body)

    @staticmethod
    def _render_health_callout(score: float, warn_count: int, fail_count: int, copy: dict) -> None:
        if fail_count:
            st.error(copy["status_fail"])
        elif warn_count or score < 90:
            st.warning(copy["status_warn"])
        else:
            st.success(copy["status_ok"])

    def _render_overview(self, checks: pd.DataFrame, snapshot: dict, tpl: str, copy: dict) -> None:
        score = self._readiness_score(checks)
        ok_count = int((checks["status"] == "OK").sum())
        warn_count = int((checks["status"] == "WARN").sum())
        fail_count = int((checks["status"] == "FAIL").sum())

        cols = st.columns([0.95, 0.95, 0.95, 0.95, 1.65])
        cols[0].metric(copy["readiness"], f"{score:.0f}/100")
        cols[1].metric(copy["ok"], f"{ok_count:,}")
        cols[2].metric(copy["warn"], f"{warn_count:,}")
        cols[3].metric(copy["fail"], f"{fail_count:,}")
        self._render_latest_run(cols[4], snapshot.get("latest_run", "N/A"), copy)

        self._render_health_callout(score, warn_count, fail_count, copy)
        self._render_api_readiness(snapshot.get("markets", pd.DataFrame()), copy)

        area_counts = (
            checks.groupby(["area", "status"], as_index=False)
            .agg(count=("check", "count"))
            .sort_values(["area", "status"])
        )
        fig = px.bar(
            area_counts,
            x="area",
            y="count",
            color="status",
            color_discrete_map=STATUS_COLORS,
            template=tpl,
            title=copy["checks"],
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=20), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            self._style_status(checks[["area", "check", "status", "detail", "path", "modified"]]),
            width="stretch",
            hide_index=True,
            height=520,
        )

    @staticmethod
    def _render_latest_run(container, latest_run: Any, copy: dict) -> None:
        latest_label = SystemHealthView._compact_timestamp(latest_run)
        safe_label = html.escape(latest_label)
        with container:
            st.markdown(
                f"""
                <div class="oqp-latest-run">
                    <div class="oqp-latest-run__label">{html.escape(copy["latest_run"])}</div>
                    <div class="oqp-latest-run__value">{safe_label}</div>
                </div>
                <style>
                .oqp-latest-run {{
                    border: 1px solid rgba(148, 163, 184, 0.34);
                    border-radius: 8px;
                    padding: 0.62rem 0.72rem;
                    min-height: 4.5rem;
                    background: rgba(248, 250, 252, 0.58);
                }}
                .oqp-latest-run__label {{
                    color: rgb(107, 114, 128);
                    font-size: 0.78rem;
                    line-height: 1.1;
                    margin-bottom: 0.36rem;
                }}
                .oqp-latest-run__value {{
                    color: rgb(17, 24, 39);
                    font-size: 0.96rem;
                    font-weight: 650;
                    line-height: 1.25;
                    overflow-wrap: anywhere;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    @staticmethod
    def _compact_timestamp(value: Any) -> str:
        text = str(value or "N/A").strip()
        if not text or text == "N/A":
            return "N/A"
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d %H:%M")
        return text

    def _render_api_readiness(self, market_df: pd.DataFrame, copy: dict) -> None:
        if market_df.empty or "asset_class" not in market_df.columns:
            return
        api_df = market_df[market_df["asset_class"].isin(["EQUITY_US", "OPTIONS_US"])].copy()
        if api_df.empty:
            return

        st.markdown(f"### {copy['api_readiness']}")
        st.caption(copy["api_readiness_note"])
        display_cols = [
            "status",
            "asset_class",
            "provider",
            "data_mode",
            "env_status",
            "env_source",
            "detail",
        ]
        display = api_df[[col for col in display_cols if col in api_df.columns]].rename(
            columns={
                "status": copy["status"],
                "asset_class": copy["asset_class"],
                "provider": copy["provider"],
                "data_mode": copy["data_mode"],
                "env_status": copy["env_status"],
                "env_source": copy["env_source"],
                "detail": copy["detail"],
            }
        )
        st.dataframe(self._style_status(display), width="stretch", hide_index=True, height=150)

    def _render_parquet_table(self, df: pd.DataFrame, title: str, copy: dict) -> None:
        st.markdown(f"### {title}")
        display = df.rename(
            columns={
                "label": copy["checks"],
                "status": copy["status"],
                "path": copy["path"],
                "modified": copy["modified"],
                "rows": copy["rows"],
                "columns": copy["cols"],
                "date_range": copy["date_range"],
                "asset_count": copy["assets"],
                "size_mb": copy["size"],
                "detail": copy["detail"],
            }
        )
        st.dataframe(self._style_status(display), width="stretch", hide_index=True)

    def _render_tick_table(self, df: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['tick_files']}")
        if df.empty:
            st.info(copy["no_tick"])
            return
        display = df.rename(
            columns={
                "status": copy["status"],
                "path": copy["path"],
                "modified": copy["modified"],
                "rows": copy["rows"],
                "columns": copy["cols"],
                "date_range": copy["date_range"],
                "asset_count": copy["assets"],
                "size_mb": copy["size"],
                "detail": copy["detail"],
            }
        )
        st.dataframe(self._style_status(display), width="stretch", hide_index=True, height=420)

    def _render_market_coverage(self, market_df: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['market_coverage']}")
        display = market_df.rename(
            columns={
                "status": copy["status"],
                "asset_class": copy["asset_class"],
                "role": copy["role"],
                "description": copy["description"],
                "region": copy["region"],
                "t_settlement": copy["settlement"],
                "price_limit": copy["price_limit"],
                "vectorizable": copy["vectorizable"],
                "data_mode": copy["data_mode"],
                "provider": copy["provider"],
                "env_status": copy["env_status"],
                "env_source": copy["env_source"],
                "detail": copy["detail"],
            }
        )
        st.dataframe(self._style_status(display), width="stretch", hide_index=True, height=260)

    def _render_database(
        self,
        db_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        return_issues_df: pd.DataFrame,
        copy: dict,
    ) -> None:
        st.markdown(f"### {copy['db_schema']}")
        st.dataframe(self._style_status(db_df), width="stretch", hide_index=True)
        st.markdown(f"### {copy['returns_linkage']}")
        st.dataframe(self._style_status(returns_df), width="stretch", hide_index=True)
        st.markdown(f"#### {copy['return_issues_title']}")
        if return_issues_df.empty:
            st.success(copy["return_issues_empty"])
            return
        display = return_issues_df.rename(
            columns={
                "status": copy["status"],
                "run_id": copy["run_id"],
                "factor_id": copy["factor_id"],
                "stored_path": copy["stored_path"],
                "resolved_path": copy["resolved_path"],
                "annualized_return": copy["annualized_return"],
                "sharpe_ratio": copy["sharpe_ratio"],
                "timestamp": copy["modified"],
                "detail": copy["detail"],
            }
        )
        st.dataframe(self._style_status(display), width="stretch", hide_index=True, height=220)

    def _render_latent(self, latent_df: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['latent_title']}")
        st.dataframe(self._style_status(latent_df), width="stretch", hide_index=True)

    def _render_models(self, model_df: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['model_title']}")
        with st.expander("How to read / 如何阅读", expanded=False):
            st.markdown(copy["model_manual"])

        if model_df.empty:
            st.info(copy["model_empty"])
            return

        registered = int(len(model_df))
        missing = int((model_df["status"] == "FAIL").sum()) if "status" in model_df.columns else 0
        warnings = int((model_df["status"] == "WARN").sum()) if "status" in model_df.columns else 0
        latest = str(model_df.iloc[0].get("model_name", "N/A"))
        families = int(model_df["model_type"].nunique(dropna=True)) if "model_type" in model_df.columns else 0
        cols = st.columns(4)
        cols[0].metric(copy["model_registered"], f"{registered:,}")
        cols[1].metric(copy["model_missing"], f"{missing:,}")
        cols[2].metric(copy["model_latest"], latest)
        cols[3].metric(copy["model_families"], f"{families:,}")

        if missing:
            st.error(copy["model_health_fail"])
        elif warnings:
            st.warning(copy["model_health_warn"])
        else:
            st.success(copy["model_health_ok"])

        display_cols = [
            "status",
            "model_name",
            "factor_id",
            "model_type",
            "artifact_format",
            "feature_count",
            "target_col",
            "artifact_exists",
            "legacy_exists",
            "artifact_path",
            "legacy_path",
            "data_path",
            "data_hash_short",
            "split_policy_summary",
            "metrics_summary",
            "created_at",
        ]
        display = model_df[[col for col in display_cols if col in model_df.columns]].rename(
            columns={
                "status": copy["status"],
                "model_name": copy["model_name"],
                "factor_id": copy["factor_id"],
                "model_type": copy["model_type"],
                "artifact_format": copy["artifact_format"],
                "feature_count": copy["feature_count"],
                "target_col": copy["target_col"],
                "artifact_exists": copy["artifact_exists"],
                "legacy_exists": copy["legacy_exists"],
                "artifact_path": copy["artifact_path"],
                "legacy_path": copy["legacy_path"],
                "data_path": copy["data_path"],
                "data_hash_short": copy["data_hash"],
                "split_policy_summary": copy["split_policy"],
                "metrics_summary": copy["metrics_json"],
                "created_at": copy["created_at"],
            }
        )
        st.dataframe(self._style_status(display), width="stretch", hide_index=True, height=460)

    def _render_infra(self, infra_df: pd.DataFrame, copy: dict) -> None:
        st.markdown(f"### {copy['infra_title']}")
        st.dataframe(self._style_status(infra_df), width="stretch", hide_index=True)

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _load_snapshot(base_dir: str, db_path: str) -> dict[str, Any]:
        view = SystemHealthView(base_dir=base_dir, db_path=db_path)
        checks: list[HealthCheck] = []

        parquet_df, parquet_checks = view._core_parquet_snapshot()
        checks.extend(parquet_checks)

        tick_df, tick_checks = view._tick_snapshot()
        checks.extend(tick_checks)

        market_df = view._market_coverage_snapshot(parquet_df, tick_df)

        db_df, db_checks, latest_run = view._database_schema_snapshot()
        checks.extend(db_checks)

        returns_df, return_issues_df, returns_checks = view._returns_linkage_snapshot()
        checks.extend(returns_checks)

        latent_df, latent_checks = view._latent_artifact_snapshot()
        checks.extend(latent_checks)

        model_df, model_checks = view._model_registry_snapshot()
        checks.extend(model_checks)

        infra_df, infra_checks = view._infra_snapshot()
        checks.extend(infra_checks)

        checks_df = pd.DataFrame([check.__dict__ for check in checks])
        if checks_df.empty:
            checks_df = pd.DataFrame(columns=["area", "check", "status", "detail", "path", "modified"])
        return {
            "checks": checks_df,
            "markets": market_df,
            "parquets": parquet_df,
            "ticks": tick_df,
            "db": db_df,
            "returns": returns_df,
            "return_issues": return_issues_df,
            "latent": latent_df,
            "models": model_df,
            "infra": infra_df,
            "latest_run": latest_run,
        }

    def _core_parquet_snapshot(self) -> tuple[pd.DataFrame, list[HealthCheck]]:
        rows = []
        checks = []
        for label, rel_path, required in self.CORE_PARQUETS:
            path = self.runtime_data_root / rel_path
            summary = self._parquet_summary(path)
            if not path.exists():
                status = "FAIL" if required else "WARN"
                detail = "Required artifact missing." if required else "Optional artifact missing."
            elif summary.get("rows", 0) <= 0:
                status = "FAIL" if required else "WARN"
                detail = "File exists but appears empty."
            else:
                stale = self._is_data_stale(summary.get("date_max"))
                status = "WARN" if required and stale else "OK"
                detail = "Data max date is older than 30 days." if stale else "Readable."

            row = {
                "label": label,
                "status": status,
                "path": self._rel(path),
                "modified": self._mtime_label(path),
                "rows": summary.get("rows", np.nan),
                "columns": summary.get("columns", np.nan),
                "date_range": summary.get("date_range", ""),
                "asset_count": summary.get("asset_count", np.nan),
                "size_mb": self._size_mb(path),
                "detail": detail,
            }
            rows.append(row)
            checks.append(HealthCheck("Core Data", label, status, detail, self._rel(path), row["modified"]))
        return pd.DataFrame(rows), checks

    def _tick_snapshot(self) -> tuple[pd.DataFrame, list[HealthCheck]]:
        files = sorted(self.data_cache.glob("*tick*.parquet"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        rows = []
        checks = []
        for idx, path in enumerate(files):
            summary = self._parquet_summary(path)
            parsed_range = self._parse_range_from_name(path.name)
            detail = "Newest tick file." if idx == 0 else "Older tick file."
            if summary.get("rows", 0) <= 0:
                status = "FAIL"
                detail = "Tick file exists but appears empty."
            elif idx == 0 and self._is_data_stale(summary.get("date_max"), max_days=14):
                status = "WARN"
                detail = "Newest tick data max date is older than 14 days."
            else:
                status = "OK"
            rows.append(
                {
                    "file": path.name,
                    "status": status,
                    "path": self._rel(path),
                    "modified": self._mtime_label(path),
                    "rows": summary.get("rows", np.nan),
                    "columns": summary.get("columns", np.nan),
                    "date_range": summary.get("date_range") or parsed_range,
                    "asset_count": summary.get("asset_count", np.nan),
                    "size_mb": self._size_mb(path),
                    "detail": detail,
                }
            )
        if files:
            newest = rows[0]
            checks.append(
                HealthCheck(
                    "Tick Data",
                    "Latest tick cache",
                    newest["status"],
                    newest["detail"],
                    newest["path"],
                    newest["modified"],
                )
            )
        else:
            checks.append(HealthCheck("Tick Data", "Latest tick cache", "WARN", "No tick parquet files found.", self._rel(self.data_cache)))
        return pd.DataFrame(rows), checks

    def _market_coverage_snapshot(self, parquet_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        futures_required = parquet_df[parquet_df["label"].isin(["ML Feature Matrix", "GMM Rolling Probabilities"])].copy()
        futures_fail = bool((futures_required["status"] == "FAIL").any()) if not futures_required.empty else True
        futures_warn = bool((futures_required["status"] == "WARN").any()) if not futures_required.empty else False
        if futures_fail:
            futures_status = "FAIL"
            futures_detail = "Current Chinese futures research files are missing or unreadable."
        elif futures_warn:
            futures_status = "WARN"
            futures_detail = f"Current Chinese futures files are present, but at least one required matrix is stale. Tick cache files: {len(tick_df)}."
        else:
            futures_status = "OK"
            futures_detail = f"Current Chinese futures local/static dataset is readable. Tick cache files: {len(tick_df)}."

        rows.append(
            self._market_row(
                asset_class="FUTURES_CN",
                status=futures_status,
                role="Current local/static research dataset",
                data_mode="Local parquet matrices + tick cache",
                provider="Bundled/static files",
                env_status="No API key required",
                env_source="workspace files",
                detail=futures_detail,
            )
        )

        fmp_present, fmp_source = self._env_key_presence("FMP_API_KEY")
        rows.append(
            self._market_row(
                asset_class="EQUITY_US",
                status="OK" if fmp_present else "WARN",
                role="Next-phase US equities lane",
                data_mode="API-backed; no public data bundled",
                provider="FMP",
                env_status="FMP_API_KEY configured" if fmp_present else "Missing FMP_API_KEY",
                env_source=fmp_source,
                detail=(
                    "US equity data credentials are present; wire the FMP adapter before treating this as an active dataset."
                    if fmp_present
                    else "Add FMP_API_KEY to the parent .env or runtime environment before US equity pages rely on live/vendor data."
                ),
            )
        )

        massive_present, massive_source = self._env_key_presence("MASSIVE_API_KEY")
        flat_access_present, flat_access_source = self._env_key_presence("MASSIVE_FLAT_FILES_ACCESS_KEY_ID")
        flat_secret_present, flat_secret_source = self._env_key_presence("MASSIVE_FLAT_FILES_SECRET_ACCESS_KEY")
        flat_sources = self._merge_sources([flat_access_source, flat_secret_source])
        if flat_access_present and flat_secret_present:
            flat_status = f"Flat files configured ({flat_sources})"
        elif flat_access_present or flat_secret_present:
            flat_status = "Flat files partially configured"
        else:
            flat_status = "Flat files not configured"

        rows.append(
            self._market_row(
                asset_class="OPTIONS_US",
                status="OK" if massive_present else "WARN",
                role="Next-phase US options lane",
                data_mode="API-backed option chains; event-driven/non-vectorized",
                provider="Massive",
                env_status=("MASSIVE_API_KEY configured; " if massive_present else "Missing MASSIVE_API_KEY; ") + flat_status,
                env_source=self._merge_sources([massive_source, flat_sources]),
                detail=(
                    "US options credentials are present. Taxonomy marks this lane as non-vectorizable, so pages should use event-driven option-chain handling."
                    if massive_present
                    else "Add MASSIVE_API_KEY before options pages rely on vendor data. Flat-file keys can later support historical chain ingestion."
                ),
            )
        )

        return pd.DataFrame(rows)

    def _market_row(
        self,
        asset_class: str,
        status: str,
        role: str,
        data_mode: str,
        provider: str,
        env_status: str,
        env_source: str,
        detail: str,
    ) -> dict[str, Any]:
        taxonomy = ASSET_TAXONOMY.get(asset_class, {})
        return {
            "status": status,
            "asset_class": asset_class,
            "role": role,
            "description": taxonomy.get("description", ""),
            "region": taxonomy.get("region", ""),
            "t_settlement": taxonomy.get("t_settlement", ""),
            "price_limit": self._yes_no(taxonomy.get("price_limit")),
            "vectorizable": self._yes_no(taxonomy.get("vectorizable")),
            "data_mode": data_mode,
            "provider": provider,
            "env_status": env_status,
            "env_source": env_source,
            "detail": detail,
        }

    def _env_key_presence(self, key: str) -> tuple[bool, str]:
        if self._valid_env_value(os.environ.get(key)):
            return True, "environment"

        sources = []
        for path in self._env_search_paths():
            values = self._read_env_file_keys(path)
            if self._valid_env_value(values.get(key)):
                sources.append(self._env_source_label(path))
        if sources:
            return True, self._merge_sources(sources)
        return False, "not found"

    def _env_search_paths(self) -> list[Path]:
        candidates = [self.repo_root / ".env"]
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                unique.append(path)
                seen.add(resolved)
        return unique

    @staticmethod
    def _read_env_file_keys(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        values: dict[str, str] = {}
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
        except Exception:
            return {}
        return values

    def _env_source_label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return path.name

    @staticmethod
    def _valid_env_value(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        return text.lower() not in {"none", "null", "todo", "replace_me", "your_key_here"}

    @staticmethod
    def _merge_sources(sources: list[str]) -> str:
        cleaned = [source for source in sources if source and source != "not found"]
        if not cleaned:
            return "not found"
        return ", ".join(dict.fromkeys(cleaned))

    @staticmethod
    def _yes_no(value: Any) -> str:
        if value is True:
            return "Yes"
        if value is False:
            return "No"
        return ""

    def _database_schema_snapshot(self) -> tuple[pd.DataFrame, list[HealthCheck], str]:
        rows = []
        checks = []
        latest_run = "N/A"
        db_exists = self.db_path.exists()
        if not db_exists:
            row = {
                "table": "research_memory.db",
                "status": "FAIL",
                "rows": 0,
                "missing_columns": "database missing",
                "detail": "Run init_research_db.py.",
            }
            return pd.DataFrame([row]), [HealthCheck("Database", "research_memory.db", "FAIL", row["detail"], self._rel(self.db_path))], latest_run

        try:
            with sqlite3.connect(self.db_path) as conn:
                for table, required_cols in self.REQUIRED_DB_COLUMNS.items():
                    table_exists = self._table_exists_conn(conn, table)
                    if not table_exists:
                        rows.append(
                            {
                                "table": table,
                                "status": "FAIL",
                                "rows": 0,
                                "missing_columns": "table missing",
                                "detail": "Required table is missing.",
                            }
                        )
                        checks.append(HealthCheck("Database", table, "FAIL", "Required table is missing.", self._rel(self.db_path)))
                        continue
                    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                    missing = sorted(required_cols - cols)
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    status = "FAIL" if missing else "OK"
                    detail = "Schema complete." if not missing else f"Missing columns: {', '.join(missing)}"
                    rows.append(
                        {
                            "table": table,
                            "status": status,
                            "rows": count,
                            "missing_columns": ", ".join(missing),
                            "detail": detail,
                        }
                    )
                    checks.append(HealthCheck("Database", table, status, detail, self._rel(self.db_path)))

                if self._table_exists_conn(conn, "backtest_runs"):
                    latest = conn.execute("SELECT MAX(timestamp) FROM backtest_runs").fetchone()[0]
                    latest_run = str(latest) if latest else "N/A"
        except Exception as exc:
            rows.append({"table": "database", "status": "FAIL", "rows": 0, "missing_columns": "", "detail": str(exc)})
            checks.append(HealthCheck("Database", "schema read", "FAIL", str(exc), self._rel(self.db_path)))
        return pd.DataFrame(rows), checks, latest_run

    def _returns_linkage_snapshot(self) -> tuple[pd.DataFrame, pd.DataFrame, list[HealthCheck]]:
        rows = []
        issue_rows = []
        checks = []
        if not self.db_path.exists():
            return pd.DataFrame(), pd.DataFrame(), []

        try:
            with sqlite3.connect(self.db_path) as conn:
                if not self._table_exists_conn(conn, "backtest_runs"):
                    return pd.DataFrame(), []
                runs = pd.read_sql_query(
                    """
                    SELECT run_id, factor_id, returns_file_path, annualized_return, sharpe_ratio, timestamp
                    FROM backtest_runs
                    """,
                    conn,
                )
        except Exception as exc:
            rows.append({"check": "returns_file_path", "status": "FAIL", "value": "", "detail": str(exc)})
            return pd.DataFrame(rows), pd.DataFrame(), [HealthCheck("Returns", "returns_file_path", "FAIL", str(exc))]

        if runs.empty:
            rows.append({"check": "backtest_runs", "status": "WARN", "value": "0", "detail": "No runs logged yet."})
            return pd.DataFrame(rows), pd.DataFrame(), [HealthCheck("Returns", "logged runs", "WARN", "No runs logged yet.")]

        runs["stored_path_present"] = runs["returns_file_path"].notna() & runs["returns_file_path"].astype(str).str.strip().ne("")
        runs["resolved_path"] = runs.apply(lambda row: str(self._resolve_returns_path(row["run_id"], row["returns_file_path"])), axis=1)
        runs["path_exists"] = runs["resolved_path"].map(lambda value: Path(value).exists())
        executed = runs["annualized_return"].notna() | runs["sharpe_ratio"].notna() | runs["stored_path_present"]
        executed_runs = runs[executed].copy()
        missing_stored = int((executed_runs["stored_path_present"] == False).sum())
        missing_files = int((executed_runs["path_exists"] == False).sum())
        explicit_missing_files = int((executed_runs["stored_path_present"] & (executed_runs["path_exists"] == False)).sum())
        fallback_missing_files = int(((executed_runs["stored_path_present"] == False) & (executed_runs["path_exists"] == False)).sum())
        linked = int((executed_runs["stored_path_present"] & executed_runs["path_exists"]).sum())
        missing_runs = executed_runs[executed_runs["path_exists"] == False].copy()
        for _, run in missing_runs.sort_values("timestamp", ascending=False).iterrows():
            has_stored_path = bool(run.get("stored_path_present", False))
            issue_rows.append(
                {
                    "status": "FAIL" if has_stored_path else "WARN",
                    "run_id": run.get("run_id", ""),
                    "factor_id": run.get("factor_id", ""),
                    "stored_path": run.get("returns_file_path", ""),
                    "resolved_path": self._rel(Path(run.get("resolved_path", ""))),
                    "annualized_return": run.get("annualized_return", np.nan),
                    "sharpe_ratio": run.get("sharpe_ratio", np.nan),
                    "timestamp": run.get("timestamp", ""),
                    "detail": (
                        "Database stores a return CSV path, but the file is missing."
                        if has_stored_path
                        else "Legacy run has performance evidence but no stored return path; rerun it to recreate the equity curve."
                    ),
                }
            )

        status = "OK"
        detail = "All executed runs have linked return logs."
        if explicit_missing_files:
            status = "FAIL"
            detail = f"{explicit_missing_files} executed runs have explicit return paths pointing to missing files."
        elif fallback_missing_files:
            status = "WARN"
            detail = f"{fallback_missing_files} legacy runs have no stored return path and no fallback CSV."
        elif missing_stored:
            status = "WARN"
            detail = f"{missing_stored} executed runs rely on fallback return-file guessing."

        rows.extend(
            [
                {"check": "executed runs", "status": "OK", "value": len(executed_runs), "detail": "Runs with performance or return-path evidence."},
                {"check": "linked return files", "status": status, "value": linked, "detail": detail},
                {"check": "missing stored returns_file_path", "status": "WARN" if missing_stored else "OK", "value": missing_stored, "detail": "Legacy runs may need evaluator rerun."},
                {
                    "check": "missing resolved files",
                    "status": "FAIL" if explicit_missing_files else "WARN" if fallback_missing_files else "OK",
                    "value": missing_files,
                    "detail": (
                        "Explicit stored paths are broken."
                        if explicit_missing_files
                        else "Only legacy fallback paths are missing."
                        if fallback_missing_files
                        else "All resolvable return files are present."
                    ),
                },
            ]
        )
        checks.append(HealthCheck("Returns", "return log linkage", status, detail))
        return pd.DataFrame(rows), pd.DataFrame(issue_rows), checks

    def _latent_artifact_snapshot(self) -> tuple[pd.DataFrame, list[HealthCheck]]:
        root_dir = self.runtime_artifact_root / "latent_factors"
        ui_dir = self.repo_root / "apps" / "research_dashboard" / "execution_logs" / "latent_factors"
        rows = []
        checks = []
        existing_count = 0
        for filename in self.EXPECTED_VQ_FILES:
            path = root_dir / filename
            exists = path.exists()
            existing_count += int(exists)
            rows.append(
                {
                    "artifact": filename,
                    "status": "OK" if exists else "WARN",
                    "path": self._rel(path),
                    "modified": self._mtime_label(path),
                    "size_mb": self._size_mb(path),
                    "detail": "Found." if exists else "Missing expected root artifact.",
                }
            )

        misplaced = sorted(ui_dir.glob("*")) if ui_dir.exists() else []
        for path in misplaced:
            rows.append(
                {
                    "artifact": f"ui_v2/{path.name}",
                    "status": "WARN",
                    "path": self._rel(path),
                    "modified": self._mtime_label(path),
                    "size_mb": self._size_mb(path),
                    "detail": "Found under ui_v2; Regime Analysis expects runtime/artifacts/research/alpha_lab/latent_factors.",
                }
            )

        if existing_count == len(self.EXPECTED_VQ_FILES):
            status = "OK"
            detail = "VQ-VAE artifact set is complete."
        elif existing_count > 0:
            status = "WARN"
            detail = f"Partial VQ-VAE artifact set: {existing_count}/{len(self.EXPECTED_VQ_FILES)} files found."
        else:
            status = "WARN"
            detail = "No root VQ-VAE artifacts found. Cross-check page will show guidance only."
        checks.append(HealthCheck("Latent", "VQ-VAE artifacts", status, detail, self._rel(root_dir)))
        return pd.DataFrame(rows), checks

    def _model_registry_snapshot(self) -> tuple[pd.DataFrame, list[HealthCheck]]:
        rows = []
        checks = []
        if not self.db_path.exists():
            detail = "research_memory.db missing. Run init_research_db.py."
            return pd.DataFrame(), [HealthCheck("Model Registry", "model_artifacts", "FAIL", detail, self._rel(self.db_path))]

        try:
            with sqlite3.connect(self.db_path) as conn:
                if not self._table_exists_conn(conn, "model_artifacts"):
                    detail = "model_artifacts table missing. Run init_research_db.py."
                    return pd.DataFrame(), [
                        HealthCheck("Model Registry", "model_artifacts", "WARN", detail, self._rel(self.db_path))
                    ]
                registry = pd.read_sql_query(
                    """
                    SELECT
                        artifact_id,
                        model_name,
                        factor_id,
                        model_type,
                        artifact_format,
                        artifact_path,
                        legacy_path,
                        source_module,
                        data_path,
                        data_sha256,
                        artifact_sha256,
                        artifact_size_bytes,
                        feature_count,
                        target_col,
                        split_policy_json,
                        metrics_json,
                        hyperparams_json,
                        created_at
                    FROM model_artifacts
                    ORDER BY created_at DESC
                    """,
                    conn,
                )
        except Exception as exc:
            return pd.DataFrame(), [HealthCheck("Model Registry", "model_artifacts", "FAIL", str(exc), self._rel(self.db_path))]

        if registry.empty:
            detail = "Model registry exists, but no model artifacts have been registered yet."
            return pd.DataFrame(), [HealthCheck("Model Registry", "registered artifacts", "WARN", detail, self._rel(self.db_path))]

        for _, row in registry.iterrows():
            artifact_path = self._resolve_workspace_path(row.get("artifact_path"))
            legacy_path = self._resolve_workspace_path(row.get("legacy_path"))
            artifact_exists = artifact_path.exists() if artifact_path else False
            legacy_exists = legacy_path.exists() if legacy_path else np.nan

            if artifact_exists:
                status = "OK"
                detail = "Versioned artifact is present."
            else:
                status = "FAIL"
                detail = "Registered artifact path is missing."

            if artifact_exists and legacy_path is not None and not legacy_exists:
                status = "WARN"
                detail = "Versioned artifact is present, but legacy factor path is missing."

            rows.append(
                {
                    "status": status,
                    "artifact_id": row.get("artifact_id", ""),
                    "model_name": row.get("model_name", ""),
                    "factor_id": row.get("factor_id", ""),
                    "model_type": row.get("model_type", ""),
                    "artifact_format": row.get("artifact_format", ""),
                    "feature_count": row.get("feature_count", np.nan),
                    "target_col": row.get("target_col", ""),
                    "artifact_exists": bool(artifact_exists),
                    "legacy_exists": "" if pd.isna(legacy_exists) else bool(legacy_exists),
                    "artifact_path": row.get("artifact_path", ""),
                    "legacy_path": row.get("legacy_path", ""),
                    "data_path": row.get("data_path", ""),
                    "data_hash_short": self._short_hash(row.get("data_sha256")),
                    "artifact_hash_short": self._short_hash(row.get("artifact_sha256")),
                    "artifact_size_mb": self._bytes_to_mb(row.get("artifact_size_bytes")),
                    "split_policy_summary": self._json_summary(row.get("split_policy_json")),
                    "metrics_summary": self._json_summary(row.get("metrics_json")),
                    "hyperparams_summary": self._json_summary(row.get("hyperparams_json"), max_items=3),
                    "source_module": row.get("source_module", ""),
                    "created_at": row.get("created_at", ""),
                    "detail": detail,
                }
            )

        out = pd.DataFrame(rows)
        missing_count = int((out["status"] == "FAIL").sum())
        warn_count = int((out["status"] == "WARN").sum())
        if missing_count:
            status = "FAIL"
            detail = f"{missing_count} registered model artifacts point to missing files."
        elif warn_count:
            status = "WARN"
            detail = f"{warn_count} registered artifacts have legacy-path warnings."
        else:
            status = "OK"
            detail = f"{len(out)} registered model artifacts are present."
        checks.append(HealthCheck("Model Registry", "registered artifacts", status, detail, self._rel(self.db_path)))
        return out, checks

    def _infra_snapshot(self) -> tuple[pd.DataFrame, list[HealthCheck]]:
        rows = []
        checks = []
        packaged_native_dir = self.repo_root / "src" / "oqp" / "native"
        packaged_source = packaged_native_dir / "cpp" / "quant_core.cpp"
        legacy_source = self.legacy_cpp_dir / "quant_core.cpp" if self.legacy_cpp_dir else None
        packaged_extensions = sorted(
            packaged_native_dir.glob("_quant_core*.so"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        legacy_extensions = (
            sorted(
                self.legacy_cpp_dir.glob("quant_core*.so"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )
            if self.legacy_cpp_dir
            else []
        )
        ext = packaged_extensions[0] if packaged_extensions else (legacy_extensions[0] if legacy_extensions else None)
        source = packaged_source if packaged_extensions or packaged_source.exists() or legacy_source is None else legacy_source

        if ext is None:
            cpp_status = "FAIL"
            cpp_detail = "Compiled native extension missing. Build with python setup.py build_ext --inplace."
        elif source.exists() and ext.stat().st_mtime < source.stat().st_mtime:
            cpp_status = "WARN"
            cpp_detail = "Native extension is older than quant_core.cpp; rebuild recommended."
        else:
            cpp_status = "OK"
            cpp_detail = "Compiled extension exists and is not older than source."

        rows.append(
            {
                "check": "quant_core extension",
                "status": cpp_status,
                "path": self._rel(ext) if ext else self._rel(packaged_native_dir),
                "modified": self._mtime_label(ext) if ext else "",
                "detail": cpp_detail,
            }
        )
        checks.append(HealthCheck("Infra", "quant_core extension", cpp_status, cpp_detail, self._rel(ext) if ext else self._rel(packaged_native_dir)))

        import_status, import_detail = self._check_quant_core_import()
        rows.append(
            {
                "check": "quant_core import",
                "status": import_status,
                "path": "oqp.native._quant_core; legacy lab builds are fallback only",
                "modified": "",
                "detail": import_detail,
            }
        )
        checks.append(HealthCheck("Infra", "quant_core import", import_status, import_detail, "oqp.native"))

        optuna_db = self.repo_root / "runtime" / "db" / "research" / "alpha_lab" / "optimization_memory.db"
        rows.append(
            {
                "check": "optimization_memory.db",
                "status": "OK" if optuna_db.exists() else "WARN",
                "path": self._rel(optuna_db),
                "modified": self._mtime_label(optuna_db),
                "detail": "Optuna memory exists." if optuna_db.exists() else "No optimization DB yet; calibration will create it.",
            }
        )

        tick_ml_status, tick_ml_detail = self._tick_ml_cache_status()
        rows.append(
            {
                "check": "tick ML cache",
                "status": tick_ml_status,
                "path": self._rel(self.db_path),
                "modified": self._mtime_label(self.db_path),
                "detail": tick_ml_detail,
            }
        )
        checks.append(HealthCheck("Infra", "tick ML cache", tick_ml_status, tick_ml_detail, self._rel(self.db_path)))
        return pd.DataFrame(rows), checks

    def _parquet_summary(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        out: dict[str, Any] = {"rows": np.nan, "columns": np.nan, "date_range": "", "asset_count": np.nan}
        try:
            import pyarrow.parquet as pq

            pq_file = pq.ParquetFile(path)
            schema_names = list(pq_file.schema.names)
            out["rows"] = int(pq_file.metadata.num_rows)
            out["columns"] = len(schema_names)
        except Exception:
            try:
                sample = pd.read_parquet(path)
                schema_names = list(sample.columns)
                out["rows"] = len(sample)
                out["columns"] = len(sample.columns)
            except Exception as exc:
                out["detail"] = str(exc)
                return out

        date_col = self._first_present(schema_names, ["date", "datetime", "timestamp", "trading_day", "datetime_nano"])
        asset_col = self._first_present(schema_names, ["ticker", "symbol", "instrument_id", "contract", "wind_code"])
        try:
            read_cols = [col for col in [date_col, asset_col] if col]
            if read_cols:
                cols_df = pd.read_parquet(path, columns=read_cols)
                if date_col:
                    dates = pd.to_datetime(cols_df[date_col], errors="coerce").dropna()
                    if not dates.empty:
                        out["date_min"] = dates.min()
                        out["date_max"] = dates.max()
                        out["date_range"] = f"{dates.min().date()} -> {dates.max().date()}"
                if asset_col:
                    out["asset_count"] = int(cols_df[asset_col].nunique(dropna=True))
        except Exception:
            pass
        return out

    def _check_quant_core_import(self) -> tuple[str, str]:
        try:
            from oqp.native import quant_core_status

            status = quant_core_status(
                ("compute_tick_rtv_pipeline",),
                legacy_paths=(self.legacy_cpp_dir,) if self.legacy_cpp_dir else (),
            )
            if status.ok:
                return "OK", f"{status.module_name} imports and exposes compute_tick_rtv_pipeline."
            if status.available:
                missing = ", ".join(status.missing_features)
                return "WARN", f"{status.module_name} imports, but is missing: {missing}."
            return "FAIL", f"native quant core import failed: {status.error}"
        except Exception as exc:
            return "FAIL", f"native quant core import failed: {exc}"

    def _tick_ml_cache_status(self) -> tuple[str, str]:
        if not self.db_path.exists():
            return "FAIL", "research_memory.db missing."
        try:
            with sqlite3.connect(self.db_path) as conn:
                if not self._table_exists_conn(conn, "tick_ml_studies"):
                    return "WARN", "tick_ml_studies table missing. Run init_research_db.py."
                count = conn.execute("SELECT COUNT(*) FROM tick_ml_studies").fetchone()[0]
                return ("OK" if count else "WARN", f"{count} cached tick ML studies found.")
        except Exception as exc:
            return "FAIL", str(exc)

    def _resolve_workspace_path(self, value: Any) -> Path | None:
        if value is None or pd.isna(value) or not str(value).strip():
            return None
        path = Path(str(value))
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == "execution_logs":
            return self.logs_dir / Path(*path.parts[1:])
        repo_path = self.repo_root / path
        return repo_path if repo_path.exists() else self.base_dir / path

    def _resolve_returns_path(self, run_id: str, returns_file_path: str | None) -> Path:
        if returns_file_path is not None and not pd.isna(returns_file_path) and str(returns_file_path).strip():
            path = Path(str(returns_file_path))
            if path.is_absolute():
                return path
            if path.parts and path.parts[0] == "execution_logs":
                return self.logs_dir / Path(*path.parts[1:])
            repo_path = self.repo_root / path
            return repo_path if repo_path.exists() else self.base_dir / path
        return self.logs_dir / "returns" / f"returns_{run_id}.csv"

    def _is_data_stale(self, date_value: Any, max_days: int = 30) -> bool:
        if date_value is None or pd.isna(date_value):
            return False
        date_value = pd.to_datetime(date_value, errors="coerce")
        if pd.isna(date_value):
            return False
        return (pd.Timestamp.today().normalize() - date_value.normalize()).days > max_days

    def _parse_range_from_name(self, name: str) -> str:
        match = re.search(r"(\d{8})_(\d{8})", name)
        if not match:
            return ""
        start, end = match.groups()
        try:
            return f"{pd.to_datetime(start).date()} -> {pd.to_datetime(end).date()}"
        except Exception:
            return f"{start} -> {end}"

    def _size_mb(self, path: Path | None) -> float:
        if path is None or not path.exists():
            return np.nan
        return round(path.stat().st_size / 1_000_000, 2)

    def _mtime_label(self, path: Path | None) -> str:
        if path is None or not path.exists():
            return ""
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    def _rel(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return str(path.relative_to(self.repo_root))
        except Exception:
            return str(path)

    @staticmethod
    def _short_hash(value: Any, length: int = 12) -> str:
        if value is None or pd.isna(value) or not str(value).strip():
            return ""
        return str(value)[:length]

    @staticmethod
    def _bytes_to_mb(value: Any) -> float:
        if value is None or pd.isna(value):
            return np.nan
        try:
            return round(float(value) / 1_000_000, 3)
        except Exception:
            return np.nan

    @staticmethod
    def _json_summary(value: Any, max_items: int = 5) -> str:
        if value is None or pd.isna(value) or not str(value).strip():
            return ""
        try:
            parsed = json.loads(str(value))
        except Exception:
            return str(value)[:240]

        if isinstance(parsed, dict):
            parts = []
            for key, item in list(parsed.items())[:max_items]:
                if isinstance(item, float):
                    item_text = f"{item:.4g}"
                else:
                    item_text = str(item)
                parts.append(f"{key}={item_text}")
            return "; ".join(parts)
        if isinstance(parsed, list):
            return f"{len(parsed)} items"
        return str(parsed)

    @staticmethod
    def _first_present(columns: list[str], candidates: list[str]) -> str | None:
        lowered = {col.lower(): col for col in columns}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        return None

    @staticmethod
    def _table_exists_conn(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _readiness_score(checks: pd.DataFrame) -> float:
        if checks.empty:
            return 0.0
        ranks = checks["status"].map(STATUS_RANK).fillna(0)
        return float(ranks.sum() / (2 * len(ranks)) * 100)

    @staticmethod
    def _style_status(df: pd.DataFrame):
        def style_value(value: Any) -> str:
            color = STATUS_COLORS.get(str(value), "")
            if not color:
                return ""
            return f"background-color: {color}; color: white; font-weight: 700"

        return df.style.map(style_value, subset=[col for col in df.columns if col.lower() == "status"])
