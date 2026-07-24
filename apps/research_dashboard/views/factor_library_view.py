from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from config import BASE_DIR, get_plotly_template
from oqp.research.factor_purity import build_factor_purity_review_index


COPY = {
    "EN": {
        "title": "Factor Library",
        "subtitle": "Live factor inventory and factor-level research evidence.",
        "active_factors": "Normalized factors",
        "families": "Factor families",
        "cohorts": "Comparison cohorts",
        "archived": "Archived duplicates",
        "violations": "Boundary issues",
        "average_ic": "Avg tested IC",
        "ic_coverage": "{tested} / {total} tested",
        "ic_help": (
            "Signed full-sample Pearson IC, averaged once per factor with a "
            "current-definition Phase 2 predictive-evidence bundle. Untested "
            "or stale-definition factors are excluded rather than treated as zero."
        ),
        "inventory_tab": "Inventory",
        "drilldown_tab": "Factor Drilldown",
        "inventory": "Active inventory",
        "family_mix": "Factor family mix",
        "frequency_mix": "Factors by frequency",
        "market_mix": "Factors by native market",
        "component_mix": "Research components",
        "no_artifacts": "The normalized factor manifest is not available yet.",
        "all": "All",
        "market": "Market",
        "frequency": "Frequency",
        "family": "Family",
        "cohort": "Cohort",
        "status": "Disposition",
        "active": "Active research",
        "archive": "Archived",
        "factor_id": "Factor ID",
        "subfamily": "Subfamily",
        "portfolio_layer": "Portfolio layer",
        "collection": "Collection",
        "source_alpha_number": "Source #",
        "implementation_status": "Research status",
        "block_reason": "Data / semantic blocker",
        "futures_cn_adaptation_status": "CN-futures adaptation",
        "futures_cn_adaptation_geometry": "Adaptation geometry",
        "futures_cn_adaptation_reason": "Required change",
        "reference_license": "Reference license",
        "predictive_evidence_status": "Predictive evidence",
        "predictive_evidence_current": "Evidence currency",
        "predictive_evidence_current_yes": "Current",
        "predictive_evidence_current_no": "Stale / unavailable",
        "predictive_evidence_definition_fingerprint": "Evidence definition fingerprint",
        "content_pure": "Content purity",
        "content_pure_yes": "Pure factor",
        "content_pure_no": "Boundary issue",
        "purity_issues": "Purity issues",
        "purity_review_group": "Purity review group",
        "purity_review_conclusion": "Review conclusion",
        "review_group": "Purity review group",
        "review_conclusion": "Review conclusion",
        "extracted_component_ids": "Extracted component IDs",
        "lookahead_fix": "Lookahead fix",
        "search": "Find factor",
        "output_geometry": "Factor boundary",
        "pure_signal": "Pure signal",
        "embedded_target": "Embedded target",
        "legacy_target": "Signal + legacy target",
        "embedded_hold": "Embedded holding rule",
        "count": "Count",
        "component": "Component",
        "factor_count": "Factors",
        "share": "Share",
    },
    "ZH": {
        "title": "因子库",
        "subtitle": "动态展示因子库存与单因子研究证据。",
        "active_factors": "标准化在库因子",
        "families": "因子类别",
        "cohorts": "统一比较组别",
        "archived": "去重归档",
        "violations": "边界异常",
        "average_ic": "已测平均 IC",
        "ic_coverage": "已测试 {tested} / {total}",
        "ic_help": (
            "对具备当前定义第二阶段预测证据包的因子，按每个因子一次计算全样本有符号 "
            "Pearson IC 均值。未测试或定义已变更的因子不按零值计入。"
        ),
        "inventory_tab": "因子清单",
        "drilldown_tab": "因子明细",
        "inventory": "在库因子清单",
        "family_mix": "因子类别分布",
        "frequency_mix": "因子频率分布",
        "market_mix": "因子原生市场分布",
        "component_mix": "研究组件分布",
        "no_artifacts": "尚未生成标准化因子清单。",
        "all": "全部",
        "market": "市场",
        "frequency": "频率",
        "family": "类别",
        "cohort": "比较组别",
        "status": "处理状态",
        "active": "在库研究",
        "archive": "已归档",
        "factor_id": "因子 ID",
        "subfamily": "子类别",
        "portfolio_layer": "组合层级",
        "collection": "因子集合",
        "source_alpha_number": "原始编号",
        "implementation_status": "研究状态",
        "block_reason": "数据 / 语义阻塞",
        "futures_cn_adaptation_status": "中国期货适配",
        "futures_cn_adaptation_geometry": "适配结构",
        "futures_cn_adaptation_reason": "所需改动",
        "reference_license": "参考许可证",
        "predictive_evidence_status": "预测证据",
        "predictive_evidence_current": "证据时效",
        "predictive_evidence_current_yes": "当前有效",
        "predictive_evidence_current_no": "已过期 / 不可用",
        "predictive_evidence_definition_fingerprint": "证据定义指纹",
        "content_pure": "内容纯净度",
        "content_pure_yes": "纯因子",
        "content_pure_no": "边界异常",
        "purity_issues": "纯净度问题",
        "purity_review_group": "纯净度复核组",
        "purity_review_conclusion": "复核结论",
        "review_group": "纯净度复核组",
        "review_conclusion": "复核结论",
        "extracted_component_ids": "已拆分组件 ID",
        "lookahead_fix": "前视偏差修复",
        "search": "搜索因子",
        "output_geometry": "因子边界",
        "pure_signal": "纯信号",
        "embedded_target": "内嵌目标仓位",
        "legacy_target": "信号 + 遗留目标仓位",
        "embedded_hold": "内嵌持有规则",
        "count": "数量",
        "component": "组件",
        "factor_count": "因子数",
        "share": "占比",
    },
}


@dataclass(frozen=True)
class FactorLibrarySnapshot:
    manifest: pd.DataFrame
    component_summary: dict[str, Any]
    archived_ids: tuple[str, ...]
    predictive_evidence: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _truthy(values: pd.Series) -> pd.Series:
    return values.eq(True).fillna(False) | (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )


def _display_sequence(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value)


def _enrich_factor_boundary_review(
    manifest: pd.DataFrame,
    base_dir: Path,
) -> pd.DataFrame:
    """Attach the static boundary-review ledger when the artifact is older."""

    if manifest.empty or "factor_id" not in manifest.columns:
        return manifest
    review_path = (
        base_dir / "departments" / "research" / "factors" / "purity_review.yaml"
    )
    if not review_path.is_file():
        return manifest
    try:
        review_index = build_factor_purity_review_index(
            manifest["factor_id"].dropna().astype(str).tolist(),
            review_path=review_path,
        )
    except (OSError, ValueError, yaml.YAMLError):
        return manifest

    review = pd.DataFrame(
        [
            {
                "factor_id": factor_id,
                "purity_review_group": details.get("review_group", ""),
                "purity_review_conclusion": details.get(
                    "review_conclusion",
                    "",
                ),
                "extracted_component_ids": ", ".join(
                    str(value)
                    for value in details.get("extracted_component_ids", ())
                ),
                "lookahead_fix": str(details.get("lookahead_fix", "")),
            }
            for factor_id, details in review_index.items()
        ]
    )
    output = manifest.copy()
    missing_columns = [
        column
        for column in review.columns
        if column != "factor_id" and column not in output.columns
    ]
    if missing_columns:
        output = output.merge(
            review[["factor_id", *missing_columns]],
            on="factor_id",
            how="left",
            validate="one_to_one",
        )

    stable_ids_path = (
        base_dir / "departments" / "research" / "factors" / "stable_ids.yaml"
    )
    if stable_ids_path.is_file():
        try:
            stable_ids = yaml.safe_load(
                stable_ids_path.read_text(encoding="utf-8")
            ) or {}
            boundary_batch = (
                stable_ids.get("normalization_batches", {}).get(
                    "active_factor_boundary_split_020",
                    {},
                )
            )
            lookahead_fixes = boundary_batch.get("lookahead_fixes", {})
            if isinstance(lookahead_fixes, dict):
                mapped = output["factor_id"].map(
                    lambda value: str(lookahead_fixes.get(str(value), ""))
                )
                if "lookahead_fix" not in output.columns:
                    output["lookahead_fix"] = mapped
                else:
                    output["lookahead_fix"] = (
                        output["lookahead_fix"].fillna("").astype(str).mask(
                            output["lookahead_fix"].fillna("").astype(str).eq(""),
                            mapped,
                        )
                    )
        except (OSError, yaml.YAMLError, AttributeError):
            pass

    if "content_pure" not in output.columns:
        conclusion_column = (
            "purity_review_conclusion"
            if "purity_review_conclusion" in output.columns
            else "review_conclusion"
        )
        output["content_pure"] = (
            output.get(conclusion_column, pd.Series("", index=output.index))
            .fillna("")
            .astype(str)
            .str.startswith("pure")
        )
    if "purity_issues" not in output.columns:
        output["purity_issues"] = ""
    return output


def load_factor_library_snapshot(base_dir: str = str(BASE_DIR)) -> FactorLibrarySnapshot:
    base_path = Path(base_dir)
    root = base_path / "runtime" / "artifacts" / "research"
    manifest = _read_csv(
        root / "factor_registry_normalization" / "cohort_manifest.csv"
    )
    manifest = _enrich_factor_boundary_review(manifest, base_path)
    component_summary = _read_json(
        root / "component_registry_audit" / "summary.json"
    )
    dedup_root = root / "daily_cn_futures_factor_dedup"
    dedup_summary = _read_json(dedup_root / "normalized_daily_summary.json")
    predictive_evidence = _read_csv(root / "predictive_evidence" / "evidence_index.csv")
    return FactorLibrarySnapshot(
        manifest=manifest,
        component_summary=component_summary,
        archived_ids=tuple(sorted(str(value) for value in dedup_summary.get("archived_ids", []))),
        predictive_evidence=predictive_evidence,
    )


class FactorLibraryView:
    def __init__(self, base_dir: str | Path = BASE_DIR):
        self.base_dir = Path(base_dir)

    def render(
        self,
        lang: str = "EN",
        theme_mode: str = "LIGHT",
        drilldown_renderer: Callable[[], None] | None = None,
    ) -> None:
        copy = COPY.get(lang, COPY["EN"])
        snapshot = load_factor_library_snapshot(str(self.base_dir))
        st.markdown(f"### {copy['title']}")
        st.caption(copy["subtitle"])
        if snapshot.manifest.empty:
            st.info(copy["no_artifacts"])
            return

        self._render_summary(snapshot, copy)
        tab_labels = [copy["inventory_tab"]]
        if drilldown_renderer is not None:
            tab_labels.append(copy["drilldown_tab"])
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            self._render_family_mix(snapshot.manifest, theme_mode, copy)
            frequency_col, market_col = st.columns(2)
            with frequency_col:
                self._render_dimension_mix(
                    snapshot.manifest,
                    dimension="data_frequency",
                    title=copy["frequency_mix"],
                    theme_mode=theme_mode,
                    count_label=copy["factor_count"],
                    share_label=copy["share"],
                )
            with market_col:
                self._render_dimension_mix(
                    snapshot.manifest,
                    dimension="native_market",
                    title=copy["market_mix"],
                    theme_mode=theme_mode,
                    count_label=copy["factor_count"],
                    share_label=copy["share"],
                )
            st.markdown(f"#### {copy['inventory']}")
            filtered = self._render_inventory_filters(snapshot.manifest, copy)
            self._render_inventory(filtered, copy)

        if drilldown_renderer is not None:
            with tabs[1]:
                drilldown_renderer()

    @staticmethod
    def _render_summary(snapshot: FactorLibrarySnapshot, copy: dict[str, Any]) -> None:
        cards = st.columns(6)
        cards[0].metric(copy["active_factors"], f"{len(snapshot.manifest):,}")
        cards[1].metric(copy["families"], f"{snapshot.manifest['factor_family'].nunique():,}")
        cards[2].metric(copy["cohorts"], f"{snapshot.manifest['deduplication_cohort'].nunique():,}")
        cards[3].metric(copy["archived"], f"{len(snapshot.archived_ids):,}")
        embedded = FactorLibraryView._boundary_violation_count(
            snapshot.manifest
        )
        cards[4].metric(copy["violations"], f"{int(embedded):,}")
        average_ic, tested_count = FactorLibraryView._average_tested_ic(snapshot)
        cards[5].metric(
            copy["average_ic"],
            "N/A" if pd.isna(average_ic) else f"{average_ic:.2%}",
            help=copy["ic_help"],
        )
        cards[5].caption(
            copy["ic_coverage"].format(
                tested=tested_count,
                total=len(snapshot.manifest),
            )
        )

    @staticmethod
    def _boundary_violation_count(manifest: pd.DataFrame) -> int:
        if "content_pure" in manifest.columns:
            content_pure = manifest["content_pure"]
            violations = (
                content_pure.eq(False).fillna(False)
                | content_pure.astype("string")
                .fillna("")
                .str.strip()
                .str.lower()
                .eq("false")
            )
            return int(violations.sum())
        pure_layers = {"alpha_score", "predictive_signal"}
        return int((~manifest["portfolio_layer"].isin(pure_layers)).sum())

    @staticmethod
    def _average_tested_ic(
        snapshot: FactorLibrarySnapshot,
    ) -> tuple[float, int]:
        evidence = snapshot.predictive_evidence
        if evidence.empty or not {"factor_id", "mean_ic"}.issubset(evidence.columns):
            return float("nan"), 0
        manifest = snapshot.manifest
        if "predictive_evidence_current" in manifest.columns:
            current_manifest = manifest.loc[
                _truthy(manifest["predictive_evidence_current"])
            ]
        else:
            current_manifest = manifest
        active_ids = set(current_manifest["factor_id"].dropna().astype(str))
        tested = evidence.loc[
            evidence["factor_id"].astype(str).isin(active_ids),
            ["factor_id", "mean_ic"],
        ].copy()
        tested["mean_ic"] = pd.to_numeric(tested["mean_ic"], errors="coerce")
        per_factor = tested.dropna(subset=["mean_ic"]).groupby("factor_id")["mean_ic"].mean()
        if per_factor.empty:
            return float("nan"), 0
        return float(per_factor.mean()), int(len(per_factor))

    @staticmethod
    def _render_inventory_filters(manifest: pd.DataFrame, copy: dict[str, Any]) -> pd.DataFrame:
        optional_filter_columns = [
            column
            for column in (
                "futures_cn_adaptation_status",
                "implementation_status",
                "predictive_evidence_status",
            )
            if column in manifest.columns
        ]
        controls = st.columns(3 + len(optional_filter_columns))
        market = controls[0].selectbox(
            copy["market"],
            [copy["all"], *sorted(manifest["native_market"].dropna().unique())],
            key="factor_library_market",
        )
        frequency = controls[1].selectbox(
            copy["frequency"],
            [copy["all"], *sorted(manifest["data_frequency"].dropna().unique())],
            key="factor_library_frequency",
        )
        family = controls[2].selectbox(
            copy["family"],
            [copy["all"], *sorted(manifest["factor_family"].dropna().unique())],
            key="factor_library_family",
        )
        selected_optional: dict[str, str] = {}
        for offset, column in enumerate(optional_filter_columns, start=3):
            values = sorted(
                value
                for value in manifest[column]
                .dropna()
                .astype(str)
                .unique()
                if value
            )
            selected_optional[column] = controls[offset].selectbox(
                copy[column],
                [copy["all"], *values],
                key=f"factor_library_{column}",
            )
        search = st.text_input(
            copy["search"],
            key="factor_library_search",
            placeholder="Factor ID, source #, family",
        ).strip()
        filtered = manifest.copy()
        if market != copy["all"]:
            filtered = filtered.loc[filtered["native_market"].eq(market)]
        if frequency != copy["all"]:
            filtered = filtered.loc[filtered["data_frequency"].eq(frequency)]
        if family != copy["all"]:
            filtered = filtered.loc[filtered["factor_family"].eq(family)]
        for column, selected in selected_optional.items():
            if selected != copy["all"]:
                filtered = filtered.loc[
                    filtered[column].fillna("").astype(str).eq(selected)
                ]
        if search:
            searchable_columns = [
                column
                for column in (
                    "factor_id",
                    "factor_family",
                    "factor_subfamily",
                    "collection",
                    "source_alpha_number",
                )
                if column in filtered.columns
            ]
            mask = pd.Series(False, index=filtered.index)
            for column in searchable_columns:
                mask |= (
                    filtered[column]
                    .fillna("")
                    .astype(str)
                    .str.contains(search, case=False, regex=False)
                )
            filtered = filtered.loc[mask]
        return filtered

    @staticmethod
    def _sort_by_factor_number(manifest: pd.DataFrame) -> pd.DataFrame:
        if manifest.empty or "factor_id" not in manifest.columns:
            return manifest

        output = manifest.copy()
        output["_factor_number"] = pd.to_numeric(
            output["factor_id"]
            .astype(str)
            .str.extract(r"(?i)^fac_?0*(\d+)", expand=False),
            errors="coerce",
        )
        return (
            output.sort_values(
                ["_factor_number", "factor_id"],
                ascending=[True, True],
                na_position="last",
                kind="stable",
            )
            .drop(columns="_factor_number")
            .reset_index(drop=True)
        )

    @staticmethod
    def _render_inventory(manifest: pd.DataFrame, copy: dict[str, Any]) -> None:
        output_labels = {
            "alpha_score": copy["pure_signal"],
            "predictive_signal": copy["pure_signal"],
            "direct_target": copy["embedded_target"],
            "alpha_score_with_legacy_direct_target": copy["legacy_target"],
            "event_signal_with_embedded_hold": copy["embedded_hold"],
        }
        columns = [
            "factor_id",
            "output_geometry",
            "factor_family",
            "factor_subfamily",
            "native_market",
            "data_frequency",
            "portfolio_layer",
            "deduplication_cohort",
        ]
        optional_columns = [
            "collection",
            "source_alpha_number",
            "predictive_evidence_status",
            "predictive_evidence_current",
            "predictive_evidence_definition_fingerprint",
            "content_pure",
            "purity_issues",
            "purity_review_group",
            "purity_review_conclusion",
            "review_group",
            "review_conclusion",
            "extracted_component_ids",
            "lookahead_fix",
            "futures_cn_adaptation_status",
            "futures_cn_adaptation_geometry",
            "futures_cn_adaptation_reason",
            "implementation_status",
            "block_reason",
            "reference_license",
        ]
        columns.extend(
            column for column in optional_columns if column in manifest.columns
        )
        display = manifest.copy()
        if "predictive_evidence_current" in display.columns:
            display["predictive_evidence_current"] = _truthy(
                display["predictive_evidence_current"]
            ).map(
                {
                    True: copy["predictive_evidence_current_yes"],
                    False: copy["predictive_evidence_current_no"],
                }
            )
        if "content_pure" in display.columns:
            pure_mask = _truthy(display["content_pure"])
            display["content_pure"] = pure_mask.map(
                {
                    True: copy["content_pure_yes"],
                    False: copy["content_pure_no"],
                }
            )
        if "extracted_component_ids" in display.columns:
            display["extracted_component_ids"] = display[
                "extracted_component_ids"
            ].map(_display_sequence)
        display["output_geometry"] = display["portfolio_layer"].map(
            lambda value: output_labels.get(
                str(value), str(value).replace("_", " ").title()
            )
        )
        display = FactorLibraryView._sort_by_factor_number(display)[columns].rename(
            columns={
                "factor_id": copy["factor_id"],
                "factor_family": copy["family"],
                "factor_subfamily": copy["subfamily"],
                "native_market": copy["market"],
                "data_frequency": copy["frequency"],
                "output_geometry": copy["output_geometry"],
                "portfolio_layer": copy["portfolio_layer"],
                "deduplication_cohort": copy["cohort"],
                "collection": copy["collection"],
                "source_alpha_number": copy["source_alpha_number"],
                "predictive_evidence_status": copy[
                    "predictive_evidence_status"
                ],
                "predictive_evidence_current": copy[
                    "predictive_evidence_current"
                ],
                "predictive_evidence_definition_fingerprint": copy[
                    "predictive_evidence_definition_fingerprint"
                ],
                "content_pure": copy["content_pure"],
                "purity_issues": copy["purity_issues"],
                "purity_review_group": copy["purity_review_group"],
                "purity_review_conclusion": copy[
                    "purity_review_conclusion"
                ],
                "review_group": copy["review_group"],
                "review_conclusion": copy["review_conclusion"],
                "extracted_component_ids": copy[
                    "extracted_component_ids"
                ],
                "lookahead_fix": copy["lookahead_fix"],
                "futures_cn_adaptation_status": copy[
                    "futures_cn_adaptation_status"
                ],
                "futures_cn_adaptation_geometry": copy[
                    "futures_cn_adaptation_geometry"
                ],
                "futures_cn_adaptation_reason": copy[
                    "futures_cn_adaptation_reason"
                ],
                "implementation_status": copy["implementation_status"],
                "block_reason": copy["block_reason"],
                "reference_license": copy["reference_license"],
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True, height=320)

    @staticmethod
    def _render_family_mix(manifest: pd.DataFrame, theme_mode: str, copy: dict[str, Any]) -> None:
        counts = manifest.groupby("factor_family", as_index=False).size()
        counts = counts.rename(columns={"size": "count"}).sort_values("count")
        counts["label"] = counts["factor_family"].map(
            lambda value: FactorLibraryView._dimension_label(
                "factor_family",
                value,
            )
        )
        counts["share"] = counts["count"] / counts["count"].sum()
        fig = px.bar(
            counts,
            x="count",
            y="label",
            orientation="h",
            color_discrete_sequence=["#2563eb"],
            text="count",
            custom_data=["share"],
            template=get_plotly_template(theme_mode),
            title=copy["family_mix"],
        )
        FactorLibraryView._style_distribution_chart(
            fig,
            height=max(430, 28 * len(counts) + 120),
            theme_mode=theme_mode,
            count_label=copy["factor_count"],
            share_label=copy["share"],
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    @staticmethod
    def _render_dimension_mix(
        manifest: pd.DataFrame,
        *,
        dimension: str,
        title: str,
        theme_mode: str,
        count_label: str,
        share_label: str,
    ) -> None:
        if manifest.empty or dimension not in manifest.columns:
            return
        counts = (
            manifest.assign(
                _dimension=manifest[dimension].fillna("Unspecified").astype(str)
            )
            .groupby("_dimension", as_index=False)
            .size()
            .rename(columns={"_dimension": dimension, "size": "count"})
            .sort_values("count")
        )
        counts["label"] = counts[dimension].map(
            lambda value: FactorLibraryView._dimension_label(dimension, value)
        )
        counts["share"] = counts["count"] / counts["count"].sum()
        fig = px.bar(
            counts,
            x="count",
            y="label",
            orientation="h",
            color="label",
            color_discrete_sequence=[
                "#2563eb",
                "#059669",
                "#d97706",
                "#dc2626",
                "#0891b2",
                "#7c3aed",
                "#64748b",
            ],
            text="count",
            custom_data=["share"],
            template=get_plotly_template(theme_mode),
            title=title,
        )
        FactorLibraryView._style_distribution_chart(
            fig,
            height=max(300, 52 * len(counts) + 140),
            theme_mode=theme_mode,
            count_label=count_label,
            share_label=share_label,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    @staticmethod
    def _dimension_label(dimension: str, value: Any) -> str:
        raw = str(value).strip()
        labels = {
            "data_frequency": {
                "daily": "Daily",
                "intraday_1m": "Intraday (1m)",
                "minute": "Minute",
                "tick": "Tick",
            },
            "native_market": {
                "FUTURES_CN": "CN Futures",
                "EQUITY_CN": "CN Equities",
                "EQUITY_US": "US Equities",
                "OPTIONS_CN": "CN Options",
                "OPTIONS_US": "US Options",
            },
        }
        return labels.get(dimension, {}).get(
            raw,
            raw.replace("_", " ").strip().title(),
        )

    @staticmethod
    def _style_distribution_chart(
        fig,
        *,
        height: int,
        theme_mode: str,
        count_label: str,
        share_label: str,
    ) -> None:
        dark = str(theme_mode).upper() == "DARK"
        hover_background = "#111827" if dark else "#ffffff"
        hover_text = "#f8fafc" if dark else "#111827"
        grid_color = "rgba(148, 163, 184, 0.18)"
        fig.update_traces(
            marker_line_width=0,
            texttemplate="%{x:,}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{count_label}: %{{x:,}}<br>"
                f"{share_label}: %{{customdata[0]:.1%}}"
                "<extra></extra>"
            ),
        )
        fig.update_layout(
            height=height,
            showlegend=False,
            bargap=0.32,
            margin=dict(l=10, r=42, t=54, b=35),
            hoverlabel=dict(
                bgcolor=hover_background,
                bordercolor="#cbd5e1",
                font=dict(color=hover_text, size=13),
                align="left",
            ),
            uniformtext_minsize=11,
            uniformtext_mode="show",
        )
        fig.update_xaxes(
            title=count_label,
            rangemode="tozero",
            showgrid=True,
            gridcolor=grid_color,
            zeroline=False,
        )
        fig.update_yaxes(
            title=None,
            categoryorder="total ascending",
            showgrid=False,
        )

    @staticmethod
    def _render_component_mix(summary: dict[str, Any], theme_mode: str, copy: dict[str, Any]) -> None:
        labels = {
            "factor": "Factor / 因子",
            "router": "Router / 路由",
            "router_state": "State / 状态",
            "position_policy": "Position / 仓位",
            "strategy_risk_overlay": "Overlay / 风控",
            "diagnostic": "Diagnostic / 诊断",
        }
        rows = [
            {"component": labels.get(kind, kind), "count": int(count)}
            for kind, count in (summary.get("kind_counts") or {}).items()
        ]
        frame = pd.DataFrame(rows).sort_values("count") if rows else pd.DataFrame()
        if frame.empty:
            return
        fig = px.bar(
            frame,
            x="count",
            y="component",
            orientation="h",
            color="component",
            color_discrete_sequence=[
                "#0f766e",
                "#2563eb",
                "#9333ea",
                "#ca8a04",
                "#e11d48",
                "#64748b",
            ],
            template=get_plotly_template(theme_mode),
            title=copy["component_mix"],
        )
        fig.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=50, b=20))
        fig.update_xaxes(title=copy["count"])
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)

__all__ = [
    "FactorLibrarySnapshot",
    "FactorLibraryView",
    "load_factor_library_snapshot",
]
