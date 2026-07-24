from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import streamlit as st

from config import BASE_DIR
from oqp.research.sleeves.registry import (
    sleeve_definition_fingerprint,
    sleeve_implementation_fingerprint,
)


COPY = {
    "EN": {
        "title": "Sleeve Library",
        "subtitle": "Reusable factor-to-position translations and their standalone evidence.",
        "registered": "Registered sleeves",
        "tested": "Tested configurations",
        "eligible": "Router eligible",
        "selection_rules": "Selection rules",
        "weighting_methods": "Weighting methods",
        "inventory_tab": "Inventory",
        "drilldown_tab": "Sleeve Drilldown",
        "evidence_tab": "Standalone Evidence",
        "trial_history_tab": "Trial History",
        "inventory": "Active sleeve inventory",
        "market": "Market",
        "frequency": "Frequency",
        "status": "Status",
        "all": "All",
        "sleeve_id": "Sleeve ID",
        "name": "Sleeve",
        "source_factor": "Source factor",
        "source_factors": "Source factor IDs",
        "factor_scope": "Factor scope",
        "any_compatible_factor": "Any compatible factor",
        "construction": "Construction",
        "selection": "Selection rule",
        "weighting": "Weighting",
        "outer_selection": "Top/bottom {percent}%",
        "zscore_selection": "Continuous z-score",
        "expression": "Expression",
        "decision_lag": "Decision lag",
        "tested_count": "Tests",
        "eligible_count": "Eligible tests",
        "select": "Inspect sleeve",
        "economic_role": "Economic role",
        "limitations": "Known limitations",
        "not_documented": "No economic role has been documented for this sleeve yet.",
        "contract": "Construction contract",
        "field": "Field",
        "value": "Value",
        "source": "Source",
        "fingerprints": "Sleeve fingerprints",
        "definition_fingerprint": "Definition fingerprint",
        "implementation_fingerprint": "Implementation fingerprint",
        "evidence": "Phase 4 standalone configurations",
        "evidence_empty": "No Phase 4 standalone evidence has been recorded for this sleeve.",
        "factor_id": "Factor ID",
        "standalone_status": "Standalone status",
        "router_eligible": "Router eligible",
        "validation_sharpe": "Validation net Sharpe",
        "holdout_sharpe": "Holdout net Sharpe",
        "trial_history": "Factor + sleeve trial history",
        "trial_history_empty": "No factor + sleeve trials have been recorded yet.",
        "trial_id": "Trial ID",
        "research_split": "Research split",
        "trial_status": "Trial status",
        "full_net_sharpe": "Full net Sharpe",
        "q1_net_sharpe": "Q1 net Sharpe",
        "q2_net_sharpe": "Q2 net Sharpe",
        "q3_net_sharpe": "Q3 net Sharpe",
        "q4_net_sharpe": "Q4 net Sharpe",
        "min_quartile_net_sharpe": "Minimum Q1–Q4 net Sharpe",
        "net_annual_return": "Net annual return",
        "max_drawdown": "Maximum drawdown",
        "annualized_cost": "Annualized cost",
        "average_turnover": "Average turnover",
        "artifact_path": "Artifact path",
        "yes": "Yes",
        "no": "No",
        "no_inventory": "No registered sleeve definitions were found.",
    },
    "ZH": {
        "title": "策略腿库",
        "subtitle": "可复用的因子到仓位转换组件及其独立表现证据。",
        "registered": "已注册策略腿",
        "tested": "已测试配置",
        "eligible": "可进入路由",
        "selection_rules": "选样规则数",
        "weighting_methods": "权重方式数",
        "inventory_tab": "组件清单",
        "drilldown_tab": "策略腿明细",
        "evidence_tab": "独立表现证据",
        "trial_history_tab": "试验记录",
        "inventory": "在库策略腿",
        "market": "市场",
        "frequency": "频率",
        "status": "状态",
        "all": "全部",
        "sleeve_id": "策略腿 ID",
        "name": "策略腿",
        "source_factor": "来源因子",
        "source_factors": "来源因子 ID",
        "factor_scope": "因子适用范围",
        "any_compatible_factor": "任意兼容因子",
        "construction": "构建方式",
        "selection": "选样规则",
        "weighting": "权重方式",
        "outer_selection": "最高/最低 {percent}%",
        "zscore_selection": "连续 Z 分数",
        "expression": "多空表达",
        "decision_lag": "决策延迟",
        "tested_count": "测试数",
        "eligible_count": "合格测试数",
        "select": "查看策略腿",
        "economic_role": "经济逻辑",
        "limitations": "已知局限",
        "not_documented": "该策略腿尚未记录经济逻辑。",
        "contract": "构建合约",
        "field": "字段",
        "value": "数值",
        "source": "源码",
        "fingerprints": "策略腿指纹",
        "definition_fingerprint": "定义指纹",
        "implementation_fingerprint": "实现指纹",
        "evidence": "第四阶段独立测试配置",
        "evidence_empty": "该策略腿尚未记录第四阶段独立测试证据。",
        "factor_id": "因子 ID",
        "standalone_status": "独立测试状态",
        "router_eligible": "可进入路由",
        "validation_sharpe": "验证集净夏普",
        "holdout_sharpe": "留出集净夏普",
        "trial_history": "因子 + 策略腿试验记录",
        "trial_history_empty": "尚未记录因子 + 策略腿试验。",
        "trial_id": "试验 ID",
        "research_split": "研究样本",
        "trial_status": "试验状态",
        "full_net_sharpe": "全样本净夏普",
        "q1_net_sharpe": "Q1 净夏普",
        "q2_net_sharpe": "Q2 净夏普",
        "q3_net_sharpe": "Q3 净夏普",
        "q4_net_sharpe": "Q4 净夏普",
        "min_quartile_net_sharpe": "Q1–Q4 最低净夏普",
        "net_annual_return": "净年化收益",
        "max_drawdown": "最大回撤",
        "annualized_cost": "年化成本",
        "average_turnover": "平均换手",
        "artifact_path": "产物路径",
        "yes": "是",
        "no": "否",
        "no_inventory": "未找到已注册的策略腿定义。",
    },
}


STATUS_LABELS = {
    "EN": {
        "rejected_after_cost": "Rejected after costs",
        "candidate_positive_after_cost": "Positive-cost candidate",
        "legacy_historical_sleeve": "Historical component",
        "fixed_research_default": "Research default",
        "registered_untested": "Registered, untested",
        "blocked_validation": "Blocked in validation",
    },
    "ZH": {
        "rejected_after_cost": "成本后被否决",
        "candidate_positive_after_cost": "成本后为正候选",
        "legacy_historical_sleeve": "历史组件",
        "fixed_research_default": "研究默认模板",
        "registered_untested": "已注册，待测试",
        "blocked_validation": "验证未通过",
    },
}


@dataclass(frozen=True)
class SleeveLibrarySnapshot:
    definitions: pd.DataFrame
    evidence: pd.DataFrame
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def trial_history(self) -> pd.DataFrame:
        return self.trials


TRIAL_COLUMN_ALIASES = {
    "trial_id": ("run_id",),
    "research_split": ("split",),
    "status": ("trial_status", "research_status"),
    "full_net_sharpe": ("net_sharpe", "full_sharpe", "net_sharpe_full"),
    "q1_net_sharpe": ("q1_sharpe", "net_sharpe_q1"),
    "q2_net_sharpe": ("q2_sharpe", "net_sharpe_q2"),
    "q3_net_sharpe": ("q3_sharpe", "net_sharpe_q3"),
    "q4_net_sharpe": ("q4_sharpe", "net_sharpe_q4"),
    "min_quartile_net_sharpe": (
        "minimum_quartile_sharpe",
        "min_quartile_sharpe",
        "min_q_sharpe",
    ),
    "net_annual_return": (
        "annual_return",
        "annualized_return",
        "net_annualized_return",
        "net_annualized_mean",
    ),
    "max_drawdown": ("net_max_drawdown",),
    "annualized_cost": ("annual_cost", "costs", "net_annualized_cost"),
    "average_turnover": (
        "turnover",
        "mean_turnover",
        "avg_turnover",
        "mean_daily_turnover",
        "average_daily_turnover",
    ),
    "artifact_path": ("artifact_dir",),
}


def _safe_literal(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"unresolved name: {node.id}")
        return names[node.id]
    if isinstance(node, ast.List):
        values: list[Any] = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                values.extend(_safe_literal(item.value, names))
            else:
                values.append(_safe_literal(item, names))
        return values
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal(item, names) for item in node.elts)
    if isinstance(node, ast.Subscript):
        collection = _safe_literal(node.value, names)
        key = _safe_literal(node.slice, names)
        return collection[key]
    if isinstance(node, ast.Dict):
        output: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                unpacked = _safe_literal(value, names)
                if not isinstance(unpacked, dict):
                    raise ValueError("dictionary unpack is not a mapping")
                output.update(unpacked)
            else:
                output[_safe_literal(key, names)] = _safe_literal(value, names)
        return output
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.USub, ast.UAdd),
    ):
        value = _safe_literal(node.operand, names)
        if not isinstance(value, (int, float)):
            raise ValueError("unary operand is not numeric")
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(f"unsupported metadata expression: {type(node).__name__}")


def _module_metadata(
    path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            source_id = node.module.rsplit(".", maxsplit=1)[-1]
            for alias in node.names:
                if alias.name == "FACTOR_ID":
                    names[alias.asname or alias.name] = source_id
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        try:
            names[target.id] = _safe_literal(value, names)
        except (KeyError, TypeError, ValueError):
            continue
    sleeve_id = str(names.get("SLEEVE_ID", "")).strip()
    metadata = names.get("SLEEVE_METADATA", {})
    contract = names.get("SLEEVE_CONTRACT", {})
    parameters = names.get("SLEEVE_PARAMETERS", {})
    if not sleeve_id or not isinstance(metadata, dict) or not isinstance(contract, dict):
        raise ValueError(f"{path.name} does not expose the sleeve metadata contract")
    if not isinstance(parameters, dict):
        parameters = {}
    return sleeve_id, metadata, contract, parameters


def _source_factor_ids(
    metadata: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[str, ...]:
    values = metadata.get("source_factor_ids")
    if not values:
        values = contract.get("source_factor_ids")
    if not values:
        values = [metadata.get("source_factor_id")]
    if not any(values):
        values = (
            metadata.get("source_factor_id_range")
            or contract.get("source_factor_id_range")
            or []
        )
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        values = []
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value or "").strip()
        )
    )


def _source_factor_lineage(
    metadata: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[tuple[str, ...], str, str, tuple[str, ...]]:
    source_ids = _source_factor_ids(metadata, contract)
    collection = str(
        metadata.get("source_factor_collection")
        or contract.get("source_factor_collection")
        or ""
    ).strip()
    source_range = metadata.get("source_factor_id_range") or contract.get(
        "source_factor_id_range"
    )
    range_ids = _source_factor_ids(
        {"source_factor_ids": source_range or []},
        {},
    )
    if len(range_ids) == 2:
        range_label = f"{range_ids[0]} → {range_ids[1]}"
        label = f"{collection}: {range_label}" if collection else range_label
    elif source_ids:
        label = ", ".join(source_ids)
    else:
        label = collection
    return source_ids, label, collection, range_ids


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _canonicalize_trial_ledger(trials: pd.DataFrame) -> pd.DataFrame:
    out = trials.copy()
    for canonical, aliases in TRIAL_COLUMN_ALIASES.items():
        if canonical in out:
            continue
        source = next((alias for alias in aliases if alias in out), None)
        if source is not None:
            out[canonical] = out[source]
    return out


def _load_optional_trial_ledger(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return _canonicalize_trial_ledger(pd.read_csv(path))
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _selection_rule(contract: dict[str, Any]) -> str:
    construction = str(contract.get("construction", ""))
    if construction == "top_bottom_quantile":
        fraction = float(contract.get("long_fraction", 0.0))
        return f"outer_{fraction * 100:g}_percent"
    if construction == "continuous_zscore":
        return "continuous_zscore"
    return construction


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _evidence_context(row: pd.Series) -> tuple[str, str]:
    market = _clean_text(row.get("market_vertical"))
    frequency = _clean_text(row.get("data_frequency"))
    artifact_text = _clean_text(row.get("artifact_path"))
    if not artifact_text:
        return market, frequency

    artifact_path = Path(artifact_text)
    manifest: dict[str, Any] = {}
    manifest_path = artifact_path / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}

    if not market:
        market = _clean_text(manifest.get("market_vertical"))
    if not market and len(artifact_path.parts) >= 2:
        market = artifact_path.parts[-2]

    if not frequency:
        frequency = _clean_text(
            manifest.get("data_frequency") or manifest.get("frequency")
        )
    profile_id = _clean_text(
        manifest.get("criterion", {}).get("profile_id")
        if isinstance(manifest.get("criterion"), dict)
        else ""
    )
    if not frequency and "_daily_" in f"_{profile_id}_":
        frequency = "daily"
    if not frequency and (artifact_path / "daily_diagnostics.parquet").is_file():
        frequency = "daily"
    return market, frequency


@st.cache_data(show_spinner=False)
def load_sleeve_library_snapshot(base_dir: str = str(BASE_DIR)) -> SleeveLibrarySnapshot:
    root = Path(base_dir)
    sleeve_root = root / "departments" / "research" / "strategies" / "sleeves"
    trial_ledger_path = (
        root
        / "runtime"
        / "artifacts"
        / "research"
        / "factor_sleeve_trials"
        / "trial_ledger.csv"
    )
    evidence_path = (
        root
        / "runtime"
        / "artifacts"
        / "research"
        / "router_hypotheses"
        / "sleeves.csv"
    )
    evidence = pd.read_csv(evidence_path) if evidence_path.exists() else pd.DataFrame()
    trials = _load_optional_trial_ledger(trial_ledger_path)
    if not evidence.empty and "router_eligible" in evidence:
        evidence["router_eligible"] = _as_bool(evidence["router_eligible"])
    if not evidence.empty:
        contexts = evidence.apply(_evidence_context, axis=1)
        evidence["market_vertical"] = [context[0] for context in contexts]
        evidence["data_frequency"] = [context[1] for context in contexts]

    rows: list[dict[str, Any]] = []
    legacy_aliases: dict[str, str] = {}
    for path in sorted(sleeve_root.glob("slv_*.py")):
        sleeve_id, metadata, contract, parameters = _module_metadata(path)
        (
            source_factor_ids,
            source_factor_label,
            source_factor_collection,
            source_factor_id_range,
        ) = _source_factor_lineage(metadata, contract)
        implementation_fingerprint = sleeve_implementation_fingerprint(path)
        definition_fingerprint = sleeve_definition_fingerprint(
            path,
            SimpleNamespace(
                SLEEVE_ID=sleeve_id,
                SLEEVE_METADATA=metadata,
                SLEEVE_CONTRACT=contract,
                SLEEVE_PARAMETERS=parameters,
            ),
        )
        legacy_ids = [str(value) for value in metadata.get("legacy_ids", [])]
        legacy_aliases.update({legacy_id: sleeve_id for legacy_id in legacy_ids})
        related_ids = {sleeve_id, *legacy_ids}
        related = (
            evidence.loc[evidence["sleeve_id"].astype(str).isin(related_ids)]
            if not evidence.empty and "sleeve_id" in evidence
            else pd.DataFrame()
        )
        name = str(metadata.get("name") or sleeve_id.split("_", maxsplit=2)[-1].replace("_", " "))
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "name": name,
                "status": str(metadata.get("status", "unspecified")),
                "native_market": str(metadata.get("native_market", "")),
                "data_frequency": str(metadata.get("data_frequency", "")),
                "source_factor_id": (
                    source_factor_ids[0] if source_factor_ids else ""
                ),
                "source_factor_ids": source_factor_ids,
                "source_factor_label": source_factor_label,
                "source_factor_collection": source_factor_collection,
                "source_factor_id_range": source_factor_id_range,
                "market_scope": str(metadata.get("market_scope", "")),
                "frequency_scope": str(metadata.get("frequency_scope", "")),
                "factor_scope": str(metadata.get("factor_scope", "")),
                "legacy_ids": legacy_ids,
                "economic_role": str(
                    metadata.get("economic_role")
                    or metadata.get("economic_rationale")
                    or ""
                ),
                "known_limitations": str(metadata.get("known_limitations") or ""),
                "construction": str(contract.get("construction", "")),
                "selection_rule": _selection_rule(contract),
                "normalization": str(contract.get("normalization", "")),
                "construction_geometry": str(contract.get("construction_geometry", "")),
                "expression": str(contract.get("expression", "")),
                "decision_lag": str(contract.get("decision_lag", "")),
                "holding_rule": str(contract.get("holding_rule", "")),
                "output_col": str(contract.get("output_col", "")),
                "supported_markets": ", ".join(
                    str(value) for value in contract.get("supported_markets", [])
                ),
                "tested_configurations": len(related),
                "eligible_configurations": (
                    int(related["router_eligible"].sum())
                    if not related.empty and "router_eligible" in related
                    else 0
                ),
                "contract": contract,
                "parameters": parameters,
                "definition_fingerprint": definition_fingerprint,
                "implementation_fingerprint": implementation_fingerprint,
                "sleeve_definition_fingerprint": definition_fingerprint,
                "sleeve_implementation_fingerprint": (
                    implementation_fingerprint
                ),
                "source_path": str(path.relative_to(root)),
            }
        )
    if not evidence.empty and "sleeve_id" in evidence:
        evidence["artifact_sleeve_id"] = evidence["sleeve_id"].astype(str)
        evidence["sleeve_id"] = evidence["artifact_sleeve_id"].replace(
            legacy_aliases
        )
    return SleeveLibrarySnapshot(
        definitions=pd.DataFrame(rows),
        evidence=evidence,
        trials=trials,
    )


class SleeveLibraryView:
    def __init__(self, base_dir: str | Path = BASE_DIR):
        self.base_dir = Path(base_dir)

    def render(self, lang: str = "EN") -> None:
        copy = COPY.get(lang, COPY["EN"])
        snapshot = load_sleeve_library_snapshot(str(self.base_dir))
        st.markdown(f"### {copy['title']}")
        st.caption(copy["subtitle"])
        if snapshot.definitions.empty:
            st.info(copy["no_inventory"])
            return

        self._render_summary(snapshot, copy)
        inventory_tab, drilldown_tab, evidence_tab, trial_history_tab = st.tabs(
            [
                copy["inventory_tab"],
                copy["drilldown_tab"],
                copy["evidence_tab"],
                copy["trial_history_tab"],
            ]
        )
        with inventory_tab:
            self._render_inventory(snapshot.definitions, copy, lang)
        with drilldown_tab:
            self._render_drilldown(snapshot, copy, lang)
        with evidence_tab:
            self._render_evidence(snapshot.evidence, copy, lang)
        with trial_history_tab:
            self._render_trial_history(snapshot.trials, copy, lang)

    @staticmethod
    def _render_summary(snapshot: SleeveLibrarySnapshot, copy: dict[str, str]) -> None:
        definitions = snapshot.definitions
        evidence = snapshot.evidence
        eligible = (
            int(evidence["router_eligible"].sum())
            if not evidence.empty and "router_eligible" in evidence
            else 0
        )
        cards = st.columns(5)
        cards[0].metric(copy["registered"], f"{len(definitions):,}")
        cards[1].metric(copy["tested"], f"{len(evidence):,}")
        cards[2].metric(copy["eligible"], f"{eligible:,}")
        cards[3].metric(
            copy["selection_rules"], f"{definitions['selection_rule'].nunique():,}"
        )
        cards[4].metric(
            copy["weighting_methods"], f"{definitions['normalization'].nunique():,}"
        )

    @staticmethod
    def _status_label(value: Any, lang: str) -> str:
        raw = str(value)
        return STATUS_LABELS.get(lang, STATUS_LABELS["EN"]).get(
            raw, raw.replace("_", " ").title()
        )

    @staticmethod
    def _selection_label(value: Any, copy: dict[str, str]) -> str:
        raw = str(value)
        if raw.startswith("outer_") and raw.endswith("_percent"):
            percent = raw.removeprefix("outer_").removesuffix("_percent")
            return copy["outer_selection"].format(percent=percent)
        if raw == "continuous_zscore":
            return copy["zscore_selection"]
        return raw.replace("_", " ").title()

    @staticmethod
    def _weighting_label(value: Any, lang: str) -> str:
        labels = {
            "EN": {
                "equal_weight": "Equal weight",
                "rank_weight": "Rank magnitude",
                "zscore_weight": "Z-score magnitude",
            },
            "ZH": {
                "equal_weight": "等权",
                "rank_weight": "排名幅度",
                "zscore_weight": "Z 分数幅度",
            },
        }
        raw = str(value)
        return labels.get(lang, labels["EN"]).get(raw, raw.replace("_", " ").title())

    @classmethod
    def _render_inventory(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        st.markdown(f"#### {copy['inventory']}")
        statuses = sorted(definitions["status"].dropna().unique())
        status = st.selectbox(
            copy["status"],
            [copy["all"], *statuses],
            format_func=lambda value: (
                value if value == copy["all"] else cls._status_label(value, lang)
            ),
            key="sleeve_library_status",
        )
        filtered = definitions.copy()
        if status != copy["all"]:
            filtered = filtered.loc[filtered["status"].eq(status)]
        display = filtered[
            [
                "sleeve_id",
                "name",
                "status",
                "source_factor_label",
                "selection_rule",
                "normalization",
                "tested_configurations",
                "eligible_configurations",
            ]
        ].copy()
        display["status"] = display["status"].map(lambda value: cls._status_label(value, lang))
        display["selection_rule"] = display["selection_rule"].map(
            lambda value: cls._selection_label(value, copy)
        )
        display["normalization"] = display["normalization"].map(
            lambda value: cls._weighting_label(value, lang)
        )
        display = display.rename(
            columns={
                "sleeve_id": copy["sleeve_id"],
                "name": copy["name"],
                "source_factor_label": copy["source_factors"],
                "construction": copy["construction"],
                "selection_rule": copy["selection"],
                "normalization": copy["weighting"],
                "expression": copy["expression"],
                "decision_lag": copy["decision_lag"],
                "tested_configurations": copy["tested_count"],
                "eligible_configurations": copy["eligible_count"],
                "status": copy["status"],
            }
        )
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            height=300,
        )

    @classmethod
    def _render_drilldown(
        cls,
        snapshot: SleeveLibrarySnapshot,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        definitions = snapshot.definitions
        labels = {
            row.sleeve_id: f"{row.sleeve_id} | {row.name}"
            for row in definitions.itertuples(index=False)
        }
        sleeve_id = st.selectbox(
            copy["select"],
            list(labels),
            format_func=labels.get,
            key="sleeve_library_drilldown",
        )
        row = definitions.loc[definitions["sleeve_id"].eq(sleeve_id)].iloc[0]
        cards = st.columns(4)
        facts = (
            (copy["status"], cls._status_label(row["status"], lang)),
            (copy["selection"], cls._selection_label(row["selection_rule"], copy)),
            (copy["weighting"], cls._weighting_label(row["normalization"], lang)),
            (copy["tested_count"], f"{int(row['tested_configurations']):,}"),
        )
        for column, (label, value) in zip(cards, facts, strict=True):
            column.caption(label)
            column.markdown(f"**{value}**")
        st.markdown(f"#### {copy['economic_role']}")
        if row["economic_role"]:
            st.write(row["economic_role"])
        else:
            st.warning(copy["not_documented"])
        if row["known_limitations"]:
            st.markdown(f"#### {copy['limitations']}")
            st.write(row["known_limitations"])
        factor_scope = (
            copy["any_compatible_factor"]
            if row["factor_scope"] == "any_compatible_factor"
            else row["factor_scope"] or "N/A"
        )
        st.caption(f"{copy['factor_scope']}: {factor_scope}")
        source_factors = row["source_factor_label"] or "N/A"
        st.caption(f"{copy['source_factors']}: {source_factors}")
        st.markdown(f"#### {copy['contract']}")
        contract = pd.DataFrame(
            [
                {copy["field"]: key, copy["value"]: str(value)}
                for key, value in row["contract"].items()
            ]
        )
        st.dataframe(contract, width="stretch", hide_index=True)
        st.markdown(f"#### {copy['fingerprints']}")
        st.caption(
            f"{copy['definition_fingerprint']}: "
            f"{row['definition_fingerprint']}"
        )
        st.caption(
            f"{copy['implementation_fingerprint']}: "
            f"{row['implementation_fingerprint']}"
        )
        st.caption(f"{copy['source']}: {row['source_path']}")

        related = (
            snapshot.evidence.loc[snapshot.evidence["sleeve_id"].astype(str).eq(sleeve_id)]
            if not snapshot.evidence.empty
            else pd.DataFrame()
        )
        st.markdown(f"#### {copy['evidence']}")
        if related.empty:
            st.info(copy["evidence_empty"])
        else:
            cls._render_evidence_table(related, copy, lang)

    @classmethod
    def _render_evidence(
        cls,
        evidence: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        st.markdown(f"#### {copy['evidence']}")
        if evidence.empty:
            st.info(copy["evidence_empty"])
            return
        controls = st.columns(2)
        sleeves = sorted(evidence["sleeve_id"].astype(str).unique())
        selected_sleeve = controls[0].selectbox(
            copy["sleeve_id"],
            [copy["all"], *sleeves],
            key="sleeve_evidence_sleeve",
        )
        eligibility = controls[1].selectbox(
            copy["router_eligible"],
            [copy["all"], copy["yes"], copy["no"]],
            key="sleeve_evidence_eligibility",
        )
        filtered = evidence.copy()
        if selected_sleeve != copy["all"]:
            filtered = filtered.loc[filtered["sleeve_id"].astype(str).eq(selected_sleeve)]
        if eligibility != copy["all"]:
            filtered = filtered.loc[
                filtered["router_eligible"].eq(eligibility == copy["yes"])
            ]
        cls._render_evidence_table(filtered, copy, lang)

    @classmethod
    def _render_evidence_table(
        cls,
        evidence: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        columns = [
            "factor_id",
            "sleeve_id",
            "market_vertical",
            "data_frequency",
            "standalone_status",
            "router_eligible",
            "validation_net_sharpe",
            "holdout_net_sharpe",
        ]
        display = evidence[[column for column in columns if column in evidence]].copy()
        if "standalone_status" in display:
            display["standalone_status"] = display["standalone_status"].map(
                lambda value: cls._status_label(value, lang)
            )
        if "router_eligible" in display:
            display["router_eligible"] = display["router_eligible"].map(
                {True: copy["yes"], False: copy["no"]}
            )
        display = display.rename(
            columns={
                "factor_id": copy["factor_id"],
                "sleeve_id": copy["sleeve_id"],
                "market_vertical": copy["market"],
                "data_frequency": copy["frequency"],
                "standalone_status": copy["standalone_status"],
                "router_eligible": copy["router_eligible"],
                "validation_net_sharpe": copy["validation_sharpe"],
                "holdout_net_sharpe": copy["holdout_sharpe"],
            }
        )
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                copy["validation_sharpe"]: st.column_config.NumberColumn(format="%.2f"),
                copy["holdout_sharpe"]: st.column_config.NumberColumn(format="%.2f"),
            },
        )

    @classmethod
    def _render_trial_history(
        cls,
        trials: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        st.markdown(f"#### {copy['trial_history']}")
        if trials.empty:
            st.info(copy["trial_history_empty"])
            return

        columns = [
            "trial_id",
            "factor_id",
            "sleeve_id",
            "research_split",
            "status",
            "full_net_sharpe",
            "q1_net_sharpe",
            "q2_net_sharpe",
            "q3_net_sharpe",
            "q4_net_sharpe",
            "min_quartile_net_sharpe",
            "net_annual_return",
            "max_drawdown",
            "annualized_cost",
            "average_turnover",
            "artifact_path",
        ]
        display = trials[[column for column in columns if column in trials]].copy()
        metric_columns = [
            "full_net_sharpe",
            "q1_net_sharpe",
            "q2_net_sharpe",
            "q3_net_sharpe",
            "q4_net_sharpe",
            "min_quartile_net_sharpe",
            "net_annual_return",
            "max_drawdown",
            "annualized_cost",
            "average_turnover",
        ]
        for column in metric_columns:
            if column in display:
                display[column] = pd.to_numeric(display[column], errors="coerce")
        if "status" in display:
            display["status"] = display["status"].map(
                lambda value: cls._status_label(value, lang)
            )
        if "min_quartile_net_sharpe" in display:
            display = display.sort_values(
                "min_quartile_net_sharpe",
                ascending=False,
                na_position="last",
                kind="stable",
            )
        for column in (
            "net_annual_return",
            "max_drawdown",
            "annualized_cost",
            "average_turnover",
        ):
            if column in display:
                display[column] *= 100.0
        display = display.rename(
            columns={
                "trial_id": copy["trial_id"],
                "factor_id": copy["factor_id"],
                "sleeve_id": copy["sleeve_id"],
                "research_split": copy["research_split"],
                "status": copy["trial_status"],
                "full_net_sharpe": copy["full_net_sharpe"],
                "q1_net_sharpe": copy["q1_net_sharpe"],
                "q2_net_sharpe": copy["q2_net_sharpe"],
                "q3_net_sharpe": copy["q3_net_sharpe"],
                "q4_net_sharpe": copy["q4_net_sharpe"],
                "min_quartile_net_sharpe": copy["min_quartile_net_sharpe"],
                "net_annual_return": copy["net_annual_return"],
                "max_drawdown": copy["max_drawdown"],
                "annualized_cost": copy["annualized_cost"],
                "average_turnover": copy["average_turnover"],
                "artifact_path": copy["artifact_path"],
            }
        )
        sharpe_columns = [
            copy["full_net_sharpe"],
            copy["q1_net_sharpe"],
            copy["q2_net_sharpe"],
            copy["q3_net_sharpe"],
            copy["q4_net_sharpe"],
            copy["min_quartile_net_sharpe"],
        ]
        percentage_columns = [
            copy["net_annual_return"],
            copy["max_drawdown"],
            copy["annualized_cost"],
            copy["average_turnover"],
        ]
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                **{
                    column: st.column_config.NumberColumn(format="%.2f")
                    for column in sharpe_columns
                    if column in display
                },
                **{
                    column: st.column_config.NumberColumn(format="%.2f%%")
                    for column in percentage_columns
                    if column in display
                },
                **(
                    {
                        copy["artifact_path"]: st.column_config.TextColumn(
                            width="large"
                        )
                    }
                    if copy["artifact_path"] in display
                    else {}
                ),
            },
        )


__all__ = [
    "SleeveLibrarySnapshot",
    "SleeveLibraryView",
    "load_sleeve_library_snapshot",
]
