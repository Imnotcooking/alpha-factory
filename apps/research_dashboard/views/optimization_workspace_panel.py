from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from oqp.optimization import OptimizationMethodRegistry
from oqp.research.optimization_objectives import OptimizationObjectiveRegistry
from oqp.research.optional_optimization import (
    DEFAULT_STUDY_CONFIG_ROOT,
    Phase8FoldConfig,
    audit_phase8_readiness,
    build_component_study_definition,
    build_factor_study_definition,
    default_study_id,
    load_component_options,
    load_dataset_options,
    load_frozen_component_options,
    parameter_schema_rows,
    resolve_selected_component_schema,
    study_definition_payload,
    write_frozen_study_definition,
    write_phase8_readiness,
)


COPY = {
    "EN": {
        "title": "Optimisation",
        "subtitle": (
            "Design and review controlled optimisation studies. Choose the job "
            "first; the workspace then couples it to a suitable method, objective "
            "profile and validation boundary."
        ),
        "boundary": (
            "One study may change one component layer only. Dataset, universe, "
            "timing, costs, upstream components and the final holdout remain frozen."
        ),
        "purpose_selector": "What are you adjusting?",
        "purpose": "Research purpose",
        "layer": "Mutable layer",
        "job": "Problem type",
        "primary": "Registry default method",
        "benchmark": "Required baseline",
        "status": "Governance",
        "why": "Why this method fits",
        "method_fit": "Method fit",
        "role": "Role",
        "method": "Method",
        "family": "Search family",
        "engine": "Implementation",
        "use_when": "Use when",
        "warning": "Main warning",
        "recommended": "Primary",
        "baseline": "Baseline",
        "alternative": "Alternative",
        "objective": "Objective and hard gates",
        "objective_profile": "Objective profile",
        "economic_question": "Economic question",
        "objectives": "Objectives",
        "constraints": "Hard gates",
        "upstream": "Required frozen components",
        "no_profile": (
            "This job does not yet use a common research objective profile. "
            "Follow its specialised protocol or define and review a profile before "
            "a generic study can run."
        ),
        "blocked": "This optimisation purpose is currently blocked",
        "workflow": "Frozen study sequence",
        "workflow_value": (
            "Freeze inputs -> select one mutable layer -> run equal-budget baseline "
            "and primary search -> validate chronologically -> freeze one candidate "
            "-> evaluate the final holdout once"
        ),
        "ledger": "Study ledger",
        "declared": "Declared",
        "enabled": "Enabled",
        "completed": "Completed",
        "frozen": "Candidates frozen",
        "holdout": "Holdout evaluations",
        "no_studies": (
            "No optimisation study has been declared yet. The workspace currently "
            "defines how a valid study must be designed; it does not invent a search "
            "automatically."
        ),
        "all_jobs": "All supported optimisation jobs",
        "not_applicable": "Not applicable",
        "profile_missing": "Objective-readiness artifacts have not been generated.",
        "definition": "Frozen study definition",
        "definition_help": (
            "This creates an immutable, disabled YAML definition. It does not "
            "launch an optimisation search."
        ),
        "component": "Mutable component",
        "component_status": "Component status",
        "source": "Source",
        "schema": "Declared parameter space",
        "schema_fingerprint": "Schema fingerprint",
        "tunable_parameters": "Tunable parameters",
        "no_components": "No registered components exist for this optimisation job.",
        "no_schema": (
            "This component has no valid declarative tunable-parameter schema, "
            "so a study definition cannot be generated."
        ),
        "upstream_candidates": "Frozen upstream candidates",
        "no_upstream": (
            "No eligible frozen upstream candidate exists yet. Complete and freeze "
            "the required upstream study before defining this optimisation."
        ),
        "select_upstream": (
            "Select enough distinct frozen upstream components to satisfy every "
            "objective-profile requirement."
        ),
        "incompatible_upstream": (
            "The selected upstream candidates do not share one development dataset "
            "and one untouched final holdout."
        ),
        "duplicate_upstream": (
            "Select only one frozen candidate version for each upstream component."
        ),
        "dataset": "Registered development dataset",
        "no_datasets": "No compatible fingerprinted dataset is registered.",
        "dataset_warning": (
            "This dataset has fewer than 10 instruments and is unlikely to support "
            "a reliable cross-sectional study."
        ),
        "study_id": "Study ID",
        "capital": "Research capital",
        "max_weight": "Maximum contract weight",
        "trials": "Trial budget",
        "seed": "Random seed",
        "freeze_date": "Protocol freeze date",
        "holdout_start": "Forward holdout starts",
        "folds": "Chronological validation",
        "fold_count": "Inner folds",
        "training_periods": "Minimum training periods",
        "validation_periods": "Validation periods",
        "purge_periods": "Purge periods",
        "embargo_periods": "Embargo periods",
        "generate": "Save frozen study definition",
        "generated": "Frozen study definition saved",
        "generated_disabled": (
            "The definition is saved with enabled=false. A separate review must "
            "approve its evaluator and upstream evidence before execution."
        ),
        "preview": "Generated YAML",
        "definition_error": "Study definition could not be generated",
    },
    "ZH": {
        "title": "优化",
        "subtitle": (
            "用于设计和审查受控优化研究。先选择任务，再匹配合适的优化方法、"
            "目标配置和验证边界。"
        ),
        "boundary": (
            "每项研究只能修改一个组件层。数据集、品种池、时序、成本、上游组件"
            "和最终留出集必须保持冻结。"
        ),
        "purpose_selector": "本次需要调整什么？",
        "purpose": "研究目的",
        "layer": "可变组件层",
        "job": "问题类型",
        "primary": "方法注册表默认值",
        "benchmark": "必要基准",
        "status": "治理状态",
        "why": "为什么适合",
        "method_fit": "方法匹配",
        "role": "角色",
        "method": "方法",
        "family": "搜索类型",
        "engine": "实现",
        "use_when": "适用情况",
        "warning": "主要风险",
        "recommended": "主要方法",
        "baseline": "基准",
        "alternative": "备选",
        "objective": "目标与硬性门槛",
        "objective_profile": "目标配置",
        "economic_question": "经济问题",
        "objectives": "优化目标",
        "constraints": "硬性门槛",
        "upstream": "必须冻结的上游组件",
        "no_profile": (
            "该任务尚未使用统一研究目标配置。必须遵循其专用流程，或先定义并审查"
            "目标配置，之后才能运行通用优化研究。"
        ),
        "blocked": "该优化任务当前被阻止",
        "workflow": "冻结研究流程",
        "workflow_value": (
            "冻结输入 -> 选择一个可变层 -> 使用相同预算运行基准与主要方法 -> "
            "按时间验证 -> 冻结一个候选 -> 最终留出集只评估一次"
        ),
        "ledger": "研究台账",
        "declared": "已声明",
        "enabled": "已启用",
        "completed": "已完成",
        "frozen": "已冻结候选",
        "holdout": "留出集评估",
        "no_studies": "尚未声明优化研究。本页面目前定义合法研究的设计方式，不会自动发明搜索。",
        "all_jobs": "全部优化任务",
        "not_applicable": "不适用",
        "profile_missing": "尚未生成优化目标审计文件。",
        "definition": "冻结研究定义",
        "definition_help": "该步骤生成不可变且默认关闭的 YAML，不会启动优化搜索。",
        "component": "可变组件",
        "component_status": "组件状态",
        "source": "源码",
        "schema": "已声明参数空间",
        "schema_fingerprint": "参数结构指纹",
        "tunable_parameters": "可调参数",
        "no_components": "该优化任务没有已注册组件。",
        "no_schema": "该组件没有有效的声明式可调参数结构，因此不能生成研究定义。",
        "upstream_candidates": "已冻结上游候选",
        "no_upstream": "尚无合资格的冻结上游候选。必须先完成并冻结所需的上游研究。",
        "select_upstream": "请选择足够且互不重复的冻结上游组件，以满足目标配置的全部要求。",
        "incompatible_upstream": "所选上游候选没有共享同一开发数据集和同一未触碰最终留出集。",
        "duplicate_upstream": "每个上游组件只能选择一个冻结候选版本。",
        "dataset": "已注册开发数据集",
        "no_datasets": "没有兼容且已生成指纹的数据集。",
        "dataset_warning": "该数据集少于 10 个品种，不适合可靠的横截面研究。",
        "study_id": "研究 ID",
        "capital": "研究资金",
        "max_weight": "单品种最大权重",
        "trials": "试验预算",
        "seed": "随机种子",
        "freeze_date": "协议冻结日期",
        "holdout_start": "前瞻留出集起始日",
        "folds": "时间顺序验证",
        "fold_count": "内部折数",
        "training_periods": "最少训练期数",
        "validation_periods": "验证期数",
        "purge_periods": "清除期数",
        "embargo_periods": "隔离期数",
        "generate": "保存冻结研究定义",
        "generated": "已保存冻结研究定义",
        "generated_disabled": "定义以 enabled=false 保存。执行前仍需单独审查评估器和上游证据。",
        "preview": "生成的 YAML",
        "definition_error": "无法生成研究定义",
    },
}


STATUS_LABELS = {
    "EN": {
        "governed": "Governed",
        "specialized": "Specialised",
        "experimental": "Experimental",
        "blocked": "Blocked",
    },
    "ZH": {
        "governed": "已定义治理",
        "specialized": "专用流程",
        "experimental": "实验性",
        "blocked": "已阻止",
    },
}


@st.cache_data(show_spinner=False)
def load_optimization_workspace(artifact_root: str) -> dict[str, Any]:
    root = Path(artifact_root).expanduser().resolve()
    phase8_root = root / "optional_optimization"
    objectives_root = root / "optimization_objectives"
    result: dict[str, Any] = {}
    readiness_path = phase8_root / "readiness.json"
    studies_path = phase8_root / "studies.csv"
    if readiness_path.exists():
        result["readiness"] = json.loads(readiness_path.read_text(encoding="utf-8"))
    if studies_path.exists():
        result["studies"] = pd.read_csv(studies_path)

    objective_paths = {
        "objective_readiness": objectives_root / "readiness.json",
        "profiles": objectives_root / "profiles.csv",
        "objectives": objectives_root / "objectives.csv",
        "constraints": objectives_root / "constraints.csv",
        "upstream": objectives_root / "upstream_requirements.csv",
    }
    for key, path in objective_paths.items():
        if not path.exists():
            continue
        result[key] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.suffix == ".json"
            else pd.read_csv(path)
        )
    return result


def _method_rows(
    registry: OptimizationMethodRegistry,
    purpose,
    copy: dict[str, str],
) -> pd.DataFrame:
    roles = [(copy["recommended"], purpose.primary_method)]
    if purpose.benchmark_method:
        roles.append((copy["baseline"], purpose.benchmark_method))
    roles.extend(
        (copy["alternative"], method_id)
        for method_id in purpose.alternative_methods
    )
    rows = []
    for role, method_id in roles:
        method = registry.resolve_method(method_id)
        rows.append(
            {
                copy["role"]: role,
                copy["method"]: method.label,
                copy["family"]: method.family,
                copy["engine"]: method.engine,
                copy["use_when"]: method.good_for,
                copy["warning"]: method.warning,
            }
        )
    return pd.DataFrame(rows)


def _render_profile(
    snapshot: dict[str, Any],
    profile_id: str | None,
    copy: dict[str, str],
) -> None:
    st.markdown(f"#### {copy['objective']}")
    if not profile_id:
        st.info(copy["no_profile"])
        return

    profiles = snapshot.get("profiles")
    objectives = snapshot.get("objectives")
    constraints = snapshot.get("constraints")
    upstream = snapshot.get("upstream")
    if not all(isinstance(frame, pd.DataFrame) for frame in (profiles, objectives, constraints, upstream)):
        st.warning(copy["profile_missing"])
        return

    profile_rows = profiles.loc[profiles["profile_id"].eq(profile_id)]
    if profile_rows.empty:
        st.warning(copy["profile_missing"])
        return
    profile = profile_rows.iloc[0]
    cols = st.columns([0.33, 0.67])
    cols[0].caption(copy["objective_profile"])
    cols[0].markdown(f"**{profile_id}**")
    cols[1].caption(copy["economic_question"])
    cols[1].markdown(str(profile["economic_question"]))

    objective_rows = objectives.loc[
        objectives["profile_id"].eq(profile_id),
        ["priority", "objective", "metric", "direction"],
    ].sort_values("priority")
    st.caption(copy["objectives"])
    st.dataframe(objective_rows, use_container_width=True, hide_index=True)

    detail_tabs = st.tabs([copy["constraints"], copy["upstream"]])
    with detail_tabs[0]:
        gate_rows = constraints.loc[
            constraints["profile_id"].eq(profile_id)
        ].drop(columns=["profile_id", "layer"], errors="ignore")
        if gate_rows.empty:
            st.caption(copy["not_applicable"])
        else:
            st.dataframe(gate_rows, use_container_width=True, hide_index=True)
    with detail_tabs[1]:
        upstream_rows = upstream.loc[
            upstream["profile_id"].eq(profile_id)
        ].drop(columns=["profile_id", "layer"], errors="ignore")
        if upstream_rows.empty:
            st.caption(copy["not_applicable"])
        else:
            st.dataframe(upstream_rows, use_container_width=True, hide_index=True)


def _render_ledger(snapshot: dict[str, Any], copy: dict[str, str]) -> None:
    st.markdown(f"#### {copy['ledger']}")
    readiness = snapshot.get("readiness", {})
    cards = st.columns(5)
    values = [
        (copy["declared"], readiness.get("declared_studies", 0)),
        (copy["enabled"], readiness.get("enabled_studies", 0)),
        (copy["completed"], readiness.get("completed_searches", 0)),
        (copy["frozen"], readiness.get("frozen_candidates", 0)),
        (copy["holdout"], readiness.get("final_holdout_evaluations", 0)),
    ]
    for card, (label, value) in zip(cards, values, strict=True):
        card.metric(label, value)

    studies = snapshot.get("studies")
    if not isinstance(studies, pd.DataFrame) or studies.empty:
        st.caption(copy["no_studies"])
    else:
        st.dataframe(studies, use_container_width=True, hide_index=True)


def _render_study_definition_builder(
    purpose_id: str,
    purpose,
    artifact_root: str | Path,
    copy: dict[str, str],
) -> dict[str, Any] | None:
    st.markdown(f"#### {copy['definition']}")

    components = load_component_options(purpose_id)
    if not components:
        st.warning(copy["no_components"])
        return None
    component_map = {item.component_id: item for item in components}
    component_id = st.selectbox(
        copy["component"],
        list(component_map),
        key=f"optimization_component_{purpose_id}",
    )
    component = component_map[component_id]

    try:
        schema = resolve_selected_component_schema(purpose_id, component_id)
    except Exception as exc:
        detail = st.columns([0.24, 0.24, 0.52])
        detail[0].caption(copy["component_status"])
        detail[0].markdown(f"**{component.research_status or 'registered'}**")
        detail[1].caption(copy["tunable_parameters"])
        detail[1].markdown("**0**")
        detail[2].caption(copy["source"])
        detail[2].code(component.source_path, language=None)
        st.warning(f"{copy['no_schema']} {exc}")
        return None
    schema_rows = pd.DataFrame(parameter_schema_rows(schema))
    tunable_count = len(schema.tunable_names)
    if tunable_count == 0:
        st.warning(copy["no_schema"])
        return None
    if purpose.status != "governed":
        message = purpose.blocking_reason or copy["no_profile"]
        st.warning(f"{copy['blocked']}: {message}")
        return None

    objective_profile = OptimizationObjectiveRegistry.load().resolve(
        purpose.objective_profile_id
    )
    frozen_components: dict[str, str] = {}
    upstream_datasets: dict[str, str] = {}
    upstream_holdouts: dict[str, str] = {}
    upstream_holdout_start: date | None = None
    requirements = tuple(objective_profile.upstream_requirements)
    if requirements:
        accepted_prefixes = tuple(
            dict.fromkeys(
                prefix
                for requirement in requirements
                for prefix in requirement.accepted_prefixes
            )
        )
        candidates = load_frozen_component_options(
            artifact_root=Path(artifact_root).expanduser().resolve()
            / "optional_optimization",
            accepted_prefixes=accepted_prefixes,
        )
        st.caption(
            f"{copy['upstream_candidates']}: "
            + "; ".join(
                f"{requirement.name} >= {requirement.minimum_count} "
                f"({', '.join(requirement.accepted_prefixes)})"
                for requirement in requirements
            )
        )
        if not candidates:
            st.warning(copy["no_upstream"])
            return None
        candidate_map = {item.source_path: item for item in candidates}
        selected_candidates = st.multiselect(
            copy["upstream_candidates"],
            list(candidate_map),
            format_func=lambda value: candidate_map[value].label,
            key=f"optimization_upstream_{purpose_id}_{component_id}",
        )
        selected = [candidate_map[value] for value in selected_candidates]
        selected_ids = [item.component_id for item in selected]
        if len(set(selected_ids)) != len(selected_ids):
            st.warning(copy["duplicate_upstream"])
            return None
        unmet = [
            requirement
            for requirement in requirements
            if sum(
                component_id.startswith(requirement.accepted_prefixes)
                for component_id in selected_ids
            )
            < requirement.minimum_count
        ]
        if unmet:
            st.warning(copy["select_upstream"])
            return None
        dataset_fingerprints = {item.dataset_fingerprint for item in selected}
        holdout_fingerprints = {item.holdout_fingerprint for item in selected}
        holdout_starts = {item.holdout_start for item in selected}
        if (
            len(dataset_fingerprints) != 1
            or len(holdout_fingerprints) != 1
            or len(holdout_starts) != 1
        ):
            st.warning(copy["incompatible_upstream"])
            return None
        frozen_components = {
            item.component_id: item.candidate_fingerprint for item in selected
        }
        upstream_datasets = {
            item.component_id: item.dataset_fingerprint for item in selected
        }
        upstream_holdouts = {
            item.component_id: item.holdout_fingerprint for item in selected
        }
        upstream_holdout_start = pd.Timestamp(
            next(iter(holdout_starts))
        ).date()

    datasets = load_dataset_options(
        market_vertical=component.market_vertical,
        data_frequency=component.data_frequency,
    )
    if upstream_datasets:
        selected_dataset_fingerprint = next(iter(upstream_datasets.values()))
        datasets = tuple(
            item
            for item in datasets
            if item.aggregate_sha256 == selected_dataset_fingerprint
        )
    if not datasets:
        st.warning(
            copy["incompatible_upstream"]
            if upstream_datasets
            else copy["no_datasets"]
        )
        return None
    dataset_map = {item.manifest_path: item for item in datasets}
    selected_manifest = st.selectbox(
        copy["dataset"],
        list(dataset_map),
        format_func=lambda value: dataset_map[value].label,
        key=f"optimization_dataset_{component_id}",
    )
    dataset = dataset_map[selected_manifest]
    if (dataset.instrument_count or 0) < 10:
        st.warning(copy["dataset_warning"])

    today = date.today()
    earliest_holdout = today + timedelta(days=1)
    if upstream_holdout_start is not None and upstream_holdout_start <= today:
        st.warning(copy["incompatible_upstream"])
        return None
    default_holdout = upstream_holdout_start or earliest_holdout
    with st.form(f"optimization_definition_form_{component_id}"):
        identity = st.columns([0.58, 0.21, 0.21])
        study_id = identity[0].text_input(
            copy["study_id"],
            value=default_study_id(component_id, today),
        )
        identity[1].date_input(
            copy["freeze_date"],
            value=today,
            disabled=True,
        )
        holdout_start = identity[2].date_input(
            copy["holdout_start"],
            value=default_holdout,
            min_value=default_holdout,
            max_value=default_holdout if upstream_holdout_start else None,
            disabled=upstream_holdout_start is not None,
        )

        controls = st.columns(4)
        capital = controls[0].number_input(
            copy["capital"],
            min_value=1_000.0,
            value=10_000_000.0,
            step=100_000.0,
        )
        max_weight = controls[1].number_input(
            copy["max_weight"],
            min_value=0.001,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.3f",
        )
        trials = controls[2].number_input(
            copy["trials"],
            min_value=5,
            max_value=500,
            value=50,
            step=5,
        )
        seed = controls[3].number_input(
            copy["seed"],
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )

        with st.expander(copy["folds"]):
            fold_controls = st.columns(5)
            fold_count = fold_controls[0].number_input(
                copy["fold_count"], min_value=2, max_value=10, value=4
            )
            training_periods = fold_controls[1].number_input(
                copy["training_periods"], min_value=20, value=252, step=21
            )
            validation_periods = fold_controls[2].number_input(
                copy["validation_periods"], min_value=5, value=63, step=5
            )
            purge_periods = fold_controls[3].number_input(
                copy["purge_periods"], min_value=0, value=1
            )
            embargo_periods = fold_controls[4].number_input(
                copy["embargo_periods"], min_value=0, value=5
            )
        submitted = st.form_submit_button(
            copy["generate"],
            use_container_width=True,
        )

    with st.expander(copy["schema"]):
        detail = st.columns([0.24, 0.24, 0.52])
        detail[0].caption(copy["component_status"])
        detail[0].markdown(f"**{component.research_status or 'registered'}**")
        detail[1].caption(copy["tunable_parameters"])
        detail[1].markdown(f"**{tunable_count}**")
        detail[2].caption(copy["source"])
        detail[2].code(component.source_path, language=None)
        st.caption(f"{copy['schema_fingerprint']}: `{schema.fingerprint}`")
        st.dataframe(schema_rows, use_container_width=True, hide_index=True)
    st.caption(copy["definition_help"])

    if not submitted:
        return None
    try:
        builder = (
            build_factor_study_definition
            if purpose.layer == "factor"
            else build_component_study_definition
        )
        builder_kwargs = {
            "study_id": study_id,
            "purpose_id": purpose_id,
            "component_id": component_id,
            "dataset_manifest_path": selected_manifest,
            "holdout_start": holdout_start.isoformat(),
            "frozen_on": today.isoformat(),
            "max_trials": int(trials),
            "seed": int(seed),
            "initial_capital": float(capital),
            "capital_currency": "CNY",
            "max_position_weight": float(max_weight),
            "fold_config": Phase8FoldConfig(
                fold_count=int(fold_count),
                minimum_training_periods=int(training_periods),
                validation_periods=int(validation_periods),
                purge_periods=int(purge_periods),
                embargo_periods=int(embargo_periods),
            ),
        }
        if purpose.layer != "factor":
            builder_kwargs.update(
                {
                    "frozen_component_fingerprints": frozen_components,
                    "frozen_component_dataset_fingerprints": upstream_datasets,
                    "frozen_component_holdout_fingerprints": upstream_holdouts,
                }
            )
        spec, _, metadata = builder(
            **builder_kwargs,
        )
        path = write_frozen_study_definition(spec, metadata)
        artifact_path = Path(artifact_root).expanduser().resolve()
        summary, studies = audit_phase8_readiness(
            DEFAULT_STUDY_CONFIG_ROOT,
            artifact_path / "optional_optimization",
        )
        write_phase8_readiness(
            summary,
            studies,
            artifact_path / "optional_optimization",
        )
        load_optimization_workspace.clear()
        st.success(f"{copy['generated']}: `{path}`")
        st.caption(copy["generated_disabled"])
        with st.expander(copy["preview"]):
            st.code(
                yaml.safe_dump(
                    study_definition_payload(spec, metadata),
                    sort_keys=False,
                    allow_unicode=False,
                ),
                language="yaml",
            )
        return load_optimization_workspace(str(artifact_root))
    except Exception as exc:
        st.error(f"{copy['definition_error']}: {exc}")
        return None


def render_optimization_workspace_panel(
    artifact_root: str | Path,
    *,
    lang: str = "EN",
) -> None:
    copy = COPY.get(lang, COPY["EN"])
    registry = OptimizationMethodRegistry.load()
    snapshot = load_optimization_workspace(str(artifact_root))

    st.markdown(f"### {copy['title']}")
    st.caption(copy["subtitle"])

    purpose_ids = list(registry.purposes)
    purpose_labels = {
        purpose_id: registry.resolve_purpose(purpose_id).label
        for purpose_id in purpose_ids
    }
    selected_id = st.selectbox(
        copy["purpose_selector"],
        purpose_ids,
        format_func=lambda value: purpose_labels[value],
    )
    purpose = registry.resolve_purpose(selected_id)
    primary = registry.resolve_method(purpose.primary_method)
    benchmark = (
        registry.resolve_method(purpose.benchmark_method)
        if purpose.benchmark_method
        else None
    )

    updated_snapshot = _render_study_definition_builder(
        selected_id,
        purpose,
        artifact_root,
        copy,
    )
    if updated_snapshot is not None:
        snapshot = updated_snapshot
    _render_ledger(snapshot, copy)

    st.divider()
    summary = st.columns([0.18, 0.25, 0.22, 0.20, 0.15])
    values = [
        (copy["layer"], purpose.layer.title()),
        (copy["job"], purpose.job_type),
        (copy["primary"], primary.label),
        (
            copy["benchmark"],
            benchmark.label if benchmark else copy["not_applicable"],
        ),
        (
            copy["status"],
            STATUS_LABELS.get(lang, STATUS_LABELS["EN"]).get(
                purpose.status, purpose.status.title()
            ),
        ),
    ]
    for column, (label, value) in zip(summary, values, strict=True):
        column.caption(label)
        column.markdown(f"**{value}**")

    st.markdown(f"#### {copy['purpose']}")
    st.write(purpose.purpose)
    st.caption(f"{copy['why']}: {purpose.rationale}")
    if purpose.blocking_reason:
        message = f"{copy['blocked']}: {purpose.blocking_reason}"
        if purpose.status == "blocked":
            st.error(message)
        else:
            st.warning(message)

    st.markdown(f"#### {copy['method_fit']}")
    st.dataframe(
        _method_rows(registry, purpose, copy),
        use_container_width=True,
        hide_index=True,
    )
    _render_profile(snapshot, purpose.objective_profile_id, copy)

    st.markdown(f"#### {copy['workflow']}")
    st.code(copy["workflow_value"], language=None)
    st.info(copy["boundary"])

    with st.expander(copy["all_jobs"]):
        inventory = pd.DataFrame(registry.purpose_inventory())
        st.dataframe(inventory, use_container_width=True, hide_index=True)


__all__ = ["load_optimization_workspace", "render_optimization_workspace_panel"]
