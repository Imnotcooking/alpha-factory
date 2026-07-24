from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pandas as pd
import streamlit as st
import yaml

from config import BASE_DIR, DB_PATH
from views.router_library_view import load_router_library_snapshot
from views.sleeve_library_view import load_sleeve_library_snapshot

from oqp.execution.transaction_costs import (
    TransactionCostProfile,
    TransactionCostRegistry,
)
from oqp.research.factor_portfolios import (
    factor_inventory,
    strategy_risk_overlay_inventory,
)
from oqp.research.ml import list_ml_experiments
from oqp.research.strategy_composition import (
    StrategyAllocatorConfig,
    StrategyBranchConfig,
    StrategyBuilderConfig,
    StrategyCoreConfig,
    StrategyCoreType,
    StrategyExecutionConfig,
    build_strategy_backtest_command,
    strategy_execution_support,
    write_strategy_builder_config,
)


COPY = {
    "EN": {
        "title": "Strategy Construction",
        "identity": "Strategy identity",
        "strategy_id": "Strategy ID",
        "name": "Strategy name",
        "market": "Market",
        "core": "Position-producing core",
        "core_type": "Core type",
        "direct": "Embedded-target factor (legacy)",
        "factor_sleeve": "Factor + sleeve",
        "factor_blend": "Multi-factor blend",
        "stat_arb": "Statistical arbitrage",
        "ml": "Fitted ML predictor",
        "routed": "Routed components",
        "factor": "Factor",
        "factors": "Factors",
        "sleeve": "Sleeve",
        "model": "Registered ML experiment",
        "model_empty": "No fitted ML experiment with stored OOS predictions is registered.",
        "router": "Router",
        "router_state": "Causal router-state file",
        "branch_a": "Branch A",
        "branch_b": "Branch B",
        "branch_source": "Branch source",
        "score_branch": "Factor score portfolio",
        "direct_branch": "Direct factor targets",
        "optional_sleeve": "Sleeve translation",
        "none": "None",
        "risk": "Risk and allocation",
        "overlays": "Risk overlays",
        "gross": "Maximum gross leverage",
        "margin_limit": "Maximum margin utilisation (%)",
        "cash_reserve": "Minimum cash reserve (%)",
        "contract_cap": "Maximum contract weight",
        "execution": "Execution",
        "capital": "Capital",
        "currency": "Currency",
        "cost_profile": "Transaction-cost profile",
        "slippage": "Slippage ticks per side",
        "data": "Market data",
        "data_file": "Data file",
        "data_prompt": "Select one compatible dataset",
        "data_missing": "No compatible local dataset is registered for this market.",
        "core_metric": "Core",
        "branch_metric": "Branches",
        "overlay_metric": "Overlays",
        "status_metric": "Executor",
        "ready": "Ready",
        "pending": "Pending adapter",
        "select_data_status": "Select data",
        "cost_blocked_status": "Costs blocked",
        "incomplete_status": "Incomplete",
        "command": "Python command",
        "config": "Generated configuration",
        "run": "Run strategy backtest",
        "output": "Backtest output",
        "invalid": "Complete the highlighted component fields to generate a runnable draft.",
        "source_empty": "No compatible component is registered for this path and market.",
    },
    "ZH": {
        "title": "策略构建",
        "identity": "策略身份",
        "strategy_id": "策略 ID",
        "name": "策略名称",
        "market": "市场",
        "core": "仓位生成核心",
        "core_type": "核心类型",
        "direct": "内嵌仓位因子（遗留）",
        "factor_sleeve": "因子 + 策略腿",
        "factor_blend": "多因子组合",
        "stat_arb": "统计套利",
        "ml": "已拟合机器学习预测器",
        "routed": "路由组件",
        "factor": "因子",
        "factors": "因子",
        "sleeve": "策略腿",
        "model": "已登记机器学习实验",
        "model_empty": "当前没有登记且保存了样本外预测的机器学习实验。",
        "router": "路由器",
        "router_state": "因果路由状态文件",
        "branch_a": "分支 A",
        "branch_b": "分支 B",
        "branch_source": "分支来源",
        "score_branch": "因子分数组合",
        "direct_branch": "因子直接生成仓位",
        "optional_sleeve": "策略腿转换",
        "none": "无",
        "risk": "风险与配置",
        "overlays": "风险覆盖",
        "gross": "最大总杠杆",
        "margin_limit": "保证金占用上限（%）",
        "cash_reserve": "最低现金储备（%）",
        "contract_cap": "单合约最大权重",
        "execution": "执行",
        "capital": "资金",
        "currency": "币种",
        "cost_profile": "交易成本配置",
        "slippage": "单边滑点（跳）",
        "data": "市场数据",
        "data_file": "数据文件",
        "data_prompt": "选择一个兼容数据集",
        "data_missing": "该市场尚未登记兼容的本地数据集。",
        "core_metric": "核心",
        "branch_metric": "分支",
        "overlay_metric": "风险覆盖",
        "status_metric": "执行器",
        "ready": "可运行",
        "pending": "适配器待接入",
        "select_data_status": "请选择数据",
        "cost_blocked_status": "成本配置受阻",
        "incomplete_status": "配置不完整",
        "command": "Python 命令",
        "config": "生成配置",
        "run": "运行策略回测",
        "output": "回测输出",
        "invalid": "请补全组件字段后生成可运行策略草案。",
        "source_empty": "该市场和路径下没有登记兼容组件。",
    },
}


CORE_LABELS = {
    "direct_factor": "direct",
    "factor_sleeve": "factor_sleeve",
    "factor_blend": "factor_blend",
    "statistical_arbitrage": "stat_arb",
    "ml_predictive": "ml",
    "routed_components": "routed",
}


@st.cache_data(show_spinner=False)
def _factor_inventory() -> pd.DataFrame:
    return factor_inventory()


@st.cache_data(show_spinner=False)
def _sleeve_inventory() -> pd.DataFrame:
    return load_sleeve_library_snapshot(str(BASE_DIR)).definitions


@st.cache_data(show_spinner=False)
def _router_inventory() -> pd.DataFrame:
    return load_router_library_snapshot(str(BASE_DIR)).definitions


@st.cache_data(show_spinner=False)
def _overlay_inventory() -> pd.DataFrame:
    return strategy_risk_overlay_inventory()


@st.cache_data(show_spinner=False)
def _ml_experiments() -> pd.DataFrame:
    rows = list_ml_experiments(DB_PATH, limit=500)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    predictions = frame["predictions_path"].fillna("").astype(str)
    ready = frame["status"].astype(str).str.lower().isin({"completed", "trained"})
    ready &= predictions.map(lambda value: bool(value) and Path(value).is_file())
    ready &= frame["factor_id"].fillna("").astype(str).str.startswith("fac_")
    return frame.loc[ready].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _data_files(base_dir: str = str(BASE_DIR)) -> tuple[str, ...]:
    root = Path(base_dir) / "runtime" / "data"
    if not root.exists():
        return ()
    values: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".parquet", ".csv"}:
            values.append(path.relative_to(Path(base_dir)).as_posix())
        elif path.is_file() and path.name == "materialization.json":
            values.append(path.relative_to(Path(base_dir)).as_posix())
    return tuple(sorted(values))


def _factor_label(row: pd.Series) -> str:
    return f"{row['factor_id']} | {row['name']}"


def _factor_options(frame: pd.DataFrame) -> tuple[list[str], dict[str, pd.Series]]:
    if frame.empty:
        return [], {}
    ordered = frame.sort_values("factor_id")
    labels = [_factor_label(row) for _, row in ordered.iterrows()]
    return labels, {label: row for label, (_, row) in zip(labels, ordered.iterrows(), strict=True)}


def _market_factors(inventory: pd.DataFrame, market: str) -> pd.DataFrame:
    if inventory.empty:
        return inventory
    supported = inventory["supported_markets"].fillna("").astype(str)
    mask = supported.str.split(",").map(
        lambda values: market in {value.strip() for value in values} or "*" in values
    )
    mask &= inventory["load_error"].fillna("").eq("")
    return inventory.loc[mask].reset_index(drop=True)


def _market_data(files: tuple[str, ...], market: str) -> list[str]:
    token = {
        "FUTURES_CN": "runtime/data/futures_cn/",
        "FUTURES_US": "runtime/data/futures_us/",
        "EQUITY_CN": "runtime/data/equity_cn/",
        "EQUITY_US": "runtime/data/equity_us/",
        "OPTIONS_CN": "runtime/data/options_cn/",
        "OPTIONS_US": "runtime/data/options_us/",
    }.get(market, "")
    return [value for value in files if not token or value.startswith(token)]


def _stable_strategy_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"str_{slug or 'research_draft'}"


def _select_factor(
    label: str,
    frame: pd.DataFrame,
    *,
    key: str,
) -> str | None:
    labels, lookup = _factor_options(frame)
    if not labels:
        return None
    selected = st.selectbox(label, labels, key=key)
    return str(lookup[selected]["factor_id"])


def _select_sleeve(label: str, *, key: str) -> str | None:
    inventory = _sleeve_inventory()
    if inventory.empty:
        return None
    lookup = {
        f"{row['sleeve_id']} | {row['name']}": str(row["sleeve_id"])
        for _, row in inventory.sort_values("sleeve_id").iterrows()
    }
    selected = st.selectbox(label, list(lookup), key=key)
    return lookup[selected]


def _branch_controls(
    label: str,
    factors: pd.DataFrame,
    copy: dict[str, str],
    *,
    key: str,
) -> StrategyBranchConfig | None:
    st.markdown(f"#### {label}")
    source = st.selectbox(
        copy["branch_source"],
        [copy["score_branch"], copy["direct_branch"]],
        key=f"{key}_source",
    )
    if source == copy["direct_branch"]:
        eligible = factors.loc[factors["execution_mode"].eq("direct")]
        factor_id = _select_factor(copy["factor"], eligible, key=f"{key}_factor")
        if not factor_id:
            return None
        return StrategyBranchConfig(
            branch_id=key,
            factor_ids=(factor_id,),
            execution_mode="direct",
        )
    eligible = factors.loc[factors["portfolio_layer"].eq("alpha_score")]
    factor_id = _select_factor(copy["factor"], eligible, key=f"{key}_factor")
    if not factor_id:
        return None
    sleeve_inventory = _sleeve_inventory()
    sleeve_lookup = {
        f"{row['sleeve_id']} | {row['name']}": str(row["sleeve_id"])
        for _, row in sleeve_inventory.sort_values("sleeve_id").iterrows()
    }
    options = [copy["none"], *sleeve_lookup]
    sleeve_label = st.selectbox(
        copy["optional_sleeve"], options, key=f"{key}_sleeve"
    )
    return StrategyBranchConfig(
        branch_id=key,
        factor_ids=(factor_id,),
        sleeve_id=sleeve_lookup.get(sleeve_label),
        execution_mode="risk_desk",
    )


def _core_controls(
    core_type: StrategyCoreType,
    factors: pd.DataFrame,
    copy: dict[str, str],
) -> StrategyCoreConfig | None:
    if core_type == StrategyCoreType.DIRECT_FACTOR:
        eligible = factors.loc[factors["execution_mode"].eq("direct")]
        factor_id = _select_factor(copy["factor"], eligible, key="str_direct_factor")
        if not factor_id:
            return None
        return StrategyCoreConfig(
            core_type=core_type,
            branches=(
                StrategyBranchConfig("core", (factor_id,), execution_mode="direct"),
            ),
        )

    if core_type == StrategyCoreType.FACTOR_SLEEVE:
        eligible = factors.loc[
            factors["portfolio_layer"].eq("alpha_score")
            & factors["evaluation_geometry"].eq("cross_sectional")
        ]
        factor_id = _select_factor(copy["factor"], eligible, key="str_sleeve_factor")
        sleeve_id = _select_sleeve(copy["sleeve"], key="str_sleeve")
        if not factor_id or not sleeve_id:
            return None
        return StrategyCoreConfig(
            core_type=core_type,
            branches=(
                StrategyBranchConfig(
                    "core",
                    (factor_id,),
                    sleeve_id=sleeve_id,
                    execution_mode="risk_desk",
                ),
            ),
        )

    if core_type == StrategyCoreType.FACTOR_BLEND:
        eligible = factors.loc[factors["portfolio_layer"].eq("alpha_score")]
        labels, lookup = _factor_options(eligible)
        selected = st.multiselect(copy["factors"], labels, key="str_blend_factors")
        if len(selected) < 2:
            return None
        factor_ids = tuple(str(lookup[value]["factor_id"]) for value in selected)
        return StrategyCoreConfig(
            core_type=core_type,
            branches=(StrategyBranchConfig("core", factor_ids),),
        )

    if core_type == StrategyCoreType.STATISTICAL_ARBITRAGE:
        eligible = factors.loc[factors["execution_mode"].eq("statarb")]
        factor_id = _select_factor(copy["factor"], eligible, key="str_statarb_factor")
        if not factor_id:
            return None
        return StrategyCoreConfig(
            core_type=core_type,
            branches=(
                StrategyBranchConfig("core", (factor_id,), execution_mode="statarb"),
            ),
        )

    if core_type == StrategyCoreType.ML_PREDICTIVE:
        experiments = _ml_experiments()
        if experiments.empty:
            st.caption(copy["model_empty"])
            return None
        lookup = {
            f"{row['experiment_id']} | {row['model_type']} | {row['factor_id']}": row
            for _, row in experiments.iterrows()
        }
        selected = st.selectbox(copy["model"], list(lookup), key="str_ml_experiment")
        row = lookup[selected]
        sleeve_inventory = _sleeve_inventory()
        sleeve_lookup = {
            f"{item['sleeve_id']} | {item['name']}": str(item["sleeve_id"])
            for _, item in sleeve_inventory.sort_values("sleeve_id").iterrows()
        }
        sleeve_label = st.selectbox(
            copy["optional_sleeve"],
            [copy["none"], *sleeve_lookup],
            key="str_ml_sleeve",
        )
        return StrategyCoreConfig(
            core_type=core_type,
            branches=(
                StrategyBranchConfig(
                    "core",
                    (str(row["factor_id"]),),
                    sleeve_id=sleeve_lookup.get(sleeve_label),
                ),
            ),
            ml_experiment_id=str(row["experiment_id"]),
            ml_predictions_path=str(row["predictions_path"]),
        )

    routers = _router_inventory()
    route_candidates = routers.loc[
        routers["supported_markets"].map(
            lambda values: not values or "*" in values or factors.attrs.get("market") in values
        )
    ] if not routers.empty else routers
    router_lookup = {
        f"{row['router_id']} | {row['name']}": str(row["router_id"])
        for _, row in route_candidates.sort_values("router_id").iterrows()
    }
    if not router_lookup:
        return None
    router_label = st.selectbox(copy["router"], list(router_lookup), key="str_router")
    router_state = st.text_input(copy["router_state"], key="str_router_state")
    columns = st.columns(2)
    with columns[0]:
        branch_a = _branch_controls(
            copy["branch_a"], factors, copy, key="branch_a"
        )
    with columns[1]:
        branch_b = _branch_controls(
            copy["branch_b"], factors, copy, key="branch_b"
        )
    if not branch_a or not branch_b:
        return None
    return StrategyCoreConfig(
        core_type=core_type,
        branches=(branch_a, branch_b),
        router_id=router_lookup[router_label],
        router_state_file=router_state or None,
    )


def _cost_controls(
    market: str,
    copy: dict[str, str],
) -> TransactionCostProfile:
    registry = TransactionCostRegistry.load()
    profiles = [
        profile
        for profile in registry.profiles.values()
        if profile.market_vertical == market
    ]
    if not profiles:
        raise ValueError(f"No transaction-cost profile is registered for {market}")
    default_id = registry.default_profiles.get(market)
    profiles.sort(key=lambda item: (item.profile_id != default_id, item.profile_id))
    labels = {
        f"{item.profile_id} | {item.status} | {item.engine_support}": item
        for item in profiles
    }
    selected = st.selectbox(copy["cost_profile"], list(labels), key="str_cost_profile")
    return labels[selected]


def render_strategy_composition_panel(
    artifact_root: str | Path,
    *,
    lang: str = "EN",
) -> bool:
    del artifact_root
    copy = COPY.get(lang, COPY["EN"])
    st.markdown(f"### {copy['title']}")

    inventory = _factor_inventory()
    markets = sorted(
        {
            value.strip()
            for values in inventory["supported_markets"].fillna("").astype(str)
            for value in values.split(",")
            if value.strip() and value.strip() != "*"
        }
    )
    identity = st.columns([0.26, 0.46, 0.28])
    strategy_name = identity[0].text_input(
        copy["name"], value="Research strategy", key="str_name"
    )
    default_id = _stable_strategy_id(strategy_name)
    strategy_id = identity[1].text_input(
        copy["strategy_id"], value=default_id, key="str_id"
    )
    market = identity[2].selectbox(
        copy["market"],
        markets,
        index=markets.index("FUTURES_CN") if "FUTURES_CN" in markets else 0,
        key="str_market",
    )

    factors = _market_factors(inventory, market)
    factors.attrs["market"] = market
    st.markdown(f"#### {copy['core']}")
    core_values = list(StrategyCoreType)
    core_type = st.selectbox(
        copy["core_type"],
        core_values,
        format_func=lambda value: copy[CORE_LABELS[value.value]],
        key="str_core_type",
    )
    core = _core_controls(core_type, factors, copy)

    st.markdown(f"#### {copy['risk']}")
    futures_margin_budget = market == "FUTURES_CN"
    risk_columns = st.columns(
        [0.46, 0.18, 0.18, 0.18]
        if futures_margin_budget
        else [0.50, 0.25, 0.25]
    )
    overlays = _overlay_inventory()
    compatible_overlays = overlays.loc[
        overlays["supported_markets"].fillna("").astype(str).map(
            lambda values: market in {item.strip() for item in values.split(",")}
            or "*" in values
        )
    ]
    overlay_lookup = {
        f"{row['overlay_id']} | {row['name']}": str(row["overlay_id"])
        for _, row in compatible_overlays.sort_values("overlay_id").iterrows()
    }
    selected_overlays = risk_columns[0].multiselect(
        copy["overlays"], list(overlay_lookup), key="str_overlays"
    )
    if futures_margin_budget:
        margin_percent = risk_columns[1].number_input(
            copy["margin_limit"],
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            key="str_margin_limit",
        )
        risk_columns[2].metric(
            copy["cash_reserve"],
            f"{100.0 - margin_percent:.0f}%",
        )
        max_gross = None
        max_margin_utilization = margin_percent / 100.0
        contract_column = risk_columns[3]
    else:
        max_gross = risk_columns[1].number_input(
            copy["gross"],
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="str_max_gross",
        )
        max_margin_utilization = None
        contract_column = risk_columns[2]
    max_contract = contract_column.number_input(
        copy["contract_cap"], min_value=0.01, max_value=1.0, value=0.10, step=0.01
    )

    st.markdown(f"#### {copy['execution']}")
    execution = st.columns([0.22, 0.16, 0.34, 0.28])
    capital = execution[0].number_input(
        copy["capital"], min_value=1_000.0, value=10_000_000.0, step=100_000.0
    )
    with execution[2]:
        cost_profile = _cost_controls(market, copy)
    profile_id = cost_profile.profile_id
    currency = cost_profile.currency
    slippage_default = float(cost_profile.slippage.get("ticks_per_side", 0.0))
    execution[1].text_input(copy["currency"], value=currency, disabled=True)
    slippage = execution[3].number_input(
        copy["slippage"],
        min_value=0.0,
        value=slippage_default,
        step=0.1,
        disabled=True,
    )

    market_files = _market_data(_data_files(), market)
    data_file = (
        st.selectbox(
            copy["data_file"],
            market_files,
            index=None,
            placeholder=copy["data_prompt"],
            key="str_data_file",
        )
        if market_files
        else ""
    )
    if not market_files:
        st.caption(copy["data_missing"])

    config: StrategyBuilderConfig | None = None
    error = ""
    if core is not None:
        try:
            config = StrategyBuilderConfig(
                strategy_id=strategy_id,
                name=strategy_name,
                market_vertical=market,
                core=core,
                risk_overlays=tuple(overlay_lookup[value] for value in selected_overlays),
                allocator=StrategyAllocatorConfig(
                    max_gross_leverage=max_gross,
                    max_contract_weight=max_contract,
                    max_margin_utilization=max_margin_utilization,
                ),
                execution=StrategyExecutionConfig(
                    capital=capital,
                    capital_currency=currency,
                    transaction_cost_profile=profile_id,
                    slippage_ticks_per_side=slippage,
                ),
            )
        except ValueError as exc:
            error = str(exc)

    support = strategy_execution_support(config) if config else None
    summary = st.columns(4)
    summary[0].metric(copy["core_metric"], copy[CORE_LABELS[core_type.value]])
    summary[1].metric(copy["branch_metric"], len(core.branches) if core else 0)
    summary[2].metric(copy["overlay_metric"], len(selected_overlays))
    execution_ready = bool(
        support
        and support.runnable
        and cost_profile.research_net_ready
        and data_file
    )
    if not support:
        executor_status = copy["incomplete_status"]
    elif not support.runnable:
        executor_status = copy["pending"]
    elif not cost_profile.research_net_ready:
        executor_status = copy["cost_blocked_status"]
    elif not data_file:
        executor_status = copy["select_data_status"]
    else:
        executor_status = copy["ready"]
    summary[3].metric(
        copy["status_metric"],
        executor_status,
    )

    if config is None:
        st.caption(error or copy["invalid"])
        st.button(copy["run"], disabled=True, width="stretch")
        return False

    config_path = (
        Path(BASE_DIR)
        / "runtime"
        / "configs"
        / "research"
        / "strategy_builder"
        / f"{config.strategy_id}.yaml"
    )
    write_strategy_builder_config(config, config_path)
    relative_config = config_path.relative_to(Path(BASE_DIR))
    command = build_strategy_backtest_command(
        config_path=relative_config,
        data_file=data_file or "<compatible-data-file>",
    )
    command_text = "PYTHONPATH=src:. " + shlex.join(command)

    if support and not support.runnable:
        st.caption(support.reason)
    if not cost_profile.research_net_ready:
        actions = cost_profile.readiness_actions()
        st.caption(actions[0] if actions else f"{profile_id} is not research-net ready.")
    with st.expander(copy["config"], expanded=False):
        st.code(
            yaml.safe_dump({"strategy": config.to_dict()}, sort_keys=False),
            language="yaml",
        )
    st.markdown(f"#### {copy['command']}")
    st.code(command_text, language="bash")

    runnable = execution_ready
    if st.button(copy["run"], disabled=not runnable, width="stretch"):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = "src:."
        process_command = [
            sys.executable,
            "scripts/research/run_strategy_backtest.py",
            "--config",
            relative_config.as_posix(),
            "--data-file",
            data_file,
        ]
        with st.spinner(copy["run"]):
            try:
                completed = subprocess.run(
                    process_command,
                    cwd=BASE_DIR,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=3_600,
                    check=False,
                )
                output = "\n".join(
                    value.strip()
                    for value in (completed.stdout, completed.stderr)
                    if value.strip()
                )
                st.session_state["strategy_builder_output"] = output
                st.session_state["strategy_builder_return_code"] = completed.returncode
            except subprocess.TimeoutExpired:
                st.session_state["strategy_builder_output"] = "Backtest timed out after 3600 seconds."
                st.session_state["strategy_builder_return_code"] = 124

    output = st.session_state.get("strategy_builder_output")
    if output:
        st.markdown(f"#### {copy['output']}")
        st.code(output, language="text")
    return True


__all__ = ["render_strategy_composition_panel"]
