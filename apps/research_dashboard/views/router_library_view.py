from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import BASE_DIR


COPY = {
    "EN": {
        "title": "Router Library",
        "subtitle": (
            "Causal router states, allocation routers, position policies, "
            "and post-construction risk overlays."
        ),
        "routers_tab": "Routers",
        "states_tab": "Router States",
        "position_policies_tab": "Position Policies",
        "overlays_tab": "Risk Overlays",
        "registered": "Registered routers",
        "discrete": "Discrete rules",
        "continuous": "Continuous rules",
        "markets": "Markets",
        "frequencies": "Frequencies",
        "inventory_tab": "Inventory",
        "drilldown_tab": "Router Drilldown",
        "inventory": "Active router inventory",
        "market": "Market",
        "frequency": "Frequency",
        "status": "Status",
        "all": "All",
        "router_id": "Router ID",
        "name": "Router",
        "indicator": "Routing input",
        "allocation": "Allocation rule",
        "sleeve_count": "Sleeve slots",
        "decision_lag": "Decision lag",
        "periods": "{count} period(s)",
        "select": "Inspect router",
        "rationale": "Economic rationale",
        "limitations": "Known limitations",
        "no_limitations": "No router-specific limitation has been documented.",
        "contract": "Routing contract",
        "parameters": "Parameter schema",
        "parameter": "Parameter",
        "default": "Default",
        "type": "Type",
        "tunable": "Tunable",
        "choices": "Choices",
        "field": "Field",
        "value": "Value",
        "source": "Source",
        "yes": "Yes",
        "no": "No",
        "no_inventory": "No registered router definitions were found.",
        "state_registered": "Registered router states",
        "state_inventory": "Active router-state inventory",
        "state_drilldown_tab": "State Drilldown",
        "state_id": "Router State ID",
        "state_name": "Router state",
        "state_col": "State output",
        "output_type": "Output type",
        "state_contract": "Router-state contract",
        "select_state": "Inspect router state",
        "no_state_inventory": "No registered router-state definitions were found.",
        "policy_registered": "Registered position policies",
        "policy_inventory": "Active position-policy inventory",
        "policy_drilldown_tab": "Policy Drilldown",
        "policy_id": "Position Policy ID",
        "policy_name": "Position policy",
        "policy_contract": "Position-policy contract",
        "select_policy": "Inspect position policy",
        "no_policy_inventory": "No registered position-policy definitions were found.",
        "source_factor_ids": "Source factor IDs",
        "economic_role": "Economic role",
        "no_economic_role": "No economic role has been documented.",
        "overlay_registered": "Registered overlays",
        "portfolio_scalar": "Portfolio-wide",
        "sign_flip": "May flip signs",
        "gross_increase": "May increase gross",
        "overlay_inventory": "Active risk-overlay inventory",
        "overlay_drilldown_tab": "Overlay Drilldown",
        "overlay_id": "Overlay ID",
        "overlay_name": "Risk overlay",
        "scope": "Scope",
        "decision_time": "Decision time",
        "effective_time": "Effective time",
        "no_overlay_inventory": "No registered risk overlays were found.",
    },
    "ZH": {
        "title": "路由器库",
        "subtitle": "因果路由状态、资金分配路由器、仓位策略与策略生成后的风险覆盖组件。",
        "routers_tab": "路由器",
        "states_tab": "路由状态",
        "position_policies_tab": "仓位策略",
        "overlays_tab": "风险覆盖",
        "registered": "已注册路由器",
        "discrete": "离散规则",
        "continuous": "连续规则",
        "markets": "覆盖市场",
        "frequencies": "状态频率",
        "inventory_tab": "组件清单",
        "drilldown_tab": "路由器明细",
        "inventory": "在库路由器",
        "market": "市场",
        "frequency": "频率",
        "status": "状态",
        "all": "全部",
        "router_id": "路由器 ID",
        "name": "路由器",
        "indicator": "路由输入",
        "allocation": "分配规则",
        "sleeve_count": "策略腿槽位",
        "decision_lag": "决策延迟",
        "periods": "{count} 个周期",
        "select": "查看路由器",
        "rationale": "经济逻辑",
        "limitations": "已知局限",
        "no_limitations": "尚未记录该路由器的特定局限。",
        "contract": "路由合约",
        "parameters": "参数结构",
        "parameter": "参数",
        "default": "默认值",
        "type": "类型",
        "tunable": "可调节",
        "choices": "候选值",
        "field": "字段",
        "value": "数值",
        "source": "源码",
        "yes": "是",
        "no": "否",
        "no_inventory": "未找到已注册的路由器定义。",
        "state_registered": "已注册路由状态",
        "state_inventory": "在库路由状态",
        "state_drilldown_tab": "路由状态明细",
        "state_id": "路由状态 ID",
        "state_name": "路由状态",
        "state_col": "状态输出",
        "output_type": "输出类型",
        "state_contract": "路由状态合约",
        "select_state": "查看路由状态",
        "no_state_inventory": "未找到已注册的路由状态定义。",
        "policy_registered": "已注册仓位策略",
        "policy_inventory": "在库仓位策略",
        "policy_drilldown_tab": "仓位策略明细",
        "policy_id": "仓位策略 ID",
        "policy_name": "仓位策略",
        "policy_contract": "仓位策略合约",
        "select_policy": "查看仓位策略",
        "no_policy_inventory": "未找到已注册的仓位策略定义。",
        "source_factor_ids": "来源因子 ID",
        "economic_role": "经济作用",
        "no_economic_role": "尚未记录经济作用。",
        "overlay_registered": "已注册风险覆盖",
        "portfolio_scalar": "组合整体缩放",
        "sign_flip": "允许反转方向",
        "gross_increase": "允许提高总敞口",
        "overlay_inventory": "在库风险覆盖",
        "overlay_drilldown_tab": "风险覆盖明细",
        "overlay_id": "风险覆盖 ID",
        "overlay_name": "风险覆盖",
        "scope": "作用范围",
        "decision_time": "决策时点",
        "effective_time": "生效时点",
        "no_overlay_inventory": "未找到已注册的风险覆盖组件。",
    },
}


STATUS_LABELS = {
    "EN": {
        "rejected_as_robust_router": "Rejected: not robust",
        "rejected_as_production_router": "Rejected: production",
        "rejected_fixed_candidate": "Rejected candidate",
        "historical_prototype_decomposition": "Historical prototype",
        "hypothesis": "Hypothesis",
        "registered_untested": "Registered, untested",
    },
    "ZH": {
        "rejected_as_robust_router": "已否决：稳健性不足",
        "rejected_as_production_router": "已否决：不适合生产",
        "rejected_fixed_candidate": "已否决候选",
        "historical_prototype_decomposition": "历史原型",
        "hypothesis": "研究假设",
        "registered_untested": "已注册，待测试",
    },
}


ALLOCATION_LABELS = {
    "EN": {
        "discrete_state_switch": "Discrete state switch",
        "continuous_state_blend": "Continuous state blend",
    },
    "ZH": {
        "discrete_state_switch": "离散状态切换",
        "continuous_state_blend": "连续状态混合",
    },
}


INVENTORY_FIELDS = (
    "router_id",
    "name",
    "status",
    "routing_indicator",
    "allocation_style",
    "sleeve_count",
    "decision_lag_periods",
    "market_label",
    "frequency",
)


@dataclass(frozen=True)
class RouterLibrarySnapshot:
    definitions: pd.DataFrame
    router_states: pd.DataFrame
    position_policies: pd.DataFrame
    risk_overlays: pd.DataFrame


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
    raise ValueError(f"unsupported router metadata expression: {type(node).__name__}")


def _module_declarations(path: Path) -> dict[str, Any]:
    """Read literal declarations without importing or executing a component."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: dict[str, Any] = {}
    for node in tree.body:
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
    return names


def _component_definition(
    path: Path,
    *,
    id_name: str,
    metadata_name: str,
    contract_name: str,
    parameter_names: tuple[str, ...] = (),
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    names = _module_declarations(path)
    component_id = str(names.get(id_name, "")).strip()
    metadata = names.get(metadata_name, {})
    contract = names.get(contract_name, {})
    parameters = next(
        (
            names[name]
            for name in parameter_names
            if isinstance(names.get(name), dict)
        ),
        {},
    )
    if (
        not component_id
        or not isinstance(metadata, dict)
        or not isinstance(contract, dict)
    ):
        raise ValueError(
            f"{path.name} does not expose the {id_name.lower()} metadata contract"
        )
    if not isinstance(parameters, dict):
        parameters = {}
    return component_id, metadata, contract, parameters


def _module_definition(
    path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    return _component_definition(
        path,
        id_name="ROUTER_ID",
        metadata_name="ROUTER_METADATA",
        contract_name="ROUTER_CONTRACT",
        parameter_names=("ROUTER_PARAMETERS",),
    )


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if str(item or "").strip()
        )
    )


def _component_name(component_id: str) -> str:
    parts = component_id.split("_", maxsplit=2)
    return (parts[-1] if len(parts) == 3 else component_id).replace("_", " ")


def _state_input(parameters: dict[str, Any], contract: dict[str, Any]) -> str:
    state_parameter = parameters.get("state_col", {})
    if isinstance(state_parameter, dict) and state_parameter.get("default"):
        return str(state_parameter["default"])
    return str(contract.get("state_col", "state"))


def _load_router_states(root: Path, router_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((router_root / "states").glob("rst_*.py")):
        state_id, metadata, contract, parameters = _component_definition(
            path,
            id_name="ROUTER_STATE_ID",
            metadata_name="ROUTER_STATE_METADATA",
            contract_name="ROUTER_STATE_CONTRACT",
            parameter_names=("ROUTER_STATE_PARAMETERS",),
        )
        markets = _string_values(
            metadata.get("supported_markets")
            or contract.get("supported_markets")
            or []
        )
        source_factor_ids = _string_values(
            metadata.get("source_factor_ids")
            or [
                value
                for value in metadata.get("legacy_ids", [])
                if str(value).startswith("fac_")
            ]
        )
        rows.append(
            {
                "state_id": state_id,
                "name": str(
                    metadata.get("name") or _component_name(state_id)
                ),
                "status": str(metadata.get("status", "registered_untested")),
                "native_market": str(metadata.get("native_market", "")),
                "data_frequency": str(metadata.get("data_frequency", "")),
                "supported_markets": markets,
                "market_label": ", ".join(markets),
                "state_col": str(contract.get("state_col", "")),
                "output_type": str(contract.get("output_type", "")),
                "decision_lag": str(contract.get("decision_lag", "")),
                "source_factor_ids": source_factor_ids,
                "source_factor_label": ", ".join(source_factor_ids),
                "economic_role": str(
                    metadata.get("economic_role")
                    or metadata.get("economic_rationale")
                    or ""
                ),
                "known_limitations": str(
                    metadata.get("known_limitations")
                    or metadata.get("known_failure")
                    or ""
                ),
                "contract": contract,
                "parameters": parameters,
                "source_path": str(path.relative_to(root)),
            }
        )
    return pd.DataFrame(rows)


def _load_position_policies(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policy_root = root / "departments" / "research" / "position_policies"
    for path in sorted(policy_root.glob("pos_*.py")):
        policy_id, metadata, contract, parameters = _component_definition(
            path,
            id_name="POSITION_POLICY_ID",
            metadata_name="POSITION_POLICY_METADATA",
            contract_name="POSITION_POLICY_CONTRACT",
            parameter_names=("POSITION_POLICY_PARAMETERS",),
        )
        markets = _string_values(
            metadata.get("supported_markets")
            or contract.get("supported_markets")
            or []
        )
        source_factor_ids = _string_values(
            metadata.get("source_factor_ids")
            or [
                value
                for value in metadata.get("legacy_ids", [])
                if str(value).startswith("fac_")
            ]
        )
        rows.append(
            {
                "policy_id": policy_id,
                "name": str(
                    metadata.get("name") or _component_name(policy_id)
                ),
                "status": str(metadata.get("status", "registered_untested")),
                "native_market": str(metadata.get("native_market", "")),
                "data_frequency": str(metadata.get("data_frequency", "")),
                "supported_markets": markets,
                "market_label": ", ".join(markets),
                "scope": str(
                    contract.get("scope")
                    or contract.get("output_type")
                    or ""
                ),
                "output_type": str(contract.get("output_type", "")),
                "decision_lag": str(contract.get("decision_lag", "")),
                "allow_sign_flip": bool(
                    contract.get("allow_sign_flip", False)
                ),
                "allow_gross_increase": bool(
                    contract.get("allow_gross_increase", False)
                ),
                "source_factor_ids": source_factor_ids,
                "source_factor_label": ", ".join(source_factor_ids),
                "economic_role": str(
                    metadata.get("economic_role")
                    or metadata.get("economic_rationale")
                    or ""
                ),
                "known_limitations": str(
                    metadata.get("known_limitations")
                    or metadata.get("known_failure")
                    or ""
                ),
                "contract": contract,
                "parameters": parameters,
                "source_path": str(path.relative_to(root)),
            }
        )
    return pd.DataFrame(rows)


def _load_risk_overlays(root: Path) -> pd.DataFrame:
    """Build the overlay inventory statically, without importing overlay code."""

    rows: list[dict[str, Any]] = []
    overlay_root = root / "departments" / "research" / "strategy_overlays"
    for path in sorted(overlay_root.glob("ovl_*.py")):
        overlay_id, metadata, contract, parameters = _component_definition(
            path,
            id_name="OVERLAY_ID",
            metadata_name="OVERLAY_METADATA",
            contract_name="OVERLAY_CONTRACT",
            parameter_names=("DEFAULT_PARAMETERS",),
        )
        markets = _string_values(
            metadata.get("supported_markets")
            or contract.get("supported_markets")
            or []
        )
        source_factor_ids = _string_values(
            metadata.get("source_factor_ids")
            or metadata.get("legacy_factor_ids")
            or []
        )
        rows.append(
            {
                "overlay_id": overlay_id,
                "name": str(metadata.get("name") or overlay_id),
                "status": str(metadata.get("status") or "unclassified"),
                "frequency": str(
                    metadata.get("frequency") or "unclassified"
                ),
                "supported_markets": ", ".join(markets),
                "scope": str(contract.get("scope", "")),
                "decision_time": str(contract.get("decision_time", "")),
                "effective_time": str(contract.get("effective_time", "")),
                "allow_sign_flip": bool(
                    contract.get("allow_sign_flip", False)
                ),
                "allow_gross_increase": bool(
                    contract.get("allow_gross_increase", False)
                ),
                "source_factor_ids": source_factor_ids,
                "economic_rationale": str(
                    metadata.get("economic_rationale")
                    or metadata.get("economic_role")
                    or ""
                ),
                "known_limitations": str(
                    metadata.get("known_limitations")
                    or metadata.get("known_failure")
                    or ""
                ),
                "contract": contract,
                "parameters": parameters,
                "source": str(path),
                "load_error": "",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_router_library_snapshot(base_dir: str = str(BASE_DIR)) -> RouterLibrarySnapshot:
    root = Path(base_dir)
    router_root = root / "departments" / "research" / "routers"
    rows: list[dict[str, Any]] = []
    for path in sorted(router_root.glob("rtr_*.py")):
        router_id, metadata, contract, parameters = _module_definition(path)
        markets = tuple(
            str(value)
            for value in (
                metadata.get("supported_markets")
                or contract.get("supported_markets")
                or []
            )
        )
        rows.append(
            {
                "router_id": router_id,
                "name": str(metadata.get("name") or router_id),
                "status": str(metadata.get("status", "registered_untested")),
                "supported_markets": markets,
                "market_label": ", ".join(markets),
                "frequency": str(metadata.get("frequency", "")),
                "routing_indicator": str(
                    metadata.get("routing_indicator")
                    or _state_input(parameters, contract)
                ),
                "allocation_style": str(
                    metadata.get("allocation_style", "state_allocation")
                ),
                "sleeve_count": int(metadata.get("sleeve_count", 0)),
                "decision_lag_periods": int(
                    contract.get("decision_lag_periods", 0)
                ),
                "economic_rationale": str(
                    metadata.get("economic_rationale")
                    or metadata.get("economic_mechanism")
                    or ""
                ),
                "known_limitations": str(
                    metadata.get("known_limitations")
                    or metadata.get("known_failure")
                    or ""
                ),
                "contract": contract,
                "parameters": parameters,
                "source_path": str(path.relative_to(root)),
            }
        )
    return RouterLibrarySnapshot(
        definitions=pd.DataFrame(rows),
        router_states=_load_router_states(root, router_root),
        position_policies=_load_position_policies(root),
        risk_overlays=_load_risk_overlays(root),
    )


class RouterLibraryView:
    def __init__(self, base_dir: str | Path = BASE_DIR):
        self.base_dir = Path(base_dir)

    def render(self, lang: str = "EN") -> None:
        copy = COPY.get(lang, COPY["EN"])
        snapshot = load_router_library_snapshot(str(self.base_dir))
        st.markdown(f"### {copy['title']}")
        st.caption(copy["subtitle"])
        if (
            snapshot.definitions.empty
            and snapshot.router_states.empty
            and snapshot.position_policies.empty
            and snapshot.risk_overlays.empty
        ):
            st.info(copy["no_inventory"])
            return
        router_tab, state_tab, policy_tab, overlay_tab = st.tabs(
            [
                copy["routers_tab"],
                copy["states_tab"],
                copy["position_policies_tab"],
                copy["overlays_tab"],
            ]
        )
        with router_tab:
            if snapshot.definitions.empty:
                st.info(copy["no_inventory"])
            else:
                self._render_summary(snapshot.definitions, copy)
                inventory_tab, drilldown_tab = st.tabs(
                    [copy["inventory_tab"], copy["drilldown_tab"]]
                )
                with inventory_tab:
                    self._render_inventory(snapshot.definitions, copy, lang)
                with drilldown_tab:
                    self._render_drilldown(snapshot.definitions, copy, lang)
        with state_tab:
            if snapshot.router_states.empty:
                st.info(copy["no_state_inventory"])
            else:
                self._render_component_summary(
                    snapshot.router_states,
                    copy["state_registered"],
                    copy,
                )
                inventory_tab, drilldown_tab = st.tabs(
                    [copy["inventory_tab"], copy["state_drilldown_tab"]]
                )
                with inventory_tab:
                    self._render_state_inventory(
                        snapshot.router_states,
                        copy,
                        lang,
                    )
                with drilldown_tab:
                    self._render_state_drilldown(
                        snapshot.router_states,
                        copy,
                        lang,
                    )
        with policy_tab:
            if snapshot.position_policies.empty:
                st.info(copy["no_policy_inventory"])
            else:
                self._render_component_summary(
                    snapshot.position_policies,
                    copy["policy_registered"],
                    copy,
                )
                inventory_tab, drilldown_tab = st.tabs(
                    [copy["inventory_tab"], copy["policy_drilldown_tab"]]
                )
                with inventory_tab:
                    self._render_policy_inventory(
                        snapshot.position_policies,
                        copy,
                        lang,
                    )
                with drilldown_tab:
                    self._render_policy_drilldown(
                        snapshot.position_policies,
                        copy,
                        lang,
                    )
        with overlay_tab:
            if snapshot.risk_overlays.empty:
                st.info(copy["no_overlay_inventory"])
            else:
                self._render_overlay_summary(snapshot.risk_overlays, copy)
                inventory_tab, drilldown_tab = st.tabs(
                    [copy["inventory_tab"], copy["overlay_drilldown_tab"]]
                )
                with inventory_tab:
                    self._render_overlay_inventory(
                        snapshot.risk_overlays, copy, lang
                    )
                with drilldown_tab:
                    self._render_overlay_drilldown(
                        snapshot.risk_overlays, copy, lang
                    )

    @staticmethod
    def _render_summary(definitions: pd.DataFrame, copy: dict[str, str]) -> None:
        market_count = len(
            {
                market
                for markets in definitions["supported_markets"]
                for market in markets
            }
        )
        allocation = definitions["allocation_style"].astype(str)
        cards = st.columns(5)
        cards[0].metric(copy["registered"], f"{len(definitions):,}")
        cards[1].metric(copy["discrete"], f"{allocation.str.startswith('discrete').sum():,}")
        cards[2].metric(
            copy["continuous"], f"{allocation.str.startswith('continuous').sum():,}"
        )
        cards[3].metric(copy["markets"], f"{market_count:,}")
        cards[4].metric(copy["frequencies"], f"{definitions['frequency'].nunique():,}")

    @staticmethod
    def _render_component_summary(
        definitions: pd.DataFrame,
        registered_label: str,
        copy: dict[str, str],
    ) -> None:
        markets = {
            market
            for values in definitions["supported_markets"]
            for market in values
        }
        cards = st.columns(3)
        cards[0].metric(registered_label, f"{len(definitions):,}")
        cards[1].metric(copy["markets"], f"{len(markets):,}")
        cards[2].metric(
            copy["frequencies"],
            f"{definitions['data_frequency'].nunique():,}",
        )

    @classmethod
    def _render_state_inventory(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        st.markdown(f"#### {copy['state_inventory']}")
        statuses = sorted(definitions["status"].dropna().unique())
        status = st.selectbox(
            copy["status"],
            [copy["all"], *statuses],
            format_func=lambda value: (
                value if value == copy["all"] else cls._status_label(value, lang)
            ),
            key="router_state_library_status",
        )
        filtered = definitions.copy()
        if status != copy["all"]:
            filtered = filtered.loc[filtered["status"].eq(status)]
        display = filtered[
            [
                "state_id",
                "name",
                "status",
                "native_market",
                "data_frequency",
                "state_col",
                "output_type",
                "decision_lag",
                "source_factor_label",
            ]
        ].copy()
        display["status"] = display["status"].map(
            lambda value: cls._status_label(value, lang)
        )
        for column in (
            "native_market",
            "data_frequency",
            "state_col",
            "output_type",
            "decision_lag",
        ):
            display[column] = display[column].map(cls._plain_label)
        display = display.rename(
            columns={
                "state_id": copy["state_id"],
                "name": copy["state_name"],
                "status": copy["status"],
                "native_market": copy["market"],
                "data_frequency": copy["frequency"],
                "state_col": copy["state_col"],
                "output_type": copy["output_type"],
                "decision_lag": copy["decision_lag"],
                "source_factor_label": copy["source_factor_ids"],
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=340,
        )

    @classmethod
    def _render_policy_inventory(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        st.markdown(f"#### {copy['policy_inventory']}")
        statuses = sorted(definitions["status"].dropna().unique())
        status = st.selectbox(
            copy["status"],
            [copy["all"], *statuses],
            format_func=lambda value: (
                value if value == copy["all"] else cls._status_label(value, lang)
            ),
            key="position_policy_library_status",
        )
        filtered = definitions.copy()
        if status != copy["all"]:
            filtered = filtered.loc[filtered["status"].eq(status)]
        display = filtered[
            [
                "policy_id",
                "name",
                "status",
                "native_market",
                "data_frequency",
                "scope",
                "decision_lag",
                "source_factor_label",
            ]
        ].copy()
        display["status"] = display["status"].map(
            lambda value: cls._status_label(value, lang)
        )
        for column in (
            "native_market",
            "data_frequency",
            "scope",
            "decision_lag",
        ):
            display[column] = display[column].map(cls._plain_label)
        display = display.rename(
            columns={
                "policy_id": copy["policy_id"],
                "name": copy["policy_name"],
                "status": copy["status"],
                "native_market": copy["market"],
                "data_frequency": copy["frequency"],
                "scope": copy["scope"],
                "decision_lag": copy["decision_lag"],
                "source_factor_label": copy["source_factor_ids"],
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=340,
        )

    @classmethod
    def _render_state_drilldown(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        labels = {
            row.state_id: f"{row.state_id} | {row.name}"
            for row in definitions.itertuples(index=False)
        }
        state_id = st.selectbox(
            copy["select_state"],
            list(labels),
            format_func=labels.get,
            key="router_state_library_drilldown",
        )
        row = definitions.loc[definitions["state_id"].eq(state_id)].iloc[0]
        facts = (
            (copy["status"], cls._status_label(row["status"], lang)),
            (
                copy["market"],
                row["native_market"] or row["market_label"] or "N/A",
            ),
            (copy["frequency"], cls._plain_label(row["data_frequency"])),
            (copy["state_col"], cls._plain_label(row["state_col"])),
        )
        cls._render_component_drilldown(
            row,
            facts,
            copy,
            copy["state_contract"],
        )

    @classmethod
    def _render_policy_drilldown(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        labels = {
            row.policy_id: f"{row.policy_id} | {row.name}"
            for row in definitions.itertuples(index=False)
        }
        policy_id = st.selectbox(
            copy["select_policy"],
            list(labels),
            format_func=labels.get,
            key="position_policy_library_drilldown",
        )
        row = definitions.loc[definitions["policy_id"].eq(policy_id)].iloc[0]
        facts = (
            (copy["status"], cls._status_label(row["status"], lang)),
            (copy["scope"], cls._plain_label(row["scope"])),
            (copy["decision_lag"], cls._plain_label(row["decision_lag"])),
            (
                copy["market"],
                row["native_market"] or row["market_label"] or "N/A",
            ),
        )
        cls._render_component_drilldown(
            row,
            facts,
            copy,
            copy["policy_contract"],
        )

    @staticmethod
    def _render_component_drilldown(
        row: pd.Series,
        facts: tuple[tuple[str, str], ...],
        copy: dict[str, str],
        contract_label: str,
    ) -> None:
        cards = st.columns(len(facts))
        for column, (label, value) in zip(cards, facts, strict=True):
            column.caption(label)
            column.markdown(f"**{value or 'N/A'}**")
        st.markdown(f"#### {copy['economic_role']}")
        if row["economic_role"]:
            st.write(row["economic_role"])
        else:
            st.caption(copy["no_economic_role"])
        st.markdown(f"#### {copy['limitations']}")
        st.write(row["known_limitations"] or copy["no_limitations"])
        if row["source_factor_label"]:
            st.caption(
                f"{copy['source_factor_ids']}: "
                f"{row['source_factor_label']}"
            )
        st.markdown(f"#### {contract_label}")
        contract = pd.DataFrame(
            [
                {copy["field"]: key, copy["value"]: str(value)}
                for key, value in row["contract"].items()
            ]
        )
        st.dataframe(contract, use_container_width=True, hide_index=True)
        st.caption(f"{copy['source']}: {row['source_path']}")

    @staticmethod
    def _render_overlay_summary(
        definitions: pd.DataFrame,
        copy: dict[str, str],
    ) -> None:
        cards = st.columns(4)
        cards[0].metric(copy["overlay_registered"], f"{len(definitions):,}")
        cards[1].metric(
            copy["portfolio_scalar"],
            f"{definitions['scope'].eq('portfolio_scalar').sum():,}",
        )
        cards[2].metric(
            copy["sign_flip"],
            f"{definitions['allow_sign_flip'].eq(True).sum():,}",
        )
        cards[3].metric(
            copy["gross_increase"],
            f"{definitions['allow_gross_increase'].eq(True).sum():,}",
        )

    @staticmethod
    def _status_label(value: Any, lang: str) -> str:
        raw = str(value)
        return STATUS_LABELS.get(lang, STATUS_LABELS["EN"]).get(
            raw, raw.replace("_", " ").title()
        )

    @staticmethod
    def _plain_label(value: Any) -> str:
        return str(value).replace("_", " ").title()

    @staticmethod
    def _allocation_label(value: Any, lang: str) -> str:
        raw = str(value)
        return ALLOCATION_LABELS.get(lang, ALLOCATION_LABELS["EN"]).get(
            raw, raw.replace("_", " ").title()
        )

    @classmethod
    def _render_inventory(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        st.markdown(f"#### {copy['inventory']}")
        market_options = sorted(
            {
                market
                for markets in definitions["supported_markets"]
                for market in markets
            }
        )
        frequency_options = sorted(definitions["frequency"].dropna().unique())
        status_options = sorted(definitions["status"].dropna().unique())
        controls = st.columns(3)
        market = controls[0].selectbox(
            copy["market"],
            [copy["all"], *market_options],
            key="router_library_market",
        )
        frequency = controls[1].selectbox(
            copy["frequency"],
            [copy["all"], *frequency_options],
            format_func=lambda value: (
                value if value == copy["all"] else cls._plain_label(value)
            ),
            key="router_library_frequency",
        )
        status = controls[2].selectbox(
            copy["status"],
            [copy["all"], *status_options],
            format_func=lambda value: (
                value if value == copy["all"] else cls._status_label(value, lang)
            ),
            key="router_library_status",
        )

        filtered = definitions.copy()
        if market != copy["all"]:
            filtered = filtered.loc[
                filtered["supported_markets"].map(lambda values: market in values)
            ]
        if frequency != copy["all"]:
            filtered = filtered.loc[filtered["frequency"].eq(frequency)]
        if status != copy["all"]:
            filtered = filtered.loc[filtered["status"].eq(status)]

        display = filtered[list(INVENTORY_FIELDS)].copy()
        display["status"] = display["status"].map(
            lambda value: cls._status_label(value, lang)
        )
        display["routing_indicator"] = display["routing_indicator"].map(
            cls._plain_label
        )
        display["allocation_style"] = display["allocation_style"].map(
            lambda value: cls._allocation_label(value, lang)
        )
        display["frequency"] = display["frequency"].map(cls._plain_label)
        display["decision_lag_periods"] = display["decision_lag_periods"].map(
            lambda value: copy["periods"].format(count=int(value))
        )
        display = display.rename(
            columns={
                "router_id": copy["router_id"],
                "name": copy["name"],
                "status": copy["status"],
                "routing_indicator": copy["indicator"],
                "allocation_style": copy["allocation"],
                "sleeve_count": copy["sleeve_count"],
                "decision_lag_periods": copy["decision_lag"],
                "market_label": copy["market"],
                "frequency": copy["frequency"],
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True, height=340)

    @classmethod
    def _render_drilldown(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        labels = {
            row.router_id: f"{row.router_id} | {row.name}"
            for row in definitions.itertuples(index=False)
        }
        router_id = st.selectbox(
            copy["select"],
            list(labels),
            format_func=labels.get,
            key="router_library_drilldown",
        )
        row = definitions.loc[definitions["router_id"].eq(router_id)].iloc[0]
        cards = st.columns(4)
        facts = (
            (copy["status"], cls._status_label(row["status"], lang)),
            (copy["indicator"], cls._plain_label(row["routing_indicator"])),
            (copy["allocation"], cls._allocation_label(row["allocation_style"], lang)),
            (copy["sleeve_count"], f"{int(row['sleeve_count']):,}"),
        )
        for column, (label, value) in zip(cards, facts, strict=True):
            column.caption(label)
            column.markdown(f"**{value}**")

        st.markdown(f"#### {copy['rationale']}")
        st.write(row["economic_rationale"])
        st.markdown(f"#### {copy['limitations']}")
        if row["known_limitations"]:
            st.write(row["known_limitations"])
        else:
            st.caption(copy["no_limitations"])

        st.markdown(f"#### {copy['contract']}")
        contract = pd.DataFrame(
            [
                {copy["field"]: key, copy["value"]: str(value)}
                for key, value in row["contract"].items()
            ]
        )
        st.dataframe(contract, use_container_width=True, hide_index=True)

        st.markdown(f"#### {copy['parameters']}")
        parameter_rows = []
        for name, specification in row["parameters"].items():
            spec = specification if isinstance(specification, dict) else {}
            parameter_rows.append(
                {
                    copy["parameter"]: name,
                    copy["default"]: str(spec.get("default", "")),
                    copy["type"]: str(spec.get("type", "")),
                    copy["tunable"]: copy["yes"] if spec.get("tunable") else copy["no"],
                    copy["choices"]: ", ".join(
                        str(value) for value in spec.get("choices", [])
                    ),
                }
            )
        if parameter_rows:
            st.dataframe(
                pd.DataFrame(parameter_rows),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(f"{copy['source']}: {row['source_path']}")

    @classmethod
    def _render_overlay_inventory(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        st.markdown(f"#### {copy['overlay_inventory']}")
        markets = sorted(
            {
                item.strip()
                for value in definitions["supported_markets"].fillna("")
                for item in str(value).split(",")
                if item.strip()
            }
        )
        frequencies = sorted(definitions["frequency"].dropna().unique())
        statuses = sorted(definitions["status"].dropna().unique())
        controls = st.columns(3)
        market = controls[0].selectbox(
            copy["market"],
            [copy["all"], *markets],
            key="overlay_library_market",
        )
        frequency = controls[1].selectbox(
            copy["frequency"],
            [copy["all"], *frequencies],
            format_func=lambda value: (
                value if value == copy["all"] else cls._plain_label(value)
            ),
            key="overlay_library_frequency",
        )
        status = controls[2].selectbox(
            copy["status"],
            [copy["all"], *statuses],
            format_func=lambda value: (
                value if value == copy["all"] else cls._status_label(value, lang)
            ),
            key="overlay_library_status",
        )
        filtered = definitions.copy()
        if market != copy["all"]:
            filtered = filtered.loc[
                filtered["supported_markets"].map(
                    lambda values: market
                    in {item.strip() for item in str(values).split(",")}
                )
            ]
        if frequency != copy["all"]:
            filtered = filtered.loc[filtered["frequency"].eq(frequency)]
        if status != copy["all"]:
            filtered = filtered.loc[filtered["status"].eq(status)]
        display = filtered[
            [
                "overlay_id",
                "name",
                "status",
                "scope",
                "decision_time",
                "effective_time",
                "supported_markets",
                "frequency",
            ]
        ].copy()
        display["status"] = display["status"].map(
            lambda value: cls._status_label(value, lang)
        )
        for column in ("scope", "decision_time", "effective_time", "frequency"):
            display[column] = display[column].map(cls._plain_label)
        display = display.rename(
            columns={
                "overlay_id": copy["overlay_id"],
                "name": copy["overlay_name"],
                "status": copy["status"],
                "scope": copy["scope"],
                "decision_time": copy["decision_time"],
                "effective_time": copy["effective_time"],
                "supported_markets": copy["market"],
                "frequency": copy["frequency"],
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True, height=300)

    @classmethod
    def _render_overlay_drilldown(
        cls,
        definitions: pd.DataFrame,
        copy: dict[str, str],
        lang: str,
    ) -> None:
        labels = {
            row.overlay_id: f"{row.overlay_id} | {row.name}"
            for row in definitions.itertuples(index=False)
        }
        overlay_id = st.selectbox(
            copy["select"],
            list(labels),
            format_func=labels.get,
            key="overlay_library_drilldown",
        )
        row = definitions.loc[definitions["overlay_id"].eq(overlay_id)].iloc[0]
        cards = st.columns(4)
        facts = (
            (copy["status"], cls._status_label(row["status"], lang)),
            (copy["scope"], cls._plain_label(row["scope"])),
            (copy["decision_time"], cls._plain_label(row["decision_time"])),
            (copy["effective_time"], cls._plain_label(row["effective_time"])),
        )
        for column, (label, value) in zip(cards, facts, strict=True):
            column.caption(label)
            column.markdown(f"**{value}**")
        st.markdown(f"#### {copy['rationale']}")
        st.write(row["economic_rationale"])
        st.markdown(f"#### {copy['limitations']}")
        st.write(row["known_limitations"] or copy["no_limitations"])
        st.markdown(f"#### {copy['contract']}")
        contract = pd.DataFrame(
            [
                {copy["field"]: key, copy["value"]: str(value)}
                for key, value in row["contract"].items()
            ]
        )
        st.dataframe(contract, use_container_width=True, hide_index=True)
        st.markdown(f"#### {copy['parameters']}")
        parameters = pd.DataFrame(
            [
                {copy["parameter"]: key, copy["default"]: str(value)}
                for key, value in row["parameters"].items()
            ]
        )
        if not parameters.empty:
            st.dataframe(parameters, use_container_width=True, hide_index=True)
        st.caption(f"{copy['source']}: {row['source']}")


__all__ = [
    "INVENTORY_FIELDS",
    "RouterLibrarySnapshot",
    "RouterLibraryView",
    "load_router_library_snapshot",
]
