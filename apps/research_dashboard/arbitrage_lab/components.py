from __future__ import annotations

import math

import pandas as pd
import streamlit as st


ARBITRAGE_TYPE_LABELS_ZH = {
    "calendar": "跨期",
    "cross_product": "跨品种",
    "statistical": "统计套利",
}

INTERPRETATION_LABELS_ZH = {
    "Insufficient data": "数据不足",
    "Watchlist: relationship may be valid, but current dislocation is mild": "观察名单：关系可能有效，但当前偏离较温和",
    "Fragile: dislocated, but mean reversion is weak or unproven": "脆弱：已有偏离，但均值回复较弱或尚未验证",
    "Fragile: hedge ratio is moving too much": "脆弱：对冲比率变化过大",
    "Promising for research: strong dislocation with acceptable stability": "可进一步研究：偏离较强，稳定性尚可",
    "Worth reviewing: some evidence, needs drilldown and cost checks": "值得复核：有初步证据，需要钻取和成本检查",
    "Low priority: weak combined evidence": "低优先级：综合证据偏弱",
}

CANDIDATE_TABLE_COLUMNS = {
    "EN": {
        "rank": "rank",
        "candidate_id": "candidate",
        "arbitrage_type": "type",
        "sector_pair": "sector/family",
        "opportunity_score": "score",
        "latest_z": "latest z",
        "correlation": "correlation",
        "half_life": "half-life",
        "beta_drift": "beta drift",
        "round_turn_cost_bps": "est. cost bps",
        "interpretation": "interpretation",
    },
    "ZH": {
        "rank": "排名",
        "candidate_id": "候选组合",
        "arbitrage_type": "类型代码",
        "arbitrage_type_label": "类型",
        "sector_pair": "板块/族群",
        "opportunity_score": "分数",
        "latest_z": "最新 z",
        "correlation": "相关性",
        "half_life": "半衰期",
        "beta_drift": "beta 漂移",
        "round_turn_cost_bps": "估算成本 bps",
        "interpretation": "解读",
    },
}


def render_explainer(title: str, body: str, *, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.markdown(body)


def render_interpretation(text: str, *, lang: str = "EN") -> None:
    display_text = translate_interpretation(text, lang=lang)
    lowered = str(text).lower()
    if "promising" in lowered:
        st.success(display_text)
    elif "fragile" in lowered:
        st.warning(display_text)
    elif "watchlist" in lowered or "worth reviewing" in lowered:
        st.info(display_text)
    else:
        st.caption(display_text)


def fmt_num(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "Inf"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}%}"


def fmt_int(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def candidate_display_table(candidates: pd.DataFrame, limit: int = 50, *, lang: str = "EN") -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    cols = [
        "rank",
        "candidate_id",
        "arbitrage_type",
        "sector_pair",
        "opportunity_score",
        "latest_z",
        "correlation",
        "half_life",
        "beta_drift",
        "round_turn_cost_bps",
        "interpretation",
    ]
    out = candidates[[col for col in cols if col in candidates.columns]].head(limit).copy()
    normalized_lang = "ZH" if str(lang).upper() == "ZH" else "EN"
    if normalized_lang == "ZH":
        if "arbitrage_type" in out.columns:
            insert_at = out.columns.get_loc("arbitrage_type") + 1
            out.insert(
                insert_at,
                "arbitrage_type_label",
                out["arbitrage_type"].map(translate_arbitrage_type),
            )
        if "interpretation" in out.columns:
            out["interpretation"] = out["interpretation"].map(
                lambda value: translate_interpretation(value, lang=normalized_lang)
            )
    rename = CANDIDATE_TABLE_COLUMNS[normalized_lang]
    return out.rename(columns=rename)


def metric_with_caption(label: str, value: str, caption: str) -> None:
    st.metric(label, value)
    st.caption(caption)


def translate_arbitrage_type(value: object, *, lang: str = "ZH") -> str:
    text = str(value or "")
    if str(lang).upper() != "ZH":
        return text
    return ARBITRAGE_TYPE_LABELS_ZH.get(text, text)


def translate_interpretation(value: object, *, lang: str = "ZH") -> str:
    text = str(value or "")
    if str(lang).upper() != "ZH":
        return text
    return INTERPRETATION_LABELS_ZH.get(text, text)
