from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


COPY = {
    "EN": {
        "title": "Validation & Promotion (Phase 10)",
        "boundary": (
            "Full-sample Sharpe is not a promotion gate. Advancement requires "
            "chronological validation, one frozen holdout, robust and reproducible "
            "incremental value, then paper trading before production review."
        ),
        "status": "Phase 10 status",
        "reviews": "Reviews",
        "paper": "Paper eligible",
        "production": "Production review",
        "failed": "Failed results",
        "lifecycle": "Research lifecycle",
        "policy": "Frozen router promotion policy",
        "ledger": "Promotion decision ledger",
        "empty": (
            "No router has reached Phase 10 yet. The gate is ready and is waiting "
            "for a frozen Phase 6/8 evidence bundle and perturbation plan."
        ),
        "failure": (
            "Negative economic results remain in the ledger; they are not deleted "
            "or relabelled as missing evidence."
        ),
        "missing": "Phase 10 readiness artifacts have not been generated yet.",
        "lifecycle_steps": [
            "Discovery",
            "Chronological validation",
            "Frozen holdout",
            "Paper trading",
            "Production review",
        ],
    },
    "ZH": {
        "title": "验证与晋级（第十阶段）",
        "boundary": (
            "全样本夏普不是晋级门槛。晋级必须依次通过时序验证、一次冻结留出集、"
            "稳健且可复现的增量价值检验，再进入模拟盘，最后才是生产评审。"
        ),
        "status": "第十阶段状态",
        "reviews": "晋级评审数",
        "paper": "可进入模拟盘",
        "production": "可进入生产评审",
        "failed": "失败研究结果",
        "lifecycle": "研究生命周期",
        "policy": "冻结路由晋级规则",
        "ledger": "晋级决策台账",
        "empty": (
            "目前尚无路由进入第十阶段。门槛已就绪，正在等待冻结的第六/第八阶段"
            "证据包和参数扰动方案。"
        ),
        "failure": "经济检验失败的结果会保留在台账中，不会被删除或改写为证据缺失。",
        "missing": "尚未生成第十阶段审计文件。",
        "lifecycle_steps": [
            "研究发现",
            "时序验证",
            "冻结留出集",
            "模拟盘",
            "生产评审",
        ],
    },
}


@st.cache_data(show_spinner=False)
def load_validation_promotion_review(
    artifact_root: str,
) -> dict[str, Any] | None:
    root = Path(artifact_root).expanduser().resolve() / "validation_promotion"
    paths = {
        "readiness": root / "readiness.json",
        "ledger": root / "promotion_ledger.csv",
        "policy": root / "promotion_policy.csv",
    }
    if any(not path.exists() for path in paths.values()):
        return None
    return {
        "readiness": json.loads(paths["readiness"].read_text(encoding="utf-8")),
        "ledger": pd.read_csv(paths["ledger"]),
        "policy": pd.read_csv(paths["policy"]),
    }


def render_validation_promotion_panel(
    artifact_root: str | Path,
    *,
    lang: str = "EN",
) -> bool:
    copy = COPY.get(lang, COPY["EN"])
    snapshot = load_validation_promotion_review(str(artifact_root))
    st.markdown(f"### {copy['title']}")
    st.info(copy["boundary"])
    if snapshot is None:
        st.warning(copy["missing"])
        return False

    readiness = snapshot["readiness"]
    status_labels = {
        "EN": {"awaiting_evidence": "WAITING", "attention_required": "ATTENTION"},
        "ZH": {"awaiting_evidence": "等待证据", "attention_required": "需要关注"},
    }
    status_value = status_labels.get(lang, status_labels["EN"]).get(
        str(readiness["status"]), str(readiness["status"]).upper()
    )
    first_row = st.columns(3)
    first_row[0].metric(copy["status"], status_value)
    first_row[1].metric(copy["reviews"], readiness["review_count"])
    first_row[2].metric(copy["failed"], readiness["failed_research_result_count"])
    second_row = st.columns(2)
    second_row[0].metric(copy["paper"], readiness["paper_eligible_count"])
    second_row[1].metric(
        copy["production"], readiness["production_review_eligible_count"]
    )

    st.markdown(f"#### {copy['lifecycle']}")
    st.markdown(" &rarr; ".join(copy["lifecycle_steps"]))
    st.caption(copy["failure"])

    ledger = snapshot["ledger"]
    if ledger.empty:
        st.success(copy["empty"])
    else:
        st.markdown(f"#### {copy['ledger']}")
        st.dataframe(ledger, use_container_width=True, hide_index=True)

    st.markdown(f"#### {copy['policy']}")
    policy = snapshot["policy"]
    visible = policy.loc[
        ~policy["parameter"].isin(["date_col", "product_col", "split_col"])
    ]
    st.dataframe(
        visible[["parameter", "value", "profile_id"]],
        use_container_width=True,
        hide_index=True,
    )
    return True


__all__ = [
    "load_validation_promotion_review",
    "render_validation_promotion_panel",
]
