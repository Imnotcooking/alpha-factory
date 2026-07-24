"""Unified model-family inventory for the ML Hub."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from oqp.research.ml.catalog import (
    ModelCategory,
    research_experiment_catalog,
    research_model_catalog,
)


_CATEGORY_LABELS = {
    "EN": {
        ModelCategory.SUPERVISED_PREDICTOR: "Supervised prediction",
        ModelCategory.REGIME_ESTIMATOR: "Regime inference",
        ModelCategory.LATENT_REPRESENTATION: "Latent representation",
        ModelCategory.STATE_SPACE_ESTIMATOR: "State-space estimation",
        ModelCategory.STUDY_CONTROL: "Study control",
    },
    "ZH": {
        ModelCategory.SUPERVISED_PREDICTOR: "监督式预测",
        ModelCategory.REGIME_ESTIMATOR: "市场状态推断",
        ModelCategory.LATENT_REPRESENTATION: "潜在表征学习",
        ModelCategory.STATE_SPACE_ESTIMATOR: "状态空间估计",
        ModelCategory.STUDY_CONTROL: "研究对照组",
    },
}


def model_catalog_frame(language: str = "EN") -> pd.DataFrame:
    """Return a dashboard-ready implementation inventory."""

    labels = _CATEGORY_LABELS.get(language, _CATEGORY_LABELS["EN"])
    return pd.DataFrame(
        [
            {
                "model_key": item.model_key,
                "model": item.display_name,
                "ml_family": labels[item.category],
                "learning": item.learning_paradigm.value,
                "task": item.task,
                "target_required": item.requires_target,
                "input_geometry": item.input_geometry,
                "output": item.output_contract,
                "scope": item.scope.value,
                "implementation": item.implementation_path,
            }
            for item in research_model_catalog()
        ]
    )


def experiment_design_frame() -> pd.DataFrame:
    """Return registered designs separately from the executed-run ledger."""

    return pd.DataFrame(
        [
            {
                "experiment": item.display_name,
                "design_status": item.design_status,
                "empirical_status": item.empirical_status,
                "primary_states": item.primary_state_count,
                "primary_metric": item.primary_metric,
                "model_count": len(item.model_keys),
                "implementation": item.implementation_path,
            }
            for item in research_experiment_catalog()
        ]
    )


def render_model_catalog_panel(copy: dict[str, str], *, language: str) -> None:
    """Render implementation and design inventories without claiming results."""

    st.markdown(f"#### {copy['model_library_title']}")
    st.caption(copy["model_library_caption"])

    catalog = model_catalog_frame(language)
    metrics = st.columns(5)
    metrics[0].metric(copy["model_library_implementations"], f"{len(catalog):,}")
    metrics[1].metric(
        copy["model_library_supervised"],
        f"{sum(item.category is ModelCategory.SUPERVISED_PREDICTOR for item in research_model_catalog()):,}",
    )
    metrics[2].metric(
        copy["model_library_regimes"],
        f"{sum(item.category is ModelCategory.REGIME_ESTIMATOR for item in research_model_catalog()):,}",
    )
    metrics[3].metric(
        copy["model_library_latent"],
        f"{sum(item.category is ModelCategory.LATENT_REPRESENTATION for item in research_model_catalog()):,}",
    )
    metrics[4].metric(
        copy["model_library_state_space"],
        f"{sum(item.category is ModelCategory.STATE_SPACE_ESTIMATOR for item in research_model_catalog()):,}",
    )

    family_options = [copy["model_library_all"], *catalog["ml_family"].unique()]
    selected_family = st.selectbox(
        copy["model_library_filter"],
        family_options,
        key="ml_hub_model_library_family",
    )
    visible = catalog
    if selected_family != copy["model_library_all"]:
        visible = catalog[catalog["ml_family"] == selected_family]
    st.dataframe(
        visible[
            [
                "model",
                "ml_family",
                "learning",
                "task",
                "target_required",
                "scope",
                "implementation",
            ]
        ],
        hide_index=True,
    )

    with st.expander(copy["model_taxonomy_title"], expanded=False):
        st.markdown(copy["model_taxonomy_help"])

    st.markdown(f"#### {copy['experiment_designs_title']}")
    st.caption(copy["experiment_designs_caption"])
    designs = experiment_design_frame()
    st.dataframe(designs, hide_index=True)
    for design in research_experiment_catalog():
        with st.expander(
            f"{design.display_name} — {design.design_status}",
            expanded=False,
        ):
            st.markdown(
                f"**{copy['experiment_empirical_status']}:** "
                f"`{design.empirical_status}`"
            )
            st.markdown(
                f"**{copy['experiment_primary_metric']}:** {design.primary_metric}"
            )
            st.markdown(f"**{copy['experiment_comparisons']}:**")
            for comparison in design.comparisons:
                st.markdown(f"- {comparison}")


__all__ = [
    "experiment_design_frame",
    "model_catalog_frame",
    "render_model_catalog_panel",
]
