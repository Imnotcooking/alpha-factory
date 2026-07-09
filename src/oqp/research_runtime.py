"""Runtime path contracts for local alpha research workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from oqp.config import REPO_ROOT
from oqp.data.runtime_paths import default_futures_cn_index_daily_file


@dataclass(frozen=True, slots=True)
class AlphaResearchRuntimePaths:
    data_root: Path
    artifact_root: Path
    db_path: Path

    @property
    def default_daily_data_file(self) -> Path:
        return default_futures_cn_index_daily_file()

    @property
    def feature_matrix_path(self) -> Path:
        return self.data_root / "feature_store" / "ML_Feature_Matrix.parquet"

    @property
    def xgboost_model_output_path(self) -> Path:
        return self.artifact_root / "models" / "xgb_base_model.json"

    @property
    def xgboost_feature_importance_path(self) -> Path:
        return (
            self.artifact_root
            / "feature_importance"
            / "feature_importance_fac_054.csv"
        )


def alpha_research_runtime_paths() -> AlphaResearchRuntimePaths:
    return AlphaResearchRuntimePaths(
        data_root=Path(
            os.environ.get(
                "ALPHA_RUNTIME_DATA_ROOT",
                REPO_ROOT / "runtime" / "data",
            )
        ),
        artifact_root=Path(
            os.environ.get(
                "ALPHA_RUNTIME_ARTIFACT_ROOT",
                REPO_ROOT / "runtime" / "artifacts" / "research",
            )
        ),
        db_path=Path(
            os.environ.get(
                "ALPHA_RESEARCH_DB_PATH",
                REPO_ROOT
                / "runtime"
                / "db"
                / "research"
                / "research_memory.db",
            )
        ),
    )
