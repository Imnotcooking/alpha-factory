"""Runtime path contracts for the daily latent-regime paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oqp.research.artifacts import slugify
from oqp.research_runtime import AlphaResearchRuntimePaths, alpha_research_runtime_paths


VALID_RUN_MODES = ("synthetic", "synthetic_smoke", "validation", "holdout")


@dataclass(frozen=True, slots=True)
class DailyRegimeRuntimePaths:
    artifact_root: Path
    data_root: Path

    @property
    def datasets_dir(self) -> Path:
        return self.artifact_root / "datasets"

    @property
    def folds_dir(self) -> Path:
        return self.artifact_root / "folds"

    @property
    def models_dir(self) -> Path:
        return self.artifact_root / "models"

    @property
    def synthetic_dir(self) -> Path:
        return self.artifact_root / "synthetic"

    @property
    def validation_dir(self) -> Path:
        return self.artifact_root / "validation"

    @property
    def holdout_dir(self) -> Path:
        return self.artifact_root / "holdout"

    @property
    def trial_ledgers_dir(self) -> Path:
        return self.artifact_root / "trial_ledgers"

    @property
    def manifests_dir(self) -> Path:
        return self.artifact_root / "manifests"

    @property
    def generated_data_dir(self) -> Path:
        return self.data_root / "daily_latent_regimes"

    def mode_dir(self, mode: str) -> Path:
        normalized = str(mode).strip().lower()
        if normalized not in VALID_RUN_MODES:
            raise ValueError(
                f"Unknown daily-regime run mode {mode!r}; expected one of {VALID_RUN_MODES}."
            )
        return {
            "synthetic": self.synthetic_dir,
            "synthetic_smoke": self.synthetic_dir,
            "validation": self.validation_dir,
            "holdout": self.holdout_dir,
        }[normalized]

    def run_dir(self, mode: str, run_id: str) -> Path:
        normalized_id = slugify(run_id, fallback="run")
        return self.mode_dir(mode) / normalized_id

    def ensure_directories(self, *, include_holdout: bool = False) -> tuple[Path, ...]:
        """Create runtime directories only when explicitly requested."""

        paths = [
            self.artifact_root,
            self.datasets_dir,
            self.folds_dir,
            self.models_dir,
            self.synthetic_dir,
            self.validation_dir,
            self.trial_ledgers_dir,
            self.manifests_dir,
            self.generated_data_dir,
        ]
        if include_holdout:
            paths.append(self.holdout_dir)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return tuple(paths)


def daily_regime_runtime_paths(
    *,
    base_paths: AlphaResearchRuntimePaths | None = None,
    artifact_root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> DailyRegimeRuntimePaths:
    """Resolve paper paths without creating files or directories."""

    base = base_paths or alpha_research_runtime_paths()
    resolved_artifact_root = (
        Path(artifact_root).expanduser()
        if artifact_root is not None
        else base.artifact_root / "daily_latent_regimes"
    )
    resolved_data_root = (
        Path(data_root).expanduser() if data_root is not None else base.data_root
    )
    return DailyRegimeRuntimePaths(
        artifact_root=resolved_artifact_root,
        data_root=resolved_data_root,
    )


__all__ = [
    "DailyRegimeRuntimePaths",
    "VALID_RUN_MODES",
    "daily_regime_runtime_paths",
]
