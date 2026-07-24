"""Native runtime probes for optional tree-based regression adapters."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from oqp.research.ml.tree_based.factory import MLModelFactory


_PROBES = {
    "lightgbm": """
import tempfile
from pathlib import Path
import pandas as pd
from oqp.research.ml.regression.base import ValidationConfig
from oqp.research.ml.tree_based.lightgbm import LGBMModelConfig, LightGBMRegressorTrainer
dates = pd.date_range('2026-01-01', periods=14, freq='B')
frame = pd.DataFrame([
    {'date': date, 'ticker': str(asset), 'f_a': float(day + asset),
     'f_b': float((day + 1) * (asset + 1)), 'target_1d_rank': float(asset)}
    for day, date in enumerate(dates) for asset in range(4)
])
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'probe.parquet'
    frame.to_parquet(path)
    config = LGBMModelConfig(
        validation=ValidationConfig(min_train_days=8, test_window_days=3, purge_gap_days=1),
        params={'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'num_threads': 1},
        num_boost_round=5,
        early_stopping_rounds=2,
    )
    LightGBMRegressorTrainer(path, config=config).train()
""",
    "xgboost": """
import numpy as np
from xgboost import XGBRegressor
x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, -0.5]], dtype=np.float64)
y = np.array([0.0, 1.0, 0.5, 1.5], dtype=np.float64)
XGBRegressor(n_estimators=1, max_depth=1, n_jobs=1).fit(x, y)
""",
}


@dataclass(frozen=True, slots=True)
class ModelRuntimeStatus:
    model_type: str
    available: bool
    detail: str
    returncode: int | None = None


def probe_model_runtime(
    model_type: str,
    *,
    timeout_seconds: float = 20.0,
) -> ModelRuntimeStatus:
    """Probe native model code in a subprocess so a segfault cannot kill the CLI."""

    canonical = MLModelFactory.normalize_model_type(model_type)
    probe = _PROBES[canonical]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ModelRuntimeStatus(
            model_type=canonical,
            available=False,
            detail=f"native runtime probe exceeded {timeout_seconds:g}s",
        )

    if completed.returncode == 0:
        return ModelRuntimeStatus(
            model_type=canonical,
            available=True,
            detail="native runtime probe passed",
            returncode=0,
        )

    detail = (completed.stderr or completed.stdout).strip().splitlines()
    reason = detail[-1] if detail else "native runtime exited unexpectedly"
    if completed.returncode in {-11, 139}:
        reason = "native runtime crashed with a segmentation fault"
    return ModelRuntimeStatus(
        model_type=canonical,
        available=False,
        detail=reason,
        returncode=completed.returncode,
    )


def require_model_runtime(model_type: str) -> ModelRuntimeStatus:
    status = probe_model_runtime(model_type)
    if not status.available:
        raise RuntimeError(
            f"{status.model_type} is installed but not runnable: {status.detail}. "
            "Repair the research Python environment before retraining this adapter."
        )
    return status


__all__ = ["ModelRuntimeStatus", "probe_model_runtime", "require_model_runtime"]
