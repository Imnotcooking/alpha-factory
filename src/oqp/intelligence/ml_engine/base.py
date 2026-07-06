"""Base contracts for model inference engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from oqp.intelligence.base import BaseEngine


@dataclass(frozen=True, slots=True)
class ModelInferenceRequest:
    model_id: str
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelInferenceResult:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BaseMLEngine(BaseEngine):
    category = "ml_engine"
