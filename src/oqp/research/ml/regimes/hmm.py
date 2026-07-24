"""Compatibility import for legacy :mod:`hmmlearn` market wrappers."""

from oqp.research.ml.regimes.legacy.hmmlearn_models import (
    MarketGMMHMM,
    MarketHMM,
)

__all__ = ["MarketGMMHMM", "MarketHMM"]
