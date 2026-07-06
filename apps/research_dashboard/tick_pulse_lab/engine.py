"""Compatibility wrapper for promoted tick-pulse hypothesis engine helpers."""

from oqp.research.tick_pulse.engine import *  # noqa: F401,F403
from oqp.research.tick_pulse.engine import (  # noqa: F401
    _build_research_sweep,
    _evaluate_horizon_summary,
    _feature_group_keys,
    _is_bearish_hypothesis,
    _is_relative_velocity_fade_hypothesis,
    _is_relative_velocity_hypothesis,
    _pct_text,
)
