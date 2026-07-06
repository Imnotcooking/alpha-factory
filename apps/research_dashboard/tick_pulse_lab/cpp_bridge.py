"""Compatibility wrapper for promoted native tick-pulse evaluators."""

from oqp.research.tick_pulse.cpp_bridge import (  # noqa: F401
    compute_heuristic_cpp,
    compute_relative_velocity_cpp,
)
