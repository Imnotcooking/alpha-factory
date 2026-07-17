"""Broker-free demo profile and deterministic onboarding fixtures."""

from oqp.demo.doctor import DoctorCheck, run_doctor
from oqp.demo.profile import DemoPaths, demo_environment, demo_paths, read_profile_marker
from oqp.demo.seed import DemoSeedResult, seed_demo_profile

__all__ = [
    "DemoPaths",
    "DemoSeedResult",
    "DoctorCheck",
    "demo_environment",
    "demo_paths",
    "read_profile_marker",
    "run_doctor",
    "seed_demo_profile",
]
