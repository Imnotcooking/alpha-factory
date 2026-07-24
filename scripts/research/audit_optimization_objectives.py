from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.research.optimization_objectives import (  # noqa: E402
    DEFAULT_OBJECTIVE_REGISTRY,
    audit_optimization_objectives,
    write_optimization_objective_readiness,
)


DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "runtime/artifacts/research/optimization_objectives"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Phase 9 layer-specific optimisation objectives."
    )
    parser.add_argument(
        "--registry-path", type=Path, default=DEFAULT_OBJECTIVE_REGISTRY
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_optimization_objectives(args.registry_path)
    summary, profiles, objectives, constraints, upstream = result
    destination = write_optimization_objective_readiness(
        summary,
        profiles,
        objectives,
        constraints,
        upstream,
        args.artifact_root,
    )
    print(f"Phase 9 status: {summary['status']}")
    print(f"Active profiles: {summary['active_profiles']}")
    print(f"Objectives: {summary['objective_count']}")
    print(f"Hard constraints: {summary['hard_constraint_count']}")
    print(f"Artifacts: {destination}")
    return 0 if summary["status"] == "active" else 1


if __name__ == "__main__":
    raise SystemExit(main())
