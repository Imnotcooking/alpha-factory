from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.research.optional_optimization import (  # noqa: E402
    audit_phase8_readiness,
    write_phase8_readiness,
)


DEFAULT_CONFIG_ROOT = REPO_ROOT / "departments/research/optimization_studies"
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "runtime/artifacts/research/optional_optimization"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 8 optimization state.")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, studies = audit_phase8_readiness(
        args.config_root, args.artifact_root
    )
    destination = write_phase8_readiness(summary, studies, args.artifact_root)
    print(f"Phase 8 status: {summary['status']}")
    print(f"Declared studies: {summary['declared_studies']}")
    print(f"Enabled studies: {summary['enabled_studies']}")
    print(f"Completed searches: {summary['completed_searches']}")
    print(f"Frozen candidates: {summary['frozen_candidates']}")
    print(f"Final holdout evaluations: {summary['final_holdout_evaluations']}")
    print(f"Artifacts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
