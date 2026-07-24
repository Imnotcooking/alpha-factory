from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.research.strategy_routing import (  # noqa: E402
    audit_router_readiness,
    write_router_readiness_snapshot,
)


DEFAULT_PHASE4_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "standalone_sleeve_tests"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "router_hypotheses"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the research library can enter Phase 6."
    )
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--frozen-hypothesis-count",
        type=int,
        default=0,
        help="Count only hypotheses dated before a still-untouched holdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, sleeves = audit_router_readiness(
        args.phase4_root,
        frozen_hypothesis_count=args.frozen_hypothesis_count,
    )
    destination = write_router_readiness_snapshot(
        summary, sleeves, args.output_root
    )
    print(f"Phase 6 readiness: {summary['status']}")
    print(f"Standalone sleeves: {summary['standalone_sleeves']}")
    print(f"Eligible sleeves: {summary['eligible_sleeves']}")
    print(f"Eligible pairs: {summary['eligible_pairs']}")
    print(f"Frozen hypotheses: {summary['frozen_hypotheses']}")
    for blocker in summary["blockers"]:
        print(f"BLOCKED: {blocker}")
    print(f"Artifacts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
