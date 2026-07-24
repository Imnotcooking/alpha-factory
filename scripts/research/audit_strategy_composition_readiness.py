from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.research.strategy_composition import (  # noqa: E402
    audit_strategy_composition_readiness,
    write_strategy_composition_readiness,
)


DEFAULT_ROUTER_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "router_hypotheses"
)
DEFAULT_RECIPE_ROOT = REPO_ROOT / "departments" / "research" / "strategies" / "compositions"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "strategy_composition"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether frozen components can enter Phase 7."
    )
    parser.add_argument("--router-root", type=Path, default=DEFAULT_ROUTER_ROOT)
    parser.add_argument("--recipe-root", type=Path, default=DEFAULT_RECIPE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, components = audit_strategy_composition_readiness(
        args.router_root, args.recipe_root
    )
    destination = write_strategy_composition_readiness(
        summary, components, args.output_root
    )
    print(f"Phase 7 readiness: {summary['status']}")
    print(f"Eligible Phase 6 routers: {summary['eligible_phase6_routers']}")
    print(f"Declared strategy recipes: {summary['declared_strategy_recipes']}")
    print(f"Admissible strategy recipes: {summary['admissible_strategy_recipes']}")
    for blocker in summary["blockers"]:
        print(f"BLOCKED: {blocker}")
    print(f"Artifacts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
