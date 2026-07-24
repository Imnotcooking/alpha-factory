from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.research.validation_promotion import (  # noqa: E402
    DEFAULT_PROMOTION_POLICY_REGISTRY,
    audit_validation_promotion,
    write_validation_promotion_readiness,
)


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "runtime/artifacts/research/validation_promotion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Phase 10 validation and promotion decisions."
    )
    parser.add_argument("--review-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--policy-path", type=Path, default=DEFAULT_PROMOTION_POLICY_REGISTRY
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, ledger, policies = audit_validation_promotion(
        args.review_root, args.policy_path
    )
    destination = write_validation_promotion_readiness(
        summary, ledger, policies, args.review_root
    )
    print(f"Phase 10 status: {summary['status']}")
    print(f"Promotion reviews: {summary['review_count']}")
    print(f"Paper eligible: {summary['paper_eligible_count']}")
    print(
        "Production-review eligible: "
        f"{summary['production_review_eligible_count']}"
    )
    print(f"Failed research results: {summary['failed_research_result_count']}")
    print(f"Artifacts: {destination}")
    return 0 if summary["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
