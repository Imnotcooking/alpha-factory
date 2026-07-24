from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / "evidence/public_evidence.json"


def test_public_evidence_contains_only_aggregate_allowlisted_sections() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert set(payload) == {
        "version",
        "evidence_status",
        "claim_status",
        "sample",
        "design",
        "gross_performance",
        "conditional_returns_pct",
        "proxy_stability",
        "q4_influence",
        "cross_sectional_anatomy",
        "robustness",
    }
    assert payload["design"]["threshold_search_performed"] is False
    assert payload["design"]["universe_selection_performed"] is False
    assert len(payload["conditional_returns_pct"]["primary_proxy"]) == 4
    assert len(payload["conditional_returns_pct"]["equal_product_proxy"]) == 4


def test_public_project_does_not_reference_private_research_surfaces() -> None:
    forbidden = [
        "runtime" + "/",
        "departments/research/" + "factors",
        ".par" + "quet",
        "target" + "_weight",
        "fee" + "_open",
        "fee" + "_close",
        "instrument" + "master",
        "2026-" + "03",
    ]
    text_suffixes = {".md", ".json", ".py", ".tex", ".bib", ".yaml", ".yml"}
    for path in PROJECT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in content, f"private token {token!r} found in {path}"
        assert re.search(r"[a-f0-9]{64}", content) is None, f"source hash found in {path}"


def test_public_figure_builder_reads_only_public_evidence() -> None:
    source = (PROJECT / "scripts/build_public_figures.py").read_text(encoding="utf-8")
    assert "public_evidence.json" in source
    assert "read_parquet" not in source
    assert "read_csv" not in source
    assert "departments.research" not in source
