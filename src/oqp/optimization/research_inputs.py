"""Validation helpers for frozen inputs consumed by optimization studies."""

from __future__ import annotations

from typing import Any


def require_dataset_fingerprint(data: Any) -> str:
    attrs = getattr(data, "attrs", {})
    fingerprint = str(attrs.get("dataset_fingerprint") or "").strip()
    if not fingerprint:
        raise ValueError(
            "Optimization requires data.attrs['dataset_fingerprint']; "
            "register or load the dataset through the research data layer first"
        )
    return fingerprint


__all__ = ["require_dataset_fingerprint"]
