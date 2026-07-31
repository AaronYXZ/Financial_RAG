"""Validation for evaluation artifact manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eligibility_manifest(path: Path) -> dict[str, Any]:
    """Load an eligibility manifest and validate its denominator."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "track",
        "model_id",
        "eligible_case_count",
        "eligible_case_ids",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Eligibility manifest is missing keys: {missing}")
    case_ids = payload["eligible_case_ids"]
    if not isinstance(case_ids, list) or not all(
        isinstance(item, str) for item in case_ids
    ):
        raise ValueError("eligible_case_ids must be a list of strings")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Eligibility manifest contains duplicate case IDs")
    if payload["eligible_case_count"] != len(case_ids):
        raise ValueError("eligible_case_count does not match eligible_case_ids")
    return payload
