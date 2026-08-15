"""Fail-closed public-data clearance checks for AITA source material."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any, Iterable


AITA_DATA_CLEARANCE_SCHEMA_VERSION = "aita-data-clearance-v1"
AITA_SOURCE_PREFIX = "aita-bench/data/"
AITA_SOURCE_EXCLUSIONS = {"aita-bench/data/README.md"}
REQUIRED_FIELDS = (
    "reviewer_identity",
    "decision_date",
    "covered_path_patterns",
    "source_provenance",
    "privacy_review",
    "terms_policy_basis",
    "transformation_redaction_notes",
    "evidence_references",
)


class DataClearanceError(ValueError):
    """Raised when a source tree cannot be represented as cleared public data."""


def tracked_aita_source_paths(paths: Iterable[str]) -> list[str]:
    return sorted(
        normalized
        for path in paths
        if (normalized := path.replace("\\", "/")).startswith(AITA_SOURCE_PREFIX)
        and normalized not in AITA_SOURCE_EXCLUSIONS
    )


def _require_schema(record: dict[str, Any]) -> None:
    if record.get("schema_version") != AITA_DATA_CLEARANCE_SCHEMA_VERSION:
        raise DataClearanceError("AITA data clearance schema is missing or unsupported")
    status = record.get("status")
    if status not in {"not_cleared", "cleared", "excluded"}:
        raise DataClearanceError("AITA data clearance status is invalid")
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise DataClearanceError("AITA data clearance fields missing: " + ", ".join(missing))
    patterns = record.get("covered_path_patterns")
    references = record.get("evidence_references")
    if not isinstance(patterns, list) or not all(isinstance(item, str) and item for item in patterns):
        raise DataClearanceError("covered_path_patterns must be a nonempty string list")
    if not isinstance(references, list) or not all(isinstance(item, str) and item for item in references):
        raise DataClearanceError("evidence_references must be a nonempty string list")


def validate_public_data_clearance(
    record: dict[str, Any],
    exported_aita_paths: Iterable[str],
) -> None:
    """Validate a human-authored clearance record against exported source paths."""
    if not isinstance(record, dict):
        raise DataClearanceError("AITA data clearance record must be an object")
    _require_schema(record)
    paths = tracked_aita_source_paths(exported_aita_paths)
    status = record["status"]
    if status == "excluded":
        if paths:
            raise DataClearanceError("AITA excluded data remains in the source export")
        return
    if paths and status != "cleared":
        raise DataClearanceError("AITA source material is not cleared for public release")
    if not paths:
        return

    patterns = record["covered_path_patterns"]
    uncovered = [path for path in paths if not any(fnmatch(path, pattern) for pattern in patterns)]
    if uncovered:
        raise DataClearanceError("AITA source path is not covered by the clearance record")
    for field in (
        "reviewer_identity",
        "decision_date",
        "source_provenance",
        "privacy_review",
        "terms_policy_basis",
        "transformation_redaction_notes",
    ):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip() or "pending" in value.lower():
            raise DataClearanceError(f"cleared AITA data requires completed {field}")
    if not record["reviewer_identity"].startswith("human:"):
        raise DataClearanceError("cleared AITA data requires human: reviewer_identity")

