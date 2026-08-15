import json
from pathlib import Path

import pytest

from suite_tools.data_clearance import (
    AITA_DATA_CLEARANCE_SCHEMA_VERSION,
    DataClearanceError,
    tracked_aita_source_paths,
    validate_public_data_clearance,
)


ROOT = Path(__file__).resolve().parents[1]


def _record(status="not_cleared"):
    return {
        "schema_version": AITA_DATA_CLEARANCE_SCHEMA_VERSION,
        "status": status,
        "reviewer_identity": "pending-human-review",
        "decision_date": None,
        "covered_path_patterns": [
            "aita-bench/data/AITA-YTA_sample.csv",
            "aita-bench/data/curated/**",
        ],
        "source_provenance": "Reddit-derived AITA source material; see dataset card.",
        "privacy_review": "pending",
        "terms_policy_basis": "pending human review",
        "transformation_redaction_notes": "Construct reversal; no release clearance asserted.",
        "evidence_references": ["docs/DATA_RIGHTS_AND_PRIVACY.md"],
    }


def test_source_release_refuses_uncleared_aita_data():
    with pytest.raises(DataClearanceError, match="not cleared"):
        validate_public_data_clearance(
            _record(),
            ["aita-bench/data/curated/example.csv"],
        )


def test_clearance_record_must_cover_every_exported_aita_source_path():
    record = _record("cleared")
    record["covered_path_patterns"] = ["aita-bench/data/AITA-YTA_sample.csv"]
    with pytest.raises(DataClearanceError, match="not covered"):
        validate_public_data_clearance(
            record,
            ["aita-bench/data/curated/example.csv"],
        )


def test_cleared_record_requires_completed_human_decision_fields():
    record = _record("cleared")
    with pytest.raises(DataClearanceError, match="requires completed"):
        validate_public_data_clearance(
            record,
            ["aita-bench/data/curated/example.csv"],
        )
    record["decision_date"] = "2026-08-13"
    record["reviewer_identity"] = "human:reviewer-id"
    record["privacy_review"] = "completed; no direct identifiers approved"
    record["terms_policy_basis"] = "human-authored basis reference"
    validate_public_data_clearance(
        record,
        ["aita-bench/data/curated/example.csv"],
    )


def test_excluded_status_requires_all_covered_data_absent():
    with pytest.raises(DataClearanceError, match="excluded data remains"):
        validate_public_data_clearance(
            _record("excluded"),
            ["aita-bench/data/curated/example.csv"],
        )
    validate_public_data_clearance(_record("excluded"), [])


def test_tracked_aita_paths_are_covered_by_repository_clearance_record():
    record = json.loads((ROOT / "manifests" / "aita-data-clearance.json").read_text())
    paths = tracked_aita_source_paths(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
    )
    assert paths
    assert record["status"] == "cleared"
    validate_public_data_clearance(record, paths)
