import pytest

from suite_tools.artifact_identity import (
    ArtifactIdentityError,
    evaluate_run_artifact_identity,
    reconcile_condition_identity,
)
from suite_tools.run_contract import IDENTITY_PROJECTION_VERSION


MODEL = {
    "provider_api": "openai_compatible",
    "condition_id": "model-high",
    "condition_hash": "sha256:abc",
    "condition_metadata": {"effort": "high"},
    "request_options": {"reasoning_effort": "high"},
}


def test_reconcile_restores_only_missing_identity_from_rendered_condition():
    artifact = {"condition_id": "model-high"}

    restored = reconcile_condition_identity(
        artifact,
        MODEL,
        context="unit-1",
        restore_missing=True,
    )

    assert set(restored) == {
        "condition_hash",
        "condition_metadata",
        "provider_api",
        "request_options",
    }
    assert artifact["condition_hash"] == "sha256:abc"
    assert artifact["condition_metadata"] == {"effort": "high"}


def test_reconcile_fails_closed_on_conflicting_identity():
    artifact = {"condition_id": "model-low", "condition_hash": "sha256:abc"}

    with pytest.raises(ArtifactIdentityError) as exc_info:
        reconcile_condition_identity(
            artifact,
            MODEL,
            context="unit-1",
            restore_missing=True,
        )

    assert exc_info.value.conflicting_fields == ("condition_id",)


def test_validation_mode_rejects_missing_identity():
    with pytest.raises(ArtifactIdentityError) as exc_info:
        reconcile_condition_identity(
            {},
            MODEL,
            context="score-input",
            restore_missing=False,
        )

    assert "condition_id" in exc_info.value.missing_fields
    assert "condition_hash" in exc_info.value.missing_fields


def test_run_artifact_identity_checks_expected_transcripts(tmp_path):
    run_dir = tmp_path / "run"
    transcript = run_dir / "transcripts" / "unit.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"completed": true}')
    contract = {
        "identity": {"model_conditions": [{
            "key": "m-high",
            "condition_id": "m-high",
            "condition_hash": "sha256:m-high",
        }]},
        "modules": [{
            "module": "sus",
            "expected_units": [{
                "unit_id": "sus:m-high:bridge:run1",
                "model_key": "m-high",
                "expected_transcript_path": "transcripts/unit.json",
            }],
        }],
    }

    report = evaluate_run_artifact_identity(run_dir, contract=contract)

    assert report["conformant"] is False
    assert {issue["kind"] for issue in report["issues"]} == {
        "missing_condition_id",
        "missing_condition_hash",
    }

    transcript.write_text(
        '{"condition_id":"m-high","condition_hash":"sha256:m-high"}'
    )
    assert evaluate_run_artifact_identity(run_dir, contract=contract)["conformant"] is True


def test_current_contract_rejects_uncheckable_condition_identity(tmp_path):
    run_dir = tmp_path / "run"
    transcript = run_dir / "unit.json"
    run_dir.mkdir()
    transcript.write_text('{"completed": true}')
    contract = {
        "provenance": {"projection_version": IDENTITY_PROJECTION_VERSION},
        "identity": {"model_conditions": [{"key": "m"}]},
        "modules": [{
            "module": "aita",
            "expected_units": [{
                "unit_id": "aita:m:item0:side_a",
                "model_key": "m",
                "expected_transcript_path": "unit.json",
            }],
        }],
    }

    report = evaluate_run_artifact_identity(run_dir, contract=contract)

    assert report["conformant"] is False
    assert report["uncheckable_artifacts"] == 1
    assert report["issues"] == [{
        "kind": "uncheckable_condition_identity",
        "module": "aita",
        "unit_id": "aita:m:item0:side_a",
        "path": "unit.json",
        "missing_fields": ["condition_hash", "condition_id", "route_hash"],
    }]


def test_legacy_contract_keeps_uncheckable_identity_visible_without_retroactive_failure(
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "unit.json").write_text('{"completed": true}')
    contract = {
        "identity": {"model_conditions": [{"key": "m"}]},
        "modules": [{
            "module": "aita",
            "expected_units": [{
                "unit_id": "aita:m:item0:side_a",
                "model_key": "m",
                "expected_transcript_path": "unit.json",
            }],
        }],
    }

    report = evaluate_run_artifact_identity(run_dir, contract=contract)

    assert report["conformant"] is True
    assert report["uncheckable_artifacts"] == 1
