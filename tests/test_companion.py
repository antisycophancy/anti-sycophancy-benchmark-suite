import json
from pathlib import Path
import subprocess
import sys

import pytest

from suite_tools import companion


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_contract(run_dir: Path, *, run_id: str = "run-1") -> Path:
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": run_id,
            "expected_models": [
                {
                    "key": "target-low",
                    "model_id": "provider/target",
                    "endpoint": "openai_compatible",
                    "condition_id": "target-low",
                    "condition_metadata": {"provider_route": "direct"},
                }
            ],
            "expected_judges": [
                {
                    "role": "panel",
                    "model_id": "provider/judge",
                    "config": {
                        "provider_api": "openai_compatible",
                        "api_key_env": "SECRET_JUDGE_KEY",
                        "condition_metadata": {"provider_route": "gateway"},
                    },
                }
            ],
            "modules": [
                {
                    "module": "aita",
                    "stage": "generation",
                    "expected_units": [
                        {
                            "unit_id": "aita:target:item0:side_a",
                            "expected_transcript_path": "target_item0_side_a.json",
                            "planned_turns": 1,
                        }
                    ],
                }
            ],
        },
    )
    return run_dir


def _complete_generation(run_dir: Path) -> None:
    _write_json(run_dir / "target_item0_side_a.json", {"turns": [{}]})
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "schema_version": "benchmark-run-status-v1",
            "status": "completed",
            "stage": "generation",
            "validity": "not_score_ready",
            "updated_at": "2026-07-31T12:00:00Z",
        },
    )


def test_resume_derives_phase_from_current_run_artifacts(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")

    first = companion.start_workflow(
        state_root,
        workflow_id="paper-finish",
        goal="collection",
        run_dirs=[run_dir],
    )
    assert first["phase"] == "prepared"
    assert first["next_action"] == "preflight_or_generate"
    assert first["runs"][0]["counts"] == {
        "done": 0,
        "terminal_model_signal": 0,
        "owed": 1,
    }

    _complete_generation(run_dir)
    resumed = companion.resume_workflow(state_root, "paper-finish")

    assert resumed["phase"] == "needs_scoring"
    assert resumed["next_action"] == "review_generation_and_approve_scoring"
    assert resumed["runs"][0]["phase"] == "needs_scoring"
    assert resumed["runs"][0]["counts"]["done"] == 1


def test_evidence_package_is_the_only_final_workflow_goal(tmp_path):
    state_root = tmp_path / "companion"

    started = companion.start_workflow(
        state_root,
        workflow_id="evidence-package",
        goal="evidence_package",
    )

    assert started["goal"] == "evidence_package"
    assert "publication" not in companion.GOALS
    with pytest.raises(ValueError, match="goal must be one of"):
        companion.start_workflow(
            state_root,
            workflow_id="old-publication-name",
            goal="publication",
        )


def test_completed_workflow_points_to_evidence_packaging():
    assert companion._aggregate_phase([{"phase": "completed"}]) == (
        "completed",
        "prepare_evidence_package",
    )


def test_structured_onboarding_choices_survive_without_storing_free_text(tmp_path):
    state_root = tmp_path / "companion"
    companion.start_workflow(
        state_root,
        workflow_id="onboarding",
        goal="onboarding",
    )

    companion.record_choice(
        state_root,
        "onboarding",
        key="connection_route",
        value="provider_direct",
    )
    companion.record_choice(
        state_root,
        "onboarding",
        key="target_provider",
        value="anthropic",
    )
    resumed = companion.resume_workflow(state_root, "onboarding")

    assert resumed["choices"] == {
        "connection_route": "provider_direct",
        "target_provider": "anthropic",
    }
    with pytest.raises(ValueError, match="unsupported choice value"):
        companion.record_choice(
            state_root,
            "onboarding",
            key="target_provider",
            value="my private provider notes",
        )


def test_resume_fails_closed_when_completed_status_disagrees_with_units(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "status": "completed",
            "stage": "generation",
            "validity": "not_score_ready",
        },
    )
    companion.start_workflow(
        state_root,
        workflow_id="conflict",
        goal="recovery",
        run_dirs=[run_dir],
    )

    resumed = companion.resume_workflow(state_root, "conflict")

    assert resumed["phase"] == "attention"
    assert resumed["runs"][0]["phase"] == "ledger_conflict"
    assert resumed["next_action"] == "inspect_authoritative_ledgers"


def test_approval_is_single_use_and_bound_to_contract_bytes(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    companion.start_workflow(
        state_root,
        workflow_id="approval",
        goal="smoke",
        run_dirs=[run_dir],
    )

    granted = companion.grant_approval(
        state_root,
        "approval",
        run_dir=run_dir,
        stage="generation",
        confirmed_by_user=True,
    )
    approval_id = granted["approval_id"]
    assert companion.resume_workflow(state_root, "approval")["approvals"][0]["state"] == "active"

    companion.consume_approval(state_root, "approval", approval_id=approval_id)
    assert companion.resume_workflow(state_root, "approval")["approvals"][0]["state"] == "consumed"
    with pytest.raises(ValueError, match="already consumed"):
        companion.consume_approval(state_root, "approval", approval_id=approval_id)


def test_claim_marker_prevents_reuse_if_interrupted_before_event_append(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    companion.start_workflow(
        state_root,
        workflow_id="interrupted-consume",
        goal="smoke",
        run_dirs=[run_dir],
    )
    approval_id = companion.grant_approval(
        state_root,
        "interrupted-consume",
        run_dir=run_dir,
        stage="generation",
        confirmed_by_user=True,
    )["approval_id"]
    companion.consume_approval(
        state_root,
        "interrupted-consume",
        approval_id=approval_id,
    )

    events_path = state_root / "interrupted-consume" / "EVENTS.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events = [event for event in events if event["event"] != "approval_consumed"]
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    resumed = companion.resume_workflow(state_root, "interrupted-consume")
    assert resumed["approvals"][0]["state"] == "consumed"
    with pytest.raises(ValueError, match="already consumed"):
        companion.consume_approval(
            state_root,
            "interrupted-consume",
            approval_id=approval_id,
        )


def test_contract_change_invalidates_unconsumed_approval(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    companion.start_workflow(
        state_root,
        workflow_id="drift",
        goal="collection",
        run_dirs=[run_dir],
    )
    companion.grant_approval(
        state_root,
        "drift",
        run_dir=run_dir,
        stage="generation",
        confirmed_by_user=True,
    )

    contract_path = run_dir / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["run_id"] = "changed-after-approval"
    _write_json(contract_path, contract)

    resumed = companion.resume_workflow(state_root, "drift")

    assert resumed["approvals"][0]["state"] == "stale_contract"
    assert resumed["active_approvals"] == []


def test_approval_requires_an_explicit_user_confirmation(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    companion.start_workflow(
        state_root,
        workflow_id="no-consent",
        goal="smoke",
        run_dirs=[run_dir],
    )

    with pytest.raises(ValueError, match="explicit user confirmation"):
        companion.grant_approval(
            state_root,
            "no-consent",
            run_dir=run_dir,
            stage="generation",
            confirmed_by_user=False,
        )


def test_scoring_approval_is_rejected_before_generation_is_complete(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    companion.start_workflow(
        state_root,
        workflow_id="ordered-gates",
        goal="collection",
        run_dirs=[run_dir],
    )

    with pytest.raises(ValueError, match="scoring approval requires needs_scoring"):
        companion.grant_approval(
            state_root,
            "ordered-gates",
            run_dir=run_dir,
            stage="scoring",
            confirmed_by_user=True,
        )

    _complete_generation(run_dir)
    granted = companion.grant_approval(
        state_root,
        "ordered-gates",
        run_dir=run_dir,
        stage="scoring",
        confirmed_by_user=True,
    )
    assert granted["resume"]["active_approvals"][0]["stage"] == "scoring"


def test_companion_files_exclude_secrets_prompts_and_transcripts(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    companion.start_workflow(
        state_root,
        workflow_id="privacy",
        goal="smoke",
        run_dirs=[run_dir],
    )
    companion.grant_approval(
        state_root,
        "privacy",
        run_dir=run_dir,
        stage="generation",
        confirmed_by_user=True,
    )
    companion.resume_workflow(state_root, "privacy")

    stored = "\n".join(
        path.read_text(encoding="utf-8")
        for path in state_root.rglob("*")
        if path.is_file()
    ).lower()
    assert "secret_judge_key" not in stored
    for forbidden in ("api_key", "messages", "prompt", "transcript"):
        assert forbidden not in stored


def test_cli_resumes_the_active_workflow_in_a_fresh_process(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    base = [
        sys.executable,
        "-m",
        "suite_tools.companion",
        "--state-root",
        str(state_root),
    ]

    started = subprocess.run(
        [
            *base,
            "start",
            "fresh-context",
            "--goal",
            "smoke",
            "--run",
            str(run_dir),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert started.returncode == 0, started.stderr

    resumed = subprocess.run(
        [*base, "resume", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    payload = json.loads(resumed.stdout)
    assert payload["workflow_id"] == "fresh-context"
    assert payload["phase"] == "prepared"
    assert (state_root / "fresh-context" / "EVENTS.jsonl").is_file()
    assert (state_root / "fresh-context" / "RESUME.json").is_file()


def test_listing_workflows_does_not_change_the_active_workflow(tmp_path):
    state_root = tmp_path / "companion"
    first_run = _run_contract(tmp_path / "runs" / "first", run_id="first")
    second_run = _run_contract(tmp_path / "runs" / "second", run_id="second")
    companion.start_workflow(
        state_root,
        workflow_id="z-first",
        goal="smoke",
        run_dirs=[first_run],
    )
    companion.start_workflow(
        state_root,
        workflow_id="a-second",
        goal="collection",
        run_dirs=[second_run],
    )

    companion.list_workflows(state_root)

    assert companion.active_workflow_id(state_root) == "a-second"


def test_cli_prints_the_approval_receipt_for_the_operator(tmp_path):
    state_root = tmp_path / "companion"
    run_dir = _run_contract(tmp_path / "runs" / "run-1")
    companion.start_workflow(
        state_root,
        workflow_id="receipt",
        goal="smoke",
        run_dirs=[run_dir],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suite_tools.companion",
            "--state-root",
            str(state_root),
            "approve",
            "receipt",
            "--run",
            str(run_dir),
            "--stage",
            "generation",
            "--confirmed-by-user",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "approval:" in result.stdout
    assert "workflow: receipt" in result.stdout


def test_default_state_root_follows_the_open_benchmark_checkout(tmp_path, monkeypatch):
    workspace = tmp_path / "benchmark"
    nested = workspace / "aita-bench"
    nested.mkdir(parents=True)
    (workspace / "suite_tools").mkdir()
    (workspace / "suite_models.yaml").write_text("models: {}\n", encoding="utf-8")
    monkeypatch.chdir(nested)

    assert companion.default_state_root() == workspace / ".benchmark-companion"
