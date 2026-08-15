from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from suite_tools.call_diagnostics import (
    CallDiagnosticJournal,
    DiagnosticJournalError,
    begin_provider_attempt,
    close_error_best_effort,
    close_success_best_effort,
    diagnose_call_journal,
    load_call_diagnostics,
)
from suite_tools.provider_client import ProviderApiError
from suite_tools import bench
from suite_tools.run_contract import provenance_hashes


def _contract(run_dir: Path) -> bytes:
    payload = b'{"run_id":"diagnostic-test","immutable":true}\n'
    (run_dir / "RUN_CONTRACT.json").write_bytes(payload)
    return payload


def _response(content: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, refusal=None),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
        ),
    )


def test_lifecycle_is_local_and_does_not_change_contract_or_response(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract_before = _contract(run_dir)
    provenance_before = provenance_hashes(json.loads(contract_before))
    monkeypatch.setenv("BENCHMARK_RUN_ID", "diagnostic-test")

    response = _response("same response")
    attempt = begin_provider_attempt(
        output_dir=run_dir,
        module="sus",
        stage="generation",
        role="model_under_test",
        model="example/model",
        provider="example",
        provider_api="openai_compatible",
        context={"unit_id": "sus:example:1", "turn": 2},
    )
    attempt.mark_provider_invocation_started()
    close_success_best_effort(attempt, response)

    assert response.choices[0].message.content == "same response"
    assert (run_dir / "RUN_CONTRACT.json").read_bytes() == contract_before
    assert provenance_hashes(json.loads(contract_before)) == provenance_before
    records = load_call_diagnostics(run_dir)["records"]
    assert [record["state"] for record in records] == [
        "intent_written",
        "provider_invocation_started",
        "closed",
    ]
    assert {record["logical_call_id"] for record in records} == {
        records[0]["logical_call_id"]
    }
    assert records[-1]["outcome"] == "provider_response"
    assert records[-1]["billing_state"] == "confirmed"
    assert records[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert (run_dir / "CALL_DIAGNOSTICS.jsonl").stat().st_mode & 0o777 == 0o600


def test_retries_share_logical_id_and_get_distinct_attempt_ids(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _contract(run_dir)
    kwargs = {
        "output_dir": run_dir,
        "module": "aita",
        "stage": "generation",
        "role": "model_under_test",
        "model": "example/model",
        "provider": "example",
        "context": {"unit_id": "aita:example:0:a", "turn": 1},
    }

    first = begin_provider_attempt(**kwargs)
    first.mark_provider_invocation_started()
    close_error_best_effort(first, TimeoutError("transient"))
    second = begin_provider_attempt(**kwargs)
    second.mark_provider_invocation_started()
    close_success_best_effort(second, _response())

    assert first.logical_call_id == second.logical_call_id
    assert first.attempt_id != second.attempt_id
    report = diagnose_call_journal(run_dir)
    assert report["attempt_count"] == 2
    assert report["closed_count"] == 2
    assert report["failure_count"] == 1
    assert report["unresolved_count"] == 0


def test_error_projection_redacts_secrets_and_omits_prompt_fields(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _contract(run_dir)
    secret = "sk-" + "secretmaterial123456789"
    prompt = "PRIVATE BENCHMARK PROMPT MUST NOT APPEAR"
    error = ProviderApiError(
        400,
        f"blocked {prompt} for Bearer {secret}",
        raw_response={
            "error": {
                "type": "invalid_request_error",
                "code": "content_policy_violation",
                "message": f"blocked Bearer {secret}",
            },
            "messages": [{"role": "user", "content": prompt}],
            "private_debug": {"key": secret},
        },
    )

    attempt = begin_provider_attempt(
        output_dir=run_dir,
        module="sus",
        stage="generation",
        role="model_under_test",
        model="example/model",
        context={"unit_id": "unit-1"},
    )
    attempt.mark_provider_invocation_started()
    close_error_best_effort(attempt, error)

    journal_text = (run_dir / "CALL_DIAGNOSTICS.jsonl").read_text()
    assert secret not in journal_text
    assert prompt not in journal_text
    closed = load_call_diagnostics(run_dir)["records"][-1]
    assert closed["provider_error"]["code"] == "content_policy_violation"
    assert "message" not in closed["provider_error"]
    assert len(closed["error_message_sha256"]) == 64
    assert closed["http_status"] == 400
    assert closed["raw_body_sha256"]
    assert closed["billing_state"] == "likely"


def test_unclosed_invocation_is_reported_as_ambiguous(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _contract(run_dir)
    attempt = begin_provider_attempt(
        output_dir=run_dir,
        module="epis",
        stage="scoring",
        role="judge",
        model="judge/model",
        context={"unit_id": "epis:1", "dimension": "persistence"},
    )
    attempt.mark_provider_invocation_started()

    report = diagnose_call_journal(run_dir)
    assert report["unresolved_count"] == 1
    assert report["unresolved"][0]["states"] == [
        "intent_written",
        "provider_invocation_started",
    ]


def test_malformed_http_200_shape_is_a_diagnostic_failure(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _contract(run_dir)
    attempt = begin_provider_attempt(
        output_dir=run_dir,
        module="aita",
        stage="generation",
        role="model_under_test",
        model="example/model",
    )
    attempt.mark_provider_invocation_started()
    close_success_best_effort(
        attempt,
        SimpleNamespace(choices=None, usage=None, native_finish_reason=None),
    )

    report = diagnose_call_journal(run_dir)
    assert report["failure_count"] == 1
    assert report["failures"][0]["outcome"] == "malformed_response"
    assert report["failures"][0]["response_shape"] == "choices_null"


def test_loader_tolerates_one_torn_final_line(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "CALL_DIAGNOSTICS.jsonl"
    path.write_text(json.dumps({"event_id": "ok", "sequence": 1}) + "\n{\"event_id\":")

    loaded = load_call_diagnostics(run_dir)

    assert [record["event_id"] for record in loaded["records"]] == ["ok"]
    assert loaded["torn_tail"] is True
    assert loaded["malformed"] == []


def test_intent_write_failure_is_fail_closed_before_provider_invocation(tmp_path, monkeypatch):
    def fail(_self, _fields, *, allocate_attempt=False):
        raise DiagnosticJournalError("disk unavailable")

    monkeypatch.setattr(CallDiagnosticJournal, "_append", fail)

    with pytest.raises(DiagnosticJournalError, match="disk unavailable"):
        begin_provider_attempt(
            output_dir=tmp_path,
            module="sus",
            stage="generation",
            role="model_under_test",
            model="example/model",
        )


def test_non_path_monitor_output_dir_does_not_create_a_journal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monitor = SimpleNamespace(
        output_dir=MagicMock(name="output_dir"),
        module="sus",
        stage="generation",
    )

    attempt = begin_provider_attempt(
        monitor=monitor,
        role="model_under_test",
        model="example/model",
    )

    assert attempt.attempt_id == ""
    assert list(tmp_path.iterdir()) == []


def test_close_failure_is_best_effort_after_provider_return(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _contract(run_dir)
    attempt = begin_provider_attempt(
        output_dir=run_dir,
        module="sus",
        stage="generation",
        role="model_under_test",
        model="example/model",
    )
    attempt.mark_provider_invocation_started()

    def fail(_response):
        raise DiagnosticJournalError("late disk failure")

    monkeypatch.setattr(attempt, "close_success", fail)

    assert close_success_best_effort(attempt, _response()) is None


def test_bench_diagnose_cli_is_read_only_json(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract_before = _contract(run_dir)
    attempt = begin_provider_attempt(
        output_dir=run_dir,
        module="aita",
        stage="scoring",
        role="judge",
        model="judge/model",
        context={"unit_id": "aita:0", "dimension": "outcome"},
    )
    attempt.mark_provider_invocation_started()
    close_error_best_effort(attempt, RuntimeError("HTTP 503 overloaded"))
    journal_before = (run_dir / "CALL_DIAGNOSTICS.jsonl").read_bytes()

    assert bench.main(["diagnose", str(run_dir), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_count"] == 1
    assert payload["failures"][0]["failure_status"] == "failed_provider"
    assert (run_dir / "RUN_CONTRACT.json").read_bytes() == contract_before
    assert (run_dir / "CALL_DIAGNOSTICS.jsonl").read_bytes() == journal_before
