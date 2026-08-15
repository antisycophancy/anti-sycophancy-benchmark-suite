import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from types import SimpleNamespace

from suite_tools.run_monitor import (
    MonitoredOpenAIClient,
    RunMonitor,
    classify_failure_status,
    is_non_retryable_provider_error,
    response_usage_to_dict,
    sanitize_error_message,
)


def test_run_monitor_writes_status_and_events(tmp_path):
    monitor = RunMonitor(
        tmp_path,
        module="aita",
        stage="generation",
        metadata={"models": ["m"]},
    )
    monitor.record("turn_saved", model="m", item_idx=1, side="side_a", turn=1)
    monitor.mark_completed()

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]

    assert status["schema_version"] == "benchmark-run-ledger-v1"
    assert status["status"] == "completed"
    assert status["validity"] == "score_ready"
    assert status["counters"]["events.turn_saved"] == 1
    assert [event["event"] for event in events] == [
        "stage_started",
        "turn_saved",
        "stage_completed",
    ]


def test_run_monitor_marks_billing_failures_not_score_ready(tmp_path):
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    monitor.mark_failed("Error code: 402 - insufficient credits")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())

    assert status["status"] == "failed_billing"
    assert status["validity"] == "not_score_ready"


def test_run_monitor_treats_429_insufficient_quota_as_billing(tmp_path):
    class QuotaError(Exception):
        status_code = 429
        raw_response = {
            "error": {
                "code": "insufficient_quota",
                "message": "You exceeded your current quota",
            }
        }

    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    monitor.mark_failed(QuotaError("You exceeded your current quota"))

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_billing"
    assert status["validity"] == "not_score_ready"


def test_mark_failed_preserves_rate_limited_status(tmp_path):
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    monitor.mark_failed(Exception("HTTP 429: rate limit"))

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())

    assert status["status"] == "failed_rate_limited"
    assert status["validity"] == "not_score_ready"


def test_classify_failure_status_identifies_timeout_text_before_provider_fallback():
    assert classify_failure_status(RuntimeError("request timed out after 300 seconds")) == "failed_timeout"


def test_classify_failure_status_identifies_httpx_timeout_exception():
    assert classify_failure_status(httpx.ReadTimeout("read timed out")) == "failed_timeout"


def test_run_monitor_marks_timeout_failures_not_score_ready(tmp_path):
    monitor = RunMonitor(tmp_path, module="sus", stage="generation")
    monitor.mark_failed(httpx.ConnectTimeout("connection timed out"))

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())

    assert status["status"] == "failed_timeout"
    assert status["validity"] == "not_score_ready"


def test_run_monitor_redacts_provider_account_ids_and_keys(tmp_path):
    fake_user_id = "user_" + ("A" * 32)
    fake_api_key = "sk-" + "or-v1-secretvalue"
    raw = (
        "Error code: 400 - {'error': {'message': 'bad'}, "
        f"'user_id': '{fake_user_id}', "
        f"'api_key': '{fake_api_key}'}}"
    )
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    monitor.record("model_batch_failed", failure_reason=raw, nested={"message": raw})
    monitor.mark_failed(raw, model_errors=[raw])

    status_text = (tmp_path / "RUN_STATUS.json").read_text()
    events_text = (tmp_path / "RUN_EVENTS.jsonl").read_text()

    assert fake_user_id not in status_text
    assert fake_user_id not in events_text
    assert fake_api_key not in status_text
    assert fake_api_key not in events_text
    assert "'<redacted>'" in status_text
    assert "'api_key': '<redacted>'" in status_text


def test_run_monitor_redacts_openrouter_key_management_urls(tmp_path):
    fake_key_id = "0" * 40
    raw = (
        "HTTP 403: key limit exceeded. Manage it using "
        f"https://openrouter.ai/workspaces/default/keys/{fake_key_id}"
    )
    monitor = RunMonitor(tmp_path, module="sus", stage="generation")
    monitor.mark_failed(raw)

    status_text = (tmp_path / "RUN_STATUS.json").read_text()
    events_text = (tmp_path / "RUN_EVENTS.jsonl").read_text()

    assert fake_key_id not in status_text
    assert fake_key_id not in events_text
    assert "openrouter.ai/workspaces/default/keys/<redacted>" in status_text


def test_sanitize_error_message_redacts_google_keys_and_headers():
    fake_google_key = "AIza" + ("x" * 35)
    redacted = sanitize_error_message(
        f"google error key={fake_google_key}; x-goog-api-key: fake123456789"
    )

    assert fake_google_key not in redacted
    assert "AIza<redacted>" in redacted
    assert "fake123456789" not in redacted
    assert "x-goog-api-key: <redacted>" in redacted


def test_sanitize_error_message_preserves_provider_status_context():
    fake_user_id = "user_" + ("B" * 32)
    redacted = sanitize_error_message(
        f"Error code: 502 for {fake_user_id}"
    )

    assert "502" in redacted
    assert fake_user_id not in redacted


def test_run_monitor_marks_cooperative_stop_not_score_ready(tmp_path):
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    monitor.mark_stopped("operator requested stop before next paid call")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]

    assert status["status"] == "stopped"
    assert status["validity"] == "not_score_ready"
    assert events[-1]["event"] == "stage_stopped"


def test_run_monitor_records_usage_cost_and_preserves_it_across_stages(tmp_path):
    generation = RunMonitor(tmp_path, module="aita", stage="generation")
    generation.record_usage(
        "google/gemini-3-flash-preview",
        {"cost": 0.0123, "prompt_tokens": 10, "completion_tokens": 5},
        role="model_under_test",
    )

    scoring = RunMonitor(tmp_path, module="aita", stage="scoring")
    scoring.record_usage(
        "google/gemini-3.1-pro-preview",
        {"cost": 0.004, "prompt_tokens": 20, "completion_tokens": 2},
        role="judge",
    )

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())

    assert status["stage"] == "scoring"
    assert status["cost"]["total_cost_usd"] == 0.0163
    assert status["cost"]["total_calls"] == 2
    assert status["cost"]["tokens_in"] == 30
    assert status["cost"]["tokens_out"] == 7
    assert status["cost"]["cost_by_role"]["model_under_test"] == 0.0123
    assert status["cost"]["cost_by_role"]["judge"] == 0.004


def test_run_monitor_records_native_gemini_thinking_and_billable_output(tmp_path):
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    monitor.record_usage(
        "google/gemini-3.1-pro-preview",
        SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=4,
            thoughts_tokens=6,
            billable_completion_tokens=10,
            estimated_cost=0.001,
            cost_source="gemini_standard_estimate_2026_06",
        ),
        role="model_under_test",
        provider="google",
    )

    cost = json.loads((tmp_path / "RUN_STATUS.json").read_text())["cost"]
    assert cost["tokens_out"] == 4
    assert cost["thinking_tokens_out"] == 6
    assert cost["billable_tokens_out"] == 10
    for dimension, key in {
        "stage": "generation",
        "provider": "google",
        "model": "google/gemini-3.1-pro-preview",
        "source": "gemini_standard_estimate_2026_06",
    }.items():
        bucket = cost[f"usage_by_{dimension}"][key]
        assert bucket["tokens_out"] == 4
        assert bucket["thinking_tokens_out"] == 6
        assert bucket["billable_tokens_out"] == 10


@pytest.mark.parametrize(
    "usage",
    [
        {
            "prompt_tokens": 30,
            "completion_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
        {
            "input_tokens": 30,
            "output_tokens": 12,
            "output_tokens_details": {"reasoning_tokens": 7},
        },
        {
            "prompt_tokens": 30,
            "completion_tokens": 12,
            "reasoning_tokens": 7,
        },
    ],
)
def test_run_monitor_splits_openai_reasoning_from_visible_output(tmp_path, usage):
    monitor = RunMonitor(tmp_path, module="sus", stage="scoring")

    monitor.record_usage(
        "openai/gpt-reasoning",
        {**usage, "cost": 0.002, "cost_source": "provider_reported"},
        role="judge",
        provider="openai",
    )

    cost = json.loads((tmp_path / "RUN_STATUS.json").read_text())["cost"]
    assert cost["tokens_in"] == 30
    assert cost["tokens_out"] == 5
    assert cost["thinking_tokens_out"] == 7
    assert cost["billable_tokens_out"] == 12
    assert cost["usage_by_stage"]["scoring"]["tokens_out"] == 5
    assert cost["usage_by_provider"]["openai"]["thinking_tokens_out"] == 7
    assert cost["usage_by_model"]["openai/gpt-reasoning"][
        "billable_tokens_out"
    ] == 12
    assert cost["usage_by_source"]["provider_reported"]["tokens_out"] == 5


def test_run_monitor_still_ignores_arbitrary_empty_usage_records(tmp_path):
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    monitor.record_usage("not-a-physical-call", {}, role="unknown")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert "cost" not in status


def test_run_monitor_splits_reported_estimated_and_unknown_costs(tmp_path):
    generation = RunMonitor(tmp_path, module="aita", stage="generation")
    generation.record_usage(
        "google/gemini-flash-lite",
        {
            "cost": 0.01,
            "cost_source": "provider_reported",
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
        role="model_under_test",
        provider="openrouter",
    )
    generation.record_usage(
        "support/model",
        {"prompt_tokens": 25, "completion_tokens": 5},
        role="support",
        provider="google",
    )
    scoring = RunMonitor(tmp_path, module="aita", stage="scoring")
    scoring.record_usage(
        "judge/model",
        {
            "cost": 0.02,
            "cost_source": "gemini_standard_estimate_2026_06",
            "prompt_tokens": 200,
            "completion_tokens": 10,
        },
        role="judge",
        provider="google",
    )

    cost = json.loads((tmp_path / "RUN_STATUS.json").read_text())["cost"]
    assert cost["total_cost_usd"] == 0.03
    assert cost["reported_cost_usd"] == 0.01
    assert cost["estimated_cost_usd"] == 0.02
    assert cost["unknown_cost_calls"] == 1
    assert cost["cost_by_stage"] == {"generation": 0.01, "scoring": 0.02}
    assert cost["cost_by_provider"] == {"openrouter": 0.01, "google": 0.02}
    assert cost["usage_by_role"]["support"]["calls"] == 1
    assert cost["usage_by_role"]["support"]["cost_state"] == "unknown"


def test_run_monitor_reported_cost_wins_over_conflicting_estimate(tmp_path):
    monitor = RunMonitor(tmp_path, module="sus", stage="scoring")

    monitor.record_usage(
        "provider/model",
        {"cost": 0.01, "estimated_cost": 0.03},
        role="judge",
        provider="openrouter",
    )

    cost = monitor.status["cost"]
    assert cost["total_cost_usd"] == 0.01
    assert cost["reported_cost_usd"] == 0.01
    assert cost["estimated_cost_usd"] == 0
    assert cost["reported_cost_usd"] + cost["estimated_cost_usd"] == cost["total_cost_usd"]
    assert cost["usage_by_role"]["judge"]["reported_calls"] == 1
    assert cost["usage_by_role"]["judge"]["estimated_calls"] == 0
    assert cost["invalid_usage_fields"]["conflicting_cost_sources"] == 1


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), -1, "not-money"])
def test_run_monitor_treats_invalid_cost_as_unknown_and_normalizes_tokens(
    tmp_path, bad_cost
):
    monitor = RunMonitor(tmp_path, module="sus", stage="generation")

    monitor.record_usage(
        "unknown/model",
        {
            "cost": bad_cost,
            "prompt_tokens": "12",
            "completion_tokens": -5,
        },
        role="model_under_test",
        provider="custom",
    )

    raw = (tmp_path / "RUN_STATUS.json").read_text()
    cost = json.loads(raw)["cost"]
    assert "NaN" not in raw and "Infinity" not in raw
    assert cost["total_cost_usd"] == 0
    assert cost["reported_cost_usd"] == 0
    assert cost["estimated_cost_usd"] == 0
    assert cost["unknown_cost_calls"] == 1
    assert cost["tokens_in"] == 12
    assert cost["tokens_out"] == 0
    assert cost["usage_anomaly_count"] >= 2
    assert cost["invalid_usage_fields"]["cost"] == 1
    assert cost["invalid_usage_fields"]["completion_tokens"] == 1


def test_monitored_openai_client_records_response_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                usage=SimpleNamespace(
                    model_dump=lambda: {
                        "cost": 0.0005,
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                    }
                )
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monitor = RunMonitor(tmp_path, module="aita", stage="scoring")
    client = MonitoredOpenAIClient(fake_client, monitor, role="judge")

    client.chat.completions.create(
        model="judge/model",
        max_tokens=8192,
        extra_body={"reasoning": {"effort": "high"}},
    )
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]

    assert status["cost"]["total_cost_usd"] == 0.0005
    assert status["cost"]["cost_by_model"]["judge/model"] == 0.0005
    assert status["cost"]["cost_by_role"]["judge"] == 0.0005
    receipts = [event for event in events if event["event"] == "effective_request"]
    assert receipts[0]["effective_max_output_tokens"] == 8192
    assert receipts[0]["effective_reasoning_effort"] == "high"
    lease_events = (tmp_path / "leases" / "PAID_CALL_LEASE_EVENTS.jsonl").read_text()
    assert "lease_acquired" in lease_events
    assert "lease_released" in lease_events


def test_monitored_openai_client_attributes_provider_from_client_base_url(tmp_path, monkeypatch):
    """Real SDK clients expose base_url on the client object (not on
    chat.completions); the lease must attribute the call to that provider
    instead of defaulting to openrouter."""
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(usage=None)

    fake_client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    monitor = RunMonitor(tmp_path, module="epistemic", stage="scoring")
    client = MonitoredOpenAIClient(fake_client, monitor, role="judge")

    client.chat.completions.create(model="gpt-5.5")

    lease_events = [
        json.loads(line)
        for line in (tmp_path / "leases" / "PAID_CALL_LEASE_EVENTS.jsonl").read_text().splitlines()
    ]
    acquired = [event for event in lease_events if event.get("event") == "lease_acquired"]
    assert acquired
    assert acquired[-1]["provider"] == "openai"
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["cost"]["total_calls"] == 1
    assert status["cost"]["unknown_cost_calls"] == 1
    assert status["cost"]["usage_by_provider"]["openai"]["calls"] == 1


def test_monitored_openai_client_records_error_then_retry_success_once_each(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

    class FakeCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider failure")
            return SimpleNamespace(usage=None)

    fake_client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    monitor = RunMonitor(tmp_path, module="aita", stage="scoring")
    client = MonitoredOpenAIClient(fake_client, monitor, role="judge")

    with pytest.raises(RuntimeError, match="temporary provider failure") as exc_info:
        client.chat.completions.create(model="gpt-5.5")
    client.chat.completions.create(model="gpt-5.5")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert getattr(exc_info.value, "usage_recorded", False) is True
    assert status["cost"]["total_calls"] == 2
    assert status["cost"]["unknown_cost_calls"] == 2


def test_monitored_openai_client_records_terminal_error_usage_once(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

    class ProviderError(RuntimeError):
        usage = {"prompt_tokens": 7, "completion_tokens": 2, "cost": 0.001}

    class FakeCompletions:
        def create(self, **kwargs):
            raise ProviderError("terminal provider failure")

    fake_client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    monitor = RunMonitor(tmp_path, module="epistemic", stage="scoring")
    client = MonitoredOpenAIClient(fake_client, monitor, role="judge")

    with pytest.raises(ProviderError, match="terminal provider failure") as exc_info:
        client.chat.completions.create(model="judge/model")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert getattr(exc_info.value, "usage_recorded", False) is True
    assert status["cost"]["total_calls"] == 1
    assert status["cost"]["tokens_in"] == 7
    assert status["cost"]["tokens_out"] == 2
    assert status["cost"]["total_cost_usd"] == 0.001


def test_monitored_context_is_logged_locally_but_never_sent_to_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

    fake_client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    monitor = RunMonitor(tmp_path, module="epistemic", stage="scoring")
    client = MonitoredOpenAIClient(fake_client, monitor, role="judge")

    client.chat.completions.create(
        model="judge/model",
        messages=[{"role": "user", "content": "unchanged"}],
        _benchmark_request_context={
            "unit_id": "epis:unit-1",
            "dimension": "persistence",
            "provider": "openai",
        },
    )

    assert captured == {
        "model": "judge/model",
        "messages": [{"role": "user", "content": "unchanged"}],
    }
    diagnostic = [
        json.loads(line)
        for line in (tmp_path / "CALL_DIAGNOSTICS.jsonl").read_text().splitlines()
    ][0]
    assert diagnostic["context"] == {
        "unit_id": "epis:unit-1",
        "dimension": "persistence",
    }


def test_response_usage_to_dict_preserves_openrouter_extra_fields():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "cost": 0.0000055,
                "cost_details": {"upstream_inference_cost": 0.0000055},
            }
        )
    )

    usage = response_usage_to_dict(response)

    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 1
    assert usage["cost"] == 0.0000055


def test_classify_failure_status_separates_adapter_and_missing_scores():
    assert classify_failure_status("Adapter rejected incomplete backend response") == "failed_invalid"
    assert classify_failure_status("missing score: resistance_a") == "failed_scoring"
    assert classify_failure_status("not a valid model ID") == "failed_invalid"
    assert classify_failure_status("Error code: 429 - Rate limit exceeded") == "failed_rate_limited"


def test_invalid_model_errors_are_non_retryable():
    class BadRequest(Exception):
        status_code = 400

        def __str__(self):
            return "not a valid model ID"

    assert is_non_retryable_provider_error(BadRequest()) is True


def test_run_monitor_handles_concurrent_events(tmp_path):
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(monitor.record, "turn_saved", item_idx=i, turn=1)
            for i in range(12)
        ]
        for future in futures:
            future.result()

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]

    assert status["counters"]["events.turn_saved"] == 12
    assert len([event for event in events if event["event"] == "turn_saved"]) == 12
    assert len({event["sequence"] for event in events}) == len(events)


def test_timeout_errors_are_non_retryable():
    assert is_non_retryable_provider_error(httpx.ReadTimeout("read timed out")) is True
    assert is_non_retryable_provider_error(RuntimeError("request timed out after 300 seconds")) is True


# ---------------------------------------------------------------------------
# Task 4: chain-derived attempt numbering under file lock
# ---------------------------------------------------------------------------

from suite_tools.run_monitor import load_attempts


def test_rerun_same_stage_archives_and_increments(tmp_path):
    first = RunMonitor(tmp_path, module="aita", stage="generation")
    first.mark_failed(RuntimeError("quota dead"), status="failed_provider")
    second = RunMonitor(tmp_path, module="aita", stage="generation")

    assert second.status["attempt_number"] == 2
    attempts = load_attempts(tmp_path)
    assert len(attempts) == 1
    assert attempts[0]["schema_version"] == "benchmark-run-attempt-v1"
    assert attempts[0]["status"]["status"] == "failed_provider"
    assert attempts[0]["status"]["attempt_number"] == 1


def test_interleaved_stages_keep_independent_attempt_chains(tmp_path):
    """generation(1) -> scoring(1) -> generation rerun must be generation 2:
    the number comes from the CHAIN, not the last RUN_STATUS."""
    RunMonitor(tmp_path, module="aita", stage="generation").mark_completed()
    RunMonitor(tmp_path, module="aita", stage="scoring").mark_completed()
    generation_again = RunMonitor(tmp_path, module="aita", stage="generation")

    assert generation_again.status["attempt_number"] == 2
    scoring_again = RunMonitor(tmp_path, module="aita", stage="scoring")
    assert scoring_again.status["attempt_number"] == 2


def test_events_carry_attempt_number(tmp_path):
    RunMonitor(tmp_path, module="aita", stage="generation").mark_completed()
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    monitor.record("turn_saved", model="m", item_idx=1)
    events = [json.loads(line) for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "turn_saved"
    assert events[-1]["attempt_number"] == 2


def test_second_live_monitor_is_rejected_not_raced(tmp_path):
    """Concurrent same-dir monitors: exactly one wins; the loser raises
    RunMonitorConflictError instead of racing RUN_STATUS/RUN_EVENTS."""
    from suite_tools.run_monitor import RunMonitorConflictError

    RunMonitor(tmp_path, module="aita", stage="generation").mark_failed(RuntimeError("x"))
    outcomes = []

    def boot():
        try:
            outcomes.append(RunMonitor(tmp_path, module="aita", stage="generation").attempt_number)
        except RunMonitorConflictError:
            outcomes.append("conflict")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: boot(), range(2)))
    assert sorted(outcomes, key=str) == [2, "conflict"]


def test_stale_running_status_is_archived_not_conflict(tmp_path, monkeypatch):
    """A crashed process leaves status running with an old updated_at; the
    next monitor must pick up (attempt 2), not refuse."""
    crashed = RunMonitor(tmp_path, module="aita", stage="generation")
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    status["updated_at"] = "2020-01-01T00:00:00+00:00"
    (tmp_path / "RUN_STATUS.json").write_text(json.dumps(status))

    revived = RunMonitor(tmp_path, module="aita", stage="generation")
    assert revived.attempt_number == 2


def test_takeover_env_overrides_conflict(tmp_path, monkeypatch):
    RunMonitor(tmp_path, module="aita", stage="generation")  # fresh + running
    monkeypatch.setenv("BENCHMARK_MONITOR_TAKEOVER", "1")
    taken = RunMonitor(tmp_path, module="aita", stage="generation")
    assert taken.attempt_number == 2


def test_load_attempts_empty_when_fresh(tmp_path):
    RunMonitor(tmp_path, module="aita", stage="generation")
    assert load_attempts(tmp_path) == []


# ---------------------------------------------------------------------------
# Stale ATTEMPTS.lock recovery (final-review item 1)
# ---------------------------------------------------------------------------

def test_stale_attempts_lock_is_cleared_and_init_succeeds_quickly(tmp_path):
    """A hard-killed process leaves ATTEMPTS.lock behind with an old mtime.
    RunMonitor must clear it and complete init in well under 5 seconds."""
    lock_path = tmp_path / "ATTEMPTS.lock"
    lock_path.touch()
    old_time = time.time() - 120  # 120 seconds in the past
    os.utime(str(lock_path), (old_time, old_time))

    start = time.monotonic()
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"stale-lock recovery took {elapsed:.2f}s (expected < 5s)"
    assert monitor.attempt_number == 1
    assert not lock_path.exists()  # released by context cleanup inside __init__


def test_fresh_attempts_lock_raises_runtime_error_after_deadline(tmp_path, monkeypatch):
    """A fresh (recently created) ATTEMPTS.lock should cause RuntimeError after
    the configured deadline. We use a very short deadline so the test is fast."""
    monkeypatch.setenv("BENCHMARK_ATTEMPT_LOCK_DEADLINE_SECONDS", "0.2")
    lock_path = tmp_path / "ATTEMPTS.lock"
    lock_path.touch()
    # Leave mtime at now so the stale-check won't clear it.

    with pytest.raises(RuntimeError, match="attempt lock"):
        RunMonitor(tmp_path, module="aita", stage="generation")


def test_record_block_writes_unit_id_into_block_and_event(tmp_path):
    from suite_tools.run_monitor import RunMonitor
    monitor = RunMonitor(output_dir=tmp_path, module="sus", stage="generation")  # match the real ctor
    monitor.record_block(
        unit={"scenario": "bridge", "run_number": 3},
        unit_id="sus:m:bridge:run3",
        evidence={"evidence_class": "model_signal", "category": "refusal"},
        model="m",
        evidence_pointer="transcripts/m_bridge_run3.json",
    )
    import json
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().splitlines()[0])
    assert block["unit_id"] == "sus:m:bridge:run3"
    events = [json.loads(x) for x in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()]
    recorded = [e for e in events if e["event"] == "block_recorded"]
    assert recorded and recorded[0]["unit_id"] == "sus:m:bridge:run3"
