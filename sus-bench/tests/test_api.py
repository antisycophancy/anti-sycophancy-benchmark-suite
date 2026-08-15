from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from sus_bench import api
from suite_tools.provider_client import (
    OPENAI_RESPONSES_URL,
    ProviderApiError,
    extract_raw_response,
)


@contextmanager
def _no_lease(*args, **kwargs):
    yield


class _FakeChatClient:
    def __init__(self, captured, response):
        self.captured = captured
        self.response = response
        self.base_url = "https://openrouter.ai/api/v1"
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.captured["create_kwargs"] = kwargs
        return self.response


def _fake_openai_factory(captured, response):
    def factory(**kwargs):
        captured["client_kwargs"] = kwargs
        client = _FakeChatClient(captured, response)
        client.base_url = kwargs.get("base_url")
        return client

    return factory


def _chat_response(text, *, usage=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage or {},
    )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parse_llm_json_rejects_nonstandard_numeric_constants(constant):
    with pytest.raises(json.JSONDecodeError):
        api.parse_llm_json(f'prefix {{"target_utility": {constant}}} suffix')


def test_call_openrouter_dispatches_to_anthropic_messages(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "text", "text": "Native Claude response"},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    api.reset_cost_tracker()

    text, latency = api.call_openrouter(
        "claude-opus-4-8",
        [
            {"role": "system", "content": "System context"},
            {"role": "user", "content": "Hello"},
        ],
        "secret-key",
        base_url=api.ANTHROPIC_MESSAGES_URL,
        request_options={
            "max_tokens": 128,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
        },
        role="model_under_test",
    )

    assert text == "Native Claude response"
    assert latency >= 0
    assert captured["url"] == api.ANTHROPIC_MESSAGES_URL
    assert captured["headers"]["x-api-key"] == "secret-key"
    assert captured["headers"]["anthropic-version"] == api.ANTHROPIC_VERSION
    assert captured["payload"]["system"] == "System context"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert captured["payload"]["thinking"] == {"type": "adaptive"}
    assert captured["payload"]["output_config"] == {"effort": "xhigh"}

    summary = api.get_cost_tracker().summary()
    assert summary["tokens_in"] == 100
    assert summary["tokens_out"] == 20
    assert summary["cost_by_model"]["claude-opus-4-8"] == 0.001


def test_call_openrouter_tracks_native_opus_5_cost(monkeypatch):
    class Response:
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {
                "content": [{"type": "text", "text": "Native Claude response"}],
                "usage": {"input_tokens": 100_000, "output_tokens": 10_000},
            }

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post",
        lambda *args, **kwargs: Response(),
    )
    api.reset_cost_tracker()

    api.call_openrouter(
        "claude-opus-5",
        [{"role": "user", "content": "Hello"}],
        "secret-key",
        base_url=api.ANTHROPIC_MESSAGES_URL,
        role="model_under_test",
    )

    summary = api.get_cost_tracker().summary()
    assert summary["cost_by_model"]["claude-opus-5"] == 0.75
    assert summary["unknown_cost_calls"] == 0


def test_anthropic_rate_limit_records_cooldown(monkeypatch):
    recorded = {}

    class Response:
        status_code = 429
        text = '{"error":{"message":"rate limit"}}'
        headers = {"retry-after": "7"}

        def json(self):
            return {}

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr("suite_tools.provider_client.httpx.post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        api,
        "record_rate_limit_cooldown",
        lambda **kwargs: recorded.update(kwargs),
    )

    try:
        api.call_openrouter(
            "claude-opus-4-8",
            [{"role": "user", "content": "Hello"}],
            "secret-key",
            base_url=api.ANTHROPIC_MESSAGES_URL,
        )
    except api.BenchmarkApiError as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("429 should raise BenchmarkApiError")

    assert recorded["provider"] == "anthropic"
    assert recorded["model"] == "claude-opus-4-8"
    assert recorded["headers"] == {"retry-after": "7"}


def test_insufficient_quota_429_does_not_record_rate_limit_cooldown(monkeypatch):
    recorded = []

    class Response:
        status_code = 429
        text = '{"error":{"code":"insufficient_quota","message":"You exceeded your quota"}}'
        headers = {}

        def json(self):
            return {
                "error": {
                    "code": "insufficient_quota",
                    "message": "You exceeded your quota",
                }
            }

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post",
        lambda *args, **kwargs: Response(),
    )
    monkeypatch.setattr(
        api,
        "record_rate_limit_cooldown",
        lambda **kwargs: recorded.append(kwargs),
    )

    with pytest.raises(api.BenchmarkApiError):
        api.call_openrouter(
            "gpt-5.6-sol",
            [{"role": "user", "content": "Hello"}],
            "secret-key",
            base_url=OPENAI_RESPONSES_URL,
        )

    assert recorded == []


def test_anthropic_empty_text_response_is_provider_failure(monkeypatch):
    class Response:
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "redacted_thinking", "data": "..."},
                ],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr("suite_tools.provider_client.httpx.post", lambda *args, **kwargs: Response())
    api.reset_cost_tracker()

    try:
        api.call_openrouter(
            "claude-opus-4-8",
            [{"role": "user", "content": "Hello"}],
            "secret-key",
            base_url=api.ANTHROPIC_MESSAGES_URL,
            role="model_under_test",
        )
    except api.BenchmarkApiError as exc:
        assert exc.status_code == 502
        assert "no text blocks" in exc.text
        assert "stop_reason=max_tokens" in exc.text
        assert "thinking" in exc.text
    else:
        raise AssertionError("empty Anthropic text should fail before scoring")

    summary = api.get_cost_tracker().summary()
    assert summary["cost_by_model"]["claude-opus-4-8"] == 0.001


def test_call_openrouter_dispatches_to_gemini_generate_content(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {
                "modelVersion": "gemini-3.1-pro-preview-20260601",
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"thought": True, "text": "internal thinking"},
                                {"text": "Native Gemini response"},
                            ]
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 20,
                    "thoughtsTokenCount": 12,
                    "totalTokenCount": 132,
                },
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    api.reset_cost_tracker()

    text, latency = api.call_openrouter(
        "gemini-3.1-pro-preview",
        [
            {"role": "system", "content": "System context"},
            {"role": "user", "content": "Hello"},
        ],
        "secret-key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        request_options={
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "high",
                    "includeThoughts": False,
                }
            }
        },
        role="model_under_test",
    )

    assert text == "Native Gemini response"
    assert latency >= 0
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.1-pro-preview:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "secret-key"
    assert captured["payload"] == {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "systemInstruction": {"parts": [{"text": "System context"}]},
        "generationConfig": {
            "maxOutputTokens": 4096,
            "thinkingConfig": {
                "thinkingLevel": "high",
                "includeThoughts": False,
            },
        },
    }

    summary = api.get_cost_tracker().summary()
    assert summary["tokens_in"] == 100
    assert summary["tokens_out"] == 20
    assert summary["thinking_tokens_out"] == 12
    assert summary["cost_by_model"]["gemini-3.1-pro-preview"] == 0.000584


def test_openai_compatible_defaults_to_bounded_max_tokens(monkeypatch):
    captured = {}
    response = _chat_response(
        "ok",
        usage={"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.001},
    )

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(api, "_openai_factory", _fake_openai_factory(captured, response))
    api.reset_cost_tracker()

    text, _latency = api.call_openrouter(
        "provider/model",
        [{"role": "user", "content": "Hello"}],
        "secret-key",
        base_url=api.OPENROUTER_URL,
        role="model_under_test",
    )

    assert text == "ok"
    assert captured["client_kwargs"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["create_kwargs"]["max_tokens"] == 4096


def test_openai_compatible_attaches_scheduler_context_to_paid_call_lease(monkeypatch):
    captured = {}
    response = _chat_response(
        "ok",
        usage={"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.001},
    )

    @contextmanager
    def capture_lease(*args, **kwargs):
        captured["lease"] = kwargs
        yield

    monkeypatch.setenv("BENCHMARK_RUN_ID", "run-123")
    monkeypatch.setenv("BENCHMARK_MODULE", "sus")
    monkeypatch.setenv("BENCHMARK_OUTPUT_DIR", "/tmp/sus-output")
    monkeypatch.setenv("BENCHMARK_CONTRACT_PATH", "/tmp/sus-output/RUN_CONTRACT.json")
    monkeypatch.setattr(api, "paid_call_lease", capture_lease)
    monkeypatch.setattr(api, "_openai_factory", _fake_openai_factory(captured, response))
    api.reset_cost_tracker()

    api.call_openrouter(
        "provider/model",
        [{"role": "user", "content": "Hello"}],
        "secret-key",
        base_url=api.OPENROUTER_URL,
        role="model_under_test",
    )

    assert captured["lease"]["provider"] == "openrouter"
    assert captured["lease"]["model"] == "provider/model"
    assert captured["lease"]["role"] == "model_under_test"
    assert captured["lease"]["module"] == "sus"
    assert captured["lease"]["run_id"] == "run-123"
    assert captured["lease"]["output_dir"] == "/tmp/sus-output"
    assert captured["lease"]["contract_path"] == "/tmp/sus-output/RUN_CONTRACT.json"


def test_openai_compatible_request_options_can_override_max_tokens(monkeypatch):
    captured = {}
    response = _chat_response(
        "ok",
        usage={"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.001},
    )

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(api, "_openai_factory", _fake_openai_factory(captured, response))
    api.reset_cost_tracker()

    api.call_openrouter(
        "provider/model",
        [{"role": "user", "content": "Hello"}],
        "secret-key",
        base_url=api.OPENROUTER_URL,
        request_options={"max_tokens": 128},
        role="model_under_test",
    )

    assert captured["create_kwargs"]["max_tokens"] == 128


def test_direct_openai_gpt5_uses_max_completion_tokens(monkeypatch):
    captured = {}
    response = _chat_response(
        "ok",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    )

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(api, "_openai_factory", _fake_openai_factory(captured, response))
    api.reset_cost_tracker()

    text, _latency = api.call_openrouter(
        "gpt-5.5",
        [{"role": "user", "content": "Hello"}],
        "secret-key",
        base_url="https://api.openai.com/v1/chat/completions",
        role="model_under_test",
    )

    assert text == "ok"
    assert captured["client_kwargs"]["base_url"] == "https://api.openai.com/v1"
    assert captured["create_kwargs"]["max_completion_tokens"] == 4096
    assert "max_tokens" not in captured["create_kwargs"]
    summary = api.get_cost_tracker().summary()
    assert summary["thinking_tokens_out"] == 1
    assert summary["cost_by_model"]["gpt-5.5"] == 0.00011


def test_cost_tracker_records_responses_api_reasoning_tokens():
    """CostTracker.record() accumulates reasoning_tokens from Responses API path.

    The Responses client returns usage as a plain dict with reasoning_tokens
    at the top level (not nested under completion_tokens_details). This tests
    that the SUS cost ledger correctly sums them into thinking_tokens_out.
    """
    from sus_bench.api import CostTracker

    tracker = CostTracker()
    # Simulate what _responses_usage() + estimate_usage_cost() return:
    # top-level reasoning_tokens key (not nested under completion_tokens_details)
    usage = {
        "prompt_tokens": 50,
        "completion_tokens": 30,
        "total_tokens": 80,
        "reasoning_tokens": 20,  # Responses API path
        "cost": 0.005,
    }
    tracker.record("openai/gpt-5.6", usage, role="model_under_test")

    summary = tracker.summary()
    assert summary["thinking_tokens_out"] == 20
    assert summary["tokens_in"] == 50
    assert summary["tokens_out"] == 30


def test_cost_tracker_flags_unpriced_calls_without_flagging_known_zero_cost():
    from sus_bench.api import CostTracker

    tracker = CostTracker()
    tracker.record(
        "new/direct-model",
        {"prompt_tokens": 50, "completion_tokens": 5},
        role="model_under_test",
    )
    tracker.record(
        "known/free-outcome",
        {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "cost": 0.0,
            "cost_source": "anthropic_refusal_no_charge",
        },
        role="model_under_test",
    )

    summary = tracker.summary()
    assert summary["unknown_cost_calls"] == 1
    assert summary["unknown_cost_by_model"] == {"new/direct-model": 1}


def test_cost_tracker_preserves_reported_and_estimated_cost_provenance():
    from sus_bench.api import CostTracker

    tracker = CostTracker()
    tracker.record(
        "openrouter/model",
        {"prompt_tokens": 50, "completion_tokens": 5, "cost": 0.012},
        role="judge",
    )
    tracker.record(
        "direct/model",
        {
            "prompt_tokens": 20,
            "completion_tokens": 2,
            "cost": 0.003,
            "cost_source": "direct_provider_estimate_2026_08",
        },
        role="model_under_test",
    )

    summary = tracker.summary()
    assert summary["reported_cost_usd"] == 0.012
    assert summary["estimated_cost_usd"] == 0.003
    assert summary["cost_by_source"] == {
        "direct_provider_estimate_2026_08": 0.003,
        "provider_reported": 0.012,
    }


def test_cost_tracker_prefers_reported_cost_and_preserves_microcosts():
    from sus_bench.api import CostTracker

    tracker = CostTracker()
    tracker.record(
        "provider/model",
        {
            "cost": 0.00001,
            "estimated_cost": 0.00003,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        },
        role="judge",
    )

    summary = tracker.summary()
    assert summary["total_cost_usd"] == 0.00001
    assert summary["reported_cost_usd"] == 0.00001
    assert summary["estimated_cost_usd"] == 0
    assert summary["reported_cost_usd"] + summary["estimated_cost_usd"] == summary["total_cost_usd"]
    assert summary["invalid_usage_fields"]["conflicting_cost_sources"] == 1


def test_cost_tracker_bare_estimate_uses_pricing_snapshot_source():
    from sus_bench.api import CostTracker

    tracker = CostTracker()
    tracker.record("model-a", {"estimated_cost": 0.003}, role="judge")

    summary = tracker.summary()
    assert summary["reported_cost_usd"] == 0
    assert summary["estimated_cost_usd"] == 0.003
    assert summary["cost_by_source"] == {"pricing_snapshot": 0.003}


def test_periodic_credit_stop_is_deferred_until_before_next_call(monkeypatch):
    from sus_bench.api import CostTracker, CreditExhaustedError

    tracker = CostTracker()
    tracker.set_api_key("key")
    tracker._check_every_n_calls = 1
    monkeypatch.setattr(
        tracker,
        "_check_credit_balance",
        lambda: (_ for _ in ()).throw(CreditExhaustedError("low balance")),
    )

    # A paid response is always accounted first; it is never discarded by the
    # balance check that its own call count makes due.
    tracker.record("model-a", {"cost": 0.01}, role="model_under_test")
    assert tracker.summary()["total_calls"] == 1

    with pytest.raises(CreditExhaustedError, match="low balance"):
        tracker.check_credit_if_due()


def test_success_without_usage_counts_unknown_paid_attempt_in_both_ledgers(
    tmp_path,
    monkeypatch,
):
    from suite_tools.run_monitor import RunMonitor

    response = _chat_response("ok", usage=None)
    monitor = RunMonitor(tmp_path, module="sus", stage="generation")
    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(api, "_openai_factory", _fake_openai_factory({}, response))
    api.reset_cost_tracker()

    text, _latency = api.call_openrouter(
        "provider/model",
        [{"role": "user", "content": "Hello"}],
        "secret-key",
        base_url=api.OPENROUTER_URL,
        role="model_under_test",
        monitor=monitor,
    )

    assert text == "ok"
    tracker_cost = api.get_cost_tracker().summary()
    monitor_cost = monitor.status["cost"]
    assert tracker_cost["total_calls"] == 1
    assert tracker_cost["unknown_cost_calls"] == 1
    assert monitor_cost["total_calls"] == 1
    assert monitor_cost["unknown_cost_calls"] == 1
    assert monitor_cost["usage_by_role"]["model_under_test"]["calls"] == 1


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), -1, "bad"])
def test_cost_tracker_treats_invalid_cost_as_unknown_and_normalizes_tokens(bad_cost):
    from sus_bench.api import CostTracker

    tracker = CostTracker()
    tracker.record(
        "unknown/model",
        {
            "cost": bad_cost,
            "prompt_tokens": "10",
            "completion_tokens": -2,
            "reasoning_tokens": "bad",
        },
        role="judge",
    )

    summary = tracker.summary()
    assert summary["total_cost_usd"] == 0
    assert summary["reported_cost_usd"] == 0
    assert summary["estimated_cost_usd"] == 0
    assert summary["unknown_cost_calls"] == 1
    assert summary["tokens_in"] == 10
    assert summary["tokens_out"] == 0
    assert summary["thinking_tokens_out"] == 0
    assert summary["usage_anomaly_count"] >= 3


def test_openai_compatible_empty_text_response_is_provider_failure(monkeypatch):
    response = _chat_response(
        "",
        usage={"prompt_tokens": 10, "completion_tokens": 0, "cost": 0.001},
        finish_reason="length",
    )

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(api, "_openai_factory", _fake_openai_factory({}, response))
    api.reset_cost_tracker()

    try:
        api.call_openrouter(
            "provider/model",
            [{"role": "user", "content": "Hello"}],
            "secret-key",
            base_url=api.OPENROUTER_URL,
            role="model_under_test",
        )
    except api.BenchmarkApiError as exc:
        assert exc.status_code == 502
        assert "empty message content" in exc.text
        assert "finish_reason=length" in exc.text
    else:
        raise AssertionError("empty OpenAI-compatible text should fail before scoring")

    summary = api.get_cost_tracker().summary()
    assert summary["cost_by_model"]["provider/model"] == 0.001


def test_call_provider_dispatches_to_openai_responses(monkeypatch):
    captured = {}
    receipt_events = []
    monitor = SimpleNamespace(
        record=lambda event, **fields: receipt_events.append(
            {"event": event, **fields}
        )
    )

    class Response:
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {
                "model": "gpt-5.6-luna-2026-07",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "content": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Native Responses reply"}],
                    },
                ],
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 12},
                    "total_tokens": 120,
                },
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return Response()

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    api.reset_cost_tracker()

    text, latency = api.call_openrouter(
        "gpt-5.6-luna",
        [
            {"role": "system", "content": "System context"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Continue"},
        ],
        "secret-key",
        base_url="https://api.openai.com/v1/responses",
        reasoning_effort="max",
        request_options={"max_tokens": 64000, "reasoning_effort": "max"},
        role="model_under_test",
        monitor=monitor,
        request_context={
            "condition_id": "gpt-5-6-luna-openai-native-max",
            "model_key": "gpt-5-6-luna-native-max",
        },
    )

    assert text == "Native Responses reply"
    assert latency >= 0
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    # system -> instructions; multi-turn roles preserved in input
    assert captured["payload"]["instructions"] == "System context"
    assert captured["payload"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Hi there"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "Continue"}]},
    ]
    assert captured["payload"]["reasoning"] == {"effort": "max"}
    assert captured["payload"]["max_output_tokens"] == 64000
    assert "reasoning_effort" not in captured["payload"]
    assert receipt_events == [
        {
            "event": "effective_request",
            "receipt_schema_version": "benchmark-effective-request-v1",
            "role": "model_under_test",
            "model": "gpt-5.6-luna",
            "controls_hash": receipt_events[0]["controls_hash"],
            "effective_max_output_tokens": 64000,
            "effective_reasoning_effort": "max",
            "condition_id": "gpt-5-6-luna-openai-native-max",
            "model_key": "gpt-5-6-luna-native-max",
            "call_attempt": 1,
            "provider": "openai",
            "provider_api": "openai_responses",
        }
    ]

    summary = api.get_cost_tracker().summary()
    assert summary["tokens_in"] == 100
    # reasoning tokens are part of output billing, not double-counted
    assert summary["tokens_out"] == 20


# ── T1: BenchmarkApiError raw_response attribute ─────────────────────────────


def test_benchmark_api_error_has_raw_response_attr():
    """BenchmarkApiError must expose raw_response (default None)."""
    err = api.BenchmarkApiError(403, "forbidden")
    assert hasattr(err, "raw_response")
    assert err.raw_response is None


# ── T1: ProviderApiError narrowing preserves raw_response ────────────────────


def test_provider_api_error_narrowing_preserves_raw_response(tmp_path, monkeypatch):
    """When a ProviderApiError is caught and narrowed to BenchmarkApiError,
    its raw_response must be forwarded onto the new error."""
    body = {"error": {"type": "permission_error", "message": "forbidden"}}

    def fail_client(**kwargs):
        class _Client:
            base_url = "https://openrouter.ai/api/v1"
            chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=staticmethod(
                        lambda **kw: (_ for _ in ()).throw(
                            ProviderApiError(403, "forbidden", raw_response=body)
                        )
                    )
                )
            )

        return _Client()

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(api, "_openai_factory", fail_client)
    api.reset_cost_tracker()
    from suite_tools.run_monitor import RunMonitor
    monitor = RunMonitor(tmp_path, module="sus", stage="generation")

    with pytest.raises(api.BenchmarkApiError) as exc_info:
        api.call_openrouter(
            "provider/model",
            [{"role": "user", "content": "hello"}],
            "secret-key",
            monitor=monitor,
        )

    err = exc_info.value
    assert err.status_code == 403
    assert hasattr(err, "raw_response")
    assert err.raw_response == body
    tracker_cost = api.get_cost_tracker().summary()
    assert tracker_cost["total_calls"] == 1
    assert tracker_cost["unknown_cost_calls"] == 1
    assert monitor.status["cost"]["total_calls"] == 1
    assert monitor.status["cost"]["unknown_cost_calls"] == 1


# ── T1: Generic SDK exception narrowing uses extract_raw_response ─────────────


def test_generic_sdk_exception_narrowing_extracts_raw_response(monkeypatch):
    """When a non-ProviderApiError SDK exception has a .body dict, that dict
    must be preserved on the resulting BenchmarkApiError via extract_raw_response."""
    sdk_body = {"error": {"type": "permission_denied", "metadata": {"reasons": ["policy"]}}}

    class FakeSDKError(Exception):
        status_code = 403
        body = sdk_body

    def fail_client(**kwargs):
        class _Client:
            base_url = "https://openrouter.ai/api/v1"
            chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=staticmethod(
                        lambda **kw: (_ for _ in ()).throw(FakeSDKError("sdk error"))
                    )
                )
            )

        return _Client()

    monkeypatch.setattr(api, "paid_call_lease", _no_lease)
    monkeypatch.setattr(api, "_openai_factory", fail_client)
    api.reset_cost_tracker()

    with pytest.raises(api.BenchmarkApiError) as exc_info:
        api.call_openrouter(
            "provider/model",
            [{"role": "user", "content": "hello"}],
            "secret-key",
        )

    err = exc_info.value
    assert err.status_code == 403
    assert hasattr(err, "raw_response")
    assert err.raw_response == sdk_body


# ── T1: _raw_openai_compatible_response captures native_finish_reason/refusal ─


def test_raw_openai_compatible_response_captures_native_finish_reason():
    """native_finish_reason on the choice must appear at top level of the snapshot."""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                native_finish_reason="end_turn",
                message=SimpleNamespace(content="text", refusal=None),
            )
        ]
    )
    raw = api._raw_openai_compatible_response(response)
    assert raw.get("native_finish_reason") == "end_turn"


def test_raw_openai_compatible_response_captures_message_refusal():
    """message.refusal must appear in the message dict when present."""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="content_filter",
                native_finish_reason=None,
                message=SimpleNamespace(content=None, refusal="I can't help with that"),
            )
        ]
    )
    raw = api._raw_openai_compatible_response(response)
    assert raw["choices"][0]["message"].get("refusal") == "I can't help with that"


def test_raw_openai_compatible_response_omits_native_finish_reason_when_absent():
    """When native_finish_reason is None/absent, it must not be added to the dict."""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="hello"),
            )
        ]
    )
    raw = api._raw_openai_compatible_response(response)
    assert "native_finish_reason" not in raw


# ── T1: _openai_compatible_empty_text_error structured raw_response ───────────


def test_openai_compatible_empty_text_error_sets_structured_raw_response():
    """_openai_compatible_empty_text_error must set raw_response with
    finish_reason, native_finish_reason, and refusal keys."""
    data = {
        "choices": [
            {
                "finish_reason": "content_filter",
                "message": {"content": None, "refusal": "I cannot assist"},
            }
        ],
        "native_finish_reason": "content_filter",
    }
    err = api._openai_compatible_empty_text_error(data)
    assert hasattr(err, "raw_response")
    assert isinstance(err.raw_response, dict)
    assert err.raw_response["finish_reason"] == "content_filter"
    assert err.raw_response["native_finish_reason"] == "content_filter"
    assert err.raw_response["refusal"] == "I cannot assist"
