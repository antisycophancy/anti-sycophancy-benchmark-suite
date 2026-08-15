import json
import math
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from suite_tools.provider_client import (
    ANTHROPIC_MESSAGES_URL,
    ANTHROPIC_PRICE_PER_TOKEN,
    GEMINI_GENERATE_CONTENT_BASE_URL,
    OPENAI_RESPONSES_URL,
    AnthropicMessagesClient,
    GeminiGenerateContentClient,
    OpenAIResponsesClient,
    ProviderApiError,
    ProviderMalformedResponseError,
    ProviderOutputBudgetExhaustedError,
    ProviderRefusalError,
    extract_raw_response,
    inspect_chat_completion_response,
    is_openai_responses_url,
    make_provider_client,
    normalize_chat_payload_for_provider,
)


def test_chat_completion_inspection_accepts_normal_sdk_shape():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                native_finish_reason=None,
                message=SimpleNamespace(content="answer", refusal=None),
            )
        ],
        native_finish_reason=None,
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )

    inspected = inspect_chat_completion_response(response)

    assert inspected.content == "answer"
    assert inspected.response_shape is None
    assert inspected.signal_payload == {
        "finish_reason": "stop",
        "native_finish_reason": None,
        "refusal": None,
    }
    assert inspected.raw_response["choices"][0]["message"]["content"] == "answer"


@pytest.mark.parametrize(
    ("response", "expected_shape"),
    [
        (None, "response_none"),
        ({}, "choices_missing"),
        ({"choices": None}, "choices_null"),
        ({"choices": {}}, "choices_wrong_type"),
        ({"choices": []}, "choices_empty"),
        ({"choices": [None]}, "choice_null"),
        ({"choices": ["bad"]}, "choice_wrong_type"),
        ({"choices": [{"finish_reason": "stop"}]}, "message_missing"),
        ({"choices": [{"message": None}]}, "message_null"),
        ({"choices": [{"message": {}}]}, "content_missing"),
        ({"choices": [{"message": {"content": []}}]}, "content_wrong_type"),
        ({"choices": [{"message": {"content": None}}]}, "empty_content"),
    ],
)
def test_chat_completion_inspection_names_malformed_shapes(response, expected_shape):
    inspected = inspect_chat_completion_response(response)

    assert inspected.content is None
    assert inspected.response_shape == expected_shape


def test_chat_completion_inspection_preserves_signal_before_shape_failure():
    response = {
        "native_finish_reason": "SAFETY",
        "choices": [{"finish_reason": "content_filter", "message": None}],
    }

    inspected = inspect_chat_completion_response(response)

    assert inspected.response_shape == "message_null"
    assert inspected.signal_payload["finish_reason"] == "content_filter"
    assert inspected.signal_payload["native_finish_reason"] == "SAFETY"
    assert inspected.raw_response == response


def test_chat_completion_inspection_preserves_sdk_extra_error_fields():
    response = ChatCompletion.model_construct(
        id="request-1",
        object="chat.completion",
        created=1,
        model="google/gemini-test",
        choices=None,
        error={"code": "upstream_empty", "message": "empty result"},
        native_finish_reason="SAFETY",
    )

    inspected = inspect_chat_completion_response(response)

    assert inspected.response_shape == "choices_null"
    assert inspected.raw_response["error"] == {
        "code": "upstream_empty",
        "message": "empty result",
    }
    assert inspected.raw_response["native_finish_reason"] == "SAFETY"


def test_malformed_response_error_carries_shape_and_raw_body():
    error = ProviderMalformedResponseError(
        "choices_null",
        "Provider returned no usable content",
        raw_response={"choices": None, "provider": "example"},
    )

    assert error.status_code == 200
    assert error.response_shape == "choices_null"
    assert error.raw_response == {"choices": None, "provider": "example"}


def test_make_provider_client_uses_openai_factory_for_openai_compatible_endpoint():
    captured = {}

    def fake_openai_factory(**kwargs):
        captured.update(kwargs)
        return "openai-client"

    client = make_provider_client(
        {
            "api_key": "key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_api": "openai_compatible",
        },
        openai_factory=fake_openai_factory,
    )

    assert client == "openai-client"
    assert captured == {
        "api_key": "key",
        "base_url": "https://openrouter.ai/api/v1",
        "max_retries": 0,
        "timeout": 120,
    }


def test_make_provider_client_allows_openai_compatible_timeout_override():
    captured = {}

    def fake_openai_factory(**kwargs):
        captured.update(kwargs)
        return "openai-client"

    client = make_provider_client(
        {
            "api_key": "key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_api": "openai_compatible",
            "timeout": 45,
        },
        openai_factory=fake_openai_factory,
    )

    assert client == "openai-client"
    assert captured["timeout"] == 45


def test_make_provider_client_uses_anthropic_messages_client_for_native_endpoint():
    client = make_provider_client(
        {
            "api_key": "anthropic-key",
            "base_url": ANTHROPIC_MESSAGES_URL,
            "provider_api": "anthropic_messages",
        },
        openai_factory=lambda **_: pytest.fail("OpenAI factory should not be used"),
    )

    assert isinstance(client, AnthropicMessagesClient)
    assert client.api_key == "anthropic-key"
    assert client.base_url == ANTHROPIC_MESSAGES_URL


def test_make_provider_client_uses_gemini_generate_content_client_for_native_endpoint():
    client = make_provider_client(
        {
            "api_key": "gemini-key",
            "base_url": GEMINI_GENERATE_CONTENT_BASE_URL,
            "provider_api": "gemini_generate_content",
        },
        openai_factory=lambda **_: pytest.fail("OpenAI factory should not be used"),
    )

    assert isinstance(client, GeminiGenerateContentClient)
    assert client.api_key == "gemini-key"
    assert client.base_url == GEMINI_GENERATE_CONTENT_BASE_URL


def test_direct_openai_gpt5_payload_uses_max_completion_tokens():
    payload = normalize_chat_payload_for_provider(
        {"model": "gpt-5.5", "messages": [], "max_tokens": 4096, "temperature": 0},
        base_url="https://api.openai.com/v1",
    )

    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 4096
    assert "temperature" not in payload


def test_direct_openai_gpt5_payload_uses_extra_body_max_completion_tokens():
    payload = normalize_chat_payload_for_provider(
        {
            "model": "gpt-5.5",
            "messages": [],
            "max_tokens": 1000,
            "extra_body": {"max_tokens": 8192, "reasoning_effort": "high"},
        },
        base_url="https://api.openai.com/v1",
    )

    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 8192
    assert payload["extra_body"] == {"reasoning_effort": "high"}


def test_direct_openai_gpt5_normalization_preserves_reused_extra_body():
    request_options = {
        "max_tokens": 128000,
        "reasoning_effort": "max",
    }

    normalized = [
        normalize_chat_payload_for_provider(
            {
                "model": "gpt-5.6-sol",
                "messages": [],
                "max_tokens": 1000,
                "extra_body": request_options,
            },
            base_url="https://api.openai.com/v1/responses",
        )
        for _ in range(2)
    ]

    assert request_options == {
        "max_tokens": 128000,
        "reasoning_effort": "max",
    }
    assert [payload["max_completion_tokens"] for payload in normalized] == [
        128000,
        128000,
    ]
    assert all(
        payload["extra_body"] == {"reasoning_effort": "max"}
        for payload in normalized
    )
    assert all(payload["extra_body"] is not request_options for payload in normalized)


def test_direct_openai_gpt5_extra_body_token_limit_used_without_top_level_limit():
    payload = normalize_chat_payload_for_provider(
        {
            "model": "gpt-5.5",
            "messages": [],
            "extra_body": {"max_tokens": 8192, "reasoning_effort": "high"},
        },
        base_url="https://api.openai.com/v1",
    )

    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 8192
    assert payload["extra_body"] == {"reasoning_effort": "high"}


def test_openrouter_gpt5_payload_keeps_openai_compatible_max_tokens():
    payload = normalize_chat_payload_for_provider(
        {"model": "openai/gpt-5.5", "messages": [], "max_tokens": 4096, "temperature": 0},
        base_url="https://openrouter.ai/api/v1",
    )

    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload
    assert payload["temperature"] == 0


def test_anthropic_output_config_payload_drops_non_default_temperature():
    payload = normalize_chat_payload_for_provider(
        {
            "model": "claude-sonnet-5",
            "messages": [],
            "max_tokens": 4096,
            "temperature": 0,
            "output_config": {"effort": "high"},
        },
        base_url="https://api.anthropic.com/v1/messages",
    )

    assert payload["max_tokens"] == 4096
    assert payload["output_config"]["effort"] == "high"
    assert "temperature" not in payload


def test_anthropic_messages_client_translates_openai_chat_payload(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "claude-opus-4-8-20260520",
                "content": [{"type": "text", "text": "held the boundary"}],
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = AnthropicMessagesClient(api_key="anthropic-key")
    response = client.chat.completions.create(
        model="claude-opus-4-8",
        messages=[
            {"role": "system", "content": "system guidance"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        max_tokens=1000,
        temperature=0,
        timeout=42,
        extra_body={
            "max_tokens": 4096,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        },
    )

    assert captured["url"] == ANTHROPIC_MESSAGES_URL
    assert captured["headers"]["x-api-key"] == "anthropic-key"
    assert captured["timeout"] == 42
    assert captured["json"] == {
        "model": "claude-opus-4-8",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "max_tokens": 4096,
        "system": "system guidance",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert response.choices[0].message.content == "held the boundary"
    assert response.usage["prompt_tokens"] == 12
    assert response.usage["completion_tokens"] == 4
    assert response.usage["total_tokens"] == 16
    assert response.usage["cost_source"] == "anthropic_estimate"
    assert response.usage["cost"] > 0


def test_anthropic_messages_client_prices_opus_5_usage(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 100_000, "output_tokens": 10_000},
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    response = AnthropicMessagesClient(api_key="anthropic-key").chat.completions.create(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert ANTHROPIC_PRICE_PER_TOKEN["claude-opus-5"] == {
        "input": 5.0 / 1_000_000,
        "output": 25.0 / 1_000_000,
    }
    assert response.usage["cost_source"] == "anthropic_estimate"
    assert response.usage["cost"] == pytest.approx(0.75)


def test_anthropic_messages_client_surfaces_provider_refusal(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "claude-fable-5",
                "stop_reason": "refusal",
                "stop_details": {"category": "reasoning_extraction"},
                "content": [],
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = AnthropicMessagesClient(api_key="anthropic-key")
    with pytest.raises(ProviderRefusalError) as excinfo:
        client.chat.completions.create(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=128000,
            extra_body={
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "max"},
            },
        )

    refusal = excinfo.value
    assert refusal.status_code == 200
    assert refusal.stop_reason == "refusal"
    assert refusal.stop_details == {"category": "reasoning_extraction"}
    assert refusal.usage["prompt_tokens"] == 12
    assert refusal.usage["completion_tokens"] == 0
    assert refusal.usage["cost"] == 0
    assert refusal.usage["cost_source"] == "anthropic_refusal_no_charge"


def test_gemini_generate_content_client_translates_openai_chat_payload(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "modelVersion": "gemini-3.1-pro-preview-20260601",
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"thought": True, "text": "internal thinking"},
                                {"text": "held the boundary"},
                            ]
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 4,
                    "thoughtsTokenCount": 6,
                    "totalTokenCount": 22,
                },
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = GeminiGenerateContentClient(api_key="gemini-key")
    response = client.chat.completions.create(
        model="gemini-3.1-pro-preview",
        messages=[
            {"role": "system", "content": "system guidance"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        max_tokens=1000,
        temperature=0,
        timeout=42,
        extra_body={
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "HIGH",
                    "includeThoughts": False,
                }
            }
        },
    )

    assert captured["url"] == (
        f"{GEMINI_GENERATE_CONTENT_BASE_URL}/models/gemini-3.1-pro-preview:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "gemini-key"
    assert captured["timeout"] == 42
    assert captured["json"] == {
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]},
        ],
        "systemInstruction": {"parts": [{"text": "system guidance"}]},
        "generationConfig": {
            "maxOutputTokens": 1000,
            "temperature": 0,
            "thinkingConfig": {
                "thinkingLevel": "HIGH",
                "includeThoughts": False,
            },
        },
    }
    assert response.choices[0].message.content == "held the boundary"
    assert response.usage["prompt_tokens"] == 12
    assert response.usage["completion_tokens"] == 4
    assert response.usage["thoughts_tokens"] == 6
    assert response.usage["billable_completion_tokens"] == 10
    assert response.usage["cost_source"] == "gemini_standard_estimate_2026_06_prompt_le_200k"
    assert response.usage["estimated_cost"] == pytest.approx(0.000144)
    assert response.usage["total_tokens"] == 22


def test_gemini_native_rejects_openrouter_style_reasoning_options():
    client = GeminiGenerateContentClient(api_key="gemini-key")

    with pytest.raises(ValueError) as exc:
        client.chat.completions.create(
            model="gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "hello"}],
            extra_body={"reasoning": {"effort": "high"}},
        )

    assert "native Gemini request_options" in str(exc.value)
    assert "reasoning" in str(exc.value)


def test_anthropic_messages_client_records_429_cooldown(monkeypatch):
    cooldowns = []

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=429,
            headers={"x-ratelimit-reset": "1"},
            text="rate limited",
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **kwargs: cooldowns.append(kwargs),
    )

    client = AnthropicMessagesClient(api_key="anthropic-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 429
    assert cooldowns
    assert cooldowns[0]["provider"] == "anthropic"
    assert cooldowns[0]["model"] == "claude-opus-4-8"
    assert cooldowns[0]["headers"] == {"x-ratelimit-reset": "1"}


@pytest.mark.parametrize(
    "client,model",
    [
        (GeminiGenerateContentClient(api_key="gemini-key"), "gemini-2.5-flash-lite"),
        (AnthropicMessagesClient(api_key="anthropic-key"), "claude-haiku"),
    ],
)
def test_native_clients_normalize_malformed_http_200_json(client, model, monkeypatch):
    class MalformedResponse:
        status_code = 200
        headers = {}
        text = "upstream returned html"

        def json(self):
            raise json.JSONDecodeError("bad", self.text, 0)

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", lambda *args, **kwargs: MalformedResponse())

    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 502
    assert "invalid JSON" in exc.value.text
    assert exc.value.raw_response["response_shape"] == "invalid_json"
    assert len(exc.value.raw_response["raw_body_sha256"]) == 64


def test_gemini_generate_content_client_records_429_cooldown(monkeypatch):
    cooldowns = []

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=429,
            headers={"retry-after": "2"},
            text="rate limited",
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **kwargs: cooldowns.append(kwargs),
    )

    client = GeminiGenerateContentClient(api_key="gemini-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 429
    assert cooldowns[0]["provider"] == "google"
    assert cooldowns[0]["model"] == "gemini-2.5-flash-lite"
    assert cooldowns[0]["headers"] == {"retry-after": "2"}


def test_estimate_usage_cost_prices_gpt_5_6_family():
    from suite_tools.provider_client import estimate_usage_cost

    for model, expected in [
        ("gpt-5.6-sol", 100_000 * 5.0 / 1e6 + 10_000 * 30.0 / 1e6),
        ("gpt-5.6-terra", 100_000 * 2.5 / 1e6 + 10_000 * 15.0 / 1e6),
        ("gpt-5.6-luna", 100_000 * 1.0 / 1e6 + 10_000 * 6.0 / 1e6),
    ]:
        usage = {"prompt_tokens": 100_000, "completion_tokens": 10_000}
        enriched = estimate_usage_cost(model, usage)
        assert enriched.get("estimated_cost") == pytest.approx(expected), model
        assert enriched.get("cost_source") == "openai_standard_estimate_2026_07"


def test_estimate_usage_cost_gpt_5_6_none_effort_same_pricing_as_tier():
    # The none-effort conditions (gpt-5-6-{sol,terra,luna}-native-none) share the
    # same model_id as the other native-effort conditions; pricing must resolve
    # identically so cost accounting is consistent across all effort arms.
    from suite_tools.provider_client import estimate_usage_cost

    usage = {"prompt_tokens": 50_000, "completion_tokens": 5_000}
    for model_id, expected_input, expected_output in [
        ("gpt-5.6-sol", 5.0, 30.0),
        ("gpt-5.6-terra", 2.5, 15.0),
        ("gpt-5.6-luna", 1.0, 6.0),
    ]:
        expected = 50_000 * expected_input / 1e6 + 5_000 * expected_output / 1e6
        enriched = estimate_usage_cost(model_id, usage)
        assert enriched.get("estimated_cost") == pytest.approx(expected), model_id
        assert enriched.get("cost_source") == "openai_standard_estimate_2026_07", model_id


def test_estimate_usage_cost_preserves_valid_provider_zero():
    from suite_tools.provider_client import estimate_usage_cost

    usage = {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0}

    assert estimate_usage_cost("gpt-5.6-luna", usage) == usage


def test_invalid_provider_cost_is_flagged_while_estimate_remains_finite(tmp_path):
    from sus_bench.api import CostTracker
    from suite_tools.provider_client import estimate_usage_cost
    from suite_tools.run_monitor import RunMonitor

    enriched = estimate_usage_cost(
        "gpt-5.6-luna",
        {"prompt_tokens": 100, "completion_tokens": 10, "cost": float("nan")},
    )
    assert math.isnan(enriched["cost"])
    assert enriched["estimated_cost"] == pytest.approx(0.00016)

    tracker = CostTracker()
    monitor = RunMonitor(tmp_path, module="sus", stage="generation")
    tracker.record("gpt-5.6-luna", enriched, role="model_under_test")
    monitor.record_usage(
        "gpt-5.6-luna",
        enriched,
        role="model_under_test",
        provider="openai",
    )

    tracker_summary = tracker.summary()
    monitor_cost = json.loads((tmp_path / "RUN_STATUS.json").read_text())["cost"]
    assert tracker_summary["reported_cost_usd"] == 0
    assert monitor_cost["reported_cost_usd"] == 0
    assert tracker_summary["estimated_cost_usd"] == pytest.approx(0.00016)
    assert monitor_cost["estimated_cost_usd"] == pytest.approx(0.00016)
    assert tracker_summary["invalid_usage_fields"]["cost"] == 1
    assert monitor_cost["invalid_usage_fields"]["cost"] == 1
    json.dumps(monitor_cost, allow_nan=False)


# ── OpenAI Responses API ─────────────────────────────────────────────────────


def _responses_completed(text, usage):
    return {
        "model": "gpt-5.6-luna-2026-07",
        "status": "completed",
        "incomplete_details": None,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": usage,
    }


def test_make_provider_client_uses_openai_responses_client_for_provider_api():
    client = make_provider_client(
        {
            "api_key": "openai-key",
            "base_url": OPENAI_RESPONSES_URL,
            "provider_api": "openai_responses",
        },
        openai_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("used factory")),
    )
    assert isinstance(client, OpenAIResponsesClient)
    assert client.base_url == OPENAI_RESPONSES_URL


def test_is_openai_responses_url_matches_responses_endpoint():
    assert is_openai_responses_url("https://api.openai.com/v1/responses")
    assert not is_openai_responses_url("https://api.openai.com/v1/chat/completions")
    assert not is_openai_responses_url("https://openrouter.ai/api/v1")
    assert not is_openai_responses_url(
        "https://api.openai.com.attacker.example/v1/responses"
    )


def test_openai_responses_client_translates_single_turn_payload(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: _responses_completed(
                "held the boundary",
                {
                    "input_tokens": 20,
                    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 25,
                },
            ),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": "system guidance"},
            {"role": "user", "content": "hello"},
        ],
        max_tokens=64000,
        reasoning_effort="max",
        timeout=42,
    )

    assert captured["url"] == OPENAI_RESPONSES_URL
    assert captured["headers"]["Authorization"] == "Bearer openai-key"
    assert captured["timeout"] == 42
    assert captured["json"] == {
        "model": "gpt-5.6-luna",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ],
        "instructions": "system guidance",
        "max_output_tokens": 64000,
        "reasoning": {"effort": "max"},
    }
    assert response.choices[0].message.content == "held the boundary"
    # reasoning_tokens are part of output billing; output_tokens already includes them.
    assert response.usage["prompt_tokens"] == 20
    assert response.usage["completion_tokens"] == 5
    assert response.usage["reasoning_tokens"] == 3
    assert response.usage["total_tokens"] == 25
    assert response.usage["cost_source"] == "openai_standard_estimate_2026_07"
    # luna: 20 in * 1/1e6 + 5 out * 6/1e6
    assert response.usage["estimated_cost"] == pytest.approx(20 * 1 / 1e6 + 5 * 6 / 1e6)


def test_openai_responses_client_multi_turn_preserves_roles(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: _responses_completed(
                "Your name is Sam.",
                {"input_tokens": 33, "output_tokens": 9, "total_tokens": 42},
            ),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "My name is Sam."},
            {"role": "assistant", "content": "Nice to meet you, Sam."},
            {"role": "user", "content": "What is my name?"},
        ],
        reasoning_effort="low",
        max_tokens=64,
    )

    assert captured["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "My name is Sam."}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Nice to meet you, Sam."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "What is my name?"}]},
    ]
    assert response.choices[0].message.content == "Your name is Sam."


def test_openai_responses_client_reads_effort_and_limit_from_extra_body(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: _responses_completed(
                "ok", {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
            ),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    client.chat.completions.create(
        model="gpt-5.6-terra",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"reasoning_effort": "xhigh", "max_tokens": 64000},
    )

    assert captured["json"]["reasoning"] == {"effort": "xhigh"}
    assert captured["json"]["max_output_tokens"] == 64000
    # translated fields must not leak into the Responses payload
    assert "reasoning_effort" not in captured["json"]
    assert "max_tokens" not in captured["json"]


def test_openai_responses_client_consumes_shadowed_extra_body_token_limits(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: _responses_completed(
                "ok", {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
            ),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    client.chat.completions.create(
        model="gpt-5.6-terra",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1000,
        extra_body={"max_tokens": 128000, "reasoning_effort": "high"},
    )

    assert captured["json"]["max_output_tokens"] == 1000
    assert captured["json"]["reasoning"] == {"effort": "high"}
    assert "max_tokens" not in captured["json"]


def test_openai_responses_client_accepts_max_completion_tokens_kwarg(monkeypatch):
    # sus/aita/epis run normalize_chat_payload_for_provider first, which rewrites
    # max_tokens -> max_completion_tokens for api.openai.com gpt-5 models.
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["json"] = json
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: _responses_completed(
                "ok", {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
            ),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    client.chat.completions.create(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=64000,
        reasoning_effort="high",
    )

    assert captured["json"]["max_output_tokens"] == 64000
    assert "max_completion_tokens" not in captured["json"]


def test_openai_responses_usage_bills_cached_at_discount_without_cache_write_surcharge(monkeypatch):
    # Billing correctness: cached_tokens billed at cached_input rate; cache_write
    # is NOT a separate OpenAI charge (already covered by uncached input) and must
    # not be surcharged. Provenance is preserved in input_tokens_details.
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: _responses_completed(
                "ok",
                {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cache_write_tokens": 300, "cached_tokens": 400},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 40},
                    "total_tokens": 1100,
                },
            ),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    response = client.chat.completions.create(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )
    usage = response.usage
    # sol: input 5/1e6, cached_input 0.5/1e6, output 30/1e6
    uncached = 1000 - 400
    expected = uncached * 5 / 1e6 + 400 * 0.5 / 1e6 + 100 * 30 / 1e6
    assert usage["estimated_cost"] == pytest.approx(expected)
    # cache_write surfaced for provenance, never added to cost
    assert usage["input_tokens_details"]["cache_write_tokens"] == 300


def test_openai_responses_client_maps_content_filter_to_refusal(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "gpt-5.6-sol",
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [{"type": "reasoning", "content": []}],
                "usage": {"input_tokens": 30, "output_tokens": 0, "total_tokens": 30},
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderRefusalError) as excinfo:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
    refusal = excinfo.value
    assert refusal.status_code == 200
    assert "content_filter" in refusal.text
    assert refusal.usage["prompt_tokens"] == 30


def test_openai_responses_client_maps_refusal_part_to_refusal(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderRefusalError) as excinfo:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert "I can't help with that." in excinfo.value.text


def test_openai_responses_output_budget_exhausted_is_budget_error(monkeypatch):
    # incomplete + max_output_tokens + reasoning-only (no output_text) surfaces as
    # ProviderOutputBudgetExhaustedError. It is empirically a STOCHASTIC runaway
    # reasoning loop, so it is a RETRYABLE (bounded) outcome — deliberately NOT a
    # ProviderRefusalError subclass (refusals are immediately terminal). It is
    # HTTP-200-shaped with billed usage attached so cost accounting stays correct.
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "gpt-5.6-luna",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "reasoning", "content": []}],
                "usage": {
                    "input_tokens": 16,
                    "output_tokens": 16,
                    "output_tokens_details": {"reasoning_tokens": 16},
                    "total_tokens": 32,
                },
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderOutputBudgetExhaustedError) as excinfo:
        client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
    error = excinfo.value
    # HTTP-200-shaped and billable, but decoupled from the refusal taxonomy so the
    # runners' `isinstance(e, ProviderRefusalError): raise` short-circuit does NOT
    # make it immediately terminal.
    assert isinstance(error, ProviderApiError)
    assert not isinstance(error, ProviderRefusalError)
    assert error.status_code == 200
    assert "max_output_tokens" in error.text
    assert error.usage["prompt_tokens"] == 16
    assert error.usage["completion_tokens"] == 16
    assert error.usage.get("reasoning_tokens") == 16
    # It must NOT be classified as a non-retryable provider error: the runner
    # applies a bounded retry (BENCHMARK_OUTPUT_BUDGET_RETRIES) before excluding it.
    from suite_tools.run_monitor import is_non_retryable_provider_error

    assert not is_non_retryable_provider_error(error)


def test_openai_responses_transient_502_is_not_budget_exhausted(monkeypatch):
    # A genuine HTTP 5xx server error (status != 200, not an incomplete-budget
    # reply) must stay a plain retryable ProviderApiError, never budget-exhausted.
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=502,
            headers={},
            text="Bad Gateway",
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError) as excinfo:
        client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "hi"}],
        )
    error = excinfo.value
    assert not isinstance(error, ProviderRefusalError)
    assert not isinstance(error, ProviderOutputBudgetExhaustedError)
    assert error.status_code == 502


def test_openai_responses_empty_text_without_budget_reason_stays_retryable_502(monkeypatch):
    # incomplete WITHOUT max_output_tokens and no text is still the generic
    # retryable empty-content 502 — not the terminal budget-exhausted outcome.
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "gpt-5.6-luna",
                "status": "incomplete",
                "incomplete_details": {"reason": "other"},
                "output": [{"type": "reasoning", "content": []}],
                "usage": {"input_tokens": 4, "output_tokens": 4, "total_tokens": 8},
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError) as excinfo:
        client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "hi"}],
        )
    error = excinfo.value
    assert not isinstance(error, ProviderOutputBudgetExhaustedError)
    assert error.status_code == 502


def test_openai_responses_client_records_429_cooldown(monkeypatch):
    cooldowns = []

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=429,
            headers={"retry-after": "3"},
            text="rate limited",
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **kwargs: cooldowns.append(kwargs),
    )

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert exc.value.status_code == 429
    assert cooldowns
    assert cooldowns[0]["provider"] == "openai"
    assert cooldowns[0]["model"] == "gpt-5.6-sol"


def test_openai_insufficient_quota_429_skips_rate_limit_cooldown(monkeypatch):
    cooldowns = []

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=429,
            headers={},
            text='{"error":{"code":"insufficient_quota"}}',
            json=lambda: {
                "error": {
                    "code": "insufficient_quota",
                    "message": "You exceeded your current quota",
                }
            },
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **kwargs: cooldowns.append(kwargs),
    )

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError):
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert cooldowns == []


def test_openai_responses_client_normalizes_malformed_http_200_json(monkeypatch):
    class MalformedResponse:
        status_code = 200
        headers = {}
        text = "upstream returned html"

        def json(self):
            raise json.JSONDecodeError("bad", self.text, 0)

    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post", lambda *a, **k: MalformedResponse()
    )
    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert exc.value.status_code == 502
    assert "invalid JSON" in exc.value.text
    assert exc.value.raw_response["response_shape"] == "invalid_json"
    assert len(exc.value.raw_response["raw_body_sha256"]) == 64


# ── Content-policy 400 → ProviderRefusalError (not FatalBenchmarkApiError) ────


def _make_400_response(body: dict, headers: dict | None = None):
    import json as _json

    raw = _json.dumps(body)
    return SimpleNamespace(
        status_code=400,
        headers=headers or {},
        text=raw,
        json=lambda: body,
    )


def test_openai_responses_client_cyber_policy_400_raises_provider_refusal(monkeypatch):
    """HTTP 400 with code=cyber_policy must raise ProviderRefusalError, not ProviderApiError."""
    body = {
        "error": {
            "message": "This content was flagged by our safety systems.",
            "type": "invalid_request_error",
            "code": "cyber_policy",
        }
    }
    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post",
        lambda *a, **k: _make_400_response(body),
    )
    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderRefusalError) as exc:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "harmful request"}],
        )
    assert "content-policy" in exc.value.text.lower() or "cyber_policy" in exc.value.text.lower()


def test_openai_responses_client_content_policy_code_400_raises_provider_refusal(monkeypatch):
    """HTTP 400 with code=content_policy must raise ProviderRefusalError."""
    body = {
        "error": {
            "message": "Request blocked by content policy.",
            "type": "invalid_request_error",
            "code": "content_policy",
        }
    }
    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post",
        lambda *a, **k: _make_400_response(body),
    )
    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderRefusalError):
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "harmful request"}],
        )


def test_openai_responses_client_flagged_message_400_raises_provider_refusal(monkeypatch):
    """HTTP 400 whose error message contains 'content was flagged' must raise ProviderRefusalError."""
    body = {
        "error": {
            "message": "This content was flagged by our moderation system.",
            "type": "invalid_request_error",
            "code": "invalid_request_error",
        }
    }
    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post",
        lambda *a, **k: _make_400_response(body),
    )
    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderRefusalError):
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "harmful request"}],
        )


def test_openai_responses_client_malformed_request_400_remains_hard_error(monkeypatch):
    """HTTP 400 from a bad parameter (not content policy) must NOT become ProviderRefusalError."""
    body = {
        "error": {
            "message": "Unknown field 'max_tokens'. Did you mean 'max_output_tokens'?",
            "type": "invalid_request_error",
            "code": "unknown_parameter",
        }
    }
    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post",
        lambda *a, **k: _make_400_response(body),
    )
    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert exc.value.status_code == 400
    assert not isinstance(exc.value, ProviderRefusalError)


def test_openai_responses_client_200_completion_unaffected_by_content_policy_check(monkeypatch):
    """Normal 200 completions must still succeed; content-policy check must not interfere."""
    monkeypatch.setattr(
        "suite_tools.provider_client.httpx.post",
        lambda *a, **k: SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: _responses_completed(
                "all good",
                {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            ),
        ),
    )
    client = OpenAIResponsesClient(api_key="openai-key")
    response = client.chat.completions.create(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert response.choices[0].message.content == "all good"


# ── T1: extract_raw_response ──────────────────────────────────────────────────


def test_extract_raw_response_returns_existing_dict_raw_response():
    """exc already has a dict raw_response → pass it through unchanged."""
    body = {"error": {"code": "permission_denied"}}
    exc = ProviderApiError(403, "forbidden", raw_response=body)
    assert extract_raw_response(exc) is body


def test_extract_raw_response_reads_openai_sdk_body_attribute():
    """OpenAI SDK exceptions expose .body as a dict."""
    body = {"error": {"type": "invalid_api_key", "code": "invalid_api_key"}}
    exc = RuntimeError("SDK error")
    exc.body = body  # type: ignore[attr-defined]
    result = extract_raw_response(exc)
    assert result == body


def test_extract_raw_response_reads_response_json_fallback():
    """Fall back to exc.response.json() when .body is absent."""
    body = {"error": {"message": "forbidden"}}
    exc = RuntimeError("SDK error")
    exc.response = SimpleNamespace(json=lambda: body)  # type: ignore[attr-defined]
    result = extract_raw_response(exc)
    assert result == body


def test_extract_raw_response_returns_none_when_no_structured_body():
    """Plain exceptions with no raw_response, body, or response → None."""
    exc = RuntimeError("plain error")
    assert extract_raw_response(exc) is None


def test_extract_raw_response_returns_none_for_non_json_response():
    """response.json() that throws ValueError → None, no crash."""

    def bad_json():
        raise ValueError("not JSON")

    exc = RuntimeError("error")
    exc.response = SimpleNamespace(json=bad_json)  # type: ignore[attr-defined]
    assert extract_raw_response(exc) is None


def test_extract_raw_response_ignores_non_dict_body():
    """.body that is not a dict (e.g. a string) → None rather than returning it."""
    exc = RuntimeError("error")
    exc.body = "plain string"  # type: ignore[attr-defined]
    assert extract_raw_response(exc) is None


def test_extract_raw_response_ignores_none_raw_response():
    """raw_response=None on a ProviderApiError → fall through and return None."""
    exc = ProviderApiError(503, "service unavailable")
    # raw_response defaults to None — extract_raw_response should return None
    assert extract_raw_response(exc) is None


# ── T1: ProviderApiError raw_response field ───────────────────────────────────


def test_provider_api_error_has_raw_response_none_by_default():
    err = ProviderApiError(403, "forbidden")
    assert hasattr(err, "raw_response")
    assert err.raw_response is None


def test_provider_api_error_stores_raw_response_when_provided():
    body = {"error": {"code": "permission_denied"}}
    err = ProviderApiError(403, "forbidden", raw_response=body)
    assert err.raw_response is body


# ── T1: D12 — subclass raw_response preserved through super() ─────────────────


def test_refusal_error_raw_response_preserved_through_super():
    """ProviderRefusalError must NOT clobber its own raw_response via super()."""
    raw = {"stop_reason": "refusal", "stop_details": {"category": "harm"}}
    exc = ProviderRefusalError("refused", raw_response=raw)
    assert isinstance(exc.raw_response, dict)
    assert exc.raw_response.get("stop_reason") == "refusal"
    assert exc.stop_reason == "refusal"
    assert exc.stop_details == {"category": "harm"}


def test_refusal_error_raw_response_defaults_to_empty_dict():
    """raw_response=None → self.raw_response is {} (not None)."""
    exc = ProviderRefusalError("refused")
    assert exc.raw_response == {}
    assert exc.stop_reason is None
    assert exc.stop_details == {}


def test_budget_exhausted_raw_response_preserved_through_super():
    """ProviderOutputBudgetExhaustedError must NOT clobber raw_response via super()."""
    raw = {"stop_reason": "max_output_tokens", "usage": {"output_tokens": 8000}}
    exc = ProviderOutputBudgetExhaustedError("budget exhausted", raw_response=raw)
    assert isinstance(exc.raw_response, dict)
    assert exc.raw_response.get("stop_reason") == "max_output_tokens"
    assert exc.stop_reason == "max_output_tokens"


def test_budget_exhausted_raw_response_defaults_to_empty_dict():
    """raw_response=None → self.raw_response is {} (not None)."""
    exc = ProviderOutputBudgetExhaustedError("budget exhausted")
    assert exc.raw_response == {}
    assert exc.stop_reason is None


# ── T1: Gemini non-200 attaches raw_response ─────────────────────────────────


def test_gemini_non_200_attaches_parsed_json_raw_response(monkeypatch):
    """Gemini 403 with JSON body → ProviderApiError.raw_response is the parsed dict."""
    body = {"error": {"status": "PERMISSION_DENIED", "message": "billing disabled"}}

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=403,
            headers={},
            text=json.dumps(body),
            json=lambda: body,
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **_: None,
    )

    client = GeminiGenerateContentClient(api_key="gemini-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 403
    assert isinstance(exc.value.raw_response, dict)
    assert exc.value.raw_response == body


def test_gemini_non_200_non_json_body_raw_response_is_none(monkeypatch):
    """Gemini 503 with non-JSON body → raw_response is None, no crash."""

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=503,
            headers={},
            text="Service Unavailable",
            json=lambda: (_ for _ in ()).throw(json.JSONDecodeError("bad", "doc", 0)),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = GeminiGenerateContentClient(api_key="gemini-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 503
    assert exc.value.raw_response is None


# ── T1: Anthropic non-200 attaches raw_response ──────────────────────────────


def test_anthropic_non_200_attaches_parsed_json_raw_response(monkeypatch):
    """Anthropic 403 with JSON body → ProviderApiError.raw_response is the parsed dict."""
    body = {"error": {"type": "permission_error", "message": "not permitted"}}

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=403,
            headers={},
            text=json.dumps(body),
            json=lambda: body,
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **_: None,
    )

    client = AnthropicMessagesClient(api_key="anthropic-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 403
    assert isinstance(exc.value.raw_response, dict)
    assert exc.value.raw_response == body


def test_anthropic_non_200_non_json_body_raw_response_is_none(monkeypatch):
    """Anthropic 503 with non-JSON body → raw_response is None, no crash."""

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=503,
            headers={},
            text="Service Unavailable",
            json=lambda: (_ for _ in ()).throw(ValueError("not JSON")),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = AnthropicMessagesClient(api_key="anthropic-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 503
    assert exc.value.raw_response is None


# ── T1: _anthropic_empty_text_error structured raw_response ──────────────────


def test_anthropic_empty_text_error_sets_structured_raw_response(monkeypatch):
    """_anthropic_empty_text_error must set raw_response with finish_reason,
    native_finish_reason, and refusal keys (matching the openai-compatible shape)."""
    from suite_tools.provider_client import _anthropic_empty_text_error

    data = {
        "stop_reason": "max_tokens",
        "content": [{"type": "thinking", "thinking": "internal"}],
    }
    err = _anthropic_empty_text_error(data)

    assert hasattr(err, "raw_response")
    assert isinstance(err.raw_response, dict)
    assert err.raw_response["finish_reason"] == "max_tokens"
    assert err.raw_response["native_finish_reason"] == "max_tokens"
    assert "refusal" in err.raw_response


# ── T1 follow-up: OpenAI Responses non-200 raw_response ──────────────────────


def test_openai_responses_non_200_attaches_parsed_json_raw_response(monkeypatch):
    """Responses API 403 with JSON body → ProviderApiError.raw_response is the parsed dict."""
    body = {"error": {"message": "Forbidden", "type": "permission_error", "code": "permission_denied"}}

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=403,
            headers={},
            text=json.dumps(body),
            json=lambda: body,
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)
    monkeypatch.setattr(
        "suite_tools.provider_client.record_rate_limit_cooldown",
        lambda **_: None,
    )

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 403
    assert not isinstance(exc.value, ProviderRefusalError)
    assert isinstance(exc.value.raw_response, dict)
    assert exc.value.raw_response == body


def test_openai_responses_non_200_non_json_body_raw_response_is_none(monkeypatch):
    """Responses API 503 with non-JSON body → raw_response is None, no crash."""

    def fake_post(*_, **__):
        return SimpleNamespace(
            status_code=503,
            headers={},
            text="Service Unavailable",
            json=lambda: (_ for _ in ()).throw(json.JSONDecodeError("bad", "doc", 0)),
        )

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = OpenAIResponsesClient(api_key="openai-key")
    with pytest.raises(ProviderApiError) as exc:
        client.chat.completions.create(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc.value.status_code == 503
    assert exc.value.raw_response is None


# ── T1 follow-up: _responses_empty_text_error structured raw_response ─────────


def test_responses_empty_text_error_sets_structured_raw_response():
    """_responses_empty_text_error must set raw_response with finish_reason,
    native_finish_reason, and refusal keys matching the openai-compatible shape."""
    from suite_tools.provider_client import _responses_empty_text_error

    data = {
        "status": "incomplete",
        "incomplete_details": {"reason": "other"},
        "output": [{"type": "reasoning"}],
    }
    err = _responses_empty_text_error(data)

    assert hasattr(err, "raw_response")
    assert isinstance(err.raw_response, dict)
    assert "finish_reason" in err.raw_response
    assert "native_finish_reason" in err.raw_response
    assert "refusal" in err.raw_response
    assert err.raw_response["native_finish_reason"] == "other"


def test_responses_empty_text_error_structured_raw_response_no_reason():
    """When incomplete_details is absent the finish_reason falls back to the status."""
    from suite_tools.provider_client import _responses_empty_text_error

    data = {"status": "incomplete", "output": [{"type": "reasoning"}]}
    err = _responses_empty_text_error(data)

    assert isinstance(err.raw_response, dict)
    assert err.raw_response["finish_reason"] == "incomplete"
    assert err.raw_response["refusal"] is None
