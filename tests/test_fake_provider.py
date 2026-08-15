import httpx
import pytest

from suite_tools.fake_provider import FakeOpenAIProvider


def test_fake_provider_serves_openai_completion_and_tracks_activity():
    with FakeOpenAIProvider(latency_seconds=0.01) as provider:
        response = httpx.post(
            provider.chat_url,
            json={
                "model": "fake/flash-lite",
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 16,
            },
            timeout=2,
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "OK"
        assert response.json()["usage"] == {
            "prompt_tokens": 8,
            "completion_tokens": 1,
            "total_tokens": 9,
            "cost": 0.0,
        }
        assert provider.snapshot()["requests"] == 1
        assert provider.snapshot()["max_active"] == 1


def test_fake_provider_can_script_a_rate_limit_response():
    with FakeOpenAIProvider(script=("rate_limit",)) as provider:
        response = httpx.post(
            provider.chat_url,
            json={"model": "fake/flash-lite", "messages": []},
            timeout=2,
        )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert response.json()["error"]["type"] == "rate_limit_error"


@pytest.mark.parametrize(
    ("action", "status_code", "body"),
    [
        ("server_error", 503, "temporary fake provider failure"),
        ("empty", 200, '"content": ""'),
        ("malformed", 200, "not-json"),
    ],
)
def test_fake_provider_can_script_response_failures(action, status_code, body):
    with FakeOpenAIProvider(script=(action,)) as provider:
        response = httpx.post(
            provider.chat_url,
            json={"model": "fake/flash-lite", "messages": []},
            timeout=2,
        )

    assert response.status_code == status_code
    assert body in response.text


def test_fake_provider_can_script_a_timeout():
    with FakeOpenAIProvider(script=("timeout",), timeout_seconds=0.2) as provider:
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(
                provider.chat_url,
                json={"model": "fake/flash-lite", "messages": []},
                timeout=0.03,
            )


def test_fake_provider_can_return_detailed_usage_variant():
    with FakeOpenAIProvider(script=("detailed_usage",)) as provider:
        response = httpx.post(
            provider.chat_url,
            json={"model": "fake/flash-lite", "messages": []},
            timeout=2,
        )

    usage = response.json()["usage"]
    assert usage["prompt_tokens_details"]["cached_tokens"] == 3
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 2
