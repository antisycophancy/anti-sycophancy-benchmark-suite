import importlib.util
import sys
from pathlib import Path

import httpx
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "adapter" / "smoke.py"
SPEC = importlib.util.spec_from_file_location("adapter_smoke", MODULE_PATH)
adapter_smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter_smoke
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter_smoke)


def smoke_client(*, mode="reference_response", model="local/example-model"):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "mode": mode})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Smoke passed."},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_reference_smoke_checks_health_models_and_chat():
    client, calls = smoke_client()

    result = adapter_smoke.run_smoke(
        base_url="http://adapter.test/v1",
        client=client,
    )

    assert result == {
        "status": "pass",
        "mode": "reference_response",
        "model": "local/example-model",
        "finish_reason": "stop",
        "assistant_text_chars": 13,
    }
    assert calls == [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
    ]


def test_proxy_smoke_requires_explicit_permission_before_chat_call():
    client, calls = smoke_client(mode="proxy")

    with pytest.raises(RuntimeError, match="--allow-proxy-call"):
        adapter_smoke.run_smoke(base_url="http://adapter.test", client=client)

    assert calls == [("GET", "/health"), ("GET", "/v1/models")]


def test_smoke_requires_expected_model_to_be_advertised():
    client, calls = smoke_client()

    with pytest.raises(RuntimeError, match="expected model is not advertised"):
        adapter_smoke.run_smoke(
            base_url="http://adapter.test",
            expected_model="missing/model",
            client=client,
        )

    assert calls == [("GET", "/health"), ("GET", "/v1/models")]


def test_api_key_can_be_loaded_from_named_environment_variable(monkeypatch):
    monkeypatch.setenv("PRIVATE_ADAPTER_KEY", "secret-value")

    assert adapter_smoke.resolve_api_key(api_key="", api_key_env="PRIVATE_ADAPTER_KEY") == "secret-value"


def test_named_api_key_environment_variable_must_be_nonempty(monkeypatch):
    monkeypatch.delenv("PRIVATE_ADAPTER_KEY", raising=False)

    with pytest.raises(RuntimeError, match="PRIVATE_ADAPTER_KEY"):
        adapter_smoke.resolve_api_key(api_key="", api_key_env="PRIVATE_ADAPTER_KEY")


def test_literal_api_key_is_not_accepted_in_process_arguments():
    parser = adapter_smoke.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--api-key", "literal"])


def test_unknown_adapter_mode_requires_paid_call_permission():
    client, calls = smoke_client(mode=None)

    with pytest.raises(RuntimeError, match="cannot prove reference mode"):
        adapter_smoke.run_smoke(base_url="http://adapter.test", client=client)

    assert calls == [("GET", "/health"), ("GET", "/v1/models")]
