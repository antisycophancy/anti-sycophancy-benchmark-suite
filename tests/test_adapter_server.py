import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "adapter"
SERVER_PATH = ADAPTER_DIR / "server.py"

DEFAULT_ENV = {
    "ADAPTER_DIAGNOSTICS_INCLUDE_DETAIL": "",
    "ADAPTER_DIAGNOSTICS_MAX_BYTES": "16777216",
    "ADAPTER_DIAGNOSTICS_PATH": "",
    "ADAPTER_DEBUG_UPSTREAM_ERRORS": "",
    "ADAPTER_HOST": "127.0.0.1",
    "ADAPTER_INBOUND_API_KEY": "",
    "ADAPTER_MAX_REQUEST_BYTES": "1048576",
    "ADAPTER_PORT": "9999",
    "EXPOSED_MODEL_ID": "local/example-model",
    "REFERENCE_RESPONSE": "Reference adapter test response.",
    "REQUEST_TIMEOUT_SECONDS": "120",
    "UPSTREAM_API_KEY": "",
    "UPSTREAM_API_KEY_ENV": "",
    "UPSTREAM_CHAT_COMPLETIONS_URL": "",
    "UPSTREAM_MODEL_ID": "",
    "UPSTREAM_OPENAI_BASE_URL": "",
}

CHAT_BODY = {
    "model": "local/example-model",
    "messages": [{"role": "user", "content": "Hello"}],
}


def load_server(monkeypatch, **env):
    monkeypatch.syspath_prepend(str(ADAPTER_DIR))
    for key, value in DEFAULT_ENV.items():
        monkeypatch.setenv(key, value)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    for module_name in [
        "backend",
        "config",
        "diagnostics",
        "model_routing",
        "openai_contract",
        "adapter_server_under_test",
    ]:
        sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location("adapter_server_under_test", SERVER_PATH)
    server = importlib.util.module_from_spec(spec)
    sys.modules["adapter_server_under_test"] = server
    assert spec.loader is not None
    spec.loader.exec_module(server)
    server.app.state.allow_unauthenticated_loopback = True
    return server


class StubResponse:
    def __init__(self, status_code=200, json_body=None, text="", json_error=False):
        self.status_code = status_code
        self._json_body = json_body if json_body is not None else {}
        self.text = text
        self.content = text.encode()
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("invalid json")
        return self._json_body


def stub_async_client(monkeypatch, server, response):
    calls = []

    class StubAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers, timeout):
            calls.append(
                {
                    "url": url,
                    "json": json,
                    "headers": headers,
                    "timeout": timeout,
                }
            )
            if isinstance(response, BaseException):
                raise response
            return response

    monkeypatch.setattr(server.backend.httpx, "AsyncClient", StubAsyncClient)
    return calls


def test_reference_mode_returns_deterministic_chat_completion(monkeypatch):
    server = load_server(monkeypatch)

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "local/example-model"
    assert body["choices"][0]["message"]["content"] == "Reference adapter test response."


def test_relative_diagnostics_path_is_resolved_from_adapter_directory(monkeypatch):
    server = load_server(
        monkeypatch,
        ADAPTER_DIAGNOSTICS_PATH="results/custom-diagnostics.jsonl",
    )

    assert sys.modules["config"].ADAPTER_DIAGNOSTICS_PATH == (
        ADAPTER_DIR / "results" / "custom-diagnostics.jsonl"
    )


def test_missing_user_message_returns_400(monkeypatch):
    server = load_server(monkeypatch)

    response = TestClient(server.app).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "assistant", "content": "No user turn"}]},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "At least one non-empty user message is required",
        "code": "missing_user_message",
    }


def test_proxy_mode_attaches_upstream_bearer_and_normalizes_upstream_body(monkeypatch):
    server = load_server(
        monkeypatch,
        UPSTREAM_API_KEY="stub-upstream-key",
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    upstream_body = {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "model": "upstream/model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "stubbed upstream"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "private_trace": "omit"},
        "private_debug": {"omit": True},
    }
    calls = stub_async_client(monkeypatch, server, StubResponse(json_body=upstream_body))

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "local/example-model"
    assert body["choices"][0]["message"]["content"] == "stubbed upstream"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 4, "completion_tokens": 2}
    assert "private_debug" not in body
    assert calls == [
        {
            "url": "https://upstream.example/v1/chat/completions",
            "json": {**CHAT_BODY, "model": "local/example-model"},
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer stub-upstream-key",
            },
            "timeout": 120.0,
        }
    ]


def test_proxy_models_response_declares_safe_mode_without_leaking_on_health(monkeypatch):
    server = load_server(
        monkeypatch,
        ADAPTER_INBOUND_API_KEY="inbound-test-key",
        UPSTREAM_API_KEY="stub-upstream-key",
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    client = TestClient(server.app)

    health = client.get("/health")
    models = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer inbound-test-key"},
    )

    assert health.json() == {"status": "ok"}
    assert models.status_code == 200
    assert models.headers["x-antisycophancy-adapter-mode"] == "proxy"


def test_proxy_preserves_full_history_and_caller_correlation(monkeypatch):
    body = {
        "model": "local/example-model",
        "conversation_id": "diagnostic-tail-001",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ],
    }
    server = load_server(
        monkeypatch,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    calls = stub_async_client(
        monkeypatch,
        server,
        StubResponse(
            json_body={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }
        ),
    )

    assert TestClient(server.app).post("/v1/chat/completions", json=body).status_code == 200
    assert calls[0]["json"] == body


@pytest.mark.parametrize("body", [CHAT_BODY, {"messages": [{"role": "user", "content": "x"}]}])
def test_request_body_larger_than_configured_limit_returns_413(monkeypatch, body):
    server = load_server(monkeypatch, ADAPTER_MAX_REQUEST_BYTES="8")

    response = TestClient(server.app).post("/v1/chat/completions", json=body)

    assert response.status_code == 413
    assert response.json() == {
        "error": "Request body exceeds adapter size limit",
        "code": "request_too_large",
    }


def test_decoded_request_body_larger_than_configured_limit_returns_413(monkeypatch):
    server = load_server(monkeypatch, ADAPTER_MAX_REQUEST_BYTES="8")

    response = TestClient(server.app).post(
        "/v1/chat/completions",
        content=json.dumps(CHAT_BODY),
        headers={"Content-Type": "application/json", "Content-Length": "1"},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"


def test_streaming_request_stops_after_limit_without_buffering_entire_body(monkeypatch):
    server = load_server(monkeypatch, ADAPTER_MAX_REQUEST_BYTES="8")

    class StreamingRequest:
        headers = {"content-type": "application/json"}

        async def stream(self):
            yield b'{"model"'
            yield b':"too-large"}'
            raise AssertionError("adapter read beyond the size-limit boundary")

    response = asyncio.run(server.chat_completions(StreamingRequest()))

    assert response.status_code == 413
    assert json.loads(response.body)["code"] == "request_too_large"


@pytest.mark.parametrize("configured", ["0", "-1", "not-an-integer"])
def test_invalid_max_request_bytes_fails_closed(monkeypatch, configured):
    with pytest.raises(ValueError, match="ADAPTER_MAX_REQUEST_BYTES must be a positive integer"):
        load_server(monkeypatch, ADAPTER_MAX_REQUEST_BYTES=configured)


def test_upstream_error_omits_detail_by_default(monkeypatch):
    upstream_key = "stub-upstream-key"
    server = load_server(
        monkeypatch,
        UPSTREAM_API_KEY=upstream_key,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(
        monkeypatch,
        server,
        StubResponse(status_code=500, text=f"stack trace containing {upstream_key}"),
    )

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "Upstream returned HTTP 500"
    assert body["code"] == "upstream_non_200"
    assert body["benchmark_action"] == "stop_run_preserve_artifacts"
    assert body["upstream_status"] == 500
    assert len(body["raw_response_sha256"]) == 64
    assert "detail" not in body


def test_upstream_error_writes_private_diagnostic_without_changing_response(
    monkeypatch,
    tmp_path,
):
    diagnostic_path = tmp_path / "adapter-diagnostics.jsonl"
    upstream_key = "sk-" + "privateupstream123456789"
    server = load_server(
        monkeypatch,
        ADAPTER_DIAGNOSTICS_PATH=str(diagnostic_path),
        ADAPTER_DIAGNOSTICS_INCLUDE_DETAIL="true",
        UPSTREAM_API_KEY=upstream_key,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(
        monkeypatch,
        server,
        StubResponse(status_code=500, text=f"private upstream error {upstream_key}"),
    )

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 502
    assert response.json().keys() == {
        "error",
        "code",
        "benchmark_action",
        "upstream_status",
        "raw_response_sha256",
    }
    record = json.loads(diagnostic_path.read_text())
    assert record["claim_source"] == "adapter_claim"
    assert record["code"] == "upstream_non_200"
    assert record["context"]["upstream_status"] == 500
    assert len(record["request_id"]) == 32
    assert upstream_key not in diagnostic_path.read_text()


def test_upstream_error_includes_detail_when_debug_enabled(monkeypatch):
    server = load_server(
        monkeypatch,
        ADAPTER_DEBUG_UPSTREAM_ERRORS="true",
        UPSTREAM_API_KEY="stub-upstream-key",
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(monkeypatch, server, StubResponse(status_code=500, text="upstream exploded"))

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 502
    assert response.json()["detail"] == "upstream exploded"


def test_inbound_auth_requires_matching_bearer_when_configured(monkeypatch):
    server = load_server(monkeypatch, ADAPTER_INBOUND_API_KEY="inbound-secret")
    client = TestClient(server.app)

    missing_response = client.post("/v1/chat/completions", json=CHAT_BODY)
    authorized_response = client.post(
        "/v1/chat/completions",
        json=CHAT_BODY,
        headers={"Authorization": "Bearer inbound-secret"},
    )

    assert missing_response.status_code == 401
    assert missing_response.json() == {
        "error": "missing or invalid adapter API key",
        "code": "adapter_unauthorized",
    }
    assert authorized_response.status_code == 200


def test_startup_guard_rejects_non_loopback_without_inbound_auth(monkeypatch):
    server = load_server(
        monkeypatch,
        ADAPTER_HOST="0.0.0.0",
    )

    with pytest.raises(RuntimeError, match="refusing non-loopback exposure"):
        server.assert_safe_exposure()

    with pytest.raises(RuntimeError, match="refusing non-loopback exposure"):
        with TestClient(server.app):
            pass


def test_startup_guard_allows_authenticated_non_loopback_binding(monkeypatch):
    server = load_server(
        monkeypatch,
        ADAPTER_HOST="0.0.0.0",
        ADAPTER_INBOUND_API_KEY="local-inbound-key",
    )

    server.assert_safe_exposure()


def test_startup_guard_rejects_unauthenticated_loopback_proxy(monkeypatch):
    server = load_server(
        monkeypatch,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )

    with pytest.raises(RuntimeError, match="refusing upstream proxy mode"):
        server.assert_safe_exposure()

    with pytest.raises(RuntimeError, match="refusing upstream proxy mode"):
        with TestClient(server.app):
            pass


def test_proxy_rejects_cross_origin_text_plain_before_upstream(monkeypatch):
    server = load_server(
        monkeypatch,
        ADAPTER_INBOUND_API_KEY="inbound-secret",
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    calls = stub_async_client(
        monkeypatch,
        server,
        StubResponse(json_body={"choices": [{"message": {"content": "unused"}}]}),
    )

    response = TestClient(server.app).post(
        "/v1/chat/completions",
        content=json.dumps(CHAT_BODY),
        headers={
            "Authorization": "Bearer inbound-secret",
            "Content-Type": "text/plain",
            "Origin": "https://attacker.example",
        },
    )

    assert response.status_code == 403
    assert response.json()["code"] == "browser_origin_not_allowed"
    assert calls == []


def test_proxy_rejects_non_json_body_before_upstream(monkeypatch):
    server = load_server(
        monkeypatch,
        ADAPTER_INBOUND_API_KEY="inbound-secret",
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    calls = stub_async_client(
        monkeypatch,
        server,
        StubResponse(json_body={"choices": [{"message": {"content": "unused"}}]}),
    )

    response = TestClient(server.app).post(
        "/v1/chat/completions",
        content=json.dumps(CHAT_BODY),
        headers={
            "Authorization": "Bearer inbound-secret",
            "Content-Type": "text/plain",
        },
    )

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"
    assert calls == []


def test_uvicorn_import_string_cannot_override_loopback_without_auth(monkeypatch):
    server = load_server(monkeypatch)
    server.app.state.allow_unauthenticated_loopback = False

    with pytest.raises(RuntimeError, match="import-string launch requires"):
        with TestClient(server.app):
            pass


def test_upstream_key_never_appears_in_error_response_body(monkeypatch):
    upstream_key = "stub-upstream-key"
    server = load_server(
        monkeypatch,
        ADAPTER_INBOUND_API_KEY="inbound-secret",
        UPSTREAM_API_KEY=upstream_key,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(
        monkeypatch,
        server,
        StubResponse(status_code=500, text=f"upstream error mentions {upstream_key}"),
    )

    response = TestClient(server.app).post(
        "/v1/chat/completions",
        json=CHAT_BODY,
        headers={"Authorization": "Bearer inbound-secret"},
    )

    assert response.status_code == 502
    assert upstream_key not in response.text


def test_upstream_key_never_appears_in_default_logs(monkeypatch, capsys):
    upstream_key = "stub-upstream-key"
    server = load_server(
        monkeypatch,
        UPSTREAM_API_KEY=upstream_key,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(
        monkeypatch,
        server,
        StubResponse(status_code=500, text=f"private body containing {upstream_key}"),
    )

    TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert upstream_key not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (StubResponse(text="not json", json_error=True), "upstream_invalid_json"),
        (StubResponse(json_body={"choices": None}, text='{"choices": null}'), "invalid_upstream_choices"),
        (
            StubResponse(
                json_body={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "content_filter",
                        }
                    ]
                },
                text='{"choices": [{"message": {"content": ""}}]}',
            ),
            "empty_upstream_content",
        ),
    ],
)
def test_malformed_success_responses_become_structured_502(monkeypatch, response, expected_code):
    server = load_server(
        monkeypatch,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(monkeypatch, server, response)

    result = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert result.status_code == 502
    body = result.json()
    assert body["code"] == expected_code
    assert body["benchmark_action"] == "stop_run_preserve_artifacts"
    assert len(body["raw_response_sha256"]) == 64
    if expected_code == "empty_upstream_content":
        assert body["finish_reason"] == "content_filter"


def test_upstream_timeout_becomes_structured_504(monkeypatch):
    server = load_server(
        monkeypatch,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(
        monkeypatch,
        server,
        server.backend.httpx.ReadTimeout("timed out"),
    )

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 504
    assert response.json()["code"] == "upstream_timeout"


def test_empty_reference_response_is_a_structured_configuration_error(monkeypatch):
    server = load_server(monkeypatch, REFERENCE_RESPONSE="   ")

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 500
    assert response.json()["code"] == "empty_reference_response"


def test_request_transform_failure_is_structured_and_hides_detail(monkeypatch):
    server = load_server(
        monkeypatch,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )

    def fail_transform(_request_body):
        raise KeyError("private-request-field")

    monkeypatch.setattr(server.backend, "build_upstream_payload", fail_transform)

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 500
    assert response.json()["code"] == "adapter_request_transform_error"
    assert "private-request-field" not in response.text


def test_response_transform_failure_is_structured_and_fingerprinted(monkeypatch):
    server = load_server(
        monkeypatch,
        UPSTREAM_OPENAI_BASE_URL="https://upstream.example/v1",
    )
    stub_async_client(
        monkeypatch,
        server,
        StubResponse(json_body={"private": "body"}, text='{"private":"body"}'),
    )

    def fail_transform(_response_body):
        raise KeyError("private-response-field")

    monkeypatch.setattr(server.backend, "parse_upstream_response", fail_transform)

    response = TestClient(server.app).post("/v1/chat/completions", json=CHAT_BODY)

    assert response.status_code == 502
    body = response.json()
    assert body["code"] == "adapter_response_transform_error"
    assert len(body["raw_response_sha256"]) == 64
    assert "private-response-field" not in response.text
