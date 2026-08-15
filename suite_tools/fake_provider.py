"""Deterministic local OpenAI-compatible server for offline load tests."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _FakeProviderHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512

    def __init__(
        self,
        address: tuple[str, int],
        *,
        latency_seconds: float,
        timeout_seconds: float,
        script: tuple[str, ...],
    ) -> None:
        super().__init__(address, _FakeProviderHandler)
        self.latency_seconds = latency_seconds
        self.timeout_seconds = timeout_seconds
        self.script = script
        self.stats_lock = threading.Lock()
        self.requests = 0
        self.active = 0
        self.max_active = 0
        self.active_seconds = 0.0
        self.measurement_started = time.monotonic()
        self.last_activity_change = self.measurement_started

    def _advance_activity_clock(self, now: float) -> None:
        self.active_seconds += self.active * (now - self.last_activity_change)
        self.last_activity_change = now

    def request_started(self) -> str:
        with self.stats_lock:
            self._advance_activity_clock(time.monotonic())
            request_index = self.requests
            self.requests += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            return self.script[request_index] if request_index < len(self.script) else "ok"

    def request_finished(self) -> None:
        with self.stats_lock:
            self._advance_activity_clock(time.monotonic())
            self.active -= 1

    def reset_stats(self) -> None:
        with self.stats_lock:
            if self.active:
                raise RuntimeError("cannot reset fake-provider stats with active requests")
            now = time.monotonic()
            self.requests = 0
            self.max_active = 0
            self.active_seconds = 0.0
            self.measurement_started = now
            self.last_activity_change = now

    def snapshot(self) -> dict[str, int | float]:
        with self.stats_lock:
            now = time.monotonic()
            self._advance_activity_clock(now)
            return {
                "requests": self.requests,
                "active": self.active,
                "max_active": self.max_active,
                "active_seconds": round(self.active_seconds, 6),
                "elapsed_seconds": round(now - self.measurement_started, 6),
            }


class _FakeProviderHandler(BaseHTTPRequestHandler):
    server: _FakeProviderHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_body(
        self,
        status: int,
        body: bytes,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        action = self.server.request_started()
        try:
            content_length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(content_length)
            try:
                request = json.loads(raw or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                request = {}
            time.sleep(self.server.latency_seconds)
            if action == "timeout":
                time.sleep(self.server.timeout_seconds)
            if action == "rate_limit":
                body = json.dumps({
                    "error": {
                        "message": "Fake provider rate limit",
                        "type": "rate_limit_error",
                    }
                }).encode("utf-8")
                self._send_body(429, body, headers={"Retry-After": "1"})
                return
            if action == "server_error":
                body = json.dumps({
                    "error": {"message": "temporary fake provider failure"}
                }).encode("utf-8")
                self._send_body(503, body)
                return
            if action == "malformed":
                self._send_body(200, b"not-json")
                return
            content = "" if action == "empty" else "OK"
            usage: dict[str, Any] = {
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "total_tokens": 9,
                "cost": 0.0,
            }
            if action == "detailed_usage":
                usage["prompt_tokens_details"] = {"cached_tokens": 3}
                usage["completion_tokens_details"] = {"reasoning_tokens": 2}
            body = json.dumps({
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": request.get("model") or "fake/model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }).encode("utf-8")
            self._send_body(200, body)
        finally:
            self.server.request_finished()


class FakeOpenAIProvider:
    """Context-managed local provider for deterministic integration tests."""

    def __init__(
        self,
        *,
        latency_seconds: float = 0.0,
        timeout_seconds: float = 1.0,
        script: tuple[str, ...] = (),
    ) -> None:
        if latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        self._server = _FakeProviderHTTPServer(
            ("127.0.0.1", 0),
            latency_seconds=latency_seconds,
            timeout_seconds=timeout_seconds,
            script=script,
        )
        self._thread: threading.Thread | None = None

    @property
    def chat_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1/chat/completions"

    def snapshot(self) -> dict[str, int | float]:
        return self._server.snapshot()

    def reset_stats(self) -> None:
        self._server.reset_stats()

    def __enter__(self) -> "FakeOpenAIProvider":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
