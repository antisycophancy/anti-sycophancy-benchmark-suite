"""Backend customization boundary for the reference adapter.

The inbound server always speaks OpenAI-compatible chat completions. Users
whose model service has a proprietary JSON contract can adapt the three small
functions below instead of changing authentication, validation, or error
handling in ``server.py``:

* ``build_upstream_payload`` translates the benchmark request.
* ``build_upstream_headers`` adds upstream authentication.
* ``parse_upstream_response`` translates the provider response.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import httpx

from config import (
    REFERENCE_RESPONSE,
    REQUEST_TIMEOUT_SECONDS,
    UPSTREAM_MODEL_ID,
    upstream_api_key,
    upstream_chat_completions_url,
)
from model_routing import resolve_upstream_model
from openai_contract import OpenAIContractError, ParsedChatCompletion, parse_chat_completion_response


@dataclass(frozen=True)
class BackendCompletion:
    """Assistant-facing fields returned to the benchmark."""

    content: str
    finish_reason: str = "stop"
    native_finish_reason: str | None = None
    refusal: str | None = None
    usage: dict[str, Any] | None = None


class AdapterBackendError(RuntimeError):
    """A safe, structured failure at the private/upstream boundary."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        detail: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.context = context or {}


def build_upstream_payload(request_body: dict[str, Any]) -> dict[str, Any]:
    """Translate the benchmark request into the upstream request body.

    The default is a transparent OpenAI-compatible proxy. Replace this function
    when a private service expects a different JSON shape. Do not add benchmark
    labels, judge prompts, or score metadata to the model-visible payload.
    """
    return {
        **request_body,
        "model": resolve_upstream_model(request_body.get("model"), UPSTREAM_MODEL_ID),
    }


def build_upstream_headers() -> dict[str, str]:
    """Build upstream headers without exposing credentials to the benchmark."""
    headers = {"Content-Type": "application/json"}
    api_key = upstream_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def parse_upstream_response(value: Any) -> BackendCompletion:
    """Translate an upstream response into the assistant-facing result.

    The default accepts OpenAI chat-completion JSON. Replace this function when
    a private service returns a different response shape, extracting only the
    assistant text, finish reason, refusal signal, and public usage counters.
    """
    parsed: ParsedChatCompletion = parse_chat_completion_response(value)
    return BackendCompletion(
        content=parsed.content,
        finish_reason=parsed.finish_reason,
        native_finish_reason=parsed.native_finish_reason,
        refusal=parsed.refusal,
        usage=parsed.usage,
    )


def _body_digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _error_context(response: httpx.Response) -> dict[str, Any]:
    return {
        "upstream_status": response.status_code,
        "raw_response_sha256": _body_digest(response.content),
    }


async def complete_chat(request_body: dict[str, Any]) -> BackendCompletion:
    """Complete one request in deterministic reference or upstream proxy mode."""
    upstream_url = upstream_chat_completions_url()
    if not upstream_url:
        if not REFERENCE_RESPONSE.strip():
            raise AdapterBackendError(
                status_code=500,
                code="empty_reference_response",
                message="REFERENCE_RESPONSE must contain assistant text",
            )
        return BackendCompletion(content=REFERENCE_RESPONSE)

    try:
        payload = build_upstream_payload(request_body)
        headers = build_upstream_headers()
    except AdapterBackendError:
        raise
    except Exception as exc:
        raise AdapterBackendError(
            status_code=500,
            code="adapter_request_transform_error",
            message="Adapter could not build the upstream request",
            detail=repr(exc),
        ) from exc

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                upstream_url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
    except httpx.TimeoutException as exc:
        raise AdapterBackendError(
            status_code=504,
            code="upstream_timeout",
            message="Upstream request timed out",
            detail=str(exc),
        ) from exc
    except httpx.HTTPError as exc:
        raise AdapterBackendError(
            status_code=502,
            code="upstream_transport_error",
            message="Upstream request failed",
            detail=str(exc),
        ) from exc

    context = _error_context(response)
    if response.status_code >= 400:
        raise AdapterBackendError(
            status_code=502,
            code="upstream_non_200",
            message=f"Upstream returned HTTP {response.status_code}",
            detail=response.text[:500],
            context=context,
        )

    try:
        response_body = response.json()
    except ValueError as exc:
        raise AdapterBackendError(
            status_code=502,
            code="upstream_invalid_json",
            message="Upstream returned invalid JSON",
            detail=response.text[:500],
            context=context,
        ) from exc

    try:
        return parse_upstream_response(response_body)
    except OpenAIContractError as exc:
        raise AdapterBackendError(
            status_code=502,
            code=exc.code,
            message=str(exc),
            detail=response.text[:500],
            context={**context, **exc.context},
        ) from exc
    except AdapterBackendError:
        raise
    except Exception as exc:
        raise AdapterBackendError(
            status_code=502,
            code="adapter_response_transform_error",
            message="Adapter could not parse the upstream response",
            detail=repr(exc),
            context=context,
        ) from exc
