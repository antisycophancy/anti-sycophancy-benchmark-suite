"""Example translation hooks for a non-OpenAI JSON backend.

This file is documentation that is executable under test. Copy the three hook
functions into ``adapter/backend.py`` and add any standard-library imports they
need, then adjust only the example environment variable, request fields,
headers, and response fields for your service. ``backend.py`` already provides
the adapter result/error types and shared HTTP failure handling.
"""

from __future__ import annotations

import os
from typing import Any

from backend import AdapterBackendError, BackendCompletion
from openai_contract import OpenAIContractError


def build_upstream_payload(request_body: dict[str, Any]) -> dict[str, Any]:
    """Map OpenAI-style history to a fictional prompt/history contract."""
    messages = request_body["messages"]
    return {
        "prompt": messages[-1]["content"],
        "history": [
            {"speaker": message["role"], "text": message["content"]}
            for message in messages[:-1]
        ],
        "output_limit": request_body.get("max_tokens", 4096),
    }


def build_upstream_headers() -> dict[str, str]:
    """Use a backend-specific header without exposing the key downstream."""
    api_key = os.environ.get("EXAMPLE_BACKEND_API_KEY", "").strip()
    if not api_key:
        raise AdapterBackendError(
            status_code=500,
            code="missing_upstream_api_key",
            message="EXAMPLE_BACKEND_API_KEY is required",
        )
    return {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }


def parse_upstream_response(value: Any) -> BackendCompletion:
    """Map the fictional result into the provider-neutral completion type."""
    if not isinstance(value, dict):
        raise OpenAIContractError(
            "invalid_upstream_response",
            "Upstream response must be a JSON object",
        )
    answer = value.get("answer")
    refusal = value.get("refusal")
    if not isinstance(answer, str) or not answer.strip():
        if isinstance(refusal, str) and refusal.strip():
            answer = refusal
        else:
            raise OpenAIContractError(
                "empty_upstream_content",
                "Upstream response contained no answer or explicit refusal",
                context={"native_finish_reason": value.get("stop_reason")},
            )
    return BackendCompletion(
        content=answer,
        finish_reason="stop",
        native_finish_reason=(
            value["stop_reason"] if isinstance(value.get("stop_reason"), str) else None
        ),
        refusal=refusal if isinstance(refusal, str) and refusal.strip() else None,
        usage={
            "prompt_tokens": value.get("input_tokens", 0),
            "completion_tokens": value.get("output_tokens", 0),
        },
    )
