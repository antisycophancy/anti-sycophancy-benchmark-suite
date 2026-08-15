"""Small helpers for OpenAI-compatible benchmark adapter responses."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any


ALLOWED_MESSAGE_ROLES = {"system", "developer", "user", "assistant", "tool"}
USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
    "cost_source",
    "cached_tokens",
    "reasoning_tokens",
    "thinking_tokens",
)


class OpenAIContractError(ValueError):
    """A request or response that does not satisfy the adapter contract."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}


@dataclass(frozen=True)
class ParsedChatCompletion:
    """The provider-neutral fields the benchmark can safely consume."""

    content: str
    finish_reason: str
    native_finish_reason: str | None
    refusal: str | None
    usage: dict[str, Any]


def last_user_message(messages: list[dict[str, Any]]) -> str:
    """Return the last user message content from an OpenAI-style message list."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def validate_chat_completion_request(value: Any) -> dict[str, Any]:
    """Validate the small OpenAI request surface emitted by this benchmark."""
    if not isinstance(value, dict):
        raise OpenAIContractError("invalid_request", "Request body must be a JSON object")

    model = value.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise OpenAIContractError("invalid_model", "model must be a non-empty string")

    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenAIContractError("invalid_messages", "messages must be a non-empty array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise OpenAIContractError(
                "invalid_message",
                f"messages[{index}] must be an object",
            )
        role = message.get("role")
        if role not in ALLOWED_MESSAGE_ROLES:
            raise OpenAIContractError(
                "invalid_message_role",
                f"messages[{index}].role is not supported",
            )
        if not isinstance(message.get("content"), str):
            raise OpenAIContractError(
                "invalid_message_content",
                f"messages[{index}].content must be a string",
            )
    if not last_user_message(messages).strip():
        raise OpenAIContractError(
            "missing_user_message",
            "At least one non-empty user message is required",
        )
    return dict(value)


def normalized_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in USAGE_FIELDS
        if key in value and isinstance(value[key], (int, float, str))
    }


def parse_chat_completion_response(value: Any) -> ParsedChatCompletion:
    """Extract a valid assistant response without forwarding arbitrary metadata."""
    if not isinstance(value, dict):
        raise OpenAIContractError(
            "invalid_upstream_response",
            "Upstream response must be a JSON object",
        )
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OpenAIContractError(
            "invalid_upstream_choices",
            "Upstream response must contain choices[0]",
        )
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise OpenAIContractError(
            "invalid_upstream_message",
            "Upstream response must contain choices[0].message",
        )

    finish_reason = choice.get("finish_reason")
    finish_reason = finish_reason if isinstance(finish_reason, str) and finish_reason else "unknown"
    native_finish_reason = value.get("native_finish_reason") or choice.get("native_finish_reason")
    native_finish_reason = (
        native_finish_reason
        if isinstance(native_finish_reason, str) and native_finish_reason
        else None
    )
    refusal = message.get("refusal")
    refusal = refusal if isinstance(refusal, str) and refusal.strip() else None
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        if refusal is not None:
            content = refusal
        else:
            raise OpenAIContractError(
                "empty_upstream_content",
                "Upstream response contained no assistant text or explicit refusal",
                context={
                    "finish_reason": finish_reason,
                    "native_finish_reason": native_finish_reason,
                    "refusal": None,
                },
            )
    return ParsedChatCompletion(
        content=content,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
        refusal=refusal,
        usage=normalized_usage(value.get("usage")),
    )


def chat_completion_response(
    *,
    model: str,
    content: str,
    completion_id_prefix: str = "bench",
    usage: dict[str, Any] | None = None,
    finish_reason: str = "stop",
    native_finish_reason: str | None = None,
    refusal: str | None = None,
) -> dict[str, Any]:
    """Build the minimal chat completion shape benchmark clients expect."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if refusal is not None:
        message["refusal"] = refusal
    response = {
        "id": f"{completion_id_prefix}-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": normalized_usage(usage) or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    if native_finish_reason is not None:
        response["native_finish_reason"] = native_finish_reason
    return response


def model_list_response(model_ids: list[str], *, owned_by: str = "local") -> dict[str, Any]:
    """Build the `/v1/models` response used by OpenAI-compatible clients."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": owned_by,
            }
            for model_id in model_ids
        ],
    }
