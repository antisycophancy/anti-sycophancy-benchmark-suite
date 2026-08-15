"""Provider client adapters used by benchmark runners.

The benchmark stores model/provider identity in suite config, but most runners
expect an OpenAI-SDK-shaped client. This module keeps provider-specific request
translation in one place so OpenRouter, direct OpenAI-compatible endpoints, and
native Anthropic Messages runs can share retry, monitor, and provenance code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import httpx

from suite_tools.credential_policy import require_credential_destination
from suite_tools.paid_call_lease import provider_from_base_url, record_rate_limit_cooldown
from suite_tools.provider_signals import is_billing_payload

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_CONTENT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_OPENROUTER_STYLE_OPTIONS = {"reasoning", "reasoning_effort", "verbosity", "max_tokens"}
DEFAULT_OPENAI_COMPATIBLE_TIMEOUT = 120

ANTHROPIC_PRICE_PER_TOKEN = {
    # Native Anthropic responses include token counts but not dollar cost.
    # These estimates are for run dashboards only; invoices remain authoritative.
    "claude-opus-4-6": {"input": 5.0 / 1_000_000, "output": 25.0 / 1_000_000},
    "claude-opus-4-7": {"input": 5.0 / 1_000_000, "output": 25.0 / 1_000_000},
    "claude-opus-4-8": {"input": 5.0 / 1_000_000, "output": 25.0 / 1_000_000},
    # Standard Opus 5 pricing as of 2026-07.
    "claude-opus-5": {"input": 5.0 / 1_000_000, "output": 25.0 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    # Introductory Sonnet 5 pricing in effect for this 2026-06 run.
    "claude-sonnet-5": {"input": 2.0 / 1_000_000, "output": 10.0 / 1_000_000},
    "claude-fable-5": {"input": 10.0 / 1_000_000, "output": 50.0 / 1_000_000},
}

OPENAI_PRICE_PER_TOKEN = {
    # Standard API pricing as of 2026-06; dashboards remain estimates.
    "gpt-5.5": {"input": 5.0 / 1_000_000, "cached_input": 0.5 / 1_000_000, "output": 30.0 / 1_000_000},
    "gpt-5.4": {"input": 2.5 / 1_000_000, "cached_input": 0.25 / 1_000_000, "output": 15.0 / 1_000_000},
    "gpt-5.4-mini": {"input": 0.75 / 1_000_000, "cached_input": 0.075 / 1_000_000, "output": 4.5 / 1_000_000},
    # GPT-5.6 family list pricing at 2026-07-09 GA; cached input at the 90% discount.
    "gpt-5.6-sol": {"input": 5.0 / 1_000_000, "cached_input": 0.5 / 1_000_000, "output": 30.0 / 1_000_000, "source": "openai_standard_estimate_2026_07"},
    "gpt-5.6-terra": {"input": 2.5 / 1_000_000, "cached_input": 0.25 / 1_000_000, "output": 15.0 / 1_000_000, "source": "openai_standard_estimate_2026_07"},
    "gpt-5.6-luna": {"input": 1.0 / 1_000_000, "cached_input": 0.1 / 1_000_000, "output": 6.0 / 1_000_000, "source": "openai_standard_estimate_2026_07"},
}

GEMINI_PRICE_PER_TOKEN = {
    # Gemini standard paid-tier text pricing; output includes thinking tokens.
    "gemini-3.5-flash": {"input": 1.5 / 1_000_000, "cached_input": 0.15 / 1_000_000, "output": 9.0 / 1_000_000},
    "gemini-3.1-flash-lite": {"input": 0.25 / 1_000_000, "cached_input": 0.025 / 1_000_000, "output": 1.5 / 1_000_000},
    "gemini-3-flash-preview": {"input": 0.5 / 1_000_000, "cached_input": 0.05 / 1_000_000, "output": 3.0 / 1_000_000},
}

GEMINI_TIERED_PRICE_PER_TOKEN = {
    "gemini-3.1-pro-preview": (
        (200_000, {"input": 2.0 / 1_000_000, "cached_input": 0.2 / 1_000_000, "output": 12.0 / 1_000_000}),
        (None, {"input": 4.0 / 1_000_000, "cached_input": 0.4 / 1_000_000, "output": 18.0 / 1_000_000}),
    ),
    "gemini-3.1-pro-preview-customtools": (
        (200_000, {"input": 2.0 / 1_000_000, "cached_input": 0.2 / 1_000_000, "output": 12.0 / 1_000_000}),
        (None, {"input": 4.0 / 1_000_000, "cached_input": 0.4 / 1_000_000, "output": 18.0 / 1_000_000}),
    ),
}


def extract_raw_response(exc: BaseException) -> dict[str, Any] | None:
    """Extract a structured provider error body from an exception, defensively.

    Checks in priority order:
    1. exc.raw_response if already a dict (ProviderApiError and subclasses).
    2. exc.body if a dict (OpenAI SDK exception shape).
    3. exc.response.json() as a fallback (SDK response wrapper).

    Returns None when no structured body is available or any extraction fails.
    Never raises.
    """
    raw = getattr(exc, "raw_response", None)
    if isinstance(raw, dict):
        return raw

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


class ProviderApiError(RuntimeError):
    """Provider error with a status code compatible with run failure classifiers."""

    def __init__(
        self,
        status_code: int,
        text: str,
        *,
        headers: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.usage = usage or {}
        self.raw_response = raw_response
        super().__init__(f"HTTP {status_code}: {text[:500]}")


class ProviderRefusalError(ProviderApiError):
    """Provider-level refusal returned as a successful API response."""

    def __init__(
        self,
        text: str,
        *,
        raw_response: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ):
        normalized_raw = raw_response if isinstance(raw_response, dict) else {}
        super().__init__(200, text, headers=headers, usage=usage, raw_response=normalized_raw)
        self.stop_reason = self.raw_response.get("stop_reason")
        stop_details = self.raw_response.get("stop_details")
        self.stop_details = stop_details if isinstance(stop_details, dict) else {}


class ProviderOutputBudgetExhaustedError(ProviderApiError):
    """Bounded-retry-then-terminal outcome: the output-token budget was spent.

    Raised when the Responses API returns ``status=incomplete`` with
    ``incomplete_reason=max_output_tokens`` and only reasoning output (no usable
    ``output_text``) — the model burned the entire output budget on reasoning.

    Empirically this is a STOCHASTIC runaway-reasoning loop, not a systematic item
    property: replaying the same "exhausted" item usually completes normally in a
    fraction of the tokens. Runners therefore retry it a bounded number of times
    (``BENCHMARK_OUTPUT_BUDGET_RETRIES``) before recording it as an excluded,
    non-halting item.

    It is deliberately NOT a subclass of :class:`ProviderRefusalError`: the runners'
    ``isinstance(e, ProviderRefusalError): raise`` short-circuit makes refusals (e.g.
    content-policy blocks) IMMEDIATELY terminal, and budget-exhaustion must instead be
    retryable. It subclasses the retryable :class:`ProviderApiError` family, stays
    HTTP-200-shaped, and carries billable usage so the spent reasoning tokens of every
    attempt are accounted for rather than lost.
    """

    def __init__(
        self,
        text: str,
        *,
        raw_response: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ):
        normalized_raw = raw_response if isinstance(raw_response, dict) else {}
        super().__init__(200, text, headers=headers, usage=usage, raw_response=normalized_raw)
        self.stop_reason = self.raw_response.get("stop_reason")


class ProviderMalformedResponseError(ProviderApiError):
    """A successful HTTP response that is not a usable chat completion."""

    def __init__(
        self,
        response_shape: str,
        text: str,
        *,
        raw_response: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ):
        self.response_shape = response_shape
        normalized_raw = raw_response if isinstance(raw_response, dict) else {}
        super().__init__(200, text, headers=headers, usage=usage, raw_response=normalized_raw)


@dataclass(frozen=True)
class ChatCompletionInspection:
    """Validated fields plus the provider body needed for failure evidence."""

    content: str | None
    signal_payload: dict[str, Any]
    raw_response: dict[str, Any]
    response_shape: str | None = None


_MISSING = object()


def _diagnostic_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 6:
        return {"_type": type(value).__name__, "_truncated": True}
    if isinstance(value, dict):
        return {
            str(key): _diagnostic_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item, depth=depth + 1) for item in value]
    return {"_type": type(value).__name__}


def _model_dump_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return _diagnostic_value(value)
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return None
    try:
        dumped = model_dump(mode="json")
    except TypeError:
        try:
            dumped = model_dump()
        except Exception:
            return None
    except Exception:
        return None
    return _diagnostic_value(dumped) if isinstance(dumped, dict) else None


def _message_mapping(message: Any) -> dict[str, Any]:
    dumped = _model_dump_mapping(message)
    if dumped is not None:
        return dumped
    result: dict[str, Any] = {}
    for field in ("role", "content", "refusal"):
        value = getattr(message, field, _MISSING)
        if value is not _MISSING:
            result[field] = _diagnostic_value(value)
    return result


def _choice_mapping(choice: Any) -> dict[str, Any]:
    dumped = _model_dump_mapping(choice)
    if dumped is not None:
        return dumped
    result: dict[str, Any] = {}
    for field in ("index", "finish_reason", "native_finish_reason"):
        value = getattr(choice, field, _MISSING)
        if value is not _MISSING:
            result[field] = _diagnostic_value(value)
    message = getattr(choice, "message", _MISSING)
    if message is not _MISSING:
        result["message"] = (
            _message_mapping(message) if message is not None else None
        )
    return result


def _chat_completion_mapping(response: Any) -> dict[str, Any]:
    dumped = _model_dump_mapping(response)
    if dumped is not None:
        return dumped
    if response is None:
        return {}

    result: dict[str, Any] = {}
    for field in ("id", "object", "created", "model", "provider", "native_finish_reason"):
        value = getattr(response, field, _MISSING)
        if value is not _MISSING:
            result[field] = _diagnostic_value(value)

    choices = getattr(response, "choices", _MISSING)
    if choices is not _MISSING:
        if isinstance(choices, (list, tuple)):
            result["choices"] = [
                _choice_mapping(choice) if choice is not None else None
                for choice in choices
            ]
        else:
            result["choices"] = _diagnostic_value(choices)

    usage = getattr(response, "usage", _MISSING)
    if usage is not _MISSING:
        dumped_usage = _model_dump_mapping(usage)
        if dumped_usage is not None:
            result["usage"] = dumped_usage
        elif usage is None:
            result["usage"] = None
        else:
            usage_fields: dict[str, Any] = {}
            for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
                value = getattr(usage, field, _MISSING)
                if value is not _MISSING:
                    usage_fields[field] = _diagnostic_value(value)
            result["usage"] = usage_fields
    return result


def inspect_chat_completion_response(response: Any) -> ChatCompletionInspection:
    """Inspect an OpenAI-compatible response without unsafe list indexing.

    The function does not decide whether a finish/refusal signal is a benchmark
    outcome. Runners consult the existing content-block policy first, then treat
    ``response_shape`` as a malformed-response failure only when no such signal
    applies.
    """

    raw = _chat_completion_mapping(response)
    if response is None:
        return ChatCompletionInspection(None, {}, raw, "response_none")

    choices = (
        response.get("choices", _MISSING)
        if isinstance(response, dict)
        else getattr(response, "choices", _MISSING)
    )
    top_native = (
        response.get("native_finish_reason")
        if isinstance(response, dict)
        else getattr(response, "native_finish_reason", None)
    )
    if choices is _MISSING:
        signal = {"native_finish_reason": top_native}
        return ChatCompletionInspection(None, signal, raw, "choices_missing")
    if choices is None:
        signal = {"native_finish_reason": top_native}
        return ChatCompletionInspection(None, signal, raw, "choices_null")
    if not isinstance(choices, (list, tuple)):
        signal = {"native_finish_reason": top_native}
        return ChatCompletionInspection(None, signal, raw, "choices_wrong_type")
    if not choices:
        signal = {"native_finish_reason": top_native}
        return ChatCompletionInspection(None, signal, raw, "choices_empty")

    choice = choices[0]
    if choice is None:
        signal = {"native_finish_reason": top_native}
        return ChatCompletionInspection(None, signal, raw, "choice_null")
    if not isinstance(choice, dict) and not hasattr(choice, "message"):
        signal = {"native_finish_reason": top_native}
        return ChatCompletionInspection(None, signal, raw, "choice_wrong_type")

    finish_reason = (
        choice.get("finish_reason")
        if isinstance(choice, dict)
        else getattr(choice, "finish_reason", None)
    )
    choice_native = (
        choice.get("native_finish_reason")
        if isinstance(choice, dict)
        else getattr(choice, "native_finish_reason", None)
    )
    message = (
        choice.get("message", _MISSING)
        if isinstance(choice, dict)
        else getattr(choice, "message", _MISSING)
    )
    signal = {
        "finish_reason": finish_reason,
        "native_finish_reason": top_native or choice_native,
        "refusal": None,
    }
    if message is _MISSING:
        return ChatCompletionInspection(None, signal, raw, "message_missing")
    if message is None:
        return ChatCompletionInspection(None, signal, raw, "message_null")
    if not isinstance(message, dict) and not hasattr(message, "content"):
        return ChatCompletionInspection(None, signal, raw, "message_wrong_type")

    content = (
        message.get("content", _MISSING)
        if isinstance(message, dict)
        else getattr(message, "content", _MISSING)
    )
    refusal = (
        message.get("refusal")
        if isinstance(message, dict)
        else getattr(message, "refusal", None)
    )
    signal["refusal"] = refusal
    if content is _MISSING:
        return ChatCompletionInspection(None, signal, raw, "content_missing")
    if content is None or content == "":
        return ChatCompletionInspection(None, signal, raw, "empty_content")
    if not isinstance(content, str):
        return ChatCompletionInspection(None, signal, raw, "content_wrong_type")
    return ChatCompletionInspection(content, signal, raw)


def _provider_response_json(response: Any, provider: str) -> dict[str, Any]:
    raw_content = getattr(response, "content", None)
    if not isinstance(raw_content, bytes) or not raw_content:
        raw_content = str(getattr(response, "text", raw_content or "")).encode(
            "utf-8",
            errors="replace",
        )
    raw_digest = hashlib.sha256(raw_content).hexdigest()
    try:
        data = response.json()
    except (TypeError, ValueError) as exc:
        raise ProviderApiError(
            502,
            f"{provider} returned invalid JSON in an HTTP 200 response",
            headers=dict(getattr(response, "headers", {}) or {}),
            raw_response={
                "response_shape": "invalid_json",
                "raw_body_sha256": raw_digest,
            },
        ) from exc
    if not isinstance(data, dict):
        raise ProviderApiError(
            502,
            f"{provider} returned a non-object JSON payload in an HTTP 200 response",
            headers=dict(getattr(response, "headers", {}) or {}),
            raw_response={
                "response_shape": "json_non_object",
                "raw_body_sha256": raw_digest,
            },
        )
    return data


def is_anthropic_messages_url(url: str | None) -> bool:
    """Return true when a configured URL targets Anthropic's Messages API."""
    parsed = urlsplit(str(url or ""))
    return (
        (parsed.hostname or "").lower().rstrip(".") == "api.anthropic.com"
        and parsed.path.rstrip("/") == "/v1/messages"
    )


def is_openai_native_url(url: str | None) -> bool:
    """Return true when a configured URL targets OpenAI's native API."""
    return (urlsplit(str(url or "")).hostname or "").lower().rstrip(".") == "api.openai.com"


def is_openai_responses_url(url: str | None) -> bool:
    """Return true when a configured URL targets OpenAI's Responses API."""
    parsed = urlsplit(str(url or ""))
    return (
        (parsed.hostname or "").lower().rstrip(".") == "api.openai.com"
        and parsed.path.rstrip("/") == "/v1/responses"
    )


def is_gemini_generate_content_url(url: str | None) -> bool:
    """Return true for native Gemini generateContent endpoints or base URLs."""
    parsed = urlsplit(str(url or ""))
    return (
        (parsed.hostname or "").lower().rstrip(".")
        == "generativelanguage.googleapis.com"
        and "/openai" not in parsed.path.lower()
    )


def model_uses_max_completion_tokens(model: str | None) -> bool:
    """Return true for direct OpenAI models that reject ``max_tokens``."""
    normalized = str(model or "").lower()
    return normalized.startswith("gpt-5")


def _anthropic_thinking_enabled(payload: dict[str, Any]) -> bool:
    thinking = payload.get("thinking")
    return isinstance(thinking, dict) and thinking.get("type") in {"adaptive", "enabled"}


def _anthropic_uses_adaptive_effort(payload: dict[str, Any]) -> bool:
    output_config = payload.get("output_config")
    return isinstance(output_config, dict) and bool(output_config.get("effort"))


def normalize_chat_payload_for_provider(payload: dict[str, Any], *, base_url: str | None) -> dict[str, Any]:
    """Normalize OpenAI-compatible chat payload fields for provider quirks.

    OpenRouter accepts ``max_tokens`` for GPT-style models, while direct OpenAI
    GPT-5-family chat completions require ``max_completion_tokens``. Keep the
    rule scoped to direct OpenAI URLs so other OpenAI-compatible providers are
    not changed. Some direct high-thinking APIs also reject explicit
    ``temperature=0``; when the provider only accepts its default temperature,
    omit the field instead of turning judge calls into avoidable 400s.
    """
    if is_openai_native_url(base_url) and model_uses_max_completion_tokens(payload.get("model")):
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            # Model and judge configs are reused across calls. Normalize a copy so
            # consuming translated fields cannot alter the configured condition.
            extra_body = dict(extra_body)
            payload["extra_body"] = extra_body
            if "max_completion_tokens" in extra_body:
                payload["max_completion_tokens"] = extra_body.pop("max_completion_tokens")
            if "max_tokens" in extra_body and "max_completion_tokens" not in payload:
                payload["max_tokens"] = extra_body.pop("max_tokens")
            else:
                extra_body.pop("max_tokens", None)
            if not extra_body:
                payload.pop("extra_body", None)
        max_tokens = payload.pop("max_tokens", None)
        if max_tokens is not None and "max_completion_tokens" not in payload:
            payload["max_completion_tokens"] = max_tokens
        if "temperature" in payload and payload["temperature"] != 1:
            payload.pop("temperature", None)
    if is_anthropic_messages_url(base_url) and (
        _anthropic_thinking_enabled(payload) or _anthropic_uses_adaptive_effort(payload)
    ):
        if "temperature" in payload and payload["temperature"] != 1:
            payload.pop("temperature", None)
    return payload


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, str]]]:
    system_parts: list[str] = []
    converted: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(str(content))
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        converted.append({"role": str(role), "content": str(content)})
    system = "\n\n".join(part for part in system_parts if part).strip() or None
    return system, converted


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _anthropic_model_key(model: str) -> str:
    return str(model).split("/")[-1].replace(".", "-")


def _model_key(model: str | None) -> str:
    return str(model or "").split("/")[-1].lower()


def _token_count(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _nested_token_count(usage: dict[str, Any], parent_key: str, child_key: str) -> int:
    parent = usage.get(parent_key)
    if not isinstance(parent, dict):
        return 0
    value = parent.get(child_key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _cached_prompt_tokens(usage: dict[str, Any]) -> int:
    return max(
        _token_count(usage, "cached_prompt_tokens"),
        _nested_token_count(usage, "prompt_tokens_details", "cached_tokens"),
        _nested_token_count(usage, "input_tokens_details", "cached_tokens"),
    )


def _billable_completion_tokens(usage: dict[str, Any]) -> int:
    explicit = _token_count(usage, "billable_completion_tokens")
    if explicit:
        return explicit
    return _token_count(usage, "completion_tokens") + _token_count(usage, "thoughts_tokens")


def _prices_for_model(model: str, prompt_tokens: int) -> tuple[str, dict[str, float]] | None:
    key = _model_key(model)
    prices = OPENAI_PRICE_PER_TOKEN.get(key)
    if prices:
        return prices.get("source", "openai_standard_estimate_2026_06"), prices
    prices = GEMINI_PRICE_PER_TOKEN.get(key)
    if prices:
        return "gemini_standard_estimate_2026_06", prices
    tiers = GEMINI_TIERED_PRICE_PER_TOKEN.get(key)
    if tiers:
        for threshold, tier_prices in tiers:
            if threshold is None or prompt_tokens <= threshold:
                suffix = "prompt_le_200k" if threshold is not None else "prompt_gt_200k"
                return f"gemini_standard_estimate_2026_06_{suffix}", tier_prices
    return None


def estimate_usage_cost(model: str, usage: dict[str, Any]) -> dict[str, Any]:
    """Attach estimated cost for direct providers when the API omits it."""
    if not usage:
        return usage
    existing_cost = usage.get("cost")
    if not isinstance(existing_cost, bool):
        try:
            parsed_cost = float(existing_cost)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(parsed_cost) and parsed_cost >= 0:
                return usage

    prompt_tokens = _token_count(usage, "prompt_tokens")
    completion_tokens = _billable_completion_tokens(usage)
    price_entry = _prices_for_model(model, prompt_tokens)
    if price_entry is None:
        return usage

    source, prices = price_entry
    cached_tokens = min(_cached_prompt_tokens(usage), prompt_tokens)
    uncached_tokens = max(prompt_tokens - cached_tokens, 0)
    estimated = (
        uncached_tokens * prices["input"]
        + cached_tokens * prices.get("cached_input", prices["input"])
        + completion_tokens * prices["output"]
    )
    enriched = dict(usage)
    enriched["estimated_cost"] = estimated
    enriched["cost_source"] = source
    return enriched


def _anthropic_usage(data: dict[str, Any], model: str) -> dict[str, Any]:
    raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    prompt_tokens = raw.get("input_tokens", 0) or 0
    completion_tokens = raw.get("output_tokens", 0) or 0
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    prices = ANTHROPIC_PRICE_PER_TOKEN.get(_anthropic_model_key(model))
    if prices:
        usage["cost"] = (
            prompt_tokens * prices["input"]
            + completion_tokens * prices["output"]
        )
        usage["cost_source"] = "anthropic_estimate"
    return usage


def _anthropic_empty_text_error(data: dict[str, Any]) -> ProviderApiError:
    content_types: list[str] = []
    for block in data.get("content") or []:
        if isinstance(block, dict):
            block_type = block.get("type")
            if isinstance(block_type, str) and block_type:
                content_types.append(block_type)
    types = ",".join(sorted(set(content_types))) or "none"
    stop_reason = data.get("stop_reason") or "unknown"
    err = ProviderApiError(
        502,
        "Anthropic native response contained no text blocks; "
        f"stop_reason={stop_reason}; content_types={types}",
        raw_response={
            "finish_reason": stop_reason,
            "native_finish_reason": stop_reason,
            "refusal": None,
        },
    )
    return err


def _gemini_parts_from_content(content: Any) -> list[dict[str, str]]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        parts: list[dict[str, str]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"text": item})
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append({"text": item["text"]})
            else:
                parts.append({"text": str(item)})
        return parts or [{"text": ""}]
    return [{"text": str(content)}]


def _to_gemini_contents(messages: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        parts = _gemini_parts_from_content(message.get("content", ""))
        if role == "system":
            system_parts.extend(parts)
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        )
    system = {"parts": system_parts} if system_parts else None
    return system, contents


def _merge_generation_config(payload: dict[str, Any], updates: dict[str, Any]) -> None:
    current = payload.setdefault("generationConfig", {})
    if not isinstance(current, dict):
        raise ValueError("generationConfig must be a mapping")
    current.update(updates)


def _apply_gemini_request_options(payload: dict[str, Any], request_options: dict[str, Any] | None) -> None:
    if not request_options:
        return
    if not isinstance(request_options, dict):
        raise ValueError("extra_body/request_options must be a mapping")
    unsupported = sorted(GEMINI_OPENROUTER_STYLE_OPTIONS.intersection(request_options))
    if unsupported:
        raise ValueError(
            "native Gemini request_options must use Gemini generateContent fields; "
            f"unsupported OpenAI/OpenRouter-style fields: {', '.join(unsupported)}"
        )
    for key, value in request_options.items():
        if key == "generationConfig":
            if not isinstance(value, dict):
                raise ValueError("generationConfig must be a mapping")
            _merge_generation_config(payload, value)
        elif key == "thinkingConfig":
            if not isinstance(value, dict):
                raise ValueError("thinkingConfig must be a mapping")
            _merge_generation_config(payload, {"thinkingConfig": value})
        else:
            payload[key] = value


def gemini_generate_content_url(model: str, base_url: str | None = None) -> str:
    root = (base_url or GEMINI_GENERATE_CONTENT_BASE_URL).rstrip("/")
    if root.endswith(":generateContent"):
        return root
    return f"{root}/models/{quote(str(model), safe='')}:generateContent"


def build_gemini_generate_content_payload(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    request_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system, contents = _to_gemini_contents(messages)
    payload: dict[str, Any] = {"contents": contents}
    if system:
        payload["systemInstruction"] = system
    generation_config: dict[str, Any] = {}
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if temperature is not None:
        generation_config["temperature"] = temperature
    if generation_config:
        payload["generationConfig"] = generation_config
    _apply_gemini_request_options(payload, request_options)
    return payload


def extract_gemini_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
    for candidate in candidates[:1]:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        for part in content.get("parts") or []:
            if isinstance(part, dict) and part.get("thought"):
                continue
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def gemini_usage(data: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    raw = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
    prompt_tokens = raw.get("promptTokenCount", 0) or 0
    completion_tokens = raw.get("candidatesTokenCount", 0) or 0
    total_tokens = raw.get("totalTokenCount")
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
    }
    thoughts_tokens = raw.get("thoughtsTokenCount")
    if thoughts_tokens is not None:
        usage["thoughts_tokens"] = thoughts_tokens
        usage["billable_completion_tokens"] = completion_tokens + (thoughts_tokens or 0)
    return estimate_usage_cost(model or data.get("modelVersion") or "", usage)


def gemini_empty_text_error(data: dict[str, Any]) -> ProviderApiError:
    candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
    first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    finish_reason = first.get("finishReason") or "unknown"
    prompt_feedback = data.get("promptFeedback") if isinstance(data.get("promptFeedback"), dict) else {}
    block_reason = prompt_feedback.get("blockReason") or "none"
    err = ProviderApiError(
        502,
        "Gemini native response contained no non-thought text parts; "
        f"finish_reason={finish_reason}; prompt_block_reason={block_reason}",
    )
    err.raw_response = {
        "candidates": candidates,
        "promptFeedback": prompt_feedback,
    }
    return err


@dataclass
class GeminiGenerateContentClient:
    """OpenAI-SDK-shaped wrapper around Gemini's native generateContent API."""

    api_key: str
    base_url: str = GEMINI_GENERATE_CONTENT_BASE_URL
    timeout: int = 120

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=_GeminiGenerateContentCompletions(self))


class _GeminiGenerateContentCompletions:
    def __init__(self, parent: GeminiGenerateContentClient) -> None:
        self._parent = parent
        self.base_url = parent.base_url

    def create(self, **kwargs: Any) -> Any:
        model = str(kwargs["model"])
        should_record_cooldown = kwargs.get("record_rate_limit_cooldown", True)
        payload = build_gemini_generate_content_payload(
            kwargs.get("messages") or [],
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            request_options=kwargs.get("extra_body") or {},
        )
        timeout = kwargs.get("timeout") or self._parent.timeout
        url = gemini_generate_content_url(model, self._parent.base_url)
        response = httpx.post(
            url,
            headers={
                "x-goog-api-key": self._parent.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            try:
                parsed_body: dict[str, Any] | None = response.json()
                if not isinstance(parsed_body, dict):
                    parsed_body = None
            except Exception:
                parsed_body = None
            error = ProviderApiError(
                response.status_code,
                response.text[:500],
                headers=dict(response.headers),
                raw_response=parsed_body,
            )
            if (
                response.status_code == 429
                and should_record_cooldown
                and not is_billing_payload(error)
            ):
                record_rate_limit_cooldown(
                    provider=provider_from_base_url(self._parent.base_url),
                    model=model,
                    headers=dict(response.headers),
                    error=error,
                )
            raise error

        data = _provider_response_json(response, "Gemini native provider")
        content = extract_gemini_text(data)
        if not content:
            error = gemini_empty_text_error(data)
            error.usage = gemini_usage(data, model=model)
            raise error
        return SimpleNamespace(
            model=data.get("modelVersion", model),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=gemini_usage(data, model=model),
            raw_response=data,
        )


@dataclass
class AnthropicMessagesClient:
    """OpenAI-SDK-shaped wrapper around Anthropic's native Messages API."""

    api_key: str
    base_url: str = ANTHROPIC_MESSAGES_URL
    timeout: int = 120

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=_AnthropicMessagesCompletions(self))


class _AnthropicMessagesCompletions:
    def __init__(self, parent: AnthropicMessagesClient) -> None:
        self._parent = parent
        self.base_url = parent.base_url

    def create(self, **kwargs: Any) -> Any:
        model = str(kwargs["model"])
        should_record_cooldown = kwargs.get("record_rate_limit_cooldown", True)
        messages = kwargs.get("messages") or []
        extra_body = kwargs.get("extra_body") or {}
        if extra_body and not isinstance(extra_body, dict):
            raise ValueError("extra_body/request_options must be a mapping")

        system, anthropic_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system:
            payload["system"] = system
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        payload.update(extra_body)
        normalize_chat_payload_for_provider(payload, base_url=self._parent.base_url)

        timeout = kwargs.get("timeout") or self._parent.timeout
        response = httpx.post(
            self._parent.base_url,
            headers={
                "x-api-key": self._parent.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            try:
                parsed_body: dict[str, Any] | None = response.json()
                if not isinstance(parsed_body, dict):
                    parsed_body = None
            except Exception:
                parsed_body = None
            error = ProviderApiError(
                response.status_code,
                response.text[:500],
                headers=dict(response.headers),
                raw_response=parsed_body,
            )
            if (
                response.status_code == 429
                and should_record_cooldown
                and not is_billing_payload(error)
            ):
                record_rate_limit_cooldown(
                    provider=provider_from_base_url(self._parent.base_url),
                    model=model,
                    headers=dict(response.headers),
                    error=error,
                )
            raise error

        data = _provider_response_json(response, "Anthropic native provider")
        usage = _anthropic_usage(data, model)
        if data.get("stop_reason") == "refusal":
            refusal_usage = dict(usage)
            raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            if not raw_usage.get("output_tokens"):
                refusal_usage["cost"] = 0.0
                refusal_usage["cost_source"] = "anthropic_refusal_no_charge"
            stop_details = data.get("stop_details")
            classifier = (
                stop_details.get("category") or stop_details.get("classifier")
                if isinstance(stop_details, dict)
                else None
            )
            detail = f"; classifier={classifier}" if classifier else ""
            raise ProviderRefusalError(
                f"Anthropic native provider refusal; stop_reason=refusal{detail}",
                raw_response=data,
                usage=refusal_usage,
            )
        content = _extract_anthropic_text(data)
        if not content:
            error = _anthropic_empty_text_error(data)
            error.usage = usage
            raise error
        return SimpleNamespace(
            model=data.get("model", model),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
            raw_response=data,
        )


def _to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Map OpenAI chat messages to Responses ``instructions`` + ``input``.

    System messages are concatenated into ``instructions`` (the Responses API
    contract for system guidance). Every other turn becomes an ``input`` message
    item that carries its own role, so multi-turn conversation history round
    trips: user text uses ``input_text`` parts, assistant text uses
    ``output_text`` parts.
    """
    instruction_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            instruction_parts.append(str(content))
            continue
        if role == "assistant":
            item_role, part_type = "assistant", "output_text"
        else:
            item_role, part_type = "user", "input_text"
        input_items.append(
            {"role": item_role, "content": [{"type": part_type, "text": str(content)}]}
        )
    instructions = "\n\n".join(part for part in instruction_parts if part).strip() or None
    return instructions, input_items


def _responses_output_item_types(data: dict[str, Any]) -> list[str]:
    types: list[str] = []
    for item in data.get("output") or []:
        if isinstance(item, dict):
            item_type = item.get("type")
            if isinstance(item_type, str) and item_type:
                types.append(item_type)
    return types


def _extract_responses_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts).strip()


def refusal_from_http_error(*, status_code: int, body: dict[str, Any] | None) -> ProviderRefusalError:
    """Build a refusal error that PRESERVES the provider's structured error
    (code, message) so downstream evidence classification can report the
    policy category instead of a generic 'refusal'."""
    payload = body if isinstance(body, dict) else {}
    error_obj = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    code = error_obj.get("code")
    base_msg = str(error_obj.get("message") or payload.get("message") or f"HTTP {status_code} content policy refusal")
    code_tag = f" [{code}]" if code else ""
    message = f"content-policy block{code_tag}: {base_msg}"
    return ProviderRefusalError(message, raw_response=payload)


def _is_openai_content_policy_400(response: Any) -> bool:
    """Return True when an HTTP 400 from the Responses API is a content-policy block.

    Only moderation/content-policy codes are matched so that genuine malformed-request
    400s (unknown fields, bad parameter types, missing required params, etc.) are NOT
    silently turned into refusals and remain hard errors.
    """
    try:
        data = response.json()
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    err = data.get("error")
    if not isinstance(err, dict):
        return False
    code = str(err.get("code") or "").lower()
    typ = str(err.get("type") or "").lower()
    msg = str(err.get("message") or "").lower()
    # Explicit content-policy / moderation codes returned by OpenAI
    if code in {"cyber_policy", "content_policy", "content_filter"}:
        return True
    if typ in {"cyber_policy", "content_policy", "content_filter"}:
        return True
    # Moderation signal phrases in the message (any code/type)
    if any(s in msg for s in ("content was flagged", "flagged by our content", "content_policy", "cyber_policy")):
        return True
    return False


def _responses_refusal_text(data: dict[str, Any]) -> str | None:
    """Return refusal text when the Responses reply is a provider refusal.

    A refusal is either an explicit ``refusal`` content part or an
    ``incomplete`` status whose reason is ``content_filter``. A
    ``max_output_tokens`` truncation is NOT a refusal.
    """
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "refusal":
                text = part.get("refusal")
                return str(text) if text else "OpenAI Responses provider refusal"
    incomplete = data.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
    if data.get("status") == "incomplete" and reason == "content_filter":
        return "OpenAI Responses provider refusal; incomplete reason=content_filter"
    return None


def _responses_usage(data: dict[str, Any], model: str) -> dict[str, Any]:
    raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    input_tokens = raw.get("input_tokens", 0) or 0
    # output_tokens already includes reasoning tokens; they are billed as output
    # and must not be double-counted.
    output_tokens = raw.get("output_tokens", 0) or 0
    total_tokens = raw.get("total_tokens")
    usage: dict[str, Any] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
    }
    input_details = raw.get("input_tokens_details")
    if isinstance(input_details, dict):
        # Passed through so estimate_usage_cost bills cached_tokens at the cached
        # rate. cache_write_tokens is surfaced for provenance only: OpenAI does
        # not bill cache writes separately (they are covered by uncached input),
        # so no surcharge is applied.
        usage["input_tokens_details"] = dict(input_details)
    output_details = raw.get("output_tokens_details")
    if isinstance(output_details, dict):
        usage["output_tokens_details"] = dict(output_details)
        reasoning_tokens = output_details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            usage["reasoning_tokens"] = reasoning_tokens
    return estimate_usage_cost(model, usage)


def _responses_output_budget_exhausted(data: dict[str, Any]) -> bool:
    """Return True when the reply is a terminal max_output_tokens truncation.

    Precisely: ``status=incomplete`` AND ``incomplete_details.reason ==
    max_output_tokens``. Callers only consult this after confirming there is no
    usable ``output_text`` and that the reply is not a ``content_filter`` refusal,
    so it never reclassifies a refusal or a partial-but-usable truncation.
    """
    incomplete = data.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
    return data.get("status") == "incomplete" and reason == "max_output_tokens"


def _responses_empty_text_error(data: dict[str, Any]) -> ProviderApiError:
    status = data.get("status") or "unknown"
    incomplete = data.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
    types = ",".join(_responses_output_item_types(data)) or "none"
    detail = f"; incomplete_reason={reason}" if reason else ""
    finish_reason = reason or status
    return ProviderApiError(
        502,
        "OpenAI Responses reply contained no output_text; "
        f"status={status}{detail}; output_types={types}",
        raw_response={
            "finish_reason": finish_reason,
            "native_finish_reason": reason,
            "refusal": None,
        },
    )


def _responses_max_output_tokens(kwargs: dict[str, Any], extra_body: dict[str, Any]) -> Any:
    body_limits = {
        key: extra_body.pop(key)
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens")
        if key in extra_body
    }
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = kwargs.get(key)
        if value is not None:
            return value
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        if key in body_limits:
            return body_limits[key]
    return None


def _responses_effort(kwargs: dict[str, Any], extra_body: dict[str, Any]) -> Any:
    # Always consume the translatable extra_body keys so they never leak into the
    # Responses payload, even when the explicit kwarg wins.
    body_effort = extra_body.pop("reasoning_effort", None)
    reasoning_option = extra_body.pop("reasoning", None)
    effort = kwargs.get("reasoning_effort")
    if effort is None:
        effort = body_effort
    if effort is None and isinstance(reasoning_option, dict):
        effort = reasoning_option.get("effort")
    return effort


@dataclass
class OpenAIResponsesClient:
    """OpenAI-SDK-shaped wrapper around OpenAI's native Responses API.

    The Responses API accepts ``reasoning.effort=max`` for the gpt-5.6 tiers,
    which Chat Completions rejects, so the whole gpt-5.6 family routes here.
    """

    api_key: str
    base_url: str = OPENAI_RESPONSES_URL
    timeout: int = 120

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=_OpenAIResponsesCompletions(self))


class _OpenAIResponsesCompletions:
    def __init__(self, parent: OpenAIResponsesClient) -> None:
        self._parent = parent
        self.base_url = parent.base_url

    def create(self, **kwargs: Any) -> Any:
        model = str(kwargs["model"])
        should_record_cooldown = kwargs.get("record_rate_limit_cooldown", True)
        messages = kwargs.get("messages") or []
        extra_body = kwargs.get("extra_body") or {}
        if extra_body and not isinstance(extra_body, dict):
            raise ValueError("extra_body/request_options must be a mapping")
        extra_body = dict(extra_body)

        instructions, input_items = _to_responses_input(messages)
        payload: dict[str, Any] = {"model": model, "input": input_items}
        if instructions:
            payload["instructions"] = instructions
        max_output_tokens = _responses_max_output_tokens(kwargs, extra_body)
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        effort = _responses_effort(kwargs, extra_body)
        if effort is not None:
            payload["reasoning"] = {"effort": str(effort)}
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        # Any remaining provider-specific request fields pass through verbatim.
        for key, value in extra_body.items():
            payload.setdefault(key, value)

        timeout = kwargs.get("timeout") or self._parent.timeout
        response = httpx.post(
            self._parent.base_url,
            headers={
                "Authorization": f"Bearer {self._parent.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            if response.status_code == 400 and _is_openai_content_policy_400(response):
                try:
                    parsed_body = response.json()
                except Exception:
                    parsed_body = {}
                err = refusal_from_http_error(status_code=400, body=parsed_body)
                err.headers = dict(response.headers)
                raise err
            try:
                parsed_body: dict[str, Any] | None = response.json()
                if not isinstance(parsed_body, dict):
                    parsed_body = None
            except Exception:
                parsed_body = None
            error = ProviderApiError(
                response.status_code,
                response.text[:500],
                headers=dict(response.headers),
                raw_response=parsed_body,
            )
            if (
                response.status_code == 429
                and should_record_cooldown
                and not is_billing_payload(error)
            ):
                record_rate_limit_cooldown(
                    provider=provider_from_base_url(self._parent.base_url),
                    model=model,
                    headers=dict(response.headers),
                    error=error,
                )
            raise error

        data = _provider_response_json(response, "OpenAI Responses provider")
        usage = _responses_usage(data, model)
        refusal_text = _responses_refusal_text(data)
        if refusal_text is not None:
            raise ProviderRefusalError(refusal_text, raw_response=data, usage=usage)
        content = _extract_responses_text(data)
        if not content:
            if _responses_output_budget_exhausted(data):
                # Terminal: reasoning consumed the whole output budget. Attach the
                # billed usage (reasoning tokens were spent) and let the runner
                # exclude the item instead of retrying it as a transient 502.
                raise ProviderOutputBudgetExhaustedError(
                    "OpenAI Responses output budget exhausted; "
                    "incomplete_reason=max_output_tokens; reasoning-only output",
                    raw_response=data,
                    usage=usage,
                )
            error = _responses_empty_text_error(data)
            error.usage = usage
            raise error
        return SimpleNamespace(
            model=data.get("model", model),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
            raw_response=data,
        )


def make_provider_client(
    model_cfg: dict[str, Any],
    *,
    openai_factory: Callable[..., Any],
) -> Any:
    """Create the appropriate SDK-shaped client for a rendered model config."""
    provider_api = model_cfg.get("provider_api", "openai_compatible")
    base_url = model_cfg.get("base_url")
    api_key_env = model_cfg.get("api_key_env")
    if isinstance(api_key_env, str) and api_key_env:
        require_credential_destination(api_key_env, base_url or "https://openrouter.ai/api/v1")
    if provider_api == "gemini_generate_content" or is_gemini_generate_content_url(base_url):
        return GeminiGenerateContentClient(
            api_key=model_cfg["api_key"],
            base_url=base_url or GEMINI_GENERATE_CONTENT_BASE_URL,
        )
    if provider_api == "anthropic_messages" or is_anthropic_messages_url(base_url):
        return AnthropicMessagesClient(
            api_key=model_cfg["api_key"],
            base_url=base_url or ANTHROPIC_MESSAGES_URL,
        )
    if provider_api == "openai_responses" or is_openai_responses_url(base_url):
        return OpenAIResponsesClient(
            api_key=model_cfg["api_key"],
            base_url=base_url or OPENAI_RESPONSES_URL,
        )
    return openai_factory(
        api_key=model_cfg["api_key"],
        base_url=base_url,
        max_retries=0,
        timeout=model_cfg.get("timeout", DEFAULT_OPENAI_COMPATIBLE_TIMEOUT),
    )
