"""Structured-payload classification for provider error bodies (spec 015 §4, plan 014a).

Rules evaluated in order; returns a result dict or None.
None means no rule recognized the body at all — the caller falls through to
legacy heuristics. An explicit UNKNOWN dict (e.g. ambiguous_permission_denied)
means the payload *was* recognized but is genuinely ambiguous — the legacy
403→auth heuristic must NOT see it.

Every result dict includes a ``retry_policy`` field (plan 020 D8):
    {"kind": "terminal" | "bounded_retry" | "stochastic_retry", "max_retries": int}
terminal ⇒ max_retries=0. Consumers MUST check this field before scheduling
retries; the ``stochastic`` boolean is preserved for back-compat where it was
previously emitted (derived from retry_policy kind == stochastic_retry).

014a providers with NO typed signal:
    Mistral — refusal text under finish_reason="stop"; indistinguishable from
              normal completion at signal level (no typed code).
    Grok/xAI — message.refusal field only; no content_filter finish_reason
               documented. UNVERIFIED distinct signal.
    Kimi/Moonshot — content_filter UNVERIFIED (3rd-party sources only).
These providers fall entirely to the judge/text layer + review queue. Do NOT
add rules for them here without verified provider documentation.

This module must NOT import suite_tools.evidence; evidence imports us.
"""

from __future__ import annotations

import json
from typing import Any

SIGNALS_TABLE_VERSION = "provider-signals-v2"

# ── Retry policy constants (plan 020 D8) ──────────────────────────────────────
_TERMINAL = {"kind": "terminal", "max_retries": 0}
_BOUNDED_RETRY_1 = {"kind": "bounded_retry", "max_retries": 1}
_STOCHASTIC_RETRY_2 = {"kind": "stochastic_retry", "max_retries": 2}

# OpenRouter error_type values that indicate a model-level content signal.
_OR_MODEL_SIGNAL_TYPES = {"content_policy_violation", "refusal"}

# Per-type retry policy for OpenRouter signals (014a: content_policy_violation
# is borderline-stochastic → bounded_retry(1); refusal is model decision →
# terminal(0)).
_OR_RETRY_POLICY: dict[str, dict[str, Any]] = {
    "content_policy_violation": _BOUNDED_RETRY_1,
    "refusal": _TERMINAL,
}

# Google finish reasons that are stochastic model signals (014a: Gemini is
# threshold-based, borderline STOCHASTIC; a single trigger is NOT proof of
# a permanent block — safetySettings-dependent).
# RECITATION added per 014a: same threshold-stochastic tier as SAFETY.
_GEMINI_STOCHASTIC_FINISH_REASONS = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "RECITATION",
}

# Billing/dunning markers in Google PERMISSION_DENIED messages.
_BILLING_MARKERS = ("dunning", "billing", "payment")

# Provider billing codes/types that can arrive on HTTP 429 as well as HTTP 402.
_BILLING_ERROR_CODES = {
    "billing_hard_limit_reached",
    "insufficient_credits",
    "insufficient_quota",
}
_BILLING_ERROR_TYPES = {"billing_error"}

# OpenAI error codes that map to model signals (deterministic 400s → terminal).
_OPENAI_MODEL_SIGNAL_CODES = {"content_policy_violation", "invalid_prompt"}

# Qwen-family error codes that map to model signals.
# 014a: pre-inference inspection fires before model call; all deterministic.
_QWEN_MODEL_SIGNAL_CODES = {
    "data_inspection_failed",
    "IPInfringementSuspect",
    "FaqRuleBlocked",
    "CustomRoleBlocked",
}


def _result(evidence_class: str, category: str, **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "evidence_class": evidence_class,
        "category": category,
        "signal_source": SIGNALS_TABLE_VERSION,
    }
    d.update(extra)
    return d


def _extract_body(error: object) -> dict[str, Any] | None:
    """Extract a structured body from an error, checking multiple SDK shapes.

    Priority:
    1. error.raw_response if a dict (ProviderApiError and subclasses — set by T1).
    2. error.body if a dict (OpenAI/SDK exception shape, e.g. PermissionDeniedError).
    3. error.response.json() as a fallback (SDK response wrapper).
    4. Parse str(error) as JSON (last resort).

    Returns None when no structured body is available. Never raises.
    """
    raw = getattr(error, "raw_response", None)
    if isinstance(raw, dict):
        return raw

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body

    response = getattr(error, "response", None)
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Last resort: try to parse the error text as JSON.
    text = str(error)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def classify_payload(error: object) -> dict[str, Any] | None:
    """Classify a provider error body against the signals table.

    Returns a result dict if any rule matches; None if no rule matched at all.
    A result dict with evidence_class="unknown" is still a match — it prevents
    the caller from falling through to legacy status heuristics.

    Every result includes ``retry_policy`` (plan 020 D8).
    """
    raw = _extract_body(error)

    if not isinstance(raw, dict):
        return None

    error_obj = raw.get("error")
    if not isinstance(error_obj, dict):
        error_obj = {}

    metadata = error_obj.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    # ── Rule 0: terminal provider billing signals ─────────────────────────────
    # OpenAI returns insufficient_quota with HTTP 429, while Anthropic uses
    # HTTP 402 + billing_error. Neither is a throughput limit.
    status_code = getattr(error, "status_code", None)
    error_type = error_obj.get("type")
    code = error_obj.get("code")
    if (
        status_code == 402
        or error_type in _BILLING_ERROR_TYPES
        or code in _BILLING_ERROR_CODES
    ):
        provider = (
            "anthropic" if error_type in _BILLING_ERROR_TYPES
            else "openai" if code in _BILLING_ERROR_CODES
            else "unknown"
        )
        provider_code = error_type or code or status_code
        return _result(
            "environment",
            "billing",
            retry_policy=_TERMINAL,
            provider=provider,
            provider_code=str(provider_code),
        )

    # ── Rule 1: OpenRouter top-level or nested error_type ─────────────────────
    # Check error_type at error.metadata level (OpenRouter wraps errors).
    # content_policy_violation is borderline-stochastic → bounded_retry(1).
    # refusal is a model decision → terminal(0).
    or_error_type = metadata.get("error_type") or raw.get("error_type")
    if or_error_type in _OR_MODEL_SIGNAL_TYPES:
        retry_policy = _OR_RETRY_POLICY.get(or_error_type, _TERMINAL)
        return _result("model_signal", str(or_error_type), retry_policy=retry_policy,
                       provider="openrouter", provider_code=str(or_error_type))

    # ── Rule 2: OpenRouter guardrail permission_denied ─────────────────────────
    # Guardrail block → terminal; halt and route to review queue.
    if or_error_type == "permission_denied" and (
        metadata.get("reasons") or metadata.get("flagged_input")
    ):
        return _result("model_signal", "guardrail_permission_denied",
                       retry_policy=_TERMINAL,
                       provider="openrouter", provider_code=str(or_error_type))

    # ── Rules 3 & 4: Google PERMISSION_DENIED ─────────────────────────────────
    google_status = error_obj.get("status")
    if google_status == "PERMISSION_DENIED":
        msg = (error_obj.get("message") or "").lower()
        if any(marker in msg for marker in _BILLING_MARKERS):
            return _result("environment", "billing", retry_policy=_TERMINAL,
                           provider="google", provider_code=str(google_status))
        # Recognized-but-ambiguous: explicit UNKNOWN so legacy auth never fires.
        # keep in review queue — terminal to prevent retry spin.
        return _result("unknown", "ambiguous_permission_denied", retry_policy=_TERMINAL,
                       provider="google", provider_code=str(google_status))

    # ── Rule 4b: DeepSeek choices[0].finish_reason == "content_filter" ────────
    # DeepSeek documents content_filter as deterministic (014a: "appears
    # deterministic"). Native API response bodies carry finish_reason in
    # choices[0], distinct from OpenRouter-normalized top-level finish_reason.
    # This rule fires BEFORE rule 5 so DeepSeek native bodies get terminal(0)
    # rather than bounded_retry(1).
    choices_list = raw.get("choices")
    if isinstance(choices_list, list) and choices_list:
        first_choice = choices_list[0]
        if isinstance(first_choice, dict) and first_choice.get("finish_reason") == "content_filter":
            return _result("model_signal", "content_filter", retry_policy=_TERMINAL,
                           provider="deepseek", provider_code="content_filter")

    # ── Rule 5: top-level finish_reason or native_finish_reason == "content_filter"
    # native_finish_reason consulted per plan 020 D8 (rules 5 and 8).
    # Top-level finish_reason = OpenRouter-normalized path; content_filter is
    # borderline-stochastic per 014a for OpenAI-sourced signals → bounded_retry(1).
    top_finish = raw.get("finish_reason") or raw.get("native_finish_reason")
    if top_finish == "content_filter":
        return _result("model_signal", "content_filter", retry_policy=_BOUNDED_RETRY_1,
                       provider="openrouter", provider_code=str(top_finish))

    # ── Rule 6: Anthropic content-filter 400 ──────────────────────────────────
    # 014a: "400 invalid_request_error 'Output blocked by content filtering
    # policy'" — Anthropic uses error.type (not error.code) and a specific
    # message. Deterministic block → terminal(0).
    # Note: Anthropic's cyber-content 400 carries a code field and is caught
    # by rule 7 below (via refusal_from_http_error on the client side).
    error_message = (error_obj.get("message") or "").lower()
    if "content filtering policy" in error_message:
        return _result("model_signal", "content_filter", retry_policy=_TERMINAL,
                       provider="anthropic", provider_code="content_filter")

    # ── Rule 7: OpenAI error.code ─────────────────────────────────────────────
    # Deterministic 400-level codes → terminal(0).
    if code in _OPENAI_MODEL_SIGNAL_CODES:
        return _result("model_signal", str(code), retry_policy=_TERMINAL,
                       provider="openai", provider_code=str(code))

    # ── Rule 8: Qwen error.code ───────────────────────────────────────────────
    # Pre-inference inspection codes; all deterministic → terminal(0).
    if code in _QWEN_MODEL_SIGNAL_CODES:
        return _result("model_signal", str(code), retry_policy=_TERMINAL,
                       provider="qwen", provider_code=str(code))

    # ── Rule 9: OpenAI Responses API refusal output item ─────────────────────
    # 014a: "output item type='refusal' + refusal text; status=incomplete +
    # incomplete_details.reason". Explicit model refusal → terminal(0).
    # Note: the Responses client normalizes this to ProviderRefusalError on the
    # happy path (via _responses_refusal_text); this rule catches raw Responses
    # bodies arriving via SDK exceptions or non-standard paths.
    output_items = raw.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            # Direct refusal output item (type=refusal at output level)
            if item.get("type") == "refusal":
                return _result("model_signal", "refusal", retry_policy=_TERMINAL,
                               provider="openai", provider_code="refusal")
            # Refusal content part nested inside a message output item
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "refusal":
                    return _result("model_signal", "refusal", retry_policy=_TERMINAL,
                                   provider="openai", provider_code="refusal")

    # ── Rule 9b: OpenAI Responses API status=incomplete + incomplete_details.reason
    # == "content_filter". The Responses client surfaces this as ProviderRefusalError
    # (via _responses_refusal_text), so this rule primarily classifies raw Responses
    # API bodies arriving via SDK exceptions, test probes, or alternative paths.
    # Same bounded-stochastic tier as Rule 5 (borderline stochastic, retry once).
    if raw.get("status") == "incomplete":
        _incomplete = raw.get("incomplete_details")
        if isinstance(_incomplete, dict) and _incomplete.get("reason") == "content_filter":
            return _result("model_signal", "content_filter", retry_policy=_BOUNDED_RETRY_1,
                           provider="openai", provider_code="content_filter")

    # ── Rule 10: Gemini candidates[0].finishReason or native_finish_reason ────
    # native_finish_reason at top level covers OpenRouter-proxied Gemini signals
    # (plan 020 D8: native_finish_reason consulted in rules 5 and 8).
    # Gemini is threshold-stochastic (014a) → stochastic_retry(2) + stochastic=True.
    native_finish = raw.get("native_finish_reason")
    if native_finish in _GEMINI_STOCHASTIC_FINISH_REASONS:
        return _result("model_signal", str(native_finish),
                       stochastic=True, retry_policy=_STOCHASTIC_RETRY_2,
                       provider="google", provider_code=str(native_finish))

    candidates = raw.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            finish_reason = first.get("finishReason")
            if finish_reason in _GEMINI_STOCHASTIC_FINISH_REASONS:
                return _result("model_signal", str(finish_reason),
                               stochastic=True, retry_policy=_STOCHASTIC_RETRY_2,
                               provider="google", provider_code=str(finish_reason))

    # ── Rule 11: Gemini promptFeedback.blockReason ────────────────────────────
    # Pre-generation blocks are same threshold-stochastic tier as candidate-level
    # signals → stochastic_retry(2).
    prompt_feedback = raw.get("promptFeedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
        block_reason = prompt_feedback["blockReason"]
        return _result("model_signal", str(block_reason),
                       stochastic=True, retry_policy=_STOCHASTIC_RETRY_2,
                       provider="google", provider_code=str(block_reason))

    # No rule matched.
    return None


def is_billing_payload(error: object) -> bool:
    """Return true when a structured provider payload is a billing failure."""
    result = classify_payload(error)
    return bool(
        result
        and result.get("evidence_class") == "environment"
        and result.get("category") == "billing"
    )
