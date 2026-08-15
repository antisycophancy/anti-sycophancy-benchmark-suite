"""Evidence-class failure taxonomy + action policy (spec 015 §4, plan 014 §2)."""

from __future__ import annotations

from typing import Any

import httpx

from suite_tools.provider_client import (
    ProviderMalformedResponseError,
    ProviderOutputBudgetExhaustedError,
    ProviderRefusalError,
)
from suite_tools.provider_signals import classify_payload
from suite_tools.run_monitor import classify_failure_status

MODEL_SIGNAL = "model_signal"
ENVIRONMENT = "environment"
INSTRUMENT_DEFECT = "instrument_defect"
UNKNOWN = "unknown"

_STATUS_TO_EVIDENCE = {
    "failed_auth": (ENVIRONMENT, "auth"),
    "failed_billing": (ENVIRONMENT, "billing"),
    "failed_rate_limited": (ENVIRONMENT, "rate_limit"),
    "failed_invalid": (INSTRUMENT_DEFECT, "invalid_config"),
    "failed_scoring": (INSTRUMENT_DEFECT, "scoring"),
}

# Narrow, evidence-backed markers only (plan 014 class C examples). The broad
# "invalid request" marker was rejected in review as overmatching.
_PAYLOAD_MARKERS = (
    "unsupported parameter",
    "max_tokens' is not supported",
    "use 'max_completion_tokens'",
)

_ACTIONS = {
    (ENVIRONMENT, "rate_limit"): "retry_bounded",
    (ENVIRONMENT, "provider_5xx"): "retry_bounded",
    (ENVIRONMENT, "malformed_response"): "retry_bounded",
    (ENVIRONMENT, "timeout_connect"): "retry_bounded",
    (ENVIRONMENT, "timeout_read"): "terminal_owed",
    (ENVIRONMENT, "auth"): "terminal_owed",
    (ENVIRONMENT, "billing"): "terminal_owed",
}


def _refusal_category(error: ProviderRefusalError) -> str:
    raw = error.raw_response if isinstance(error.raw_response, dict) else {}
    error_obj = raw.get("error") if isinstance(raw.get("error"), dict) else {}
    code = error_obj.get("code")
    if code:
        return str(code)
    # Consult native_finish_reason in addition to finish_reason (plan 020 D8).
    finish_reason = raw.get("finish_reason") or raw.get("native_finish_reason")
    if finish_reason:
        return str(finish_reason)
    if error.stop_reason:
        return str(error.stop_reason)
    return "refusal"


def classify_evidence(error: object) -> dict[str, Any]:
    """Classify a failure by evidentiary value. Typed exceptions take
    precedence; string heuristics are fallback; ambiguity yields UNKNOWN."""
    if isinstance(error, ProviderOutputBudgetExhaustedError):
        return {"evidence_class": MODEL_SIGNAL, "category": "output_budget_exhausted"}
    if isinstance(error, ProviderRefusalError):
        return {"evidence_class": MODEL_SIGNAL, "category": _refusal_category(error)}
    if isinstance(error, (httpx.ConnectTimeout, httpx.PoolTimeout)):
        return {"evidence_class": ENVIRONMENT, "category": "timeout_connect"}
    if isinstance(error, httpx.TimeoutException):
        return {"evidence_class": ENVIRONMENT, "category": "timeout_read"}

    # SDK-shaped exceptions (with .body or .response.json() but not .raw_response)
    # are handled by classify_payload's _extract_body helper (plan 020 D8; T1 seam).
    payload = classify_payload(error)
    if payload is not None:
        return payload
    if isinstance(error, ProviderMalformedResponseError):
        return {"evidence_class": ENVIRONMENT, "category": "malformed_response"}

    # NEW (plan 020 D1): status==403 with no classifiable body → unknown/ambiguous_403.
    # Action: halt → review queue. The OPERATIONAL layer (classify_failure_status)
    # remains unchanged and still maps 403 → failed_auth; two layers, one intent.
    # 401 is intentionally excluded: always environment/auth (no ambiguity).
    status_code = getattr(error, "status_code", None)
    if status_code == 403:
        return {"evidence_class": UNKNOWN, "category": "ambiguous_403"}

    text = str(error).lower()
    if (status_code == 400 or "error code: 400" in text) and any(
        marker in text for marker in _PAYLOAD_MARKERS
    ):
        return {"evidence_class": INSTRUMENT_DEFECT, "category": "payload"}
    if isinstance(status_code, int) and status_code >= 500:
        return {"evidence_class": ENVIRONMENT, "category": "provider_5xx"}

    legacy = classify_failure_status(error)
    if legacy == "failed_timeout":
        return {"evidence_class": ENVIRONMENT, "category": "timeout_read"}
    if legacy in _STATUS_TO_EVIDENCE:
        evidence_class, category = _STATUS_TO_EVIDENCE[legacy]
        return {"evidence_class": evidence_class, "category": category}
    return {"evidence_class": UNKNOWN, "category": "unclassified"}


def _action_from_evidence_class(evidence: dict[str, Any]) -> str:
    """Derive action from evidence class + category (no retry_policy consulted).

    This is the canonical class-based action logic. action_policy_for delegates
    here when retry_policy is absent or kind is terminal.
    """
    evidence_class = evidence.get("evidence_class")
    if evidence_class == MODEL_SIGNAL:
        if evidence.get("stochastic"):
            return "retry_bounded"
        return "record_outcome"
    if evidence_class in (INSTRUMENT_DEFECT, UNKNOWN):
        return "halt"
    return _ACTIONS.get((evidence_class, evidence.get("category")), "terminal_owed")


def action_policy_for(evidence: dict[str, Any]) -> tuple[str, int]:
    """Return (action, max_retries) driven by retry_policy when present.

    retry_policy is embedded in evidence dicts returned by classify_payload
    (plan 020 D8). When retry_policy is absent (typed exceptions, legacy paths)
    or kind is terminal, falls through to class-based action with max_retries=0.

    T3 consumers: the shared policy executor owns the per-unit attempt counter;
    max_retries here is the BOUND from the signals table, not a remaining count.

    SHARP EDGE FOR T3:
    ProviderRefusalError is constructed AFTER the policy decision is made; it
    carries no retry_policy in its evidence dict (it is NOT routed through
    classify_payload). Pre-construction consultation must go through
    classify_payload(raw_body) to get the retry bound. Calling action_policy_for
    on evidence from a constructed ProviderRefusalError always returns
    ("record_outcome", 0) — i.e. terminal — even for content_filter bodies.
    T3's executor must consult classify_payload BEFORE raising the refusal
    exception, not after catching it.
    """
    retry_policy = evidence.get("retry_policy")
    if isinstance(retry_policy, dict):
        kind = retry_policy.get("kind")
        max_retries = int(retry_policy.get("max_retries", 0))
        if kind in ("bounded_retry", "stochastic_retry"):
            return "retry_bounded", max_retries
        # terminal: fall through to class-based action, max_retries=0

    # No retry_policy or terminal kind: use existing class-based logic.
    # max_retries=0 means "no per-signal bound; use runner counter (legacy)."
    return _action_from_evidence_class(evidence), 0


def action_for(evidence: dict[str, Any]) -> str:
    """Runner action for THIS attempt (backward-compatible, class-based only).

    Uses pure evidence-class logic — does NOT consult retry_policy. Existing
    runner call sites (aita, epis, sus) use this; their retry behavior is
    unchanged until T3's policy executor is wired in.

    Callers needing policy-aware retry bounds must call action_policy_for,
    which returns (action, max_retries) and does consult retry_policy.
    """
    return _action_from_evidence_class(evidence)
