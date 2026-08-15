"""End-to-end evidence integration tests for T2 (signals v2 + evidence integration).

Production-shaped test cases per T2 brief:
- SDK PermissionDeniedError with guardrail body → model_signal/guardrail_permission_denied
- SDK PermissionDeniedError with dunning body → environment/billing
- Bare 403 no classifiable body → unknown/ambiguous_403 (evidence) while
  classify_failure_status still yields failed_auth (operational layer — UNCHANGED)
- content_filter via native_finish_reason only → bounded_retry(1)
- Gemini SAFETY → stochastic_retry(2)
- Gemini RECITATION → stochastic_retry(2)
- action_policy_for returns (action, max_retries)
- Blast-radius regression: classify_failure_status output map unchanged
"""
import pytest

from suite_tools.evidence import (
    ENVIRONMENT,
    INSTRUMENT_DEFECT,
    MODEL_SIGNAL,
    UNKNOWN,
    action_for,
    action_policy_for,
    classify_evidence,
)
from suite_tools.provider_client import ProviderRefusalError
from suite_tools.run_monitor import classify_failure_status


# ── Helper factories ──────────────────────────────────────────────────────────

def _exc(status=None, raw=None, body=None, text="HTTP error"):
    """Build a bare exception with status_code, raw_response and/or body."""
    class E(Exception):
        pass
    e = E(text)
    if status is not None:
        e.status_code = status
    if raw is not None:
        e.raw_response = raw
    if body is not None:
        e.body = body
    return e


# ── SDK-shaped exceptions (.body not .raw_response) ──────────────────────────

def test_sdk_permission_denied_guardrail_body_classifies_model_signal():
    """SDK PermissionDeniedError with guardrail body → model_signal/guardrail_permission_denied.

    OpenRouter sends a structured JSON body on 403. Real SDK exceptions carry
    the body in exc.body, NOT exc.raw_response. T1 added extract_raw_response;
    T2 wires it so classify_evidence (via classify_payload) sees the body.

    This is the live misclassification fixed by T2: without this wiring, the
    403 status_code falls through to classify_failure_status → failed_auth →
    environment/auth, silently retrying a terminal guardrail block.
    """
    e = _exc(status=403,
             body={"error": {"metadata": {
                 "error_type": "permission_denied",
                 "reasons": ["guardrail"],
             }}})
    result = classify_evidence(e)
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "guardrail_permission_denied"


def test_sdk_permission_denied_dunning_body_classifies_environment_billing():
    """SDK exc with Google dunning body in .body → environment/billing (not auth)."""
    e = _exc(status=403,
             body={"error": {
                 "code": 403,
                 "status": "PERMISSION_DENIED",
                 "message": "Lightning dunning decision is deny for project: projects/955415727012",
             }})
    result = classify_evidence(e)
    assert result["evidence_class"] == ENVIRONMENT
    assert result["category"] == "billing"


def test_openai_insufficient_quota_classifies_as_terminal_billing():
    e = _exc(
        status=429,
        raw={"error": {"code": "insufficient_quota", "message": "You exceeded your quota"}},
    )
    result = classify_evidence(e)
    assert result["evidence_class"] == ENVIRONMENT
    assert result["category"] == "billing"
    assert action_policy_for(result) == ("terminal_owed", 0)
    assert classify_failure_status(e) == "failed_billing"


# ── Bare 403 with no classifiable body ────────────────────────────────────────

def test_bare_403_no_body_is_ambiguous_403_in_evidence_layer():
    """status=403 + no classifiable body → unknown/ambiguous_403 (evidence layer).

    This is distinct from the operational layer: classify_failure_status still
    returns 'failed_auth' for bare 403s (D1 blast-radius rule; that map feeds
    scoring summaries and dashboards and must not change). Evidence layer yields
    ambiguous_403 so the failure goes to the review queue, not silently retried
    as an auth failure.
    """
    e = _exc(status=403, text="HTTP 403: Permission Denied")
    result = classify_evidence(e)
    assert result["evidence_class"] == UNKNOWN
    assert result["category"] == "ambiguous_403"


def test_bare_403_classify_failure_status_still_returns_failed_auth():
    """Operational layer classify_failure_status(403) → 'failed_auth' unchanged.

    D1 blast-radius constraint: aita scoring.py:435,514; sus runner.py:535,618;
    run_monitor.py:103-114,302-314; live_dashboard.py:1637 all consume this map.
    It must not change. This test is a regression guard.
    """
    e = _exc(status=403, text="HTTP 403: Permission Denied")
    assert classify_failure_status(e) == "failed_auth"


def test_classified_403_does_not_become_ambiguous_403():
    """A 403 with a recognizable payload skips ambiguous_403 (signals table wins)."""
    e = _exc(status=403,
             raw={"error": {"metadata": {"error_type": "permission_denied",
                                          "reasons": ["guardrail"]}}})
    result = classify_evidence(e)
    # Must be guardrail, NOT ambiguous_403
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "guardrail_permission_denied"


# ── native_finish_reason in _refusal_category ─────────────────────────────────

def test_refusal_category_reads_native_finish_reason():
    """_refusal_category consults native_finish_reason when finish_reason absent.

    OpenRouter-via-OpenAI may set native_finish_reason without a top-level
    finish_reason. The D8 requirement: native_finish_reason consulted in
    _refusal_category (evidence.py:47-58).
    """
    e = ProviderRefusalError("blocked", raw_response={"native_finish_reason": "content_filter"})
    result = classify_evidence(e)
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "content_filter"


def test_refusal_category_prefers_finish_reason_over_native():
    """finish_reason takes precedence when both are present."""
    e = ProviderRefusalError("blocked",
                             raw_response={"finish_reason": "refusal",
                                           "native_finish_reason": "stop"})
    result = classify_evidence(e)
    assert result["category"] == "refusal"


# ── content_filter via native_finish_reason (non-refusal exception) ───────────

def test_content_filter_via_native_finish_reason_only():
    """native_finish_reason=content_filter (no top-level finish_reason) → model_signal/content_filter.

    OpenRouter can emit {native_finish_reason: 'content_filter'} without a
    top-level finish_reason. Rule 5 now consults native_finish_reason.
    """
    e = _exc(raw={"native_finish_reason": "content_filter"})
    result = classify_evidence(e)
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "content_filter"


def test_content_filter_via_native_finish_reason_bounded_retry_policy():
    """content_filter via native_finish_reason → bounded_retry(1) in retry_policy."""
    e = _exc(raw={"native_finish_reason": "content_filter"})
    result = classify_evidence(e)
    assert result.get("retry_policy") == {"kind": "bounded_retry", "max_retries": 1}


# ── Gemini signals end-to-end ─────────────────────────────────────────────────

def test_gemini_safety_via_classify_evidence_has_stochastic_retry_2():
    """Gemini SAFETY → model_signal/SAFETY with stochastic_retry(2)."""
    e = _exc(raw={"candidates": [{"finishReason": "SAFETY"}]})
    result = classify_evidence(e)
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "SAFETY"
    assert result.get("retry_policy") == {"kind": "stochastic_retry", "max_retries": 2}


def test_gemini_recitation_via_classify_evidence():
    """Gemini RECITATION → model_signal/RECITATION, stochastic_retry(2)."""
    e = _exc(raw={"candidates": [{"finishReason": "RECITATION"}]})
    result = classify_evidence(e)
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "RECITATION"
    assert result.get("retry_policy") == {"kind": "stochastic_retry", "max_retries": 2}


# ── action_policy_for ─────────────────────────────────────────────────────────

def test_action_policy_for_exists_and_returns_tuple():
    """action_policy_for(evidence) → (action: str, max_retries: int)."""
    evidence = {"evidence_class": MODEL_SIGNAL, "category": "refusal"}
    result = action_policy_for(evidence)
    assert isinstance(result, tuple), "action_policy_for must return a tuple"
    assert len(result) == 2
    action, max_retries = result
    assert isinstance(action, str)
    assert isinstance(max_retries, int)


def test_action_for_uses_class_logic_not_retry_policy():
    """action_for must NOT consult retry_policy — it is a backward-compat function.

    Runners written before T2 call action_for; their retry semantics must not
    silently change. Policy-driven retry is T3's executor's job; it calls
    action_policy_for to get the (action, max_retries) tuple.

    For content_filter: action_for returns 'record_outcome' (class-based, no policy);
    action_policy_for returns ('retry_bounded', 1) (policy-aware, for T3).
    """
    content_filter_ev = {
        "evidence_class": MODEL_SIGNAL,
        "category": "content_filter",
        "retry_policy": {"kind": "bounded_retry", "max_retries": 1},
    }
    # action_for ignores retry_policy → class-based result
    assert action_for(content_filter_ev) == "record_outcome"
    # action_policy_for exposes the bound for T3's executor
    action, max_retries = action_policy_for(content_filter_ev)
    assert action == "retry_bounded"
    assert max_retries == 1


def test_action_policy_for_bounded_retry_returns_max_retries():
    """bounded_retry(1) in retry_policy → action_policy_for returns ('retry_bounded', 1)."""
    evidence = {
        "evidence_class": MODEL_SIGNAL,
        "category": "content_filter",
        "retry_policy": {"kind": "bounded_retry", "max_retries": 1},
    }
    action, max_retries = action_policy_for(evidence)
    assert action == "retry_bounded"
    assert max_retries == 1


def test_action_policy_for_stochastic_retry_returns_max_retries():
    """stochastic_retry(2) → action_policy_for returns ('retry_bounded', 2)."""
    evidence = {
        "evidence_class": MODEL_SIGNAL,
        "category": "SAFETY",
        "stochastic": True,
        "retry_policy": {"kind": "stochastic_retry", "max_retries": 2},
    }
    action, max_retries = action_policy_for(evidence)
    assert action == "retry_bounded"
    assert max_retries == 2


def test_action_policy_for_terminal_returns_zero_retries():
    """terminal retry_policy → action from evidence class logic, max_retries=0."""
    evidence = {
        "evidence_class": MODEL_SIGNAL,
        "category": "refusal",
        "retry_policy": {"kind": "terminal", "max_retries": 0},
    }
    action, max_retries = action_policy_for(evidence)
    assert action == "record_outcome"  # MODEL_SIGNAL non-stochastic → record_outcome
    assert max_retries == 0


def test_action_policy_for_no_retry_policy_fallback_to_class_logic():
    """When retry_policy absent, action_policy_for uses existing class-based logic."""
    # stochastic MODEL_SIGNAL (old format without retry_policy)
    ev = {"evidence_class": MODEL_SIGNAL, "category": "SAFETY", "stochastic": True}
    action, max_retries = action_policy_for(ev)
    assert action == "retry_bounded"
    assert max_retries == 0  # legacy: no per-signal bound, uses runner counter

    # ENVIRONMENT rate_limit
    ev2 = {"evidence_class": ENVIRONMENT, "category": "rate_limit"}
    action2, max_retries2 = action_policy_for(ev2)
    assert action2 == "retry_bounded"
    assert max_retries2 == 0


def test_action_for_content_filter_is_record_outcome_backward_compat():
    """content_filter via classify_evidence → action_for returns 'record_outcome'.

    action_for is backward-compat (class-based only; ignores retry_policy).
    Policy-driven retry is T3's executor's job via action_policy_for.
    """
    e = _exc(raw={"finish_reason": "content_filter"})
    evidence = classify_evidence(e)
    # action_for: backward-compat class-based result
    assert action_for(evidence) == "record_outcome"
    # action_policy_for: exposes the bounded retry for T3's executor
    action, max_retries = action_policy_for(evidence)
    assert action == "retry_bounded"
    assert max_retries == 1


# ── Blast-radius regression: classify_failure_status map unchanged ────────────

def test_t3_contract_constructed_refusal_error_has_no_bounded_policy():
    """action_policy_for(classify_evidence(ProviderRefusalError)) is always terminal/(0).

    T3 SHARP EDGE: ProviderRefusalError is constructed AFTER the policy decision.
    Pre-construction consultation must go through classify_payload on the raw body
    (that is where retry_policy lives). Do NOT construct the refusal early and then
    call action_policy_for expecting a bounded policy — it will always return terminal.

    This test pins the contract so T3's executor cannot cut itself on this edge.
    """
    # Even with content_filter body, a constructed ProviderRefusalError yields terminal.
    # retry_policy is NOT present in evidence from ProviderRefusalError path.
    e_content_filter = ProviderRefusalError("blocked", raw_response={"finish_reason": "content_filter"})
    evidence = classify_evidence(e_content_filter)
    action, max_retries = action_policy_for(evidence)
    assert action == "record_outcome", \
        "Constructed ProviderRefusalError must not yield bounded policy (T3 sharp edge)"
    assert max_retries == 0

    # Same for generic refusal
    e_refusal = ProviderRefusalError("blocked", raw_response={})
    action2, max_retries2 = action_policy_for(classify_evidence(e_refusal))
    assert action2 == "record_outcome"
    assert max_retries2 == 0

    # Contrast: classify_payload on the same raw body DOES return bounded_retry(1)
    # This is what T3's executor should consult before constructing the exception.
    from suite_tools.provider_signals import classify_payload
    class _RawBody:
        raw_response = {"finish_reason": "content_filter"}
    payload_evidence = classify_payload(_RawBody())
    assert payload_evidence is not None
    payload_action, payload_max = action_policy_for(payload_evidence)
    assert payload_action == "retry_bounded"
    assert payload_max == 1


def test_classify_failure_status_blast_radius_regression():
    """The operational status vocabulary from classify_failure_status must not change.

    D1: this map feeds aita scoring.py:435,514; sus runner.py:535,618;
    run_monitor.py:103-114,302-314; live_dashboard.py:1637. Any change here
    breaks the entire result chain. This test is a regression guard asserting
    each representative status is unchanged.

    Note: 403 → 'failed_auth' is INTENTIONALLY preserved here. The evidence
    layer intercepts it at classify_evidence (→ ambiguous_403 or model_signal),
    but the operational layer is untouched. Two layers, one intent.
    """
    cases = [
        # (error_description, expected_status)
        (_exc(status=401, text="HTTP 401 Unauthorized"), "failed_auth"),
        (_exc(status=403, text="HTTP 403 Forbidden"), "failed_auth"),
        (_exc(text="401 Unauthorized"), "failed_auth"),
        (_exc(text="403 Forbidden"), "failed_auth"),
        (_exc(status=402, text="HTTP 402"), "failed_billing"),
        (_exc(text="Insufficient credits"), "failed_billing"),
        (
            _exc(
                status=429,
                raw={"error": {"code": "insufficient_quota"}},
                text="You exceeded your current quota",
            ),
            "failed_billing",
        ),
        (_exc(status=429, text="HTTP 429 Rate limited"), "failed_rate_limited"),
        (_exc(text="Error code: 429 - too many requests"), "failed_rate_limited"),
        (_exc(text="rate limit exceeded"), "failed_rate_limited"),
        (_exc(text="Adapter rejected incomplete artifact"), "failed_invalid"),
        (_exc(status=400, text="not a valid model ID"), "failed_invalid"),
        (_exc(text="missing score: resistance_a"), "failed_scoring"),
        (_exc(text="request timed out after 300 seconds"), "failed_timeout"),
        (_exc(status=500, text="HTTP 500 Internal Server Error"), "failed_provider"),
        (_exc(status=503, text="HTTP 503 Service Unavailable"), "failed_provider"),
    ]
    for err, expected in cases:
        got = classify_failure_status(err)
        assert got == expected, \
            f"classify_failure_status({err}) = {got!r}, expected {expected!r} — BLAST RADIUS VIOLATION"
