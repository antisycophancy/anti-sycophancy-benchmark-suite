"""Table-driven tests for provider_signals v2.

Every row in 014a-provider-refusal-signal-matrix.md must have a corresponding
test here. Tests are written RED-first (T2 TDD phase).
"""
from suite_tools.provider_signals import classify_payload, SIGNALS_TABLE_VERSION


def _err(raw=None, text="HTTP error", status=None):
    class E(Exception):
        pass
    e = E(text)
    if raw is not None:
        e.raw_response = raw
    if status is not None:
        e.status_code = status
    return e


def _sdk_err(body=None, status=None, text="HTTP error"):
    """Simulate an SDK exception that has .body but NOT .raw_response."""
    class E(Exception):
        pass
    e = E(text)
    if body is not None:
        e.body = body
    if status is not None:
        e.status_code = status
    return e


# ── Version ───────────────────────────────────────────────────────────────────

def test_signals_table_version_is_v2():
    assert SIGNALS_TABLE_VERSION == "provider-signals-v2"


# ── retry_policy present on every classified result ───────────────────────────

def test_all_rules_emit_retry_policy():
    """Every matched rule must include a retry_policy dict."""
    cases = [
        _err(raw={"error": {"metadata": {"error_type": "content_policy_violation"}}}),
        _err(raw={"error": {"metadata": {"error_type": "refusal"}}}),
        _err(raw={"error": {"metadata": {"error_type": "permission_denied", "reasons": ["guardrail"]}}}),
        _err(raw={"error": {"type": "billing_error"}}, status=402),
        _err(raw={"error": {"code": "insufficient_quota"}}, status=429),
        _err(raw={"error": {"code": 403, "status": "PERMISSION_DENIED",
                            "message": "Lightning dunning decision is deny for project: projects/1"}}),
        _err(raw={"error": {"code": 403, "status": "PERMISSION_DENIED",
                            "message": "caller does not have permission"}}),
        _err(raw={"finish_reason": "content_filter"}),
        _err(raw={"error": {"code": "content_policy_violation"}}),
        _err(raw={"error": {"code": "invalid_prompt"}}),
        _err(raw={"error": {"code": "data_inspection_failed"}}),
        _err(raw={"candidates": [{"finishReason": "SAFETY"}]}),
        _err(raw={"candidates": [{"finishReason": "RECITATION"}]}),
        _err(raw={"promptFeedback": {"blockReason": "SAFETY"}}),
    ]
    for e in cases:
        result = classify_payload(e)
        assert result is not None, f"No result for {e.raw_response}"
        assert "retry_policy" in result, f"Missing retry_policy in {result}"
        rp = result["retry_policy"]
        assert rp["kind"] in ("terminal", "bounded_retry", "stochastic_retry"), f"Bad kind in {rp}"
        assert isinstance(rp["max_retries"], int), f"Bad max_retries in {rp}"
        if rp["kind"] == "terminal":
            assert rp["max_retries"] == 0, f"Terminal must have 0 retries: {rp}"


# ── OpenRouter rules (014a row: OpenRouter) ───────────────────────────────────

def test_openrouter_content_policy_violation_bounded_retry():
    """content_policy_violation is borderline-stochastic per 014a → bounded_retry(1)."""
    e = _err(raw={"error": {"metadata": {"error_type": "content_policy_violation"}}})
    result = classify_payload(e)
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_policy_violation"
    assert result["retry_policy"] == {"kind": "bounded_retry", "max_retries": 1}


def test_openrouter_refusal_is_terminal():
    """OpenRouter refusal = model refusal, deterministic → terminal(0)."""
    e = _err(raw={"error": {"metadata": {"error_type": "refusal"}}})
    result = classify_payload(e)
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "refusal"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_openrouter_guardrail_permission_denied_is_terminal():
    """OpenRouter permission_denied guardrail → terminal(0); review queue."""
    e = _err(raw={"error": {"metadata": {"error_type": "permission_denied", "reasons": ["guardrail"]}}})
    result = classify_payload(e)
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "guardrail_permission_denied"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_openrouter_permission_denied_via_flagged_input():
    """OpenRouter permission_denied via flagged_input → same terminal guardrail signal."""
    e = _err(raw={"error": {"metadata": {"error_type": "permission_denied",
                                          "flagged_input": "some prompt text"}}})
    result = classify_payload(e)
    assert result["category"] == "guardrail_permission_denied"
    assert result["retry_policy"]["kind"] == "terminal"


# ── Google PERMISSION_DENIED rules (014a row: Gemini) ─────────────────────────

def test_google_billing_is_terminal():
    """Google dunning PERMISSION_DENIED → environment/billing, terminal(0)."""
    e = _err(raw={"error": {"code": 403, "status": "PERMISSION_DENIED",
                            "message": "Lightning dunning decision is deny for project: projects/1"}})
    result = classify_payload(e)
    assert result["evidence_class"] == "environment"
    assert result["category"] == "billing"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_anthropic_billing_error_is_terminal():
    """Anthropic HTTP 402 billing_error is not retryable."""
    e = _err(
        raw={"error": {"type": "billing_error", "message": "Account has insufficient credits"}},
        status=402,
    )
    result = classify_payload(e)
    assert result["evidence_class"] == "environment"
    assert result["category"] == "billing"
    assert result["provider"] == "anthropic"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_openai_insufficient_quota_is_terminal_billing_not_rate_limit():
    """OpenAI can return insufficient_quota with HTTP 429; it still requires refill."""
    e = _err(
        raw={"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}},
        status=429,
    )
    result = classify_payload(e)
    assert result["evidence_class"] == "environment"
    assert result["category"] == "billing"
    assert result["provider"] == "openai"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_bare_google_permission_denied_is_explicit_unknown_terminal():
    """Bare PERMISSION_DENIED → explicit UNKNOWN halt, terminal(0); legacy 403→auth blocked."""
    e = _err(raw={"error": {"code": 403, "status": "PERMISSION_DENIED",
                            "message": "caller does not have permission"}})
    result = classify_payload(e)
    assert result["evidence_class"] == "unknown"
    assert result["category"] == "ambiguous_permission_denied"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


# ── Rule 5: finish_reason content_filter (OpenAI / OpenRouter) ────────────────

def test_finish_reason_content_filter_bounded_retry():
    """top-level finish_reason=content_filter → model_signal, bounded_retry(1).

    014a: 'content_filter borderline-stochastic' for OpenAI; same treatment via
    native_finish_reason for OpenRouter-served OpenAI-upstream.
    """
    result = classify_payload(_err(raw={"finish_reason": "content_filter"}))
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_filter"
    assert result["retry_policy"] == {"kind": "bounded_retry", "max_retries": 1}


def test_native_finish_reason_content_filter_when_no_top_level():
    """native_finish_reason consulted when finish_reason absent (rule 5, 014a D8).

    OpenRouter may emit finish_reason='error' but native_finish_reason='content_filter'
    when the upstream (OpenAI) signal was content_filter.
    """
    result = classify_payload(_err(raw={"native_finish_reason": "content_filter"}))
    assert result is not None
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_filter"
    assert result["retry_policy"] == {"kind": "bounded_retry", "max_retries": 1}


# ── OpenAI error codes (014a row: OpenAI Chat) ────────────────────────────────

def test_openai_content_policy_violation_code_is_terminal():
    """400 error code content_policy_violation = deterministic block → terminal(0)."""
    e = _err(raw={"error": {"code": "content_policy_violation"}})
    result = classify_payload(e)
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_policy_violation"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_openai_invalid_prompt_code_is_terminal():
    """400 error code invalid_prompt = deterministic block → terminal(0)."""
    e = _err(raw={"error": {"code": "invalid_prompt"}})
    result = classify_payload(e)
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


# ── Qwen/DashScope codes (014a row: Qwen/DashScope) ──────────────────────────

def test_qwen_codes_are_terminal():
    """Qwen pre-inference inspection codes = deterministic 400 → terminal(0).

    014a: 'data_inspection_failed, IPInfringementSuspect, FaqRuleBlocked,
    CustomRoleBlocked' — pre-inference inspection fires before model call.
    """
    for code in ("data_inspection_failed", "IPInfringementSuspect",
                 "FaqRuleBlocked", "CustomRoleBlocked"):
        e = _err(raw={"error": {"code": code}})
        result = classify_payload(e)
        assert result is not None, f"No result for {code}"
        assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}, \
            f"Wrong policy for Qwen code {code}: {result['retry_policy']}"


# ── Gemini finishReason (014a row: Gemini) ────────────────────────────────────

def test_gemini_safety_is_stochastic_retry_2():
    """SAFETY → stochastic_retry(2); stochastic=True preserved for back-compat.

    014a: 'threshold-based, borderline STOCHASTIC; single SAFETY ≠ permanent block;
    safetySettings-dependent'. 2 attempts give statistical signal without burning
    budget on truly hard-blocked items.
    """
    result = classify_payload(_err(raw={"candidates": [{"finishReason": "SAFETY"}]}))
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "SAFETY"
    assert result["stochastic"] is True  # back-compat
    assert result["retry_policy"] == {"kind": "stochastic_retry", "max_retries": 2}


def test_gemini_prohibited_content_stochastic_retry():
    """PROHIBITED_CONTENT is same stochastic tier as SAFETY per 014a Gemini row."""
    result = classify_payload(_err(raw={"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]}))
    assert result["retry_policy"] == {"kind": "stochastic_retry", "max_retries": 2}
    assert result["stochastic"] is True


def test_gemini_blocklist_stochastic_retry():
    """BLOCKLIST (keyword match) is same stochastic tier."""
    result = classify_payload(_err(raw={"candidates": [{"finishReason": "BLOCKLIST"}]}))
    assert result["retry_policy"] == {"kind": "stochastic_retry", "max_retries": 2}


def test_gemini_spii_stochastic_retry():
    """SPII (sensitive PII) is same stochastic tier."""
    result = classify_payload(_err(raw={"candidates": [{"finishReason": "SPII"}]}))
    assert result["retry_policy"] == {"kind": "stochastic_retry", "max_retries": 2}


def test_gemini_recitation_added_as_stochastic_retry():
    """RECITATION = content-level quality signal, same threshold-stochastic tier.

    014a: Gemini is 'threshold-based, borderline STOCHASTIC' overall. RECITATION
    is a finishReason in the generativelanguagepb FinishReason enum; it fires when
    the model output is flagged for reciting copyrighted/training content. Like SAFETY,
    it is not proof of a permanent block (sampling variation can clear it), so
    stochastic_retry(2) matches the 014a Gemini stochastic tier.
    """
    result = classify_payload(_err(raw={"candidates": [{"finishReason": "RECITATION"}]}))
    assert result is not None, "RECITATION must be a recognized Gemini signal (014a addition)"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "RECITATION"
    assert result["retry_policy"] == {"kind": "stochastic_retry", "max_retries": 2}
    assert result["stochastic"] is True  # same tier as SAFETY


def test_gemini_native_finish_reason_consulted_in_rule_8():
    """native_finish_reason at top level → classified via rule 8 when no candidates.

    OpenRouter serving Gemini may surface the Gemini finishReason as
    native_finish_reason without a candidates array (stream-error path).
    """
    result = classify_payload(_err(raw={"native_finish_reason": "SAFETY"}))
    assert result is not None, "native_finish_reason SAFETY must classify via rule 8"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "SAFETY"
    assert result["retry_policy"]["kind"] == "stochastic_retry"


# ── promptFeedback.blockReason variants (014a row: Gemini) ────────────────────

def test_prompt_feedback_blockreason_safety_stochastic():
    """promptFeedback.blockReason SAFETY → stochastic_retry(2).

    014a: 'promptFeedback.blockReason' is a pre-generation block (before candidates
    are produced). Same threshold-stochastic tier as candidiate-level SAFETY.
    """
    result = classify_payload(_err(raw={"promptFeedback": {"blockReason": "SAFETY"}}))
    assert result["evidence_class"] == "model_signal"
    assert result["retry_policy"] == {"kind": "stochastic_retry", "max_retries": 2}


def test_prompt_feedback_blockreason_prohibited_content_stochastic():
    """promptFeedback.blockReason PROHIBITED_CONTENT → stochastic_retry(2)."""
    result = classify_payload(_err(raw={"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}))
    assert result["retry_policy"] == {"kind": "stochastic_retry", "max_retries": 2}


def test_prompt_feedback_blockreason_other_values_stochastic():
    """Other promptFeedback.blockReason values → stochastic_retry(2) same tier."""
    for reason in ("BLOCKLIST", "SPII"):
        result = classify_payload(_err(raw={"promptFeedback": {"blockReason": reason}}))
        assert result is not None
        assert result["retry_policy"]["kind"] == "stochastic_retry", \
            f"blockReason={reason} should be stochastic: {result}"


# ── SDK exception shapes (.body not .raw_response) ────────────────────────────

def test_sdk_exception_body_attribute_reaches_signals_table():
    """SDK exception with .body dict (no .raw_response) is classified via signals table.

    014a gap 1: OpenRouter 403 permission_denied arrives as SDK PermissionDeniedError
    with exc.body containing the structured body. T1 added extract_raw_response;
    T2 wires classify_payload to consume it so the guardrail body reaches the table.
    """
    e = _sdk_err(body={"error": {"metadata": {"error_type": "permission_denied",
                                               "reasons": ["guardrail"]}}},
                 status=403)
    result = classify_payload(e)
    assert result is not None, "SDK .body must reach the signals table"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "guardrail_permission_denied"


def test_sdk_exception_body_billing_reaches_signals_table():
    """SDK .body with Google dunning shape → environment/billing."""
    e = _sdk_err(body={"error": {"code": 403, "status": "PERMISSION_DENIED",
                                  "message": "dunning decision is deny for project: foo"}},
                 status=403)
    result = classify_payload(e)
    assert result is not None
    assert result["evidence_class"] == "environment"
    assert result["category"] == "billing"


# ── No-typed-signal providers documented in-source (014a row) ─────────────────

def test_mistral_grok_kimi_unrecognized_returns_none():
    """Mistral/Grok/Kimi emit no typed signal (014a).

    These providers use finish_reason='stop' with refusal text — indistinguishable
    from a normal completion at the signal level. classify_payload correctly returns
    None; the caller falls through to judge/text-layer + review queue.
    This test documents that expectation and guards against future 'fixes'.
    """
    # Mistral-style: finish_reason=stop, no error code
    mistral = _err(raw={"choices": [{"finish_reason": "stop", "message": {"content": "I can't help with that."}}]})
    assert classify_payload(mistral) is None, "Mistral text-level refusal must return None (no typed signal)"

    # Grok-style: refusal in message.refusal but no content_filter finish_reason
    grok = _err(raw={"choices": [{"finish_reason": "stop",
                                   "message": {"refusal": "I can't assist with that."}}]})
    assert classify_payload(grok) is None, "Grok refusal must return None (no typed signal per 014a)"


# ── Missing 014a rows (review-round additions) ────────────────────────────────

def test_anthropic_content_filter_400_is_model_signal_terminal():
    """Anthropic error-shape content-filter 400 → model_signal/content_filter, terminal.

    014a: "400 invalid_request_error 'Output blocked by content filtering policy'"
    Anthropic's block is deterministic (documented behavior) → terminal(0).
    Currently falls to legacy failed_invalid/halt without this rule.

    Anthropic uses error.type (not error.code) with a specific message.
    """
    body = {"error": {
        "type": "invalid_request_error",
        "message": "Output blocked by content filtering policy",
    }}
    result = classify_payload(_err(raw=body))
    assert result is not None, "Anthropic content-filter 400 must be recognized"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_filter"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_openai_responses_refusal_item_is_model_signal_terminal():
    """OpenAI Responses API refusal output item → model_signal/refusal, terminal.

    014a: "output item type='refusal' + refusal text; status=incomplete +
    incomplete_details.reason". Direct model refusal in the output → terminal(0).

    This covers raw Responses bodies arriving via SDK exceptions or non-standard
    paths; the Responses client normalizes to ProviderRefusalError on the happy path.
    """
    # Direct refusal item at output level
    body = {
        "status": "incomplete",
        "output": [{"type": "refusal", "refusal": "I cannot help with this request."}],
    }
    result = classify_payload(_err(raw=body))
    assert result is not None, "Responses refusal-item must be recognized"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "refusal"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_openai_responses_refusal_item_nested_in_message():
    """Refusal content part nested inside a message output item → same terminal signal."""
    body = {
        "status": "incomplete",
        "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}],
    }
    result = classify_payload(_err(raw=body))
    assert result is not None
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "refusal"
    assert result["retry_policy"]["kind"] == "terminal"


def test_deepseek_choices_content_filter_is_terminal():
    """DeepSeek choices[0].finish_reason='content_filter' → terminal (014a: deterministic).

    014a: "DeepSeek: finish_reason='content_filter' (documented); appears deterministic".
    Native API response bodies carry finish_reason in choices[0] (OpenAI-compat shape),
    distinct from OpenRouter-normalized top-level finish_reason (bounded_retry(1)).
    Rule 4b fires BEFORE rule 5 so DeepSeek native bodies get terminal(0).
    """
    body = {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
    result = classify_payload(_err(raw=body))
    assert result is not None, "choices[0].finish_reason=content_filter must be recognized"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_filter"
    assert result["retry_policy"] == {"kind": "terminal", "max_retries": 0}


def test_top_level_finish_reason_content_filter_is_still_bounded_retry():
    """Rule 5 (top-level finish_reason) is bounded_retry(1) regardless of rule 4b.

    OpenRouter normalizes to top-level finish_reason; DeepSeek rule 4b only
    fires on choices[0].finish_reason. The two rules are disjoint by body shape.
    """
    result = classify_payload(_err(raw={"finish_reason": "content_filter"}))
    assert result is not None
    assert result["retry_policy"] == {"kind": "bounded_retry", "max_retries": 1}


# ── provider / provider_code fields (Issue 3 — plan 020 D2) ──────────────────

def test_provider_and_provider_code_present_openrouter_rule1():
    """OpenRouter content_policy_violation → provider='openrouter', provider_code matches."""
    e = _err(raw={"error": {"metadata": {"error_type": "content_policy_violation"}}})
    result = classify_payload(e)
    assert result is not None
    assert result.get("provider") == "openrouter", f"Expected provider=openrouter, got {result.get('provider')}"
    assert result.get("provider_code") == "content_policy_violation", (
        f"Expected provider_code=content_policy_violation, got {result.get('provider_code')}"
    )


def test_provider_and_provider_code_present_openrouter_rule2():
    """OpenRouter guardrail permission_denied → provider='openrouter', provider_code='permission_denied'."""
    e = _err(raw={"error": {"metadata": {"error_type": "permission_denied", "reasons": ["guardrail"]}}})
    result = classify_payload(e)
    assert result is not None
    assert result.get("provider") == "openrouter"
    assert result.get("provider_code") == "permission_denied"


def test_provider_and_provider_code_present_google_billing():
    """Google billing PERMISSION_DENIED → provider='google', provider_code='PERMISSION_DENIED'."""
    e = _err(raw={"error": {"code": 403, "status": "PERMISSION_DENIED",
                            "message": "Lightning dunning decision is deny for project: projects/1"}})
    result = classify_payload(e)
    assert result is not None
    assert result.get("provider") == "google"
    assert result.get("provider_code") == "PERMISSION_DENIED"


def test_provider_and_provider_code_present_deepseek_rule4b():
    """DeepSeek choices[0] content_filter → provider='deepseek', provider_code='content_filter'."""
    body = {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
    result = classify_payload(_err(raw=body))
    assert result is not None
    assert result.get("provider") == "deepseek"
    assert result.get("provider_code") == "content_filter"


def test_provider_and_provider_code_present_openrouter_rule5():
    """Top-level finish_reason=content_filter → provider='openrouter', provider_code='content_filter'."""
    result = classify_payload(_err(raw={"finish_reason": "content_filter"}))
    assert result is not None
    assert result.get("provider") == "openrouter"
    assert result.get("provider_code") == "content_filter"


def test_provider_and_provider_code_present_anthropic_rule6():
    """Anthropic content-filter 400 → provider='anthropic', provider_code='content_filter'."""
    body = {"error": {"type": "invalid_request_error",
                      "message": "Output blocked by content filtering policy"}}
    result = classify_payload(_err(raw=body))
    assert result is not None
    assert result.get("provider") == "anthropic"
    assert result.get("provider_code") == "content_filter"


def test_provider_and_provider_code_present_openai_rule7():
    """OpenAI error.code content_policy_violation → provider='openai', provider_code='content_policy_violation'."""
    result = classify_payload(_err(raw={"error": {"code": "content_policy_violation"}}))
    assert result is not None
    assert result.get("provider") == "openai"
    assert result.get("provider_code") == "content_policy_violation"


def test_provider_and_provider_code_present_qwen_rule8():
    """Qwen code data_inspection_failed → provider='qwen', provider_code='data_inspection_failed'."""
    result = classify_payload(_err(raw={"error": {"code": "data_inspection_failed"}}))
    assert result is not None
    assert result.get("provider") == "qwen"
    assert result.get("provider_code") == "data_inspection_failed"


def test_provider_and_provider_code_present_openai_responses_rule9():
    """OpenAI Responses refusal item → provider='openai', provider_code='refusal'."""
    body = {"status": "incomplete",
            "output": [{"type": "refusal", "refusal": "I cannot help."}]}
    result = classify_payload(_err(raw=body))
    assert result is not None
    assert result.get("provider") == "openai"
    assert result.get("provider_code") == "refusal"


def test_provider_and_provider_code_present_gemini_finishreason_rule10():
    """Gemini candidates finishReason SAFETY → provider='google', provider_code='SAFETY'."""
    result = classify_payload(_err(raw={"candidates": [{"finishReason": "SAFETY"}]}))
    assert result is not None
    assert result.get("provider") == "google"
    assert result.get("provider_code") == "SAFETY"


def test_provider_and_provider_code_present_gemini_native_finish_rule10():
    """native_finish_reason SAFETY (OpenRouter-proxied Gemini) → provider='google', provider_code='SAFETY'."""
    result = classify_payload(_err(raw={"native_finish_reason": "SAFETY"}))
    assert result is not None
    assert result.get("provider") == "google"
    assert result.get("provider_code") == "SAFETY"


def test_provider_and_provider_code_present_gemini_prompt_feedback_rule11():
    """Gemini promptFeedback SAFETY → provider='google', provider_code='SAFETY'."""
    result = classify_payload(_err(raw={"promptFeedback": {"blockReason": "SAFETY"}}))
    assert result is not None
    assert result.get("provider") == "google"
    assert result.get("provider_code") == "SAFETY"


def test_real_openai_sdk_permission_denied_error_body_reaches_signals_table():
    """Real openai.PermissionDeniedError.body dict is classified via signals table.

    Validates the .body contract at type level: the real SDK exception carries
    the guardrail body in exc.body (not exc.raw_response), and _extract_body
    must find it via the .body attribute path.
    """
    import json
    import httpx
    from openai import PermissionDeniedError

    guardrail_body = {"error": {"metadata": {
        "error_type": "permission_denied",
        "reasons": ["guardrail"],
        "flagged_input": "test prompt",
    }}}
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    mock_response = httpx.Response(403,
                                   content=json.dumps(guardrail_body).encode(),
                                   request=request)
    exc = PermissionDeniedError(message="403 Forbidden", response=mock_response,
                                 body=guardrail_body)

    # Verify the type-level contract: .body is set, .raw_response is not
    assert isinstance(exc.body, dict), "openai SDK exc must carry body as dict"
    assert not hasattr(exc, "raw_response"), "openai SDK exc must NOT have raw_response"

    result = classify_payload(exc)
    assert result is not None, "Real openai.PermissionDeniedError.body must be classified"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "guardrail_permission_denied"
    assert result["retry_policy"]["kind"] == "terminal"
