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


def test_openrouter_content_policy_is_model_signal():
    e = _err(raw={"error": {"metadata": {"error_type": "content_policy_violation"}}})
    result = classify_payload(e)
    # v2: exact match includes retry_policy; check key fields + shape
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_policy_violation"
    assert result["signal_source"] == SIGNALS_TABLE_VERSION
    assert "retry_policy" in result  # v2 addition


def test_openrouter_guardrail_permission_denied_is_model_signal():
    e = _err(raw={"error": {"metadata": {"error_type": "permission_denied", "reasons": ["guardrail"]}}})
    result = classify_payload(e)
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "guardrail_permission_denied"


def test_google_dunning_permission_denied_is_environment_billing():
    e = _err(raw={"error": {"code": 403, "status": "PERMISSION_DENIED",
                            "message": "Lightning dunning decision is deny for project: projects/955415727012"}})
    result = classify_payload(e)
    assert result["evidence_class"] == "environment"
    assert result["category"] == "billing"
    assert result["signal_source"] == SIGNALS_TABLE_VERSION
    assert "retry_policy" in result  # v2 addition


def test_bare_google_permission_denied_is_explicit_unknown():
    e = _err(raw={"error": {"code": 403, "status": "PERMISSION_DENIED",
                            "message": "caller does not have permission"}})
    result = classify_payload(e)
    assert result["evidence_class"] == "unknown"
    assert result["category"] == "ambiguous_permission_denied"
    assert result["signal_source"] == SIGNALS_TABLE_VERSION
    assert "retry_policy" in result  # v2 addition


def test_finish_reason_content_filter_is_model_signal():
    assert classify_payload(_err(raw={"finish_reason": "content_filter"}))["category"] == "content_filter"


def test_gemini_safety_finish_reason_is_stochastic_model_signal():
    result = classify_payload(_err(raw={"candidates": [{"finishReason": "SAFETY"}]}))
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "SAFETY"
    assert result["stochastic"] is True


def test_openai_and_qwen_error_codes():
    assert classify_payload(_err(raw={"error": {"code": "invalid_prompt"}}))["category"] == "invalid_prompt"
    assert classify_payload(_err(raw={"error": {"code": "data_inspection_failed"}}))["category"] == "data_inspection_failed"


def test_json_text_fallback_when_no_raw_response():
    e = _err(text='{"error": {"metadata": {"error_type": "refusal"}}}')
    assert classify_payload(e)["category"] == "refusal"


def test_unrecognized_body_returns_none():
    assert classify_payload(_err(raw={"error": {"message": "boom"}})) is None
