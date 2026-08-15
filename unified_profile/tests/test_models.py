from unified_profile.models import canonicalize_model_id


def test_canonicalize_historical_claude_ids():
    assert canonicalize_model_id("anthropic/claude-opus-4-6") == "anthropic/claude-opus-4.6"
    assert canonicalize_model_id("anthropic/claude-sonnet-4-6") == "anthropic/claude-sonnet-4.6"


def test_canonicalize_aita_aliases_and_labels():
    assert canonicalize_model_id("opus-4-6") == "anthropic/claude-opus-4.6"
    assert canonicalize_model_id("Opus 4.6") == "anthropic/claude-opus-4.6"
    assert canonicalize_model_id("gemini-flash") == "google/gemini-3-flash-preview"
    assert canonicalize_model_id("GPT-5.5") == "openai/gpt-5.5"
