import pytest

from suite_tools.credential_policy import (
    CredentialDestinationError,
    destination_policy_error,
    require_credential_destination,
)


@pytest.mark.parametrize(
    ("env_name", "url"),
    [
        ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ("OPENAI_API_KEY", "https://api.openai.com/v1/responses"),
        ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/messages"),
        ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta"),
    ],
)
def test_official_credentials_accept_only_canonical_https_origin(env_name, url):
    require_credential_destination(env_name, url)


@pytest.mark.parametrize(
    "url",
    [
        "https://openrouter.ai.attacker.example/v1",
        "https://attacker.example/path/openrouter.ai/v1",
        "http://openrouter.ai/api/v1",
        "https://openrouter.ai:8443/api/v1",
        "https://user@openrouter.ai/api/v1",
    ],
)
def test_official_credential_rejects_deceptive_or_noncanonical_destination(url):
    with pytest.raises(CredentialDestinationError):
        require_credential_destination("OPENROUTER_API_KEY", url)


def test_custom_credential_defaults_to_loopback_and_remote_requires_allowlist(monkeypatch):
    monkeypatch.delenv("BENCHMARK_ALLOWED_ENDPOINT_HOSTS", raising=False)
    require_credential_destination("MY_MODEL_KEY", "http://127.0.0.1:9000/v1")
    assert destination_policy_error("MY_MODEL_KEY", "https://models.example/v1")

    monkeypatch.setenv("BENCHMARK_ALLOWED_ENDPOINT_HOSTS", "models.example")
    require_credential_destination("MY_MODEL_KEY", "https://models.example/v1")

