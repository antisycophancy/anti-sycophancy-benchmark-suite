import urllib.request
from datetime import datetime

import pytest

from suite_tools.openrouter_preflight import (
    _NoRedirectHandler,
    collect_openrouter_refs,
    sanitize_key_info,
    validate_openrouter_url,
    validate_openrouter_catalog,
)


def test_openrouter_preflight_disables_redirects_to_protect_authorization_header():
    handler = _NoRedirectHandler()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": "Bearer secret"},
    )

    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://attacker.example/capture",
    )

    assert redirected is None


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://openrouter.ai/api/v1/models",
        "https://openrouter.ai.evil.example/api/v1/models",
        "https://user:secret@openrouter.ai/api/v1/models",
        "https://openrouter.ai:8443/api/v1/models",
    ],
)
def test_openrouter_preflight_rejects_unsafe_or_noncanonical_urls(url):
    with pytest.raises(ValueError, match="OpenRouter preflight URL"):
        validate_openrouter_url(url)


def test_openrouter_preflight_accepts_canonical_https_urls():
    url = "https://openrouter.ai/api/v1/models"
    assert validate_openrouter_url(url) == url


def _config():
    return {
        "schema_version": 1,
        "defaults": {"endpoint": "openrouter", "max_parallel": 2},
        "endpoints": {
            "openrouter": {
                "openai_base_url": "https://openrouter.ai/api/v1",
                "chat_completions_url": "https://openrouter.ai/api/v1/chat/completions",
                "api_key_env": "OPENROUTER_API_KEY",
            },
            "local_openai_compatible": {
                "openai_base_url": "http://localhost:9999/v1",
                "chat_completions_url": "http://localhost:9999/v1/chat/completions",
                "api_key_env": "LOCAL_OPENAI_COMPATIBLE_API_KEY",
            },
        },
        "agents": {"seeker": {"model_id": "google/gemini-3-flash-preview"}},
        "judge_sets": {
            "calibration": {
                "primary": "google/gemini-3.1-pro-preview",
                "panel": ["google/gemini-3.1-pro-preview"],
            }
        },
        "model_groups": {"smoke": ["raw", "local"]},
        "models": {
            "raw": {"model_id": "openai/gpt-5.5"},
            "local": {
                "model_id": "local/example-model",
                "endpoint": "local_openai_compatible",
            },
        },
    }


def _catalog():
    return {
        "data": [
            {
                "id": "openai/gpt-5.5",
                "name": "GPT-5.5",
                "pricing": {"prompt": "0.000001", "completion": "0.00001"},
            },
            {
                "id": "google/gemini-3-flash-preview",
                "name": "Gemini 3 Flash",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000003"},
            },
            {
                "id": "google/gemini-3.1-pro-preview",
                "canonical_slug": "google/gemini-3.1-pro-preview",
                "name": "Gemini 3.1 Pro",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
            {
                "id": "anthropic/claude-haiku-4.5",
                "name": "Claude Haiku 4.5",
                "pricing": {"prompt": "0.000001", "completion": "0.000005"},
            },
        ]
    }


def test_collect_openrouter_refs_skips_non_openrouter_models():
    refs, skipped = collect_openrouter_refs(_config())

    assert "openai/gpt-5.5" in refs
    assert "google/gemini-3-flash-preview" in refs
    assert "local/example-model" not in refs
    assert skipped == ["models.local: local/example-model (local_openai_compatible)"]


def test_collect_openrouter_refs_resolves_direct_judge_aliases():
    config = _config()
    config["endpoints"]["anthropic_native"] = {
        "messages_url": "https://api.anthropic.com/v1/messages",
        "api_key_env": "ANTHROPIC_API_KEY",
    }
    config["judge_models"] = {
        "judge-opus-native-high": {
            "model_id": "claude-opus-4-7",
            "endpoint": "anthropic_native",
        },
        "judge-flash-via-openrouter": {
            "model_id": "google/gemini-3-flash-preview",
            "endpoint": "openrouter",
        },
    }
    config["judge_sets"]["direct_high"] = {
        "primary": "judge-opus-native-high",
        "panel": ["judge-opus-native-high", "judge-flash-via-openrouter"],
    }

    refs, skipped = collect_openrouter_refs(config)

    # Direct-provider judge aliases must be skipped, not flagged missing.
    assert "judge-opus-native-high" not in refs
    assert any("judge-opus-native-high -> claude-opus-4-7 (anthropic_native)" in s for s in skipped)
    # OpenRouter-routed judge aliases resolve to their model_id for validation.
    assert "google/gemini-3-flash-preview" in refs


def test_collect_openrouter_refs_includes_agent_profiles():
    config = _config()
    config["agent_profiles"] = {
        "haiku_45": {
            "description": "candidate",
            "agents": {
                "seeker": {"model_id": "anthropic/claude-haiku-4.5"},
            },
        },
    }

    refs, _skipped = collect_openrouter_refs(config)

    assert refs["anthropic/claude-haiku-4.5"] == {
        "agent_profiles.haiku_45.agents.seeker"
    }


def test_collect_openrouter_refs_includes_module_agents():
    config = _config()
    config["module_agents"] = {
        "aita": {
            "seeker": {"model_id": "anthropic/claude-haiku-4.5"},
        },
    }

    refs, _skipped = collect_openrouter_refs(config)

    assert refs["anthropic/claude-haiku-4.5"] == {"module_agents.aita.seeker"}


def test_validate_openrouter_catalog_reports_ok_with_prices():
    report = validate_openrouter_catalog(_config(), _catalog())

    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["schema_version"] == "benchmark-pricing-snapshot-v1"
    assert report["units"] == "per_token"
    assert report["provider"] == "openrouter"
    assert datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00")).tzinfo is not None
    assert len(report["checked"]) == 3
    assert report["skipped"] == ["models.local: local/example-model (local_openai_compatible)"]


def test_validate_openrouter_catalog_warns_on_missing_pricing_without_failing():
    catalog = _catalog()
    catalog["data"][0]["pricing"] = {"request": "0"}

    report = validate_openrouter_catalog(_config(), catalog)

    assert report["errors"] == []
    assert "No OpenRouter prompt/completion pricing" in report["warnings"][0]


def test_validate_openrouter_catalog_strict_pricing_fails_on_missing_pricing():
    catalog = _catalog()
    catalog["data"][0]["pricing"] = {"request": "0"}

    report = validate_openrouter_catalog(_config(), catalog, strict_pricing=True)

    assert "No OpenRouter prompt/completion pricing" in report["errors"][0]


@pytest.mark.parametrize("bad_price", ["-1", "NaN", "Infinity", "-Infinity"])
def test_validate_openrouter_catalog_rejects_negative_or_nonfinite_pricing(bad_price):
    catalog = _catalog()
    catalog["data"][0]["pricing"]["prompt"] = bad_price

    with pytest.raises(ValueError, match="non-negative finite"):
        validate_openrouter_catalog(_config(), catalog)


@pytest.mark.parametrize("bad_price", ["-1", "NaN", "Infinity"])
def test_price_overrides_reject_negative_or_nonfinite_values(bad_price):
    from suite_tools.openrouter_preflight import parse_price_overrides

    with pytest.raises(ValueError, match="non-negative finite"):
        parse_price_overrides([f"model/example:{bad_price}:0.1"])


def test_validate_openrouter_catalog_reports_missing_model_slug():
    catalog = _catalog()
    catalog["data"] = [item for item in catalog["data"] if item["id"] != "openai/gpt-5.5"]

    report = validate_openrouter_catalog(_config(), catalog)

    assert report["errors"] == ["Missing OpenRouter model `openai/gpt-5.5` used by models.raw"]


def test_sanitize_key_info_redacts_label_and_keeps_usage_fields():
    info = sanitize_key_info(
        {
            "data": {
                "label": "my secret-ish key label",
                "limit": 500,
                "limit_reset": None,
                "limit_remaining": 4.5,
                "include_byok_in_limit": False,
                "usage": 451.1,
                "usage_daily": 3.0,
                "usage_weekly": 7.0,
                "usage_monthly": 210.0,
                "byok_usage": 1.0,
                "byok_usage_daily": 0.1,
                "byok_usage_weekly": 0.2,
                "byok_usage_monthly": 0.3,
                "is_free_tier": False,
                "rate_limit": {"requests": -1, "interval": "10s"},
            }
        }
    )

    assert info["available"] is True
    assert info["label_present"] is True
    assert "label" not in info
    assert info["limit"] == 500
    assert info["limit_remaining"] == 4.5
    assert info["credit_limit_low"] is True
    assert info["credit_limit_exhausted"] is False
    assert info["byok_usage_monthly"] == 0.3
    assert info["rate_limit_deprecated_present"] is True
