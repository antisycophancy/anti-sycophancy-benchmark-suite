import pytest

from suite_tools.artifact_privacy import assert_public_artifact_safe, scan_public_artifact_payload


def test_public_artifact_privacy_rejects_secret_like_values():
    payload = {
        "viewers": [
            {
                "items": [
                    {
                        "records": [
                            {
                                "messages": [
                                    {
                                        "role": "assistant",
                                        "content": "Authorization: Bearer " + ("x" * 32),
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    issues = scan_public_artifact_payload(payload)

    assert issues
    with pytest.raises(ValueError, match="Public artifact privacy check failed"):
        assert_public_artifact_safe(payload)


def test_public_artifact_privacy_rejects_fake_google_key():
    fake_google_key = "AIza" + ("x" * 35)
    payload = {"metadata": {"provider_error": f"bad key {fake_google_key}"}}

    issues = scan_public_artifact_payload(payload)

    assert issues
    assert issues[0].reason == "secret-looking value"


def test_public_artifact_privacy_rejects_private_network_hosts():
    payload = {"endpoint": "http://192.168.1.10:9999/v1"}

    issues = scan_public_artifact_payload(payload)

    assert issues
    assert issues[0].reason == "private/internal marker"


def test_public_artifact_privacy_rejects_private_prompt_fields():
    payload = {"raw_prompt": "unreleased scenario text"}

    with pytest.raises(ValueError, match="raw_prompt"):
        assert_public_artifact_safe(payload)


def test_absolute_home_path_caught_when_real_local_path():
    # A real operator run path must be flagged.
    for value in (
        "/Users/me/runs/pilot/RUN_CONTRACT.json",
        "path=/home/runner/results/x.json",
        "/Users/alice/secret/run",
    ):
        issues = scan_public_artifact_payload({"p": value})
        assert any(i.reason == "absolute home path" for i in issues), value


def test_absolute_home_path_does_not_false_positive_on_url_segments():
    # Lowercase URL path segments and non-boundary mentions must pass (case-sensitive,
    # boundary-anchored regex).
    for value in (
        "https://api.example.test/users/profile",
        "https://cdn.example.test/home/banner.png",
        "see docs at example.test/Users/guide",   # /Users not at a delimiter boundary
    ):
        issues = scan_public_artifact_payload({"u": value})
        assert not any(i.reason == "absolute home path" for i in issues), value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_public_artifact_privacy_rejects_nonfinite_numbers(value):
    issues = scan_public_artifact_payload({"score": value})

    assert any(issue.reason == "non-finite numeric value" for issue in issues)
    with pytest.raises(ValueError, match="non-finite numeric value"):
        assert_public_artifact_safe({"score": value})
