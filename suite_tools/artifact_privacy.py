"""Privacy checks for public benchmark artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


SECRET_VALUE_RE = re.compile(
    r"(?i)("
    r"sk-[a-z0-9_-]{20,}|"
    r"sk-or-v1-[a-z0-9_-]{20,}|"
    r"AIza[0-9a-z_-]{30,}|"
    r"authorization:\s*bearer\s+[a-z0-9._-]{20,}|"
    r"api[_-]?key\s*[:=]\s*['\"]?[a-z0-9._-]{20,}"
    r")"
)
PRIVATE_HOST_RE = re.compile(
    r"(?i)(?:https?://|//|\b)"
    r"(?:"
    r"localhost|"
    r"127\.0\.0\.1|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"[a-z0-9.-]+\.internal"
    r")"
    r"(?::\d+)?(?:[/?#]\S*)?\b"
)
# Absolute home-directory paths (Sol finding 5). A public artifact must never
# carry an operator's local filesystem layout — e.g. ``str(contract_path.resolve())``
# leaking ``/Users/<name>/...`` even after members are reduced to bundle-local ids.
# Case-sensitive and path-boundary anchored so lowercase URL segments like
# ``https://api.example.test/users/profile`` do NOT false-positive: the match must
# start at a delimiter (line start, whitespace, quote, ``=``/``:``/``(``/``[``) and
# carry a trailing ``/`` after the first path segment. Defense in depth: flagged
# wherever a payload string matches, aborting a bundle before it can be written.
ABSOLUTE_HOME_PATH_RE = re.compile(r"(?:^|[\s\"'=:(\[])/(?:Users|home)/[A-Za-z0-9_.-]+/")
PRIVATE_TEXT_MARKERS = (
    "private_question_bank",
    "/internal/",
    "internal/",
    "authorization:",
)
PRIVATE_FIELD_NAMES = {
    "api_key",
    "authorization",
    "developer_prompt",
    "headers",
    "jwt",
    "password",
    "private_prompt",
    # Raw provider-response bodies can echo prompt/model content verbatim.  They
    # are LOCAL-only (plan 020 D2); only the raw_body_sha256 digest is public.
    # Forbidding the field name here means audit_bundle_tree catches such a body
    # ANYWHERE in public JSON — including nested inside another allowlisted
    # object — as defense in depth behind the bundle's field allowlist.
    "raw_body",
    "raw_body_excerpt",
    "raw_prompt",
    "raw_response",
    "request_headers",
    "response_body",
    "secret",
    "source_prompt",
    "system_prompt",
    "token",
}


@dataclass(frozen=True)
class ArtifactPrivacyIssue:
    """One public artifact privacy finding."""

    path: str
    reason: str


def _field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def scan_public_artifact_payload(payload: Any) -> list[ArtifactPrivacyIssue]:
    """Return obvious private/secrets issues in a public artifact payload."""
    issues: list[ArtifactPrivacyIssue] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if _field_name(key) in PRIVATE_FIELD_NAMES:
                    issues.append(ArtifactPrivacyIssue(child_path, "private field name"))
                walk(item, child_path)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(value, float) and not math.isfinite(value):
            issues.append(ArtifactPrivacyIssue(path, "non-finite numeric value"))
            return
        if not isinstance(value, str):
            return

        lowered = value.lower()
        if SECRET_VALUE_RE.search(value):
            issues.append(ArtifactPrivacyIssue(path, "secret-looking value"))
        elif any(marker in lowered for marker in PRIVATE_TEXT_MARKERS) or PRIVATE_HOST_RE.search(value):
            issues.append(ArtifactPrivacyIssue(path, "private/internal marker"))
        if ABSOLUTE_HOME_PATH_RE.search(value):
            issues.append(ArtifactPrivacyIssue(path, "absolute home path"))

    walk(payload, "")
    return issues


def assert_public_artifact_safe(payload: Any) -> None:
    """Raise if a payload contains obvious private/secrets material."""
    issues = scan_public_artifact_payload(payload)
    if issues:
        sample = "; ".join(f"{issue.path}: {issue.reason}" for issue in issues[:5])
        raise ValueError(f"Public artifact privacy check failed: {sample}")


def assert_text_public_safe(text: str) -> None:
    """Gate serialized non-JSON text BEFORE it is written to a public artifact.

    Checks the raw string against the three primary privacy regexes — secret
    values, private/internal hosts, and absolute home-directory paths.  Raises
    ``ValueError`` if any pattern matches so a caller can abort before the byte
    hits disk.  Use this for CSV, Markdown, HTML, or any non-JSON serialization.
    """
    reasons: list[str] = []
    if SECRET_VALUE_RE.search(text):
        reasons.append("secret-looking value")
    if PRIVATE_HOST_RE.search(text):
        reasons.append("private/internal host marker")
    if ABSOLUTE_HOME_PATH_RE.search(text):
        reasons.append("absolute home path")
    if reasons:
        raise ValueError(
            f"Public artifact privacy check failed: {'; '.join(reasons)}"
        )
