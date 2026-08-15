"""Credential-to-destination policy for provider calls.

Model configs are data, not authority to send an environment secret anywhere.
Official provider variables are pinned to their canonical HTTPS origin. Custom
variables default to loopback and require a separate operator allowlist for a
remote HTTPS host.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit


OFFICIAL_KEY_HOSTS = {
    "OPENROUTER_API_KEY": frozenset({"openrouter.ai"}),
    "OPENAI_API_KEY": frozenset({"api.openai.com"}),
    "ANTHROPIC_API_KEY": frozenset({"api.anthropic.com"}),
    "GEMINI_API_KEY": frozenset({"generativelanguage.googleapis.com"}),
    "GOOGLE_API_KEY": frozenset({"generativelanguage.googleapis.com"}),
}
ALLOWED_ENDPOINT_HOSTS_ENV = "BENCHMARK_ALLOWED_ENDPOINT_HOSTS"


class CredentialDestinationError(ValueError):
    """An environment credential is not authorized for the configured URL."""


def normalized_url_host(url: str | None) -> tuple[str, str, int | None] | None:
    """Return ``(scheme, hostname, port)`` for an HTTP(S) URL without userinfo."""
    try:
        parsed = urlsplit(str(url or ""))
        port = parsed.port
    except ValueError:
        return None
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return parsed.scheme.lower(), hostname, port


def is_loopback_host(hostname: str) -> bool:
    hostname = hostname.lower().rstrip(".")
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def configured_allowed_endpoint_hosts(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    source = os.environ if environ is None else environ
    raw = str(source.get(ALLOWED_ENDPOINT_HOSTS_ENV, "") or "")
    return tuple(
        sorted({part.lower().rstrip(".") for part in re.split(r"[\s,]+", raw) if part})
    )


def destination_policy_error(
    api_key_env: str,
    base_url: str,
    allowed_endpoint_hosts: Iterable[str] = (),
) -> str | None:
    parsed = normalized_url_host(base_url)
    if parsed is None:
        return "provider endpoint must be an HTTP(S) URL without userinfo"
    scheme, hostname, port = parsed

    official_hosts = OFFICIAL_KEY_HOSTS.get(str(api_key_env))
    if official_hosts is not None:
        if scheme != "https" or port not in {None, 443} or hostname not in official_hosts:
            expected = ", ".join(sorted(official_hosts))
            return (
                f"${api_key_env} is bound to canonical HTTPS host(s) {expected}; "
                f"refusing {base_url}"
            )
        return None

    if is_loopback_host(hostname):
        return None

    allowed = {
        str(host).lower().rstrip(".")
        for host in (*configured_allowed_endpoint_hosts(), *allowed_endpoint_hosts)
    }
    if scheme == "https" and port in {None, 443} and hostname in allowed:
        return None
    return (
        f"${api_key_env} may reach only loopback by default; explicitly trust "
        f"remote HTTPS host {hostname!r} with ${ALLOWED_ENDPOINT_HOSTS_ENV}"
    )


def require_credential_destination(
    api_key_env: str,
    base_url: str,
    *,
    allowed_endpoint_hosts: Iterable[str] = (),
) -> None:
    error = destination_policy_error(api_key_env, base_url, allowed_endpoint_hosts)
    if error:
        raise CredentialDestinationError(error)

