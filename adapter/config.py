"""Provider-neutral reference adapter configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ADAPTER_HOST = os.environ.get("ADAPTER_HOST", "127.0.0.1").strip()
ADAPTER_DEBUG_UPSTREAM_ERRORS = (
    os.environ.get("ADAPTER_DEBUG_UPSTREAM_ERRORS", "").lower() in {"1", "true", "yes", "on"}
)
ADAPTER_INBOUND_API_KEY = os.environ.get("ADAPTER_INBOUND_API_KEY", "").strip()


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


ADAPTER_MAX_REQUEST_BYTES = _positive_int_env(
    "ADAPTER_MAX_REQUEST_BYTES",
    1024 * 1024,
)
ADAPTER_PORT = int(os.environ.get("ADAPTER_PORT", "9999"))
EXPOSED_MODEL_ID = (
    os.environ.get("EXPOSED_MODEL_ID", "local/example-model").strip()
    or "local/example-model"
)
# Blank means pass the benchmark's requested model through unchanged. Set a
# value when the public adapter id differs from the private upstream id.
UPSTREAM_MODEL_ID = os.environ.get("UPSTREAM_MODEL_ID", "").strip()
UPSTREAM_OPENAI_BASE_URL = os.environ.get("UPSTREAM_OPENAI_BASE_URL", "").rstrip("/")
UPSTREAM_CHAT_COMPLETIONS_URL = os.environ.get("UPSTREAM_CHAT_COMPLETIONS_URL", "").strip()
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")
UPSTREAM_API_KEY_ENV = os.environ.get("UPSTREAM_API_KEY_ENV", "")
REFERENCE_RESPONSE = os.environ.get(
    "REFERENCE_RESPONSE",
    "Reference OpenAI-compatible adapter response. Configure UPSTREAM_OPENAI_BASE_URL "
    "or point the benchmark directly at your model endpoint for real runs.",
)
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120"))
_diagnostics_value = os.environ.get(
    "ADAPTER_DIAGNOSTICS_PATH",
    str(Path(__file__).parent / "results" / "adapter-diagnostics.jsonl"),
).strip()
if _diagnostics_value:
    _diagnostics_path = Path(_diagnostics_value)
    ADAPTER_DIAGNOSTICS_PATH = (
        _diagnostics_path
        if _diagnostics_path.is_absolute()
        else Path(__file__).parent / _diagnostics_path
    )
else:
    ADAPTER_DIAGNOSTICS_PATH = None
ADAPTER_DIAGNOSTICS_INCLUDE_DETAIL = (
    os.environ.get("ADAPTER_DIAGNOSTICS_INCLUDE_DETAIL", "").lower()
    in {"1", "true", "yes", "on"}
)
try:
    ADAPTER_DIAGNOSTICS_MAX_BYTES = int(
        os.environ.get("ADAPTER_DIAGNOSTICS_MAX_BYTES", str(16 * 1024 * 1024))
    )
except ValueError:
    ADAPTER_DIAGNOSTICS_MAX_BYTES = 16 * 1024 * 1024


def upstream_api_key() -> str:
    if UPSTREAM_API_KEY_ENV:
        return os.environ.get(UPSTREAM_API_KEY_ENV, "")
    return UPSTREAM_API_KEY


def upstream_chat_completions_url() -> str:
    if UPSTREAM_CHAT_COMPLETIONS_URL:
        return UPSTREAM_CHAT_COMPLETIONS_URL
    if UPSTREAM_OPENAI_BASE_URL:
        return f"{UPSTREAM_OPENAI_BASE_URL}/chat/completions"
    return ""
