"""Provider-neutral model id helpers for a reference OpenAI-compatible adapter."""

from __future__ import annotations


def resolve_upstream_model(requested_model: str | None, default_model: str) -> str:
    """Return the model id sent upstream.

    Public benchmark users can either expose a fixed adapter model id or forward
    the benchmark's requested model id through to an OpenAI-compatible upstream.
    """
    if default_model:
        return default_model
    return (requested_model or "").strip() or "local/example-model"


def list_adapter_model_ids(exposed_model_id: str) -> list[str]:
    """Return model ids this reference adapter advertises."""
    return [(exposed_model_id or "local/example-model").strip()]
