"""Model ID canonicalization for cross-module benchmark joins."""

from __future__ import annotations

MODEL_ALIASES = {
    # Historical local/OpenRouter spellings observed in this repo.
    "anthropic/claude-opus-4-6": "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    # AITA aliases / labels.
    "opus-4-6": "anthropic/claude-opus-4.6",
    "claude-opus-4-6": "anthropic/claude-opus-4.6",
    "Opus 4.6": "anthropic/claude-opus-4.6",
    "opus-4-7": "anthropic/claude-opus-4.7",
    "claude-opus-4-7": "anthropic/claude-opus-4.7",
    "Opus 4.7": "anthropic/claude-opus-4.7",
    "sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "gpt-5-4": "openai/gpt-5.4",
    "GPT-5.4": "openai/gpt-5.4",
    "gpt-5-5": "openai/gpt-5.5",
    "GPT-5.5": "openai/gpt-5.5",
    "gemini-flash": "google/gemini-3-flash-preview",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "Gemini 3 Flash": "google/gemini-3-flash-preview",
    "gemini-3-1-pro": "google/gemini-3.1-pro-preview",
    "Gemini 3.1 Pro": "google/gemini-3.1-pro-preview",
    "Therapeutic Harness: Claude Opus 4.6": "therapeutic-harness/th-opus-4-6",
}

AITA_MODEL_MAP = MODEL_ALIASES


def canonicalize_model_id(value: str) -> str:
    """Return the canonical model ID used for unified profile joins."""
    return MODEL_ALIASES.get(value, value)


def model_label(model_id: str, fallback: str | None = None) -> str:
    """Return a compact display label for a canonical model ID."""
    labels = {
        "anthropic/claude-opus-4.6": "Opus 4.6",
        "anthropic/claude-opus-4.7": "Opus 4.7",
        "anthropic/claude-sonnet-4.6": "Sonnet 4.6",
        "openai/gpt-5.4": "GPT-5.4",
        "openai/gpt-5.5": "GPT-5.5",
        "google/gemini-3-flash-preview": "Gemini 3 Flash",
        "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
        "therapeutic-harness/th-opus-4-6": "Therapeutic Harness: Claude Opus 4.6",
    }
    return labels.get(model_id, fallback or model_id)
