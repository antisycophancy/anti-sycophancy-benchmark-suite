"""Vendor/family-name scrubbing for judge-visible model text.

Blinding has two layers. The first removes the *registered* identifiers for the
model under test (its registry slug and label) — see ``model_blind_patterns`` /
``_blind_text`` in each module's scoring code. That layer is exact and cheap,
but it only knows the strings the harness configured.

It is defeated by prose self-identification: a model that writes "I'm Claude,
made by Anthropic" hands the judge its vendor even though no registered slug
appeared. This module is the second layer — a fixed list of vendor and model-
family names, matched case-insensitively **on word boundaries only**.

Two deliberate constraints:

* **Word boundaries are load-bearing.** Substring matching mangles the evidence
  the judge is supposed to grade ("metaphor" → "MODELphor"). Every term here
  must survive appearing inside a longer ordinary word.
* **Apply to model-authored text only.** Scenario and user text is the judge's
  evidence; a poster who says "my husband works at Google" must keep saying it.

This is best-effort, not a guarantee. A model that describes itself without
naming its vendor ("I was trained by a San Francisco lab") still reaches the
judge unblinded. Treat it as raising the cost of leakage, not eliminating it.
"""

from __future__ import annotations

import re

BLIND_TOKEN = "MODEL"

# Vendor and model-family names, longest first so multi-word phrases win.
# Standalone terms that are also common English words ("moonshot", "opus",
# "sonnet", "flash", "pro") are deliberately EXCLUDED and covered only by their
# multi-word form, because scrubbing them would damage ordinary advisor prose.
VENDOR_IDENTITY_TERMS: tuple[str, ...] = (
    "google deepmind",
    "moonshot ai",
    "meta ai",
    "anthropic",
    "chatgpt",
    "deepmind",
    "deepseek",
    "microsoft",
    "alibaba",
    "mistral",
    "mixtral",
    "copilot",
    "gemini",
    "claude",
    "openai",
    "google",
    "tongyi",
    "llama",
    "grok",
    "qwen",
    "bard",
    "meta",
    "kimi",
    "xai",
    "gpt",
)

# Optional trailing version token so "GPT-5.5" and "Llama 3.1" are removed
# whole rather than leaving a bare "-5.5" behind.
_VERSION_SUFFIX = r"(?:[-\s]?\d+(?:\.\d+)*)?"

_VENDOR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in VENDOR_IDENTITY_TERMS) + r")"
    + _VERSION_SUFFIX
    + r"\b",
    re.IGNORECASE,
)


def blind_vendor_identity(text: str | None) -> str:
    """Replace vendor/family self-identification in model-authored text.

    Matches on word boundaries, case-insensitively. Ordinary words that merely
    contain a vendor name as a substring are left intact.
    """
    if text is None:
        return ""
    return _VENDOR_RE.sub(BLIND_TOKEN, str(text))
