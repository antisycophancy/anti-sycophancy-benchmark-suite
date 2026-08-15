"""Guardrails for model-visible benchmark payloads.

These checks are intentionally about what the model-under-test can see at
inference time. Result artifacts may contain benchmark metadata, but the
messages sent to the target model should look like an ordinary conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


MODEL_VISIBLE_LEAK_RE = re.compile(
    r"\b("
    r"benchmark|harness|judge|judging|score|scoring|scored|phase|"
    r"dataset|artifact|transcript artifact|runner|evaluation|eval|"
    r"experiment condition|control group|treatment group|red team"
    r")\b",
    re.IGNORECASE,
)

CONTROL_ROLES = {"system", "developer", "tool"}


@dataclass(frozen=True)
class PayloadLeak:
    """A benchmark/evaluation term visible to the model-under-test."""

    index: int
    role: str
    term: str
    excerpt: str


def _excerpt(text: Any, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def scan_model_visible_payload(
    messages: Iterable[dict[str, Any]],
    *,
    roles: set[str] | None = None,
) -> list[PayloadLeak]:
    """Return model-visible benchmark-context leaks in control messages.

    By default this scans only non-user control roles, because user benchmark
    prompts can legitimately contain words like "test" or "score." Pass a
    wider role set when auditing already-generated assistant history.
    """
    checked_roles = CONTROL_ROLES if roles is None else roles
    leaks: list[PayloadLeak] = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "")
        if role not in checked_roles:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for match in MODEL_VISIBLE_LEAK_RE.finditer(content):
            leaks.append(
                PayloadLeak(
                    index=index,
                    role=role,
                    term=match.group(1),
                    excerpt=_excerpt(content),
                )
            )
    return leaks


def assert_blind_model_payload(messages: Iterable[dict[str, Any]]) -> None:
    """Raise AssertionError if control messages reveal benchmark context."""
    leaks = scan_model_visible_payload(messages)
    if not leaks:
        return
    details = "; ".join(
        f"{leak.role}[{leak.index}] contains `{leak.term}`: {leak.excerpt}"
        for leak in leaks
    )
    raise AssertionError(f"Model-visible payload leaks benchmark context: {details}")
