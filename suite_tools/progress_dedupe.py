"""Shared unit-identity key and countable sets for progress deduplication.

Progress counters count DISTINCT units in completed/reused/terminal states
across ALL attempts.  A unit that appears in attempt 1 and again in attempt 2
(as a reuse event) counts once.  The canonical identity is ``unit_id`` when
present; otherwise a fallback tuple of (model, scenario, …) fields is used so
that legacy streams without ``unit_id`` keep working.

Cost paths and evidence paths remain UNFILTERED — only the progress-counting
chokepoints (scheduler and dashboard) consume this module.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Event-name sets
# ---------------------------------------------------------------------------

COMPLETED_EVENTS: frozenset[str] = frozenset(
    {
        "conversation_completed",
        "run_completed",
        "sus_run_completed",
    }
)

REUSED_EVENTS: frozenset[str] = frozenset(
    {
        "conversation_reused",
        "sus_run_reused",
        "conversation_reused_provider_refusal",
        "conversation_reused_output_budget_exhausted",
    }
)

TERMINAL_SIGNAL_EVENTS: frozenset[str] = frozenset(
    {
        "block_recorded",
        "conversation_output_budget_exhausted",
    }
)

SCORING_COMPLETED_EVENTS: frozenset[str] = frozenset(
    {
        "score_saved",
        "score_reused",
        "result_saved",
    }
)

# Union used by completed_unit_keys
_COUNTABLE_EVENTS: frozenset[str] = COMPLETED_EVENTS | REUSED_EVENTS | TERMINAL_SIGNAL_EVENTS

# All event names that progress tracking cares about (superset for filter lists)
ALL_PROGRESS_EVENTS: frozenset[str] = (
    _COUNTABLE_EVENTS | SCORING_COMPLETED_EVENTS
)

# Event-name sets for active-unit counting.
ACTIVE_STARTED_EVENTS: frozenset[str] = frozenset(
    {"conversation_started", "paid_call_started", "run_started"}
)
ACTIVE_FINISHED_EVENTS: frozenset[str] = frozenset(
    {
        "conversation_completed",
        "conversation_failed",
        "conversation_incomplete",
        "paid_call_completed",
        "score_saved",
        "score_failed",
        "run_completed",
    }
)

# ---------------------------------------------------------------------------
# Unit-identity key
# ---------------------------------------------------------------------------


def event_unit_key(event: dict[str, Any]) -> tuple[Any, ...]:
    """Return a hashable identity key for *event*.

    Prefers the canonical ``unit_id`` field when present; otherwise falls back
    to a tuple of (model, scenario, test_type, item_idx, run_number, side,
    role) so that legacy streams without ``unit_id`` continue to work.
    """
    if event.get("unit_id"):
        return ("unit", event["unit_id"])
    return (
        event.get("model_id") or event.get("model") or event.get("model_key"),
        event.get("scenario"),
        event.get("test_type"),
        event.get("item_idx"),
        event.get("run_number"),
        event.get("side"),
        event.get("role"),
    )


# Fallback key emitted when there is no unit_id and no other identifying field.
# All-None means we cannot deduplicate: each such event must remain unique.
_ALL_NONE_KEY: tuple[None, ...] = (None, None, None, None, None, None, None)

# ---------------------------------------------------------------------------
# Countable-set reduction
# ---------------------------------------------------------------------------


def completed_unit_keys(events: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    """Return the set of distinct unit keys that have reached a terminal state.

    Counts events in ``COMPLETED_EVENTS | REUSED_EVENTS | TERMINAL_SIGNAL_EVENTS``.
    Environment-failure events (``conversation_failed``, etc.) are excluded.
    The set deduplicates the same unit across multiple attempts by canonical
    ``unit_id`` when present.

    For legacy events with no ``unit_id`` and no fallback identifying fields
    (model/scenario/item_idx/…), each event is treated as a distinct unit so
    that streams without any identity information still count correctly.
    """
    keys: set[tuple[Any, ...]] = set()
    _seq = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("event") or "")
        if name in _COUNTABLE_EVENTS:
            key = event_unit_key(event)
            if key == _ALL_NONE_KEY:
                # No identifying information — use a unique per-event key so
                # legacy streams are not collapsed.
                key = ("_seq", _seq)
                _seq += 1
            keys.add(key)
    return keys


def active_unit_count(events: list[dict[str, Any]]) -> int:
    """Count units that have started but not yet reached a finished state.

    For events with ``unit_id``: set-based dedup — a retried unit that emits
    a second ``conversation_started`` is NOT double-counted as two active units.

    For events without ``unit_id``: count-based fallback (started - finished),
    preserving legacy behaviour for streams that lack identity fields.
    """
    started_known: set[str] = set()
    finished_known: set[str] = set()
    started_count = 0
    finished_count = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("event") or "")
        unit_id = event.get("unit_id")
        if unit_id:
            if name in ACTIVE_STARTED_EVENTS:
                started_known.add(unit_id)
            elif name in ACTIVE_FINISHED_EVENTS:
                finished_known.add(unit_id)
        else:
            if name in ACTIVE_STARTED_EVENTS:
                started_count += 1
            elif name in ACTIVE_FINISHED_EVENTS:
                finished_count += 1

    active_known = max(0, len(started_known - finished_known))
    active_unknown = max(0, started_count - finished_count)
    return active_known + active_unknown
