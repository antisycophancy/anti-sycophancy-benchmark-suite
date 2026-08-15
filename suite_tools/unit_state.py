"""Per-module unit completeness and terminal-state predicates.

These pure functions are the single source of truth for deciding whether a
recorded conversation/result counts as completed, terminal, or still owed.
Tasks 5, 6, and 7 wire these predicates into runners and owed_units logic.
"""

UNIT_STATE_VERSION = "unit-state-v1"


# ---------------------------------------------------------------------------
# Shared terminal-signal helpers
# ---------------------------------------------------------------------------

def is_terminal_model_signal(state: str) -> bool:
    """Return True iff *state* represents a terminal model-signal outcome."""
    return state == "terminal_model_signal"


def terminal_reuse_event_name(conv: dict) -> str:
    """Return the event name to emit when reusing a terminal conversation."""
    if conv.get("output_budget_exhausted") is True:
        return "conversation_reused_output_budget_exhausted"
    return "conversation_reused_provider_refusal"


# ---------------------------------------------------------------------------
# Internal helpers (shared between AITA and EPIS)
# ---------------------------------------------------------------------------

def _is_provider_refusal(conv: dict) -> bool:
    if conv.get("provider_refusal") is True:
        return True
    reason = str(conv.get("failure_reason") or "").lower()
    return "provider refusal" in reason or "stop_reason=refusal" in reason


def _is_output_budget_exhausted(conv: dict) -> bool:
    if conv.get("output_budget_exhausted") is True:
        return True
    reason = str(conv.get("failure_reason") or "").lower()
    return "output budget exhausted" in reason


def _is_aita_epis_terminal(conv: dict) -> bool:
    return _is_provider_refusal(conv) or _is_output_budget_exhausted(conv)


# ---------------------------------------------------------------------------
# AITA
# ---------------------------------------------------------------------------

def aita_unit_state(conv: dict, planned_turns: int) -> str:
    """Return 'completed' | 'terminal_model_signal' | 'owed' for one AITA conv.

    Terminal is checked first: a provider refusal or budget-exhaustion outcome
    is terminal even when the turn count is below the planned threshold.
    """
    if _is_aita_epis_terminal(conv):
        return "terminal_model_signal"
    if len(conv.get("turns", [])) >= planned_turns:
        return "completed"
    return "owed"


# ---------------------------------------------------------------------------
# EPIS
# ---------------------------------------------------------------------------

def epis_unit_state(conv: dict, planned_turns: int) -> str:
    """Return 'completed' | 'terminal_model_signal' | 'owed' for one EPIS conv.

    The terminal predicates mirror the EPIS runner's
    ``_is_provider_refusal_conversation`` / ``_is_output_budget_exhausted_conversation``
    checks, minus the ``completed is not False`` guard (that guard is runner-
    internal bookkeeping; the unit-state predicate operates on stored records
    that may not set ``completed``).
    """
    if _is_aita_epis_terminal(conv):
        return "terminal_model_signal"
    if len(conv.get("turns", [])) >= planned_turns:
        return "completed"
    return "owed"


# ---------------------------------------------------------------------------
# SUS
# ---------------------------------------------------------------------------

_SUS_TERMINAL_SCORE_STATES = {
    "excluded_provider_refusal",
    "excluded_output_budget_exhausted",
}


def sus_unit_state(result: dict, planned_escalations: int) -> str:
    """Return 'completed' | 'terminal_model_signal' | 'owed' for one SUS result.

    *result* is a persisted transcript artifact dict (the object written by
    ``_write_live_transcript_artifact``).

    Terminal is checked first.  Completed requires:
      - ``"elicit"`` key present in ``phases``
      - count of ``escalate_*`` keys in ``phases`` >= ``planned_escalations``
    """
    score_state = result.get("score_state")
    if score_state in _SUS_TERMINAL_SCORE_STATES:
        return "terminal_model_signal"

    phases = result.get("phases") or {}
    if "elicit" not in phases:
        return "owed"
    escalate_count = sum(1 for k in phases if k.startswith("escalate_"))
    if escalate_count >= planned_escalations:
        return "completed"
    return "owed"
