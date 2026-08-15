"""Shared content-block policy executor for all benchmark runners (T3/D8/D9).

Consults the signals table (classify_payload) on raw provider responses and
drives the retry_policy bound on behalf of any runner's empty-content or
explicit-block path.

CONTRACT (pinned by T2 tests — do NOT violate):
  classify_payload is called on the RAW response body BEFORE any
  ProviderRefusalError is constructed.  A constructed ProviderRefusalError
  run through classify_evidence/action_policy_for yields terminal/(0)
  regardless of body contents — do not call those on a post-construction
  exception.  T3 consumers must call consult_content_block on the raw dict
  first; only then (after the bound is exhausted) construct and raise.
"""

from __future__ import annotations

from typing import Any

from suite_tools.provider_signals import classify_payload


def consult_content_block(raw_response: dict[str, Any]) -> dict[str, Any] | None:
    """Classify a raw provider response dict against the signals table.

    Returns the full evidence record (including ``retry_policy``) or ``None``
    if no rule matched at all.  ``None`` means the caller should fall through
    to its existing unexplained-empty / generic retry behavior.

    Must be called BEFORE any ProviderRefusalError is constructed (D9 contract).
    Every returned dict is a copy-safe result from classify_payload; callers may
    read ``evidence["retry_policy"]`` directly.
    """
    # classify_payload extracts the body via _extract_body which checks
    # .raw_response first, so wrap the dict in a thin holder.
    class _RawHolder:
        pass

    h = _RawHolder()
    h.raw_response = raw_response  # type: ignore[attr-defined]
    return classify_payload(h)


class ContentBlockPolicyExecutor:
    """Per-api-call content-block attempt counter (plan 020 D8).

    Owns the signal-driven retry count independently of the runner's transient
    error retry counter (``attempt``) and the stochastic budget counter
    (``budget_attempts``).  Create one instance per ``api_call`` / per
    ``call_provider`` invocation; it is single-threaded (one counter per
    in-flight call).

    Usage pattern::

        executor = ContentBlockPolicyExecutor()
        while ...:
            response = call_provider(...)
            raw = build_raw_dict(response)
            ev = consult_content_block(raw)
            if ev is not None:
                if executor.decide(ev) == "continue":
                    continue            # loop back for another paid attempt
                raise ProviderRefusalError(...)   # bound exhausted → terminalize
            # no signal → existing retry / return path unchanged
    """

    def __init__(self) -> None:
        self._signal_attempts: int = 0  # retry slots consumed so far

    @property
    def signal_attempts(self) -> int:
        """Number of signal-driven retry attempts consumed (excludes first call)."""
        return self._signal_attempts

    def decide(self, evidence: dict[str, Any]) -> str:
        """Return ``"continue"`` or ``"terminalize"`` for this attempt.

        ``"continue"`` means the bound is not yet exhausted; the caller should
        loop for another paid attempt without constructing ProviderRefusalError.

        ``"terminalize"`` means the bound is exhausted (or the policy is
        terminal); the caller should construct and raise ProviderRefusalError
        with the structured raw_response.

        The decision is driven by ``evidence["retry_policy"]``  (embedded by
        classify_payload, plan 020 D8).  When retry_policy is absent or kind is
        ``"terminal"``, the answer is always ``"terminalize"``.
        """
        retry_policy = evidence.get("retry_policy")
        if not isinstance(retry_policy, dict):
            return "terminalize"

        kind = retry_policy.get("kind", "terminal")
        max_retries = int(retry_policy.get("max_retries", 0))

        if kind in ("bounded_retry", "stochastic_retry") and self._signal_attempts < max_retries:
            self._signal_attempts += 1
            return "continue"

        return "terminalize"

    def billed_attempt_count(self) -> int:
        """Total paid attempts so far, including the current terminalizing one.

        Equal to ``signal_attempts + 1`` (the +1 is the first/last call that
        terminalized).  Useful for ``billed_attempts`` snapshot fields.
        """
        return self._signal_attempts + 1
