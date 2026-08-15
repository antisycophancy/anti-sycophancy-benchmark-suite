"""Review model + effective-evidence projection (plan 020, Task T5, D4 + D5).

This is the ONE place where human judgment (``BLOCK_REVIEWS.jsonl`` dispositions)
joins recorded facts (``BLOCKS.jsonl`` blocks ∪ ``attempt_failure_classified``
events) into a single *effective* state that ``owed_units``, ``score_rows`` and
``bench blockers`` consume.  Every clause below encodes a Sol review round; they
are implemented verbatim from the plan.

Two views (``project(run_dir) -> ProjectionResult``):

* ``events_by_ref`` — every fact keyed by its D4 ``event_ref``, carrying the
  active review (head of the supersession chain), the effective class/category,
  the scope (unit / member / unmappable-legacy) and a resolution status.
* ``units_by_id`` — the attempt-aware per-unit reduction consumers need:
  latest-attempt carrier, strictly-later-attempt completion, retry obligations,
  v1 artifact compat, same-attempt conflict integrity, disposition semantics.

D4 fact identity:
  * v2 facts carry immutable ids → ``blocks-id:<block_id>`` / ``events-id:<event_id>``.
  * legacy facts use hash-anchored physical line refs →
    ``blocks-line:<n>:<sha8>`` / ``events-line:<n>:<sha8>`` where ``n`` is the
    1-based physical line index and ``sha8`` is the first 8 hex of the sha256 of
    the raw line bytes.  A hash mismatch on resolve is a hard error naming the
    drift.

Review records (``benchmark-block-review-v2``, same ``BLOCK_REVIEWS.jsonl``
file): appended atomically under an ``O_EXCL`` lock held across validate-then-
append so two concurrent invocations cannot create two active heads.  v1
backfill reviews (``benchmark-block-review-v1``, no ``event_ref``) are
grandfathered through the composite ``(module, model, unit_id, category,
backfill_id)`` to exactly one physical backfilled block.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BLOCKS_FILENAME = "BLOCKS.jsonl"
RUN_EVENTS_FILENAME = "RUN_EVENTS.jsonl"
BLOCK_REVIEWS_FILENAME = "BLOCK_REVIEWS.jsonl"
BLOCK_REVIEWS_LOCK_FILENAME = "BLOCK_REVIEWS.lock"
CONTRACT_FILENAME = "RUN_CONTRACT.json"
RUN_STATUS_FILENAME = "RUN_STATUS.json"

REVIEW_SCHEMA_VERSION = "benchmark-block-review-v2"
REVIEW_SCHEMA_VERSION_V1 = "benchmark-block-review-v1"

FACT_EVENT_NAME = "attempt_failure_classified"

VALID_DISPOSITIONS = frozenset(
    {"safety_declination", "retry", "instrument_defect", "needs_escalation"}
)
# Dispositions that adjudicate a fact as no-longer-an-open-blocker.
_RESOLVING_DISPOSITIONS = frozenset({"safety_declination", "instrument_defect"})
# Underlying categories/classes that force ``safety_declination`` to supply a
# ``resolved_category`` (plan D5; brief disposition semantics).
_UNCLASSIFIED_CATEGORIES = frozenset({"unclassified", "ambiguous_403", None, ""})
_UNKNOWN_CLASSES = frozenset({"unknown"})

_KNOWN_MODULES = frozenset({"aita", "epis", "sus"})

_DEFAULT_LOCK_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EventRefError(ValueError):
    """Malformed or unresolvable ``event_ref``."""


class EventRefDriftError(EventRefError):
    """A hash-anchored line ref no longer matches the physical line (drift)."""


class ReviewValidationError(ValueError):
    """A review record failed validation (bad disposition, missing resolved
    category, unmappable retry target, ambiguous backfill resolution, ...)."""


class DuplicateActiveReviewError(ReviewValidationError):
    """A second active review for a fact was appended without ``supersede``."""


class ReviewLockError(RuntimeError):
    """The ``BLOCK_REVIEWS.lock`` could not be acquired before the deadline."""


# ---------------------------------------------------------------------------
# D4 — event_ref parse / format / resolve
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventRef:
    source: str          # "blocks" | "events"
    kind: str            # "id" | "line"
    ident: str | None = None
    line: int | None = None
    sha8: str | None = None

    def canonical(self) -> str:
        if self.kind == "id":
            return f"{self.source}-id:{self.ident}"
        return f"{self.source}-line:{self.line}:{self.sha8}"


@dataclass
class ResolvedFact:
    source: str
    index: int           # 1-based physical line
    record: dict[str, Any]
    event_ref: str


def line_sha8(raw_line: bytes) -> str:
    """First 8 hex chars of the sha256 of the raw physical line bytes (D4)."""
    if isinstance(raw_line, str):
        raw_line = raw_line.encode("utf-8")
    return hashlib.sha256(raw_line).hexdigest()[:8]


def parse_event_ref(ref: str) -> EventRef:
    """Parse a D4 ``event_ref`` string; raise :class:`EventRefError` on garbage."""
    if not isinstance(ref, str) or ":" not in ref:
        raise EventRefError(f"malformed event_ref: {ref!r}")
    prefix, rest = ref.split(":", 1)
    if prefix in ("blocks-id", "events-id"):
        source = prefix.split("-", 1)[0]
        if not rest:
            raise EventRefError(f"empty id in event_ref: {ref!r}")
        return EventRef(source=source, kind="id", ident=rest)
    if prefix in ("blocks-line", "events-line"):
        source = prefix.split("-", 1)[0]
        parts = rest.split(":")
        if len(parts) != 2:
            raise EventRefError(f"malformed line event_ref: {ref!r}")
        line_str, sha8 = parts
        try:
            line = int(line_str)
        except ValueError as exc:
            raise EventRefError(f"non-integer line in event_ref: {ref!r}") from exc
        if not sha8:
            raise EventRefError(f"empty sha8 in event_ref: {ref!r}")
        return EventRef(source=source, kind="line", line=line, sha8=sha8)
    raise EventRefError(f"unknown event_ref prefix: {ref!r}")


def _source_filename(source: str) -> str:
    return BLOCKS_FILENAME if source == "blocks" else RUN_EVENTS_FILENAME


def _physical_lines(path: Path) -> list[str]:
    """Return the physical lines of *path* (without trailing newlines)."""
    if not path.exists():
        return []
    return path.read_text().splitlines()


def _iter_source_records(run_dir: Path, source: str):
    """Yield ``(line_index_1based, raw_line, record_or_None)`` for a fact source."""
    path = run_dir / _source_filename(source)
    for idx, raw in enumerate(_physical_lines(path), start=1):
        stripped = raw.strip()
        if not stripped:
            yield idx, raw, None
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            yield idx, raw, None
            continue
        if not isinstance(record, dict):
            yield idx, raw, None
            continue
        yield idx, raw, record


def _record_is_fact(source: str, record: dict[str, Any]) -> bool:
    if source == "blocks":
        return True
    return record.get("event") == FACT_EVENT_NAME


def _record_id_field(source: str) -> str:
    return "block_id" if source == "blocks" else "event_id"


def resolve_event_ref(run_dir: Path | str, ref: str | EventRef) -> ResolvedFact:
    """Resolve an ``event_ref`` to its physical fact record in *run_dir*.

    id-form refs scan the source file for the matching ``block_id``/``event_id``.
    line-form refs read the 1-based physical line and verify the sha8 hash; a
    mismatch raises :class:`EventRefDriftError` naming the drift.
    """
    run_dir = Path(run_dir)
    parsed = ref if isinstance(ref, EventRef) else parse_event_ref(ref)
    id_field = _record_id_field(parsed.source)
    filename = _source_filename(parsed.source)

    if parsed.kind == "id":
        for idx, _raw, record in _iter_source_records(run_dir, parsed.source):
            if record is None:
                continue
            if not _record_is_fact(parsed.source, record):
                continue
            if str(record.get(id_field)) == parsed.ident:
                return ResolvedFact(parsed.source, idx, record,
                                    parsed.canonical())
        raise EventRefError(
            f"event_ref {parsed.canonical()} not found in {filename}"
        )

    # line-form: read the physical line and verify the hash.
    lines = _physical_lines(run_dir / filename)
    if parsed.line is None or parsed.line < 1 or parsed.line > len(lines):
        raise EventRefError(
            f"event_ref line {parsed.line} out of range for {filename} "
            f"({len(lines)} lines)"
        )
    raw = lines[parsed.line - 1]
    actual = line_sha8(raw.encode("utf-8"))
    if actual != parsed.sha8:
        raise EventRefDriftError(
            f"event_ref drift: {filename} line {parsed.line} hash {actual} "
            f"!= expected {parsed.sha8} — the ledger line changed under the "
            f"review pointer; re-review against the current fact."
        )
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EventRefError(
            f"event_ref {filename} line {parsed.line} is not valid JSON"
        ) from exc
    return ResolvedFact(parsed.source, parsed.line, record, parsed.canonical())


# ---------------------------------------------------------------------------
# Fact model
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    source: str                 # "blocks" | "events"
    line: int
    record: dict[str, Any]
    event_ref: str
    unit_id: str | None
    scope: str                  # "unit" | "member" | "unmappable_legacy"
    attempt_number: int
    evidence_class: str | None
    category: str | None
    module: str | None
    model: str | None
    # True iff the record carried an explicit ``attempt_number`` (a v2 fact).
    # v1/legacy facts are attempt-less (attempt 0) and keep the completed-
    # artifact-wins compat behaviour; v2 facts require strict attempt ordering.
    has_attempt: bool = False


def _fact_event_ref(source: str, line: int, raw: str, record: dict[str, Any]) -> str:
    id_field = _record_id_field(source)
    ident = record.get(id_field)
    if ident:
        return f"{source}-id:{ident}"
    return f"{source}-line:{line}:{line_sha8(raw.encode('utf-8'))}"


def _classify_scope(source: str, unit_id: str | None) -> str:
    if unit_id:
        return "unit"
    # A block is inherently a unit-level denial; a missing unit_id cannot be
    # mapped → unmappable-legacy.  A halt event without a unit_id is a run-level
    # (member-scoped) fact (e.g. current aita halt records).
    return "unmappable_legacy" if source == "blocks" else "member"


def _load_facts(run_dir: Path) -> list[Fact]:
    facts: list[Fact] = []
    for source in ("blocks", "events"):
        for idx, raw, record in _iter_source_records(run_dir, source):
            if record is None or not _record_is_fact(source, record):
                continue
            unit_id = record.get("unit_id")
            unit_id = str(unit_id) if unit_id else None
            raw_attempt = record.get("attempt_number")
            has_attempt = raw_attempt is not None
            try:
                attempt = int(raw_attempt)
            except (TypeError, ValueError):
                attempt = 0
                has_attempt = False
            facts.append(Fact(
                source=source,
                line=idx,
                record=record,
                event_ref=_fact_event_ref(source, idx, raw, record),
                unit_id=unit_id,
                scope=_classify_scope(source, unit_id),
                attempt_number=attempt,
                evidence_class=record.get("evidence_class"),
                category=record.get("category"),
                module=record.get("module"),
                model=record.get("model"),
                has_attempt=has_attempt,
            ))
    return facts


# ---------------------------------------------------------------------------
# Reviews — load, backfill grandfather, supersession, active head
# ---------------------------------------------------------------------------

def load_reviews(run_dir: Path | str) -> list[dict[str, Any]]:
    """Return every review record (v1 + v2) from ``BLOCK_REVIEWS.jsonl``."""
    run_dir = Path(run_dir)
    path = run_dir / BLOCK_REVIEWS_FILENAME
    reviews: list[dict[str, Any]] = []
    if not path.exists():
        return reviews
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            reviews.append(record)
    return reviews


def _load_reviews_strict(run_dir: Path) -> list[dict[str, Any]]:
    """Like :func:`load_reviews` but fail-closed: a non-blank line that is not a
    JSON object is a hard error (final-gate F4b) rather than a silent skip, so a
    truncated/partial review write cannot make a resolved fact look unresolved
    (or vice-versa).  Used by :func:`project`; the lenient loader stays for the
    display-only ``bench review`` list path."""
    path = run_dir / BLOCK_REVIEWS_FILENAME
    reviews: list[dict[str, Any]] = []
    if not path.exists():
        return reviews
    for idx, raw in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ReviewValidationError(
                f"malformed review line {idx} in {BLOCK_REVIEWS_FILENAME}: "
                f"not valid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ReviewValidationError(
                f"malformed review line {idx} in {BLOCK_REVIEWS_FILENAME}: "
                f"not a JSON object"
            )
        reviews.append(record)
    return reviews


def _is_v1_backfill_review(review: dict[str, Any]) -> bool:
    return (
        review.get("schema_version") == REVIEW_SCHEMA_VERSION_V1
        or (review.get("backfill_id") is not None and not review.get("event_ref"))
    )


def _backfill_block_composite(fact: Fact) -> tuple | None:
    """The (module, model, unit_id, category, backfill_id) composite of a
    physical *backfilled* block fact, else None if it is not backfilled."""
    record = fact.record
    if not record.get("backfilled") or not record.get("backfill_id"):
        return None
    return (
        record.get("module"), record.get("model"), record.get("unit_id"),
        record.get("category"), record.get("backfill_id"),
    )


def _resolve_backfill_review(review: dict[str, Any], facts: list[Fact]) -> str:
    """Resolve a v1 backfill review to the ONE physical backfilled block's
    ``event_ref`` via the composite.  Ambiguity is a hard error listing
    candidates; no match is a hard error too (plan D4)."""
    composite = (
        review.get("module"), review.get("model"), review.get("unit_id"),
        review.get("category"), review.get("backfill_id"),
    )
    candidates = [
        f for f in facts
        if f.source == "blocks" and _backfill_block_composite(f) == composite
    ]
    if len(candidates) == 1:
        return candidates[0].event_ref
    if not candidates:
        raise ReviewValidationError(
            f"backfill review composite {composite} matched no physical "
            f"backfilled block"
        )
    raise ReviewValidationError(
        f"backfill review composite {composite} is ambiguous — matched "
        f"{len(candidates)} backfilled blocks at lines "
        f"{[c.line for c in candidates]}; refuse to guess"
    )


def _fact_for_ref_in(facts: list[Fact], ref: str) -> Fact | None:
    """Match an event_ref (id- or line-form) to a loaded fact, else None."""
    try:
        parsed = parse_event_ref(ref)
    except EventRefError:
        return None
    id_field = _record_id_field(parsed.source)
    if parsed.kind == "id":
        return next(
            (f for f in facts
             if f.source == parsed.source
             and str(f.record.get(id_field)) == parsed.ident),
            None,
        )
    return next(
        (f for f in facts if f.source == parsed.source and f.line == parsed.line),
        None,
    )


def _canonical_review_ref(review: dict[str, Any], facts: list[Fact]) -> str | None:
    """Return the canonical target ``event_ref`` (the matched fact's own
    ``event_ref``, i.e. id-form when the fact carries a v2 id) for *review*.

    v1 backfill reviews resolve through the composite; v2 reviews normalize
    their stored ref to the fact's canonical ref so a line-form review always
    groups with an id-form fact (and vice-versa).  Returns None when the ref
    resolves to no physical fact.
    """
    if _is_v1_backfill_review(review):
        try:
            return _resolve_backfill_review(review, facts)
        except ReviewValidationError:
            return None
    raw = review.get("event_ref")
    if not raw:
        return None
    fact = _fact_for_ref_in(facts, raw)
    return fact.event_ref if fact is not None else raw


def _validate_v2_record(review: dict[str, Any]) -> None:
    """Reject a malformed/incomplete v2 review (final-gate F4b): a record that
    cannot be safely resolved must be an integrity error, never silently used."""
    disposition = review.get("disposition")
    if disposition not in VALID_DISPOSITIONS:
        raise ReviewValidationError(
            f"v2 review has invalid disposition {disposition!r}; expected one of "
            f"{sorted(VALID_DISPOSITIONS)}"
        )
    if not review.get("review_id"):
        raise ReviewValidationError(
            "malformed v2 review: missing review_id (cannot participate in a "
            "supersession chain)"
        )
    if not review.get("event_ref"):
        raise ReviewValidationError("malformed v2 review: missing event_ref")


def _active_review_for_ref(reviews: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the single active head of the supersession chain for one ref.

    The head is the v2 review no other v2 review supersedes.  A dangling
    supersession target, a supersession cycle, or multiple surviving heads is an
    integrity error — never a silent timestamp/file-order pick and never a
    silently-accepted phantom resolver (final-gate F4b + F4-round2).  With no v2
    records, the last v1 backfill record is the head.
    """
    if not reviews:
        return None
    v2 = [r for r in reviews if r.get("review_id")]
    if not v2:
        return reviews[-1]  # v1 backfill only

    by_id = {r["review_id"]: r for r in v2}

    # Fail-closed on the read path (final-gate2 F4): every supersedes_review_id
    # must resolve to a review PRESENT in this same canonical-ref chain.  A
    # dangling target (nonexistent id, or one grouped under a different ref) was
    # previously accepted at LOAD time — a hand-edited/corrupted ledger could
    # make a phantom-superseding record the active resolver.  Now it is an
    # integrity error, the same family as the cycle/multiple-head errors.
    for start in v2:
        seen = {start["review_id"]}
        sid = start.get("supersedes_review_id")
        while sid:
            if sid in seen:
                raise ReviewValidationError(
                    f"supersession cycle detected among reviews for "
                    f"{start.get('event_ref')} (review {start.get('review_id')})"
                )
            nxt = by_id.get(sid)
            if nxt is None:
                raise ReviewValidationError(
                    f"dangling supersession target {sid!r} for review "
                    f"{start.get('review_id')!r} on {start.get('event_ref')}: no "
                    f"such review in this event_ref's chain — refuse to accept a "
                    f"phantom superseding record as the resolver"
                )
            seen.add(sid)
            sid = nxt.get("supersedes_review_id")

    superseded_ids = {
        r["supersedes_review_id"] for r in v2 if r.get("supersedes_review_id")
    }
    heads = [r for r in v2 if r["review_id"] not in superseded_ids]
    if len(heads) > 1:
        raise ReviewValidationError(
            f"multiple active review heads for one event_ref "
            f"({sorted(str(h.get('review_id')) for h in heads)}); refuse to "
            f"resolve by timestamp — supersede the extras or remove them"
        )
    if not heads:
        raise ReviewValidationError(
            "no active review head (all v2 reviews superseded — supersession cycle)"
        )
    return heads[0]


def _reviews_by_ref(reviews: list[dict[str, Any]], facts: list[Fact]) -> dict[str, list[dict]]:
    """Group reviews under their canonical target ``event_ref`` (grandfathering
    v1 via composite; validating + canonicalizing v2 records)."""
    # Ledger-wide integrity (final-gate2 F4): a duplicate v2 review_id is
    # ambiguous — a supersedes_review_id could name either copy, so which is the
    # head is undecidable.  Reject the whole ledger rather than pick one; this is
    # the same integrity family as the dangling/cycle/multiple-head errors and
    # the bundle gate fails closed on it (never a silent drop).
    seen_ids: set[str] = set()
    for review in reviews:
        rid = review.get("review_id")
        if rid:
            if rid in seen_ids:
                raise ReviewValidationError(
                    f"duplicate review_id {rid!r} in {BLOCK_REVIEWS_FILENAME}; "
                    f"review ids must be unique across the ledger"
                )
            seen_ids.add(rid)

    grouped: dict[str, list[dict]] = {}
    for review in reviews:
        if _is_v1_backfill_review(review):
            ref = _resolve_backfill_review(review, facts)
        else:
            _validate_v2_record(review)
            raw = review.get("event_ref")
            fact = _fact_for_ref_in(facts, raw)
            ref = fact.event_ref if fact is not None else raw
        grouped.setdefault(ref, []).append(review)
    return grouped


# ---------------------------------------------------------------------------
# Effective class / category / resolution status
# ---------------------------------------------------------------------------

def _effective_class(fact: Fact, review: dict[str, Any] | None) -> str | None:
    if review is None:
        return fact.evidence_class
    disposition = review.get("disposition")
    if disposition == "safety_declination":
        return "model_signal"
    if disposition == "instrument_defect":
        return "instrument_defect"
    # retry / needs_escalation do not reclassify the underlying evidence.
    return fact.evidence_class


def _effective_category(fact: Fact, review: dict[str, Any] | None) -> str | None:
    if review is None:
        return fact.category
    if review.get("disposition") == "safety_declination":
        resolved = review.get("resolved_category")
        if resolved:
            return resolved
    return fact.category


def _resolution_status(fact: Fact, review: dict[str, Any] | None) -> str:
    if review is not None:
        disposition = review.get("disposition")
        if disposition == "needs_escalation":
            return "unresolved"
        if disposition == "retry":
            return "pending_retry"
        if disposition in _RESOLVING_DISPOSITIONS:
            return "resolved"
    # No active review: an unknown-class fact is unresolved (needs review); any
    # other known class is settled.
    if fact.evidence_class in _UNKNOWN_CLASSES:
        return "unresolved"
    return "resolved"


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@dataclass
class FactView:
    event_ref: str
    source: str
    scope: str
    unit_id: str | None
    member_id: str
    fact: dict[str, Any]
    active_review: dict[str, Any] | None
    disposition: str | None
    effective_class: str | None
    effective_category: str | None
    resolution_status: str
    attempt_number: int
    line: int = 0
    has_attempt: bool = False


@dataclass
class UnitView:
    unit_id: str
    module: str | None
    scope: str
    member_id: str
    state: str                      # completed|terminal_model_signal|owed|pending_retry|instrument_defect|unresolved
    reason: str
    artifact: Path | None
    carrier: FactView | None
    disposition: str | None
    effective_category: str | None
    integrity_error: bool = False


@dataclass
class MemberObligation:
    member_id: str
    event_ref: str
    kind: str                       # "retry"
    fulfilled: bool
    fact: dict[str, Any]


@dataclass
class ProjectionResult:
    member_id: str
    events_by_ref: dict[str, FactView] = field(default_factory=dict)
    units_by_id: dict[str, UnitView] = field(default_factory=dict)
    member_obligations: list[MemberObligation] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Unit reduction helpers
# ---------------------------------------------------------------------------

_STATE_COMPLETED = "completed"
_STATE_TERMINAL = "terminal_model_signal"
_STATE_OWED = "owed"
_STATE_PENDING_RETRY = "pending_retry"
_STATE_INSTRUMENT = "instrument_defect"
_STATE_UNRESOLVED = "unresolved"


def _select_carrier(fact_views: list[FactView]) -> tuple[FactView | None, bool, str]:
    """Return ``(carrier, integrity_error, reason)`` for a unit's facts.

    Facts sort by attempt_number; the carrier is a fact at the maximum attempt.
    Multiple max-attempt facts that share (evidence_class, category) are
    compatible duplicates → BLOCKS-over-events, insertion-order independent.
    Conflicting max-attempt facts are an integrity error (review queue).
    """
    if not fact_views:
        return None, False, ""
    max_attempt = max(fv.attempt_number for fv in fact_views)
    top = [fv for fv in fact_views if fv.attempt_number == max_attempt]
    signatures = {(fv.fact.get("evidence_class"), fv.fact.get("category")) for fv in top}
    if len(signatures) > 1:
        detail = ", ".join(
            f"{fv.source}#{fv.fact.get('evidence_class')}/{fv.fact.get('category')}"
            for fv in sorted(top, key=lambda f: (f.source, f.line))
        )
        return (
            sorted(top, key=lambda f: (f.source != "blocks", f.line))[0],
            True,
            f"same-attempt terminal conflict at attempt {max_attempt}: {detail}",
        )
    blocks_top = sorted((fv for fv in top if fv.source == "blocks"), key=lambda f: f.line)
    if blocks_top:
        return blocks_top[0], False, ""
    return sorted(top, key=lambda f: f.line)[0], False, ""


def _read_artifact(artifact: Path | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    try:
        data = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _dispatch_state(module: str | None, unit: dict[str, Any], data: dict[str, Any]) -> str | None:
    """Run the module unit_state predicate; None for unknown modules (never
    raise here — owed_units owns the unknown-module error surface)."""
    if module not in _KNOWN_MODULES:
        return None
    from suite_tools.owed_units import _dispatch_unit_state  # noqa: PLC0415
    return _dispatch_unit_state(module, unit, data)


def _artifact_attempt(data: dict[str, Any]) -> int:
    try:
        return int(data.get("attempt_number") or 0)
    except (TypeError, ValueError):
        return 0


def _carrier_discharged(
    *,
    carrier: FactView | None,
    disposition: str | None,
    artifact: Path | None,
    module: str | None,
    unit: dict[str, Any] | None,
) -> bool:
    """Return True iff a completed artifact discharges *carrier* (plan D5, F3).

    Strict attempt ordering applies when the carrier carries an explicit
    attempt_number (a V2 fact) or is under a ``retry`` review: the completed
    artifact must be stamped with a STRICTLY LATER attempt.  Attempt-less (v1)
    carriers with no retry review keep the legacy completed-artifact-wins compat.
    """
    if carrier is None:
        return False
    data = _read_artifact(artifact)
    if data is None or _dispatch_state(module, unit or {}, data) != "completed":
        return False
    strict = disposition == "retry" or carrier.has_attempt
    if strict:
        return _artifact_attempt(data) > carrier.attempt_number
    return True


def _reduce_unit(
    *,
    unit_id: str,
    unit: dict[str, Any] | None,
    module: str | None,
    member_id: str,
    run_dir: Path,
    fact_views: list[FactView],
) -> UnitView:
    from suite_tools.owed_units import _resolve_artifact  # noqa: PLC0415

    artifact = _resolve_artifact(run_dir, unit) if unit is not None else None
    carrier, integrity_error, conflict_reason = _select_carrier(fact_views)

    def mk(state, reason, *, disposition=None, effective_category=None, integ=False):
        return UnitView(
            unit_id=unit_id, module=module, scope="unit", member_id=member_id,
            state=state, reason=reason, artifact=artifact, carrier=carrier,
            disposition=disposition,
            effective_category=(
                effective_category if effective_category is not None
                else (carrier.effective_category if carrier else None)
            ),
            integrity_error=integ,
        )

    if integrity_error:
        return mk(_STATE_UNRESOLVED, conflict_reason, integ=True)

    disposition = carrier.disposition if carrier else None

    # Does a completed artifact discharge the carrier?  Strict attempt ordering
    # (plan D5, final-gate F3): a V2 carrier (a fact stamped with attempt_number)
    # OR any retried carrier requires the artifact to come from a STRICTLY LATER
    # attempt — otherwise an older/same-attempt completion could resurrect a
    # declined unit into the scoring denominators.  The attempt-less completed-
    # artifact-wins compat survives ONLY for attempt-less (v1) carriers with no
    # retry review.
    discharged = _carrier_discharged(
        carrier=carrier, disposition=disposition, artifact=artifact,
        module=module, unit=unit,
    )

    # --- retry: strict attempt-aware discharge -----------------------------
    if disposition == "retry":
        if discharged:
            # The later outcome REPLACES the obligation and does NOT inherit the
            # retry review (Sol r2-2).
            return mk(_STATE_COMPLETED, "completed", disposition=None)
        return mk(_STATE_PENDING_RETRY, "pending_retry", disposition="retry")

    # --- carrier is a blocking model_signal BLOCKS fact --------------------
    if (
        carrier is not None
        and carrier.source == "blocks"
        and carrier.effective_class == "model_signal"
    ):
        if discharged:
            return mk(_STATE_COMPLETED, "completed_after_block", disposition=disposition)
        return mk(_STATE_TERMINAL, "BLOCKS.jsonl model_signal entry", disposition=disposition)

    # --- carrier reclassified as instrument_defect -------------------------
    if carrier is not None and carrier.effective_class == "instrument_defect":
        if discharged:
            return mk(_STATE_COMPLETED, "completed_after_block", disposition=disposition)
        return mk(_STATE_INSTRUMENT, "instrument_defect", disposition=disposition)

    # --- carrier is a unit-scoped model_signal EVENT (not a block) ---------
    if (
        carrier is not None
        and carrier.source == "events"
        and carrier.effective_class == "model_signal"
    ):
        if discharged:
            return mk(_STATE_COMPLETED, "completed", disposition=disposition)
        return mk(_STATE_TERMINAL, "RUN_EVENTS.jsonl model_signal halt", disposition=disposition)

    # --- carrier left unresolved by an active review (needs_escalation) ----
    # Gate on an active review: a *bare* unknown-class fact with no review is
    # ignored by the owed view exactly as the pre-projection owed_units ignored
    # non-model_signal blocks (byte-identity).  Its unresolved status still
    # surfaces on events_by_ref for the publication gate (T7).
    if (
        carrier is not None
        and carrier.resolution_status == "unresolved"
        and carrier.active_review is not None
    ):
        return mk(_STATE_UNRESOLVED, "needs_escalation, unresolved", disposition=disposition)

    # --- no blocking carrier: predicate path (mirrors owed_units) ----------
    if artifact is None:
        return mk(_STATE_OWED, "artifact missing")
    data = _read_artifact(artifact)
    if data is None:
        return mk(_STATE_OWED, "artifact unreadable")
    raw_state = _dispatch_state(module, unit or {}, data)
    if raw_state == "completed":
        return mk(_STATE_COMPLETED, "completed")
    if raw_state == "terminal_model_signal":
        return mk(_STATE_TERMINAL, "unit_state terminal")
    return mk(_STATE_OWED, "incomplete")


# ---------------------------------------------------------------------------
# project()
# ---------------------------------------------------------------------------

def _member_id(run_dir: Path) -> str:
    contract_path = run_dir / CONTRACT_FILENAME
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text())
            run_id = contract.get("run_id")
            if run_id:
                return str(run_id)
        except (OSError, json.JSONDecodeError):
            pass
    return run_dir.name


def _load_contract_units(run_dir: Path) -> tuple[dict[str, tuple[dict, str]], list[str]]:
    """Return ``{unit_id: (unit_dict, module)}`` and the ordered unit_id list."""
    contract_path = run_dir / CONTRACT_FILENAME
    if not contract_path.exists():
        return {}, []
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}, []
    from suite_tools.owed_units import _try_normalize_module  # noqa: PLC0415

    by_id: dict[str, tuple[dict, str]] = {}
    order: list[str] = []
    for mod_entry in contract.get("modules") or []:
        if not isinstance(mod_entry, dict):
            continue
        module = _try_normalize_module(str(mod_entry.get("module") or ""))
        for unit in mod_entry.get("expected_units") or []:
            if not isinstance(unit, dict):
                continue
            uid = unit.get("unit_id")
            if not uid:
                continue
            uid = str(uid)
            by_id[uid] = (unit, module)
            order.append(uid)
    return by_id, order


def _run_completed_after(
    run_dir: Path,
    attempt: int,
    *,
    stage: str | None = None,
) -> bool:
    """True iff the same stage completed on a strictly later attempt.

    ``RUN_STATUS.json`` describes only the latest stage.  After generation is
    followed by scoring its attempt counter can therefore be lower than the
    generation attempt that discharged a member-level retry.  The append-only
    event ledger preserves the stage-local completion we need for that check.
    """
    events_path = run_dir / RUN_EVENTS_FILENAME
    if events_path.exists():
        try:
            lines = events_path.read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                event = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict) or event.get("event") != "stage_completed":
                continue
            if stage is not None and event.get("stage") != stage:
                continue
            try:
                completed_attempt = int(event.get("attempt_number", 1))
            except (TypeError, ValueError):
                continue
            if completed_attempt > attempt:
                return True

    status_path = run_dir / RUN_STATUS_FILENAME
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(status, dict):
        return False
    if status.get("status") != "completed":
        return False
    if stage is not None and status.get("stage") not in (None, stage):
        return False
    try:
        current = int(status.get("attempt_number", 1))
    except (TypeError, ValueError):
        current = 1
    return current > attempt


def project(run_dir: Path | str) -> ProjectionResult:
    """Join facts (BLOCKS ∪ classified failure events) with reviews into the
    two effective views (plan D5)."""
    run_dir = Path(run_dir)
    member_id = _member_id(run_dir)
    result = ProjectionResult(member_id=member_id)

    facts = _load_facts(run_dir)
    reviews = _load_reviews_strict(run_dir)  # fail-closed on malformed lines
    reviews_by_ref = _reviews_by_ref(reviews, facts)  # may raise on ambiguity

    # --- events_by_ref: one FactView per physical fact ---------------------
    fact_views: list[FactView] = []
    for fact in facts:
        active = _active_review_for_ref(reviews_by_ref.get(fact.event_ref, []))
        fv = FactView(
            event_ref=fact.event_ref,
            source=fact.source,
            scope=fact.scope,
            unit_id=fact.unit_id,
            member_id=member_id,
            fact=fact.record,
            active_review=active,
            disposition=(active or {}).get("disposition") if active else None,
            effective_class=_effective_class(fact, active),
            effective_category=_effective_category(fact, active),
            resolution_status=_resolution_status(fact, active),
            attempt_number=fact.attempt_number,
            line=fact.line,
            has_attempt=fact.has_attempt,
        )
        fact_views.append(fv)
        result.events_by_ref[fact.event_ref] = fv

    # --- units_by_id: attempt-aware per-unit reduction ---------------------
    unit_facts: dict[str, list[FactView]] = {}
    for fv in fact_views:
        if fv.scope == "unit" and fv.unit_id:
            unit_facts.setdefault(fv.unit_id, []).append(fv)

    contract_units, _order = _load_contract_units(run_dir)

    seen: set[str] = set()
    # Contract-declared units first (their unit dicts drive artifact resolution).
    for uid, (unit, module) in contract_units.items():
        seen.add(uid)
        result.units_by_id[uid] = _reduce_unit(
            unit_id=uid, unit=unit, module=module, member_id=member_id,
            run_dir=run_dir, fact_views=unit_facts.get(uid, []),
        )
    # Unit-scoped facts whose unit_id is not in the contract (best-effort).
    for uid, fvs in unit_facts.items():
        if uid in seen:
            continue
        module = fvs[0].fact.get("module")
        result.units_by_id[uid] = _reduce_unit(
            unit_id=uid, unit=None, module=module, member_id=member_id,
            run_dir=run_dir, fact_views=fvs,
        )

    # --- member obligations: member-scoped retry reviews -------------------
    for fv in fact_views:
        if fv.scope != "member" or fv.disposition != "retry":
            continue
        fulfilled = _run_completed_after(
            run_dir,
            fv.attempt_number,
            stage=fv.fact.get("stage"),
        )
        result.member_obligations.append(MemberObligation(
            member_id=member_id, event_ref=fv.event_ref, kind="retry",
            fulfilled=fulfilled, fact=fv.fact,
        ))

    return result


# ---------------------------------------------------------------------------
# D4 — locked, validated review append
# ---------------------------------------------------------------------------

def _acquire_review_lock(lock_path: Path, timeout: float) -> int:
    """Acquire the review lock via ``O_EXCL``, waiting up to *timeout*.

    Final-gate F4: NO blind time-based steal.  A review append is a fast
    validate-then-append-one-line under the lock, and ``append_review`` always
    removes the lock in its ``finally``; the only way a lock outlives its writer
    is a hard-kill mid-append.  Stealing on age alone could yank the lock from a
    slow-but-live writer and let two active heads through — the very race the
    lock exists to prevent — so instead we wait to the deadline and raise a
    clear, actionable error.  The operator removes the orphaned lock explicitly.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                raise ReviewLockError(
                    f"could not acquire {lock_path} within {timeout}s; another "
                    f"`bench review` invocation may hold it. If no writer is "
                    f"running (a process was hard-killed mid-append), remove the "
                    f"lock file manually: rm {lock_path}"
                )
            time.sleep(0.02)


def _validate_review(run_dir: Path, review: dict[str, Any],
                     existing: list[dict[str, Any]], facts: list[Fact]) -> None:
    disposition = review.get("disposition")
    if disposition not in VALID_DISPOSITIONS:
        raise ReviewValidationError(
            f"invalid disposition {disposition!r}; expected one of "
            f"{sorted(VALID_DISPOSITIONS)}"
        )
    ref = review.get("event_ref")
    if not ref:
        raise ReviewValidationError("v2 review requires an event_ref")
    resolved = resolve_event_ref(run_dir, ref)  # raises on drift / not found

    # Locate the target fact for scope + underlying-category validation.
    target = next((f for f in facts if f.event_ref == resolved.event_ref), None)
    scope = target.scope if target else _classify_scope(
        resolved.source, resolved.record.get("unit_id"))

    if disposition == "retry" and scope == "unmappable_legacy":
        raise ReviewValidationError(
            f"retry disposition rejected for unmappable-legacy fact {ref}; map "
            f"it first (add unit_id) or use a non-retry disposition"
        )

    if disposition == "safety_declination":
        underlying_class = resolved.record.get("evidence_class")
        underlying_cat = resolved.record.get("category")
        needs_resolved = (
            underlying_class in _UNKNOWN_CLASSES
            or underlying_cat in _UNCLASSIFIED_CATEGORIES
        )
        if needs_resolved and not review.get("resolved_category"):
            raise ReviewValidationError(
                f"safety_declination on unclassified/ambiguous fact {ref} "
                f"(class={underlying_class}, category={underlying_cat}) requires "
                f"resolved_category"
            )

    # Supersession / duplicate-active-head guard (atomic under the lock).  All
    # comparisons are on the CANONICAL ref so a line-form review and an id-form
    # fact are treated as the same target (final-gate F4a).
    canonical = _canonical_review_ref(review, facts) or resolved.event_ref
    existing_v2_same = [
        r for r in existing
        if r.get("review_id") and _canonical_review_ref(r, facts) == canonical
    ]
    supersede_id = review.get("supersedes_review_id")
    if supersede_id:
        # A supersede must name an EXISTING review that is the CURRENT ACTIVE
        # HEAD of the SAME canonical ref — never a nonexistent/stale/foreign id
        # (that was the gate bypass Sol reproduced).
        target = next(
            (r for r in existing_v2_same if r.get("review_id") == supersede_id), None
        )
        if target is None:
            raise ReviewValidationError(
                f"supersedes_review_id {supersede_id!r} does not name an existing "
                f"review on {canonical}; refuse to append a phantom supersession"
            )
        head = _active_review_for_ref(existing_v2_same)  # may raise on prior corruption
        if head is None or head.get("review_id") != supersede_id:
            raise DuplicateActiveReviewError(
                f"supersedes_review_id {supersede_id!r} is not the current active "
                f"head for {canonical} "
                f"(head={None if head is None else head.get('review_id')!r})"
            )
    elif existing_v2_same:
        # No supersede: a live active head means this would create a second head.
        active = _active_review_for_ref(existing_v2_same)
        if active is not None:
            raise DuplicateActiveReviewError(
                f"an active review already exists for {canonical}; pass "
                f"supersedes_review_id to supersede it"
            )


def append_review(
    run_dir: Path | str,
    review: dict[str, Any],
    *,
    lock_timeout: float = _DEFAULT_LOCK_TIMEOUT,
    validate: bool = True,
) -> dict[str, Any]:
    """Validate then append a v2 review under an ``O_EXCL`` lock (plan D4).

    The lock is held across validate-then-append so two concurrent invocations
    cannot create two active heads; on lock contention the loser raises
    :class:`ReviewLockError`, and a duplicate active head raises
    :class:`DuplicateActiveReviewError`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    record = dict(review)
    record.setdefault("schema_version", REVIEW_SCHEMA_VERSION)
    record.setdefault("review_id", uuid.uuid4().hex)

    lock_path = run_dir / BLOCK_REVIEWS_LOCK_FILENAME
    lock_fd = _acquire_review_lock(lock_path, lock_timeout)
    try:
        existing = load_reviews(run_dir)
        facts = _load_facts(run_dir)
        # Stamp the D4-required fields the caller omitted from the resolved fact,
        # so every appended review is a complete v2 record (final-gate F4c).
        _stamp_review_fields(record, facts)
        if validate:
            _validate_review(run_dir, record, existing, facts)
        with (run_dir / BLOCK_REVIEWS_FILENAME).open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    return record


def _stamp_review_fields(record: dict[str, Any], facts: list[Fact]) -> None:
    """Fill the D4-required review fields the caller omitted (final-gate F4c).

    ``reviewed_at`` gets the current UTC time; ``module``/``model``/``category``
    are copied from the target fact so the stored review always names the fact's
    own module/model/category (never silently blank).  All use setdefault so an
    explicit caller value wins.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    record.setdefault("reviewed_at", datetime.now(timezone.utc).isoformat())
    ref = record.get("event_ref")
    if not ref:
        return
    fact = _fact_for_ref_in(facts, ref)
    if fact is None:
        return
    for key in ("module", "model", "category"):
        value = fact.record.get(key)
        if not record.get(key) and value is not None:
            record[key] = value


# ---------------------------------------------------------------------------
# Public API (plan 020 T7 hardening + finding 7)
# ---------------------------------------------------------------------------
# Stable, non-underscore names for cross-module consumers — most importantly the
# bundle publication gate (a publication-safety path that must not silently break
# on an internals refactor) and ``bench review``.  The underscore names remain as
# the internal definitions and backward-compatible aliases.
UNKNOWN_CLASSES = _UNKNOWN_CLASSES
RESOLVING_DISPOSITIONS = _RESOLVING_DISPOSITIONS
load_facts = _load_facts
reviews_by_ref = _reviews_by_ref


def is_publication_blocking(fv: "FactView") -> str | None:
    """Return a reason string if *fv* blocks the D6 publication gate, else ``None``.

    Encodes three clauses at the individual-fact level so the fact-level gate
    consumers (``bench review``'s ``gate_blocking`` field and the bundle gate's
    per-fact loop) share one authoritative predicate and cannot drift.

    Clauses (mirrors ``_apply_publication_gate`` in bundle.py):

    * **clause b** — active ``needs_escalation`` review (any evidence class)
    * **clause a** — ``unknown``-class fact with no active resolving review
    * **clause c** — ``instrument_defect`` effective class (blocks until fixed
      and re-run; a review that sets the disposition to ``instrument_defect``
      keeps the fact blocking until the defect is corrected)

    Note: ``retry`` discharge is attempt-aware and is evaluated at the
    *UnitView* level (``uv.state`` in bundle.py), not at the FactView level.
    A FactView with ``disposition=='retry'`` does NOT return a blocking reason
    here because the FactView cannot determine whether the retry was discharged
    by a strictly-later completed attempt.

    ``safety_declination`` adjudicates a fact as a true model-signal and is
    NOT publication-blocking; it is the one disposition that clears a fact from
    the gate and from the ``bench review`` default queue.

    **bench blockers suppression**: ``bench blockers`` uses
    ``_QUEUE_CLEARING_DISPOSITIONS`` (only ``safety_declination``) to decide
    whether to hide a halt, NOT this predicate, because blockers shows all
    active-concern halts (including bare model_signal halts that do not block
    publication).
    """
    if fv.disposition == "needs_escalation":
        return "active needs_escalation review"
    if (
        fv.effective_class in _UNKNOWN_CLASSES
        and fv.disposition not in _RESOLVING_DISPOSITIONS
        and fv.disposition != "retry"
    ):
        return "unknown-class fact with no active resolving review"
    if fv.effective_class == "instrument_defect":
        return "instrument_defect fact (blocks until fixed and re-run)"
    return None
