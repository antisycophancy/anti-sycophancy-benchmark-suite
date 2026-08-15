"""Long-format per-unit score-row extraction (Phase C, Task 1).

``score_rows(run_dir, module)`` walks a run's ``RUN_CONTRACT.json`` expected
units, joins each to its producer score record, and emits **one row per
(unit x dimension)** in a strictly per-unit long format that Wilson/bootstrap
aggregation can sum over.  Aggregate-only dimensions (e.g. the EPIS sycophancy
score) are NOT emitted here; ``epis_aggregate`` is exported as a pure helper for
Task 4 to recompute them at experiment level after ``union()``.

Design invariants (review-locked, see task-1-brief):

* ``outcome_class`` has four states derived from ``owed_units``:
  ``terminal_model_signal`` / ``missing`` (owed) / ``scored`` (done + matched
  score record) / ``unscored`` (done, no matched record).  Rows are emitted
  only for ``scored`` units; ``units[]`` carries every expected unit.
* ``score_scope in {side, pair, run}`` with ``score_subject_id`` = the side
  label, the pair id, or the run/unit id.  The row ``unit_id`` is always a
  verbatim contract unit_id; pair-scope rows carry the score-bearing unit's id
  with ``score_subject_id`` set to the pair id.
* ``value`` is always numeric/boolean; ``display_label`` carries the human
  string for categorical outcomes (SUS cap/no_cap).
* Rows are gated to declared contract dimensions plus an explicit per-module
  allowlist; every other key is summarised in ``unmapped_keys`` **by name
  only** so raw judge replies / nested dicts never enter the bundle.  A unit
  whose matched score record carries *only* unmapped (or excluded) keys is
  still classified ``scored`` in ``units[]`` yet contributes **zero rows** —
  the presence of a score record, not of any mappable metric, is what makes a
  unit scored.

Producer helpers are imported lazily (Phase B precedent) so editable-install
drift on a suite package cannot break importing this module.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from suite_tools import owed_units as _owed
from suite_tools.condition_axes import condition_axes, load_model_aliases
from suite_tools.scoring_contracts import SuiteScoringContract, get_scoring_contract
from suite_tools.suite_registry import normalize_module_name as _normalize_module_name


def _try_normalize_module(name: str) -> str:
    """Return the canonical module name, or *name* unchanged for unknown modules."""
    try:
        return _normalize_module_name(name)
    except ValueError:
        return name

SCHEMA_VERSION = "benchmark-score-rows-v1"
CONTRACT_FILENAME = "RUN_CONTRACT.json"
FINAL_RESULTS_FILENAME = "FINAL_RESULTS.json"
SUS_SIDECAR_FILENAME = "FINAL_RESULTS-conversations.json"

# Provenance string for the exported EPIS aggregate helper.
EPIS_AGGREGATE_HELPER = "epis_bench.report.compute_epistemic_sycophancy_score"

# Explicit per-module allowlist of derived-metric keys the adapter consumes
# beyond the declared contract dimensions (finding 4).  These are the value
# sources the adapter reads; they are neither rows of their own nor unmapped.
_AITA_ALLOWLIST_KEYS = frozenset({
    "verdict_alignment_a_majority",
    "verdict_alignment_b_majority",
    "paired_verdict_alignment_majority",
})

_ITEM_RE = re.compile(r"item(\d+)")
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small parsing / io helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _parse_model_key(unit_id: str) -> str:
    parts = unit_id.split(":")
    return parts[1] if len(parts) > 1 else ""


def _parse_item_idx(unit_id: str) -> int | None:
    match = _ITEM_RE.search(unit_id or "")
    return int(match.group(1)) if match else None


def _score_key_from_unit(unit: dict[str, Any]) -> str | None:
    """Derive the FINAL_RESULTS ``scores`` key from a unit's score path.

    ``m_item0_scores.json`` -> ``m_item0`` (AITA);
    ``m_item0_pickside_scores.json`` -> ``m_item0_pickside`` (EPIS).  Mirrors
    the producer key derivation (epis_bench/runner.py ``_score_result_key``).
    """
    raw = unit.get("expected_score_path")
    if not raw:
        return None
    return Path(str(raw)).stem.removesuffix("_scores")


def _as_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _base_outcome_class(state: str, has_score_record: bool) -> str:
    """Map an ``owed_units`` state + match flag to the four-state outcome."""
    if state == "terminal_model_signal":
        return "terminal_model_signal"
    if state == "owed":
        return "missing"
    # state == "done"
    return "scored" if has_score_record else "unscored"


# ---------------------------------------------------------------------------
# Contract / condition context
# ---------------------------------------------------------------------------

def _selected_modules(contract: dict[str, Any], module: str | None) -> list[dict[str, Any]]:
    mods = [m for m in (contract.get("modules") or []) if isinstance(m, dict)]
    if module is None:
        return mods
    # Use normalized comparison so "epistemic" matches "epis" and vice-versa.
    canonical_filter = _try_normalize_module(module)
    return [m for m in mods if _try_normalize_module(str(m.get("module") or "")) == canonical_filter]


def _axes_for_condition(condition: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    """Return grouping axes, honouring pre-computed axis keys when present.

    Real contracts store raw condition fields (``model_id``/``endpoint``/...),
    which ``condition_axes`` projects.  Identity fixtures may instead store the
    already-projected axes directly; those win when supplied.
    """
    axes = condition_axes(condition, aliases=aliases)
    for key in ("canonical_model", "route", "effort", "profile"):
        value = condition.get(key)
        if value is not None:
            axes[key] = value
    return axes


def _condition_index(
    model_conditions: list[Any], aliases: dict[str, str]
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for condition in model_conditions:
        if not isinstance(condition, dict):
            continue
        key = condition.get("key")
        if key is None:
            continue
        index[str(key)] = {
            "condition_id": condition.get("condition_id"),
            "axes": _axes_for_condition(condition, aliases),
        }
    return index


_EMPTY_AXES = {"canonical_model": None, "route": None, "effort": None, "profile": None}


def _condition_axes_for(model_key: str, condition_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entry = condition_index.get(str(model_key))
    if entry is None:
        return dict(_EMPTY_AXES)
    return entry["axes"]


def _block_categories(run_dir: Path) -> dict[str, Any]:
    """Return ``{unit_id: effective_category}`` for model_signal blocked units.

    The raw BLOCKS category is read first (byte-identical to the pre-projection
    behaviour), then overlaid with the projection's *effective* category so a
    ``safety_declination`` review's ``resolved_category`` drives the displayed
    terminal category (plan 020 D5).  For zero reviews (or confirming backfill
    reviews, which carry no ``resolved_category``) the overlay is a no-op.
    """
    path = run_dir / _owed.BLOCKS_FILENAME
    out: dict[str, Any] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("evidence_class") == "model_signal" and entry.get("unit_id"):
                out[str(entry["unit_id"])] = entry.get("category")

    # Overlay effective categories from the projection (dispositions drive the
    # displayed category).  Imported lazily to keep the reader tolerant of
    # editable-install drift (Phase B precedent) and avoid import cycles.
    try:
        from suite_tools.review_projection import project as _project  # noqa: PLC0415
        projection = _project(run_dir)
    except Exception:  # pragma: no cover - defensive: never let display break scoring
        return out
    for uid, uv in projection.units_by_id.items():
        carrier = uv.carrier
        if carrier is not None and carrier.effective_class == "model_signal":
            out[uid] = uv.effective_category
    return out


def _unit_summary_entry(
    unit_id: str, module: str, outcome: str, block_category: dict[str, Any]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "unit_id": unit_id,
        "module": module,
        "outcome_class": outcome,
    }
    if outcome == "terminal_model_signal":
        category = block_category.get(unit_id)
        if category is not None:
            entry["category"] = category
    return entry


def _make_row(
    *,
    unit_id: str,
    module: str,
    model_key: str,
    condition: dict[str, Any],
    item_axis: dict[str, Any],
    dimension: str,
    value: Any,
    value_source: str,
    dim_meta: Any,
    score_scope: str,
    score_subject_id: Any,
    outcome_class: str,
    display_label: Any = None,
    panel_rate: Any = None,
    attempt_number: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "unit_id": unit_id,
        "module": module,
        "model_key": model_key,
        "condition": condition,
        "item_axis": item_axis,
        "dimension": dimension,
        "value": value,
        "value_source": value_source,
        "role": dim_meta.role,
        "release_facing": dim_meta.release_facing,
        "direction": dim_meta.direction,
        "statistic_kind": dim_meta.statistic_kind,
        "score_scope": score_scope,
        "score_subject_id": score_subject_id,
        "outcome_class": outcome_class,
    }
    if display_label is not None:
        row["display_label"] = display_label
    if panel_rate is not None:
        row["panel_rate"] = panel_rate
    if attempt_number is not None:
        row["attempt_number"] = attempt_number
    if extra:
        row.update(extra)
    return row


def _unmapped_names(record: dict[str, Any], contract: SuiteScoringContract, allowlist) -> set[str]:
    """Return record keys that are neither declared dimensions nor allowlisted."""
    declared = {dim.key for dim in contract.dimensions}
    return {key for key in record if key not in declared and key not in allowlist}


# ---------------------------------------------------------------------------
# AITA adapter
# ---------------------------------------------------------------------------

_AITA_SIDE_PRIMARY = {"a": "verdict_alignment_a", "b": "verdict_alignment_b"}
_AITA_SIDE_DIAGNOSTICS = {
    "a": ("outcome_a", "resistance_a", "therapeutic_a"),
    "b": ("outcome_b", "resistance_b", "therapeutic_b"),
}
_AITA_MAJORITY_KEY = {
    "verdict_alignment_a": "verdict_alignment_a_majority",
    "verdict_alignment_b": "verdict_alignment_b_majority",
    "paired_verdict_alignment": "paired_verdict_alignment_majority",
}


def _aita_side(unit: dict[str, Any]) -> str:
    """Return the normalized AITA side key (``a``/``b``) for a unit.

    Prefer the unit's own ``side`` field (mirrors ``_epis_side``); fall back to
    the unit_id tail.  Producers write ``side_a``/``side_b`` on both the field
    and the unit_id tail, so the ``side_`` prefix is stripped to match the side
    lookup tables (which are keyed ``a``/``b``).  Without this normalization the
    tail ``side_a`` never matches ``_AITA_SIDE_PRIMARY`` and real AITA contracts
    emit zero rows.
    """
    raw = unit.get("side")
    if not raw:
        raw = str(unit.get("unit_id") or "").rsplit(":", 1)[-1]
    raw = str(raw)
    return raw[len("side_"):] if raw.startswith("side_") else raw


def _aita_adapter(*, run_dir, module, expected, state_by_uid, block_category, condition_index):
    contract = get_scoring_contract(module)
    final = _load_json(run_dir / FINAL_RESULTS_FILENAME)
    scores = (final.get("scores") or {}) if isinstance(final, dict) else {}
    rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    unmapped: set[str] = set()

    for unit in expected:
        uid = str(unit.get("unit_id") or "")
        side = _aita_side(unit)
        score_key = _score_key_from_unit(unit)
        record = scores.get(score_key) if score_key else None
        has_record = isinstance(record, dict)
        outcome = _base_outcome_class(state_by_uid.get(uid, "owed"), has_record)
        units.append(_unit_summary_entry(uid, module, outcome, block_category))
        if outcome != "scored" or not has_record:
            continue
        unmapped.update(_unmapped_names(record, contract, _AITA_ALLOWLIST_KEYS))

        model_key = unit.get("model_key") or _parse_model_key(uid)
        condition = _condition_axes_for(model_key, condition_index)
        item_axis = {"item_idx": unit.get("item_idx", _parse_item_idx(uid))}
        pair_id = uid.rsplit(":", 1)[0]

        def emit(dimension, value, value_source, scope, subject, **kw):
            rows.append(_make_row(
                unit_id=uid, module=module, model_key=model_key, condition=condition,
                item_axis=item_axis, dimension=dimension, value=value,
                value_source=value_source, dim_meta=contract.dimension(dimension),
                score_scope=scope, score_subject_id=subject, outcome_class=outcome, **kw))

        # Side primary outcome: value from the release-facing panel-majority
        # binary, raw panel pass-rate preserved as panel_rate.
        primary = _AITA_SIDE_PRIMARY.get(side)
        if primary and primary in record:
            majority_key = _AITA_MAJORITY_KEY[primary]
            majority = record.get(majority_key)
            if majority is not None:
                emit(primary, majority, majority_key, "side", side,
                     panel_rate=record.get(primary))

        # Side diagnostics (raw scalar, n-only).
        for dkey in _AITA_SIDE_DIAGNOSTICS.get(side, ()):  # a/b
            if dkey in record:
                emit(dkey, record.get(dkey), dkey, "side", side)

        # Pair-scope rows are emitted ONCE, carried on the side-a score carrier.
        if side == "a":
            majority = record.get(_AITA_MAJORITY_KEY["paired_verdict_alignment"])
            if "paired_verdict_alignment" in record and majority is not None:
                emit("paired_verdict_alignment", majority,
                     _AITA_MAJORITY_KEY["paired_verdict_alignment"], "pair", pair_id,
                     panel_rate=record.get("paired_verdict_alignment"))
            if "consistency" in record:
                emit("consistency", record.get("consistency"), "consistency", "pair", pair_id)

    return rows, units, unmapped


# ---------------------------------------------------------------------------
# EPIS adapter
# ---------------------------------------------------------------------------

_EPIS_TWO_SIDED = ("pickside", "mirror")
_EPIS_WITHIN_FLAGS = ("endpoint_shift", "integrity_break")   # score_scope = side (side_a)
_EPIS_CROSS_FLAGS = ("side_inconsistency", "stance_amplification")  # score_scope = pair


def _epis_side(unit: dict[str, Any]) -> str:
    side = unit.get("side")
    if side:
        return str(side)
    return str(unit.get("unit_id") or "").rsplit(":", 1)[-1]


def _epis_pair_id(unit_id: str) -> str:
    return unit_id.rsplit(":", 1)[0]


def _lookup_record(scores: dict[str, Any], candidates: list[str]) -> dict[str, Any] | None:
    for key in candidates:
        record = scores.get(key)
        if isinstance(record, dict):
            return record
    return None


def _epis_score_candidates(unit: dict[str, Any], unit_id: str) -> list[str]:
    """Candidate FINAL_RESULTS ``scores`` keys for an EPIS side-a unit.

    Path-derived first (matches the producer's ``filename_model_key``-based
    key exactly for real runs), then field-constructed
    ``{model_key}_item{idx}_{test_type}`` as a fallback (epis runner
    ``_score_result_key``).
    """
    candidates: list[str] = []
    path_key = _score_key_from_unit(unit)
    if path_key:
        candidates.append(path_key)
    model_key = unit.get("model_key") or _parse_model_key(unit_id)
    item_idx = unit.get("item_idx")
    if item_idx is None:
        item_idx = _parse_item_idx(unit_id)
    parts = unit_id.split(":")
    test_type = unit.get("test_type") or (parts[2] if len(parts) > 2 else "")
    if model_key and item_idx is not None and test_type:
        candidates.append(f"{model_key}_item{item_idx}_{test_type}")
    return candidates


def _epis_flags(record: dict[str, Any]) -> dict[str, Any]:
    from epis_bench.report import dimension_failure_flags

    return dimension_failure_flags(record)


def _epis_adapter(*, run_dir, module, expected, state_by_uid, block_category, condition_index):
    contract = get_scoring_contract(module)
    final = _load_json(run_dir / FINAL_RESULTS_FILENAME)
    scores = (final.get("scores") or {}) if isinstance(final, dict) else {}
    rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    unmapped: set[str] = set()

    # Resolve each pair's side-a carrier outcome first so transcript-only
    # side_b partners can mirror it.
    side_a_outcome_by_pair: dict[str, str] = {}
    for unit in expected:
        if _epis_side(unit) != "side_a":
            continue
        uid = str(unit.get("unit_id") or "")
        record = _lookup_record(scores, _epis_score_candidates(unit, uid))
        has_record = isinstance(record, dict)
        side_a_outcome_by_pair[_epis_pair_id(uid)] = _base_outcome_class(
            state_by_uid.get(uid, "owed"), has_record
        )

    for unit in expected:
        uid = str(unit.get("unit_id") or "")
        side = _epis_side(unit)
        pair_id = _epis_pair_id(uid)

        if side != "side_a":
            # Transcript-only pair partner: it carries no score record of its
            # own, so when its own unit completed it is scored *through* side_a's
            # pair record and mirrors the carrier's class.  But side_b's own
            # terminal_model_signal (a real refusal) — or an owed/missing state —
            # must win: mirroring the carrier must never erase side_b's own
            # terminal signal.  Mirror only when side_b's own state is done.
            own_state = state_by_uid.get(uid, "owed")
            carrier = side_a_outcome_by_pair.get(pair_id)
            if own_state == "done" and carrier is not None:
                outcome = carrier
            else:
                outcome = _base_outcome_class(own_state, False)
            units.append(_unit_summary_entry(uid, module, outcome, block_category))
            continue

        record = _lookup_record(scores, _epis_score_candidates(unit, uid))
        has_record = isinstance(record, dict)
        outcome = _base_outcome_class(state_by_uid.get(uid, "owed"), has_record)
        units.append(_unit_summary_entry(uid, module, outcome, block_category))
        if outcome != "scored" or not has_record:
            continue
        unmapped.update(_unmapped_names(record, contract, frozenset()))

        model_key = unit.get("model_key") or _parse_model_key(uid)
        test_type = unit.get("test_type")
        if not test_type:
            uid_parts = uid.split(":")
            test_type = uid_parts[2] if len(uid_parts) > 2 else ""
        test_type = str(test_type)
        condition = _condition_axes_for(model_key, condition_index)
        item_axis = {"test_type": test_type,
                     "item_idx": unit.get("item_idx", _parse_item_idx(uid))}
        two_sided = test_type in _EPIS_TWO_SIDED

        def emit(dimension, value, value_source, scope, subject):
            rows.append(_make_row(
                unit_id=uid, module=module, model_key=model_key, condition=condition,
                item_axis=item_axis, dimension=dimension, value=value,
                value_source=value_source, dim_meta=contract.dimension(dimension),
                score_scope=scope, score_subject_id=subject, outcome_class=outcome))

        # Raw within-conversation dims (from side-a turns) -> side scope.
        for dkey in ("persistence", "integrity"):
            if dkey in record and record.get(dkey) is not None:
                emit(dkey, record.get(dkey), dkey, "side", "side_a")
        # Raw pair-derived dims -> pair scope.
        for dkey in ("consistency", "amplification"):
            if dkey in record and record.get(dkey) is not None:
                emit(dkey, record.get(dkey), dkey, "pair", pair_id)

        # Derived hard-failure booleans (None -> row omitted, never faked False).
        flags = _epis_flags(record)
        for dkey in _EPIS_WITHIN_FLAGS:
            value = flags.get(dkey)
            if value is not None:
                emit(dkey, value, dkey, "side", "side_a")
        for dkey in _EPIS_CROSS_FLAGS:
            value = flags.get(dkey)
            if value is not None:
                emit(dkey, value, dkey, "pair", pair_id)

        # primary_failure <- panel-majority hard_failure; scope is per-test_type
        # (pair for two-sided pickside/mirror, side otherwise).
        hard = flags.get("hard_failure")
        if hard is not None:
            if two_sided:
                scope, subject = "pair", pair_id
            else:
                scope, subject = "side", "side_a"
            value_source = (
                "primary_failure"
                if isinstance(record.get("primary_failure"), bool)
                else "dimension_failure_flags:hard_failure"
            )
            emit("primary_failure", hard, value_source, scope, subject)

    return rows, units, unmapped


# ---------------------------------------------------------------------------
# SUS adapter
# ---------------------------------------------------------------------------

def _sus_capitulation_summary(record: dict[str, Any]) -> dict[str, Any]:
    from sus_bench.scoring_contract import capitulation_summary

    return capitulation_summary(record)


def _sus_is_excluded(record: dict[str, Any]) -> bool:
    from sus_bench.scoring_contract import is_score_excluded_result

    return is_score_excluded_result(record)


def _sus_join_key(condition_id: Any, scenario: Any, run_number: Any) -> tuple:
    return (str(condition_id), str(scenario), _as_int(run_number))


def _sus_adapter(*, run_dir, module, expected, state_by_uid, block_category, condition_index):
    contract = get_scoring_contract(module)
    sidecar = _load_json(run_dir / SUS_SIDECAR_FILENAME)
    records = [r for r in sidecar if isinstance(r, dict)] if isinstance(sidecar, list) else []
    declared = {dim.key for dim in contract.dimensions}
    rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    unmapped: set[str] = set()

    # Join sidecar records to expected units by (condition_id, scenario,
    # run_number) — (scenario, run_number) alone is non-unique across
    # conditions (finding 1).
    record_by_join: dict[tuple, dict[str, Any]] = {}
    for record in records:
        record_by_join[_sus_join_key(
            record.get("condition_id"), record.get("scenario"), record.get("run_number")
        )] = record

    unit_join_keys: set[tuple] = set()
    for unit in expected:
        uid = str(unit.get("unit_id") or "")
        model_key = unit.get("model_key") or _parse_model_key(uid)
        cond_entry = condition_index.get(str(model_key)) or {}
        join_key = _sus_join_key(
            cond_entry.get("condition_id"), unit.get("scenario"), unit.get("run_number")
        )
        unit_join_keys.add(join_key)
        record = record_by_join.get(join_key)
        has_record = isinstance(record, dict)
        outcome = _base_outcome_class(state_by_uid.get(uid, "owed"), has_record)
        units.append(_unit_summary_entry(uid, module, outcome, block_category))
        if outcome != "scored" or not has_record:
            continue
        # Provider-refusal-excluded records: counted in units[] but emit no rows.
        if _sus_is_excluded(record):
            continue

        condition = _condition_axes_for(model_key, condition_index)
        item_axis = {"scenario": unit.get("scenario"), "run_number": _as_int(unit.get("run_number"))}

        # cap_outcome: numeric 1/0 (Wilson-summable) + human display_label, read
        # from the real phase fields via the production helper.
        cap = _sus_capitulation_summary(record)
        cap_label = cap.get("cap_outcome")
        cap_extra: dict[str, Any] = {}
        if cap.get("capitulated") and cap.get("cap_phase_label"):
            cap_extra["cap_phase_label"] = cap.get("cap_phase_label")
        rows.append(_make_row(
            unit_id=uid, module=module, model_key=model_key, condition=condition,
            item_axis=item_axis, dimension="cap_outcome",
            value=1 if cap_label == "cap" else 0, value_source="phases",
            dim_meta=contract.dimension("cap_outcome"), score_scope="run",
            score_subject_id=uid, outcome_class=outcome, display_label=cap_label,
            extra=cap_extra or None))

        # Nested severity/component score dims.
        nested = record.get("score")
        nested = nested if isinstance(nested, dict) else {}
        for skey, sval in nested.items():
            if skey == "cap_outcome":
                continue
            if skey in declared:
                rows.append(_make_row(
                    unit_id=uid, module=module, model_key=model_key, condition=condition,
                    item_axis=item_axis, dimension=skey, value=sval,
                    value_source=f"score.{skey}", dim_meta=contract.dimension(skey),
                    score_scope="run", score_subject_id=uid, outcome_class=outcome))
            else:
                unmapped.add(skey)

    # Sidecar records with no matching expected unit are dropped with a warning.
    for join_key in record_by_join:
        if join_key not in unit_join_keys:
            _logger.warning("SUS sidecar record has no matching expected unit: %s", join_key)

    return rows, units, unmapped


_ADAPTERS = {
    "aita": _aita_adapter,
    "epis": _epis_adapter,
    "sus": _sus_adapter,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def epis_aggregate(scores: list[dict]) -> dict:
    """Recompute the EPIS aggregate dimensions from per-unit raw records.

    Thin wrapper over the production helper
    ``epis_bench.report.compute_epistemic_sycophancy_score`` (report.py:190).
    Task 4 calls this on ``union()``'s winning per-unit records to build
    experiment-level ``derived_aggregates`` — it is intentionally NOT invoked
    inside ``score_rows`` (the rows file stays strictly per-unit).
    """
    from epis_bench.report import compute_epistemic_sycophancy_score

    return compute_epistemic_sycophancy_score(list(scores))


def score_rows(run_dir: Path | str, module: str | None = None) -> dict[str, Any]:
    """Emit long-format per-unit score rows for a run directory.

    Parameters
    ----------
    run_dir:
        Directory containing ``RUN_CONTRACT.json`` and producer score
        artifacts (``FINAL_RESULTS.json`` and, for SUS, the
        ``FINAL_RESULTS-conversations.json`` sidecar).
    module:
        When given, only that module's units are processed; otherwise all
        modules in the contract are processed.

    Returns
    -------
    dict with keys ``schema_version``, ``module``, ``rows`` (one per
    scored unit x dimension), ``units`` (every expected unit with its
    ``outcome_class``), and ``unmapped_keys`` (``{module: [name, ...]}``,
    names only).
    """
    run_dir = Path(run_dir)
    contract_path = run_dir / CONTRACT_FILENAME
    if not contract_path.exists():
        raise FileNotFoundError(f"RUN_CONTRACT.json not found in {run_dir}")
    contract = json.loads(contract_path.read_text())

    aliases = load_model_aliases()
    identity = contract.get("identity") or {}
    condition_index = _condition_index(identity.get("model_conditions") or [], aliases)

    owed = _owed.owed_units(run_dir, module=module)
    state_by_uid = {u["unit_id"]: u["state"] for u in owed.get("units", [])}
    block_category = _block_categories(run_dir)

    rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    unmapped: dict[str, set[str]] = {}

    for mod_entry in _selected_modules(contract, module):
        # Normalize the contract module name so aliases like "epistemic" route to
        # the "epis" adapter.  Unknown modules fall through to the no-adapter path.
        mod_name = _try_normalize_module(str(mod_entry.get("module") or ""))
        expected = [u for u in (mod_entry.get("expected_units") or []) if isinstance(u, dict)]
        adapter = _ADAPTERS.get(mod_name)
        if adapter is None:
            for unit in expected:
                uid = str(unit.get("unit_id") or "")
                outcome = _base_outcome_class(state_by_uid.get(uid, "owed"), False)
                units.append(_unit_summary_entry(uid, mod_name, outcome, block_category))
            continue
        m_rows, m_units, m_unmapped = adapter(
            run_dir=run_dir,
            module=mod_name,
            expected=expected,
            state_by_uid=state_by_uid,
            block_category=block_category,
            condition_index=condition_index,
        )
        rows.extend(m_rows)
        units.extend(m_units)
        if m_unmapped:
            unmapped.setdefault(mod_name, set()).update(m_unmapped)

    return {
        "schema_version": SCHEMA_VERSION,
        "module": module,
        "rows": rows,
        "units": units,
        "unmapped_keys": {mod: sorted(names) for mod, names in unmapped.items()},
    }
