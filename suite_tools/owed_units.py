"""Owed-units computation for pre-spend status view.

Walk RUN_CONTRACT.json ``expected_units`` for each module, resolve each
unit's artifact (prefer ``expected_transcript_path``), classify via the
unit_state predicates (the single source of truth), and return a schemed
dict.

BLOCKS.jsonl entries with a matching ``unit_id`` and
``evidence_class="model_signal"`` mark the unit ``terminal_model_signal``
**only when** the unit's artifact is absent or itself terminal.  If a
*completed* artifact exists despite a block entry the artifact wins —
reason ``"completed_after_block"`` — reflecting a deliberate
re-execution (takeover/review-retry) that produced newer evidence.
A stale block entry must never gate re-spend or packaging when the
transcript confirms completion.

CLI::

    python3 -m suite_tools.owed_units <run_dir> [--module MOD] [--json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from suite_tools import unit_state as _us
from suite_tools.suite_registry import normalize_module_name as _normalize_module_name

SCHEMA_VERSION = "benchmark-owed-units-v1"
BLOCKS_FILENAME = "BLOCKS.jsonl"
CONTRACT_FILENAME = "RUN_CONTRACT.json"

_KNOWN_MODULES = {"aita", "epis", "sus"}


def _try_normalize_module(name: str) -> str:
    """Return the canonical module name, or *name* unchanged for unknown modules."""
    try:
        return _normalize_module_name(name)
    except ValueError:
        return name

_STATE_DONE = "done"
_STATE_TERMINAL = "terminal_model_signal"
_STATE_OWED = "owed"

# Map the projection's effective per-unit state (suite_tools.review_projection)
# onto owed_units' three-state schema.  For zero reviews (or confirming backfill
# safety_declination reviews) only the first three keys are reachable, so the
# output is byte-identical to the pre-projection behaviour (plan 020 D5,
# acceptance 6).  The remaining keys carry real v2 dispositions: a pending retry
# or an instrument defect re-enters ``owed`` (needs re-run); an unresolved
# integrity/unknown halt stays gate-blocking as ``terminal_model_signal``.
_PROJECTION_STATE_MAP = {
    "completed": _STATE_DONE,
    "terminal_model_signal": _STATE_TERMINAL,
    "owed": _STATE_OWED,
    "pending_retry": _STATE_OWED,
    "instrument_defect": _STATE_OWED,
    "unresolved": _STATE_TERMINAL,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_blocked_unit_ids(run_dir: Path) -> set[str]:
    """Return unit_ids appearing in BLOCKS.jsonl as model_signal blocks."""
    blocks_path = run_dir / BLOCKS_FILENAME
    blocked: set[str] = set()
    if not blocks_path.exists():
        return blocked
    with blocks_path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("evidence_class") == "model_signal" and entry.get("unit_id"):
                blocked.add(str(entry["unit_id"]))
    return blocked


def _resolve_artifact(run_dir: Path, unit: dict[str, Any]) -> Path | None:
    """Return the first existing artifact path for a unit, else None.

    Preference order: expected_transcript_path > expected_score_path >
    expected_summary_path.  Relative paths are resolved from run_dir.
    """
    for key in ("expected_transcript_path", "expected_score_path", "expected_summary_path"):
        raw = unit.get(key)
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = run_dir / path
        if path.exists():
            return path
        if key == "expected_transcript_path":
            legacy_sus_path = _resolve_legacy_sus_transcript(path, unit)
            if legacy_sus_path is not None:
                return legacy_sus_path
    return None


def _resolve_legacy_sus_transcript(
    expected_path: Path,
    unit: dict[str, Any],
) -> Path | None:
    """Resolve SUS artifacts from contracts prepared before option-qualified paths."""
    unit_id = str(unit.get("unit_id") or "")
    if not unit_id.startswith("sus:") or expected_path.suffix != ".json":
        return None
    try:
        candidates = sorted(expected_path.parent.glob(f"{expected_path.stem}_*.json"))
    except OSError:
        return None

    matches: list[Path] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and str(payload.get("unit_id") or "") == unit_id:
            matches.append(candidate)

    # Ambiguous alternate artifacts fail closed rather than selecting a condition.
    return matches[0] if len(matches) == 1 else None


def _dispatch_unit_state(module: str, unit: dict[str, Any], data: dict[str, Any]) -> str:
    """Call the canonical unit_state predicate for *module* and return its result.

    Raises
    ------
    ValueError
        When *module* is not one of the known benchmark modules.
    """
    if module == "sus":
        planned = int(unit.get("planned_escalations") or 0)
        return _us.sus_unit_state(data, planned)
    if module == "epis":
        planned = int(unit.get("planned_turns") or 0)
        return _us.epis_unit_state(data, planned)
    if module == "aita":
        planned = int(unit.get("planned_turns") or 0)
        return _us.aita_unit_state(data, planned)
    raise ValueError(
        f"Unknown module {module!r}; expected one of {sorted(_KNOWN_MODULES)}"
    )


def _classify_unit(
    unit: dict[str, Any],
    *,
    run_dir: Path,
    module: str,
    blocked_unit_ids: set[str],
) -> tuple[str, Path | None, str]:
    """Return ``(state, artifact_path_or_None, reason)`` for one contract unit.

    When a BLOCKS.jsonl entry exists for the unit:
    - artifact absent → terminal_model_signal
    - artifact present and completed → done, reason "completed_after_block"
      (newer evidence from a re-execution overrides the stale block)
    - artifact present and terminal/owed → terminal_model_signal (block wins)
    """
    unit_id: str = unit.get("unit_id") or ""
    artifact = _resolve_artifact(run_dir, unit)

    if unit_id and unit_id in blocked_unit_ids:
        if artifact is not None:
            try:
                data = json.loads(artifact.read_text())
                raw_state = _dispatch_unit_state(module, unit, data)
                if raw_state == "completed":
                    return _STATE_DONE, artifact, "completed_after_block"
            except (OSError, json.JSONDecodeError):
                pass
        # No completed artifact: BLOCKS entry wins as terminal
        return _STATE_TERMINAL, artifact, "BLOCKS.jsonl model_signal entry"

    # No BLOCKS entry: evaluate artifact via unit_state predicate.
    if artifact is None:
        return _STATE_OWED, None, "artifact missing"

    try:
        data = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError):
        return _STATE_OWED, artifact, "artifact unreadable"

    raw_state = _dispatch_unit_state(module, unit, data)

    if raw_state == "completed":
        return _STATE_DONE, artifact, "completed"
    if raw_state == "terminal_model_signal":
        return _STATE_TERMINAL, artifact, "unit_state terminal"
    return _STATE_OWED, artifact, "incomplete"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def owed_units(run_dir: Path | str, *, module: str | None = None) -> dict[str, Any]:
    """Classify each expected unit from ``RUN_CONTRACT.json``.

    Parameters
    ----------
    run_dir:
        Directory containing ``RUN_CONTRACT.json`` and optional
        ``BLOCKS.jsonl``.
    module:
        When given, only the matching ``modules[]`` entry is processed.
        When ``None``, all modules are processed.

    Returns
    -------
    dict with keys: ``schema_version``, ``module``, ``counts``,
    ``units`` (list of ``{unit_id, state, artifact, reason}``).

    Raises
    ------
    FileNotFoundError
        When ``RUN_CONTRACT.json`` is absent from *run_dir*.
    """
    run_dir = Path(run_dir)
    contract_path = run_dir / CONTRACT_FILENAME
    if not contract_path.exists():
        raise FileNotFoundError(
            f"RUN_CONTRACT.json not found in {run_dir}"
        )

    contract = json.loads(contract_path.read_text())

    # The projection (suite_tools.review_projection) is the ONE place BLOCK_REVIEWS
    # judgment joins the recorded facts into an effective per-unit state; owed_units
    # is a thin formatter over its ``units_by_id`` view (plan 020 D5).  Imported
    # lazily to avoid an import cycle (review_projection reuses this module's
    # artifact/predicate helpers).
    from suite_tools.review_projection import project as _project  # noqa: PLC0415

    projection = _project(run_dir)
    units_by_id = projection.units_by_id

    all_modules: list[dict[str, Any]] = [
        m for m in (contract.get("modules") or [])
        if isinstance(m, dict)
    ]
    if module is not None:
        # Normalize the requested filter so aliases ("epistemic") match canonical
        # names ("epis") and vice-versa, using the same try-normalize helper.
        canonical_filter = _try_normalize_module(module)
        selected = [
            m for m in all_modules
            if _try_normalize_module(str(m.get("module") or "")) == canonical_filter
        ]
    else:
        selected = all_modules

    counts: dict[str, int] = {_STATE_DONE: 0, _STATE_TERMINAL: 0, _STATE_OWED: 0}
    units_out: list[dict[str, Any]] = []

    for mod_entry in selected:
        # Normalize the contract module name before dispatch so aliases like
        # "epistemic" route to the correct "epis" predicate.
        mod_name = _try_normalize_module(str(mod_entry.get("module") or ""))
        for unit in mod_entry.get("expected_units") or []:
            if not isinstance(unit, dict):
                continue
            # Preserve the pre-projection unknown-module error surface: the
            # projection tolerates unknown modules (never raises), so owed_units
            # raises here exactly as _dispatch_unit_state used to.
            if mod_name not in _KNOWN_MODULES:
                raise ValueError(
                    f"Unknown module {mod_name!r}; expected one of "
                    f"{sorted(_KNOWN_MODULES)}"
                )
            uid = unit.get("unit_id")
            unit_view = units_by_id.get(str(uid)) if uid else None
            if unit_view is not None:
                state = _PROJECTION_STATE_MAP.get(unit_view.state, _STATE_OWED)
                artifact = unit_view.artifact
                reason = unit_view.reason
            else:
                # No unit_id (or not projected): a block cannot attach without a
                # unit_id, so the pre-projection predicate path is exact here.
                state, artifact, reason = _classify_unit(
                    unit, run_dir=run_dir, module=mod_name, blocked_unit_ids=set()
                )
            counts[state] += 1
            units_out.append({
                "unit_id": uid,
                "state": state,
                "artifact": str(artifact) if artifact is not None else None,
                "reason": reason,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "module": module,
        "counts": counts,
        "units": units_out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify owed units from a run directory (pre-spend status view)."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing RUN_CONTRACT.json",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Filter to a specific module name (default: all)",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output result as JSON",
    )
    args = parser.parse_args(argv)

    result = owed_units(args.run_dir, module=args.module)

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        module_label = result.get("module") or "all"
        counts = result.get("counts") or {}
        print(f"module: {module_label}")
        print(f"  done:                  {counts.get(_STATE_DONE, 0)}")
        print(f"  terminal_model_signal: {counts.get(_STATE_TERMINAL, 0)}")
        print(f"  owed:                  {counts.get(_STATE_OWED, 0)}")
        units = result.get("units") or []
        if units:
            print()
            col_uid = max(len(u.get("unit_id") or "") for u in units)
            col_uid = max(col_uid, len("unit_id"))
            col_st = max(len(u.get("state") or "") for u in units)
            col_st = max(col_st, len("state"))
            header = f"{'unit_id':<{col_uid}}  {'state':<{col_st}}  reason"
            print(header)
            print("-" * len(header))
            for u in units:
                uid = u.get("unit_id") or ""
                st = u.get("state") or ""
                reason = u.get("reason") or ""
                print(f"{uid:<{col_uid}}  {st:<{col_st}}  {reason}")


if __name__ == "__main__":
    _main()
