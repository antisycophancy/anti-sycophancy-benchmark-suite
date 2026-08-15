"""Experiment manifest management.

An experiment groups multiple benchmark runs under a shared instrument
definition, a conditions table, and a target.  The manifest is stored as
``EXPERIMENT.json`` and every mutation appends a timestamped entry to
``EXPERIMENT_LOG.jsonl``.

CLI verbs
---------
    python -m suite_tools.experiment init <dir> --id ID --title TITLE
        [--from-run <run_dir>]... [--target-items N]
    python -m suite_tools.experiment adopt <dir> <run_dir> --role ROLE
    python -m suite_tools.experiment supersede <dir> <member_path>
        --by <member_path> --reason REASON
    python -m suite_tools.experiment status <dir>
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from datetime import timezone as _tz
from pathlib import Path
from typing import Any

from suite_tools.run_contract import (
    IDENTITY_PROJECTION_VERSION,
    load_run_contract,
    provenance_hashes,
    provenance_identity_from_contract,
    stable_json_hash,
    summarize_contract,
)
from suite_tools.run_monitor import atomic_write_json, utc_now
from suite_tools.owed_units import owed_units as _compute_owed_units
from suite_tools.suite_registry import normalize_module_name as _normalize_module_name


def _try_normalize_module(name: str) -> str:
    """Return the canonical module name, or *name* unchanged for unknown modules."""
    try:
        return _normalize_module_name(name)
    except ValueError:
        return name

EXPERIMENT_SCHEMA_VERSION = "benchmark-experiment-v1"
EXPERIMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_experiment_id(value: Any) -> str:
    """Return a portable experiment slug or fail before any filesystem use."""
    if not isinstance(value, str) or not EXPERIMENT_ID_RE.fullmatch(value):
        raise ValueError(
            "experiment_id must be a 1-128 character ASCII slug beginning with "
            "a letter or digit and containing only letters, digits, '.', '_', or '-'"
        )
    return value
EXPERIMENT_UNION_SCHEMA_VERSION = "benchmark-experiment-union-v1"
EXPERIMENT_FILENAME = "EXPERIMENT.json"
EXPERIMENT_LOG_FILENAME = "EXPERIMENT_LOG.jsonl"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InstrumentMismatch(Exception):
    """Raised when a member's recomputed benchmark_condition_hash does not match
    the experiment's stored hash for that module.

    The error message always contains ``"benchmark_condition_hash"`` so callers
    can assert on it.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_manifest(exp_dir: Path) -> dict[str, Any]:
    return json.loads((exp_dir / EXPERIMENT_FILENAME).read_text())


def _save_manifest(exp_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(exp_dir / EXPERIMENT_FILENAME, manifest)


def _append_log(exp_dir: Path, event: dict[str, Any]) -> None:
    log_path = exp_dir / EXPERIMENT_LOG_FILENAME
    with open(log_path, "a") as fh:
        fh.write(json.dumps({**event, "timestamp": utc_now()}) + "\n")


def _derive_target_items(contract: dict[str, Any]) -> int:
    """Derive ``target.n_items`` from the seeding contract (module-aware, R3-4 spec).

    SUS:   scenarios × runs_per_scenario
    AITA:  n_items × 2  (side_a + side_b)
    EPIS:  units per model condition (len(expected_units) / n_model_conditions),
           or fallback to len(expected_units)
    """
    modules = contract.get("modules") or []
    if not modules:
        return 0
    module_entry = modules[0]
    module = module_entry.get("module") or ""
    identity = contract.get("identity") or {}
    sample = identity.get("sample_spec") or {}

    if module == "sus":
        scenario_ids = sample.get("scenario_ids") or []
        runs = int(sample.get("runs") or sample.get("run_count") or 1)
        return len(scenario_ids) * runs

    if module == "aita":
        # items × 2 sides; use n_items from sample_spec when available
        n_items = sample.get("n_items")
        if n_items is not None:
            return int(n_items) * 2
        # Fall back: count distinct item_idx values across expected_units
        expected_units = module_entry.get("expected_units") or []
        item_indices = {u.get("item_idx") for u in expected_units if isinstance(u, dict)}
        return len(item_indices) * 2

    if module in ("epis", "epistemic"):
        # epis has pickside/mirror (2 sides) and delusion/others (1 side).
        # Fall back to len(expected_units) per model condition.
        expected_units = module_entry.get("expected_units") or []
        identity_mc = (contract.get("identity") or {}).get("model_conditions") or []
        n_conditions = len(identity_mc) or 1
        return len(expected_units) // n_conditions

    # Generic fallback
    expected_units = module_entry.get("expected_units") or []
    return len(expected_units)


def _read_run_status(run_dir: Path) -> dict[str, Any]:
    """Read RUN_STATUS.json from *run_dir*; return {} if absent or unreadable."""
    path = run_dir / "RUN_STATUS.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


_EPOCH_MIN = datetime.min.replace(tzinfo=_tz.utc)


def _parse_ts(raw: str | None) -> datetime:
    """Parse an ISO-8601 timestamp to a tz-aware datetime.

    Both ``"...Z"`` and ``"...+00:00"`` are accepted as UTC.  On ``None``,
    empty string, or any parse failure the epoch-minimum (UTC) is returned so
    that unparsable timestamps sort last (lose all comparisons).  This prevents
    a third-party-written ``started_at`` in a non-standard format from silently
    mis-ranking members.
    """
    if not raw:
        return _EPOCH_MIN
    try:
        # Replace trailing 'Z' so fromisoformat works on Python < 3.11;
        # on 3.11+ both forms are handled natively.
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return _EPOCH_MIN


def _resolve_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the winning candidate by spec §5.2: latest ``started_at``.

    Each candidate dict must have:
      ``member``          str  — path to RUN_CONTRACT.json
      ``started_at``      str | None  — from RUN_STATUS.json (ISO-8601)
      ``run_started_at``  str | None  — from RUN_STATUS.json (tiebreaker)
      ``manifest_index``  int  — position in manifest (stable last-resort tiebreaker)
      ``attempt``         int  — attempt_number from RUN_STATUS.json (NOT used for ordering)
      ``state``           str  — unit state

    Plus any extra keys passed through (e.g. ``condition_id``, ``module``).

    Timestamps are parsed via ``_parse_ts`` (tz-aware; unparsable → epoch-oldest).
    Both ``"...Z"`` and ``"...+00:00"`` representations of the same instant
    compare equal, so ties resolve deterministically by ``manifest_index``
    (lower index = earlier in manifest = wins on tie).

    Sort key (timestamps descending; index ascending):
      1. ``started_at``     — latest tz-aware datetime wins; unparsable sorts last
      2. ``run_started_at`` — tiebreaker; unparsable sorts last
      3. ``manifest_index`` — lower index wins on full tie
    """
    if not candidates:
        raise ValueError("_resolve_winner called with empty candidates list")

    def sort_key(c: dict[str, Any]) -> tuple:
        started = _parse_ts(c.get("started_at"))
        run_started = _parse_ts(c.get("run_started_at"))
        # Invert manifest_index so that lower index sorts last when inverted,
        # meaning lower index wins when we take max().
        return (started, run_started, -c.get("manifest_index", 0))

    return max(candidates, key=sort_key)


def _conditions_from_hashes(hashes: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-model conditions from a ``provenance_hashes`` result.

    Returns a list of ``{key, condition_id, condition_hash}`` dicts where:
    - ``key``           = model key string
    - ``condition_id``  = explicit condition_id if present in the model
                         definition, else falls back to ``key``
    - ``condition_hash``= per-model projection hash (``hash`` field from
                         ``model_condition_hashes``)
    """
    entries = []
    for mc_hash in hashes.get("model_condition_hashes") or []:
        if not isinstance(mc_hash, dict):
            continue
        key = str(mc_hash.get("key") or "")
        condition_id = str(mc_hash.get("condition_id") or key)
        condition_hash = str(mc_hash.get("hash") or "")
        entries.append({"key": key, "condition_id": condition_id, "condition_hash": condition_hash})
    return entries


def _model_key_to_condition_id(contract: dict[str, Any]) -> dict[str, str]:
    """Return a ``model_key → condition_id`` map from a contract's identity."""
    identity = provenance_identity_from_contract(contract)
    hashes = provenance_hashes(identity)
    mapping: dict[str, str] = {}
    for mc_hash in hashes.get("model_condition_hashes") or []:
        if not isinstance(mc_hash, dict):
            continue
        key = str(mc_hash.get("key") or "")
        condition_id = str(mc_hash.get("condition_id") or key)
        mapping[key] = condition_id
        mapping[condition_id] = condition_id
    return mapping


def _unit_id_to_condition_id(contract: dict[str, Any]) -> dict[str, str]:
    """Map expected unit ids to conditions without parsing legacy ids.

    Early SUS contracts placed a short condition hash in ``unit_id`` while
    retaining the human model key in the expected-unit record.  Reading the
    record is authoritative and keeps those runs attributable after adoption.
    """
    key_to_cid = _model_key_to_condition_id(contract)
    mapping: dict[str, str] = {}
    for module_entry in contract.get("modules") or []:
        if not isinstance(module_entry, dict):
            continue
        for unit in module_entry.get("expected_units") or []:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id") or "")
            model_key = str(unit.get("model_key") or "")
            condition_id = str(
                unit.get("condition_id")
                or key_to_cid.get(model_key)
                or model_key
            )
            if unit_id and condition_id:
                mapping[unit_id] = condition_id
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init(
    exp_dir: Path,
    *,
    experiment_id: str,
    title: str,
    from_runs: list[Path] | None = None,
    target_items: int | None = None,
) -> None:
    """Initialise a new experiment manifest in *exp_dir*.

    Parameters
    ----------
    exp_dir:
        Directory to create (must not already contain EXPERIMENT.json).
    experiment_id:
        Short stable identifier (e.g. ``"sus-frontier-2026-q3"``).
    title:
        Human-readable title.
    from_runs:
        Optional list of run dirs whose contracts seed the instrument hashes
        and (when ``target_items`` is None) the target item count.
        Each run dir must contain ``RUN_CONTRACT.json``.
        ``provenance_hashes(contract)`` is called on the loaded contract dict
        directly — the function accepts full contracts as well as raw identity
        objects, routing through ``provenance_identity_from_contract`` when
        the schema version is not the identity schema.
    target_items:
        Explicit ``target.n_items`` override.  When None and *from_runs* is
        given, derived from the first seeding run's ``identity.sample_spec``.
    """
    experiment_id = validate_experiment_id(experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)

    instrument_hashes: dict[str, str] = {}
    instrument_modules: list[str] = []
    seeding_contract: dict[str, Any] | None = None

    for run_dir in (from_runs or []):
        contract_path = run_dir / "RUN_CONTRACT.json"
        contract = load_run_contract(contract_path)
        # provenance_hashes() accepts both contract dicts and raw identity
        # objects; for contracts it routes through provenance_identity_from_contract.
        hashes = provenance_hashes(contract)
        for module_entry in contract.get("modules") or []:
            module = module_entry.get("module")
            if module and isinstance(module, str):
                instrument_hashes[module] = hashes["benchmark_condition_hash"]
                if module not in instrument_modules:
                    instrument_modules.append(module)
        if seeding_contract is None:
            seeding_contract = contract

    # Determine target item count
    if target_items is not None:
        n_items = target_items
    elif seeding_contract is not None:
        n_items = _derive_target_items(seeding_contract)
    else:
        n_items = 0

    manifest: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "title": title,
        "projection_version": IDENTITY_PROJECTION_VERSION,
        "instrument": {
            "modules": instrument_modules,
            "hashes": instrument_hashes,
        },
        "conditions": [],
        "target": {"n_items": n_items},
        "members": [],
    }

    _save_manifest(exp_dir, manifest)
    _append_log(exp_dir, {
        "event": "init",
        "experiment_id": experiment_id,
        "title": title,
        "from_runs": [str(r) for r in (from_runs or [])],
    })


def adopt(
    exp_dir: Path,
    run_dir: Path,
    *,
    role: str,
) -> None:
    """Adopt a member run into the experiment.

    Adoption verifies the run's per-module ``benchmark_condition_hash`` by
    RECOMPUTING it via ``provenance_hashes`` (never trusting stored hashes)
    and raises ``InstrumentMismatch`` on any mismatch, printing the differing
    hash name (``benchmark_condition_hash``).

    ``conditions[]`` is auto-unioned with the new member's model-condition
    summaries.  The manifest is written atomically and a log entry appended.
    """
    manifest = _load_manifest(exp_dir)

    contract_path = run_dir / "RUN_CONTRACT.json"
    contract = load_run_contract(contract_path)

    # Recompute hashes for this run (never compare stored legacy hashes)
    hashes = provenance_hashes(contract)
    computed_bch = hashes["benchmark_condition_hash"]

    # Verify per-module instrument hash
    for module_entry in contract.get("modules") or []:
        module = module_entry.get("module")
        if not module:
            continue
        expected_bch = manifest["instrument"]["hashes"].get(module)
        if expected_bch is not None and computed_bch != expected_bch:
            raise InstrumentMismatch(
                f"benchmark_condition_hash mismatch for module {module!r}: "
                f"experiment has {expected_bch!r}, run recomputes to {computed_bch!r}. "
                f"Differing hash: benchmark_condition_hash"
            )

    # Canonical member fingerprint
    summary = summarize_contract(contract, contract_path=contract_path)
    contract_fingerprint = summary["contract_fingerprint"]

    # Auto-union conditions
    new_conditions = _conditions_from_hashes(hashes)
    existing_condition_ids = {c["condition_id"] for c in manifest.get("conditions") or []}
    for cond in new_conditions:
        if cond["condition_id"] not in existing_condition_ids:
            manifest.setdefault("conditions", []).append(cond)
            existing_condition_ids.add(cond["condition_id"])

    # Record member
    manifest.setdefault("members", []).append({
        "path": str(contract_path.resolve()),
        "contract_fingerprint": contract_fingerprint,
        "projection_version": IDENTITY_PROJECTION_VERSION,
        "role": role,
    })

    _save_manifest(exp_dir, manifest)
    _append_log(exp_dir, {
        "event": "adopt",
        "path": str(contract_path.resolve()),
        "role": role,
        "contract_fingerprint": contract_fingerprint,
    })


def supersede(
    exp_dir: Path,
    member_path: Path,
    *,
    by: Path,
    reason: str,
) -> None:
    """Mark *member_path* as superseded by *by* for the given *reason*.

    The member is kept in ``members[]`` but excluded from ``status`` aggregation.
    The manifest is written atomically and a log entry appended.
    """
    manifest = _load_manifest(exp_dir)

    member_path_str = str(member_path.resolve())
    by_path_str = str(by.resolve())

    found = False
    for member in manifest.get("members") or []:
        if member.get("path") == member_path_str:
            member["superseded_by"] = by_path_str
            member["reason"] = reason
            found = True
            break

    if not found:
        raise KeyError(f"Member {member_path_str!r} not found in experiment")

    _save_manifest(exp_dir, manifest)
    _append_log(exp_dir, {
        "event": "supersede",
        "member_path": member_path_str,
        "by": by_path_str,
        "reason": reason,
    })


def union(exp_dir: Path) -> dict[str, Any]:
    """Compute a per-unit winner decision across all active members.

    Returns a dict with schema_version ``benchmark-experiment-union-v1``
    containing:

    units : list[{unit_id, module, condition_id, state, chosen_member,
                  attempt, reason, candidates}]
        Every expected unit across active (non-superseded, non-moved) members
        appears exactly once.  ``chosen_member`` is the winning member's
        RUN_CONTRACT.json path; ``reason`` is ``"sole_provider"`` or
        ``"latest_started_at"`` (spec §5.2).  ``candidates`` lists every
        member that held the unit with its ``started_at``/``attempt``/``state``.
    collisions : list[{unit_id, kept_member, dropped_members}]
        Units that appeared in more than one active member.
    warnings : list[{member_path, reason, ...}]
        Moved-member entries (``file_missing`` or ``fingerprint_mismatch``).
    member_errors : list[{path, error}]
        Members whose ``owed_units`` call raised an exception.
    """
    manifest = _load_manifest(exp_dir)

    active_members = [
        (idx, m)
        for idx, m in enumerate(manifest.get("members") or [])
        if not m.get("superseded_by")
    ]

    warnings: list[dict[str, Any]] = []
    member_errors: list[dict[str, Any]] = []

    # unit_id → list of candidate dicts (one per active member providing that unit)
    unit_candidates: dict[str, list[dict[str, Any]]] = {}

    for manifest_index, member in active_members:
        member_path_str = member["path"]
        member_contract_path = Path(member_path_str)
        stored_fingerprint = member.get("contract_fingerprint")

        if not member_contract_path.exists():
            warnings.append({"member_path": member_path_str, "reason": "file_missing"})
            continue

        try:
            contract = load_run_contract(member_contract_path)
            current_summary = summarize_contract(contract, contract_path=member_contract_path)
            current_fingerprint = current_summary["contract_fingerprint"]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            warnings.append({"member_path": member_path_str, "reason": "file_missing"})
            continue

        if stored_fingerprint and current_fingerprint != stored_fingerprint:
            warnings.append({
                "member_path": member_path_str,
                "reason": "fingerprint_mismatch",
                "stored_fingerprint": stored_fingerprint,
                "current_fingerprint": current_fingerprint,
            })
            continue

        run_dir = member_contract_path.parent
        run_status = _read_run_status(run_dir)
        started_at = run_status.get("started_at")
        run_started_at = run_status.get("run_started_at")
        attempt = int(run_status.get("attempt_number") or 1)
        key_to_cid = _model_key_to_condition_id(contract)
        unit_to_cid = _unit_id_to_condition_id(contract)

        for module_entry in contract.get("modules") or []:
            module = module_entry.get("module")
            if not module:
                continue
            # Normalize at entry point so downstream dispatch handles aliases
            # ("epistemic" → "epis") without each consumer needing to know.
            canonical_module = _try_normalize_module(module)
            try:
                result = _compute_owed_units(run_dir, module=canonical_module)
            except Exception as exc:
                member_errors.append({"path": member_path_str, "error": str(exc)})
                continue

            for unit in result.get("units") or []:
                unit_id = unit.get("unit_id") or ""
                state = unit.get("state") or "owed"
                parts = unit_id.split(":")
                model_key = parts[1] if len(parts) > 1 else ""
                condition_id = unit_to_cid.get(unit_id) or key_to_cid.get(model_key, model_key)

                unit_candidates.setdefault(unit_id, []).append({
                    "member": member_path_str,
                    "started_at": started_at,
                    "run_started_at": run_started_at,
                    "manifest_index": manifest_index,
                    "attempt": attempt,
                    "state": state,
                    "condition_id": condition_id,
                    "module": canonical_module,
                })

    # Resolve winner per unit and build output
    units: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []

    for unit_id, candidates in unit_candidates.items():
        winner = _resolve_winner(candidates)
        reason = "sole_provider" if len(candidates) == 1 else "latest_started_at"
        units.append({
            "unit_id": unit_id,
            "module": winner["module"],
            "condition_id": winner["condition_id"],
            "state": winner["state"],
            "chosen_member": winner["member"],
            "attempt": winner["attempt"],
            "reason": reason,
            "candidates": [
                {
                    "member": c["member"],
                    "attempt": c["attempt"],
                    "started_at": c["started_at"],
                    "state": c["state"],
                }
                for c in candidates
            ],
        })
        if len(candidates) > 1:
            # Use object identity to exclude exactly the winning candidate,
            # so duplicate-path cases (same run adopted twice) still list dropped members.
            non_winner = [c["member"] for c in candidates if c is not winner]
            collisions.append({
                "unit_id": unit_id,
                "kept_member": winner["member"],
                "dropped_members": non_winner,
            })

    return {
        "schema_version": EXPERIMENT_UNION_SCHEMA_VERSION,
        "experiment_id": manifest.get("experiment_id"),
        "units": units,
        "collisions": collisions,
        "warnings": warnings,
        "member_errors": member_errors,
    }


def status(exp_dir: Path) -> dict[str, Any]:
    """Compute completeness per condition from all active members.

    Returned dict keys
    ------------------
    schema_version, experiment_id
        Manifest metadata.
    completeness : list[{condition_id, module, done, owed, terminal}]
        One entry per (condition_id, module) pair across non-superseded,
        non-moved members.  ``terminal`` counts units classified
        ``terminal_model_signal`` (never re-execute); they are excluded from
        ``owed``.  Duplicate units (same unit_id in multiple members) resolve
        with "newest non-superseded member wins" (last in manifest order).
    collisions : list[{unit_id, kept_member, dropped_members}]
        Every unit_id that appeared in more than one active member.
    warnings : list[{member_path, reason, ...}]
        Moved-member entries.  ``reason`` is one of:
        - ``"file_missing"`` — contract file absent or unreadable.
        - ``"fingerprint_mismatch"`` — file exists but the recomputed
          ``summarize_contract(...)["contract_fingerprint"]`` differs from the
          value stored at adopt time.
        Moved members are excluded from unit aggregation.
    member_errors : list[{path, error}]
        Members whose ``owed_units`` call raised an exception (corrupt
        contract, unknown module, etc.).  These members are skipped for
        unit aggregation but not treated as moved.
    instrument, conditions, target
        Forwarded from the manifest.
    """
    manifest = _load_manifest(exp_dir)

    # Collect active (non-superseded) members in manifest order
    active_members = [
        (idx, m)
        for idx, m in enumerate(manifest.get("members") or [])
        if not m.get("superseded_by")
    ]

    warnings: list[dict[str, Any]] = []
    member_errors: list[dict[str, Any]] = []

    # unit_id → list of candidate dicts (for _resolve_winner; spec §5.2)
    unit_candidates: dict[str, list[dict[str, Any]]] = {}

    for manifest_index, member in active_members:
        member_path_str = member["path"]
        member_contract_path = Path(member_path_str)
        stored_fingerprint = member.get("contract_fingerprint")

        # --- Moved-member check 1: file absent ---
        if not member_contract_path.exists():
            warnings.append({
                "member_path": member_path_str,
                "reason": "file_missing",
            })
            continue

        # --- Load contract and verify fingerprint ---
        try:
            contract = load_run_contract(member_contract_path)
            current_summary = summarize_contract(contract, contract_path=member_contract_path)
            current_fingerprint = current_summary["contract_fingerprint"]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            warnings.append({
                "member_path": member_path_str,
                "reason": "file_missing",
            })
            continue

        # --- Moved-member check 2: fingerprint mismatch ---
        if stored_fingerprint and current_fingerprint != stored_fingerprint:
            warnings.append({
                "member_path": member_path_str,
                "reason": "fingerprint_mismatch",
                "stored_fingerprint": stored_fingerprint,
                "current_fingerprint": current_fingerprint,
            })
            continue

        run_dir = member_contract_path.parent
        run_status = _read_run_status(run_dir)
        key_to_cid = _model_key_to_condition_id(contract)
        unit_to_cid = _unit_id_to_condition_id(contract)

        for module_entry in contract.get("modules") or []:
            module = module_entry.get("module")
            if not module:
                continue
            # Normalize at entry point so downstream dispatch handles aliases.
            canonical_module = _try_normalize_module(module)
            try:
                result = _compute_owed_units(run_dir, module=canonical_module)
            except Exception as exc:
                member_errors.append({"path": member_path_str, "error": str(exc)})
                continue

            for unit in result.get("units") or []:
                unit_id = unit.get("unit_id") or ""
                state = unit.get("state") or "owed"

                # Extract model_key from unit_id (format: module:model_key:...)
                parts = unit_id.split(":")
                model_key = parts[1] if len(parts) > 1 else ""
                condition_id = unit_to_cid.get(unit_id) or key_to_cid.get(model_key, model_key)

                unit_candidates.setdefault(unit_id, []).append({
                    "member": member_path_str,
                    "started_at": run_status.get("started_at"),
                    "run_started_at": run_status.get("run_started_at"),
                    "manifest_index": manifest_index,
                    "attempt": int(run_status.get("attempt_number") or 1),
                    "state": state,
                    "condition_id": condition_id,
                    "module": canonical_module,
                })

    # Resolve winner per unit using latest-started_at rule (spec §5.2).
    # Winners are computed once per unit and reused for both unit_states and
    # collisions — no duplicate _resolve_winner calls.
    unit_winners: dict[str, dict[str, Any]] = {
        unit_id: _resolve_winner(candidates)
        for unit_id, candidates in unit_candidates.items()
    }

    unit_states: dict[str, dict[str, Any]] = {
        unit_id: {
            "state": winner["state"],
            "condition_id": winner["condition_id"],
            "module": winner["module"],
            "kept_member": winner["member"],
        }
        for unit_id, winner in unit_winners.items()
    }

    # Build collisions: unit_ids seen in more than one active member
    collisions: list[dict[str, Any]] = []
    for unit_id, candidates in unit_candidates.items():
        if len(candidates) > 1:
            winner = unit_winners[unit_id]
            # Use object identity to handle same-path duplicate adopts correctly
            dropped = [c["member"] for c in candidates if c is not winner]
            collisions.append({
                "unit_id": unit_id,
                "kept_member": winner["member"],
                "dropped_members": dropped,
            })

    # Aggregate done/owed/terminal counts per (condition_id, module).
    # terminal_model_signal units are excluded from owed — they will never
    # be re-executed and must not inflate the remaining-work count.
    condition_module_counts: dict[tuple[str, str], dict[str, int]] = {}
    for info in unit_states.values():
        cid = info["condition_id"]
        mod = info["module"]
        key = (cid, mod)
        if key not in condition_module_counts:
            condition_module_counts[key] = {"done": 0, "owed": 0, "terminal": 0}
        state = info["state"]
        if state == "done":
            condition_module_counts[key]["done"] += 1
        elif state == "terminal_model_signal":
            condition_module_counts[key]["terminal"] += 1
        else:
            condition_module_counts[key]["owed"] += 1

    completeness = [
        {
            "condition_id": cid,
            "module": mod,
            "done": counts["done"],
            "owed": counts["owed"],
            "terminal": counts["terminal"],
        }
        for (cid, mod), counts in condition_module_counts.items()
    ]

    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": manifest.get("experiment_id"),
        "completeness": completeness,
        "collisions": collisions,
        "warnings": warnings,
        "member_errors": member_errors,
        "instrument": manifest.get("instrument"),
        "conditions": manifest.get("conditions"),
        "target": manifest.get("target"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m suite_tools.experiment",
        description="Experiment manifest management.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    # --- init ---
    p_init = sub.add_parser("init", help="Initialise a new experiment manifest.")
    p_init.add_argument("dir", type=Path, help="Experiment directory to create.")
    p_init.add_argument("--id", dest="experiment_id", required=True)
    p_init.add_argument("--title", required=True)
    p_init.add_argument("--from-run", dest="from_runs", action="append", type=Path, default=[])
    p_init.add_argument("--target-items", dest="target_items", type=int, default=None)

    # --- adopt ---
    p_adopt = sub.add_parser("adopt", help="Adopt a member run.")
    p_adopt.add_argument("dir", type=Path)
    p_adopt.add_argument("run_dir", type=Path)
    p_adopt.add_argument("--role", required=True, choices=["pilot", "expansion", "repair"])

    # --- supersede ---
    p_sup = sub.add_parser("supersede", help="Mark a member as superseded.")
    p_sup.add_argument("dir", type=Path)
    p_sup.add_argument("member_path", type=Path)
    p_sup.add_argument("--by", type=Path, required=True)
    p_sup.add_argument("--reason", required=True)

    # --- status ---
    p_status = sub.add_parser("status", help="Print completeness per condition.")
    p_status.add_argument("dir", type=Path)
    p_status.add_argument("--json", dest="output_json", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.verb == "init":
            init(
                args.dir,
                experiment_id=args.experiment_id,
                title=args.title,
                from_runs=args.from_runs or None,
                target_items=args.target_items,
            )
            print(f"Initialised experiment {args.experiment_id!r} in {args.dir}")

        elif args.verb == "adopt":
            adopt(args.dir, args.run_dir, role=args.role)
            print(f"Adopted {args.run_dir} as {args.role!r}")

        elif args.verb == "supersede":
            supersede(args.dir, args.member_path, by=args.by, reason=args.reason)
            print(f"Superseded {args.member_path}")

        elif args.verb == "status":
            st = status(args.dir)
            if args.output_json:
                print(json.dumps(st, indent=2))
            else:
                print(f"Experiment: {st['experiment_id']}")
                for row in st.get("completeness") or []:
                    print(
                        f"  {row['condition_id']} ({row.get('module', '')}): "
                        f"done={row['done']} owed={row['owed']} terminal={row.get('terminal', 0)}"
                    )
                if st.get("collisions"):
                    print(f"  collisions: {len(st['collisions'])}")
                if st.get("warnings"):
                    for w in st["warnings"]:
                        print(f"  WARNING ({w.get('reason')}): {w.get('member_path')}")
                if st.get("member_errors"):
                    for e in st["member_errors"]:
                        print(f"  MEMBER_ERROR: {e.get('path')}: {e.get('error')}")

    except InstrumentMismatch as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
