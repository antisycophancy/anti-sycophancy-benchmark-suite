"""Registry commands for the benchmark harness.

Verbs answer spec §6's surface (runs / experiments / status / blockers /
adopt / supersede / verify / package).  Each library function returns a plain
dict carrying ``schema_version: SCHEMA_VERSION``; the CLI renders it.

CLI::

    python3 -m suite_tools.bench <verb> [args] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from suite_tools.live_dashboard import (
    DISPOSITION_FILENAME,
    DISPOSITION_SCHEMA_VERSION,
)
from suite_tools.owed_units import owed_units as _owed_units
from suite_tools.run_contract import (
    CONTRACT_FILENAME,
    IDENTITY_PROJECTION_VERSION,
    LEGACY_IDENTITY_PROJECTION_VERSION,
    compare_provenance,
    provenance_hashes,
    provenance_hashes_for_version,
    provenance_identity_from_contract,
)
from suite_tools.suite_registry import FIRST_PARTY_SUITES, REPO_ROOT
from suite_tools import experiment as _experiment
from suite_tools.assert_hash_panel import item_universe_report
from suite_tools.call_diagnostics import diagnose_call_journal

SCHEMA_VERSION = "benchmark-registry-view-v1"
MAX_SCAN_DEPTH = 6

# ``instrument_defect`` is publication-blocking and must NOT be suppressed from
# the blockers list.  The single authoritative predicate is
# ``review_projection.is_publication_blocking``; it is the only gate logic
# that should be consulted here.  (Finding 7: the old local copy included
# ``instrument_defect`` incorrectly.)
from suite_tools.review_projection import is_publication_blocking as _is_publication_blocking  # noqa: E402


# ---------------------------------------------------------------------------
# Default roots
# ---------------------------------------------------------------------------


def default_roots() -> list[Path]:
    """Return deduplicated default scan roots.

    Includes each first-party suite's ``results_root`` (``<suite>/results/``)
    plus the top-level ``REPO_ROOT/results/``.  De-duplicated by resolved path
    so a symlinked or coincident directory is only visited once.
    """
    seen: set[Path] = set()
    roots: list[Path] = []

    candidates: list[Path] = [suite.results_root for suite in FIRST_PARTY_SUITES]
    candidates.append(REPO_ROOT / "results")

    for root in candidates:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            roots.append(root)

    return roots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json_safe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON file.  Returns ``(data, None)`` or ``(None, error_str)``."""
    try:
        raw = path.read_text()
        data = json.loads(raw)
        return (data if isinstance(data, dict) else {}), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _load_disposition(output_dir: Path) -> dict[str, Any]:
    """Mirror the canonical ``_load_disposition`` from live_dashboard.py.

    Returns an empty dict when the file is absent, unreadable, or carries a
    schema version other than ``DISPOSITION_SCHEMA_VERSION``.
    """
    path = output_dir / DISPOSITION_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if data.get("schema_version") != DISPOSITION_SCHEMA_VERSION:
        return {}
    return data


def _is_rejected(disposition: dict[str, Any]) -> bool:
    return disposition.get("disposition") == "rejected_from_analysis"


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL event file.  Skips malformed lines silently."""
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    events.append(event)
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return events


def _scan_runs(roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk *roots* and collect run directories.

    Operational contract
    --------------------
    * **Deterministic**: entries sorted at each level (``sorted(Path.iterdir())``).
    * **Pruned**: dirs whose name starts with ``_archive*`` or that are symlinks are
      skipped (``is_symlink()`` — symlinks are never followed).
    * **Bounded depth**: at most ``MAX_SCAN_DEPTH`` levels below each root.
    * **Rejection**: a directory is skipped when its ``RUN_DISPOSITION.json`` has
      ``disposition == "rejected_from_analysis"`` and the matching schema version.
    * **Per-path error capture**: malformed JSON, permission errors, and unreadable
      contracts append ``{path, error}`` to the ``scan_warnings`` list instead of
      aborting.

    Returns ``(runs, scan_warnings)``.
    """
    runs: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > MAX_SCAN_DEPTH:
            return
        try:
            entries = sorted(directory.iterdir())
        except (PermissionError, OSError) as exc:
            warnings.append({"path": str(directory), "error": str(exc)})
            return

        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.name.startswith("_archive"):
                continue
            if not entry.is_dir():
                continue

            contract_path = entry / CONTRACT_FILENAME
            if contract_path.exists():
                # This is a run directory — process it (do not recurse further).
                data, error = _load_json_safe(contract_path)
                if error is not None:
                    warnings.append({"path": str(entry), "error": error})
                    continue

                disposition = _load_disposition(entry)
                if _is_rejected(disposition):
                    continue

                run_id = (data or {}).get("run_id") or entry.name
                runs.append({
                    "run_id": run_id,
                    "path": str(entry),
                    "modules": [
                        m.get("module")
                        for m in ((data or {}).get("modules") or [])
                        if isinstance(m, dict)
                    ],
                    "schema_version": (data or {}).get("schema_version"),
                })
            else:
                # Not a run directory — recurse deeper.
                _walk(entry, depth + 1)

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        _walk(root, 1)

    return runs, warnings


# ---------------------------------------------------------------------------
# Public library functions
# ---------------------------------------------------------------------------


def runs(
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Deterministic scan of *roots* returning all non-rejected run directories.

    Parameters
    ----------
    roots:
        Directories to scan.  Defaults to ``default_roots()``.

    Returns
    -------
    dict with ``schema_version``, ``runs``, ``scan_warnings``.
    """
    if roots is None:
        roots = default_roots()
    run_list, scan_warnings = _scan_runs(roots)
    return {
        "schema_version": SCHEMA_VERSION,
        "runs": run_list,
        "scan_warnings": scan_warnings,
    }


def experiments(
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Find ``EXPERIMENT.json`` directories and return ``experiment.status()`` for each.

    Parameters
    ----------
    roots:
        Directories to scan for experiments.  Defaults to ``default_roots()``.
    """
    if roots is None:
        roots = default_roots()

    found: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        # Collect and sort by path string for deterministic ordering, then
        # prune paths whose ancestor components are _archive* dirs or symlinks
        # (mirrors the _scan_runs pruning contract).
        candidates = sorted(root.rglob("EXPERIMENT.json"), key=lambda p: str(p))
        for exp_json in candidates:
            exp_dir = exp_json.parent
            # Prune: skip if any ancestor between root and exp_dir is a symlink
            # or has a name starting with "_archive".
            pruned = False
            try:
                rel_parts = exp_dir.relative_to(root).parts
            except ValueError:
                rel_parts = ()
            cumpath = root
            for part in rel_parts:
                cumpath = cumpath / part
                if part.startswith("_archive"):
                    pruned = True
                    break
                if cumpath.is_symlink():
                    pruned = True
                    break
            if pruned:
                continue
            try:
                exp_status = _experiment.status(exp_dir)
                found.append(exp_status)
            except Exception as exc:
                warnings.append({"path": str(exp_dir), "error": str(exc)})

    return {
        "schema_version": SCHEMA_VERSION,
        "experiments": found,
        "scan_warnings": warnings,
    }


def diagnose(run_dir: Path | str) -> dict[str, Any]:
    """Read local provider-call diagnostics without mutating the run."""
    return diagnose_call_journal(Path(run_dir))


def status(run_dir: Path | str) -> dict[str, Any]:
    """Alias for :func:`owed_units.owed_units` — owed units before spend.

    This is the implementation of ``bench status <run>`` from spec §6.
    """
    return _owed_units(run_dir)


def blockers(
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Return every run whose **latest attempt** emitted an ``action=halt`` event.

    Halt detection reads ``RUN_STATUS.json`` (for the current attempt number)
    and ``RUN_EVENTS.jsonl`` (for ``attempt_failure_classified`` events).  A halt
    on a superseded earlier attempt that a later attempt resolved is not a blocker.

    Returns
    -------
    dict with ``schema_version``, ``blockers``, ``scan_warnings``.
    Each blocker entry carries: ``run_id``, ``path``, ``action``, ``category``,
    ``failure_reason``, ``evidence_class``, ``attempt_number``.
    """
    if roots is None:
        roots = default_roots()

    run_list, scan_warnings = _scan_runs(roots)
    blocker_list: list[dict[str, Any]] = []

    for run in run_list:
        run_path = Path(run["path"])

        # Current attempt number comes from RUN_STATUS.json
        status_path = run_path / "RUN_STATUS.json"
        if not status_path.exists():
            continue
        try:
            status_data = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(status_data, dict):
            continue

        current_attempt = status_data.get("attempt_number", 1)
        current_stage = status_data.get("stage")

        # Consume the projection's event facts so the suppression check
        # (_QUEUE_CLEARING_DISPOSITIONS) can use the FactView's disposition.
        # Only an explicit safety_declination adjudication removes a halt
        # from the list; instrument_defect and bare model_signal halts stay
        # visible because they are still active concerns for the operator.
        # Guarded so a malformed review file can never *hide* a real blocker:
        # on any projection error we fall back to the raw scan (no fv → never
        # suppress any event, fail-closed).
        try:
            from suite_tools.review_projection import project as _project  # noqa: PLC0415
            # Carry the full FactView so the suppression check has disposition.
            candidates: list[tuple[dict[str, Any], Any]] = [
                (fv.fact, fv)
                for fv in _project(run_path).events_by_ref.values()
                if fv.source == "events"
            ]
        except Exception:
            candidates = [
                (event, None)
                for event in _load_events(run_path / "RUN_EVENTS.jsonl")
            ]

        for event, fv in candidates:
            if (
                event.get("event") == "attempt_failure_classified"
                and event.get("action") == "halt"
                and event.get("attempt_number") == current_attempt
                and (
                    not current_stage
                    or not event.get("stage")
                    or event.get("stage") == current_stage
                )
            ):
                # Suppress only if the fact has been explicitly adjudicated away
                # (safety_declination is the only queue-clearing disposition).
                # Note: is_publication_blocking is NOT the right predicate here
                # because a bare model_signal halt is not publication-blocking
                # but IS still an active concern that should appear in blockers.
                # instrument_defect is publication-blocking and stays visible;
                # model_signal with no review is an active concern and stays visible.
                # fv is None in the fallback path → never suppress (fail-closed).
                if fv is not None and fv.disposition in _QUEUE_CLEARING_DISPOSITIONS:
                    continue  # adjudicated away (safety_declination)
                blocker_list.append({
                    "run_id": run["run_id"],
                    "path": run["path"],
                    "action": event["action"],
                    "category": event.get("category", ""),
                    "failure_reason": event.get("failure_reason", ""),
                    "evidence_class": event.get("evidence_class", ""),
                    "attempt_number": current_attempt,
                })
                break  # Only report the first halt event per run

    return {
        "schema_version": SCHEMA_VERSION,
        "blockers": blocker_list,
        "scan_warnings": scan_warnings,
    }


def adopt(
    exp_dir: Path | str,
    run_dir: Path | str,
    *,
    role: str,
) -> dict[str, Any]:
    """Passthrough to :func:`experiment.adopt`.

    Adopts *run_dir* into the experiment at *exp_dir* with the given *role*.
    """
    _experiment.adopt(Path(exp_dir), Path(run_dir), role=role)
    return {"schema_version": SCHEMA_VERSION, "result": "ok"}


def supersede(
    exp_dir: Path | str,
    member_path: Path | str,
    *,
    by: Path | str,
    reason: str,
) -> dict[str, Any]:
    """Passthrough to :func:`experiment.supersede`.

    Marks *member_path* as superseded by *by* for *reason*.
    """
    _experiment.supersede(
        Path(exp_dir),
        Path(member_path),
        by=Path(by),
        reason=reason,
    )
    return {"schema_version": SCHEMA_VERSION, "result": "ok"}


def verify_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    """Audit an emitted bundle tree for privacy and payload integrity.

    Walks the filesystem (not ``git ls-files``), so it inspects gitignored bundles
    that ``release_audit`` never sees.  Current bundles must carry a complete
    SHA-256 inventory for every non-report payload file.
    """
    from suite_tools import bundle as _bundle

    privacy_issues = _bundle.audit_bundle_tree(bundle_dir)
    integrity_issues = _bundle.audit_bundle_integrity(bundle_dir)
    provenance_issues = _bundle.audit_bundle_provenance(bundle_dir)
    issues = privacy_issues + integrity_issues + provenance_issues
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle": str(bundle_dir),
        "issues": [{"path": issue.path, "reason": issue.reason} for issue in issues],
        "privacy_clean": not privacy_issues,
        "integrity_clean": not integrity_issues,
        "provenance_clean": not provenance_issues,
        "clean": not issues,
    }


def verify(
    dirs: list[Path | str],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Verify one or two run directories.

    One directory
        Recompute ``provenance_hashes`` and compare to any stored ``provenance``
        block in the contract; report drift.

    Two directories
        Run :func:`run_contract.compare_provenance` for the hash certificate
        **plus** the new :func:`assert_hash_panel.item_universe_report` for item
        universe equality.  Does **not** call ``assert_comparison_identity``.

    Parameters
    ----------
    dirs:
        List of 1 or 2 run directory paths.
    strict:
        When ``True`` and two dirs are given, exit with code 1 if the runs are
        not comparable.
    """
    dirs = [Path(d) for d in dirs]

    if len(dirs) == 1:
        return _verify_one(dirs[0])
    elif len(dirs) == 2:
        return _verify_two(dirs[0], dirs[1], strict=strict)
    else:
        raise ValueError(f"verify requires 1 or 2 directories, got {len(dirs)}")


def _load_contract(run_dir: Path) -> dict[str, Any]:
    """Load the contract from a run directory; raise on failure."""
    contract_path = run_dir / CONTRACT_FILENAME
    data = json.loads(contract_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Contract at {contract_path} is not a JSON object")
    return data


def _verify_one(run_dir: Path) -> dict[str, Any]:
    """Recompute provenance hashes and report drift vs stored contract values."""
    try:
        contract = _load_contract(run_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"schema_version": SCHEMA_VERSION, "error": str(exc)}

    identity = provenance_identity_from_contract(contract)
    current_projection = provenance_hashes(identity)
    stored = contract.get("provenance") or {}
    from suite_tools.request_receipts import evaluate_request_conformance
    from suite_tools.artifact_identity import evaluate_run_artifact_identity

    request_conformance = evaluate_request_conformance(run_dir)
    artifact_identity = evaluate_run_artifact_identity(run_dir, contract=contract)

    stored_projection_version = stored.get("projection_version")
    if stored_projection_version is None:
        stored_projection_version = LEGACY_IDENTITY_PROJECTION_VERSION
    try:
        recomputed_for_stored_version = provenance_hashes_for_version(
            identity,
            str(stored_projection_version),
        )
    except ValueError:
        recomputed_for_stored_version = {}

    drift: dict[str, dict[str, Any]] = {}
    all_keys = set(recomputed_for_stored_version) | set(stored)
    for key in sorted(all_keys):
        r = recomputed_for_stored_version.get(key)
        s = stored.get(key)
        if r != s:
            drift[key] = {"recomputed": r, "stored": s}

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": contract.get("run_id"),
        "stored_projection_version": stored_projection_version,
        "verification_projection_supported": bool(recomputed_for_stored_version),
        "drift": drift,
        "current_projection": current_projection,
        "request_conformance": request_conformance,
        "artifact_identity": artifact_identity,
        "clean": (
            bool(recomputed_for_stored_version)
            and len(drift) == 0
            and request_conformance["conformant"]
            and artifact_identity["conformant"]
        ),
    }


def _verify_two(dir_a: Path, dir_b: Path, *, strict: bool) -> dict[str, Any]:
    """Compare two run dirs via hash certificate + item universe report."""
    try:
        contract_a = _load_contract(dir_a)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"schema_version": SCHEMA_VERSION, "error": f"Could not load {dir_a}: {exc}"}
    try:
        contract_b = _load_contract(dir_b)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"schema_version": SCHEMA_VERSION, "error": f"Could not load {dir_b}: {exc}"}

    hash_cert = compare_provenance(contract_a, contract_b)
    item_univ = item_universe_report(contract_a, contract_b)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hash_certificate": hash_cert,
        "item_universe": item_univ,
    }

    if strict and not hash_cert.get("comparable"):
        sys.exit(1)

    return result


def package(
    exp_dir: Path | str,
    out_dir: Path | str,
    *,
    include_transcripts: bool = False,
    include_review_rationale: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    """Emit a self-contained experiment bundle (delegates to ``bundle.emit``)."""
    from suite_tools import bundle as _bundle

    return _bundle.emit(
        exp_dir,
        out_dir,
        include_transcripts=include_transcripts,
        include_review_rationale=include_review_rationale,
        write_report=write_report,
    )


# ---------------------------------------------------------------------------
# review (T6 — bench review verb)
# ---------------------------------------------------------------------------

_REVIEW_SOURCES_FILENAMES = {"blocks": "BLOCKS.jsonl", "events": "RUN_EVENTS.jsonl"}

# The ONLY disposition that clears a fact from the review queue, blockers,
# and the publication gate simultaneously.  ``instrument_defect`` is
# publication-blocking and must NOT be in this set — it stays visible in the
# queue and in blockers until the defect is corrected and the run is re-run.
# (Finding 7: ``RESOLVING_DISPOSITIONS`` from review_projection includes
# ``instrument_defect`` for gate-clause-a logic; this is the separate,
# narrower concept of "human adjudication that fully clears the fact".)
_QUEUE_CLEARING_DISPOSITIONS = frozenset({"safety_declination"})

# Unit states the bundle gate accepts as publishable (mirrors bundle.py).
# Any other UnitView state (owed, pending_retry, instrument_defect, unresolved)
# blocks publication at the unit level (F6: gate_blocking must capture this).
_PUBLISHABLE_UNIT_STATES = frozenset({"completed", "terminal_model_signal"})


def _is_triage_resolved(fv: Any) -> bool:
    """True iff the fact has been conclusively cleared by a safety_declination.

    ``instrument_defect`` is publication-blocking (finding 7) and therefore
    NOT triage-resolved — it remains in the default queue and in blockers.
    """
    return fv.disposition in _QUEUE_CLEARING_DISPOSITIONS


def _disposition_status(fv: Any) -> str:
    """Human-readable disposition_status string for a review queue row.

    Values: ``unreviewed`` | ``needs_escalation`` | ``pending_retry`` |
    ``instrument_defect`` | ``resolved`` (safety_declination only — the sole
    disposition that fully clears a fact from the gate and the queue).
    """
    d = fv.disposition
    if d is None:
        return "unreviewed"
    if d == "needs_escalation":
        return "needs_escalation"
    if d == "retry":
        return "pending_retry"
    if d in _QUEUE_CLEARING_DISPOSITIONS:
        return "resolved"
    # ``instrument_defect`` and any future forward-compatible dispositions pass
    # through so the caller sees the actual disposition name.
    return d


def _compute_gate_blocking(
    fv: Any,
    proj: Any,  # ProjectionResult
) -> tuple[bool, str | None]:
    """Return ``(gate_blocking, gate_reason)`` for a review queue row.

    Three layers are checked in priority order; the first firing layer names
    the reason so the operator knows which gate clause will actually refuse
    publication:

    * ``"fact"``   — :func:`is_publication_blocking` fires (needs_escalation,
                     unknown-no-resolving-review, instrument_defect)
    * ``"unit"``   — the fact's unit has a non-publishable UnitView state
                     (``owed``, ``pending_retry``, ``instrument_defect``,
                     ``unresolved``)
    * ``"member"`` — the run carries an unfulfilled member-level retry
                     obligation that would gate the whole member regardless of
                     individual unit state

    ``gate_reason`` is ``None`` when ``gate_blocking`` is False.

    Design note: ``retry`` discharge is attempt-aware and is evaluated via the
    UnitView (layer 2), not at the FactView level (layer 1).  This ensures a
    discharged retry correctly yields ``gate_blocking=False`` (because the
    unit's state will be ``completed`` after discharge), while an undischarged
    retry correctly yields ``gate_blocking=True, gate_reason="unit"``.
    """
    # Layer 1: fact-level predicate (is_publication_blocking)
    if _is_publication_blocking(fv) is not None:
        return True, "fact"

    # Layer 2: unit-level state for unit-scoped facts
    if fv.unit_id is not None:
        uv = proj.units_by_id.get(fv.unit_id)
        if uv is not None and uv.state not in _PUBLISHABLE_UNIT_STATES:
            return True, "unit"

    # Layer 3: member-level obligation (any unfulfilled → whole member blocked)
    for obligation in proj.member_obligations:
        if not obligation.fulfilled:
            return True, "member"

    return False, None


def _make_review_row(
    *,
    run_id: str,
    run_path: str,
    fv: Any,   # FactView from review_projection
    proj: Any,  # ProjectionResult — needed for unit/member gate check (F6)
) -> dict[str, Any]:
    """Build one review queue row from a FactView.  Raw body excerpt is a
    LOCAL-only field (plan D2) — it stays in the row; it never enters bundles."""
    fact = fv.fact
    gate_blocking, gate_reason = _compute_gate_blocking(fv, proj)
    return {
        "run_id": run_id,
        "run_path": run_path,
        "scope": fv.scope,
        "unit_id": fv.unit_id,
        "event_ref": fv.event_ref,
        "source": fv.source,
        "evidence_class": fv.effective_class,
        "category": fv.effective_category,
        "attempt_number": fv.attempt_number,
        "provider": fact.get("provider"),
        "provider_code": fact.get("provider_code"),
        "native_finish_reason": fact.get("native_finish_reason"),
        "signal_source": fact.get("signal_source"),
        "billed_attempts": fact.get("billed_attempts"),
        "raw_body_excerpt": fact.get("raw_body_excerpt"),
        "evidence_pointer": run_path + "/" + _REVIEW_SOURCES_FILENAMES.get(fv.source, fv.source),
        "disposition_status": _disposition_status(fv),
        "resolution_status": fv.resolution_status,
        # gate_blocking + gate_reason use _compute_gate_blocking which checks
        # all three layers (fact, unit, member) so this field cannot drift from
        # what the bundle gate will actually refuse (F6 fix).
        "gate_blocking": gate_blocking,
        "gate_reason": gate_reason,
        "active_review": fv.active_review,
    }


def review(
    # LIST mode
    roots: list[Path] | None = None,
    *,
    include_resolved: bool = False,
    evidence_class: str | None = None,
    scope: str | None = None,
    # Shared (restrict to a single run in list mode, or target in disposition mode)
    run_dir: Path | str | None = None,
    # DISPOSITION mode — all of these together trigger disposition mode
    event_ref: str | None = None,
    disposition: str | None = None,
    reviewer: str | None = None,
    reason: str | None = None,
    resolved_category: str | None = None,
    issue_ref: str | None = None,
    supersede: str | None = None,
) -> dict[str, Any]:
    """Human triage workflow for the review queue (plan 020 §3 T6 + D4/D5/D11).

    LIST mode (no ``event_ref``)
    ----------------------------
    Scans *roots* (default ``default_roots()``) for run directories, calls the
    projection on each, and emits rows for unresolved facts from both
    ``BLOCKS.jsonl`` (blocks source) and ``attempt_failure_classified`` events
    from ``RUN_EVENTS.jsonl`` (events source).  Resolved facts are hidden by
    default; pass ``include_resolved=True`` to include them.

    DISPOSITION mode (``event_ref`` supplied together with ``disposition``,
    ``reviewer``, ``reason``)
    -----------------------------------------------------------------------
    Validates the review via T5's :func:`review_projection.append_review`,
    appends it atomically under the ``O_EXCL`` lock, then re-projects the run
    so the operator immediately sees the consequence.  T5's validation errors
    (unmappable retry, missing resolved_category, duplicate active head) are
    surfaced as-is — no re-encoding of T5 rules here (D11).
    """
    from suite_tools import review_projection as _rp  # noqa: PLC0415

    # ---- DISPOSITION mode --------------------------------------------------
    if event_ref is not None:
        if run_dir is None:
            raise ValueError("--run / run_dir is required in disposition mode")
        if disposition is None or reviewer is None or reason is None:
            raise ValueError(
                "disposition, reviewer, and reason are all required in disposition mode"
            )
        run_path = Path(run_dir)

        # Normalize a line-form event_ref to id-form before storing the review.
        # The projection's _reviews_by_ref groups reviews by their literal
        # event_ref; the projection keys facts by their canonical id-form ref
        # (blocks-id:<block_id> / events-id:<event_id>) when a v2 id field is
        # present.  If we store the review with a line-form ref and the fact has
        # a block_id/event_id, the two never match → the review is invisible to
        # the projection and the echo shows the wrong state.  Normalize here so
        # the stored ref always matches the fact's canonical key.
        resolved = None
        try:
            resolved = _rp.resolve_event_ref(run_path, event_ref)
            id_field = "block_id" if resolved.source == "blocks" else "event_id"
            id_value = resolved.record.get(id_field)
            stored_ref: str = (
                f"{resolved.source}-id:{id_value}" if id_value else event_ref
            )
        except Exception:
            stored_ref = event_ref  # let append_review surface any ref error

        from datetime import datetime, timezone  # noqa: PLC0415
        record: dict[str, Any] = {
            "schema_version": _rp.REVIEW_SCHEMA_VERSION,
            "event_ref": stored_ref,
            "disposition": disposition,
            "reviewer": reviewer,
            "rationale": reason,
            # D4-required fields the CLI must stamp (final-gate F4c): reviewed_at
            # is the append time; module/model/category come from the target fact
            # so the stored review always names the fact's own identity.
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        if resolved is not None:
            for key in ("module", "model", "category"):
                value = resolved.record.get(key)
                if value is not None:
                    record[key] = value
        if resolved_category is not None:
            record["resolved_category"] = resolved_category
        if issue_ref is not None:
            record["issue_ref"] = issue_ref
        if supersede is not None:
            record["supersedes_review_id"] = supersede

        # append_review validates, stamps review_id/schema_version, and acquires
        # the O_EXCL lock; T5's errors propagate unchanged.
        written = _rp.append_review(run_path, record)

        # The canonical ref is the one actually stored (id-form when available),
        # so the projection lookup always hits the right FactView.
        canonical_ref = stored_ref

        # Re-project to show the operator the immediate consequence.
        proj = _rp.project(run_path)
        fv = proj.events_by_ref.get(canonical_ref)
        new_state: dict[str, Any]
        if fv is not None:
            new_state = {
                "event_ref": canonical_ref,
                "resolution_status": fv.resolution_status,
                "effective_class": fv.effective_class,
                "effective_category": fv.effective_category,
                "disposition": fv.disposition,
            }
        else:
            new_state = {"event_ref": canonical_ref, "resolution_status": "unknown"}

        return {
            "schema_version": SCHEMA_VERSION,
            "written_review": written,
            "new_effective_state": new_state,
        }

    # ---- LIST mode ---------------------------------------------------------
    # Determine scan roots.  If both roots and run_dir are given, roots takes
    # precedence for the scan and run_dir acts as a post-scan filter.  If only
    # run_dir is given, we treat it as a single run (no full tree scan needed).
    filter_path: Path | None = None
    if run_dir is not None:
        filter_path = Path(run_dir).resolve()

    if roots is not None:
        run_list, scan_warnings = _scan_runs(roots)
    elif run_dir is not None:
        # Single-run shortcut: build a minimal run_list entry without a full
        # directory scan so --run works even when the run is not under a
        # standard results root.
        run_path = Path(run_dir)
        run_list, scan_warnings = _scan_runs([run_path.parent])
    else:
        run_list, scan_warnings = _scan_runs(default_roots())

    # Apply --run filter after scan.
    if filter_path is not None:
        run_list = [r for r in run_list if Path(r["path"]).resolve() == filter_path]

    rows: list[dict[str, Any]] = []

    for run_info in run_list:
        rpath = Path(run_info["path"])
        try:
            proj = _rp.project(rpath)
        except Exception:
            scan_warnings.append({
                "path": str(rpath),
                "error": "projection failed — skipping",
            })
            continue

        for ref, fv in proj.events_by_ref.items():
            # Apply scope filter.  The projection uses "unmappable_legacy" as the
            # canonical name; the CLI flag accepts the shorter "unmappable" alias.
            if scope is not None:
                canonical_scope = fv.scope
                filter_scope = "unmappable_legacy" if scope == "unmappable" else scope
                if canonical_scope != filter_scope:
                    continue
            # Apply evidence_class filter (use underlying class from the fact record
            # so filter works before disposition-driven reclassification too).
            raw_class = fv.fact.get("evidence_class")
            eff_class = fv.effective_class
            if evidence_class is not None:
                if raw_class != evidence_class and eff_class != evidence_class:
                    continue

            # Determine triage resolution: a fact is hidden from the default
            # queue only when a human has explicitly applied a resolving review
            # (safety_declination or instrument_defect).  Known-class facts
            # with NO review are still "unreviewed" — a human hasn't looked at
            # them — so they appear in the queue.  This differs intentionally
            # from the projection's own resolution_status, which marks any
            # non-unknown-class fact as "resolved" for gate purposes.
            if _is_triage_resolved(fv) and not include_resolved:
                continue

            row = _make_review_row(
                run_id=run_info["run_id"],
                run_path=run_info["path"],
                fv=fv,
                proj=proj,
            )
            rows.append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "scan_warnings": scan_warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(result: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        for key, value in result.items():
            if key == "schema_version":
                continue
            if isinstance(value, list):
                print(f"{key}: ({len(value)} items)")
                for item in value[:10]:
                    print(f"  {json.dumps(item, default=str)}")
                if len(value) > 10:
                    print(f"  ... and {len(value) - 10} more")
            else:
                print(f"{key}: {json.dumps(value, default=str)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m suite_tools.bench",
        description="Benchmark registry and comparison commands.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    # runs
    p_runs = sub.add_parser("runs", help="Scan for run directories.")
    p_runs.add_argument("--root", dest="roots", action="append", type=Path, default=None)
    p_runs.add_argument("--json", dest="as_json", action="store_true")

    # experiments
    p_exp = sub.add_parser("experiments", help="Find EXPERIMENT.json directories.")
    p_exp.add_argument("--root", dest="roots", action="append", type=Path, default=None)
    p_exp.add_argument("--json", dest="as_json", action="store_true")

    # status (alias for owed_units)
    p_status = sub.add_parser("status", help="Owed units for a run (alias for owed_units).")
    p_status.add_argument("run", type=Path, help="Run directory.")
    p_status.add_argument("--json", dest="as_json", action="store_true")

    # diagnose (private operational journal; never part of bundle identity)
    p_diagnose = sub.add_parser(
        "diagnose",
        help="Explain provider-call failures and ambiguous in-flight attempts for one run.",
    )
    p_diagnose.add_argument("run", type=Path, help="Run directory.")
    p_diagnose.add_argument("--json", dest="as_json", action="store_true")

    # blockers
    p_block = sub.add_parser("blockers", help="Runs whose latest attempt halted.")
    p_block.add_argument("--root", dest="roots", action="append", type=Path, default=None)
    p_block.add_argument("--json", dest="as_json", action="store_true")

    # adopt (passthrough to experiment.adopt)
    p_adopt = sub.add_parser("adopt", help="Adopt a run into an experiment.")
    p_adopt.add_argument("exp", type=Path, help="Experiment directory.")
    p_adopt.add_argument("run", type=Path, help="Run directory.")
    p_adopt.add_argument("--role", required=True, help="Role for this member.")
    p_adopt.add_argument("--json", dest="as_json", action="store_true")

    # supersede (passthrough to experiment.supersede)
    p_sup = sub.add_parser("supersede", help="Supersede a member in an experiment.")
    p_sup.add_argument("exp", type=Path, help="Experiment directory.")
    p_sup.add_argument("member", type=Path, help="Member path to supersede.")
    p_sup.add_argument("--by", required=True, type=Path, help="Path of superseding member.")
    p_sup.add_argument("--reason", required=True, help="Reason for supersession.")
    p_sup.add_argument("--json", dest="as_json", action="store_true")

    # verify
    p_verify = sub.add_parser("verify", help="Verify run directories or a bundle tree.")
    p_verify.add_argument("dirs", nargs="*", type=Path, help="Run directory or pair of directories.")
    p_verify.add_argument("--bundle", type=Path, default=None,
                          help="Audit an emitted bundle tree instead of run dirs.")
    p_verify.add_argument("--json", dest="as_json", action="store_true")
    p_verify.add_argument("--strict", action="store_true",
                          help="Exit 1 if two dirs are not comparable.")

    # package (delegates to bundle emitter)
    p_pkg = sub.add_parser("package", help="Emit a self-contained experiment bundle.")
    p_pkg.add_argument("exp", type=Path, help="Experiment directory.")
    p_pkg.add_argument("--out", required=True, type=Path, help="Output directory.")
    p_pkg.add_argument("--include-transcripts", action="store_true",
                       help="Include conversation transcripts (review.html).")
    p_pkg.add_argument("--include-review-rationale", action="store_true",
                       help="Include free-text review rationale in "
                            "data/block_reviews.jsonl (stamps the manifest).")
    p_pkg.add_argument("--no-report", dest="write_report", action="store_false",
                       help="Skip the markdown summary report.")
    p_pkg.add_argument("--json", dest="as_json", action="store_true")

    # review (triage queue — list mode and disposition mode; plan 020 §3 T6 + D11)
    p_rev = sub.add_parser("review", help="Triage the unresolved evidence queue.")
    # Shared
    p_rev.add_argument("--run", dest="run_dir", type=Path, default=None,
                       help="Restrict to this run directory (list) or target run (disposition).")
    p_rev.add_argument("--root", dest="roots", action="append", type=Path, default=None,
                       help="Scan root(s) in list mode.")
    p_rev.add_argument("--json", dest="as_json", action="store_true")
    # List mode filters
    p_rev.add_argument("--all", dest="include_resolved", action="store_true",
                       help="Include resolved facts (default: unresolved only).")
    p_rev.add_argument("--class", dest="evidence_class", default=None,
                       help="Filter by evidence_class.")
    p_rev.add_argument("--scope", dest="scope", default=None,
                       choices=["unit", "member", "unmappable"],
                       help="Filter by scope.")
    # Disposition mode
    p_rev.add_argument("--event-ref", dest="event_ref", default=None,
                       help="D4 event_ref of the fact to disposition.")
    p_rev.add_argument("--disposition", dest="disposition", default=None,
                       choices=["safety_declination", "retry", "instrument_defect",
                                "needs_escalation"],
                       help="Disposition to apply.")
    p_rev.add_argument("--by", dest="reviewer", default=None,
                       help="Reviewer identifier.")
    p_rev.add_argument("--reason", dest="reason", default=None,
                       help="Rationale for the disposition.")
    p_rev.add_argument("--resolved-category", dest="resolved_category", default=None,
                       help="Resolved category (required for safety_declination on unclassified/ambiguous facts).")
    p_rev.add_argument("--issue-ref", dest="issue_ref", default=None,
                       help="Structured issue reference (e.g. ISSUE-42).")
    p_rev.add_argument("--supersede", dest="supersede", default=None,
                       help="review_id of the active review this supersedes.")

    args = parser.parse_args(argv)

    if args.verb == "runs":
        result = runs(roots=args.roots)
    elif args.verb == "experiments":
        result = experiments(roots=args.roots)
    elif args.verb == "status":
        result = status(args.run)
    elif args.verb == "diagnose":
        result = diagnose(args.run)
    elif args.verb == "blockers":
        result = blockers(roots=args.roots)
    elif args.verb == "adopt":
        result = adopt(args.exp, args.run, role=args.role)
    elif args.verb == "supersede":
        result = supersede(args.exp, args.member, by=args.by, reason=args.reason)
    elif args.verb == "verify":
        if args.bundle is not None:
            result = verify_bundle(args.bundle)
        elif args.dirs:
            result = verify(args.dirs, strict=args.strict)
        else:
            parser.error("verify requires run directories or --bundle DIR")
    elif args.verb == "package":
        result = package(
            args.exp,
            args.out,
            include_transcripts=args.include_transcripts,
            include_review_rationale=args.include_review_rationale,
            write_report=args.write_report,
        )
    elif args.verb == "review":
        result = review(
            roots=args.roots,
            include_resolved=args.include_resolved,
            evidence_class=args.evidence_class,
            scope=args.scope,
            run_dir=args.run_dir,
            event_ref=args.event_ref,
            disposition=args.disposition,
            reviewer=args.reviewer,
            reason=args.reason,
            resolved_category=args.resolved_category,
            issue_ref=args.issue_ref,
            supersede=args.supersede,
        )
    else:
        parser.print_help()
        return 1

    as_json = getattr(args, "as_json", False)
    _print_result(result, as_json=as_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
