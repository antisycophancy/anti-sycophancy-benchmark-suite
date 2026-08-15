"""Bundle emitter — turn an experiment into a shareable, self-contained artifact.

``python3 -m suite_tools.bundle EXPERIMENT_DIR --out DIR`` reduces an experiment
to a public, privacy-scrubbed bundle: bundle-local member ids (``m1``, ``m2`` …)
in place of every absolute path, a union-scoped view of scores/outcomes/blocks,
experiment-level derived aggregates computed AFTER ``experiment.union()`` from the
winning rows only, and projected run contracts. Every payload is gated through
``artifact_privacy.assert_public_artifact_safe`` before it is written and the whole
emitted tree is re-audited by ``audit_bundle_tree`` before the atomic rename.

Review-locked invariants (three Sol rounds):

* **Atomic emission (finding 9).** Everything is written under a sibling
  ``.<name>.tmp/`` staging directory; the bundle-tree audit runs over the staged
  tree; only on success is it ``os.replace``d to the final name. Any abort (privacy
  violation, audit failure) removes the staging dir and leaves NO partial bundle.
* **Bundle-local ids only.** No absolute paths in any payload. ``union()``'s member
  references (``chosen_member``/``candidates``/``kept_member``/``dropped_members``/
  ``member_path``/``path``) are rewritten to ``member_id`` in the manifest.
* **Projected contracts.** ``identity.execution`` and top-level ``source_command``
  are dropped; the recomputed provenance panel is embedded per member.
* **Derived aggregates from union winners only (round-2 B3).** The EPIS aggregate is
  recomputed once per (module, condition) group over the winning per-unit records —
  never per member — so a superseded/duplicate member cannot shift the experiment
  aggregate.
* **Blocks scoped to union winners (round-3 finding 5).** A collision loser's stale
  block for a superseded unit never reaches the bundle.
* **Absolute-path scan (finding 5).** ``artifact_privacy.ABSOLUTE_HOME_PATH_RE``
  flags any leaked ``/Users`` or ``/home`` path — defense in depth.
* Without ``--include-transcripts`` there is no conversation text anywhere; the tree
  audit and a sentinel grep both confirm it.

Phase-D publication safety (Task 7 — D2, D6, D7):

* **Allowlist projection (D2).** ``data/blocks.jsonl`` and the new
  ``data/evidence.jsonl`` publish ONLY ``_BLOCK_PUBLIC_FIELDS`` /
  ``_EVENT_PUBLIC_FIELDS``.  The v2 ``raw_body_excerpt`` (which can echo prompt
  content) and every unknown/future key are dropped by construction — only the
  provenance digest ``raw_body_sha256`` survives.  ``data/block_reviews.jsonl``
  publishes the active review + full supersession chain with free-text
  ``rationale`` projected out; ``--include-review-rationale`` opts it in AND
  stamps ``contains_review_rationale`` in the manifest (mirror of the transcripts
  flag).  Every reviewable fact a published review references also ships.
* **Three-clause hard gate (D6), no bypass.** Before the promote, a bundle is
  refused if any included fact is unknown-class with no resolving review, has an
  active ``needs_escalation`` review, or if any winning unit is effectively owed /
  pending_retry / instrument_defect, or a member-level retry obligation is
  pending.  The error lists every offender as ``(member_id, unit_id?, event_ref,
  reason)``.
* **RunSnapshot consistency (D6).** One immutable per-member ``RunSnapshot``
  (RUN_STATUS + ledger/review bytes + fingerprints + the captured projection) is
  built at gate time; the projection and gate consume only the snapshot.  A
  non-terminal-completed member is refused, and immediately before the promote
  every fingerprint is re-verified — any drift (an attempt started mid-bundle)
  aborts with staging cleanup.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from suite_tools import experiment as _experiment_mod
from suite_tools.artifact_privacy import (
    ABSOLUTE_HOME_PATH_RE,
    PRIVATE_HOST_RE,
    SECRET_VALUE_RE,
    ArtifactPrivacyIssue,
    assert_public_artifact_safe,
    assert_text_public_safe as _assert_text_public_safe,
    scan_public_artifact_payload,
)
from suite_tools.assert_hash_panel import item_universe_report
from suite_tools.run_contract import (
    CONTRACT_FILENAME,
    IDENTITY_PROJECTION_VERSION,
    LEGACY_IDENTITY_PROJECTION_VERSION,
    REPO_ROOT,
    compare_provenance,
    load_run_contract,
    provenance_hashes,
    provenance_hashes_for_version,
)
from suite_tools.run_monitor import utc_now
from suite_tools.score_rows import (
    EPIS_AGGREGATE_HELPER,
    epis_aggregate,
    score_rows as compute_score_rows,
)

BUNDLE_SCHEMA_VERSION = "benchmark-bundle-v1"
# Truthful, self-describing exclusion policy: the ``definition`` states exactly
# the behavior score_rows/bundle implement, so the manifest id is not a phantom.
EXCLUSION_POLICY = {
    "id": "responsive-subset-v1",
    "definition": (
        "behavioral score rows and denominators include outcome_class=scored "
        "units only; terminal_model_signal units are reported as declination "
        "rates beside scores, never in behavioral denominators; unscored units "
        "are pending-scoring and excluded from both"
    ),
}
BLOCKS_FILENAME = "BLOCKS.jsonl"
BLOCK_REVIEWS_FILENAME = "BLOCK_REVIEWS.jsonl"
RUN_EVENTS_FILENAME = "RUN_EVENTS.jsonl"
RUN_STATUS_FILENAME = "RUN_STATUS.json"

# ---------------------------------------------------------------------------
# Public allowlists (plan 020 D2, D7).  Projection is an ALLOWLIST, not a
# denylist: only these keys survive into a bundle, so any unknown/future field
# (raw_body_excerpt above all) is dropped by construction.  ``member_id`` and
# ``event_ref`` are stamped by the bundle emitter (not part of the raw record).
# ---------------------------------------------------------------------------

# BLOCKS.jsonl records.  raw_body_excerpt is DELIBERATELY absent — the raw body
# can echo prompt content and is LOCAL-only; only its provenance digest
# (raw_body_sha256) is published.  Extend this tuple when a genuinely public
# corrected-field is added upstream; until then unknown keys drop.
_BLOCK_PUBLIC_FIELDS = (
    "schema_version", "block_id", "timestamp", "module", "stage",
    "attempt_number", "model", "unit", "unit_id", "evidence_class",
    "category", "evidence_pointer", "provider", "provider_code",
    "native_finish_reason", "signal_source", "retry_policy_kind",
    "stochastic", "billed_attempts", "raw_body_sha256",
    "backfilled", "backfill_id",
)

# attempt_failure_classified event facts.  Same discipline; raw_body_excerpt AND
# failure_reason are dropped (both can echo model/prompt text — D2 publishes only
# provider/class/category/native_finish_reason/signal_source/policy-kind/digest).
_EVENT_PUBLIC_FIELDS = (
    "schema_version", "event_id", "event", "timestamp", "module", "stage",
    "attempt_number", "model", "unit_id", "evidence_class", "category",
    "action", "provider", "provider_code", "native_finish_reason",
    "signal_source", "retry_policy_kind", "stochastic", "billed_attempts",
    "raw_body_sha256", "item_idx", "side", "scenario", "test_type",
)

# BLOCK_REVIEWS.jsonl records.  ``rationale`` is free text and is projected out
# by default (only stamped in on the --include-review-rationale opt-in); the
# structured ``issue_ref`` always publishes.
_REVIEW_PUBLIC_FIELDS = (
    "schema_version", "review_id", "module", "model", "unit_id", "category",
    "resolved_category", "disposition", "reviewer", "issue_ref", "reviewed_at",
    "supersedes_review_id", "backfill_id", "backfilled",
)

# A winning unit may be published only when it is scored (completed) or a
# legitimate declination (terminal_model_signal); every other effective state
# (owed / pending_retry / instrument_defect / unresolved integrity) blocks.
_PUBLISHABLE_UNIT_STATES = frozenset({"completed", "terminal_model_signal"})

# ``unit`` is the ONE nested object inside a public block record.  Allowlisting
# it wholesale would let a nested raw body ride along; instead it is projected to
# these scalar identity keys only (plan 020 D2/D7 hardening, finding 6).  Every
# other block/event/review allowlist field is already a scalar.
_UNIT_IDENTITY_FIELDS = (
    "unit_id", "model_key", "item_idx", "side", "scenario", "run",
    "test_type", "planned_turns", "planned_escalations",
)


# Nominally-scalar allowlisted fields are capped so no pathological (possibly
# raw-body) string can be smuggled through one; every legitimate value (enum,
# id, path, sha, timestamp) is far shorter.
_MAX_PUBLIC_STR = 4096


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _coerce_public_scalar(value: Any) -> Any:
    """Return *value* if it is a (length-capped) scalar, else ``None``.

    A dict/list value in a nominally-scalar allowlisted field is DROPPED — it can
    only be a smuggling vector for nested content (finding 5/6 hardening).
    """
    if isinstance(value, str):
        return value[:_MAX_PUBLIC_STR]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return None


def _project_unit_identity(unit: Any) -> Any:
    """Reduce a block ``unit`` to its scalar identity fields, or ``None``.

    A non-dict ``unit`` is malformed (a legitimate unit is always a dict) and is
    DROPPED wholesale — a bare string/list ``unit`` could itself be a raw body
    (finding 5).  For a dict, only :data:`_UNIT_IDENTITY_FIELDS` with scalar
    values survive, so no content nested at any depth can ride along (finding 6).
    """
    if not isinstance(unit, dict):
        return None
    return {
        key: _coerce_public_scalar(unit[key])
        for key in _UNIT_IDENTITY_FIELDS
        if key in unit and _is_scalar(unit[key])
    }

# Fields that can carry conversation/transcript text.  These are stripped from any
# source score record before it is gated (so real transcript content never causes a
# false-positive abort) and, in the default no-transcript bundle, are never written.
_TRANSCRIPT_FIELD_NAMES = frozenset({
    "conversation",
    "conversations",
    "dialogue",
    "history",
    "messages",
    "model_response",
    "prompt",
    "raw_prompt",
    "raw_response",
    "response",
    "responses",
    "transcript",
    "transcripts",
    "turns",
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _parse_model_key(unit_id: str) -> str:
    parts = str(unit_id or "").split(":")
    return parts[1] if len(parts) > 1 else ""


def _strip_transcripts(value: Any) -> Any:
    """Recursively drop transcript-bearing keys from a JSON-like value."""
    if isinstance(value, dict):
        return {
            key: _strip_transcripts(item)
            for key, item in value.items()
            if key not in _TRANSCRIPT_FIELD_NAMES
        }
    if isinstance(value, list):
        return [_strip_transcripts(item) for item in value]
    return value


def _unit_index(contract: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return ``{unit_id: (module, unit_dict)}`` for a contract's expected units."""
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    for module_entry in contract.get("modules") or []:
        if not isinstance(module_entry, dict):
            continue
        module = str(module_entry.get("module") or "")
        for unit in module_entry.get("expected_units") or []:
            if isinstance(unit, dict) and unit.get("unit_id"):
                index[str(unit["unit_id"])] = (module, unit)
    return index


def _tool_version() -> str:
    """Return a ``git describe``/commit string for provenance, or ``"unknown"``."""
    for args in (
        ["git", "describe", "--tags", "--always", "--dirty"],
        ["git", "rev-parse", "HEAD"],
    ):
        try:
            proc = subprocess.run(
                args,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        value = (proc.stdout or "").strip()
        if value:
            return value
    return "unknown"


def _next_version(out_dir: Path, experiment_id: str) -> int:
    """Return the next ``v{N}`` for an experiment id (auto-increment on existing)."""
    prefix = f"bundle-{experiment_id}-v"
    highest = 0
    for child in out_dir.glob(f"{prefix}*"):
        if not child.is_dir():
            continue
        suffix = child.name[len(prefix):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def _sanitize_selection_value(value: Any) -> Any:
    """Recursively sanitize selection values.

    Any string that starts with ``"/"`` (i.e. an absolute filesystem path) is
    reduced to its basename.  Non-path strings, numbers, booleans, lists, and
    dicts are traversed/left unchanged so non-path selection content (item
    hashes, test-type labels, etc.) is preserved verbatim.
    """
    if isinstance(value, str):
        return Path(value).name if value.startswith("/") else value
    if isinstance(value, dict):
        return {k: _sanitize_selection_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_selection_value(item) for item in value]
    return value


_PUBLIC_MODULE_KEYS = frozenset({
    "dataset_mode",
    "escalation_mode",
    "expected_artifacts",
    "expected_units",
    "module",
    "runs",
    "scenarios",
    "selection",
    "stage",
})
_PUBLIC_EXPECTED_UNIT_KEYS = frozenset({
    "condition_hash",
    "condition_id",
    "escalation_mode",
    "expected_score_path",
    "expected_summary_path",
    "expected_trace_path",
    "expected_transcript_path",
    "ground_truth",
    "item_hash",
    "item_idx",
    "model_id",
    "model_key",
    "pair_id",
    "planned_escalations",
    "planned_turns",
    "route_hash",
    "run_number",
    "scenario",
    "side",
    "side_prompt_hash",
    "source_pair_hash",
    "test_type",
    "unit_id",
})
_PUBLIC_EXPECTED_ARTIFACT_KEYS = frozenset({
    "bytes",
    "kind",
    "path",
    "required_for",
    "sha256",
})


def _project_allowlisted_mapping(value: Any, allowed: frozenset[str]) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: _sanitize_selection_value(item)
        for key, item in value.items()
        if key in allowed
    }


def _project_module_entry(module: dict[str, Any]) -> dict[str, Any]:
    """Project one module entry for public inclusion.

    Unknown module, unit, and artifact extensions are dropped fail-closed.
    The identity sample manifest already carries path-free content hashes and
    selection provenance, so operational dataset/output fields are omitted.
    """
    result: dict[str, Any] = {}
    for key, val in module.items():
        if key not in _PUBLIC_MODULE_KEYS:
            continue
        if key == "expected_units" and isinstance(val, list):
            result[key] = [
                _project_allowlisted_mapping(item, _PUBLIC_EXPECTED_UNIT_KEYS)
                for item in val
            ]
        elif key == "expected_artifacts" and isinstance(val, list):
            result[key] = [
                _project_allowlisted_mapping(item, _PUBLIC_EXPECTED_ARTIFACT_KEYS)
                for item in val
            ]
        elif key == "selection":
            result[key] = _sanitize_selection_value(val)
        else:
            result[key] = _sanitize_selection_value(val)
    return result


_PUBLIC_MODEL_METADATA_KEYS = frozenset({
    "api_family",
    "effort",
    "effort_policy",
    "model_family",
    "provider_fallback",
    "provider_route",
    "role",
    "source_official_model_id",
    "thinking_config_path",
})
_PUBLIC_MODEL_CONDITION_KEYS = frozenset({
    "canonical_model",
    "condition_hash",
    "condition_id",
    "condition_metadata",
    "effort",
    "key",
    "label",
    "model_id",
    "profile_hash",
    "provider_api",
    "provider_condition_hash",
    "request_options",
    "route_hash",
    "served_model_version",
    "served_profile_hash",
    "served_weights_fingerprint",
    "system_fingerprint",
})


def _project_model_condition_for_bundle(condition: Any) -> Any:
    """Remove local routing detail while retaining public tested-system identity."""
    if not isinstance(condition, dict):
        return condition
    projected = {
        key: _sanitize_selection_value(value)
        for key, value in condition.items()
        if key in _PUBLIC_MODEL_CONDITION_KEYS
    }
    # Endpoint aliases, registry extensions, transport settings, and profile
    # IDs are operator-controlled. Keep only the explicit scientific/public
    # fields above; hashes retain comparability without publishing those IDs.
    metadata = projected.get("condition_metadata")
    if isinstance(metadata, dict):
        projected["condition_metadata"] = {
            key: value
            for key, value in metadata.items()
            if key in _PUBLIC_MODEL_METADATA_KEYS
        }
    return projected


_PUBLIC_JUDGE_PANEL_KEYS = frozenset({
    "analyzer",
    "configs",
    "flip_generator",
    "judge_prompt_hashes",
    "judges",
    "panel",
    "primary",
    "rubric",
    "rubric_source_ids",
    "rubric_source_registry",
    "rubric_version",
    "seeker",
})
_PUBLIC_JUDGE_CONFIG_KEYS = frozenset({
    "condition_hash",
    "condition_id",
    "condition_metadata",
    "model_id",
    "provider_api",
    "profile_hash",
    "request_options",
    "route_hash",
    "served_profile_hash",
    "served_model_version",
    "served_weights_fingerprint",
    "system_fingerprint",
})


def _project_judge_config_for_bundle(config: Any) -> Any:
    if not isinstance(config, dict):
        return config
    projected = {
        key: _sanitize_selection_value(value)
        for key, value in config.items()
        if key in _PUBLIC_JUDGE_CONFIG_KEYS
    }
    metadata = projected.get("condition_metadata")
    if isinstance(metadata, dict):
        projected["condition_metadata"] = {
            key: value
            for key, value in metadata.items()
            if key in _PUBLIC_MODEL_METADATA_KEYS
        }
    return projected


def _project_judge_panel_for_bundle(panel: Any) -> Any:
    """Publish scientific judge identity without operator transport details."""
    if not isinstance(panel, dict):
        return panel
    projected = {
        key: _sanitize_selection_value(value)
        for key, value in panel.items()
        if key in _PUBLIC_JUDGE_PANEL_KEYS
    }
    if isinstance(projected.get("configs"), list):
        projected["configs"] = [
            _project_judge_config_for_bundle(config)
            for config in projected["configs"]
        ]
    if isinstance(projected.get("analyzer"), dict):
        projected["analyzer"] = _project_judge_config_for_bundle(
            projected["analyzer"]
        )
    return projected


def _project_score_row_for_bundle(row: dict[str, Any]) -> dict[str, Any]:
    """Project score grouping axes without publishing a private endpoint.

    ``score_rows`` intentionally preserves the concrete route for local
    analysis.  Public bundles retain that a profiled adapter was used, but not
    its localhost or internal address.
    """
    projected = dict(row)
    condition = projected.get("condition")
    if not isinstance(condition, dict):
        return projected
    public_condition = dict(condition)
    route = public_condition.get("route")
    public_routes = {
        "anthropic_native",
        "google_gemini_native",
        "local_openai_compatible",
        "openai_compatible",
        "openai_native",
        "openai_responses",
        "openrouter",
        "profile_adapter",
    }
    profile = public_condition.get("profile")
    if isinstance(route, str) and route not in public_routes:
        public_condition["route"] = (
            "profile_adapter" if isinstance(profile, dict) and profile.get("profile_id")
            else "openai_compatible"
        )
    if isinstance(profile, dict):
        public_profile = {
            key: value
            for key, value in profile.items()
            if key in {"profile_hash"}
        }
        public_condition["profile"] = public_profile or None
    projected["condition"] = public_condition
    return projected


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _project_contract_for_bundle(contract: dict[str, Any]) -> dict[str, Any]:
    """Project a run contract for public inclusion.

    Keeps ``schema_version``/``run_id``/``modules``/``identity`` (minus
    ``execution``) — enough to recompute provenance — drops ``source_command`` and
    any other top-level key, and embeds a freshly recomputed provenance panel.

    Path sanitization (BUG 1):
    * ``modules[*].output_dir`` is dropped (always an absolute local path).
    * ``modules[*].selection`` strings that start with ``"/"`` are reduced to
      basename.
    * ``identity.sample_spec.selection`` strings that start with ``"/"`` are
      reduced to basename.

    The source contract is never mutated — all changes happen on projected copies.
    """
    stored_provenance = contract.get("provenance")
    if not isinstance(stored_provenance, dict) or not stored_provenance:
        raise ValueError("source contract provenance missing")
    stored_version = str(
        stored_provenance.get("projection_version")
        or LEGACY_IDENTITY_PROJECTION_VERSION
    )
    try:
        expected_provenance = provenance_hashes_for_version(
            contract,
            stored_version,
        )
    except ValueError as exc:
        raise ValueError(f"source contract provenance drift: {exc}") from exc
    if expected_provenance != stored_provenance:
        raise ValueError(
            "source contract provenance drift: stored panel does not match identity"
        )

    projected: dict[str, Any] = {}
    for key in ("schema_version", "run_id"):
        if key in contract:
            projected[key] = contract[key]
    # Project modules: drop output_dir, sanitize selection
    if "modules" in contract:
        projected["modules"] = [
            _project_module_entry(m) if isinstance(m, dict) else m
            for m in (contract["modules"] or [])
        ]
    identity = dict(contract.get("identity") or {})
    identity.pop("execution", None)
    # Sanitize selection in sample_spec so absolute data-file paths become basenames
    if "sample_spec" in identity:
        ss = identity["sample_spec"]
        if isinstance(ss, dict) and "selection" in ss:
            ss = dict(ss)
            ss["selection"] = _sanitize_selection_value(ss["selection"])
            identity["sample_spec"] = ss
    if "model_conditions" in identity:
        identity["model_conditions"] = [
            _project_model_condition_for_bundle(condition)
            for condition in (identity.get("model_conditions") or [])
        ]
    if "judge_panel" in identity:
        identity["judge_panel"] = _project_judge_panel_for_bundle(
            identity.get("judge_panel")
        )
    projected["identity"] = identity
    projected["projection_version"] = IDENTITY_PROJECTION_VERSION
    # Recompute from the projected contract (execution already dropped) so no local
    # execution detail can enter even the hashed panel.
    projected["provenance"] = provenance_hashes(projected)
    return projected


# ---------------------------------------------------------------------------
# Bundle-tree audit (filesystem, independent of git tracking)
# ---------------------------------------------------------------------------


def _scan_text_payload(rel: str, text: str) -> list[ArtifactPrivacyIssue]:
    issues: list[ArtifactPrivacyIssue] = []
    if SECRET_VALUE_RE.search(text):
        issues.append(ArtifactPrivacyIssue(rel, "secret-looking value"))
    if PRIVATE_HOST_RE.search(text):
        issues.append(ArtifactPrivacyIssue(rel, "private/internal marker"))
    if ABSOLUTE_HOME_PATH_RE.search(text):
        issues.append(ArtifactPrivacyIssue(rel, "absolute home path"))
    return issues




def audit_bundle_tree(bundle_dir: Path | str) -> list[ArtifactPrivacyIssue]:
    """Walk an emitted bundle tree and scan every file for public-safety issues.

    JSON/JSONL payloads go through ``scan_public_artifact_payload`` (private field
    names, secrets, private hosts, absolute home paths); all other text files are
    matched against ``SECRET_VALUE_RE`` + ``PRIVATE_HOST_RE`` +
    ``ABSOLUTE_HOME_PATH_RE``.  Because it walks the filesystem (not
    ``git ls-files``), it inspects gitignored bundles that ``release_audit`` never
    sees.  Returned issue ``path`` values are bundle-relative so the offending file
    is always identifiable.
    """
    root = Path(bundle_dir)
    issues: list[ArtifactPrivacyIssue] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="ignore")
        suffix = path.suffix.lower()

        if suffix in (".json", ".jsonl"):
            lines = text.splitlines() if suffix == ".jsonl" else [text]
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    issues.extend(_scan_text_payload(rel, stripped))
                    continue
                for sub in scan_public_artifact_payload(payload):
                    issues.append(ArtifactPrivacyIssue(f"{rel}::{sub.path}", sub.reason))
        elif suffix == ".csv":
            for row_number, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
                for column, value in row.items():
                    if str(value or "").strip().lower() in {
                        "nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
                    }:
                        issues.append(ArtifactPrivacyIssue(
                            f"{rel}::row{row_number}.{column}",
                            "non-finite numeric value",
                        ))
            issues.extend(_scan_text_payload(rel, text))
        else:
            issues.extend(_scan_text_payload(rel, text))

    return issues


def _payload_files(bundle_dir: Path) -> list[Path]:
    """Return immutable bundle payload files, excluding generated reports."""
    files: list[Path] = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        if rel == "BUNDLE_MANIFEST.json" or rel == "REPORT.md" or rel.startswith("report/"):
            continue
        files.append(path)
    return files


def _payload_hash_entries(bundle_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _payload_files(bundle_dir):
        raw = path.read_bytes()
        entries.append({
            "path": path.relative_to(bundle_dir).as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return entries


def audit_bundle_integrity(bundle_dir: Path | str) -> list[ArtifactPrivacyIssue]:
    """Verify the manifest's complete payload file list and SHA-256 digests."""
    root = Path(bundle_dir)
    manifest_path = root / "BUNDLE_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [ArtifactPrivacyIssue("BUNDLE_MANIFEST.json", f"integrity manifest unreadable: {exc}")]

    expected = manifest.get("payload_files")
    if not isinstance(expected, list):
        return [ArtifactPrivacyIssue("BUNDLE_MANIFEST.json", "payload hash inventory missing")]

    expected_by_path: dict[str, dict[str, Any]] = {}
    issues: list[ArtifactPrivacyIssue] = []
    for entry in expected:
        if not isinstance(entry, dict):
            issues.append(ArtifactPrivacyIssue("BUNDLE_MANIFEST.json", "invalid payload hash entry"))
            continue
        rel = str(entry.get("path") or "")
        rel_path = Path(rel)
        if not rel or rel_path.is_absolute() or ".." in rel_path.parts:
            issues.append(ArtifactPrivacyIssue("BUNDLE_MANIFEST.json", f"unsafe payload path: {rel!r}"))
            continue
        if rel in expected_by_path:
            issues.append(ArtifactPrivacyIssue(rel, "duplicate payload hash entry"))
            continue
        expected_by_path[rel] = entry

    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in _payload_files(root)
    }
    for rel in sorted(set(expected_by_path) - set(actual_paths)):
        issues.append(ArtifactPrivacyIssue(rel, "payload file missing"))
    for rel in sorted(set(actual_paths) - set(expected_by_path)):
        issues.append(ArtifactPrivacyIssue(rel, "unlisted payload file"))
    for rel in sorted(set(actual_paths) & set(expected_by_path)):
        raw = actual_paths[rel].read_bytes()
        entry = expected_by_path[rel]
        if entry.get("bytes") != len(raw):
            issues.append(ArtifactPrivacyIssue(rel, "payload byte count mismatch"))
        if entry.get("sha256") != hashlib.sha256(raw).hexdigest():
            issues.append(ArtifactPrivacyIssue(rel, "payload SHA-256 mismatch"))
    return issues


def audit_bundle_provenance(bundle_dir: Path | str) -> list[ArtifactPrivacyIssue]:
    """Recompute every projected contract panel from its bundled identity."""
    root = Path(bundle_dir)
    issues: list[ArtifactPrivacyIssue] = []
    contract_paths = sorted((root / "provenance").glob("RUN_CONTRACT-*.json"))
    manifest_path = root / "BUNDLE_MANIFEST.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}
        members = manifest.get("members") if isinstance(manifest, dict) else None
        union = manifest.get("union") if isinstance(manifest, dict) else None
        union_units = union.get("units") if isinstance(union, dict) else None
        if not isinstance(union_units, list) or not union_units:
            issues.append(
                ArtifactPrivacyIssue(
                    "BUNDLE_MANIFEST.json",
                    "bundle union units missing",
                )
            )
        payload_files = manifest.get("payload_files") if isinstance(manifest, dict) else None
        if not isinstance(payload_files, list) or not payload_files:
            issues.append(
                ArtifactPrivacyIssue(
                    "BUNDLE_MANIFEST.json",
                    "bundle payload inventory empty",
                )
            )
        if isinstance(members, list):
            if not members:
                issues.append(
                    ArtifactPrivacyIssue(
                        "BUNDLE_MANIFEST.json",
                        "bundle member provenance missing",
                    )
                )
            expected_paths = {
                root / "provenance" / f"RUN_CONTRACT-{member['member_id']}.json"
                for member in members
                if isinstance(member, dict) and isinstance(member.get("member_id"), str)
            }
            for missing in sorted(expected_paths - set(contract_paths)):
                issues.append(
                    ArtifactPrivacyIssue(
                        missing.relative_to(root).as_posix(),
                        "projected contract missing",
                    )
                )

        outcomes_path = root / "data" / "outcomes.jsonl"
        try:
            outcome_lines = outcomes_path.read_text().splitlines()
        except OSError:
            issues.append(ArtifactPrivacyIssue("data/outcomes.jsonl", "bundle outcomes missing"))
        else:
            has_outcome = False
            malformed = False
            for line in outcome_lines:
                if not line.strip():
                    continue
                try:
                    outcome = json.loads(line)
                except json.JSONDecodeError:
                    malformed = True
                    continue
                if (
                    isinstance(outcome, dict)
                    and isinstance(outcome.get("unit_id"), str)
                    and outcome["unit_id"]
                ):
                    has_outcome = True
            if malformed:
                issues.append(
                    ArtifactPrivacyIssue("data/outcomes.jsonl", "bundle outcomes unreadable")
                )
            if not has_outcome:
                issues.append(ArtifactPrivacyIssue("data/outcomes.jsonl", "bundle outcomes empty"))
    for contract_path in contract_paths:
        rel = contract_path.relative_to(root).as_posix()
        try:
            contract = json.loads(contract_path.read_text())
        except (OSError, json.JSONDecodeError):
            issues.append(ArtifactPrivacyIssue(rel, "projected contract unreadable"))
            continue
        if not isinstance(contract, dict):
            issues.append(ArtifactPrivacyIssue(rel, "projected contract is not an object"))
            continue
        stored = contract.get("provenance")
        if not isinstance(stored, dict) or not stored:
            issues.append(ArtifactPrivacyIssue(rel, "projected contract provenance missing"))
            continue
        version = str(
            stored.get("projection_version")
            or LEGACY_IDENTITY_PROJECTION_VERSION
        )
        try:
            expected = provenance_hashes_for_version(contract, version)
        except ValueError:
            issues.append(ArtifactPrivacyIssue(rel, "unsupported provenance projection"))
            continue
        if expected != stored:
            issues.append(ArtifactPrivacyIssue(rel, "projected contract provenance drift"))
    return issues


# ---------------------------------------------------------------------------
# Derived data (all computed AFTER union())
# ---------------------------------------------------------------------------


def _winning_records(
    union_units: list[dict[str, Any]],
    *,
    contract_by_path: dict[str, dict[str, Any]],
    run_dir_by_path: dict[str, Path],
) -> list[dict[str, Any]]:
    """Collect and gate the winning per-unit score records.

    For each unit ``union()`` selected, resolve the *chosen member's*
    ``expected_score_path`` record, strip transcript fields, and gate it through
    ``assert_public_artifact_safe`` (a poisoned source record aborts the bundle
    here, before anything is written).  Collision losers are never read — they
    never enter ``union().units`` — so they cannot contaminate downstream
    aggregates.
    """
    index_cache: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {}
    records: list[dict[str, Any]] = []

    for unit in union_units:
        member_path = unit.get("chosen_member")
        contract = contract_by_path.get(member_path)
        run_dir = run_dir_by_path.get(member_path)
        if contract is None or run_dir is None:
            continue
        index = index_cache.get(member_path)
        if index is None:
            index = _unit_index(contract)
            index_cache[member_path] = index
        entry = index.get(str(unit.get("unit_id")))
        if entry is None:
            continue
        module, unit_dict = entry
        score_path_raw = unit_dict.get("expected_score_path")
        if not score_path_raw:
            continue
        score_path = Path(score_path_raw)
        if not score_path.is_absolute():
            score_path = run_dir / score_path
        record = _load_json(score_path)
        if not isinstance(record, dict):
            continue
        stripped = _strip_transcripts(record)
        # Privacy gate on the source score record (Sol: every payload before write).
        assert_public_artifact_safe(stripped)
        records.append({
            "module": module,
            "condition_id": unit.get("condition_id"),
            "model_key": _parse_model_key(str(unit.get("unit_id"))),
            "record": stripped,
        })

    return records


def _derived_aggregates(winning_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute experiment-level EPIS aggregates from the winning records only.

    Grouped by (module, condition_id, model_key); one ``epis_aggregate`` call per
    group; one row per numeric aggregate dimension.  Never per member.
    """
    groups: dict[tuple[str, Any, str], list[dict[str, Any]]] = {}
    for winning in winning_records:
        if winning["module"] not in ("epis", "epistemic"):
            continue
        key = (winning["module"], winning["condition_id"], winning["model_key"])
        groups.setdefault(key, []).append(winning["record"])

    rows: list[dict[str, Any]] = []
    for (module, condition_id, model_key), group in sorted(
        groups.items(), key=lambda item: json.dumps(item[0], default=str)
    ):
        aggregate = epis_aggregate(group)
        n = len(group)
        for dimension, value in aggregate.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            rows.append({
                "module": module,
                "condition_id": condition_id,
                "model_key": model_key,
                "dimension": dimension,
                "value": value,
                "n": n,
                "helper": EPIS_AGGREGATE_HELPER,
            })
    return rows


def _member_score_rows(
    run_dir_by_path: dict[str, Path],
    member_id_by_path: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Return ``(rows_by_path, units_by_path)`` from ``score_rows`` per member."""
    rows_by_path: dict[str, dict[str, Any]] = {}
    units_by_path: dict[str, dict[str, dict[str, Any]]] = {}
    for path in member_id_by_path:
        run_dir = run_dir_by_path.get(path)
        if run_dir is None:
            continue
        try:
            result = compute_score_rows(run_dir)
        except Exception as exc:
            raise ValueError(f"could not derive score rows for bundle member {member_id_by_path[path]}") from exc
        rows_by_path[path] = result
        units_by_path[path] = {
            str(entry.get("unit_id")): entry for entry in result.get("units") or []
        }
    return rows_by_path, units_by_path


def _scores_union(
    union_units: list[dict[str, Any]],
    rows_by_path: dict[str, dict[str, Any]],
    member_id_by_path: dict[str, str],
) -> list[dict[str, Any]]:
    """Union of members' ``score_rows``, filtered to each unit's union winner."""
    winner_by_unit = {str(u.get("unit_id")): u.get("chosen_member") for u in union_units}
    out: list[dict[str, Any]] = []
    for path, member_id in member_id_by_path.items():
        result = rows_by_path.get(path)
        if result is None:
            continue
        for row in result.get("rows") or []:
            if winner_by_unit.get(str(row.get("unit_id"))) == path:
                out.append(_project_score_row_for_bundle({**row, "member_id": member_id}))
    return out


def _outcomes(
    union_units: list[dict[str, Any]],
    units_by_path: dict[str, dict[str, dict[str, Any]]],
    member_id_by_path: dict[str, str],
) -> list[dict[str, Any]]:
    """One outcome record per expected unit from ``union()``."""
    out: list[dict[str, Any]] = []
    for unit in union_units:
        member_path = unit.get("chosen_member")
        member_id = member_id_by_path.get(member_path, "external")
        unit_entry = units_by_path.get(member_path, {}).get(str(unit.get("unit_id")), {})
        outcome_class = unit_entry.get("outcome_class") or unit.get("state")
        record: dict[str, Any] = {
            "unit_id": unit.get("unit_id"),
            "outcome_class": outcome_class,
            "member_id": member_id,
            "attempt": unit.get("attempt"),
        }
        if unit_entry.get("category") is not None:
            record["category"] = unit_entry["category"]
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# RunSnapshot — capture-once, consume-the-snapshot, recheck-before-promote (D6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FileFingerprint:
    """(name, exists, size, sha256) of one input file at capture time.

    ``name`` is the run-dir-relative path (may be nested, e.g.
    ``transcripts/x.json``) so distinct artifacts never collide on basename.
    """
    name: str
    exists: bool
    size: int
    sha256: str

    @classmethod
    def of(cls, path: Path, name: str) -> "_FileFingerprint":
        try:
            raw = path.read_bytes()
        except (OSError, FileNotFoundError):
            return cls(name, False, 0, "")
        return cls(name, True, len(raw), hashlib.sha256(raw).hexdigest())


@dataclass
class RunSnapshot:
    """An immutable per-member view captured at gate time (plan 020 D6).

    The projection and every gate clause consume THIS object, never the
    filesystem.  ``fingerprints`` are re-verified immediately before the promote
    so an attempt started (or any ledger line appended) mid-bundle aborts the
    emission with staging cleanup.
    """
    member_id: str
    run_dir: Path
    run_status: dict[str, Any]
    fingerprints: dict[str, _FileFingerprint]
    projection: Any                       # review_projection.ProjectionResult
    reviews_by_ref: dict[str, list[dict[str, Any]]]
    request_conformance: dict[str, Any] | None = None
    artifact_identity: dict[str, Any] | None = None


# Ledger/review files whose bytes define "the run state" for drift detection.
_SNAPSHOT_LEDGER_FILES = (
    RUN_STATUS_FILENAME, BLOCKS_FILENAME, RUN_EVENTS_FILENAME,
    BLOCK_REVIEWS_FILENAME,
)
# Score-summary files read by score_rows / winning_records AFTER capture.
_SNAPSHOT_SCORE_FILES = (
    "FINAL_RESULTS.json", "FINAL_RESULTS-conversations.json",
)


def _member_input_files(run_dir: Path) -> list[str]:
    """Every input file read AFTER capture whose drift must fail the promote.

    Beyond the ledger/review files this includes ``RUN_CONTRACT.json``, the
    score-summary files, and each expected artifact/transcript/score/summary
    path the contract declares — the full set ``union``, ``score_rows``,
    ``project()`` and ``_winning_records`` re-read while the bundle is built
    (finding 5c).  Returned as run-dir-relative names (dedup, order-stable).
    """
    names: list[str] = [*_SNAPSHOT_LEDGER_FILES, CONTRACT_FILENAME,
                        *_SNAPSHOT_SCORE_FILES]
    contract = _load_json(run_dir / CONTRACT_FILENAME)
    if isinstance(contract, dict):
        for module_entry in contract.get("modules") or []:
            if not isinstance(module_entry, dict):
                continue
            for unit in module_entry.get("expected_units") or []:
                if not isinstance(unit, dict):
                    continue
                for key in ("expected_transcript_path", "expected_score_path",
                            "expected_summary_path"):
                    rel = unit.get(key)
                    if rel:
                        names.append(str(rel))
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _fingerprint_member(run_dir: Path) -> dict[str, _FileFingerprint]:
    """Fingerprint every input file a member contributes to a bundle build."""
    return {
        name: _FileFingerprint.of(run_dir / name, name)
        for name in _member_input_files(run_dir)
    }


def _project_run_snapshot(
    run_dir: Path, member_id: str, fingerprints: dict[str, _FileFingerprint],
) -> RunSnapshot:
    """Project a member (fail-closed) into a RunSnapshot with *fingerprints*.

    A projection error (drifted line ref, ambiguous backfill, malformed review,
    supersession-integrity violation) propagates so a run whose effective state
    cannot be computed is never published.
    """
    from suite_tools import review_projection as _rp  # noqa: PLC0415

    run_status = _load_json(run_dir / RUN_STATUS_FILENAME)
    if not isinstance(run_status, dict):
        run_status = {}
    projection = _rp.project(run_dir)                 # may raise → abort
    reviews = _rp.load_reviews(run_dir)
    facts = _rp.load_facts(run_dir)
    reviews_by_ref = _rp.reviews_by_ref(reviews, facts)  # may raise → abort
    from suite_tools.request_receipts import evaluate_request_conformance  # noqa: PLC0415
    from suite_tools.artifact_identity import evaluate_run_artifact_identity  # noqa: PLC0415

    request_conformance = evaluate_request_conformance(run_dir)
    contract = _load_json(run_dir / CONTRACT_FILENAME)
    artifact_identity = evaluate_run_artifact_identity(
        run_dir,
        contract=contract if isinstance(contract, dict) else {},
    )
    return RunSnapshot(
        member_id=member_id,
        run_dir=run_dir,
        run_status=run_status,
        fingerprints=fingerprints,
        projection=projection,
        reviews_by_ref=reviews_by_ref,
        request_conformance=request_conformance,
        artifact_identity=artifact_identity,
    )


def _capture_run_snapshot(run_dir: Path, member_id: str) -> RunSnapshot:
    """Fingerprint every input file and project once (fail-closed).

    Fingerprints cover the ledger/review files AND the contract + artifact/score
    files that ``union``/``score_rows``/``project()``/``_winning_records`` read
    after capture, so a mutation to any of them before the promote is caught by
    the recheck (finding 5c).
    """
    return _project_run_snapshot(run_dir, member_id, _fingerprint_member(run_dir))


def _verify_member_fingerprints(
    member_id: str, run_dir: Path, fingerprints: dict[str, _FileFingerprint], *,
    when: str,
) -> None:
    """Raise if any of *fingerprints* no longer matches the file on disk."""
    for name, captured in fingerprints.items():
        current = _FileFingerprint.of(run_dir / name, name)
        if current != captured:
            raise ValueError(
                f"RunSnapshot drift for member {member_id}: {name} changed "
                f"{when} (size {captured.size}->{current.size}, "
                f"sha {captured.sha256[:8] or '-'}->{current.sha256[:8] or '-'}). "
                f"An input (attempt, ledger, contract, or artifact) changed "
                f"mid-bundle; re-run the bundle."
            )


def _recheck_run_snapshots(snapshots: list[RunSnapshot]) -> None:
    """Re-verify every captured fingerprint immediately before the promote.

    Covers EVERY active member (not just contributing winners) so a mutation to
    a losing member that would have flipped winner selection also aborts.  Any
    drift raises so the caller removes the staging dir — no partial or stale
    bundle escapes.
    """
    for snap in snapshots:
        _verify_member_fingerprints(
            snap.member_id, snap.run_dir, snap.fingerprints,
            when="between capture and the promote",
        )


# ---------------------------------------------------------------------------
# Publication gate (plan 020 D6) — three clauses, no bypass
# ---------------------------------------------------------------------------


def _gate_relevant(fv: Any, won: set[str]) -> bool:
    """A fact gates a contributing member iff it is a won unit's fact, a
    member-scoped run fact, or an unmappable-legacy fact (conservative)."""
    if fv.scope == "unit":
        return fv.unit_id in won
    return fv.scope in ("member", "unmappable_legacy")


def _run_publication_gate(
    snapshots: list[RunSnapshot],
    won_by_member_id: dict[str, set[str]],
) -> None:
    """Fail (before the promote) if any clause of the D6 hard gate is tripped.

    Clauses (over CONTRIBUTING members only — each contributes >=1 winning unit):
      a. any included fact is unknown-class with no active resolving review;
      b. any included fact has an active needs_escalation review (any class);
      c. any winning unit is effectively owed / pending_retry (not publishable),
         any member-level retry obligation is pending, or any instrument_defect
         fact touches the member.
    Members whose RUN_STATUS is present-and-not-completed are refused outright.
    """
    from suite_tools.review_projection import (  # noqa: PLC0415
        RESOLVING_DISPOSITIONS,
        UNKNOWN_CLASSES,
        is_publication_blocking as _is_pub_blocking,
    )

    offenders: list[tuple[str, str | None, str | None, str]] = []

    for snap in snapshots:
        member_id = snap.member_id
        won = won_by_member_id.get(member_id, set())

        # RunSnapshot: refuse any run that is not terminal-completed at capture
        # (fail-closed).  A real run always carries RUN_STATUS.status ("completed"
        # once RunMonitor.mark_completed runs; a failure/running status otherwise),
        # so a missing/unreadable status means the run never finished — never
        # publish it.  The fingerprint recheck is the additional race guard.
        status = (snap.run_status or {}).get("status")
        if status != "completed":
            if status:
                reason = f"run not terminal-completed (RUN_STATUS.status={status!r})"
            else:
                reason = ("run not terminal-completed "
                          "(RUN_STATUS.status absent or unreadable)")
            offenders.append((member_id, None, None, reason))
            continue

        request_conformance = snap.request_conformance or {}
        if request_conformance and not request_conformance.get("conformant", False):
            issue_count = len(request_conformance.get("issues") or [])
            offenders.append(
                (
                    member_id,
                    None,
                    None,
                    "effective requests do not conform to RUN_CONTRACT.json "
                    f"({issue_count} issue(s))",
                )
            )
            continue

        artifact_identity = snap.artifact_identity or {}
        if artifact_identity and not artifact_identity.get("conformant", False):
            issue_count = len(artifact_identity.get("issues") or [])
            offenders.append(
                (
                    member_id,
                    None,
                    None,
                    "saved transcript identities do not conform to "
                    f"RUN_CONTRACT.json ({issue_count} issue(s))",
                )
            )
            continue

        proj = snap.projection

        for ref, fv in proj.events_by_ref.items():
            if _gate_relevant(fv, won):
                # Single authoritative predicate shared with bench review's
                # gate_blocking field and bench blockers' suppression logic so
                # the three surfaces cannot drift (finding 7).  Covers clauses a
                # (unknown), b (needs_escalation), c (instrument_defect).
                blocking_reason = _is_pub_blocking(fv)
                if blocking_reason:
                    offenders.append((member_id, fv.unit_id, ref, blocking_reason))
            elif fv.effective_class == "instrument_defect":
                # Clause c is member-scoped, NOT winner-scoped: an instrument
                # defect on a unit this member LOST still taints the whole
                # contributing member and blocks publication (finding 3, D6 c3).
                offenders.append((
                    member_id, fv.unit_id, ref,
                    "instrument_defect fact (blocks until fixed and re-run)",
                ))

        # clause c — winning units must be scored or a legitimate declination.
        # A winning unit with NO UnitView (projection dropped/never produced it)
        # is NOT presumed publishable — fail closed and name it (finding 5b).
        for uid in sorted(won):
            uv = proj.units_by_id.get(uid)
            if uv is None:
                offenders.append((
                    member_id, uid, None,
                    "winning unit has no projected state (unresolved/dropped)",
                ))
                continue
            if uv.state in _PUBLISHABLE_UNIT_STATES:
                continue
            carrier_ref = uv.carrier.event_ref if uv.carrier else None
            offenders.append((
                member_id, uid, carrier_ref,
                f"winning unit effectively owed (state={uv.state})",
            ))

        # clause c — a pending member-level retry obligation gates the member
        for obligation in proj.member_obligations:
            if not obligation.fulfilled:
                offenders.append((
                    member_id, None, obligation.event_ref,
                    "member-level pending retry obligation",
                ))

    if not offenders:
        return

    seen: set[tuple] = set()
    lines: list[str] = []
    for offender in offenders:
        if offender in seen:
            continue
        seen.add(offender)
        member_id, unit_id, event_ref, reason = offender
        lines.append(
            f"  - (member={member_id}, unit_id={unit_id}, "
            f"event_ref={event_ref}): {reason}"
        )
    raise ValueError(
        "Publication gate failed — unresolved facts / owed units block this "
        "bundle (no bypass):\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Allowlist projection of facts + reviews (plan 020 D2, D7)
# ---------------------------------------------------------------------------


def _project_allowlisted(
    record: dict[str, Any], fields: tuple[str, ...], *, member_id: str,
    event_ref: str | None = None,
) -> dict[str, Any]:
    """Keep only *fields*; stamp member_id (+ event_ref) for the public record.

    Every allowlisted field is a scalar EXCEPT ``unit`` — so each nominally-scalar
    field is coerced (dict/list values dropped, strings capped) to defeat nested
    smuggling (finding 5), and ``unit`` is projected to its scalar identity keys,
    or dropped entirely if it is not even a dict (finding 5/6).
    """
    out: dict[str, Any] = {}
    for key in fields:
        if key not in record:
            continue
        if key == "unit":
            projected = _project_unit_identity(record[key])
            if projected is not None:      # a non-dict unit is dropped wholesale
                out[key] = projected
        elif _is_scalar(record[key]):
            out[key] = _coerce_public_scalar(record[key])   # caps strings
        # else: a dict/list in a nominally-scalar field is dropped entirely —
        # it can only be a nested-content smuggling vector (finding 5).
    out["member_id"] = member_id
    if event_ref is not None:
        out["event_ref"] = event_ref
    return out


def _published_facts_for_member(
    snap: RunSnapshot, won: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Return ``(blocks, evidence, published_event_refs)`` for one member.

    * blocks — winner-filtered BLOCKS facts (unit_id in *won*), allowlisted.
    * evidence — attempt_failure_classified facts that are member-scoped or
      won-unit-scoped, allowlisted (raw_body_excerpt + failure_reason dropped).
    Both come from the captured projection, so the bundle agrees with the
    projection every consumer sees.
    """
    blocks: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    published_refs: set[str] = set()
    for ref, fv in snap.projection.events_by_ref.items():
        if fv.source == "blocks":
            if fv.scope == "unit" and fv.unit_id in won:
                blocks.append(_project_allowlisted(
                    fv.fact, _BLOCK_PUBLIC_FIELDS,
                    member_id=snap.member_id, event_ref=ref))
                published_refs.add(ref)
        else:  # events source (attempt_failure_classified)
            if (fv.scope == "member") or (fv.scope == "unit" and fv.unit_id in won):
                evidence.append(_project_allowlisted(
                    fv.fact, _EVENT_PUBLIC_FIELDS,
                    member_id=snap.member_id, event_ref=ref))
                published_refs.add(ref)
    return blocks, evidence, published_refs


def _published_reviews_for_member(
    snap: RunSnapshot, published_refs: set[str], *, include_rationale: bool,
) -> list[dict[str, Any]]:
    """Full supersession chains for every review targeting a published fact,
    allowlisted; rationale is included only under the opt-in."""
    reviews: list[dict[str, Any]] = []
    for ref, chain in snap.reviews_by_ref.items():
        if ref not in published_refs:
            continue
        for review in chain:
            projected = _project_allowlisted(
                review, _REVIEW_PUBLIC_FIELDS,
                member_id=snap.member_id, event_ref=ref)
            if include_rationale and review.get("rationale") is not None:
                projected["rationale"] = review["rationale"]
            reviews.append(projected)
    return reviews


def _projected_bundle_facts(
    snapshots: list[RunSnapshot],
    won_by_member_id: dict[str, set[str]],
    *,
    include_rationale: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build ``(blocks, evidence, block_reviews)`` from the captured snapshots."""
    blocks: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for snap in snapshots:
        won = won_by_member_id.get(snap.member_id, set())
        mem_blocks, mem_evidence, refs = _published_facts_for_member(snap, won)
        blocks.extend(mem_blocks)
        evidence.extend(mem_evidence)
        reviews.extend(_published_reviews_for_member(
            snap, refs, include_rationale=include_rationale))
    return blocks, evidence, reviews


# ---------------------------------------------------------------------------
# Member-reference rewriting (paths -> bundle-local ids)
# ---------------------------------------------------------------------------


def _rewrite_union(
    union_result: dict[str, Any],
    member_id_by_path: dict[str, str],
) -> dict[str, Any]:
    """Rewrite every member path in a ``union()`` result to a bundle-local id."""

    def rw(path: Any) -> str:
        return member_id_by_path.get(path, "external")

    units = [
        {
            **{k: v for k, v in unit.items() if k not in ("chosen_member", "candidates")},
            "chosen_member": rw(unit.get("chosen_member")),
            "candidates": [
                {**{k: v for k, v in c.items() if k != "member"}, "member": rw(c.get("member"))}
                for c in unit.get("candidates") or []
            ],
        }
        for unit in union_result.get("units") or []
    ]
    collisions = [
        {
            **{k: v for k, v in coll.items() if k not in ("kept_member", "dropped_members")},
            "kept_member": rw(coll.get("kept_member")),
            "dropped_members": [rw(m) for m in coll.get("dropped_members") or []],
        }
        for coll in union_result.get("collisions") or []
    ]
    warnings = [
        {**{k: v for k, v in warn.items() if k != "member_path"}, "member_path": rw(warn.get("member_path"))}
        for warn in union_result.get("warnings") or []
    ]
    member_errors = [
        {**{k: v for k, v in err.items() if k != "path"}, "member": rw(err.get("path"))}
        for err in union_result.get("member_errors") or []
    ]
    return {
        "schema_version": union_result.get("schema_version"),
        "units": units,
        "collisions": collisions,
        "warnings": warnings,
        "member_errors": member_errors,
    }


# ---------------------------------------------------------------------------
# Writers (each payload gated before write)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    assert_public_artifact_safe(payload)
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        assert_public_artifact_safe(record)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str, allow_nan=False) + "\n")


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert_public_artifact_safe(rows)
    if not rows:
        path.write_text("")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: (value if isinstance(value, (str, int, float, bool)) or value is None
                  else json.dumps(value, default=str, allow_nan=False))
            for key, value in row.items()
        })
    text = buffer.getvalue()
    _assert_text_public_safe(text)  # gate serialized CSV text before write
    path.write_text(text)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


def _certificate(projected_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pairwise comparability certificate across members' projected contracts."""
    ids = sorted(projected_by_id, key=lambda member_id: int(member_id[1:]))
    pairwise: list[dict[str, Any]] = []
    for id_a, id_b in itertools.combinations(ids, 2):
        comparison = compare_provenance(projected_by_id[id_a], projected_by_id[id_b])
        universe = item_universe_report(projected_by_id[id_a], projected_by_id[id_b])
        pairwise.append({
            "member_a": id_a,
            "member_b": id_b,
            **comparison,
            "item_universe": universe,
        })
    return {"member_count": len(ids), "pairwise": pairwise}


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _assert_no_dropped_members(
    union_result: dict[str, Any],
    member_id_by_path: dict[str, str],
) -> None:
    """Abort (fail closed) if ``union`` dropped or could not verify any active
    member (finding 5a).

    ``member_errors`` (a member's ``owed_units``/projection raised — e.g. an
    invalid review, a supersession-integrity violation, an ambiguous backfill)
    and ``warnings`` (missing or fingerprint-mismatched contract) both mean an
    active member's units silently vanished from ``union``.  Emitting the
    remainder would ship a bundle that quietly omits real, unresolved data — so
    the whole emission is refused, naming every offender.
    """
    def _label(path: Any) -> str:
        return member_id_by_path.get(path, f"external:{path}")

    offenders: list[str] = []
    for err in union_result.get("member_errors") or []:
        offenders.append(
            f"  - (member={_label(err.get('path'))}): union error: "
            f"{err.get('error')}"
        )
    for warn in union_result.get("warnings") or []:
        offenders.append(
            f"  - (member={_label(warn.get('member_path'))}): union warning: "
            f"{warn.get('reason')}"
        )
    if offenders:
        raise ValueError(
            "Publication gate failed — union dropped or could not verify an "
            "active member; refusing to ship a silently-shrunk bundle "
            "(no bypass):\n" + "\n".join(offenders)
        )


def _contract_uses_sealed_aita_pack(contract: dict[str, Any]) -> bool:
    for module in contract.get("modules") or []:
        if not isinstance(module, dict) or module.get("module") != "aita":
            continue
        dataset_manifest = module.get("dataset_manifest")
        if (
            isinstance(dataset_manifest, dict)
            and dataset_manifest.get("distribution_mode") == "sealed_public_pack"
        ):
            return True
    return False


def _build_bundle(
    staging: Path,
    exp_dir: Path,
    experiment_id: str,
    version: int,
    *,
    include_transcripts: bool,
    include_review_rationale: bool,
    write_report: bool,
) -> list[RunSnapshot]:
    staging.mkdir(parents=True)
    data_dir = staging / "data"
    prov_dir = staging / "provenance"
    data_dir.mkdir()
    prov_dir.mkdir()

    # --- pull EXPERIMENT.json into the fingerprint boundary ----------------
    # The member list is parsed from the SAME bytes that are fingerprinted, and
    # union() independently re-reads the experiment — so the manifest is verified
    # (below) to be identical to what union consumed.  A member added between the
    # initial read and union() can no longer slip in an ungated winner, and the
    # pre-promote recheck covers the manifest too.
    exp_manifest_name = _experiment_mod.EXPERIMENT_FILENAME
    exp_bytes = (exp_dir / exp_manifest_name).read_bytes()
    manifest_fingerprint = _FileFingerprint(
        name=exp_manifest_name, exists=True, size=len(exp_bytes),
        sha256=hashlib.sha256(exp_bytes).hexdigest(),
    )
    manifest_doc = json.loads(exp_bytes)
    members = [m for m in (manifest_doc.get("members") or []) if isinstance(m, dict)]

    member_id_by_path: dict[str, str] = {}
    run_dir_by_path: dict[str, Path] = {}
    active_members: list[tuple[str, str, Path]] = []   # (path, member_id, run_dir)
    for index, member in enumerate(members):
        path = member.get("path")
        member_id = f"m{index + 1}"
        is_active = not member.get("superseded_by")
        if is_active:
            contract_path = Path(path) if isinstance(path, str) and path.strip() else None
            if (
                contract_path is None
                or path != path.strip()
                or not contract_path.is_absolute()
                or contract_path.name != CONTRACT_FILENAME
                or contract_path.is_symlink()
                or not contract_path.is_file()
            ):
                raise ValueError(
                    "Publication gate failed — active member "
                    f"{member_id} path must be an absolute, nonempty string "
                    "naming an existing regular RUN_CONTRACT.json; "
                    f"got {path!r}."
                )
        if not isinstance(path, str) or not path:
            continue
        member_id_by_path[path] = member_id
        run_dir = Path(path).parent
        run_dir_by_path[path] = run_dir
        # Active = the members union() will consider (same non-superseded rule).
        if is_active:
            active_members.append((path, member_id, run_dir))
    if not active_members:
        raise ValueError(
            "Publication gate failed — experiment has no active members; "
            "adopt a completed run before packaging."
        )

    # --- capture every ACTIVE member's inputs BEFORE union (finding 2) ------
    # union() picks winners by reading these same files; capturing first and
    # re-verifying after guarantees union's winner selection and the gate's
    # projection read the SAME bytes.  A member changing between selection and
    # capture could otherwise establish a new winner with unresolved evidence
    # that no captured fingerprint would flag.
    active_fingerprints: dict[str, dict[str, _FileFingerprint]] = {
        member_id: _fingerprint_member(run_dir)
        for _path, member_id, run_dir in active_members
    }

    union_result = _experiment_mod.union(exp_dir)
    union_units = union_result.get("units")

    # --- finding 2 + manifest: nothing changed during winner selection -----
    _verify_member_fingerprints(
        "<manifest>", exp_dir, {exp_manifest_name: manifest_fingerprint},
        when="during winner selection (union)",
    )
    for _path, member_id, run_dir in active_members:
        _verify_member_fingerprints(
            member_id, run_dir, active_fingerprints[member_id],
            when="during winner selection (union)",
        )

    # --- load projected contracts AFTER the fingerprint boundary -----------
    # (finding: stale contract read.)  RUN_CONTRACT.json is fingerprinted per
    # active member, so loading here — inside the verified-stable window and
    # covered by the pre-promote recheck — guarantees the published provenance
    # matches the fingerprinted bytes; no read window precedes the boundary.
    contract_by_path: dict[str, dict[str, Any]] = {}
    for path in member_id_by_path:
        contract = load_run_contract(Path(path))
        if contract:
            contract_by_path[path] = contract

    sealed_aita_member = any(
        _contract_uses_sealed_aita_pack(contract)
        for contract in contract_by_path.values()
    )
    if include_transcripts and sealed_aita_member:
        raise ValueError(
            "Publication gate failed - sealed AITA pack runs cannot publish raw transcripts; "
            "keep local review artifacts private or use a separately reviewed encrypted export."
        )
    if include_review_rationale and sealed_aita_member:
        raise ValueError(
            "Publication gate failed - sealed AITA pack runs cannot publish free-text review rationale; "
            "the numeric and categorical evidence projection remains available."
        )

    # --- fail closed on any broken ACTIVE member (finding 5a) --------------
    # union() catches a member's projection/owed_units error and drops that
    # member's units (or warns on a missing/tampered contract); publishing the
    # remainder would SILENTLY shrink the bundle and hide the broken member.
    _assert_no_dropped_members(union_result, member_id_by_path)
    if not isinstance(union_units, list) or not union_units:
        raise ValueError(
            "Publication gate failed — experiment union contains no units; "
            "refusing to emit an empty evidence bundle."
        )

    # --- contributing members + their winning unit ids ---------------------
    won_by_member_id: dict[str, set[str]] = {}
    for unit in union_units:
        member_path = unit.get("chosen_member")
        member_id = member_id_by_path.get(member_path)
        if member_id is None:
            # A union winner not in the captured manifest is an UNMAPPED winner
            # (a member added/changed mid-bundle) — never silently skipped; fail
            # closed naming it (finding: manifest race / unmapped winner).
            raise ValueError(
                "Publication gate failed — union selected a winner not in the "
                f"captured manifest (chosen_member={member_path!r}, "
                f"unit_id={unit.get('unit_id')!r}); the experiment changed "
                "mid-bundle. Re-run the bundle (no bypass)."
            )
        won_by_member_id.setdefault(member_id, set()).add(str(unit.get("unit_id")))

    # --- RunSnapshot per ACTIVE member (D6) --------------------------------
    # Contributing members are projected (they gate + publish); losing active
    # members carry fingerprints only — but ALL of them are rechecked before the
    # promote, so a late mutation to a loser that would flip a winner aborts.
    # Every snapshot reuses the pre-union fingerprints (verified stable above)
    # so union's inputs and the gate's inputs are provably the same bytes.
    all_snapshots: list[RunSnapshot] = []
    gate_snapshots: list[RunSnapshot] = []
    for _path, member_id, run_dir in active_members:
        fps = active_fingerprints[member_id]
        if member_id in won_by_member_id:
            snap = _project_run_snapshot(run_dir, member_id, fps)
            gate_snapshots.append(snap)
        else:
            snap = RunSnapshot(
                member_id=member_id, run_dir=run_dir, run_status={},
                fingerprints=fps, projection=None, reviews_by_ref={},
            )
        all_snapshots.append(snap)

    # The manifest itself rides the pre-promote recheck (finding: manifest race).
    all_snapshots.append(RunSnapshot(
        member_id="<manifest>", run_dir=exp_dir, run_status={},
        fingerprints={exp_manifest_name: manifest_fingerprint},
        projection=None, reviews_by_ref={},
    ))

    # --- three-clause hard gate (D6): abort BEFORE any write ---------------
    _run_publication_gate(gate_snapshots, won_by_member_id)

    # --- winning records (GATE here; poison aborts before any write) ---
    winning_records = _winning_records(
        union_units,
        contract_by_path=contract_by_path,
        run_dir_by_path=run_dir_by_path,
    )
    derived_aggregates = _derived_aggregates(winning_records)

    rows_by_path, units_by_path = _member_score_rows(run_dir_by_path, member_id_by_path)
    scores = _scores_union(union_units, rows_by_path, member_id_by_path)
    outcomes = _outcomes(union_units, units_by_path, member_id_by_path)
    # Allowlisted, winner/member-scoped facts + judgment trail (D2, D7).
    blocks, evidence, block_reviews = _projected_bundle_facts(
        gate_snapshots, won_by_member_id, include_rationale=include_review_rationale,
    )

    # --- projected contracts (GATE + write) ---
    projected_by_id: dict[str, dict[str, Any]] = {}
    for path, member_id in member_id_by_path.items():
        contract = contract_by_path.get(path)
        if contract is None:
            continue
        projected = _project_contract_for_bundle(contract)
        _write_json(prov_dir / f"RUN_CONTRACT-{member_id}.json", projected)
        projected_by_id[member_id] = projected

    certificate = _certificate(projected_by_id)

    # --- data files ---
    _write_jsonl(data_dir / "scores.jsonl", scores)
    _write_scores_csv(data_dir / "scores.csv", scores)
    _write_jsonl(data_dir / "derived_aggregates.jsonl", derived_aggregates)
    _write_jsonl(data_dir / "outcomes.jsonl", outcomes)
    _write_jsonl(data_dir / "blocks.jsonl", blocks)
    _write_jsonl(data_dir / "evidence.jsonl", evidence)
    _write_jsonl(data_dir / "block_reviews.jsonl", block_reviews)

    # --- manifest (member references rewritten to bundle-local ids) ---
    # Hash the complete data/provenance payload after every payload file is written.
    # The edition manifest can then pin this one file and transitively pin all
    # analysis data and projected provenance. Human-readable reports remain
    # reproducible views over that pinned payload.
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "title": manifest_doc.get("title"),
        "version": version,
        "projection_version": IDENTITY_PROJECTION_VERSION,
        "instrument": manifest_doc.get("instrument"),
        "conditions": manifest_doc.get("conditions"),
        "target": manifest_doc.get("target"),
        "members": [
            {
                "member_id": f"m{index + 1}",
                "role": member.get("role"),
                "contract_fingerprint": member.get("contract_fingerprint"),
            }
            for index, member in enumerate(members)
        ],
        "union": _rewrite_union(union_result, member_id_by_path),
        "certificate": certificate,
        "exclusion_policy": EXCLUSION_POLICY,
        "contains_transcripts": include_transcripts,
        "contains_review_rationale": include_review_rationale,
        "payload_files": _payload_hash_entries(staging),
        "tool_version": _tool_version(),
        "created_at": utc_now(),
    }
    _write_json(staging / "BUNDLE_MANIFEST.json", manifest)

    if write_report:
        _write_report(staging, manifest)
        from suite_tools.bundle_report import write_bundle_report as _write_html_report
        _write_html_report(staging)

    if include_transcripts:
        member_id_by_run_dir = {
            str(run_dir): member_id_by_path[contract_path]
            for contract_path, run_dir in run_dir_by_path.items()
        }
        _write_transcript_review(
            staging,
            list(run_dir_by_path.values()),
            member_id_by_run_dir,
            experiment_id,
            write_report=write_report,
        )

    # Return EVERY active snapshot (contributing + losing) so the pre-promote
    # recheck (in emit) covers all winner-selection inputs (finding 2).
    return all_snapshots


def _normalize_review_path(
    path_str: str | None,
    module: str,
    member_id_by_run_dir: dict[str, str],
) -> str | None:
    """Rewrite an absolute run-dir path to a ``{member_id}/{module}/{basename}`` label."""
    if not path_str:
        return path_str
    for run_dir_str, member_id in member_id_by_run_dir.items():
        if path_str.startswith(run_dir_str + "/") or path_str == run_dir_str:
            return f"{member_id}/{module}/{Path(path_str).name}"
    return path_str  # not under any known run dir; leave as-is


def _normalize_review_record_paths(
    record: dict[str, Any],
    member_id_by_run_dir: dict[str, str],
) -> dict[str, Any]:
    """Rewrite source_path, score_path, and metadata.run_contract_provenance.path."""
    module = record.get("module") or "generic"

    def norm(path_str: str | None) -> str | None:
        return _normalize_review_path(path_str, module, member_id_by_run_dir)

    result = dict(record)
    result["source_path"] = norm(record.get("source_path"))
    result["score_path"] = norm(record.get("score_path"))

    metadata = dict(record.get("metadata") or {})
    prov = metadata.get("run_contract_provenance")
    if isinstance(prov, dict) and prov.get("path"):
        prov = dict(prov)
        prov["path"] = norm(prov["path"])
        metadata["run_contract_provenance"] = prov
    result["metadata"] = metadata

    return result


def _write_transcript_review(
    staging: Path,
    run_dirs: list[Path],
    member_id_by_run_dir: dict[str, str],
    experiment_id: str,
    *,
    write_report: bool,
) -> None:
    """Write report/review.html and (optionally) link it from report/index.html.

    Single load: ``load_review_records`` is called once; absolute run-dir paths in
    ``source_path``, ``score_path``, and ``metadata.run_contract_provenance.path``
    are rewritten to member-relative ``{member_id}/{module}/{basename}`` labels before
    the pre-render scan so they cannot trigger ``ABSOLUTE_HOME_PATH_RE``.  The scan
    then enforces ``SECRET_VALUE_RE``, ``PRIVATE_HOST_RE``, and
    ``ABSOLUTE_HOME_PATH_RE`` on every normalized record.  ``render_review_html`` is
    called directly (the scanned records ARE the rendered records) and the HTML text
    is gated through ``_assert_text_public_safe`` before writing.  The function is
    called inside the atomic staging block so any abort leaves no partial bundle.
    """
    from suite_tools.review_viewer import load_review_records, render_review_html

    # Load once and normalize absolute-path fields to member-relative labels.
    records = [
        _normalize_review_record_paths(r, member_id_by_run_dir)
        for r in load_review_records(run_dirs)
    ]

    # Pre-render privacy scan on the normalized records.
    for record in records:
        record_json = json.dumps(record, default=str, allow_nan=False)
        if SECRET_VALUE_RE.search(record_json):
            raise ValueError(
                "pre-render transcript scan: secret-looking value in review records"
            )
        if PRIVATE_HOST_RE.search(record_json):
            raise ValueError(
                "pre-render transcript scan: private/internal host in review records"
            )
        if ABSOLUTE_HOME_PATH_RE.search(record_json):
            raise ValueError(
                "pre-render transcript scan: absolute home path in review records"
            )

    # Render directly from the scanned records and gate the HTML text.
    html_text = render_review_html(records, title=f"Transcript Review: {experiment_id}")
    _assert_text_public_safe(html_text)

    review_output = staging / "report" / "review.html"
    review_output.parent.mkdir(parents=True, exist_ok=True)
    review_output.write_text(html_text, encoding="utf-8")

    # Add a link from index.html to review.html when the report was also written.
    if write_report:
        index_path = staging / "report" / "index.html"
        if index_path.exists():
            index_text = index_path.read_text()
            link_section = (
                "\n<p><strong>Transcript review:</strong>"
                ' <a href="review.html">review.html</a>'
                " — raw conversation content.</p>"
            )
            index_path.write_text(
                index_text.replace("</body>", link_section + "\n</body>", 1)
            )


def _write_report(staging: Path, manifest: dict[str, Any]) -> None:
    """Write a minimal, path-free markdown summary of the bundle."""
    union = manifest.get("union") or {}
    excl = manifest.get("exclusion_policy")
    excl_id = excl.get("id") if isinstance(excl, dict) else excl
    lines = [
        f"# Bundle {manifest.get('experiment_id')} v{manifest.get('version')}",
        "",
        f"- title: {manifest.get('title')}",
        f"- members: {len(manifest.get('members') or [])}",
        f"- units: {len(union.get('units') or [])}",
        f"- collisions: {len(union.get('collisions') or [])}",
        f"- exclusion_policy: {excl_id}",
        f"- contains_transcripts: {str(manifest.get('contains_transcripts')).lower()}",
        f"- contains_review_rationale: {str(manifest.get('contains_review_rationale')).lower()}",
        f"- projection_version: {manifest.get('projection_version')}",
        f"- tool_version: {manifest.get('tool_version')}",
        "",
    ]
    text = "\n".join(lines)
    _assert_text_public_safe(text)  # gate serialized markdown text before write
    (staging / "REPORT.md").write_text(text)


def emit(
    exp_dir: Path | str,
    out_dir: Path | str,
    *,
    include_transcripts: bool = False,
    include_review_rationale: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    """Emit a self-contained bundle for the experiment at *exp_dir* into *out_dir*.

    Writes ``bundle-{experiment_id}-v{N}/`` (N auto-increments on existing).  The
    tree is fully written under a sibling ``.<name>.tmp/`` staging dir, audited via
    :func:`audit_bundle_tree`, and only then ``os.replace``d into place.  Any abort
    (privacy violation, audit failure, gate failure, or a RunSnapshot fingerprint
    drift caught immediately before the promote) removes the staging dir and
    leaves no partial bundle.

    ``include_review_rationale`` opts free-text review ``rationale`` into
    ``data/block_reviews.jsonl`` and stamps ``contains_review_rationale`` in the
    manifest (exact mirror of ``include_transcripts``); it never affects the
    always-local raw body excerpts.  There is NO gate-bypass flag (plan 020 D6).
    """
    exp_path = Path(exp_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path.resolve()

    manifest_raw = json.loads((exp_path / _experiment_mod.EXPERIMENT_FILENAME).read_text())
    experiment_id = _experiment_mod.validate_experiment_id(
        manifest_raw.get("experiment_id")
    )

    version = _next_version(out_path, experiment_id)
    final_name = f"bundle-{experiment_id}-v{version}"
    final_dir = out_path / final_name
    staging = out_path / f".{final_name}.tmp"

    for candidate in (final_dir, staging):
        if candidate.parent.resolve() != out_path or candidate.name in {"", ".", ".."}:
            raise ValueError("bundle output must be a direct child of the output directory")

    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError("bundle staging path must be a real directory")
        shutil.rmtree(staging)

    try:
        snapshots = _build_bundle(
            staging,
            exp_path,
            experiment_id,
            version,
            include_transcripts=include_transcripts,
            include_review_rationale=include_review_rationale,
            write_report=write_report,
        )
        issues = audit_bundle_tree(staging)
        if issues:
            sample = "; ".join(f"{issue.path}: {issue.reason}" for issue in issues[:5])
            raise ValueError(f"Bundle-tree privacy check failed: {sample}")
        # Immediately before the promote, re-verify every captured fingerprint:
        # an attempt started (or any ledger line appended) mid-bundle aborts here
        # with staging cleanup (plan 020 D6).
        _recheck_run_snapshots(snapshots)
        os.replace(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_dir": str(final_dir),
        "experiment_id": experiment_id,
        "version": version,
        "contains_transcripts": include_transcripts,
        "contains_review_rationale": include_review_rationale,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m suite_tools.bundle",
        description="Emit a self-contained, privacy-scrubbed experiment bundle.",
    )
    parser.add_argument("experiment_dir", type=Path, help="Experiment directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--include-transcripts", action="store_true",
                        help="Include conversation transcripts (review.html).")
    parser.add_argument("--include-review-rationale", action="store_true",
                        help="Include free-text review rationale in "
                             "data/block_reviews.jsonl (stamps the manifest).")
    parser.add_argument("--no-report", dest="write_report", action="store_false",
                        help="Skip the markdown summary report.")
    parser.add_argument("--json", dest="as_json", action="store_true")

    args = parser.parse_args(argv)

    if args.include_transcripts:
        import sys
        print(
            "\n*** WARNING: --include-transcripts — this bundle contains raw "
            "conversation content. Review before distributing. ***\n",
            file=sys.stderr,
        )

    if args.include_review_rationale:
        import sys
        print(
            "\n*** WARNING: --include-review-rationale — this bundle contains "
            "free-text reviewer notes. Review before distributing. ***\n",
            file=sys.stderr,
        )

    result = emit(
        args.experiment_dir,
        args.out,
        include_transcripts=args.include_transcripts,
        include_review_rationale=args.include_review_rationale,
        write_report=args.write_report,
    )
    if args.as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Wrote bundle: {result['bundle_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
