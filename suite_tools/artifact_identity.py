"""Condition-identity checks for saved and resumed benchmark artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from suite_tools.model_config import MODEL_CONDITION_METADATA_FIELDS
from suite_tools.run_contract import IDENTITY_PROJECTION_VERSION


class ArtifactIdentityError(ValueError):
    """Raised when a saved artifact cannot be reconciled with its run condition."""

    def __init__(
        self,
        context: str,
        *,
        missing_fields: list[str] | tuple[str, ...] = (),
        conflicting_fields: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.context = context
        self.missing_fields = tuple(sorted(missing_fields))
        self.conflicting_fields = tuple(sorted(conflicting_fields))
        details: list[str] = []
        if self.missing_fields:
            details.append(f"missing {', '.join(self.missing_fields)}")
        if self.conflicting_fields:
            details.append(f"conflicting {', '.join(self.conflicting_fields)}")
        super().__init__(f"{context}: artifact condition identity is invalid ({'; '.join(details)})")


def expected_condition_identity(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the condition fields that a rendered model configuration freezes."""
    return {
        field: copy.deepcopy(model_config[field])
        for field in MODEL_CONDITION_METADATA_FIELDS
        if field in model_config and model_config[field] is not None
    }


def reconcile_condition_identity(
    artifact: MutableMapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    context: str,
    restore_missing: bool,
) -> tuple[str, ...]:
    """Validate an artifact against a rendered condition and optionally restore gaps.

    Missing fields may be restored only from the already-rendered model condition.
    Existing conflicting values always fail closed. The caller decides whether a
    legacy/resumed artifact may be enriched in memory or must already be complete.
    """
    expected = expected_condition_identity(model_config)
    missing = [field for field in expected if artifact.get(field) is None]
    conflicting = [
        field
        for field, value in expected.items()
        if artifact.get(field) is not None and artifact.get(field) != value
    ]
    if conflicting or (missing and not restore_missing):
        raise ArtifactIdentityError(
            context,
            missing_fields=missing if not restore_missing else (),
            conflicting_fields=conflicting,
        )
    for field in missing:
        artifact[field] = copy.deepcopy(expected[field])
    return tuple(sorted(missing))


def _contract_conditions(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    identity = contract.get("identity")
    conditions = identity.get("model_conditions") if isinstance(identity, Mapping) else None
    if not isinstance(conditions, list):
        conditions = contract.get("expected_models")
    result: dict[str, Mapping[str, Any]] = {}
    for condition in conditions or []:
        if not isinstance(condition, Mapping):
            continue
        key = condition.get("key")
        if key:
            result[str(key)] = condition
    return result


def evaluate_run_artifact_identity(
    run_dir: str | Path,
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check expected transcript identity against the run contract.

    Only fields explicitly frozen by the contract are checked. Legacy contracts
    without a condition ID or condition hash remain visible as uncheckable rather
    than being retroactively failed.
    """
    run_path = Path(run_dir)
    if contract is None:
        contract = json.loads((run_path / "RUN_CONTRACT.json").read_text())
    conditions = _contract_conditions(contract)
    issues: list[dict[str, Any]] = []
    checked_artifacts = 0
    checkable_artifacts = 0
    uncheckable_artifacts = 0
    provenance = contract.get("provenance")
    current_projection = (
        isinstance(provenance, Mapping)
        and provenance.get("projection_version") == IDENTITY_PROJECTION_VERSION
    )

    for module in contract.get("modules") or []:
        if not isinstance(module, Mapping):
            continue
        module_name = str(module.get("module") or "unknown")
        for unit in module.get("expected_units") or []:
            if not isinstance(unit, Mapping):
                continue
            relative_path = unit.get("expected_transcript_path")
            if not relative_path:
                continue
            artifact_path = run_path / str(relative_path)
            if not artifact_path.is_file():
                continue
            checked_artifacts += 1
            model_key = str(unit.get("model_key") or "")
            condition = conditions.get(model_key)
            required_fields = (
                ("condition_id", "condition_hash", "route_hash")
                if current_projection
                else ("condition_id", "condition_hash")
            )
            missing_condition_fields = [
                field
                for field in required_fields
                if condition is None or condition.get(field) is None
            ]
            if missing_condition_fields:
                uncheckable_artifacts += 1
                if current_projection:
                    issues.append({
                        "kind": "uncheckable_condition_identity",
                        "module": module_name,
                        "unit_id": unit.get("unit_id"),
                        "path": str(relative_path),
                        "missing_fields": sorted(missing_condition_fields),
                    })
                continue
            # condition_hash already binds the resolved route; transcripts need
            # the compact id/hash pair, while current contracts must also carry
            # route_hash so the condition itself is independently auditable.
            expected = {
                field: condition[field]
                for field in ("condition_id", "condition_hash")
            }
            checkable_artifacts += 1
            try:
                artifact = json.loads(artifact_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                issues.append({
                    "kind": "unreadable_transcript_identity",
                    "module": module_name,
                    "unit_id": unit.get("unit_id"),
                    "path": str(relative_path),
                    "detail": str(exc),
                })
                continue
            if not isinstance(artifact, Mapping):
                issues.append({
                    "kind": "invalid_transcript_identity_container",
                    "module": module_name,
                    "unit_id": unit.get("unit_id"),
                    "path": str(relative_path),
                })
                continue
            for field, expected_value in expected.items():
                observed = artifact.get(field)
                if observed is None:
                    issues.append({
                        "kind": f"missing_{field}",
                        "module": module_name,
                        "unit_id": unit.get("unit_id"),
                        "path": str(relative_path),
                        "expected": expected_value,
                    })
                elif observed != expected_value:
                    issues.append({
                        "kind": f"conflicting_{field}",
                        "module": module_name,
                        "unit_id": unit.get("unit_id"),
                        "path": str(relative_path),
                        "expected": expected_value,
                        "observed": observed,
                    })

    return {
        "schema_version": "benchmark-artifact-identity-v1",
        "conformant": not issues,
        "checked_artifacts": checked_artifacts,
        "checkable_artifacts": checkable_artifacts,
        "uncheckable_artifacts": uncheckable_artifacts,
        "issues": issues,
    }


def require_run_artifact_identity(run_dir: str | Path) -> dict[str, Any]:
    """Return the identity report or fail before downstream paid work."""
    report = evaluate_run_artifact_identity(run_dir)
    if report["conformant"]:
        return report
    kinds = sorted({str(issue.get("kind")) for issue in report["issues"]})
    raise ValueError(
        "saved transcript identities do not conform to RUN_CONTRACT.json "
        f"({len(report['issues'])} issue(s): {', '.join(kinds)})"
    )
