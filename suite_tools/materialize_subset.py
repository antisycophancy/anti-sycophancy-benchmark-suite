"""Materialize a publication subset from a completed AITA or EPIS run.

The source contract and artifacts stay immutable. Selected unit artifacts are
copied byte-for-byte into a new derived run, while the new contract records the
excluded model keys and SHA-256 pins every copied source artifact. No provider
calls are made.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from suite_tools.request_receipts import evaluate_request_conformance
from suite_tools.run_contract import REPO_ROOT, write_run_contract
from suite_tools.run_monitor import atomic_write_json, utc_now

SCHEMA_VERSION = "benchmark-derived-subset-v1"
_SUPPORTED_MODULES = frozenset({"aita", "epis", "epistemic"})
_COMMAND_FIELDS = (
    "execute_command",
    "execute_steps",
    "execute_cwd",
    "execute_argv",
    "score_command",
    "score_steps",
    "score_cwd",
    "score_argv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _source_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def _safe_relative_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Artifact path must be a safe relative path: {raw!r}")
    return path


def _filter_final_results(
    payload: dict[str, Any],
    *,
    retained_model_keys: set[str],
) -> dict[str, Any]:
    filtered = copy.deepcopy(payload)
    scores = filtered.get("scores")
    if isinstance(scores, dict):
        filtered["scores"] = {
            key: value
            for key, value in scores.items()
            if isinstance(value, dict)
            and str(value.get("model") or "") in retained_model_keys
        }
    metadata = filtered.get("metadata")
    if isinstance(metadata, dict):
        models = metadata.get("models")
        if isinstance(models, list):
            metadata["models"] = [
                model for model in models if str(model) in retained_model_keys
            ]
        missing = metadata.get("missing_scores")
        if isinstance(missing, list):
            metadata["missing_scores"] = [
                item
                for item in missing
                if not isinstance(item, dict)
                or str(item.get("model") or item.get("model_key") or "")
                in retained_model_keys
            ]
    return filtered


def materialize(
    *,
    source_run_dir: Path,
    output_dir: Path,
    run_id: str,
    excluded_model_keys: set[str],
    reason: str,
) -> Path:
    """Create a score-ready derived subset without touching source artifacts."""
    source_run_dir = source_run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing derived run: {output_dir}")
    if not excluded_model_keys:
        raise ValueError("At least one excluded model key is required")

    contract_path = source_run_dir / "RUN_CONTRACT.json"
    status_path = source_run_dir / "RUN_STATUS.json"
    contract = json.loads(contract_path.read_text())
    status = json.loads(status_path.read_text())
    if status.get("status") != "completed" or status.get("validity") != "score_ready":
        raise ValueError("Source run must be completed and score_ready")

    modules = contract.get("modules") or []
    if len(modules) != 1 or not isinstance(modules[0], dict):
        raise ValueError("Derived subset requires exactly one source module")
    source_module = modules[0]
    module_name = str(source_module.get("module") or "").lower()
    if module_name not in _SUPPORTED_MODULES:
        raise ValueError(f"Unsupported derived subset module: {module_name!r}")

    expected_models = [
        copy.deepcopy(model)
        for model in contract.get("expected_models") or []
        if isinstance(model, dict)
    ]
    available_model_keys = {str(model.get("key") or "") for model in expected_models}
    missing_exclusions = excluded_model_keys - available_model_keys
    if missing_exclusions:
        raise ValueError(
            "Excluded model keys are absent from source contract: "
            + ", ".join(sorted(missing_exclusions))
        )
    retained_models = [
        model
        for model in expected_models
        if str(model.get("key") or "") not in excluded_model_keys
    ]
    retained_model_keys = {str(model.get("key") or "") for model in retained_models}
    if not retained_model_keys:
        raise ValueError("Derived subset cannot exclude every model condition")

    expected_units = [
        copy.deepcopy(unit)
        for unit in source_module.get("expected_units") or []
        if isinstance(unit, dict)
        and str(unit.get("model_key") or "") in retained_model_keys
    ]
    if not expected_units:
        raise ValueError("No expected units remain after model filtering")

    identity = copy.deepcopy(contract.get("identity") or {})
    identity_conditions = [
        condition
        for condition in identity.get("model_conditions") or []
        if isinstance(condition, dict)
        and str(condition.get("key") or "") in retained_model_keys
    ]
    if {str(item.get("key") or "") for item in identity_conditions} != retained_model_keys:
        raise ValueError("Source identity model conditions do not match expected_models")
    identity["model_conditions"] = identity_conditions
    identity["execution"] = {
        "runner": "suite_tools.materialize_subset",
        "derived": True,
        "provider_calls": 0,
    }

    source_hashes: dict[str, str] = {
        "RUN_CONTRACT.json": _sha256(contract_path),
        "RUN_STATUS.json": _sha256(status_path),
    }
    copied_paths: set[Path] = set()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for unit in expected_units:
            for field in ("expected_transcript_path", "expected_score_path"):
                if not unit.get(field):
                    continue
                relative = _safe_relative_path(unit[field])
                if relative in copied_paths:
                    continue
                source = source_run_dir / relative
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Selected unit artifact is missing: {source}"
                    )
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                source_digest = _sha256(source)
                if _sha256(destination) != source_digest:
                    raise ValueError(f"Copied artifact hash mismatch: {relative}")
                source_hashes[str(relative)] = source_digest
                copied_paths.add(relative)

        final_results_path = source_run_dir / "FINAL_RESULTS.json"
        expected_artifacts = [
            {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "diagnostic"},
            {"kind": "run_events", "path": "RUN_EVENTS.jsonl", "required_for": "diagnostic"},
            {
                "kind": "derived_provenance",
                "path": "DERIVED_PROVENANCE.json",
                "required_for": "promotion",
            },
        ]
        if final_results_path.is_file():
            final_results = json.loads(final_results_path.read_text())
            if not isinstance(final_results, dict):
                raise ValueError("FINAL_RESULTS.json must contain an object")
            filtered_results = _filter_final_results(
                final_results,
                retained_model_keys=retained_model_keys,
            )
            _write_json(stage / "FINAL_RESULTS.json", filtered_results)
            source_hashes["FINAL_RESULTS.json"] = _sha256(final_results_path)
            expected_artifacts.append(
                {
                    "kind": "final_results",
                    "path": "FINAL_RESULTS.json",
                    "required_for": "promotion",
                }
            )

        derived_module = copy.deepcopy(source_module)
        derived_module["output_dir"] = "."
        derived_module["expected_units"] = expected_units
        derived_module["expected_artifacts"] = expected_artifacts

        derived_from = {
            "schema_version": SCHEMA_VERSION,
            "source_run_dir": _source_label(source_run_dir),
            "source_run_id": contract.get("run_id"),
            "excluded_model_keys": sorted(excluded_model_keys),
            "retained_model_keys": sorted(retained_model_keys),
            "reason": reason,
            "source_hashes": dict(sorted(source_hashes.items())),
            "copy_policy": "selected unit artifacts copied byte-for-byte",
            "provider_calls": 0,
        }
        derived_contract = {
            "schema_version": contract.get("schema_version") or "benchmark-run-contract-v1",
            "run_id": run_id,
            "contract_scope": "module",
            "lifecycle_state": "derived_complete",
            "source_command": "python -m suite_tools.materialize_subset",
            "identity": identity,
            "expected_models": retained_models,
            "expected_judges": copy.deepcopy(contract.get("expected_judges") or []),
            "modules": [derived_module],
            "completion_gates": copy.deepcopy(contract.get("completion_gates") or []),
            "derived_from": derived_from,
        }
        for field in _COMMAND_FIELDS:
            derived_contract.pop(field, None)
        write_run_contract(stage, derived_contract)
        _write_json(stage / "DERIVED_PROVENANCE.json", derived_from)

        conformance = evaluate_request_conformance(stage)
        if not conformance["conformant"]:
            raise ValueError(
                "Derived subset retains request-conformance failures: "
                + json.dumps(conformance["issues"], sort_keys=True)
            )

        now = utc_now()
        derived_status = {
            "schema_version": "benchmark-run-ledger-v1",
            "module": module_name,
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "output_dir": ".",
            "started_at": now,
            "updated_at": now,
            "completed_at": now,
            "expected_units": len(expected_units),
            "resolved_units": len(expected_units),
            "request_conformance": conformance,
            "settings": {"derived": True, "provider_calls": 0},
        }
        _write_json(stage / "RUN_STATUS.json", derived_status)
        events = [
            {
                "schema_version": "benchmark-run-monitor-v1",
                "sequence": 1,
                "timestamp": now,
                "event": "derived_artifacts_materialized",
                "module": module_name,
                "stage": "scoring",
                "expected_units": len(expected_units),
                "excluded_model_keys": sorted(excluded_model_keys),
                "provider_calls": 0,
            },
            {
                "schema_version": "benchmark-run-monitor-v1",
                "sequence": 2,
                "timestamp": now,
                "event": "stage_completed",
                "module": module_name,
                "stage": "scoring",
                "status": "completed",
                "validity": "score_ready",
                "attempt_number": 1,
            },
        ]
        (stage / "RUN_EVENTS.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
        )
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        required=True,
        help="Model key to exclude. Repeat to exclude multiple model keys.",
    )
    parser.add_argument("--reason", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = materialize(
        source_run_dir=args.source_run_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        excluded_model_keys=set(args.exclude_model),
        reason=args.reason,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
