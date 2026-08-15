"""Materialize a saved SUS rescore, merge, or identity repair as a derived run.

This is a no-provider-call migration for paper-facing artifacts. It can restore
missing condition identity only when one frozen source-contract condition is an
unambiguous match; conflicting identity always fails. It creates a new run
directory with a canonical ``RUN_CONTRACT.json``, individual unit artifacts,
normalized final score files, and an explicit hash receipt back to every source
file. Source contracts and score artifacts are never changed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from suite_tools.artifact_identity import reconcile_condition_identity
from suite_tools.artifact_privacy import assert_public_artifact_safe
from suite_tools.run_contract import (
    build_provenance_identity,
    file_sha256,
    provenance_hashes,
    stable_json_hash,
    write_run_contract,
)
from suite_tools.run_monitor import SCHEMA_VERSION as LEDGER_SCHEMA_VERSION
from suite_tools.run_monitor import utc_now

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "benchmark-derived-sus-v1"
_PUBLIC_CONDITION_METADATA_KEYS = frozenset({
    "effort",
    "effort_policy",
    "provider_fallback",
    "provider_route",
    "source_official_model_id",
    "source_official_model_key",
})


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(
            row.get("condition_id")
            or row.get("model_key")
            or row.get("key")
            or row.get("label")
            or ""
        ),
        str(row.get("scenario") or ""),
        int(row.get("run_number") or 0),
    )


def _model_condition_map(contract: dict[str, Any]) -> dict[str, str]:
    identity = contract.get("identity") or {}
    conditions = identity.get("model_conditions") or contract.get("expected_models") or []
    result: dict[str, str] = {}
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        key = condition.get("key")
        condition_id = condition.get("condition_id") or key
        if key and condition_id:
            result[str(key)] = str(condition_id)
        label = condition.get("label")
        if label and condition_id:
            result.setdefault(str(label), str(condition_id))
    return result


def _model_conditions(contract: dict[str, Any]) -> list[dict[str, Any]]:
    identity = contract.get("identity") or {}
    conditions = identity.get("model_conditions") or contract.get("expected_models") or []
    return [condition for condition in conditions if isinstance(condition, dict)]


def _resolve_row_condition(
    row: dict[str, Any],
    conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    condition_id = row.get("condition_id")
    if condition_id:
        candidates = [
            condition
            for condition in conditions
            if condition.get("condition_id") == condition_id
            or (condition.get("condition_id") is None and condition.get("key") == condition_id)
        ]
    else:
        observed = {
            str(value)
            for value in (
                row.get("model"), row.get("model_id"), row.get("model_key"),
                row.get("key"), row.get("label"),
            )
            if value is not None
        }
        candidates = [
            condition
            for condition in conditions
            if observed.intersection(
                str(value)
                for value in (
                    condition.get("model_id"), condition.get("id"),
                    condition.get("key"), condition.get("label"),
                )
                if value is not None
            )
        ]
    if len(candidates) != 1:
        raise ValueError(
            "SUS source row resolves to "
            f"{len(candidates)} contract model conditions: {_row_key(row)}"
        )
    return candidates[0]


def _normalize_row_identity(
    row: dict[str, Any],
    conditions: list[dict[str, Any]],
    *,
    source_contract_sha256: str,
) -> dict[str, Any]:
    normalized = copy.deepcopy(row)
    condition = _resolve_row_condition(normalized, conditions)
    restored = reconcile_condition_identity(
        normalized,
        condition,
        context=f"derived SUS row {_row_key(normalized)}",
        restore_missing=True,
    )
    if restored:
        normalized["identity_normalization"] = {
            "method": "restored_from_frozen_source_contract",
            "restored_fields": list(restored),
            "source_contract_sha256": source_contract_sha256,
        }
    return normalized


def _sus_module(contract: dict[str, Any]) -> dict[str, Any]:
    for module in contract.get("modules") or []:
        if isinstance(module, dict) and module.get("module") == "sus":
            return module
    raise ValueError("Contract does not contain a SUS module")


def _expected_key(unit: dict[str, Any], condition_by_key: dict[str, str]) -> tuple[str, str, int]:
    model_key = str(unit.get("model_key") or "")
    condition_id = condition_by_key.get(model_key)
    if not condition_id:
        raise ValueError(f"No condition_id found for model_key {model_key!r}")
    return (
        condition_id,
        str(unit.get("scenario") or ""),
        int(unit.get("run_number") or 0),
    )


def _safe_rescore_metadata(metadata: Any, source_hashes: dict[str, str | None]) -> Any:
    if not isinstance(metadata, dict):
        return metadata
    projected = {
        key: copy.deepcopy(metadata[key])
        for key in ("timestamp", "analyzer_model", "judge_panel")
        if key in metadata
    }
    projected["source_artifact_sha256"] = {
        key: value for key, value in source_hashes.items() if value
    }
    return projected


def _safe_condition_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, dict):
        return metadata
    return {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key in _PUBLIC_CONDITION_METADATA_KEYS
    }


def _sanitize_row(row: dict[str, Any], source_hashes: dict[str, str | None]) -> dict[str, Any]:
    projected = copy.deepcopy(row)
    projected.pop("union_recovery_root", None)
    projected.pop("union_source", None)
    projected.pop("score_repair_source", None)
    if "rescore_metadata" in projected:
        projected["rescore_metadata"] = _safe_rescore_metadata(
            projected.get("rescore_metadata"), source_hashes
        )
    if "condition_metadata" in projected:
        projected["condition_metadata"] = _safe_condition_metadata(
            projected.get("condition_metadata")
        )
    return projected


def _without_transcripts(value: Any) -> Any:
    transcript_keys = {
        "conversation", "conversations", "dialogue", "history", "messages",
        "model_response", "prompt", "raw_prompt", "raw_response", "response",
        "responses", "transcript", "transcripts", "turns",
    }
    if isinstance(value, dict):
        return {
            key: _without_transcripts(item)
            for key, item in value.items()
            if key not in transcript_keys
        }
    if isinstance(value, list):
        return [_without_transcripts(item) for item in value]
    return value


def _judge_receipt(rows: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    analyzers: set[str] = set()
    panels: set[tuple[str, ...]] = set()
    for row in rows:
        metadata = row.get("rescore_metadata")
        if not isinstance(metadata, dict):
            continue
        analyzer = metadata.get("analyzer_model")
        panel = metadata.get("judge_panel")
        if analyzer:
            analyzers.add(str(analyzer))
        if isinstance(panel, list):
            panels.add(tuple(str(item) for item in panel))
    if len(analyzers) > 1 or len(panels) > 1:
        raise ValueError("Selected score rows do not share one analyzer and judge panel")
    return (next(iter(analyzers), None), list(next(iter(panels), ())))


def _expected_judge_ids(contract: dict[str, Any]) -> tuple[str | None, list[str]]:
    identity_panel = (contract.get("identity") or {}).get("judge_panel") or {}
    analyzer = identity_panel.get("analyzer")
    panel = identity_panel.get("panel") or []
    return (
        str(analyzer) if analyzer else None,
        [str(item) for item in panel],
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def materialize(
    *,
    model_source_contract: Path,
    benchmark_template_contract: Path,
    score_summary: Path,
    score_sidecar: Path,
    output_dir: Path,
    run_id: str,
    source_conversations: Path | None = None,
) -> Path:
    """Create and return a derived SUS run directory without provider calls."""
    inputs = [model_source_contract, benchmark_template_contract, score_summary, score_sidecar]
    if source_conversations is not None:
        inputs.append(source_conversations)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing derived run: {output_dir}")

    source_contract = _load_json(model_source_contract)
    template_contract = _load_json(benchmark_template_contract)
    source_summary = _load_json(score_summary)
    scored_rows_raw = _load_json(score_sidecar)
    source_rows_raw = _load_json(source_conversations) if source_conversations else []
    if not isinstance(source_summary, dict):
        raise ValueError("score_summary must contain a JSON object")
    if not isinstance(scored_rows_raw, list) or not all(isinstance(row, dict) for row in scored_rows_raw):
        raise ValueError("score_sidecar must contain a JSON list of objects")
    if not isinstance(source_rows_raw, list) or not all(isinstance(row, dict) for row in source_rows_raw):
        raise ValueError("source_conversations must contain a JSON list of objects")

    source_hashes = {
        "model_source_contract": file_sha256(model_source_contract),
        "benchmark_template_contract": file_sha256(benchmark_template_contract),
        "score_summary": file_sha256(score_summary),
        "score_sidecar": file_sha256(score_sidecar),
        "source_conversations": file_sha256(source_conversations) if source_conversations else None,
    }

    source_identity = source_contract.get("identity") or {}
    template_identity = template_contract.get("identity") or {}
    source_hash_panel = provenance_hashes(source_contract)
    template_hash_panel = provenance_hashes(template_contract)
    for name in ("benchmark_spec_hash", "sample_condition_hash"):
        if source_hash_panel.get(name) != template_hash_panel.get(name):
            raise ValueError(f"Source and benchmark template disagree on {name}")

    conditions = _model_conditions(source_contract)
    if not conditions:
        raise ValueError("Source contract has no model conditions")
    source_contract_sha256 = str(source_hashes["model_source_contract"])
    scored_rows = [
        _sanitize_row(
            _normalize_row_identity(
                row,
                conditions,
                source_contract_sha256=source_contract_sha256,
            ),
            source_hashes,
        )
        for row in scored_rows_raw
    ]
    source_rows = [
        _sanitize_row(
            _normalize_row_identity(
                row,
                conditions,
                source_contract_sha256=source_contract_sha256,
            ),
            source_hashes,
        )
        for row in source_rows_raw
    ]
    observed_analyzer, observed_panel = _judge_receipt(scored_rows)
    expected_analyzer, expected_panel = _expected_judge_ids(template_contract)
    if observed_analyzer and observed_analyzer != expected_analyzer:
        raise ValueError("Score artifact analyzer does not match benchmark template")
    if observed_panel and observed_panel != expected_panel:
        raise ValueError("Score artifact judge panel does not match benchmark template")

    scored_by_key = {_row_key(row): row for row in scored_rows}
    source_by_key = {_row_key(row): row for row in source_rows}
    condition_by_key = _model_condition_map(source_contract)
    expected_units = [
        copy.deepcopy(unit)
        for unit in (_sus_module(source_contract).get("expected_units") or [])
        if isinstance(unit, dict)
    ]
    if not expected_units:
        raise ValueError("Source contract has no expected SUS units")

    from sus_bench.scoring_contract import is_score_excluded_result
    from sus_bench.stats import aggregate_runs

    selected_rows: list[dict[str, Any]] = []
    derived_units: list[dict[str, Any]] = []
    unit_artifacts: list[tuple[str, dict[str, Any]]] = []
    for index, unit in enumerate(expected_units, start=1):
        key = _expected_key(unit, condition_by_key)
        row = scored_by_key.get(key)
        if row is None:
            row = source_by_key.get(key)
            if row is None or not is_score_excluded_result(row):
                raise ValueError(f"No selected score or terminal source row for {key}")
        row = copy.deepcopy(row)
        row["unit_id"] = str(unit.get("unit_id") or "")
        assert_public_artifact_safe(_without_transcripts(row))
        selected_rows.append(row)

        artifact_name = f"transcripts/unit-{index:04d}.json"
        derived_unit = copy.deepcopy(unit)
        derived_unit.pop("expected_summary_path", None)
        derived_unit.pop("expected_score_path", None)
        derived_unit["expected_transcript_path"] = artifact_name
        derived_units.append(derived_unit)
        unit_artifacts.append((artifact_name, row))

    restored_rows = [
        row for row in selected_rows if isinstance(row.get("identity_normalization"), dict)
    ]
    restored_field_counts: dict[str, int] = {}
    for row in restored_rows:
        for field in row["identity_normalization"].get("restored_fields") or []:
            restored_field_counts[str(field)] = restored_field_counts.get(str(field), 0) + 1

    expected_keys = {_expected_key(unit, condition_by_key) for unit in expected_units}
    unexpected = set(scored_by_key) - expected_keys
    if unexpected:
        raise ValueError(f"Score sidecar contains {len(unexpected)} unexpected unit(s)")

    identity = build_provenance_identity(
        benchmark_family_id=str(template_identity.get("benchmark_family_id") or "sus"),
        benchmark_spec=template_identity.get("benchmark_spec") or {},
        sample_spec=template_identity.get("sample_spec") or source_identity.get("sample_spec") or {},
        judge_panel=template_identity.get("judge_panel") or {},
        model_conditions=list(source_identity.get("model_conditions") or []),
        execution={
            "run_id": run_id,
            "contract_scope": "module",
            "runner": "suite_tools.materialize_sus_derived",
            "derived": True,
        },
    )
    provenance = provenance_hashes(identity)
    contract = {
        "run_id": run_id,
        "lifecycle_state": "derived_complete",
        "contract_scope": "module",
        "source_command": "python -m suite_tools.materialize_sus_derived",
        "results_root": ".",
        "identity": identity,
        "provenance": provenance,
        "expected_models": copy.deepcopy(source_contract.get("expected_models") or []),
        "expected_judges": copy.deepcopy(template_contract.get("expected_judges") or []),
        "derived_from": {
            "schema_version": SCHEMA_VERSION,
            "model_source_contract": _display_path(model_source_contract),
            "benchmark_template_contract": _display_path(benchmark_template_contract),
            "score_summary": _display_path(score_summary),
            "score_sidecar": _display_path(score_sidecar),
            "source_conversations": _display_path(source_conversations) if source_conversations else None,
            "source_hashes": {key: value for key, value in source_hashes.items() if value},
            "transform": (
                "missing condition identity is restored only from one unambiguous "
                "frozen source-contract condition; score rows replace matching "
                "source rows; terminal excluded rows fill unscored keys"
            ),
            "identity_normalization": {
                "restored_row_count": len(restored_rows),
                "restored_field_counts": restored_field_counts,
                "source_contract_sha256": source_contract_sha256,
            },
        },
        "modules": [{
            "module": "sus",
            "stage": "scoring",
            "output_dir": ".",
            "scenarios": copy.deepcopy(_sus_module(source_contract).get("scenarios") or []),
            "runs": _sus_module(source_contract).get("runs"),
            "escalation_mode": _sus_module(source_contract).get("escalation_mode"),
            "expected_units": derived_units,
            "expected_artifacts": [
                {"kind": "run_contract", "path": "RUN_CONTRACT.json", "required_for": "diagnostic"},
                {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "promotion"},
                {"kind": "run_events", "path": "RUN_EVENTS.jsonl", "required_for": "promotion"},
                {"kind": "final_results", "path": "FINAL_RESULTS.json", "required_for": "promotion"},
                {"kind": "final_conversations", "path": "FINAL_RESULTS-conversations.json", "required_for": "promotion"},
                {"kind": "derived_provenance", "path": "DERIVED_PROVENANCE.json", "required_for": "promotion"},
            ],
        }],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        write_run_contract(stage, contract)
        for artifact_name, row in unit_artifacts:
            _write_json(stage / artifact_name, row)
        _write_json(stage / "FINAL_RESULTS-conversations.json", selected_rows)
        _write_json(stage / "FINAL_RESULTS.json", {
            "version": source_summary.get("version"),
            "run_id": run_id,
            "timestamp": utc_now(),
            "aggregated": aggregate_runs(selected_rows),
            "cost": source_summary.get("cost"),
            "derived": True,
        })
        derived_provenance = contract["derived_from"] | {
            "created_at": utc_now(),
            "output_contract_sha256": file_sha256(stage / "RUN_CONTRACT.json"),
            "output_score_sidecar_sha256": file_sha256(stage / "FINAL_RESULTS-conversations.json"),
        }
        _write_json(stage / "DERIVED_PROVENANCE.json", derived_provenance)
        now = utc_now()
        _write_json(stage / "RUN_STATUS.json", {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "run_id": run_id,
            "module": "sus",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "attempt_number": 1,
            "started_at": now,
            "updated_at": now,
            "completed_at": now,
            "expected_units": len(derived_units),
            "resolved_units": len(derived_units),
            "settings": {"derived": True, "provider_calls": 0},
        })
        event = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event": "derived_artifacts_materialized",
            "event_id": stable_json_hash({"run_id": run_id, "source_hashes": source_hashes})[:32],
            "timestamp": now,
            "run_id": run_id,
            "module": "sus",
            "stage": "scoring",
            "attempt_number": 1,
            "provider_calls": 0,
        }
        (stage / "RUN_EVENTS.jsonl").write_text(json.dumps(event, sort_keys=True) + "\n")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source-contract", type=Path, required=True)
    parser.add_argument("--benchmark-template-contract", type=Path, required=True)
    parser.add_argument("--score-summary", type=Path, required=True)
    parser.add_argument("--score-sidecar", type=Path, required=True)
    parser.add_argument("--source-conversations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = materialize(
        model_source_contract=args.model_source_contract,
        benchmark_template_contract=args.benchmark_template_contract,
        score_summary=args.score_summary,
        score_sidecar=args.score_sidecar,
        source_conversations=args.source_conversations,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps({"schema_version": SCHEMA_VERSION, "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
