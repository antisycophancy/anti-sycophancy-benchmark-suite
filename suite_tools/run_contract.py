"""Run contract, cooperative control, and provenance helpers.

``RUN_CONTRACT.json`` records what a run expects before paid calls begin.
``RUN_EVENTS.jsonl`` records what happened. This module gives dashboards and
future runners a shared way to compare the two without mutating scored data.

``RUN_CONTROL.json`` is intentionally small and cooperative. It can ask a
runner to stop before the next paid call or work unit, but it is not an
out-of-band process killer and does not cancel in-flight provider requests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

from suite_tools.cost_estimate import (
    build_contract_call_plan,
    estimate_call_plan,
    validate_pricing_snapshot,
)
from suite_tools.run_monitor import atomic_write_json, utc_now

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_FILENAME = "RUN_CONTRACT.json"
CONTROL_FILENAME = "RUN_CONTROL.json"
PLAN_FILENAME = "RUN_PLAN.json"
CONTRACT_SCHEMA_VERSION = "benchmark-run-contract-v1"
CONTROL_SCHEMA_VERSION = "benchmark-run-control-v1"
PLAN_SCHEMA_VERSION = "benchmark-run-plan-v1"
PROVENANCE_IDENTITY_SCHEMA_VERSION = "benchmark-provenance-identity-v1"
SAMPLE_REPLICATION_KEYS = {
    "runs",
    "run_count",
    "replicates",
    "replicate_count",
    "run_numbers",
    "run_number",
    "repetition",
    "rep",
}
MODEL_OPERATOR_KEYS = {
    "key",
    "label",
    "source",
    "source_sha256",
    "max_parallel",
}

# item_hash added to the unit sample axis (2026-07-21); cross-era comparisons recompute and fall back per spec.
IDENTITY_PROJECTION_VERSION = "benchmark-identity-projection-v4"
LEGACY_V3_IDENTITY_PROJECTION_VERSION = "benchmark-identity-projection-v3"
LEGACY_IDENTITY_PROJECTION_VERSION = "benchmark-identity-projection-v1"
PREPARED_PRICING_SCHEMA_VERSION = "benchmark-prepared-pricing-v1"

# Whitelist projections: ONLY these keys enter identity hashes. A key absent
# from these sets is excluded until consciously added (spec 015 §3.2). This
# inverts the old blacklist failure mode where operational keys
# (primary_config) and derived hash fields (condition_hash,
# provider_condition_hash) silently leaked into identity.
JUDGE_PANEL_IDENTITY_KEYS = frozenset({
    # real explicit-identity shape (aita/sus/epis contracts)
    "panel", "primary", "analyzer", "seeker",
    "configs",  # nested projection — see JUDGE_CONFIG_IDENTITY_KEYS
    "judge_prompt_hashes", "rubric_version",
    "rubric_source_ids", "rubric_source_registry",
    # legacy fallback shape (contracts without explicit identity)
    "judges", "rubric",
})
# Per-judge config entries mix identity with transport. Raw URL, key-variable,
# and label fields stay private/operator-only; route_hash binds the normalized
# destination without publishing private routing.
LEGACY_V3_JUDGE_CONFIG_IDENTITY_KEYS = frozenset(
    {"model_id", "provider_api", "condition_metadata"}
)
JUDGE_CONFIG_IDENTITY_KEYS = frozenset({
    "model_id",
    "provider_api",
    "route_hash",
    "condition_id",
    "request_options",
    "condition_metadata",
    "profile_id",
    "profile_hash",
    "served_profile_id",
    "served_profile_hash",
    "served_model_version",
    "served_weights_fingerprint",
    "system_fingerprint",
})
MODEL_CONDITION_IDENTITY_KEYS = frozenset({
    "condition_id",
    "condition_metadata",
    "endpoint",
    "model_id",
    "request_options",
    "profile_id",
    "profile_hash",
    "parent_profile_id",
    "served_profile_id",
    "served_profile_hash",
    "provider_api",
    "route_hash",
})
MODEL_PROVENANCE_SUMMARY_FIELDS = (
    "provider_api",
    "route_hash",
    "condition_id",
    "condition_hash",
    "profile_id",
    "profile_hash",
    "served_profile_id",
    "served_profile_hash",
    "system_fingerprint",
    "provider_condition_id",
    "provider_condition_hash",
    "provider_version",
    "condition_metadata",
    "request_options",
)

STOP_BEFORE_NEXT_PAID_CALL = "stop_before_next_paid_call"
PAUSE_BEFORE_NEXT_UNIT = "pause_before_next_unit"
CLEAR_CONTROL = "clear"

SECRET_ENV_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|credential)")
SECRET_FLAGS = {
    "--api-key",
    "--api_key",
    "--key",
    "--token",
    "--secret",
    "--password",
    "--openrouter-api-key",
    "--local-openai-compatible-api-key",
    "--private-adapter-api-key",
}


class RunControlStopRequested(RuntimeError):
    """Raised when cooperative control asks the runner to stop safely."""

    def __init__(self, summary: dict[str, Any]):
        self.summary = summary
        reason = summary.get("reason")
        label = summary.get("label") or "Run control stop requested"
        super().__init__(f"{label}: {reason}" if reason else str(label))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _path_for(path_or_dir: Path | str, filename: str) -> Path:
    path = Path(path_or_dir)
    return path if path.name == filename else path / filename


def load_run_contract(path_or_dir: Path | str) -> dict[str, Any]:
    """Load a contract from ``RUN_CONTRACT.json`` or its containing directory."""
    return _read_json(_path_for(path_or_dir, CONTRACT_FILENAME))


def load_run_control(path_or_dir: Path | str) -> dict[str, Any]:
    """Load cooperative control state from ``RUN_CONTROL.json`` if present."""
    return _read_json(_path_for(path_or_dir, CONTROL_FILENAME))


def load_run_plan(path_or_dir: Path | str) -> dict[str, Any]:
    """Load a run-group plan from ``RUN_PLAN.json`` or its containing directory."""
    return _read_json(_path_for(path_or_dir, PLAN_FILENAME))


def stable_json_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-like data."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_json(value: Any) -> Any:
    """Return JSON-like data with deterministic mapping/list order."""
    if isinstance(value, dict):
        return {
            str(key): _normalized_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalized_json(item) for item in value]
    return value


def _sorted_dicts(items: list[Any]) -> list[Any]:
    normalized = [_normalized_json(item) for item in items if item is not None]
    return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))


def _unit_sample_axis(unit: dict[str, Any], *, module_name: Any = None) -> dict[str, Any]:
    keys = (
        "item_idx",
        "item_hash",
        "side",
        "test_type",
        "scenario",
        "scenario_id",
        "run_number",
        "planned_turns",
        "dataset_mode",
        "pair_id",
        "source_pair_hash",
        "side_prompt_hash",
        "row_index",
    )
    axis = {"module": module_name} if module_name is not None else {}
    for key in keys:
        if unit.get(key) is not None:
            axis[key] = unit.get(key)
    return axis


def _fallback_benchmark_family_id(contract: dict[str, Any]) -> str:
    modules = [
        str(module.get("module"))
        for module in contract.get("modules") or []
        if isinstance(module, dict) and module.get("module")
    ]
    if len(set(modules)) == 1:
        return modules[0]
    return "suite" if modules else "unknown"


def _fallback_benchmark_spec(contract: dict[str, Any]) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for module in contract.get("modules") or []:
        if not isinstance(module, dict):
            continue
        spec = {
            "module": module.get("module"),
            "stage": module.get("stage"),
            "dataset_mode": module.get("dataset_mode"),
            "benchmark_spec": module.get("benchmark_spec"),
            "prompt_versions": module.get("prompt_versions"),
            "prompt_hashes": module.get("prompt_hashes"),
            "score_dimensions": module.get("score_dimensions"),
            "rubric_version": module.get("rubric_version"),
            "rubric_source_ids": module.get("rubric_source_ids"),
            "module_version": module.get("module_version"),
        }
        modules.append(_normalized_json(spec))
    return {
        "schema_version": contract.get("schema_version"),
        "benchmark_family_id": _fallback_benchmark_family_id(contract),
        "modules": _sorted_dicts(modules),
        "completion_gates": _normalized_json(contract.get("completion_gates") or []),
    }


def _fallback_sample_spec(contract: dict[str, Any]) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for module in contract.get("modules") or []:
        if not isinstance(module, dict):
            continue
        module_name = module.get("module")
        unit_axes = [
            _unit_sample_axis(unit, module_name=module_name)
            for unit in module.get("expected_units") or []
            if isinstance(unit, dict)
        ]
        modules.append(
            {
                "module": module_name,
                "dataset_mode": module.get("dataset_mode"),
                "selection": module.get("selection"),
                "scenarios": module.get("scenarios"),
                "units": _sorted_dicts(unit_axes),
            }
        )
    return {"modules": _sorted_dicts(modules)}


def _fallback_judge_panel(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "judges": _sorted_dicts(list(contract.get("expected_judges") or [])),
        "rubric": contract.get("judge_rubric") or contract.get("rubric"),
    }


def _fallback_model_conditions(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return _sorted_dicts(list(contract.get("expected_models") or []))


def _fallback_execution(contract: dict[str, Any]) -> dict[str, Any]:
    return _normalized_json(
        {
            "run_id": contract.get("run_id"),
            "created_at": contract.get("created_at"),
            "source_command": contract.get("source_command"),
            "results_root": contract.get("results_root"),
            "contract_scope": contract.get("contract_scope"),
            "execution": contract.get("execution"),
        }
    )


def build_provenance_identity(
    *,
    benchmark_family_id: str,
    benchmark_spec: dict[str, Any],
    sample_spec: dict[str, Any],
    judge_panel: dict[str, Any],
    model_conditions: list[dict[str, Any]],
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the split identity object used for long-lived comparisons."""
    return _normalized_json(
        {
            "schema_version": PROVENANCE_IDENTITY_SCHEMA_VERSION,
            "benchmark_family_id": benchmark_family_id,
            "benchmark_spec": benchmark_spec,
            "sample_spec": sample_spec,
            "judge_panel": judge_panel,
            "model_conditions": _sorted_dicts(model_conditions),
            "execution": execution or {},
        }
    )


def provenance_identity_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Return explicit or best-effort provenance identity for a contract."""
    raw_identity = contract.get("identity")
    if isinstance(raw_identity, dict):
        return build_provenance_identity(
            benchmark_family_id=str(
                raw_identity.get("benchmark_family_id")
                or _fallback_benchmark_family_id(contract)
            ),
            benchmark_spec=raw_identity.get("benchmark_spec") or _fallback_benchmark_spec(contract),
            sample_spec=raw_identity.get("sample_spec") or _fallback_sample_spec(contract),
            judge_panel=raw_identity.get("judge_panel") or _fallback_judge_panel(contract),
            model_conditions=list(raw_identity.get("model_conditions") or _fallback_model_conditions(contract)),
            execution=raw_identity.get("execution") or _fallback_execution(contract),
        )

    return build_provenance_identity(
        benchmark_family_id=_fallback_benchmark_family_id(contract),
        benchmark_spec=_fallback_benchmark_spec(contract),
        sample_spec=_fallback_sample_spec(contract),
        judge_panel=_fallback_judge_panel(contract),
        model_conditions=_fallback_model_conditions(contract),
        execution=_fallback_execution(contract),
    )


def _judge_config_identity(config: Any) -> Any:
    if not isinstance(config, dict):
        return config
    return {key: value for key, value in config.items() if key in JUDGE_CONFIG_IDENTITY_KEYS}


def _legacy_v3_judge_config_identity(config: Any) -> Any:
    if not isinstance(config, dict):
        return config
    return {
        key: value
        for key, value in config.items()
        if key in LEGACY_V3_JUDGE_CONFIG_IDENTITY_KEYS
    }


def _judge_panel_for_comparison(judge_panel: Any) -> Any:
    """Whitelist projection of judge identity (spec 015 §3.2), nested for
    per-judge configs.

    v2 change: was a blacklist that admitted unknown keys (primary_config
    leaked into published judge_panel_hash; per-config base_url/api_key_env
    still leak today). NOTE: dropping nested transport keys CHANGES
    judge_panel_hash values vs v1 — expected; cross-era comparison goes
    through compare_provenance recompute (Task 2).
    """
    normalized = _normalized_json(judge_panel or {})
    if not isinstance(normalized, dict):
        return normalized
    projected = {
        key: value
        for key, value in normalized.items()
        if key in JUDGE_PANEL_IDENTITY_KEYS
    }
    if isinstance(projected.get("configs"), list):
        projected["configs"] = [_judge_config_identity(c) for c in projected["configs"]]
    return projected


def _legacy_v3_judge_panel_for_comparison(judge_panel: Any) -> Any:
    """Reproduce the projection-v3 judge panel for stored-contract audits."""
    normalized = _normalized_json(judge_panel or {})
    if not isinstance(normalized, dict):
        return normalized
    projected = {
        key: value
        for key, value in normalized.items()
        if key in JUDGE_PANEL_IDENTITY_KEYS
    }
    if isinstance(projected.get("configs"), list):
        projected["configs"] = [
            _legacy_v3_judge_config_identity(config)
            for config in projected["configs"]
        ]
    return projected


def _sample_spec_for_condition(sample_spec: Any) -> Any:
    """Return sample identity without execution-size/repetition fields.

    This keeps an n=3 pilot and later n=17 expansion under one benchmark
    condition when the scenario/item universe is unchanged, while the exact
    sample hash below still records the concrete runset.
    """
    normalized = _normalized_json(sample_spec or {})

    def strip_runset_fields(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_runset_fields(item)
                for key, item in value.items()
                if str(key) not in SAMPLE_REPLICATION_KEYS
            }
        if isinstance(value, list):
            items = [strip_runset_fields(item) for item in value]
            deduped: list[Any] = []
            seen: set[str] = set()
            for item in items:
                marker = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(item)
            return deduped
        return value

    return strip_runset_fields(normalized)


def _model_condition_for_comparison(model_condition: Any) -> Any:
    """Whitelist projection of tested-system identity (spec 015 §3.2)."""
    normalized = _normalized_json(model_condition or {})
    if not isinstance(normalized, dict):
        return normalized
    return {
        key: value
        for key, value in normalized.items()
        if str(key) in MODEL_CONDITION_IDENTITY_KEYS
    }


def _legacy_v1_judge_panel_for_comparison(judge_panel: Any) -> Any:
    """Reproduce the pre-2026-07-18 judge projection for stored-hash audits."""
    normalized = _normalized_json(judge_panel or {})
    if not isinstance(normalized, dict):
        return normalized
    return {
        key: value
        for key, value in normalized.items()
        if key not in {"judge_set", "judge_selector", "selector", "primary_config"}
    }


def _legacy_v1_model_condition_for_comparison(model_condition: Any) -> Any:
    """Reproduce the pre-2026-07-18 model projection for stored-hash audits."""
    normalized = _normalized_json(model_condition or {})
    if not isinstance(normalized, dict):
        return normalized
    return {
        key: value
        for key, value in normalized.items()
        if str(key) not in MODEL_OPERATOR_KEYS
    }


class JudgeProvenanceError(ValueError):
    """Raised before spend when the resolved scorer differs from its contract."""

    def __init__(self, drift_fields: list[str]):
        self.drift_fields = tuple(sorted(drift_fields))
        super().__init__(
            "judge provenance mismatch in: " + ", ".join(self.drift_fields)
        )


class PreparedConfigProvenanceError(ValueError):
    """Raised before spend when a prepared model config is no longer frozen."""

    def __init__(self, issues: str | list[str]):
        normalized = [issues] if isinstance(issues, str) else list(issues)
        self.issues = tuple(sorted(str(issue) for issue in normalized))
        super().__init__("prepared config provenance invalid: " + "; ".join(self.issues))


class PreparedPricingProvenanceError(ValueError):
    """Raised before spend when frozen pricing inputs no longer authenticate."""

    def __init__(self, issues: str | list[str]):
        normalized = [issues] if isinstance(issues, str) else list(issues)
        self.issues = tuple(sorted(str(issue) for issue in normalized))
        super().__init__("prepared pricing provenance invalid: " + "; ".join(self.issues))


def _judge_panel_for_projection_version(
    judge_panel: Any,
    projection_version: str,
) -> Any:
    if projection_version == IDENTITY_PROJECTION_VERSION:
        return _judge_panel_for_comparison(judge_panel)
    if projection_version == LEGACY_V3_IDENTITY_PROJECTION_VERSION:
        return _legacy_v3_judge_panel_for_comparison(judge_panel)
    if projection_version == LEGACY_IDENTITY_PROJECTION_VERSION:
        return _legacy_v1_judge_panel_for_comparison(judge_panel)
    raise ValueError(f"Unsupported identity projection version: {projection_version}")


def validate_judge_provenance_before_spend(
    frozen_judge_panel: dict[str, Any],
    resolved_judge_panel: dict[str, Any],
    *,
    projection_version: str,
) -> None:
    """Fail if runtime scoring identity differs under the frozen projection."""
    frozen = _judge_panel_for_projection_version(
        frozen_judge_panel,
        projection_version,
    )
    resolved = _judge_panel_for_projection_version(
        resolved_judge_panel,
        projection_version,
    )
    if frozen == resolved:
        return
    if not isinstance(frozen, dict) or not isinstance(resolved, dict):
        raise JudgeProvenanceError(["judge_panel"])
    drift_fields = [
        key
        for key in sorted(set(frozen) | set(resolved))
        if frozen.get(key) != resolved.get(key)
    ]
    raise JudgeProvenanceError(drift_fields or ["judge_panel"])


def validate_run_judge_provenance_before_spend(
    path_or_dir: Path | str,
    resolved_judge_panel: dict[str, Any],
) -> bool:
    """Validate a run's frozen judge panel; return False for legacy omissions."""
    contract = load_run_contract(path_or_dir)
    identity = contract.get("identity")
    stored_provenance = contract.get("provenance")
    projection_version = (
        stored_provenance.get("projection_version")
        if isinstance(stored_provenance, dict)
        else None
    ) or LEGACY_IDENTITY_PROJECTION_VERSION
    prepared_contract = (
        contract.get("lifecycle_state") == "prepared"
        or contract.get("prepared") is True
    )
    if isinstance(stored_provenance, dict) and stored_provenance:
        try:
            expected_provenance = provenance_hashes_for_version(
                contract,
                str(projection_version),
            )
        except ValueError as exc:
            raise JudgeProvenanceError(["stored_provenance"]) from exc
        if expected_provenance != stored_provenance:
            raise JudgeProvenanceError(["stored_provenance"])
    elif projection_version == IDENTITY_PROJECTION_VERSION or prepared_contract:
        raise JudgeProvenanceError(["stored_provenance"])

    frozen_panel = identity.get("judge_panel") if isinstance(identity, dict) else None
    has_prompt_hashes = (
        isinstance(frozen_panel, dict)
        and isinstance(frozen_panel.get("judge_prompt_hashes"), dict)
        and bool(frozen_panel.get("judge_prompt_hashes"))
    )
    if not has_prompt_hashes:
        if projection_version == IDENTITY_PROJECTION_VERSION:
            raise JudgeProvenanceError(["judge_panel"])
        return False
    validate_judge_provenance_before_spend(
        frozen_panel,
        resolved_judge_panel,
        projection_version=str(projection_version),
    )
    return True


def validate_run_prepared_config_before_spend(
    path_or_dir: Path | str,
    runtime_config_path: Path | str | None = None,
) -> dict[str, Any] | bool:
    """Authenticate a prepared contract and its exact rendered config bytes.

    Runtime-owned runs have no prepared contract and return ``False``. Legacy
    v1/v3 prepared contracts without a config binding also return ``False`` so
    their frozen hashes are not silently upgraded. Current v4 prepared
    contracts fail closed on any missing, escaped, substituted, or changed
    rendered model config before provider credentials or calls are used.
    """
    contract_path = _path_for(path_or_dir, CONTRACT_FILENAME)
    if not contract_path.is_file():
        return False
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparedConfigProvenanceError("RUN_CONTRACT.json is unreadable") from exc
    if not isinstance(contract, dict):
        raise PreparedConfigProvenanceError("RUN_CONTRACT.json is not an object")

    lifecycle_state = contract.get("lifecycle_state") or contract.get("state")
    if lifecycle_state != "prepared" and contract.get("prepared") is not True:
        return False

    stored_provenance = contract.get("provenance")
    if not isinstance(stored_provenance, dict) or not stored_provenance:
        raise PreparedConfigProvenanceError("stored provenance is missing")
    projection_version = str(
        stored_provenance.get("projection_version")
        or LEGACY_IDENTITY_PROJECTION_VERSION
    )
    try:
        expected_provenance = provenance_hashes_for_version(
            contract,
            projection_version,
        )
    except ValueError as exc:
        raise PreparedConfigProvenanceError("stored provenance projection is unsupported") from exc
    if expected_provenance != stored_provenance:
        raise PreparedConfigProvenanceError("stored provenance digest does not authenticate")

    identity = contract.get("identity")
    execution = identity.get("execution") if isinstance(identity, dict) else None
    binding = execution.get("prepared_config") if isinstance(execution, dict) else None
    if not isinstance(binding, dict):
        if projection_version != IDENTITY_PROJECTION_VERSION:
            return False
        raise PreparedConfigProvenanceError("prepared config binding is missing")

    rendered_artifacts: list[dict[str, Any]] = []
    for module in contract.get("modules") or []:
        if not isinstance(module, dict):
            continue
        rendered_artifacts.extend(
            artifact
            for artifact in module.get("expected_artifacts") or []
            if isinstance(artifact, dict) and artifact.get("kind") == "rendered_models"
        )
    if len(rendered_artifacts) != 1:
        raise PreparedConfigProvenanceError(
            "contract must declare exactly one rendered_models artifact"
        )
    artifact = rendered_artifacts[0]

    raw_path = binding.get("path")
    digest = binding.get("sha256")
    byte_count = binding.get("bytes")
    if (
        not isinstance(raw_path, str)
        or not raw_path.strip()
        or Path(raw_path).is_absolute()
        or raw_path in {".", ".."}
        or ".." in Path(raw_path).parts
    ):
        raise PreparedConfigProvenanceError("prepared config path is unsafe")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PreparedConfigProvenanceError("prepared config digest is malformed")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise PreparedConfigProvenanceError("prepared config byte count is malformed")
    for field in ("path", "sha256", "bytes"):
        if artifact.get(field) != binding.get(field):
            raise PreparedConfigProvenanceError(
                f"rendered_models artifact {field} differs from identity binding"
            )

    run_dir = contract_path.parent
    trust_root = run_dir.parent.resolve()
    bound_path = trust_root / raw_path
    try:
        resolved_bound_path = bound_path.resolve(strict=True)
    except OSError as exc:
        raise PreparedConfigProvenanceError("prepared config file is missing") from exc
    if not resolved_bound_path.is_relative_to(trust_root):
        raise PreparedConfigProvenanceError("prepared config resolves outside the run group")
    if not resolved_bound_path.is_file():
        raise PreparedConfigProvenanceError("prepared config is not a regular file")

    if runtime_config_path is not None:
        try:
            resolved_runtime_path = Path(runtime_config_path).resolve(strict=True)
        except OSError as exc:
            raise PreparedConfigProvenanceError("runtime config path is missing") from exc
        if resolved_runtime_path != resolved_bound_path:
            raise PreparedConfigProvenanceError(
                "runtime path does not match the prepared config path"
            )

    raw = resolved_bound_path.read_bytes()
    if len(raw) != byte_count:
        raise PreparedConfigProvenanceError("prepared config byte count changed")
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != digest:
        raise PreparedConfigProvenanceError("prepared config digest changed")
    return {
        "verified": True,
        "path": raw_path,
        "sha256": actual_digest,
        "bytes": len(raw),
        "projection_version": projection_version,
    }


def _require_finite_numbers(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PreparedPricingProvenanceError(f"{label} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_numbers(item, label=f"{label}[{index}]")


def _expected_cost_warning(
    estimate: dict[str, Any],
    threshold_usd: float,
) -> dict[str, Any]:
    total = estimate.get("total_cost_usd")
    total = total if isinstance(total, dict) else {}
    high = float(total.get("high") or 0)
    complete = estimate.get("state") == "estimated"
    exceeded = complete and high > threshold_usd
    return {
        "state": "exceeded" if exceeded else "within" if complete else "unavailable",
        "warning_threshold_usd": threshold_usd,
        "estimated_high_usd": high if complete else None,
        "notice": "Preparation warning only; this does not stop an executing run.",
    }


def validate_run_pricing_before_spend(
    path_or_dir: Path | str,
) -> dict[str, Any] | bool:
    """Authenticate frozen pricing, call-plan, estimate, and warning inputs.

    Runtime-owned runs return ``False``. Legacy v1/v3 prepared contracts whose
    identities omit the pricing binding remain explicitly unverified rather
    than being upgraded. Current v4 prepared contracts fail closed when the
    binding is absent or any authenticated or derived pricing fact drifts.
    """
    contract_path = _path_for(path_or_dir, CONTRACT_FILENAME)
    if not contract_path.is_file():
        return False
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparedPricingProvenanceError("RUN_CONTRACT.json is unreadable") from exc
    if not isinstance(contract, dict):
        raise PreparedPricingProvenanceError("RUN_CONTRACT.json is not an object")

    lifecycle_state = contract.get("lifecycle_state") or contract.get("state")
    if lifecycle_state != "prepared" and contract.get("prepared") is not True:
        return False

    stored_provenance = contract.get("provenance")
    if not isinstance(stored_provenance, dict) or not stored_provenance:
        raise PreparedPricingProvenanceError("stored provenance is missing")
    projection_version = str(
        stored_provenance.get("projection_version")
        or LEGACY_IDENTITY_PROJECTION_VERSION
    )
    try:
        expected_provenance = provenance_hashes_for_version(
            contract,
            projection_version,
        )
    except ValueError as exc:
        raise PreparedPricingProvenanceError(
            "stored provenance projection is unsupported"
        ) from exc
    if expected_provenance != stored_provenance:
        raise PreparedPricingProvenanceError(
            "stored provenance digest does not authenticate"
        )

    identity = contract.get("identity")
    execution = identity.get("execution") if isinstance(identity, dict) else None
    binding = execution.get("prepared_pricing") if isinstance(execution, dict) else None
    if not isinstance(binding, dict):
        if projection_version != IDENTITY_PROJECTION_VERSION:
            return False
        raise PreparedPricingProvenanceError("prepared pricing binding is missing")
    if binding.get("schema_version") != PREPARED_PRICING_SCHEMA_VERSION:
        raise PreparedPricingProvenanceError(
            "prepared pricing binding schema_version is unsupported"
        )
    _require_finite_numbers(binding, label="prepared pricing binding")

    recomputed_call_plan = build_contract_call_plan(contract)
    bound_call_plan = binding.get("call_plan")
    if not isinstance(bound_call_plan, dict):
        raise PreparedPricingProvenanceError("prepared call plan binding is missing")
    if bound_call_plan != recomputed_call_plan:
        raise PreparedPricingProvenanceError(
            "prepared call plan differs from the recomputed contract plan"
        )
    if contract.get("call_plan") != bound_call_plan:
        raise PreparedPricingProvenanceError(
            "top-level call plan differs from the prepared binding"
        )

    snapshot_binding = binding.get("pricing_snapshot")
    if snapshot_binding is None:
        if any(
            key in contract
            for key in ("pricing_snapshot", "cost_estimate", "cost_warning")
        ):
            raise PreparedPricingProvenanceError(
                "unpriced binding conflicts with top-level pricing artifacts"
            )
        if "warning_threshold_usd" in binding:
            raise PreparedPricingProvenanceError(
                "warning threshold requires a frozen pricing snapshot"
            )
        return {
            "verified": True,
            "state": "not_configured",
            "projection_version": projection_version,
        }
    if not isinstance(snapshot_binding, dict):
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot binding is malformed"
        )
    if set(snapshot_binding) != {"path", "sha256", "bytes"}:
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot binding fields are malformed"
        )

    raw_path = snapshot_binding.get("path")
    digest = snapshot_binding.get("sha256")
    byte_count = snapshot_binding.get("bytes")
    if raw_path != "PRICING_SNAPSHOT.json":
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot path must be PRICING_SNAPSHOT.json"
        )
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot digest is malformed"
        )
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot byte count is malformed"
        )

    trust_root = contract_path.parent.resolve()
    frozen_path = trust_root / raw_path
    try:
        resolved_frozen_path = frozen_path.resolve(strict=True)
    except OSError as exc:
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot file is missing"
        ) from exc
    if not resolved_frozen_path.is_relative_to(trust_root):
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot resolves outside the contract directory"
        )
    if not resolved_frozen_path.is_file():
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot is not a regular file"
        )

    raw_snapshot = resolved_frozen_path.read_bytes()
    if len(raw_snapshot) != byte_count:
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot byte count changed"
        )
    actual_digest = hashlib.sha256(raw_snapshot).hexdigest()
    if actual_digest != digest:
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot digest changed"
        )
    try:
        pricing = json.loads(raw_snapshot)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot is unreadable"
        ) from exc
    if not isinstance(pricing, dict):
        raise PreparedPricingProvenanceError(
            "prepared pricing snapshot is not an object"
        )
    try:
        validate_pricing_snapshot(pricing)
    except ValueError as exc:
        raise PreparedPricingProvenanceError(str(exc)) from exc

    top_level_snapshot = contract.get("pricing_snapshot")
    if not isinstance(top_level_snapshot, dict):
        raise PreparedPricingProvenanceError(
            "top-level pricing snapshot metadata is missing"
        )
    for field in ("path", "sha256", "bytes"):
        if top_level_snapshot.get(field) != snapshot_binding.get(field):
            raise PreparedPricingProvenanceError(
                f"top-level pricing snapshot {field} differs from the prepared binding"
            )
    expected_snapshot_metadata = {
        "pricing_hash": stable_json_hash(pricing),
        "schema_version": pricing.get("schema_version"),
        "units": pricing.get("units"),
        "provider": pricing.get("provider"),
        "generated_at": pricing.get("generated_at"),
        "source": pricing.get("source"),
    }
    for field, expected_value in expected_snapshot_metadata.items():
        if top_level_snapshot.get(field) != expected_value:
            raise PreparedPricingProvenanceError(
                f"top-level pricing snapshot {field} differs from frozen bytes"
            )

    try:
        expected_estimate = estimate_call_plan(recomputed_call_plan, pricing)
    except ValueError as exc:
        raise PreparedPricingProvenanceError(str(exc)) from exc
    supplied_estimate = contract.get("cost_estimate")
    _require_finite_numbers(supplied_estimate, label="cost estimate")
    if supplied_estimate != expected_estimate:
        raise PreparedPricingProvenanceError(
            "top-level cost estimate differs from the frozen pricing inputs"
        )

    threshold = binding.get("warning_threshold_usd")
    warning_state = None
    if threshold is None:
        if "cost_warning" in contract:
            raise PreparedPricingProvenanceError(
                "top-level cost warning has no authenticated threshold"
            )
    else:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or threshold < 0
        ):
            raise PreparedPricingProvenanceError(
                "warning threshold must be non-negative and finite"
            )
        expected_warning = _expected_cost_warning(expected_estimate, float(threshold))
        supplied_warning = contract.get("cost_warning")
        _require_finite_numbers(supplied_warning, label="cost warning")
        if supplied_warning != expected_warning:
            raise PreparedPricingProvenanceError(
                "top-level cost warning differs from the authenticated threshold"
            )
        warning_state = expected_warning["state"]

    return {
        "verified": True,
        "state": expected_estimate.get("state"),
        "path": raw_path,
        "sha256": actual_digest,
        "bytes": len(raw_snapshot),
        "warning_state": warning_state,
        "projection_version": projection_version,
    }


COMPARABILITY_HASHES = ("benchmark_spec_hash", "sample_condition_hash", "judge_panel_hash")


def _provenance_hashes(
    identity_or_contract: dict[str, Any],
    *,
    projection_version: str,
    judge_panel_projector: Any,
    model_condition_projector: Any,
) -> dict[str, Any]:
    """Compute hashes with one explicit immutable identity projection."""
    looks_like_panel = identity_or_contract.get("artifact") == "provenance_hashes" or (
        "comparison_spec_hash" in identity_or_contract
        and not any(k in identity_or_contract for k in ("modules", "benchmark_spec", "expected_models"))
    )
    if looks_like_panel:
        raise ValueError(
            "Input is a provenance hash panel, not raw identity; "
            "hash panels cannot be re-hashed or compared as identity."
        )
    schema = identity_or_contract.get("schema_version")
    if schema == PROVENANCE_IDENTITY_SCHEMA_VERSION:
        identity = _normalized_json(identity_or_contract)
    elif isinstance(schema, str) and schema.startswith("benchmark-provenance-identity-"):
        raise ValueError(
            f"Unknown identity schema version {schema!r}; "
            "upgrade suite_tools before comparing this artifact."
        )
    else:
        identity = provenance_identity_from_contract(identity_or_contract)

    benchmark_spec_hash = stable_json_hash(
        {
            "benchmark_family_id": identity.get("benchmark_family_id"),
            "benchmark_spec": identity.get("benchmark_spec") or {},
        }
    )
    sample_hash = stable_json_hash(identity.get("sample_spec") or {})
    sample_condition_hash = stable_json_hash(_sample_spec_for_condition(identity.get("sample_spec")))
    judge_panel_hash = stable_json_hash(judge_panel_projector(identity.get("judge_panel")))
    comparison_spec_hash = stable_json_hash(
        {
            "benchmark_spec_hash": benchmark_spec_hash,
            "sample_hash": sample_hash,
            "judge_panel_hash": judge_panel_hash,
        }
    )
    benchmark_condition_hash = stable_json_hash(
        {
            "benchmark_spec_hash": benchmark_spec_hash,
            "sample_condition_hash": sample_condition_hash,
            "judge_panel_hash": judge_panel_hash,
        }
    )
    model_conditions = list(identity.get("model_conditions") or [])
    model_conditions_for_hash = _sorted_dicts(
        [model_condition_projector(condition) for condition in model_conditions]
    )
    model_conditions_hash = stable_json_hash(model_conditions_for_hash)
    model_condition_hashes = [
        (lambda condition_hash: {
            "key": condition.get("key") or condition.get("model_id") or str(index),
            "label": condition.get("label"),
            "model_id": condition.get("model_id"),
            "hash": condition_hash,
            "benchmark_model_condition_hash": stable_json_hash(
                {
                    "benchmark_condition_hash": benchmark_condition_hash,
                    "model_condition_hash": condition_hash,
                }
            ),
            **{
                field: condition[field]
                for field in MODEL_PROVENANCE_SUMMARY_FIELDS
                if field in condition
            },
        })(stable_json_hash(model_condition_projector(condition)))
        for index, condition in enumerate(model_conditions)
        if isinstance(condition, dict)
    ]
    return {
        "artifact": "provenance_hashes",
        "projection_version": projection_version,
        "schema_version": PROVENANCE_IDENTITY_SCHEMA_VERSION,
        "benchmark_family_id": identity.get("benchmark_family_id"),
        "benchmark_spec_hash": benchmark_spec_hash,
        "sample_hash": sample_hash,
        "sample_condition_hash": sample_condition_hash,
        "judge_panel_hash": judge_panel_hash,
        "benchmark_condition_hash": benchmark_condition_hash,
        "comparison_spec_hash": comparison_spec_hash,
        "model_conditions_hash": model_conditions_hash,
        "batch_condition_hash": stable_json_hash(
            {
                "benchmark_condition_hash": benchmark_condition_hash,
                "model_conditions_hash": model_conditions_hash,
            }
        ),
        "model_condition_hashes": model_condition_hashes,
        "run_execution_hash": stable_json_hash(identity.get("execution") or {}),
    }


def provenance_hashes(identity_or_contract: dict[str, Any]) -> dict[str, Any]:
    """Compute hashes under the current identity projection."""
    return _provenance_hashes(
        identity_or_contract,
        projection_version=IDENTITY_PROJECTION_VERSION,
        judge_panel_projector=_judge_panel_for_comparison,
        model_condition_projector=_model_condition_for_comparison,
    )


def legacy_v3_provenance_hashes(identity_or_contract: dict[str, Any]) -> dict[str, Any]:
    """Recompute projection-v3 panels for immutable stored-contract audits."""
    return _provenance_hashes(
        identity_or_contract,
        projection_version=LEGACY_V3_IDENTITY_PROJECTION_VERSION,
        judge_panel_projector=_legacy_v3_judge_panel_for_comparison,
        model_condition_projector=_model_condition_for_comparison,
    )


def legacy_v1_provenance_hashes(identity_or_contract: dict[str, Any]) -> dict[str, Any]:
    """Recompute the exact unversioned v1 panel stored by legacy contracts.

    This exists only to verify immutable historical contracts. New comparisons
    and all newly written contracts continue to use :func:`provenance_hashes`
    under :data:`IDENTITY_PROJECTION_VERSION`.
    """
    if identity_or_contract.get("schema_version") == PROVENANCE_IDENTITY_SCHEMA_VERSION:
        identity = _normalized_json(identity_or_contract)
    else:
        identity = provenance_identity_from_contract(identity_or_contract)

    benchmark_spec_hash = stable_json_hash(
        {
            "benchmark_family_id": identity.get("benchmark_family_id"),
            "benchmark_spec": identity.get("benchmark_spec") or {},
        }
    )
    sample_hash = stable_json_hash(identity.get("sample_spec") or {})
    sample_condition_hash = stable_json_hash(
        _sample_spec_for_condition(identity.get("sample_spec"))
    )
    judge_panel_hash = stable_json_hash(
        _legacy_v1_judge_panel_for_comparison(identity.get("judge_panel"))
    )
    comparison_spec_hash = stable_json_hash(
        {
            "benchmark_spec_hash": benchmark_spec_hash,
            "sample_hash": sample_hash,
            "judge_panel_hash": judge_panel_hash,
        }
    )
    benchmark_condition_hash = stable_json_hash(
        {
            "benchmark_spec_hash": benchmark_spec_hash,
            "sample_condition_hash": sample_condition_hash,
            "judge_panel_hash": judge_panel_hash,
        }
    )
    model_conditions = list(identity.get("model_conditions") or [])
    projected_conditions = _sorted_dicts(
        [
            _legacy_v1_model_condition_for_comparison(condition)
            for condition in model_conditions
        ]
    )
    model_conditions_hash = stable_json_hash(projected_conditions)
    model_condition_hashes = [
        (lambda condition_hash: {
            "key": condition.get("key") or condition.get("model_id") or str(index),
            "label": condition.get("label"),
            "model_id": condition.get("model_id"),
            "hash": condition_hash,
            "benchmark_model_condition_hash": stable_json_hash(
                {
                    "benchmark_condition_hash": benchmark_condition_hash,
                    "model_condition_hash": condition_hash,
                }
            ),
            **{
                field: condition[field]
                for field in MODEL_PROVENANCE_SUMMARY_FIELDS
                if field in condition
            },
        })(stable_json_hash(_legacy_v1_model_condition_for_comparison(condition)))
        for index, condition in enumerate(model_conditions)
        if isinstance(condition, dict)
    ]
    return {
        "schema_version": PROVENANCE_IDENTITY_SCHEMA_VERSION,
        "benchmark_family_id": identity.get("benchmark_family_id"),
        "benchmark_spec_hash": benchmark_spec_hash,
        "sample_hash": sample_hash,
        "sample_condition_hash": sample_condition_hash,
        "judge_panel_hash": judge_panel_hash,
        "benchmark_condition_hash": benchmark_condition_hash,
        "comparison_spec_hash": comparison_spec_hash,
        "model_conditions_hash": model_conditions_hash,
        "batch_condition_hash": stable_json_hash(
            {
                "benchmark_condition_hash": benchmark_condition_hash,
                "model_conditions_hash": model_conditions_hash,
            }
        ),
        "model_condition_hashes": model_condition_hashes,
        "run_execution_hash": stable_json_hash(identity.get("execution") or {}),
    }


def provenance_hashes_for_version(
    identity_or_contract: dict[str, Any],
    projection_version: str,
) -> dict[str, Any]:
    """Recompute a panel using exactly the projection declared by its artifact."""
    if projection_version == IDENTITY_PROJECTION_VERSION:
        return provenance_hashes(identity_or_contract)
    if projection_version == LEGACY_V3_IDENTITY_PROJECTION_VERSION:
        return legacy_v3_provenance_hashes(identity_or_contract)
    if projection_version == LEGACY_IDENTITY_PROJECTION_VERSION:
        return legacy_v1_provenance_hashes(identity_or_contract)
    raise ValueError(f"Unsupported identity projection version: {projection_version}")


def compare_provenance(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two raw identities/contracts by RECOMPUTING hashes under the
    current projection (spec 015 §3.2). Stored hash strings are never trusted
    across projection versions — serialization-era drift makes them
    incomparable (the v1 sample_spec enrichment break)."""
    hashes_a = provenance_hashes(a)
    hashes_b = provenance_hashes(b)
    match = {
        name: hashes_a.get(name) == hashes_b.get(name)
        for name in (
            "benchmark_spec_hash", "sample_hash", "sample_condition_hash",
            "judge_panel_hash", "benchmark_condition_hash",
            "comparison_spec_hash", "model_conditions_hash",
        )
    }
    return {
        "projection_version": IDENTITY_PROJECTION_VERSION,
        "match": match,
        "comparable": all(match[name] for name in COMPARABILITY_HASHES),
    }


def file_sha256(path: Path | str) -> str | None:
    """Return a file SHA-256, or ``None`` when the file is unavailable."""
    target = Path(path)
    try:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def redact_source_command(command: Any) -> str:
    """Redact likely secrets from a command before saving it in a contract."""
    if not isinstance(command, str) or not command.strip():
        return ""
    redacted_lines: list[str] = []
    for line in command.splitlines():
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            tokens = line.split()

        redacted: list[str] = []
        redact_next = False
        for token in tokens:
            if redact_next:
                redacted.append("<redacted>")
                redact_next = False
                continue

            if "=" in token:
                name, value = token.split("=", 1)
                lowered_name = name.lower().lstrip("-")
                if SECRET_ENV_RE.search(lowered_name):
                    redacted.append(f"{name}=<redacted>")
                    continue
                redacted.append(f"{name}={value}")
                continue

            lowered = token.lower()
            redacted.append(token)
            if lowered in SECRET_FLAGS or (
                lowered.startswith("--") and SECRET_ENV_RE.search(lowered)
            ):
                redact_next = True

        redacted_lines.append(shlex.join(redacted))

    return "\n".join(redacted_lines)


def _redact_persisted_command_fields(payload: dict[str, Any]) -> None:
    """Redact top-level and module command strings in a persisted payload."""
    command_fields = ("source_command", "execute_command", "score_command")
    for command_key in command_fields:
        if command_key in payload:
            payload[command_key] = redact_source_command(payload.get(command_key))
    modules = payload.get("modules")
    if isinstance(modules, list):
        sanitized_modules: list[Any] = []
        for module in modules:
            if not isinstance(module, dict):
                sanitized_modules.append(module)
                continue
            sanitized = dict(module)
            for command_key in command_fields:
                if command_key in sanitized:
                    sanitized[command_key] = redact_source_command(
                        sanitized.get(command_key)
                    )
            sanitized_modules.append(sanitized)
        payload["modules"] = sanitized_modules


def write_run_contract(output_dir: Path | str, contract: dict[str, Any]) -> Path:
    """Write a normalized run contract atomically and return its path."""
    payload = dict(contract)
    # Commands are persisted for operator context, never as a credential store.
    # Redact before identity construction so secrets cannot survive inside the
    # authenticated execution projection either.
    _redact_persisted_command_fields(payload)
    payload.setdefault("schema_version", CONTRACT_SCHEMA_VERSION)
    payload.setdefault("created_at", utc_now())
    payload.setdefault("contract_scope", "run_group")
    identity = provenance_identity_from_contract(payload)
    payload["identity"] = identity
    supplied_provenance = payload.get("provenance")
    if supplied_provenance is not None:
        if not isinstance(supplied_provenance, dict):
            raise ValueError("stale provenance: supplied panel is not an object")
        supplied_version = str(
            supplied_provenance.get("projection_version")
            or LEGACY_IDENTITY_PROJECTION_VERSION
        )
        try:
            expected_supplied = provenance_hashes_for_version(identity, supplied_version)
        except ValueError as exc:
            raise ValueError(f"stale provenance: {exc}") from exc
        if supplied_provenance != expected_supplied:
            raise ValueError(
                "stale provenance: supplied panel does not match the normalized identity"
            )
    payload["provenance"] = provenance_hashes(identity)
    atomic_write_json(_path_for(output_dir, CONTRACT_FILENAME), payload)
    return _path_for(output_dir, CONTRACT_FILENAME)


def write_runtime_run_contract(output_dir: Path | str, contract: dict[str, Any]) -> Path:
    """Write standalone metadata without mutating a prepared execution contract.

    Current prepared contracts use ``lifecycle_state=prepared``. The older
    top-level ``prepared=true`` marker remains accepted for compatibility.
    Unmarked contracts are runtime-owned and may be replaced.
    """
    contract_path = _path_for(output_dir, CONTRACT_FILENAME)
    existing = load_run_contract(contract_path)
    lifecycle_state = existing.get("lifecycle_state") or existing.get("state")
    if lifecycle_state == "prepared" or existing.get("prepared") is True:
        return contract_path
    return write_run_contract(output_dir, contract)


def write_run_plan(output_dir: Path | str, plan: dict[str, Any]) -> Path:
    """Write a normalized run-group plan atomically and return its path."""
    payload = dict(plan)
    payload.setdefault("schema_version", PLAN_SCHEMA_VERSION)
    payload.setdefault("created_at", utc_now())
    payload["updated_at"] = utc_now()
    _redact_persisted_command_fields(payload)
    atomic_write_json(_path_for(output_dir, PLAN_FILENAME), payload)
    return _path_for(output_dir, PLAN_FILENAME)


def write_run_control(
    output_dir: Path | str,
    *,
    action: str,
    reason: str = "",
    requested_by: str = "operator",
    applies_to: str = "run_group",
) -> Path:
    """Write cooperative local control intent for future runner checks."""
    payload = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "updated_at": utc_now(),
        "state": "cleared" if action == CLEAR_CONTROL else "requested",
        "action": action,
        "reason": reason,
        "requested_by": requested_by,
        "applies_to": applies_to,
    }
    atomic_write_json(_path_for(output_dir, CONTROL_FILENAME), payload)
    return _path_for(output_dir, CONTROL_FILENAME)


def should_stop_before_paid_call(control: dict[str, Any]) -> bool:
    """Return true when a runner should stop before spending another call."""
    return (
        control.get("schema_version") == CONTROL_SCHEMA_VERSION
        and control.get("state") == "requested"
        and control.get("action") == STOP_BEFORE_NEXT_PAID_CALL
    )


def require_no_control_stop(
    output_dir: Path | str,
    *,
    monitor: Any | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Raise before a paid call when ``RUN_CONTROL.json`` requests a stop."""
    control_path = _path_for(output_dir, CONTROL_FILENAME)
    control = load_run_control(control_path)
    if not should_stop_before_paid_call(control):
        return

    summary = summarize_control(control, control_path=control_path)
    if monitor is not None:
        fields = {
            "action": summary.get("action"),
            "state": summary.get("state"),
            "reason": summary.get("reason"),
            "requested_by": summary.get("requested_by"),
            "control_path": summary.get("path"),
        }
        if context:
            fields.update(context)
        monitor.record("control_stop_requested", **fields)
    raise RunControlStopRequested(summary)


def summarize_control(control: dict[str, Any], *, control_path: Path | None = None) -> dict[str, Any]:
    """Return dashboard-safe control state."""
    if not control:
        return {
            "present": False,
            "active": False,
            "label": "No control request",
            "action": None,
            "path": _display_path(control_path) if control_path else None,
        }

    action = control.get("action")
    state = control.get("state")
    active = state == "requested" and action in {STOP_BEFORE_NEXT_PAID_CALL, PAUSE_BEFORE_NEXT_UNIT}
    if action == STOP_BEFORE_NEXT_PAID_CALL:
        label = "Stop before next paid call"
        next_action = "Runner should finish any in-flight call, then halt before spending again."
    elif action == PAUSE_BEFORE_NEXT_UNIT:
        label = "Pause before next unit"
        next_action = "Runner should stop before the next model/item/scenario unit."
    elif action == CLEAR_CONTROL:
        label = "Control cleared"
        next_action = "No cooperative stop is active."
    else:
        label = "Unknown control request"
        next_action = "Inspect RUN_CONTROL.json before continuing."

    return {
        "present": True,
        "active": active,
        "label": label,
        "action": action,
        "state": state,
        "reason": control.get("reason"),
        "requested_by": control.get("requested_by"),
        "updated_at": control.get("updated_at"),
        "next_action": next_action,
        "path": _display_path(control_path) if control_path else None,
    }


def _display_path(path: Path | None, *, base: Path = REPO_ROOT) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _resolve_contract_path(
    value: Any,
    *,
    contract_dir: Path,
    results_root: Path | None = None,
    module_output_dir: Path | None = None,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if module_output_dir is not None:
        candidates.append(module_output_dir / path)
    if results_root is not None:
        candidates.append(results_root / path)
    if path.parts and path.parts[0] == "results":
        candidates.append(REPO_ROOT / path)
    candidates.extend([contract_dir / path, REPO_ROOT / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _module_output_dir(
    module: dict[str, Any],
    *,
    contract_dir: Path,
    results_root: Path | None,
) -> Path:
    resolved = _resolve_contract_path(
        module.get("output_dir"),
        contract_dir=contract_dir,
        results_root=results_root,
    )
    return resolved or contract_dir


def _artifact_required(artifact: dict[str, Any]) -> bool:
    return str(artifact.get("required_for") or "required").lower() not in {"optional", "nice_to_have"}


def _expected_artifact_summary(
    artifacts: list[Any],
    *,
    contract_dir: Path,
    results_root: Path | None,
    module_output_dir: Path,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for raw in artifacts:
        if not isinstance(raw, dict):
            continue
        resolved = _resolve_contract_path(
            raw.get("path"),
            contract_dir=contract_dir,
            results_root=results_root,
            module_output_dir=module_output_dir,
        )
        summary.append(
            {
                "kind": raw.get("kind") or "artifact",
                "path": _display_path(resolved) if resolved else raw.get("path"),
                "required_for": raw.get("required_for") or "required",
                "required": _artifact_required(raw),
                "present": bool(resolved and resolved.exists()),
            }
        )
    return summary


def _unit_summary(
    unit: dict[str, Any],
    *,
    contract_dir: Path,
    results_root: Path | None,
    module_output_dir: Path,
) -> dict[str, Any]:
    expected_paths = [
        ("transcript", unit.get("expected_transcript_path")),
        ("score", unit.get("expected_score_path")),
        ("summary", unit.get("expected_summary_path")),
        ("trace", unit.get("expected_trace_path")),
    ]
    paths: list[dict[str, Any]] = []
    for kind, value in expected_paths:
        resolved = _resolve_contract_path(
            value,
            contract_dir=contract_dir,
            results_root=results_root,
            module_output_dir=module_output_dir,
        )
        if resolved is None and not value:
            continue
        paths.append(
            {
                "kind": kind,
                "path": _display_path(resolved) if resolved else value,
                "present": bool(resolved and resolved.exists()),
            }
        )

    required_paths = [item for item in paths if item.get("path")]
    return {
        "unit_id": unit.get("unit_id"),
        "module": unit.get("module"),
        "model_key": unit.get("model_key"),
        "model_id": unit.get("model_id"),
        "item_idx": unit.get("item_idx"),
        "side": unit.get("side"),
        "test_type": unit.get("test_type"),
        "planned_turns": unit.get("planned_turns"),
        "paths": paths,
        "present_path_count": sum(1 for item in paths if item.get("present")),
        "expected_path_count": len(required_paths),
        "complete": bool(required_paths) and all(item.get("present") for item in required_paths),
    }


def _model_mismatches(contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected: dict[str, str] = {}
    for model in contract.get("expected_models") or []:
        if isinstance(model, dict) and model.get("key") and model.get("model_id"):
            expected[str(model["key"])] = str(model["model_id"])

    mismatches: list[dict[str, Any]] = []
    for module in contract.get("modules") or []:
        if not isinstance(module, dict):
            continue
        for unit in module.get("expected_units") or []:
            if not isinstance(unit, dict):
                continue
            key = unit.get("model_key")
            model_id = unit.get("model_id")
            if key in expected and model_id and str(model_id) != expected[str(key)]:
                mismatches.append(
                    {
                        "unit_id": unit.get("unit_id"),
                        "model_key": key,
                        "expected_model_id": expected[str(key)],
                        "unit_model_id": model_id,
                    }
                )
    return mismatches


def summarize_contract(
    contract: "dict[str, Any] | Path | str",
    *,
    contract_path: "Path | str | None" = None,
    results_root: "Path | str | None" = None,
) -> "dict[str, Any]":
    """Summarize expected-vs-observed state for a loaded run contract.

    ``contract`` may be a pre-loaded dict or a ``Path``/``str`` pointing to
    ``RUN_CONTRACT.json`` (or its containing directory), in which case the
    file is loaded automatically and ``contract_path`` is inferred.
    """
    if isinstance(contract, (Path, str)):
        inferred_path = _path_for(contract, CONTRACT_FILENAME)
        if contract_path is None:
            contract_path = inferred_path
        contract = _read_json(inferred_path)
    path = Path(contract_path) if contract_path is not None else None
    contract_dir = path.parent if path is not None else Path(contract.get("results_root") or ".")
    root = Path(results_root) if results_root is not None else None

    module_summaries: list[dict[str, Any]] = []
    total_units = 0
    complete_units = 0
    total_artifacts = 0
    present_artifacts = 0
    missing_required_artifacts: list[dict[str, Any]] = []

    for raw_module in contract.get("modules") or []:
        if not isinstance(raw_module, dict):
            continue
        module_output = _module_output_dir(raw_module, contract_dir=contract_dir, results_root=root)
        artifacts = _expected_artifact_summary(
            list(raw_module.get("expected_artifacts") or []),
            contract_dir=contract_dir,
            results_root=root,
            module_output_dir=module_output,
        )
        units = [
            _unit_summary(unit, contract_dir=contract_dir, results_root=root, module_output_dir=module_output)
            for unit in list(raw_module.get("expected_units") or [])
            if isinstance(unit, dict)
        ]

        missing_artifacts = [
            artifact for artifact in artifacts if artifact.get("required") and not artifact.get("present")
        ]
        missing_required_artifacts.extend(
            {
                **artifact,
                "module": raw_module.get("module"),
                "stage": raw_module.get("stage"),
            }
            for artifact in missing_artifacts
        )

        unit_count = len(units)
        unit_complete_count = sum(1 for unit in units if unit.get("complete"))
        artifact_count = len(artifacts)
        artifact_present_count = sum(1 for artifact in artifacts if artifact.get("present"))

        total_units += unit_count
        complete_units += unit_complete_count
        total_artifacts += artifact_count
        present_artifacts += artifact_present_count

        module_summaries.append(
            {
                "module": raw_module.get("module"),
                "stage": raw_module.get("stage"),
                "output_dir": _display_path(module_output),
                "expected_units": unit_count,
                "complete_units": unit_complete_count,
                "missing_units": unit_count - unit_complete_count,
                "expected_artifacts": artifact_count,
                "present_artifacts": artifact_present_count,
                "missing_artifacts": missing_artifacts[:12],
                "units": units[:50],
            }
        )

    mismatches = _model_mismatches(contract)
    lifecycle_state = contract.get("lifecycle_state") or contract.get("state")
    prepared = lifecycle_state == "prepared" or contract.get("prepared") is True
    attention = (not prepared) and bool(missing_required_artifacts or mismatches)
    progress_percent = 100
    denominator = total_units + total_artifacts
    if denominator:
        progress_percent = round(((complete_units + present_artifacts) / denominator) * 100)
    identity = provenance_identity_from_contract(contract)
    provenance = provenance_hashes(identity)
    contract_fingerprint = stable_json_hash(
        {
            "schema_version": contract.get("schema_version"),
            "expected_models": contract.get("expected_models") or [],
            "expected_judges": contract.get("expected_judges") or [],
            "modules": contract.get("modules") or [],
            "completion_gates": contract.get("completion_gates") or [],
        }
    )

    return {
        "present": True,
        "schema_version": contract.get("schema_version"),
        "path": _display_path(path) if path else None,
        "run_id": contract.get("run_id") or (contract_dir.name if contract_dir else None),
        "contract_scope": contract.get("contract_scope"),
        "lifecycle_state": lifecycle_state,
        "prepared": prepared,
        "created_at": contract.get("created_at"),
        "source_command": contract.get("source_command"),
        "execute_command": contract.get("execute_command"),
        "results_root": contract.get("results_root"),
        "model_selector": contract.get("model_selector"),
        "judge_set": contract.get("judge_set"),
        "expected_models": contract.get("expected_models") or [],
        "expected_judges": contract.get("expected_judges") or [],
        "completion_gates": contract.get("completion_gates") or [],
        "modules": module_summaries,
        "expected_units": total_units,
        "complete_units": complete_units,
        "missing_units": total_units - complete_units,
        "expected_artifacts": total_artifacts,
        "present_artifacts": present_artifacts,
        "missing_required_artifacts": missing_required_artifacts[:20],
        "model_mismatches": mismatches[:20],
        "attention": attention,
        "progress_percent": progress_percent,
        "identity": identity,
        "provenance": provenance,
        "contract_fingerprint": contract_fingerprint,
        "fingerprint": contract_fingerprint,
    }
