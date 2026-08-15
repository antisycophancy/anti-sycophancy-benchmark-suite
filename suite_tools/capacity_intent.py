"""Pre-run capacity intent helpers for prepared benchmark contracts.

The intent file is an advisory, side-effect-free signal. It lets an operator
or private endpoint adapter size infrastructure from a ``RUN_CONTRACT.json``
without changing benchmark prompts, questions, model payloads, judges, scoring,
or artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from suite_tools.run_contract import (
    CONTRACT_FILENAME,
    MODEL_PROVENANCE_SUMMARY_FIELDS,
    load_run_contract,
)
from suite_tools.run_monitor import atomic_write_json, utc_now

try:
    import yaml
except ImportError:  # pragma: no cover - optional operator convenience only
    yaml = None


CAPACITY_INTENT_FILENAME = "CAPACITY_INTENT.json"
CAPACITY_INTENT_SCHEMA_VERSION = "benchmark-capacity-intent-v1"

DEFAULT_CAPACITY_PROFILE: dict[str, Any] = {
    "name": "default",
    "model_id_prefixes": [],
    "endpoint_names": [],
    "endpoint_contains": [],
    "default_turns_per_unit": 1,
    "provider_calls_per_turn": 1,
    "default_max_active_calls": 20,
    "calls_per_capacity_unit": 10,
    "min_capacity_units": 0,
    "max_capacity_units": None,
}

MODEL_CONDITION_INTENT_FIELDS = (
    "key",
    "model_id",
    "endpoint",
    *MODEL_PROVENANCE_SUMMARY_FIELDS,
)


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _dedup_sorted(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_profile_selector(selector: str | None) -> tuple[Path | None, str | None]:
    if not selector:
        return None, None
    candidate = Path(selector)
    if candidate.exists():
        return candidate, None
    if ":" in selector:
        raw_path, name = selector.rsplit(":", 1)
        return Path(raw_path), name
    return candidate, None


def load_capacity_profile(selector: str | None = None, explicit_name: str | None = None) -> dict[str, Any]:
    """Load a generic capacity profile from JSON/YAML or return defaults."""
    path, selector_name = _parse_profile_selector(selector)
    profile_name = explicit_name or selector_name
    profile = dict(DEFAULT_CAPACITY_PROFILE)
    if path is None:
        if profile_name:
            profile["name"] = profile_name
        return profile
    if not path.exists():
        raise FileNotFoundError(f"Capacity profile not found: {path}")

    text = path.read_text()
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML capacity profiles")
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Capacity profile must contain a mapping: {path}")

    if "profiles" in loaded:
        profiles = loaded.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError("`profiles` must be a mapping")
        if not profile_name:
            if len(profiles) != 1:
                names = ", ".join(sorted(str(name) for name in profiles))
                raise ValueError(f"Profile name required. Available profiles: {names}")
            profile_name = str(next(iter(profiles)))
        selected = profiles.get(profile_name)
        if not isinstance(selected, dict):
            raise ValueError(f"Profile not found or invalid: {profile_name}")
    else:
        selected = loaded
        profile_name = profile_name or str(loaded.get("name") or path.stem)

    profile = _deep_merge(profile, selected)
    profile["name"] = str(profile_name or profile.get("name") or "default")
    profile["_profile_path"] = str(path)
    return normalize_capacity_profile(profile)


def normalize_capacity_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize aliases into the public profile vocabulary."""
    normalized = dict(DEFAULT_CAPACITY_PROFILE)
    normalized.update(profile)
    if "calls_per_instance" in normalized and "calls_per_capacity_unit" not in profile:
        normalized["calls_per_capacity_unit"] = normalized.get("calls_per_instance")
    if "min_instances" in normalized and "min_capacity_units" not in profile:
        normalized["min_capacity_units"] = normalized.get("min_instances")
    if "max_instances" in normalized and "max_capacity_units" not in profile:
        normalized["max_capacity_units"] = normalized.get("max_instances")
    if "target_instances" in normalized and "target_capacity_units" not in profile:
        normalized["target_capacity_units"] = normalized.get("target_instances")
    normalized["model_id_prefixes"] = _dedup_sorted(_list_of_strings(normalized.get("model_id_prefixes")))
    normalized["endpoint_names"] = _dedup_sorted(_list_of_strings(normalized.get("endpoint_names")))
    normalized["endpoint_contains"] = _dedup_sorted(_list_of_strings(normalized.get("endpoint_contains")))
    normalized["name"] = str(normalized.get("name") or "default")
    return normalized


def resolve_contract_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / CONTRACT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"RUN_CONTRACT.json not found: {path}")
    return path.resolve()


def expected_models_by_key(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for model in contract.get("expected_models") or []:
        if not isinstance(model, dict):
            continue
        key = model.get("key") or model.get("label") or model.get("model_id")
        if key:
            models[str(key)] = model
    return models


def model_condition_summary(model: dict[str, Any]) -> dict[str, Any]:
    return {
        field: model[field]
        for field in MODEL_CONDITION_INTENT_FIELDS
        if field in model and model[field] is not None
    }


def model_matches(model: dict[str, Any], profile: dict[str, Any]) -> bool:
    normalized = normalize_capacity_profile(profile)
    model_id = str(model.get("model_id") or "")
    endpoint = str(model.get("endpoint") or "")
    prefixes = normalized.get("model_id_prefixes") or []
    endpoint_names = normalized.get("endpoint_names") or []
    endpoint_needles = normalized.get("endpoint_contains") or []
    if not prefixes and not endpoint_names and not endpoint_needles:
        return True
    return (
        any(model_id.startswith(prefix) for prefix in prefixes)
        or endpoint in endpoint_names
        or any(needle and needle in endpoint for needle in endpoint_needles)
    )


def estimate_contract_capacity(contract: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Estimate matching units, turns, and provider calls for a contract."""
    normalized = normalize_capacity_profile(profile)
    models = expected_models_by_key(contract)
    matching_model_keys = {
        key
        for key, model in models.items()
        if model_matches(model, normalized)
    }
    matching_model_ids = {
        str(model.get("model_id"))
        for model in models.values()
        if model_matches(model, normalized) and model.get("model_id")
    }
    matching_model_conditions = [
        model_condition_summary(model)
        for _key, model in sorted(models.items(), key=lambda pair: pair[0])
        if model_matches(model, normalized)
    ]
    default_turns = _positive_int(normalized.get("default_turns_per_unit"), default=1)
    provider_calls_per_turn = _positive_float(normalized.get("provider_calls_per_turn"), default=1.0)

    total_units = 0
    matching_units = 0
    planned_turns = 0
    modules_summary: list[dict[str, Any]] = []

    for module in contract.get("modules") or []:
        if not isinstance(module, dict):
            continue
        units = [unit for unit in module.get("expected_units") or [] if isinstance(unit, dict)]
        module_total = len(units)
        module_matching = 0
        module_turns = 0
        for unit in units:
            total_units += 1
            unit_model_key = str(unit.get("model_key") or "")
            unit_model_id = str(unit.get("model_id") or "")
            if (
                unit_model_key in matching_model_keys
                or unit_model_id in matching_model_ids
                or model_matches(unit, normalized)
            ):
                matching_units += 1
                module_matching += 1
                unit_turns = unit.get("planned_turns")
                if not isinstance(unit_turns, (int, float)) or int(unit_turns) < 1:
                    unit_turns = default_turns
                planned_turns += int(unit_turns)
                module_turns += int(unit_turns)
        modules_summary.append(
            {
                "module": module.get("module"),
                "total_units": module_total,
                "matching_units": module_matching,
                "planned_turns": module_turns,
            }
        )

    return {
        "total_units": total_units,
        "matching_units": matching_units,
        "planned_turns": planned_turns,
        "estimated_provider_calls": math.ceil(planned_turns * provider_calls_per_turn),
        "matching_model_keys": sorted(matching_model_keys),
        "matching_model_ids": sorted(matching_model_ids),
        "matching_model_conditions": matching_model_conditions,
        "modules": modules_summary,
    }


def choose_capacity(
    estimate: dict[str, Any],
    profile: dict[str, Any],
    *,
    max_active_override: int | None = None,
    capacity_units_override: int | None = None,
) -> dict[str, Any]:
    """Choose an advisory concurrency and capacity-unit target."""
    normalized = normalize_capacity_profile(profile)
    matching_units = _nonnegative_int(estimate.get("matching_units"), default=0)
    configured_max_active = (
        max_active_override
        if max_active_override is not None
        else normalized.get("max_active_calls")
    )
    if matching_units == 0:
        max_active_calls = 0
    elif configured_max_active is None or str(configured_max_active).lower() == "auto":
        default_max_active = _positive_int(normalized.get("default_max_active_calls"), default=20)
        max_active_calls = max(1, min(matching_units, default_max_active))
    else:
        max_active_calls = _positive_int(configured_max_active, default=1)

    calls_per_capacity_unit = _positive_int(normalized.get("calls_per_capacity_unit"), default=10)
    min_capacity_units = _nonnegative_int(normalized.get("min_capacity_units"), default=0)
    raw_max_capacity_units = normalized.get("max_capacity_units")
    max_capacity_units = (
        _positive_int(raw_max_capacity_units, default=max(min_capacity_units, 1))
        if raw_max_capacity_units is not None
        else None
    )

    if matching_units == 0:
        target_capacity_units = 0
    elif capacity_units_override is not None:
        target_capacity_units = max(1, int(capacity_units_override))
    elif normalized.get("target_capacity_units") not in {None, "auto"}:
        target_capacity_units = _positive_int(normalized.get("target_capacity_units"), default=1)
    else:
        target_capacity_units = math.ceil(max_active_calls / calls_per_capacity_unit)
        target_capacity_units = max(min_capacity_units, target_capacity_units)
        if max_capacity_units is not None:
            target_capacity_units = min(max_capacity_units, target_capacity_units)

    return {
        "max_active_calls": max_active_calls,
        "calls_per_capacity_unit": calls_per_capacity_unit,
        "target_capacity_units": target_capacity_units,
        "min_capacity_units": min_capacity_units,
        "max_capacity_units": max_capacity_units,
    }


def build_capacity_intent(
    contract: dict[str, Any],
    *,
    contract_path: str | Path | None = None,
    profile: dict[str, Any] | None = None,
    max_active_override: int | None = None,
    capacity_units_override: int | None = None,
) -> dict[str, Any]:
    """Build a generic pre-run capacity intent from a prepared contract."""
    normalized_profile = normalize_capacity_profile(profile or {})
    estimate = estimate_contract_capacity(contract, normalized_profile)
    capacity = choose_capacity(
        estimate,
        normalized_profile,
        max_active_override=max_active_override,
        capacity_units_override=capacity_units_override,
    )
    path_value = str(contract_path) if contract_path is not None else None
    return {
        "schema_version": CAPACITY_INTENT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "contract_path": path_value,
        "run_id": contract.get("run_id"),
        "profile": {
            "name": normalized_profile.get("name"),
            "path": normalized_profile.get("_profile_path"),
        },
        "match": {
            "model_id_prefixes": normalized_profile.get("model_id_prefixes") or [],
            "endpoint_names": normalized_profile.get("endpoint_names") or [],
            "endpoint_contains": normalized_profile.get("endpoint_contains") or [],
            "default_is_all_models": not any(
                normalized_profile.get(key)
                for key in ("model_id_prefixes", "endpoint_names", "endpoint_contains")
            ),
        },
        "estimate": estimate,
        "capacity": capacity,
        "side_effects": "none",
        "contract_invariance": {
            "modifies_run_contract": False,
            "modifies_questions": False,
            "modifies_prompts": False,
            "modifies_model_payloads": False,
            "modifies_judges": False,
            "modifies_scoring": False,
            "modifies_artifacts": False,
        },
        "operator_note": (
            "Advisory pre-run capacity signal only. Private infrastructure "
            "wrappers may consume it, but benchmark execution and scoring use "
            "the unchanged RUN_CONTRACT.json and normal runner commands."
        ),
    }


def write_capacity_intent(output_path: Path, intent: dict[str, Any]) -> Path:
    atomic_write_json(output_path, intent)
    return output_path


def _extend(values: list[str], extra: list[str] | None) -> list[str]:
    return _dedup_sorted([*values, *(extra or [])])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a side-effect-free pre-run capacity intent from a prepared "
            "benchmark RUN_CONTRACT.json."
        )
    )
    parser.add_argument("--contract", required=True, help="Path to RUN_CONTRACT.json or its directory.")
    parser.add_argument(
        "--profile",
        help="Optional JSON/YAML profile path, or path:name when the file has a profiles mapping.",
    )
    parser.add_argument("--profile-name", help="Profile name when --profile contains multiple profiles.")
    parser.add_argument("--match-model-prefix", action="append", default=[], help="Match model_id prefix.")
    parser.add_argument("--match-endpoint", action="append", default=[], help="Match exact endpoint name.")
    parser.add_argument(
        "--match-endpoint-contains",
        action="append",
        default=[],
        help="Match endpoint names containing this substring.",
    )
    parser.add_argument("--default-turns-per-unit", type=int)
    parser.add_argument("--provider-calls-per-turn", type=float)
    parser.add_argument("--default-max-active-calls", type=int)
    parser.add_argument("--max-active-calls", type=int, help="Override advisory scheduler concurrency.")
    parser.add_argument("--calls-per-capacity-unit", type=int)
    parser.add_argument("--capacity-units", type=int, help="Override advisory target capacity units.")
    parser.add_argument("--min-capacity-units", type=int)
    parser.add_argument("--max-capacity-units", type=int)
    parser.add_argument(
        "--output",
        help=f"Output path. Defaults to {CAPACITY_INTENT_FILENAME} beside the contract.",
    )
    parser.add_argument("--json", action="store_true", help="Print intent JSON.")
    return parser


def _apply_cli_overrides(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = normalize_capacity_profile(profile)
    updated["model_id_prefixes"] = _extend(updated.get("model_id_prefixes") or [], args.match_model_prefix)
    updated["endpoint_names"] = _extend(updated.get("endpoint_names") or [], args.match_endpoint)
    updated["endpoint_contains"] = _extend(
        updated.get("endpoint_contains") or [],
        args.match_endpoint_contains,
    )
    for field in (
        "default_turns_per_unit",
        "provider_calls_per_turn",
        "default_max_active_calls",
        "calls_per_capacity_unit",
        "min_capacity_units",
        "max_capacity_units",
    ):
        value = getattr(args, field)
        if value is not None:
            updated[field] = value
    return normalize_capacity_profile(updated)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract_path = resolve_contract_path(args.contract)
    contract = load_run_contract(contract_path)
    if not contract:
        raise ValueError(f"Could not read RUN_CONTRACT.json: {contract_path}")
    profile = _apply_cli_overrides(load_capacity_profile(args.profile, args.profile_name), args)
    intent = build_capacity_intent(
        contract,
        contract_path=contract_path,
        profile=profile,
        max_active_override=args.max_active_calls,
        capacity_units_override=args.capacity_units,
    )
    output_path = Path(args.output).resolve() if args.output else contract_path.parent / CAPACITY_INTENT_FILENAME
    write_capacity_intent(output_path, intent)

    if args.json:
        print(json.dumps(intent, indent=2, sort_keys=True))
    else:
        estimate = intent["estimate"]
        capacity = intent["capacity"]
        print(f"Capacity intent written: {output_path}")
        print(f"Matched units: {estimate['matching_units']} / {estimate['total_units']}")
        print(
            "Capacity: "
            f"max_active_calls={capacity['max_active_calls']}, "
            f"target_capacity_units={capacity['target_capacity_units']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
