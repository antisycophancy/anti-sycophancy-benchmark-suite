"""Sanitized effective-request receipts and contract conformance checks.

Receipts deliberately retain only request controls that define a benchmark
condition. Prompts, messages, API keys, headers, and private routing details
never enter the run ledger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from suite_tools.provider_client import (
    is_anthropic_messages_url,
    is_gemini_generate_content_url,
)
from suite_tools.run_contract import load_run_contract, stable_json_hash

RECEIPT_EVENT = "effective_request"
RECEIPT_SCHEMA_VERSION = "benchmark-effective-request-v1"
_CONTROL_FIELDS = (
    "max_output_tokens",
    "reasoning_effort",
    "reasoning_enabled",
    "reasoning_exclude",
    "thinking_type",
    "thinking_display",
    "include_thoughts",
    "verbosity",
)
_CONTEXT_FIELDS = (
    "condition_id",
    "model_key",
    "unit_id",
    "item_idx",
    "side",
    "test_type",
    "scenario",
    "phase",
    "turn",
    "dimension",
    "call_attempt",
    "provider",
    "provider_api",
)


class RequestConformanceError(RuntimeError):
    """Executed request controls do not conform to the prepared contract."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        issues = result.get("issues") or []
        detail = "; ".join(_format_issue(issue) for issue in issues[:5])
        if len(issues) > 5:
            detail += f"; plus {len(issues) - 5} more"
        super().__init__(f"Effective request does not conform to RUN_CONTRACT.json: {detail}")


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _as_token_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _reasoning_effort(mapping: dict[str, Any]) -> str | None:
    direct = mapping.get("reasoning_effort")
    if direct is not None:
        return str(direct)

    reasoning = _as_mapping(mapping.get("reasoning"))
    if reasoning.get("effort") is not None:
        return str(reasoning["effort"])

    output_config = _as_mapping(mapping.get("output_config"))
    if output_config.get("effort") is not None:
        return str(output_config["effort"])

    generation_config = _as_mapping(mapping.get("generationConfig"))
    thinking_config = _as_mapping(generation_config.get("thinkingConfig"))
    if thinking_config.get("thinkingLevel") is not None:
        return str(thinking_config["thinkingLevel"])

    thinking_config = _as_mapping(mapping.get("thinkingConfig"))
    if thinking_config.get("thinkingLevel") is not None:
        return str(thinking_config["thinkingLevel"])
    return None


def _output_cap(mapping: dict[str, Any]) -> int | None:
    cap = _first(
        mapping,
        "max_output_tokens",
        "max_completion_tokens",
        "max_tokens",
    )
    if cap is not None:
        return _as_token_count(cap)

    generation_config = _as_mapping(mapping.get("generationConfig"))
    return _as_token_count(
        _first(
            generation_config,
            "maxOutputTokens",
            "max_output_tokens",
            "max_tokens",
        )
    )


def _additional_controls(mapping: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    reasoning = _as_mapping(mapping.get("reasoning"))
    if reasoning.get("enabled") is not None:
        controls["reasoning_enabled"] = bool(reasoning["enabled"])
    if reasoning.get("exclude") is not None:
        controls["reasoning_exclude"] = bool(reasoning["exclude"])

    thinking = _as_mapping(mapping.get("thinking"))
    if thinking.get("type") is not None:
        controls["thinking_type"] = str(thinking["type"])
    if thinking.get("display") is not None:
        controls["thinking_display"] = str(thinking["display"])

    generation_config = _as_mapping(mapping.get("generationConfig"))
    thinking_config = _as_mapping(generation_config.get("thinkingConfig"))
    if not thinking_config:
        thinking_config = _as_mapping(mapping.get("thinkingConfig"))
    if thinking_config.get("includeThoughts") is not None:
        controls["include_thoughts"] = bool(thinking_config["includeThoughts"])

    if mapping.get("verbosity") is not None:
        controls["verbosity"] = str(mapping["verbosity"])
    return controls


def effective_request_controls(
    payload: dict[str, Any],
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Project the provider-effective cap and effort from a request payload."""
    top = _as_mapping(payload)
    extra = _as_mapping(top.get("extra_body"))
    extra_overrides = (
        is_anthropic_messages_url(base_url)
        or is_gemini_generate_content_url(base_url)
    )

    sources = (extra, top) if extra_overrides else (top, extra)
    cap = next((_output_cap(source) for source in sources if _output_cap(source) is not None), None)
    effort = next(
        (_reasoning_effort(source) for source in sources if _reasoning_effort(source) is not None),
        None,
    )

    controls: dict[str, Any] = {}
    for source in sources:
        for key, value in _additional_controls(source).items():
            controls.setdefault(key, value)
    if cap is not None:
        controls["max_output_tokens"] = cap
    if effort is not None:
        controls["reasoning_effort"] = effort
    return controls


def record_effective_request(
    monitor: Any,
    payload: dict[str, Any],
    *,
    base_url: str | None = None,
    role: str,
    **context: Any,
) -> dict[str, Any]:
    """Append one prompt-free receipt immediately before a provider call."""
    controls = effective_request_controls(payload, base_url=base_url)
    fields: dict[str, Any] = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "role": str(role),
        "model": str(payload.get("model") or context.get("model") or "unknown"),
        "controls_hash": stable_json_hash(controls),
    }
    for key, value in controls.items():
        fields[f"effective_{key}"] = value
    for key in _CONTEXT_FIELDS:
        value = context.get(key)
        if value is not None:
            fields[key] = value
    monitor.record(RECEIPT_EVENT, **fields)
    return fields


def _expected_controls(entry: dict[str, Any]) -> dict[str, Any]:
    options = entry.get("request_options")
    if not isinstance(options, dict):
        return {}
    return effective_request_controls(
        options,
        base_url=str(entry.get("base_url") or entry.get("endpoint") or ""),
    )


def _expected_provider(entry: dict[str, Any]) -> str | None:
    metadata = _as_mapping(entry.get("condition_metadata"))
    route = str(metadata.get("provider_route") or "").lower()
    endpoint = str(entry.get("base_url") or entry.get("endpoint") or "").lower()
    value = f"{route} {endpoint}"
    if "openrouter" in value:
        return "openrouter"
    if "anthropic" in value:
        return "anthropic"
    if "openai" in value:
        return "openai"
    if "google" in value or "generativelanguage" in value or "gemini" in value:
        return "google"
    return None


def _requirements(contract: dict[str, Any], roles: set[str]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    if "model_under_test" in roles:
        for model in contract.get("expected_models") or []:
            if not isinstance(model, dict):
                continue
            controls = _expected_controls(model)
            if controls:
                requirements.append(
                    {
                        "role": "model_under_test",
                        "model": str(model.get("model_id") or ""),
                        "model_key": model.get("key"),
                        "condition_id": model.get("condition_id") or model.get("key"),
                        "endpoint": model.get("endpoint") or model.get("base_url"),
                        "condition_metadata": model.get("condition_metadata") or {},
                        "provider": _expected_provider(model),
                        "controls": controls,
                    }
                )
    if "judge" in roles:
        for judge in contract.get("expected_judges") or []:
            if not isinstance(judge, dict) or judge.get("role") == "seeker":
                continue
            config = judge.get("config") if isinstance(judge.get("config"), dict) else judge
            controls = _expected_controls(config)
            if controls:
                requirements.append(
                    {
                        "role": "judge",
                        "model": str(config.get("model_id") or judge.get("model_id") or ""),
                        "model_key": config.get("key"),
                        "condition_id": config.get("condition_id") or config.get("key"),
                        "endpoint": config.get("base_url") or judge.get("endpoint"),
                        "condition_metadata": config.get("condition_metadata") or {},
                        "provider": _expected_provider(config),
                        "controls": controls,
                    }
                )
    return requirements


def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "RUN_EVENTS.jsonl"
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _matching_receipts(
    requirement: dict[str, Any],
    receipts: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    role = requirement["role"]
    model = requirement["model"]
    condition_id = requirement.get("condition_id")
    same_model_requirements = [
        candidate
        for candidate in requirements
        if candidate["role"] == role and candidate["model"] == model
    ]
    matches = []
    for receipt in receipts:
        if receipt.get("role") != role or str(receipt.get("model") or "") != model:
            continue
        receipt_condition = receipt.get("condition_id")
        if condition_id and receipt_condition:
            if receipt_condition == condition_id:
                matches.append(receipt)
        elif len(same_model_requirements) == 1:
            matches.append(receipt)
    return matches


def _completed_model_calls(
    requirement: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    after_index: int | None = None,
) -> list[dict[str, Any]]:
    model_key = requirement.get("model_key")
    return [
        event
        for index, event in enumerate(events)
        if (after_index is None or index > after_index)
        if event.get("event") == "paid_call_completed"
        and event.get("role") == "model"
        and str(event.get("model_id") or event.get("model") or "") == requirement["model"]
        and (
            not model_key
            or event.get("model_key") == model_key
            or event.get("model") == model_key
        )
    ]


def _receipt_policy_start(events: list[dict[str, Any]], role: str) -> int | None:
    stages = {"generation", "run"} if role == "model_under_test" else {"scoring", "run"}
    for index, event in enumerate(events):
        if (
            event.get("event") == "stage_started"
            and event.get("stage") in stages
            and event.get("request_receipt_schema_version") == RECEIPT_SCHEMA_VERSION
        ):
            return index
    return None


def _contract_module(contract: dict[str, Any], run_dir: Path) -> str:
    for module in contract.get("modules") or []:
        if isinstance(module, dict) and module.get("module"):
            return str(module["module"]).lower()
    return run_dir.name.lower()


def _legacy_request_is_known_risky(
    requirement: dict[str, Any],
    *,
    module: str,
) -> bool:
    if module not in {"aita", "epis", "epistemic"}:
        return False
    endpoint = str(requirement.get("endpoint") or "").lower()
    metadata = _as_mapping(requirement.get("condition_metadata"))
    route = str(metadata.get("provider_route") or "").lower()
    return (
        "api.openai.com" in endpoint
        or endpoint in {"openai_native", "openai_responses"}
        or route.startswith("openai_")
    )


def _call_has_receipt(call: dict[str, Any], receipts: list[dict[str, Any]]) -> bool:
    for receipt in receipts:
        if str(receipt.get("model") or "") != str(call.get("model_id") or ""):
            continue
        if receipt.get("model_key") != call.get("model"):
            continue
        axes = ("item_idx", "side", "test_type", "scenario", "phase", "turn")
        if all(call.get(axis) == receipt.get(axis) for axis in axes if call.get(axis) is not None):
            return True
    return False


def _call_unit_label(call: dict[str, Any]) -> str:
    parts = [str(call.get("model") or call.get("model_id") or "unknown")]
    if call.get("test_type") is not None:
        parts.append(str(call["test_type"]))
    if call.get("scenario") is not None:
        parts.append(str(call["scenario"]))
    if call.get("item_idx") is not None:
        parts.append(f"item{call['item_idx']}")
    if call.get("side") is not None:
        parts.append(str(call["side"]))
    if call.get("turn") is not None:
        parts.append(f"turn{call['turn']}")
    if call.get("phase") is not None:
        parts.append(str(call["phase"]))
    return ":".join(parts)


def evaluate_request_conformance(
    run_dir: Path | str,
    *,
    roles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare recorded effective controls with explicit contract controls."""
    path = Path(run_dir)
    selected_roles = set(roles or {"model_under_test", "judge"})
    contract = load_run_contract(path)
    module = _contract_module(contract, path)
    requirements = _requirements(contract, selected_roles)
    events = _load_events(path)
    receipts = [
        event
        for event in events
        if event.get("event") == RECEIPT_EVENT and event.get("role") in selected_roles
    ]
    issues: list[dict[str, Any]] = []
    legacy_unverified_requirements = 0

    for requirement in requirements:
        matches = _matching_receipts(requirement, receipts, requirements)
        issue_base = {
            "role": requirement["role"],
            "condition_id": requirement.get("condition_id"),
            "model": requirement["model"],
        }
        policy_start = _receipt_policy_start(events, requirement["role"])
        legacy_risky = _legacy_request_is_known_risky(
            requirement,
            module=module,
        )
        completed_calls = (
            _completed_model_calls(requirement, events)
            if requirement["role"] == "model_under_test"
            else []
        )
        covered_calls = (
            completed_calls
            if legacy_risky
            else _completed_model_calls(
                requirement,
                events,
                after_index=policy_start,
            )
            if policy_start is not None and requirement["role"] == "model_under_test"
            else []
        )
        if not matches:
            if (
                legacy_risky
                or (requirement["role"] == "judge" and policy_start is not None)
            ):
                issues.append({"kind": "missing_request_receipt", **issue_base})
            elif covered_calls:
                issues.append(
                    {
                        "kind": "missing_call_receipts",
                        **issue_base,
                        "missing_count": len(covered_calls),
                        "sample_units": [
                            _call_unit_label(call)
                            for call in covered_calls[:5]
                        ],
                    }
                )
            else:
                legacy_unverified_requirements += 1
            continue
        for receipt in matches:
            expected_provider = requirement.get("provider")
            actual_provider = receipt.get("provider")
            if (
                expected_provider is not None
                and actual_provider is not None
                and actual_provider != expected_provider
            ):
                issue = {
                    "kind": "request_mismatch",
                    **issue_base,
                    "field": "provider",
                    "expected": expected_provider,
                    "actual": actual_provider,
                }
                if issue not in issues:
                    issues.append(issue)
            for field in _CONTROL_FIELDS:
                if field not in requirement["controls"]:
                    continue
                expected = requirement["controls"][field]
                actual = receipt.get(f"effective_{field}")
                if actual != expected:
                    issue = {
                        "kind": "request_mismatch",
                        **issue_base,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                    if issue not in issues:
                        issues.append(issue)

        if requirement["role"] == "model_under_test":
            missing_calls = [
                call
                for call in covered_calls
                if not _call_has_receipt(call, matches)
            ]
            if missing_calls:
                issues.append(
                    {
                        "kind": "missing_call_receipts",
                        **issue_base,
                        "missing_count": len(missing_calls),
                        "sample_units": [
                            _call_unit_label(call)
                            for call in missing_calls[:5]
                        ],
                    }
                )

    return {
        "schema_version": "benchmark-request-conformance-v1",
        "contract_present": bool(contract),
        "module": module,
        "roles": sorted(selected_roles),
        "requirement_count": len(requirements),
        "receipt_count": len(receipts),
        "legacy_unverified_requirement_count": legacy_unverified_requirements,
        "conformant": not issues,
        "issues": issues,
    }


def require_request_conformance(
    run_dir: Path | str,
    *,
    roles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Raise before further paid work when receipts do not match the contract."""
    result = evaluate_request_conformance(run_dir, roles=roles)
    if not result["conformant"]:
        raise RequestConformanceError(result)
    return result


def _format_issue(issue: dict[str, Any]) -> str:
    identity = issue.get("condition_id") or issue.get("model") or "unknown condition"
    if issue.get("kind") == "missing_request_receipt":
        return f"{identity} has no effective-request receipt"
    if issue.get("kind") == "missing_call_receipts":
        return f"{identity} has {issue.get('missing_count')} completed calls without receipts"
    return (
        f"{identity} {issue.get('field')} expected {issue.get('expected')!r}, "
        f"observed {issue.get('actual')!r}"
    )
