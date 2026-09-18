"""Conservative, public-safe condition summaries for read-only aggregation."""

from __future__ import annotations

from suite_tools.model_config import MODEL_CONDITION_METADATA_FIELDS
from suite_tools.request_receipts import effective_request_controls, stable_json_hash


def describe_condition(record: dict) -> dict:
    options = record.get("request_options") or {}
    metadata = record.get("condition_metadata") or {}
    if not isinstance(options, dict) or not isinstance(metadata, dict):
        raise ValueError("Incompatible model condition: malformed request options or metadata")
    controls = effective_request_controls(options)
    efforts = [
        str(value) for value in (
            controls.get("reasoning_effort"), record.get("reasoning_effort"), metadata.get("effort")
        ) if value is not None
    ]
    if len({value.lower() for value in efforts}) > 1:
        raise ValueError("Incompatible model condition: conflicting configured effort values")
    if efforts:
        controls["reasoning_effort"] = efforts[0]

    # Compare all recorded settings without exposing private request extensions.
    settings = {
        field: record[field]
        for field in (*MODEL_CONDITION_METADATA_FIELDS, "temperature", "reasoning_effort")
        if field in record and record[field] is not None
    }
    if settings.get("condition_hash"):
        settings.pop("condition_id", None)  # aliases are not experimental conditions
    return {
        "condition_id": record.get("condition_id"),
        "condition_hash": record.get("condition_hash"),
        "provider_api": record.get("provider_api"),
        "route_hash": record.get("route_hash"),
        "settings_hash": stable_json_hash(settings),
        "controls": controls,
        "benchmark_condition_hash": record.get("benchmark_condition_hash"),
    }


def merge_conditions(model_id: str, conditions: list[dict], *, cross_module: bool = False) -> dict:
    """Refuse mixed or partially unidentified conditions instead of pooling them."""
    if not conditions:
        return describe_condition({})
    first = conditions[0]
    for other in conditions[1:]:
        if (
            first["settings_hash"] != other["settings_hash"]
            or (not cross_module and first.get("benchmark_condition_hash") != other.get("benchmark_condition_hash"))
        ):
            raise ValueError(
                f"Incompatible model conditions for {model_id}; select one recorded condition "
                "per model and do not mix known and unknown settings."
            )
    merged = dict(first)
    if cross_module:
        merged.pop("benchmark_condition_hash", None)
    return merged


def configured_effort(condition: dict | None) -> str:
    effort = (condition or {}).get("controls", {}).get("reasoning_effort")
    return str(effort) if effort is not None else "unknown (not recorded)"
