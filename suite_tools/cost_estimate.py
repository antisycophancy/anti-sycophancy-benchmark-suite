"""Pre-run benchmark call planning and non-binding cost estimates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "benchmark-cost-estimate-v1"
RANGE_KEYS = ("low", "expected", "high")

DEFAULT_TOKEN_PROFILES: dict[str, dict[str, dict[str, int]]] = {
    "model_under_test": {
        "input": {"low": 600, "expected": 1800, "high": 8000},
        "output": {"low": 120, "expected": 400, "high": 1000},
        "input_growth": {"low": 100, "expected": 500, "high": 1800},
    },
    "support": {
        "input": {"low": 500, "expected": 1400, "high": 5000},
        "output": {"low": 50, "expected": 120, "high": 200},
        "input_growth": {"low": 100, "expected": 400, "high": 1400},
    },
    "judge": {
        "input": {"low": 1200, "expected": 3500, "high": 14000},
        "output": {"low": 40, "expected": 180, "high": 900},
        "input_growth": {"low": 0, "expected": 0, "high": 0},
    },
}


def _range(value: int | dict[str, Any]) -> dict[str, int]:
    if isinstance(value, dict):
        return {key: max(0, int(value.get(key) or 0)) for key in RANGE_KEYS}
    parsed = max(0, int(value))
    return {key: parsed for key in RANGE_KEYS}


def _provider_route(record: dict[str, Any] | None) -> str:
    record = record or {}
    metadata = record.get("condition_metadata") or {}
    route = metadata.get("provider_route") if isinstance(metadata, dict) else None
    if route:
        return str(route)
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    config_metadata = config.get("condition_metadata") or {}
    route = config_metadata.get("provider_route") if isinstance(config_metadata, dict) else None
    if route:
        return str(route)
    endpoint = str(
        config.get("base_url")
        or record.get("base_url")
        or record.get("endpoint")
        or ""
    ).lower()
    if "openrouter" in endpoint:
        return "openrouter"
    if "googleapis" in endpoint:
        return "google_direct"
    if "anthropic" in endpoint:
        return "anthropic_direct"
    if "api.openai.com" in endpoint:
        return "openai_direct"
    return "openrouter"


def _profile(
    role: str,
    token_profiles: dict[str, Any] | None,
) -> tuple[dict[str, int], dict[str, int]]:
    supplied = (token_profiles or {}).get(role) or {}
    default = DEFAULT_TOKEN_PROFILES[role]
    return (
        _range(supplied.get("input", default["input"])),
        _range(supplied.get("output", default["output"])),
    )


def _line(
    *,
    stage: str,
    role: str,
    operation: str,
    model: str,
    provider: str,
    calls: int | dict[str, Any],
    token_profiles: dict[str, Any] | None,
    conversation_calls: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    input_tokens, output_tokens = _profile(role, token_profiles)
    line = {
        "stage": stage,
        "role": role,
        "operation": operation,
        "model": model,
        "provider": provider,
        "calls": _range(calls),
        "input_tokens_per_call": input_tokens,
        "output_tokens_per_call": output_tokens,
    }
    if conversation_calls:
        supplied = (token_profiles or {}).get(role) or {}
        growth = _range(
            supplied.get("input_growth", DEFAULT_TOKEN_PROFILES[role]["input_growth"])
        )
        line["input_growth_tokens_per_turn"] = growth
        line["input_tokens_total"] = {
            key: sum(
                (count := int(call_range[key])) * input_tokens[key]
                + ((count * (count - 1)) // 2) * growth[key]
                for call_range in conversation_calls
            )
            for key in RANGE_KEYS
        }
        line["output_tokens_total"] = {
            key: sum(int(call_range[key]) * output_tokens[key] for call_range in conversation_calls)
            for key in RANGE_KEYS
        }
    return line


def build_contract_call_plan(
    contract: dict[str, Any],
    *,
    token_profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic call-count plan from a prepared run contract."""
    models = {
        str(item.get("key") or item.get("model_id")): item
        for item in contract.get("expected_models") or []
        if isinstance(item, dict)
    }
    expected_judges = [
        item for item in contract.get("expected_judges") or [] if isinstance(item, dict)
    ]
    panel = [item for item in expected_judges if item.get("role") in {"panel", "primary", "judge"}]
    seeker = next((item for item in expected_judges if item.get("role") == "seeker"), None)
    analyzer = next((item for item in expected_judges if item.get("role") == "analyzer"), None)
    flip_generator = next(
        (item for item in expected_judges if item.get("role") == "flip_generator"),
        None,
    )
    lines: list[dict[str, Any]] = []
    sus_retry_contingency_applied = False

    for module in contract.get("modules") or []:
        if not isinstance(module, dict):
            continue
        module_name = str(module.get("module") or "unknown").lower()
        units = [item for item in module.get("expected_units") or [] if isinstance(item, dict)]
        model_calls: dict[tuple[str, str], int] = defaultdict(int)
        model_conversations: dict[tuple[str, str], list[dict[str, int]]] = defaultdict(list)
        support_calls = 0
        support_conversations: list[dict[str, int]] = []
        sus_conversations: list[dict[str, int]] = []
        sus_analysis_calls: list[dict[str, int]] = []
        for unit in units:
            key = str(unit.get("model_key") or unit.get("model_id") or "unknown")
            record = models.get(key) or {}
            model_id = str(unit.get("model_id") or record.get("model_id") or key)
            planned_turns = max(1, int(unit.get("planned_turns") or 1))
            if module_name in {"aita", "aite", "epistemic", "epis"}:
                model_calls[(model_id, _provider_route(record))] += planned_turns
                model_conversations[(model_id, _provider_route(record))].append(
                    _range(planned_turns)
                )
                support_calls += max(0, planned_turns - 1)
                if planned_turns > 1:
                    support_conversations.append(_range(planned_turns - 1))
            elif module_name == "sus":
                model_calls[(model_id, _provider_route(record))] += 1
                planned_escalations = max(0, int(unit.get("planned_escalations") or 2))
                expected_escalations = (planned_escalations + 1) // 2
                conversation_range = {
                    "low": 1,
                    "expected": 2 + expected_escalations,
                    "high": 2 + planned_escalations,
                }
                model_conversations[(model_id, _provider_route(record))].append(conversation_range)
                sus_conversations.append(conversation_range)
                adaptive = str(unit.get("escalation_mode") or "adaptive") == "adaptive"
                sus_analysis_calls.append({
                    "low": 0,
                    "expected": 2 + (expected_escalations if adaptive else 0),
                    "high": 2 + (planned_escalations if adaptive else 0),
                })
            else:
                model_calls[(model_id, _provider_route(record))] += 1
                model_conversations[(model_id, _provider_route(record))].append(_range(1))

        for (model_id, provider), count in model_calls.items():
            calls: int | dict[str, int] = count
            if module_name == "sus":
                calls = {
                    key: sum(group[key] for group in model_conversations[(model_id, provider)])
                    for key in RANGE_KEYS
                }
            model_line = _line(
                stage="generation", role="model_under_test", operation=f"{module_name}_conversation",
                model=model_id, provider=provider, calls=calls, token_profiles=token_profiles,
                conversation_calls=model_conversations[(model_id, provider)],
            )
            if module_name == "sus":
                model_line["calls"]["high"] *= 2
                for key in ("input_tokens_total", "output_tokens_total"):
                    if isinstance(model_line.get(key), dict):
                        model_line[key]["high"] *= 2
                model_line["retry_contingency"] = {
                    "scope": "high",
                    "retries_per_turn": 1,
                }
                sus_retry_contingency_applied = True
            lines.append(model_line)

        if module_name == "sus" and analyzer:
            count = len(units)
            lines.append(_line(
                stage="generation", role="judge", operation="sus_compliance",
                model=str(analyzer.get("model_id") or "unknown"),
                provider=_provider_route(analyzer),
                calls={key: sum(group[key] for group in sus_conversations) for key in RANGE_KEYS},
                token_profiles=token_profiles,
            ))

        if support_calls and seeker:
            lines.append(_line(
                stage="generation", role="support", operation="seeker",
                model=str(seeker.get("model_id") or "unknown"), provider=_provider_route(seeker),
                calls=support_calls, token_profiles=token_profiles,
                conversation_calls=support_conversations,
            ))

        if (
            module_name in {"aita", "aite"}
            and module.get("dataset_mode") == "yta-synthflip"
            and flip_generator
        ):
            flip_items = {unit.get("item_idx") for unit in units}
            lines.append(_line(
                stage="generation", role="support", operation="aita_flip",
                model=str(flip_generator.get("model_id") or "unknown"),
                provider=_provider_route(flip_generator),
                calls=len(flip_items),
                token_profiles=token_profiles,
            ))

        if module_name in {"aita", "aite"}:
            item_sides: dict[tuple[Any, Any], set[str]] = defaultdict(set)
            calls_per_judge = 0
            for unit in units:
                item_key = (
                    unit.get("model_key") or unit.get("model_id"),
                    unit.get("item_idx"),
                )
                side = str(unit.get("side") or "")
                if side:
                    item_sides[item_key].add(side)
                # Outcome and therapeutic scoring apply to every selected side.
                calls_per_judge += 2
                if max(1, int(unit.get("planned_turns") or 1)) >= 2:
                    calls_per_judge += 1
                if unit.get("ground_truth") in {"NTA", "YTA"}:
                    calls_per_judge += 1
            calls_per_judge += sum(
                1
                for sides in item_sides.values()
                if {"side_a", "side_b"}.issubset(sides)
            )
            for judge in panel:
                lines.append(_line(
                    stage="scoring", role="judge", operation="aita_dimensions",
                    model=str(judge.get("model_id") or "unknown"), provider=_provider_route(judge),
                    calls=calls_per_judge, token_profiles=token_profiles,
                ))
        elif module_name in {"epistemic", "epis"}:
            groups: dict[tuple[Any, Any, Any], int] = defaultdict(int)
            for unit in units:
                groups[(unit.get("model_key") or unit.get("model_id"), unit.get("item_idx"), unit.get("test_type"))] += 1
            calls_per_judge = sum(4 if side_count > 1 else 2 for side_count in groups.values())
            for judge in panel:
                lines.append(_line(
                    stage="scoring", role="judge", operation="epistemic_dimensions",
                    model=str(judge.get("model_id") or "unknown"), provider=_provider_route(judge),
                    calls=calls_per_judge, token_profiles=token_profiles,
                ))
        elif module_name == "sus":
            if analyzer:
                count = len(units)
                lines.append(_line(
                    stage="generation", role="support", operation="sus_analysis",
                    model=str(analyzer.get("model_id") or "unknown"), provider=_provider_route(analyzer),
                    calls={key: sum(group[key] for group in sus_analysis_calls) for key in RANGE_KEYS},
                    token_profiles=token_profiles,
                ))
            for judge in panel:
                count = len(units)
                lines.append(_line(
                    stage="scoring", role="judge", operation="sus_post_analysis",
                    model=str(judge.get("model_id") or "unknown"), provider=_provider_route(judge),
                    calls={"low": count, "expected": count, "high": count * 2},
                    token_profiles=token_profiles,
                ))
        else:
            for judge in panel:
                lines.append(_line(
                    stage="scoring", role="judge", operation="generic_score",
                    model=str(judge.get("model_id") or "unknown"), provider=_provider_route(judge),
                    calls=len(units), token_profiles=token_profiles,
                ))

    total_calls = {
        key: sum(line["calls"][key] for line in lines)
        for key in RANGE_KEYS
    }
    assumptions = [
        "Call counts come from the prepared contract and current benchmark stage formulas.",
        "Token ranges are planning estimates; actual usage is recorded from provider responses.",
        "High token ranges are conservative planning assumptions, not a spend ceiling.",
    ]
    if sus_retry_contingency_applied:
        assumptions.append(
            "The SUS target-model high range includes the default one retry per failed turn; "
            "expected and low ranges assume no retry."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "planned",
        "lines": lines,
        "total_calls": total_calls,
        "assumptions": assumptions,
    }


def _decimal(value: Any, label: str) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"{label} must be a non-negative finite pricing value")
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{label} must be a non-negative finite pricing value")
    return parsed


def validate_pricing_snapshot(pricing: dict[str, Any]) -> None:
    """Reject malformed money inputs before they can weaken a spend warning."""
    units = pricing.get("units") if isinstance(pricing, dict) else None
    if units not in {"per_token", "per_million_tokens"}:
        raise ValueError(
            "pricing snapshot units must be 'per_token' or 'per_million_tokens'"
        )
    price_models = pricing.get("models")
    if not isinstance(price_models, dict) and isinstance(pricing.get("checked"), list):
        price_models = {
            str(item["model_id"]): item.get("pricing") or {}
            for item in pricing["checked"]
            if isinstance(item, dict) and item.get("model_id")
        }
    if not isinstance(price_models, dict):
        raise ValueError("pricing snapshot models must be an object or checked list")
    for model, raw_price in price_models.items():
        if not isinstance(raw_price, dict):
            continue
        _decimal(raw_price.get("prompt", raw_price.get("input")), f"{model}.prompt")
        _decimal(
            raw_price.get("completion", raw_price.get("output")),
            f"{model}.completion",
        )


def estimate_call_plan(
    call_plan: dict[str, Any],
    pricing: dict[str, Any],
) -> dict[str, Any]:
    """Price a call plan using a captured snapshot with explicit units."""
    validate_pricing_snapshot(pricing)
    units = pricing.get("units") if isinstance(pricing, dict) else None
    if units not in {"per_token", "per_million_tokens"}:
        raise ValueError(
            "pricing snapshot units must be 'per_token' or 'per_million_tokens'"
        )
    unit_factor = Decimal("1") if units == "per_token" else Decimal("0.000001")
    price_models = pricing.get("models") if isinstance(pricing, dict) else None
    if not isinstance(price_models, dict) and isinstance(pricing.get("checked"), list):
        price_models = {
            str(item["model_id"]): item.get("pricing") or {}
            for item in pricing["checked"]
            if isinstance(item, dict) and item.get("model_id")
        }
    if not isinstance(price_models, dict):
        raise ValueError("pricing snapshot models must be an object or checked list")
    totals = {key: Decimal("0") for key in RANGE_KEYS}
    by_stage: dict[str, dict[str, Decimal]] = defaultdict(lambda: {key: Decimal("0") for key in RANGE_KEYS})
    by_role: dict[str, dict[str, Decimal]] = defaultdict(lambda: {key: Decimal("0") for key in RANGE_KEYS})
    by_provider: dict[str, dict[str, Decimal]] = defaultdict(lambda: {key: Decimal("0") for key in RANGE_KEYS})
    unknown: set[str] = set()
    priced_lines: list[dict[str, Any]] = []

    for line in call_plan.get("lines") or []:
        model = str(line.get("model") or "unknown")
        price = price_models.get(model)
        price = price if isinstance(price, dict) else {}
        input_price = _decimal(
            price.get("prompt", price.get("input")), f"{model}.prompt"
        )
        output_price = _decimal(
            price.get("completion", price.get("output")), f"{model}.completion"
        )
        if input_price is None or output_price is None:
            unknown.add(model)
            priced_lines.append({**line, "cost_state": "unknown_pricing"})
            continue
        line_cost = {}
        for key in RANGE_KEYS:
            calls = Decimal(int((line.get("calls") or {}).get(key) or 0))
            input_totals = line.get("input_tokens_total")
            output_totals = line.get("output_tokens_total")
            if isinstance(input_totals, dict) and isinstance(output_totals, dict):
                input_tokens = Decimal(int(input_totals.get(key) or 0))
                output_tokens = Decimal(int(output_totals.get(key) or 0))
                value = (
                    (input_tokens * input_price) + (output_tokens * output_price)
                ) * unit_factor
            else:
                input_tokens = Decimal(int((line.get("input_tokens_per_call") or {}).get(key) or 0))
                output_tokens = Decimal(int((line.get("output_tokens_per_call") or {}).get(key) or 0))
                value = calls * (
                    (input_tokens * input_price) + (output_tokens * output_price)
                ) * unit_factor
            totals[key] += value
            by_stage[str(line.get("stage") or "unknown")][key] += value
            by_role[str(line.get("role") or "unknown")][key] += value
            by_provider[str(line.get("provider") or "unknown")][key] += value
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"cost estimate for {model} is not finite")
            line_cost[key] = round(numeric_value, 8)
        priced_lines.append({
            **line,
            "cost_state": "estimated",
            "pricing_source": price.get("source") or "snapshot",
            "cost_usd": line_cost,
        })

    def floats(values: dict[str, Decimal]) -> dict[str, float]:
        result = {}
        for key in RANGE_KEYS:
            value = float(values[key])
            if not math.isfinite(value):
                raise ValueError("cost estimate total is not finite")
            result[key] = round(value, 8)
        return result

    known_count = len(priced_lines) - len([line for line in priced_lines if line["cost_state"] == "unknown_pricing"])
    state = "estimated" if not unknown else "partial" if known_count else "unavailable"
    return {
        "schema_version": SCHEMA_VERSION,
        "pricing_units": units,
        "state": state,
        "total_cost_usd": floats(totals),
        "cost_by_stage": {key: floats(value) for key, value in sorted(by_stage.items())},
        "cost_by_role": {key: floats(value) for key, value in sorted(by_role.items())},
        "cost_by_provider": {key: floats(value) for key, value in sorted(by_provider.items())},
        "unknown_pricing": sorted(unknown),
        "lines": priced_lines,
        "notice": "Planning estimate only; provider invoices and reported response costs are authoritative.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate a prepared benchmark run without making API calls.")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--pricing-snapshot", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = json.loads(Path(args.contract).read_text())
    pricing = json.loads(Path(args.pricing_snapshot).read_text())
    call_plan = build_contract_call_plan(contract)
    print(json.dumps({"call_plan": call_plan, "estimate": estimate_call_plan(call_plan, pricing)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
