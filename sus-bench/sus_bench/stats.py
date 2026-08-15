"""Statistical aggregation for multi-run benchmark results.

Provides confidence interval calculation and run aggregation without
requiring scipy — uses hardcoded t-distribution critical values for
small sample sizes (N=2 to N=30).
"""

from __future__ import annotations

import sys
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from suite_tools.statistics import bootstrap_ci, confidence_interval

from sus_bench.scoring_contract import (
    SUS_RESPONSE_COMPONENT_WEIGHTS,
    cap_rate_summary,
    first_capitulation_phase,
    is_capitulation_result,
    is_score_excluded_result,
)

def _model_condition_key(result: dict[str, Any]) -> tuple[Any, ...]:
    """Return the model condition identity used for score aggregation."""
    return (
        result.get("condition_hash"),
        result.get("condition_id"),
        result.get("provider_api"),
        result.get("model"),
        result.get("label"),
        _freeze_jsonish(result.get("request_options")),
    )


def _freeze_jsonish(value: Any) -> Any:
    """Convert JSON-like metadata into a stable hashable shape."""
    if isinstance(value, dict):
        return tuple((key, _freeze_jsonish(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_freeze_jsonish(item) for item in value)
    return value


def aggregate_runs(results: list[dict]) -> list[dict]:
    """Aggregate multi-run results by model condition + scenario.

    Groups results and computes mean, stddev, and 95% CI for the overall SUS
    score and each sub-dimension. The condition identity intentionally includes
    label/request_options so the same provider model can be tested under
    multiple OpenRouter controls without collapsing into one aggregate.

    Args:
        results: List of result dicts from run_benchmark.

    Returns:
        List of aggregated result dicts, one per model-condition+scenario pair.
    """
    # Group by (model condition, scenario)
    groups: dict[tuple[tuple[Any, ...], str], list[dict]] = defaultdict(list)
    for r in results:
        key = (_model_condition_key(r), r.get("scenario", "unknown"))
        groups[key].append(r)

    aggregated = []
    for (_condition_key, scenario_id), runs in groups.items():
        cap_summary = cap_rate_summary(runs)
        included_runs = [r for r in runs if not is_score_excluded_result(r)]
        n = cap_summary["n_valid"]
        model_id = runs[0]["model"]
        label = runs[0].get("label", model_id)
        scenario_name = runs[0].get("scenario_name", scenario_id)

        # Collect scores
        sus_scores = [r["score"]["sus"] for r in included_runs if r.get("score")]
        irq_scores = [r["score"]["irq"] for r in included_runs if r.get("score") and "irq" in r["score"]]
        pr_scores = [r["score"]["pr"] for r in included_runs if r.get("score") and "pr" in r["score"]]
        er_scores = [r["score"]["er"] for r in included_runs if r.get("score") and "er" in r["score"]]
        ca_scores = [r["score"]["ca"] for r in included_runs if r.get("score") and "ca" in r["score"]]
        component_scores = {
            key: [r["score"][key] for r in included_runs if r.get("score") and key in r["score"]]
            for key in SUS_RESPONSE_COMPONENT_WEIGHTS
        }

        # Compute CIs (t-distribution)
        sus_mean, sus_lo, sus_hi = confidence_interval(sus_scores)
        irq_mean, _, _ = confidence_interval(irq_scores)
        pr_mean, _, _ = confidence_interval(pr_scores)
        er_mean, _, _ = confidence_interval(er_scores)
        ca_mean, _, _ = confidence_interval(ca_scores)

        # Bootstrap CI
        sus_boot_mean, sus_boot_lo, sus_boot_hi = bootstrap_ci(sus_scores)

        # Determine capitulation point from the most common phase pattern.
        capitulation_phase = _most_common_capitulation_point(included_runs)
        capitulation_count = cap_summary["cap_count"]
        capitulation_rate = round(cap_summary["cap_rate"], 3) if n else 0.0

        aggregated.append({
            "model": model_id,
            "label": label,
            "provider_api": runs[0].get("provider_api"),
            "condition_id": runs[0].get("condition_id"),
            "condition_hash": runs[0].get("condition_hash"),
            "condition_metadata": runs[0].get("condition_metadata"),
            "request_options": runs[0].get("request_options"),
            "scenario": scenario_id,
            "scenario_name": scenario_name,
            "runs": n,
            "sus_mean": sus_mean,
            "sus_ci_lower": sus_lo,
            "sus_ci_upper": sus_hi,
            "sus_bootstrap_mean": sus_boot_mean,
            "sus_bootstrap_ci_lower": sus_boot_lo,
            "sus_bootstrap_ci_upper": sus_boot_hi,
            "sus_stddev": round(statistics.stdev(sus_scores), 2) if len(sus_scores) > 1 else 0,
            "sus_response_score_mean": sus_mean,
            "irq_mean": irq_mean,
            "pr_mean": pr_mean,
            "er_mean": er_mean,
            "ca_mean": ca_mean,
            "sus_response_component_means": {
                key: confidence_interval(scores)[0] for key, scores in component_scores.items()
            },
            "capitulation_phase": capitulation_phase,
            "capitulation_count": capitulation_count,
            "capitulation_rate": capitulation_rate,
            "no_cap_count": cap_summary["no_cap_count"],
            "excluded_provider_refusal_count": cap_summary["n_excluded_provider_refusal"],
            "cap_rate": capitulation_rate,
            "cap_rate_wilson_95_ci_low": cap_summary["cap_rate_wilson_95_ci_low"],
            "cap_rate_wilson_95_ci_high": cap_summary["cap_rate_wilson_95_ci_high"],
            "capitulation_rate_wilson_95_ci_low": cap_summary["cap_rate_wilson_95_ci_low"],
            "capitulation_rate_wilson_95_ci_high": cap_summary["cap_rate_wilson_95_ci_high"],
            # Keep individual runs for detailed inspection
            "individual_results": runs,
        })

    return aggregated


def _most_common_capitulation_point(runs: list[dict]) -> str:
    """Determine the most common point where the model capitulated."""
    capitulation_points: dict[str, int] = defaultdict(int)

    for r in runs:
        capitulation = first_capitulation_phase(r.get("phases"))
        if capitulation is None:
            capitulation_points["no capitulation"] += 1
            continue
        phase_key, phase = capitulation
        if phase_key == "elicit":
            capitulation_points["initial response"] += 1
        elif phase_key == "follow":
            capitulation_points["passphrase (1 turn)"] += 1
        else:
            capitulation_points[_phase_strategy_label(phase_key, phase)] += 1

    if not capitulation_points:
        return "unknown"

    return max(capitulation_points, key=capitulation_points.get)


def _phase_strategy_label(phase_key: str, phase: dict) -> str:
    """Return the scenario-specific escalation strategy label for a phase."""
    if isinstance(phase, dict):
        strategy = phase.get("strategy")
        if strategy:
            return str(strategy)
    return phase_key.replace("_", " ")


def _is_capitulation(run: dict) -> bool:
    """Return whether a run contains phase-derived target-info capitulation."""
    return is_capitulation_result(run)
