"""Prepare no-paid benchmark run contracts for dashboard review.

The prepare step writes the intended run footprint before paid calls begin:
rendered model config, expected units, provenance hashes, and the exact command
to execute later. It does not contact providers or run benchmarks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import shlex
import os
import sys
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from suite_tools.cost_estimate import (
    build_contract_call_plan,
    estimate_call_plan,
    validate_pricing_snapshot,
)
from suite_tools.env import read_repo_env_values
from suite_tools.model_config import (
    DEFAULT_SUITE_CONFIG,
    MODEL_CONDITION_METADATA_FIELDS,
    expand_model_keys,
    load_suite_config,
    render_model_condition,
    render_module_config,
    validate_suite_config,
)
from suite_tools.paid_call_lease import (
    PAID_CALL_LIMIT_ENV_NAMES,
    paid_call_capacity_report,
)
from suite_tools.run_monitor import atomic_write_json
from suite_tools.run_contract import (
    build_provenance_identity,
    load_run_plan,
    stable_json_hash,
    write_run_contract,
    write_run_plan,
)
from suite_tools.scoring_contracts import get_scoring_contract
from suite_tools.suite_registry import REPO_ROOT, suite_root

SUS_ROOT = suite_root("sus")
AITA_ROOT = suite_root("aita")
EPIS_ROOT = suite_root("epistemic")
DEFAULT_PREPARED_ROOT = REPO_ROOT / "results" / "prepared"
PREPARED_PRICING_SCHEMA_VERSION = "benchmark-prepared-pricing-v1"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _repo_display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _stored_contract_provenance(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    provenance = contract.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("prepared contract is missing provenance")
    return provenance


def _refresh_run_plan_provenance(contract_path: Path) -> None:
    run_group_dir = contract_path.parent.parent
    plan = load_run_plan(run_group_dir)
    if not plan:
        return
    target = _repo_display(contract_path)
    changed = False
    modules = []
    for raw_module in plan.get("modules") or []:
        if not isinstance(raw_module, dict):
            modules.append(raw_module)
            continue
        module = dict(raw_module)
        if module.get("contract_path") == target:
            module["provenance"] = _stored_contract_provenance(contract_path)
            changed = True
        modules.append(module)
    if changed:
        plan["modules"] = modules
        write_run_plan(run_group_dir, plan)


def _bind_prepared_pricing(
    contract: dict[str, Any],
    *,
    call_plan: dict[str, Any],
    pricing_snapshot: dict[str, Any] | None = None,
    warning_threshold_usd: float | None = None,
) -> None:
    binding: dict[str, Any] = {
        "schema_version": PREPARED_PRICING_SCHEMA_VERSION,
        "call_plan": call_plan,
    }
    if pricing_snapshot is not None:
        binding["pricing_snapshot"] = pricing_snapshot
    if warning_threshold_usd is not None:
        binding["warning_threshold_usd"] = warning_threshold_usd
    identity = dict(contract.get("identity") or {})
    execution = dict(identity.get("execution") or {})
    execution["prepared_pricing"] = binding
    identity["execution"] = execution
    contract["identity"] = identity
    contract.pop("provenance", None)


def _attach_call_plan(contract_path: Path) -> Path:
    contract = json.loads(contract_path.read_text())
    call_plan = build_contract_call_plan(contract)
    contract["call_plan"] = call_plan
    _bind_prepared_pricing(contract, call_plan=call_plan)
    return write_run_contract(contract_path.parent, contract)


def _pricing_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot.get("checked"), list):
        return {
            "schema_version": snapshot.get("schema_version")
            or "benchmark-pricing-snapshot-v1",
            "units": snapshot.get("units"),
            "provider": snapshot.get("provider") or "openrouter",
            "generated_at": snapshot.get("generated_at"),
            "source": snapshot.get("catalog_source") or snapshot.get("source"),
            "models": {
                str(item["model_id"]): item.get("pricing") or {}
                for item in snapshot["checked"]
                if isinstance(item, dict) and item.get("model_id")
            },
        }
    return snapshot


def _attach_cost_estimate(contract_path: Path, pricing: dict[str, Any]) -> Path:
    contract = json.loads(contract_path.read_text())
    validate_pricing_snapshot(pricing)
    call_plan = build_contract_call_plan(contract)
    contract["call_plan"] = call_plan
    contract["cost_estimate"] = estimate_call_plan(call_plan, pricing)
    frozen_name = "PRICING_SNAPSHOT.json"
    frozen_path = contract_path.parent / frozen_name
    atomic_write_json(frozen_path, pricing)
    frozen_bytes = frozen_path.read_bytes()
    frozen_binding = {
        "path": frozen_name,
        "sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "bytes": len(frozen_bytes),
    }
    contract["pricing_snapshot"] = {
        **frozen_binding,
        "pricing_hash": stable_json_hash(pricing),
        "schema_version": pricing.get("schema_version"),
        "units": pricing.get("units"),
        "provider": pricing.get("provider"),
        "generated_at": pricing.get("generated_at"),
        "source": pricing.get("source"),
    }
    _bind_prepared_pricing(
        contract,
        call_plan=call_plan,
        pricing_snapshot=frozen_binding,
    )
    written_path = write_run_contract(contract_path.parent, contract)
    _refresh_run_plan_provenance(written_path)
    return written_path


def _attach_cost_warning(contract_path: Path, threshold_usd: float) -> tuple[Path, bool]:
    if not math.isfinite(threshold_usd) or threshold_usd < 0:
        raise ValueError("cost warning threshold must be non-negative and finite")
    contract = json.loads(contract_path.read_text())
    estimate = contract.get("cost_estimate")
    estimate = estimate if isinstance(estimate, dict) else {}
    total = estimate.get("total_cost_usd")
    total = total if isinstance(total, dict) else {}
    high = float(total.get("high") or 0)
    complete = estimate.get("state") == "estimated"
    exceeded = complete and high > threshold_usd
    contract["cost_warning"] = {
        "state": "exceeded" if exceeded else "within" if complete else "unavailable",
        "warning_threshold_usd": threshold_usd,
        "estimated_high_usd": high if complete else None,
        "notice": (
            "Preparation warning only; this does not stop an executing run."
        ),
    }
    identity = contract.get("identity") if isinstance(contract.get("identity"), dict) else {}
    execution = identity.get("execution") if isinstance(identity, dict) else {}
    prior_binding = (
        execution.get("prepared_pricing") if isinstance(execution, dict) else None
    )
    frozen_binding = (
        prior_binding.get("pricing_snapshot")
        if isinstance(prior_binding, dict)
        and isinstance(prior_binding.get("pricing_snapshot"), dict)
        else None
    )
    call_plan = build_contract_call_plan(contract)
    contract["call_plan"] = call_plan
    _bind_prepared_pricing(
        contract,
        call_plan=call_plan,
        pricing_snapshot=frozen_binding,
        warning_threshold_usd=threshold_usd,
    )
    written_path = write_run_contract(contract_path.parent, contract)
    _refresh_run_plan_provenance(written_path)
    return written_path, not exceeded and complete


def _cost_range_text(value: Any) -> str:
    if not isinstance(value, dict):
        return "unavailable"
    low = float(value.get("low") or 0)
    expected = float(value.get("expected") or 0)
    high = float(value.get("high") or 0)
    return f"${expected:.4f} expected (${low:.4f}-${high:.4f})"


def _print_cost_estimate(contract: dict[str, Any]) -> None:
    estimate = contract.get("cost_estimate")
    if not isinstance(estimate, dict):
        print("Planning cost estimate: unavailable (attach --pricing-snapshot)")
        return
    print(f"Planning cost estimate: {_cost_range_text(estimate.get('total_cost_usd'))}")
    by_stage = estimate.get("cost_by_stage") if isinstance(estimate.get("cost_by_stage"), dict) else {}
    print(f"  Generation: {_cost_range_text(by_stage.get('generation'))}")
    print(f"  Scoring:    {_cost_range_text(by_stage.get('scoring'))}")
    unknown = estimate.get("unknown_pricing")
    if isinstance(unknown, list) and unknown:
        print(f"  Unknown pricing: {', '.join(str(model) for model in unknown)}")
    warning = contract.get("cost_warning")
    if isinstance(warning, dict):
        threshold = float(warning.get("warning_threshold_usd") or 0)
        high = warning.get("estimated_high_usd")
        if warning.get("state") == "exceeded":
            print(
                f"  WARNING: high estimate ${float(high):.4f} exceeds warning threshold "
                f"${threshold:.4f}."
            )
        elif warning.get("state") == "unavailable":
            print("  WARNING: cannot evaluate threshold because pricing is incomplete.")
        else:
            print(f"  Warning threshold: ${threshold:.4f} (not exceeded)")
    print("  Actual provider billing may differ.")


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _quote(value: Path | str) -> str:
    return shlex.quote(str(value))


def _resolve_optional_path(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(value).resolve())


def _python_module_argv(module: str, args: list[str]) -> list[str]:
    # Bake the preparing interpreter into the contract so prepared runs work
    # regardless of the operator's venv name/location or the step cwd.
    return [sys.executable, "-m", module, *args]


def _command_step(*, cwd: Path, argv: list[str]) -> dict[str, Any]:
    return {"cwd": str(cwd.resolve()), "argv": list(argv)}


def _command_from_steps(steps: list[dict[str, Any]], *, separator: str = " && \\\n  ") -> str:
    if not steps:
        return ""
    first = steps[0]
    cwd = Path(first["cwd"])
    rendered_steps = [" ".join(_quote(item) for item in step["argv"]) for step in steps]
    return "\n".join([f"cd {_quote(cwd)}", separator.join(rendered_steps)])


def _first_step_fields(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {}
    step = steps[0]
    return {
        "cwd": step["cwd"],
        "argv": list(step["argv"]),
    }


def _structured_command_fields(prefix: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {}
    first = _first_step_fields(steps)
    fields = {
        f"{prefix}_steps": steps,
        f"{prefix}_cwd": first["cwd"],
        f"{prefix}_argv": first["argv"],
    }
    return fields


def _prepared_command_binding(
    *,
    execute_steps: list[dict[str, Any]],
    score_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the exact structured commands authenticated by run provenance."""
    return {
        "schema_version": "benchmark-prepared-commands-v1",
        "execute_steps": [
            {"cwd": step["cwd"], "argv": list(step["argv"])}
            for step in execute_steps
        ],
        "score_steps": [
            {"cwd": step["cwd"], "argv": list(step["argv"])}
            for step in score_steps
        ],
    }


def _python_module_steps(*, cwd: Path, module: str, args: list[str]) -> list[dict[str, Any]]:
    return [_command_step(cwd=cwd, argv=_python_module_argv(module, args))]


def _python_module_step(*, cwd: Path, module: str, args: list[str]) -> dict[str, Any]:
    return _python_module_steps(cwd=cwd, module=module, args=args)[0]


def _load_sus_constants() -> dict[str, Any]:
    if str(SUS_ROOT) not in sys.path:
        sys.path.insert(0, str(SUS_ROOT))
    from sus_bench import __version__ as sus_version
    from sus_bench.analyzer import ADAPTIVE_ESCALATION_PROMPT, EXTRACT_PROMPT, FOLLOW_PROMPT
    from sus_bench.scorer import (
        DEFAULT_WEIGHTS,
        POST_ANALYSIS_PROMPT,
        RUBRIC_SOURCE_IDS,
        RUBRIC_SOURCE_REGISTRY,
        RUBRIC_VERSION,
    )
    from sus_bench.scoring_contract import (
        SUS_LEGACY_DIAGNOSTIC_DIMENSIONS,
        SUS_RELEASE_SCORE_DIMENSIONS,
        SUS_RESPONSE_COMPONENT_WEIGHTS,
    )

    return {
        "version": sus_version,
        "extract_prompt": EXTRACT_PROMPT,
        "follow_prompt": FOLLOW_PROMPT,
        "adaptive_escalation_prompt": ADAPTIVE_ESCALATION_PROMPT,
        "post_analysis_prompt": POST_ANALYSIS_PROMPT,
        "rubric_version": RUBRIC_VERSION,
        "rubric_source_ids": list(RUBRIC_SOURCE_IDS),
        "rubric_source_registry": RUBRIC_SOURCE_REGISTRY,
        "score_dimensions": list(SUS_RELEASE_SCORE_DIMENSIONS),
        "score_component_weights": dict(SUS_RESPONSE_COMPONENT_WEIGHTS),
        "legacy_diagnostic_dimensions": list(SUS_LEGACY_DIAGNOSTIC_DIMENSIONS),
        "legacy_weights": dict(DEFAULT_WEIGHTS),
    }


def _load_sus_scenarios(selector: str | None) -> list[dict[str, Any]]:
    scenarios_dir = SUS_ROOT / "scenarios"
    wanted = {item.strip() for item in (selector or "").split(",") if item.strip()}
    scenarios: list[dict[str, Any]] = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            continue
        data["_filename_stem"] = path.stem
        data["_path"] = _repo_display(path)
        if not wanted or data.get("id") in wanted or path.stem in wanted:
            scenarios.append(data)
    if not scenarios:
        raise ValueError(f"No SUS scenarios matched: {selector}")
    return scenarios


def _load_aita_runner():
    if str(AITA_ROOT) not in sys.path:
        sys.path.insert(0, str(AITA_ROOT))
    from aita_bench import runner as aita_runner

    return aita_runner


def _load_epis_runner():
    if str(EPIS_ROOT) not in sys.path:
        sys.path.insert(0, str(EPIS_ROOT))
    from epis_bench import runner as epis_runner

    return epis_runner


def _load_epis_prompts():
    if str(EPIS_ROOT) not in sys.path:
        sys.path.insert(0, str(EPIS_ROOT))
    from epis_bench import prompts as epis_prompts

    return epis_prompts


def _repo_relative_or_name(path_value: str | None) -> str | None:
    """Return a repo-relative path for hashed identity payloads.

    Absolute operator paths must not flow into sample_spec: the same selection
    file prepared from a different checkout location would otherwise produce a
    different sample_hash/comparison_spec_hash. Falls back to the file name for
    paths outside the repo.
    """
    if not path_value:
        return path_value
    path = Path(path_value)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _selected_models(
    config: dict[str, Any],
    selector: str,
    *,
    module: str,
    suite_config_path: Path = DEFAULT_SUITE_CONFIG,
) -> list[dict[str, Any]]:
    keys = expand_model_keys(config, selector)
    defaults = config.get("defaults") or {}
    models = config.get("models") or {}
    selected = []
    for key in keys:
        model = dict(models[key])
        rendered = render_model_condition(config, key, module)
        endpoint = model.get("endpoint", defaults.get("endpoint", "openrouter"))
        selected.append(
            {
                "key": key,
                "label": model.get("label") or key,
                "model_id": model["model_id"],
                "endpoint": endpoint,
                "max_parallel": model.get("max_parallel", defaults.get("max_parallel")),
                "source": _repo_display(suite_config_path),
                **{
                    field: rendered[field]
                    for field in MODEL_CONDITION_METADATA_FIELDS
                    if field in rendered
                },
            }
        )
    return selected


def _write_rendered_config(
    *,
    suite_config: dict[str, Any],
    run_group_dir: Path,
    judge_set: str,
    model_selector: str,
    module: str,
    agent_profile: str | None,
) -> Path:
    config_dir = run_group_dir / "_configs" / judge_set
    config_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_module_config(
        suite_config,
        module,
        judge_set=judge_set,
        model_selector=model_selector,
        agent_profile=agent_profile,
    )
    path = config_dir / f"{module}-models.yaml"
    path.write_text(yaml.safe_dump(rendered, sort_keys=False))
    return path


def _prepared_config_binding(config_path: Path, run_group_dir: Path) -> dict[str, Any]:
    """Return the run-group-relative identity binding for rendered YAML bytes."""
    resolved_root = run_group_dir.resolve()
    resolved_path = config_path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("rendered model config must stay inside the prepared run group")
    relative_path = resolved_path.relative_to(resolved_root).as_posix()
    raw = resolved_path.read_bytes()
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _run_plan_path(run_group_dir: Path) -> Path:
    return run_group_dir / "RUN_PLAN.json"


def _contract_payload(
    contract_path: Path,
    paid_call_capacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with contract_path.open() as handle:
        contract = yaml.safe_load(handle) or {}
    payload = {
        "contract_path": _repo_display(contract_path),
        "run_plan_path": _repo_display(_run_plan_path(contract_path.parent.parent)),
        "run_id": contract.get("run_id"),
        "module": (contract.get("modules") or [{}])[0].get("module"),
        "lifecycle_state": contract.get("lifecycle_state"),
        "model_selector": contract.get("model_selector"),
        "judge_set": contract.get("judge_set"),
        "agent_profile": contract.get("agent_profile", "default"),
        "expected_units": sum(
            len(module.get("expected_units") or [])
            for module in contract.get("modules") or []
            if isinstance(module, dict)
        ),
        "execute_command": contract.get("execute_command"),
        "score_command": contract.get("score_command"),
    }
    if paid_call_capacity is not None:
        payload["paid_call_capacity"] = paid_call_capacity
    return payload


def _prepared_paid_call_capacity() -> dict[str, Any]:
    capacity_environment = read_repo_env_values(PAID_CALL_LIMIT_ENV_NAMES)
    return paid_call_capacity_report(environment=capacity_environment)


def _scheduler_run_command(
    contract_path: Path,
    paid_call_capacity: dict[str, Any],
) -> str:
    argv = [
        "./venv/bin/python",
        "-m",
        "suite_tools.scheduler",
        "run",
        "--contract",
        _repo_display(contract_path),
    ]
    limit = paid_call_capacity.get("effective_limit")
    if isinstance(limit, int) and limit > 0:
        argv.extend(["--max-active-calls", str(limit)])
    argv.append("--stop-on-attention")
    return shlex.join(argv)


def _upsert_run_plan(
    *,
    run_id: str,
    run_group_dir: Path,
    module_plan: dict[str, Any],
    source_command: str,
    model_selector: str,
    judge_set: str,
    agent_profile: str | None,
) -> Path:
    existing = load_run_plan(run_group_dir)
    modules = [
        module
        for module in list(existing.get("modules") or [])
        if isinstance(module, dict)
        and not (
            module.get("module") == module_plan.get("module")
            and module.get("output_dir") == module_plan.get("output_dir")
        )
    ]
    modules.append(module_plan)
    modules.sort(key=lambda module: str(module.get("module") or ""))

    plan = {
        **existing,
        "run_id": run_id,
        "lifecycle_state": "prepared",
        "source_command": source_command,
        "results_root": _repo_display(run_group_dir),
        "model_selector": model_selector,
        "judge_set": judge_set,
        "agent_profile": agent_profile or "default",
        "modules": modules,
        "completion_gates": [
            "review prepared plan before paid calls",
            "run modules only from rendered configs",
            "all expected contracts terminal",
            "all scored artifacts reviewed before promotion",
        ],
    }
    return write_run_plan(run_group_dir, plan)


def prepare_sus_run(
    *,
    run_id: str,
    output_root: Path,
    suite_config_path: Path,
    model_selector: str,
    judge_set: str,
    scenarios_selector: str | None,
    runs: int,
    escalation_mode: str = "adaptive",
    agent_profile: str | None = None,
    source_command: str = "",
) -> Path:
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    if escalation_mode not in {"adaptive", "static"}:
        raise ValueError("--escalation-mode must be 'adaptive' or 'static'")

    suite_config = load_suite_config(suite_config_path)
    validate_suite_config(suite_config)

    run_group_dir = output_root
    module_dir = run_group_dir / "sus"
    module_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_rendered_config(
        suite_config=suite_config,
        run_group_dir=run_group_dir,
        judge_set=judge_set,
        model_selector=model_selector,
        module="sus",
        agent_profile=agent_profile,
    )
    prepared_config = _prepared_config_binding(config_path, run_group_dir)
    rendered_sus_config = yaml.safe_load(config_path.read_text())
    scenarios = _load_sus_scenarios(scenarios_selector)
    constants = _load_sus_constants()
    from sus_bench.runner import sus_transcript_filename
    scoring_contract = get_scoring_contract("sus")
    expected_models = _selected_models(
        suite_config, model_selector, module="sus", suite_config_path=suite_config_path
    )
    analyzer = rendered_sus_config.get("analyzer")
    judge_panel = list(rendered_sus_config.get("judge_panel") or [])
    judge_configs = list(rendered_sus_config.get("judge_configs") or [])

    contract_units = []
    for model in expected_models:
        for scenario in scenarios:
            for run_number in range(1, runs + 1):
                contract_units.append(
                    {
                        "unit_id": f"sus:{model['key']}:{scenario['id']}:run{run_number}",
                        "model_key": model["key"],
                        "model_id": model["model_id"],
                        "scenario": scenario["id"],
                        "run_number": run_number,
                        "planned_escalations": len(scenario.get("escalation") or []),
                        "escalation_mode": escalation_mode,
                        "expected_transcript_path": (
                            "transcripts/"
                            + sus_transcript_filename(
                                model,
                                scenario,
                                run_number,
                                request_options=model.get("request_options"),
                            )
                        ),
                    }
                )

    scenario_ids = [scenario["id"] for scenario in scenarios]
    execute_steps = _python_module_steps(
        cwd=SUS_ROOT,
        module="sus_bench",
        args=[
            "run",
            "--models",
            str(config_path.resolve()),
            "--runs",
            str(runs),
            "--scenarios",
            ",".join(scenario_ids),
            "--escalation-mode",
            escalation_mode,
            "--output",
            str(module_dir.resolve()),
        ],
    )
    execute_command = _command_from_steps(execute_steps)
    score_steps = _python_module_steps(
        cwd=SUS_ROOT,
        module="sus_bench",
        args=[
            "score",
            "--input",
            str(module_dir.resolve()),
            "--models",
            str(config_path.resolve()),
            "--output",
            str((module_dir / "FINAL_RESULTS.json").resolve()),
        ],
    )
    score_command = _command_from_steps(score_steps)
    identity = build_provenance_identity(
        benchmark_family_id="sus",
        benchmark_spec={
            "module": "sus",
            "module_version": constants["version"],
            "escalation_mode": escalation_mode,
            "phase_prompts": {
                "extract": stable_json_hash(constants["extract_prompt"]),
                "follow": stable_json_hash(constants["follow_prompt"]),
                "adaptive_escalation": stable_json_hash(constants["adaptive_escalation_prompt"]),
                "post_analysis": stable_json_hash(constants["post_analysis_prompt"]),
            },
            "score_dimensions": constants["score_dimensions"],
            "scoring_contract": scoring_contract.as_benchmark_spec(),
            "score_component_weights": constants["score_component_weights"],
            "legacy_diagnostic_dimensions": constants["legacy_diagnostic_dimensions"],
            "legacy_score_weights": constants["legacy_weights"],
        },
        sample_spec={
            "scenario_ids": scenario_ids,
            "scenario_hashes": {
                scenario["id"]: stable_json_hash(
                    {key: value for key, value in scenario.items() if not str(key).startswith("_")}
                )
                for scenario in scenarios
            },
            "runs": runs,
        },
        judge_panel={
            "analyzer": analyzer,
            "panel": judge_panel,
            "configs": judge_configs,
            "judge_prompt_hashes": {
                "post_analysis": stable_json_hash(constants["post_analysis_prompt"]),
            },
            "rubric_version": constants["rubric_version"],
            "rubric_source_ids": constants["rubric_source_ids"],
            "rubric_source_registry": constants["rubric_source_registry"],
            "judge_set": judge_set,
        },
        model_conditions=expected_models,
        execution={
            "run_id": run_id,
            "results_root": _repo_display(run_group_dir),
            "module_output_dir": _repo_display(module_dir),
            "runner": "sus_bench.cli",
            "contract_scope": "module",
            "prepared": True,
            "escalation_mode": escalation_mode,
            "agent_profile": agent_profile or "default",
            "prepared_config": prepared_config,
            "prepared_commands": _prepared_command_binding(
                execute_steps=execute_steps,
                score_steps=score_steps,
            ),
        },
    )

    contract_path = write_run_contract(
        module_dir,
        {
            "run_id": run_id,
            "lifecycle_state": "prepared",
            "source_command": source_command,
            "execute_command": execute_command,
            **_structured_command_fields("execute", execute_steps),
            "score_command": score_command,
            **_structured_command_fields("score", score_steps),
            "results_root": _repo_display(run_group_dir),
            "contract_scope": "module",
            "model_selector": model_selector,
            "judge_set": judge_set,
            "agent_profile": agent_profile or "default",
            "identity": identity,
            "expected_models": expected_models,
            "expected_judges": [
                {"role": "analyzer", "model_id": analyzer},
                *[
                    {
                        "role": "panel",
                        "model_id": judge,
                        "config": judge_configs[index] if index < len(judge_configs) else None,
                    }
                    for index, judge in enumerate(judge_panel)
                ],
            ],
            "modules": [
                {
                    "module": "sus",
                    "stage": "generation",
                    "output_dir": ".",
                    "scenarios": scenario_ids,
                    "runs": runs,
                    "escalation_mode": escalation_mode,
                    "expected_units": contract_units,
                    "expected_artifacts": [
                        {"kind": "run_contract", "path": "RUN_CONTRACT.json", "required_for": "diagnostic"},
                        {
                            "kind": "rendered_models",
                            **prepared_config,
                            "required_for": "diagnostic",
                        },
                        {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "promotion"},
                        {"kind": "run_events", "path": "RUN_EVENTS.jsonl", "required_for": "promotion"},
                        {"kind": "summary_json", "path": "<runtime-run-id>.json", "required_for": "diagnostic"},
                        {
                            "kind": "conversations_json",
                            "path": "<runtime-run-id>-conversations.json",
                            "required_for": "scoring",
                        },
                        {"kind": "final_results", "path": "FINAL_RESULTS.json", "required_for": "promotion"},
                    ],
                }
            ],
            "completion_gates": [
                "review prepared contract before paid calls",
                "all expected SUS runs complete",
                "conversation JSON written",
                "score only after generation completes without incomplete conversations",
                "scoring status completed and score_ready",
            ],
        },
    )
    contract_path = _attach_call_plan(contract_path)
    _upsert_run_plan(
        run_id=run_id,
        run_group_dir=run_group_dir,
        module_plan={
            "module": "sus",
            "stage": "generation",
            "lifecycle_state": "prepared",
            "output_dir": _repo_display(module_dir),
            "contract_path": _repo_display(contract_path),
            "config_path": _repo_display(config_path),
            "execute_command": execute_command,
            "score_command": score_command,
            "model_selector": model_selector,
            "judge_set": judge_set,
            "agent_profile": agent_profile or "default",
            "expected_units": len(contract_units),
            "model_keys": [model["key"] for model in expected_models],
            "scenario_ids": scenario_ids,
            "runs": runs,
            "escalation_mode": escalation_mode,
            "provenance": _stored_contract_provenance(contract_path),
        },
        source_command=source_command,
        model_selector=model_selector,
        judge_set=judge_set,
        agent_profile=agent_profile,
    )
    return contract_path


def prepare_aita_run(
    *,
    run_id: str,
    output_root: Path,
    suite_config_path: Path,
    model_selector: str,
    judge_set: str,
    items: str,
    dataset_mode: str,
    data: str | None = None,
    og_data: str | None = None,
    flip_data: str | None = None,
    paired_labels: str | None = None,
    item_selection: str | None = None,
    sealed_pack: str | None = None,
    sealed_key_part_b_from_env: bool = False,
    sealed_pack_key_part_b: str | None = None,
    allow_sample_fallback: bool = False,
    agent_profile: str | None = None,
    source_command: str = "",
) -> Path:
    suite_config = load_suite_config(suite_config_path)
    validate_suite_config(suite_config)
    aita_runner = _load_aita_runner()

    run_group_dir = output_root
    module_dir = run_group_dir / "aita"
    module_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_rendered_config(
        suite_config=suite_config,
        run_group_dir=run_group_dir,
        judge_set=judge_set,
        model_selector=model_selector,
        module="aita",
        agent_profile=agent_profile,
    )
    prepared_config = _prepared_config_binding(config_path, run_group_dir)
    rendered_aita_config = yaml.safe_load(config_path.read_text())
    aita_judge_configs = list((rendered_aita_config.get("judge") or {}).get("configs") or [])
    aita_primary_judge_config = (rendered_aita_config.get("judge") or {}).get("primary_config")
    aita_expected_judges = [
        {"role": "panel", "model_id": config.get("model_id"), "config": config}
        for config in aita_judge_configs
    ] or [
        {
            "role": "panel",
            "model_id": aita_runner.JUDGE_MODEL,
            "config": aita_primary_judge_config,
        }
    ]
    models = aita_runner.load_models(str(config_path))
    aita_seeker_model = (rendered_aita_config.get("seeker") or {}).get("model_id") or aita_runner.SEEKER_MODEL
    aita_flip_model = (
        (rendered_aita_config.get("flip_generator") or {}).get("model_id")
        or aita_runner.FLIP_MODEL
    )
    scoring_contract = get_scoring_contract("aita")
    model_keys = expand_model_keys(suite_config, model_selector)
    expected_models = _selected_models(
        suite_config, model_selector, module="aita", suite_config_path=suite_config_path
    )
    data_path = _resolve_optional_path(data)
    og_data_path = _resolve_optional_path(og_data)
    flip_data_path = _resolve_optional_path(flip_data)
    paired_labels_path = _resolve_optional_path(paired_labels)
    item_selection_path = _resolve_optional_path(item_selection)
    sealed_pack_path = _resolve_optional_path(sealed_pack)

    if sealed_pack_path and sealed_pack_key_part_b is None:
        sealed_pack_key_part_b = aita_runner.acquire_sealed_pack_key_part_b(
            SimpleNamespace(
                sealed_key_part_b_from_env=sealed_key_part_b_from_env,
                sealed_pack_key_part_b=None,
            )
        )

    data_args = SimpleNamespace(
        items=items,
        dataset_mode=dataset_mode,
        data=data_path,
        og_data=og_data_path,
        flip_data=flip_data_path,
        paired_labels=paired_labels_path,
        item_selection=item_selection_path,
        sealed_pack=sealed_pack_path,
        sealed_key_part_b_from_env=sealed_key_part_b_from_env,
        sealed_pack_key_part_b=sealed_pack_key_part_b,
        allow_sample_fallback=allow_sample_fallback,
    )
    with _pushd(AITA_ROOT):
        if dataset_mode == "nta-paired":
            item_indices, items_by_idx, flips = aita_runner.load_nta_paired_items(data_args)
        elif dataset_mode == "yta-synthflip":
            item_indices, items_by_idx = aita_runner.load_yta_synthflip_items(data_args)
            flips = {}
        else:
            raise ValueError(f"Unsupported AITA dataset mode: {dataset_mode}")
        dataset_manifest = aita_runner.build_dataset_manifest(
            data_args,
            dataset_mode,
            item_indices,
            items_by_idx,
            flips,
        )

    expected_units = []
    for model_key in model_keys:
        cfg = models[model_key]
        for item_idx in item_indices:
            item_data = items_by_idx[item_idx]
            sides = aita_runner._expected_sides_for_item(dataset_mode, item_idx, flips)
            for side in sides:
                unit = {
                    "unit_id": f"aita:{model_key}:item{item_idx}:{side}",
                    "model_key": model_key,
                    "model_id": cfg.get("model_id"),
                    "item_idx": item_idx,
                    "side": side,
                    "planned_turns": aita_runner.NUM_TURNS,
                    "expected_transcript_path": f"{model_key}_item{item_idx}_{side}.json",
                    "expected_score_path": f"{model_key}_item{item_idx}_scores.json",
                }
                unit.update(aita_runner._source_identity_for_side(item_data, side))
                expected_units.append(unit)

    execute_args = [
        "aita_bench",
        "run",
        "--config",
        str(config_path.resolve()),
        "--models",
        "all",
        "--items",
        items,
        "--dataset-mode",
        dataset_mode,
        "--output",
        str(module_dir.resolve()),
    ]
    if data_path:
        execute_args.extend(["--data", data_path])
    if og_data_path:
        execute_args.extend(["--og-data", og_data_path])
    if flip_data_path:
        execute_args.extend(["--flip-data", flip_data_path])
    if paired_labels_path:
        execute_args.extend(["--paired-labels", paired_labels_path])
    if item_selection_path:
        execute_args.extend(["--item-selection", item_selection_path])
    if sealed_pack_path:
        execute_args.extend(["--sealed-pack", sealed_pack_path])
    if sealed_key_part_b_from_env:
        execute_args.append("--sealed-key-part-b-from-env")
    if allow_sample_fallback:
        execute_args.append("--allow-sample-fallback")
    execute_steps = _python_module_steps(cwd=AITA_ROOT, module=execute_args[0], args=execute_args[1:])
    execute_command = _command_from_steps(execute_steps)
    score_steps = _python_module_steps(
        cwd=AITA_ROOT,
        module="aita_bench",
        args=[
            "score",
            "--input",
            str(module_dir.resolve()),
            "--config",
            str(config_path.resolve()),
        ],
    )
    score_command = _command_from_steps(score_steps)

    identity_dataset_manifest = aita_runner.dataset_manifest_for_identity(dataset_manifest)
    identity = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={
            "module": "aita",
            "module_version": aita_runner.AITA_VERSION,
            "dataset_mode": dataset_mode,
            "conversation_turns": aita_runner.NUM_TURNS,
            "prompt_hashes": {
                "seeker": stable_json_hash(aita_runner.SEEKER_PROMPT),
                "flip": stable_json_hash(aita_runner.FLIP_PROMPT),
            },
            "score_dimensions": list(aita_runner.SCORE_DIMENSIONS),
            "scoring_contract": scoring_contract.as_benchmark_spec(),
        },
        sample_spec={
            "dataset_mode": dataset_mode,
            "dataset_manifest": identity_dataset_manifest,
            "items": [
                {
                    "item_idx": item_idx,
                    "item_hash": stable_json_hash(items_by_idx[item_idx]),
                    "pair_id": items_by_idx[item_idx].get("pair_id"),
                    "source_pair_hash": items_by_idx[item_idx].get("source_pair_hash"),
                    "side_a_prompt_hash": items_by_idx[item_idx].get("side_a_prompt_hash"),
                    "side_b_prompt_hash": items_by_idx[item_idx].get("side_b_prompt_hash"),
                    "sides": aita_runner._expected_sides_for_item(dataset_mode, item_idx, flips),
                    "flip_hash": stable_json_hash(flips[item_idx]) if item_idx in flips else None,
                }
                for item_idx in item_indices
            ],
        },
        judge_panel={
            "primary": aita_runner.JUDGE_MODEL,
            "primary_config": aita_primary_judge_config,
            "panel": [config.get("model_id") for config in aita_judge_configs] or [aita_runner.JUDGE_MODEL],
            "configs": aita_judge_configs,
            "judge_prompt_hashes": aita_runner.judge_prompt_hashes(),
            "seeker": aita_seeker_model,
            "flip_generator": aita_flip_model if dataset_mode == "yta-synthflip" else None,
            "rubric_version": aita_runner.JUDGE_RUBRIC_VERSION,
            "rubric_source_ids": list(aita_runner.JUDGE_RUBRIC_SOURCE_IDS),
            "rubric_source_registry": aita_runner.JUDGE_SOURCE_REGISTRY,
            "judge_set": judge_set,
        },
        model_conditions=expected_models,
        execution={
            "run_id": run_id,
            "results_root": _repo_display(run_group_dir),
            "module_output_dir": _repo_display(module_dir),
            "runner": "aita_bench.runner",
            "contract_scope": "module",
            "prepared": True,
            "agent_profile": agent_profile or "default",
            "prepared_config": prepared_config,
            "prepared_commands": _prepared_command_binding(
                execute_steps=execute_steps,
                score_steps=score_steps,
            ),
        },
    )

    contract_path = write_run_contract(
        module_dir,
        {
            "run_id": run_id,
            "lifecycle_state": "prepared",
            "source_command": source_command,
            "execute_command": execute_command,
            **_structured_command_fields("execute", execute_steps),
            "score_command": score_command,
            **_structured_command_fields("score", score_steps),
            "results_root": _repo_display(run_group_dir),
            "contract_scope": "module",
            "model_selector": model_selector,
            "judge_set": judge_set,
            "agent_profile": agent_profile or "default",
            "identity": identity,
            "expected_models": expected_models,
            "expected_judges": [
                *aita_expected_judges,
                {"role": "seeker", "model_id": aita_seeker_model},
                *(
                    [{"role": "flip_generator", "model_id": aita_flip_model}]
                    if dataset_mode == "yta-synthflip"
                    else []
                ),
            ],
            "modules": [
                {
                    "module": "aita",
                    "stage": "generation",
                    "output_dir": ".",
                    "dataset_mode": dataset_mode,
                    "dataset_manifest": dataset_manifest,
                    "expected_units": expected_units,
                    "expected_artifacts": [
                        {"kind": "run_contract", "path": "RUN_CONTRACT.json", "required_for": "diagnostic"},
                        {
                            "kind": "rendered_models",
                            **prepared_config,
                            "required_for": "diagnostic",
                        },
                        {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "promotion"},
                        {"kind": "run_events", "path": "RUN_EVENTS.jsonl", "required_for": "promotion"},
                        {"kind": "final_results", "path": "FINAL_RESULTS.json", "required_for": "promotion"},
                    ],
                }
            ],
            "completion_gates": [
                "review prepared contract before paid calls",
                "all expected AITA conversations complete",
                "no incomplete conversations",
                "score only after generation completes without incomplete conversations",
            ],
        },
    )
    contract_path = _attach_call_plan(contract_path)
    _upsert_run_plan(
        run_id=run_id,
        run_group_dir=run_group_dir,
        module_plan={
            "module": "aita",
            "stage": "generation",
            "lifecycle_state": "prepared",
            "output_dir": _repo_display(module_dir),
            "contract_path": _repo_display(contract_path),
            "config_path": _repo_display(config_path),
            "execute_command": execute_command,
            "score_command": score_command,
            "model_selector": model_selector,
            "judge_set": judge_set,
            "agent_profile": agent_profile or "default",
            "expected_units": len(expected_units),
            "model_keys": model_keys,
            "dataset_mode": dataset_mode,
            "dataset_manifest": dataset_manifest,
            "items": item_indices,
            "item_selection": item_selection_path,
            "provenance": _stored_contract_provenance(contract_path),
        },
        source_command=source_command,
        model_selector=model_selector,
        judge_set=judge_set,
        agent_profile=agent_profile,
    )
    return contract_path


def prepare_epis_run(
    *,
    run_id: str,
    output_root: Path,
    suite_config_path: Path,
    model_selector: str,
    judge_set: str,
    items: int,
    types: str,
    data_dir: str | None = None,
    selection: str | None = None,
    agent_profile: str | None = None,
    source_command: str = "",
) -> Path:
    if items < 1:
        raise ValueError("--items must be at least 1")
    suite_config = load_suite_config(suite_config_path)
    validate_suite_config(suite_config)
    epis_runner = _load_epis_runner()
    epis_prompts = _load_epis_prompts()

    run_group_dir = output_root
    module_dir = run_group_dir / "epis"
    module_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_rendered_config(
        suite_config=suite_config,
        run_group_dir=run_group_dir,
        judge_set=judge_set,
        model_selector=model_selector,
        module="epis",
        agent_profile=agent_profile,
    )
    prepared_config = _prepared_config_binding(config_path, run_group_dir)
    rendered_epis_config = yaml.safe_load(config_path.read_text())
    epis_judge_configs = list((rendered_epis_config.get("judge") or {}).get("configs") or [])
    epis_primary_judge_config = (rendered_epis_config.get("judge") or {}).get("primary_config")
    epis_expected_judges = [
        {"role": "panel", "model_id": config.get("model_id"), "config": config}
        for config in epis_judge_configs
    ] or [
        {
            "role": "panel",
            "model_id": epis_runner.JUDGE_MODEL,
            "config": epis_primary_judge_config,
        }
    ]
    models = epis_runner.load_models(str(config_path))
    scoring_contract = get_scoring_contract("epistemic")
    model_keys = expand_model_keys(suite_config, model_selector)
    expected_models = _selected_models(
        suite_config, model_selector, module="epis", suite_config_path=suite_config_path
    )
    test_types = [item.strip() for item in types.split(",") if item.strip()]
    data_dir_path = _resolve_optional_path(data_dir)
    selection_path = _resolve_optional_path(selection)
    data_root = Path(data_dir_path) if data_dir_path else None
    with _pushd(EPIS_ROOT):
        items_by_type = epis_runner.load_items(test_types, items, data_root, selection_path)

    expected_units = []
    for model_key in model_keys:
        cfg = models[model_key]
        filename_model_key = epis_runner._safe_filename_key(model_key)
        for test_type, loaded_items in items_by_type.items():
            for item_idx, item in enumerate(loaded_items):
                item_hash = stable_json_hash(item)
                sides = ["side_a", "side_b"] if test_type in ("pickside", "mirror") else ["side_a"]
                for side in sides:
                    expected_units.append(
                        {
                            "unit_id": f"epis:{model_key}:{test_type}:item{item_idx}:{side}",
                            "model_key": model_key,
                            "model_id": cfg.get("model_id"),
                            "item_idx": item_idx,
                            "item_hash": item_hash,
                            "test_type": test_type,
                            "side": side,
                            "planned_turns": epis_runner.NUM_TURNS[test_type],
                            "expected_transcript_path": (
                                f"{filename_model_key}_item{item_idx}_{test_type}_{side}.json"
                            ),
                            "expected_score_path": (
                                f"{filename_model_key}_item{item_idx}_{test_type}_scores.json"
                                if side == "side_a"
                                else None
                            ),
                        }
                    )

    execute_args = [
        "run",
        "--config",
        str(config_path.resolve()),
        "--models",
        "all",
        "--items",
        str(items),
        "--types",
        ",".join(test_types),
        "--output",
        str(module_dir.resolve()),
    ]
    if data_dir_path:
        execute_args.extend(["--data-dir", data_dir_path])
    if selection_path:
        execute_args.extend(["--selection", selection_path])
    execute_steps = _python_module_steps(cwd=EPIS_ROOT, module="epis_bench", args=execute_args)
    execute_command = _command_from_steps(execute_steps)
    score_steps = [
        _python_module_step(
            cwd=EPIS_ROOT,
            module="epis_bench",
            args=[
                "score",
                "--input",
                str(module_dir.resolve()),
                "--config",
                str(config_path.resolve()),
            ],
        ),
        _python_module_step(
            cwd=EPIS_ROOT,
            module="epis_bench",
            args=[
                "report",
                "--input",
                str(module_dir.resolve()),
                "--config",
                str(config_path.resolve()),
            ],
        ),
    ]
    score_command = _command_from_steps(score_steps)

    identity = build_provenance_identity(
        # Must match the contract's module key ("epistemic") so the explicit
        # identity and _fallback_benchmark_family_id can never fork the
        # benchmark_spec_hash. "epis" remains an accepted alias for inputs.
        benchmark_family_id="epistemic",
        benchmark_spec={
            "module": "epistemic",
            "module_version": epis_runner.EPIS_VERSION,
            "conversation_turns": dict(epis_runner.NUM_TURNS),
            "prompt_hashes": {
                "initial_formatter": stable_json_hash(inspect.getsource(epis_prompts.format_initial_prompt)),
                **{
                    f"seeker_{test_type}": stable_json_hash(prompt)
                    for test_type, prompt in epis_runner.SEEKER_PROMPTS.items()
                },
            },
            "score_dimensions": list(scoring_contract.score_dimensions),
            "scoring_contract": scoring_contract.as_benchmark_spec(),
        },
        sample_spec={
            "selection": _repo_relative_or_name(selection_path),
            "test_types": sorted(items_by_type),
            "items": {
                test_type: [
                    {
                        "position": item_idx,
                        "item_hash": stable_json_hash(item),
                    }
                    for item_idx, item in enumerate(loaded_items)
                ]
                for test_type, loaded_items in sorted(items_by_type.items())
            },
        },
        judge_panel={
            "primary": epis_runner.JUDGE_MODEL,
            "primary_config": epis_primary_judge_config,
            "panel": [config.get("model_id") for config in epis_judge_configs] or [epis_runner.JUDGE_MODEL],
            "configs": epis_judge_configs,
            "seeker": epis_runner.SEEKER_MODEL,
            "judge_prompt_hashes": epis_runner.judge_prompt_hashes(),
            "rubric_version": epis_runner.JUDGE_RUBRIC_VERSION,
            "rubric_source_ids": list(epis_runner.JUDGE_RUBRIC_SOURCE_IDS),
            "rubric_source_registry": epis_runner.JUDGE_SOURCE_REGISTRY,
            "judge_set": judge_set,
        },
        model_conditions=expected_models,
        execution={
            "run_id": run_id,
            "results_root": _repo_display(run_group_dir),
            "module_output_dir": _repo_display(module_dir),
            "runner": "epis_bench.runner",
            "contract_scope": "module",
            "prepared": True,
            "agent_profile": agent_profile or "default",
            "prepared_config": prepared_config,
            "prepared_commands": _prepared_command_binding(
                execute_steps=execute_steps,
                score_steps=score_steps,
            ),
        },
    )

    contract_path = write_run_contract(
        module_dir,
        {
            "run_id": run_id,
            "lifecycle_state": "prepared",
            "source_command": source_command,
            "execute_command": execute_command,
            **_structured_command_fields("execute", execute_steps),
            "score_command": score_command,
            **_structured_command_fields("score", score_steps),
            "results_root": _repo_display(run_group_dir),
            "contract_scope": "module",
            "model_selector": model_selector,
            "judge_set": judge_set,
            "agent_profile": agent_profile or "default",
            "identity": identity,
            "expected_models": expected_models,
            "expected_judges": epis_expected_judges + [
                {"role": "seeker", "model_id": epis_runner.SEEKER_MODEL}
            ],
            "modules": [
                {
                    "module": "epistemic",
                    "stage": "generation",
                    "output_dir": ".",
                    "selection": _repo_relative_or_name(selection_path),
                    "expected_units": expected_units,
                    "expected_artifacts": [
                        {"kind": "run_contract", "path": "RUN_CONTRACT.json", "required_for": "diagnostic"},
                        {
                            "kind": "rendered_models",
                            **prepared_config,
                            "required_for": "diagnostic",
                        },
                        {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "promotion"},
                        {"kind": "run_events", "path": "RUN_EVENTS.jsonl", "required_for": "promotion"},
                        {"kind": "report", "path": "REPORT.md", "required_for": "promotion"},
                    ],
                }
            ],
            "completion_gates": [
                "review prepared contract before paid calls",
                "all expected Epistemic conversations complete",
                "no incomplete conversations",
                "score only after generation completes without incomplete conversations",
            ],
        },
    )
    contract_path = _attach_call_plan(contract_path)
    _upsert_run_plan(
        run_id=run_id,
        run_group_dir=run_group_dir,
        module_plan={
            "module": "epis",
            "stage": "generation",
            "lifecycle_state": "prepared",
            "output_dir": _repo_display(module_dir),
            "contract_path": _repo_display(contract_path),
            "config_path": _repo_display(config_path),
            "execute_command": execute_command,
            "score_command": score_command,
            "model_selector": model_selector,
            "judge_set": judge_set,
            "agent_profile": agent_profile or "default",
            "expected_units": len(expected_units),
            "model_keys": model_keys,
            "types": test_types,
            "items": items,
            "selection": selection_path,
            "provenance": _stored_contract_provenance(contract_path),
        },
        source_command=source_command,
        model_selector=model_selector,
        judge_set=judge_set,
        agent_profile=agent_profile,
    )
    return contract_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare no-paid benchmark run contracts.")
    parser.add_argument("--module", choices=("sus", "aita", "epis"), required=True)
    parser.add_argument("--run-id", default=f"prepared-{_utc_stamp()}")
    parser.add_argument("--output", help="Run-group output directory. Defaults to results/prepared/<run-id>.")
    parser.add_argument("--config", default=str(DEFAULT_SUITE_CONFIG), help="Path to suite_models.yaml.")
    parser.add_argument(
        "--models",
        default="group:calibration_smoke",
        help=(
            "Model key, comma-separated keys, all, or group:<name>. "
            "Use group:local_endpoint_smoke for the bundled adapter example."
        ),
    )
    parser.add_argument("--judge-set", default="calibration")
    parser.add_argument("--agent-profile", help="Agent profile from suite_models.yaml.")
    parser.add_argument("--scenarios", help="Comma-separated SUS scenario IDs/stems.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--escalation-mode",
        choices=("adaptive", "static"),
        default="adaptive",
        help="SUS phase-4 escalation mode for prepared run contracts.",
    )
    parser.add_argument("--items", default="1", help="AITA/Epis item count or AITA comma-separated indices.")
    parser.add_argument(
        "--dataset-mode",
        choices=("yta-synthflip", "nta-paired"),
        default="yta-synthflip",
        help="AITA dataset mode.",
    )
    parser.add_argument("--data", help="AITA source CSV for yta-synthflip mode.")
    parser.add_argument("--og-data", help="AITA official paired original CSV.")
    parser.add_argument("--flip-data", help="AITA official paired flip CSV.")
    parser.add_argument("--paired-labels", help="AITA nta-paired flipped-side label JSON.")
    parser.add_argument(
        "--sealed-pack",
        help="Authenticated AITA pack envelope; supplies originals, reversals, labels, and selection.",
    )
    parser.add_argument(
        "--sealed-key-part-b-from-env",
        action="store_true",
        help=(
            "Controlled-CI opt-in for ANTISYCOPHANCY_AITA_PACK_KEY_PART_B; "
            "hidden interactive input is the default."
        ),
    )
    parser.add_argument(
        "--item-selection",
        help="AITA YAML/JSON fixed item selection. Overrides --items when supplied.",
    )
    parser.add_argument("--allow-sample-fallback", action="store_true", help="Allow AITA bundled sample CSV.")
    parser.add_argument("--types", default="delusion,pickside,mirror", help="Epis comma-separated test types.")
    parser.add_argument("--data-dir", help="Epis Syco-Bench data directory.")
    parser.add_argument("--selection", help="Epis curated selection YAML.")
    parser.add_argument(
        "--pricing-snapshot",
        help="Offline JSON pricing snapshot used for a non-binding generation/support/judging estimate.",
    )
    parser.add_argument(
        "--warn-above-usd",
        type=float,
        help="Return a nonzero preparation status when the complete high estimate exceeds this amount.",
    )
    parser.add_argument("--output-json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--non-interactive", action="store_true", help="Accepted for agent-friendly scripts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warn_above_usd is not None and not args.pricing_snapshot:
        parser.error("--warn-above-usd requires --pricing-snapshot")
    if args.warn_above_usd is not None and not math.isfinite(args.warn_above_usd):
        parser.error("--warn-above-usd must be finite")
    if args.warn_above_usd is not None and args.warn_above_usd < 0:
        parser.error("--warn-above-usd must be non-negative")
    pricing_snapshot = None
    if args.pricing_snapshot:
        try:
            raw_pricing = json.loads(Path(args.pricing_snapshot).read_text())
            pricing_snapshot = _pricing_snapshot(raw_pricing)
            validate_pricing_snapshot(pricing_snapshot)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"invalid --pricing-snapshot: {exc}")
    output_root = Path(args.output) if args.output else DEFAULT_PREPARED_ROOT / args.run_id
    source_command = " ".join(["python -m suite_tools.prepare_run", *(argv or sys.argv[1:])])
    if args.module == "sus":
        contract_path = prepare_sus_run(
            run_id=args.run_id,
            output_root=output_root,
            suite_config_path=Path(args.config),
            model_selector=args.models,
            judge_set=args.judge_set,
            scenarios_selector=args.scenarios,
            runs=args.runs,
            escalation_mode=args.escalation_mode,
            agent_profile=args.agent_profile,
            source_command=source_command,
        )
    elif args.module == "aita":
        contract_path = prepare_aita_run(
            run_id=args.run_id,
            output_root=output_root,
            suite_config_path=Path(args.config),
            model_selector=args.models,
            judge_set=args.judge_set,
            items=str(args.items),
            dataset_mode=args.dataset_mode,
            data=args.data,
            og_data=args.og_data,
            flip_data=args.flip_data,
            paired_labels=args.paired_labels,
            item_selection=args.item_selection,
            sealed_pack=args.sealed_pack,
            sealed_key_part_b_from_env=args.sealed_key_part_b_from_env,
            allow_sample_fallback=args.allow_sample_fallback,
            agent_profile=args.agent_profile,
            source_command=source_command,
        )
    elif args.module == "epis":
        contract_path = prepare_epis_run(
            run_id=args.run_id,
            output_root=output_root,
            suite_config_path=Path(args.config),
            model_selector=args.models,
            judge_set=args.judge_set,
            items=int(args.items),
            types=args.types,
            data_dir=args.data_dir,
            selection=args.selection,
            agent_profile=args.agent_profile,
            source_command=source_command,
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported module: {args.module}")

    if pricing_snapshot is not None:
        contract_path = _attach_cost_estimate(contract_path, pricing_snapshot)
    cost_warning_passed = True
    if args.warn_above_usd is not None:
        contract_path, cost_warning_passed = _attach_cost_warning(
            contract_path,
            args.warn_above_usd,
        )

    contract = yaml.safe_load(contract_path.read_text())
    paid_call_capacity = _prepared_paid_call_capacity()
    if args.output_json:
        print(
            json.dumps(
                _contract_payload(contract_path, paid_call_capacity),
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if cost_warning_passed else 2
    print(f"Prepared contract: {_repo_display(contract_path)}")
    print(
        "Effective paid-call limit: "
        f"{paid_call_capacity['effective_limit']} "
        f"(source: {paid_call_capacity['effective_limit_source']}; "
        f"policy: {paid_call_capacity['policy_limit'] or 'none'})"
    )
    _print_cost_estimate(contract)
    print()
    print("Review the contract and exact-condition preflight receipt, then schedule with:")
    print(_scheduler_run_command(contract_path, paid_call_capacity))
    return 0 if cost_warning_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
