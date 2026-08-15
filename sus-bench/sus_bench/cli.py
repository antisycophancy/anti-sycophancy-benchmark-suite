"""CLI for sus-bench.

Three subcommands:
  run    — Execute the benchmark (default)
  rescore — Re-score saved conversations with a new judge panel
  report — Generate reports from existing JSON results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from sus_bench import __version__
from sus_bench.scoring_contract import is_score_excluded_result
from suite_tools.artifact_identity import (
    reconcile_condition_identity,
    require_run_artifact_identity,
)
from suite_tools.paid_call_lease import effective_paid_call_parallelism
from suite_tools.preflight_conditions import (
    PreflightReceiptValidationError,
    validate_preflight_receipt_for_prepared_config,
)
from suite_tools.run_contract import (
    JudgeProvenanceError,
    PreparedConfigProvenanceError,
    file_sha256,
    load_run_contract,
    stable_json_hash,
    validate_run_judge_provenance_before_spend,
    validate_run_prepared_config_before_spend,
    write_runtime_run_contract,
)


def _configured_score_parallelism(score_parallelism=None) -> int:
    raw = (
        score_parallelism
        if score_parallelism is not None
        else os.environ.get("BENCHMARK_SUS_SCORE_MAX_PARALLEL")
        or os.environ.get("BENCHMARK_SCORE_MAX_PARALLEL")
        or "2"
    )
    try:
        requested = max(1, int(raw))
    except (TypeError, ValueError):
        requested = 2
    return effective_paid_call_parallelism(requested)


def _validate_rescore_condition_identity(
    results: list[dict],
    models_config: dict,
) -> None:
    """Fail before judge calls when saved rows do not match rendered conditions."""
    conditions = [
        condition
        for condition in models_config.get("models") or []
        if isinstance(condition, dict)
    ]
    if not conditions:
        return
    for index, result in enumerate(results, start=1):
        condition_id = result.get("condition_id")
        if condition_id:
            candidates = [
                condition
                for condition in conditions
                if condition.get("condition_id") == condition_id
            ]
        else:
            observed = {
                str(value)
                for value in (
                    result.get("model"),
                    result.get("model_id"),
                    result.get("model_key"),
                    result.get("label"),
                )
                if value is not None
            }
            candidates = [
                condition
                for condition in conditions
                if observed.intersection(
                    str(value)
                    for value in (
                        condition.get("id"),
                        condition.get("model_id"),
                        condition.get("key"),
                        condition.get("label"),
                    )
                    if value is not None
                )
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"SUS score input row {index} resolves to {len(candidates)} "
                "rendered model conditions; refusing judge calls"
            )
        reconcile_condition_identity(
            result,
            candidates[0],
            context=f"SUS score input row {index}",
            restore_missing=False,
        )


def _score_input_key(result: dict) -> tuple[str, str, int]:
    return (
        str(result.get("condition_id") or ""),
        str(result.get("scenario") or ""),
        int(result.get("run_number") or 0),
    )


def _hydrate_score_input_identity_from_transcripts(
    results: list[dict],
    input_dir: Path,
) -> dict[str, object]:
    """Restore aggregate-row identity only from matching saved transcripts.

    Older SUS conversation sidecars omitted served/provider identity even though
    the individual immutable transcript artifacts retained it. Scoring may copy
    those fields in memory only after the scoring-relevant conversation payload
    matches exactly. Source artifacts are never rewritten.
    """
    contract_path = input_dir / "RUN_CONTRACT.json"
    if not contract_path.is_file():
        return {"restored_rows": 0, "restored_fields": {}}

    contract = json.loads(contract_path.read_text())
    transcript_by_key: dict[tuple[str, str, int], tuple[dict, Path]] = {}
    for module in contract.get("modules") or []:
        if not isinstance(module, dict) or module.get("module") != "sus":
            continue
        for unit in module.get("expected_units") or []:
            if not isinstance(unit, dict):
                continue
            relative_path = unit.get("expected_transcript_path")
            if not relative_path:
                continue
            transcript_path = input_dir / str(relative_path)
            if not transcript_path.is_file():
                continue
            transcript = json.loads(transcript_path.read_text())
            if not isinstance(transcript, dict):
                raise ValueError(f"SUS transcript is not an object: {relative_path}")
            key = _score_input_key(transcript)
            if not key[0]:
                raise ValueError(f"SUS transcript lacks condition_id: {relative_path}")
            if key in transcript_by_key:
                raise ValueError(f"Duplicate SUS transcript identity key: {key}")
            transcript_by_key[key] = (transcript, transcript_path)

    restored_rows = 0
    restored_field_counts: dict[str, int] = {}
    for index, result in enumerate(results, start=1):
        key = _score_input_key(result)
        if not key[0] or key not in transcript_by_key:
            continue
        transcript, transcript_path = transcript_by_key[key]
        for field in ("conversation", "phases", "turn_outcomes"):
            if result.get(field) != transcript.get(field):
                raise ValueError(
                    f"SUS score input row {index} disagrees with saved transcript "
                    f"for {field}: {transcript_path}"
                )
        restored = reconcile_condition_identity(
            result,
            transcript,
            context=f"SUS score input row {index} vs saved transcript",
            restore_missing=True,
        )
        if not restored:
            continue
        restored_rows += 1
        for field in restored:
            restored_field_counts[field] = restored_field_counts.get(field, 0) + 1
        result["identity_normalization"] = {
            "method": "restored_from_saved_transcript_artifact",
            "restored_fields": list(restored),
            "source_path": str(transcript_path.relative_to(input_dir)),
            "source_sha256": file_sha256(transcript_path),
        }
    return {
        "restored_rows": restored_rows,
        "restored_fields": restored_field_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with run/report subcommands."""
    parser = argparse.ArgumentParser(
        prog="sus_bench",
        description=(
            "Susceptibility to Unsafe Suggestion Bench (sus-bench) v{version} — "
            "Do AI safety refusals preserve their boundary, or capitulate "
            "when the same request is reframed?"
        ).format(version=__version__),
    )
    parser.add_argument(
        "--version", action="version", version=f"sus-bench {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run", help="Run the benchmark against one or more models",
    )
    run_parser.add_argument(
        "--model",
        help="Test a specific model (OpenRouter ID, e.g. anthropic/claude-sonnet-4.6)",
    )
    run_parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per model+scenario (default: 3, use 1 for quick exploration)",
    )
    run_parser.add_argument(
        "--analyzer-model",
        help="Override the analyzer/judge model (default: from models.yaml)",
    )
    run_parser.add_argument(
        "--scenarios",
        help="Comma-separated scenario IDs or YAML filenames (without .yaml) to run (default: all)",
    )
    run_parser.add_argument(
        "--models",
        help="Path to models.yaml config (default: models.yaml in package root)",
    )
    run_parser.add_argument(
        "--output", "-o",
        help="Output directory for results (default: results/)",
    )
    run_parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between API calls (default: 1.0)",
    )
    run_parser.add_argument(
        "--html", action="store_true",
        help="Generate HTML dashboard after run",
    )
    run_parser.add_argument(
        "--temperature", type=float, nargs="+",
        help="Temperature(s) to test (e.g. --temperature 0.0 0.7 1.0). "
             "Runs the full benchmark at each temperature. Default: provider default.",
    )
    run_parser.add_argument(
        "--no-parallel", action="store_true",
        help="Run models sequentially instead of in parallel (default: parallel)",
    )
    run_parser.add_argument(
        "--reasoning", nargs="+",
        help="Reasoning effort levels to sweep (e.g. --reasoning none low medium high). "
             "Maps to OpenRouter reasoning.effort / Gemini thinking levels.",
    )
    run_parser.add_argument(
        "--escalation-mode",
        choices=("adaptive", "static"),
        default="adaptive",
        help=(
            "How SUS phase-4 escalation prompts are produced. "
            "'adaptive' generates context-aware user turns from the scenario strategy; "
            "'static' replays the scenario YAML prompts."
        ),
    )
    run_parser.add_argument(
        "--score-inline",
        action="store_true",
        help=(
            "Legacy/scratch mode: run the blinded judge panel during generation. "
            "Default is generation-only, then use the score command."
        ),
    )

    # --- score/rescore subcommands ---
    score_parser = subparsers.add_parser(
        "score", help="Score a completed SUS generation directory with the judge panel",
    )
    score_parser.add_argument("--input", required=True, help="Generation output directory or conversations JSON")
    score_parser.add_argument("--models", help="Path to models.yaml config with analyzer/judge_panel settings")
    score_parser.add_argument("--analyzer-model", help="Override analyzer model used as fallback/context")
    score_parser.add_argument("--judge-panel", help="Comma-separated judge model IDs. Overrides models.yaml.")
    score_parser.add_argument("--output", "-o", help="Output summary JSON path. Defaults to <input>/FINAL_RESULTS.json for directories.")
    score_parser.add_argument("--score-parallelism", type=int, help="Maximum conversation rows to score in parallel")

    rescore_parser = subparsers.add_parser(
        "rescore", help="Re-score saved conversations with the current or specified judge panel",
    )
    rescore_parser.add_argument(
        "files", nargs="+",
        help="Conversation JSON files or summary JSON files with sibling *-conversations.json files",
    )
    rescore_parser.add_argument(
        "--models",
        help="Path to models.yaml config with analyzer/judge_panel settings",
    )
    rescore_parser.add_argument(
        "--analyzer-model",
        help="Override analyzer model used as fallback/context for SUS scoring",
    )
    rescore_parser.add_argument(
        "--judge-panel",
        help="Comma-separated judge model IDs. Overrides judge_panel in models.yaml.",
    )
    rescore_parser.add_argument(
        "--output", "-o",
        help="Output summary JSON path. Defaults to results/sus-rescore-<timestamp>.json",
    )
    rescore_parser.add_argument(
        "--score-parallelism",
        type=int,
        help="Maximum conversation rows to score in parallel",
    )

    report_parser = subparsers.add_parser(
        "report", help="Generate reports from existing JSON result files",
    )
    report_parser.add_argument(
        "files", nargs="+",
        help="JSON result files to process",
    )
    report_parser.add_argument(
        "--html",
        help="Output path for HTML dashboard",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to 'run' if no subcommand given
    if args.command is None:
        args.command = "run"
        # Re-parse with run defaults
        args = parser.parse_args(["run"] + (argv or sys.argv[1:]))

    if args.command == "run":
        _cmd_run(args)
    elif args.command in {"score", "rescore"}:
        _cmd_rescore(args)
    elif args.command == "report":
        _cmd_report(args)


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the benchmark run command."""
    from datetime import datetime

    from rich.console import Console

    from sus_bench.report import print_table, write_html, write_json
    from sus_bench.api import get_cost_tracker, reset_cost_tracker, CreditExhaustedError
    from sus_bench.analyzer import ADAPTIVE_ESCALATION_PROMPT, EXTRACT_PROMPT, FOLLOW_PROMPT
    from sus_bench.runner import BenchmarkRunError, load_models_config, load_scenario, run_benchmark
    from sus_bench.scorer import (
        DEFAULT_JUDGE_PANEL,
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
    from sus_bench.stats import aggregate_runs
    from suite_tools.run_monitor import RunMonitor
    from suite_tools.run_contract import (
        PreparedConfigProvenanceError,
        RunControlStopRequested,
        build_provenance_identity,
        stable_json_hash,
        validate_run_prepared_config_before_spend,
    )
    from suite_tools.model_config import MODEL_CONDITION_METADATA_FIELDS
    from suite_tools.env import load_repo_env_files
    from suite_tools.scoring_contracts import get_scoring_contract

    console = Console()
    scoring_contract = get_scoring_contract("sus")

    # Find package root (where models.yaml and scenarios/ live)
    pkg_root = _find_package_root()

    output_dir = Path(args.output) if args.output else pkg_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_path = Path(args.models) if args.models else pkg_root / "models.yaml"
    try:
        prepared_config_receipt = validate_run_prepared_config_before_spend(
            output_dir,
            models_path,
        )
        if prepared_config_receipt:
            prepared_contract = load_run_contract(output_dir)
            prepared_module = next(
                (
                    module
                    for module in prepared_contract.get("modules") or []
                    if isinstance(module, dict) and module.get("module") == "sus"
                ),
                {},
            )
            requested_scenarios = [
                item.strip()
                for item in str(getattr(args, "scenarios", "") or "").split(",")
                if item.strip()
            ]
            overrides = [
                name
                for name, active in (
                    ("--model", bool(getattr(args, "model", None))),
                    ("--analyzer-model", bool(getattr(args, "analyzer_model", None))),
                    ("--temperature", bool(getattr(args, "temperature", None))),
                    ("--reasoning", bool(getattr(args, "reasoning", None))),
                    ("--score-inline", bool(getattr(args, "score_inline", False))),
                    (
                        "--runs",
                        getattr(args, "runs", None) != prepared_module.get("runs"),
                    ),
                    (
                        "--scenarios",
                        requested_scenarios != list(prepared_module.get("scenarios") or []),
                    ),
                    (
                        "--escalation-mode",
                        getattr(args, "escalation_mode", "adaptive")
                        != prepared_module.get("escalation_mode"),
                    ),
                )
                if active
            ]
            if overrides:
                raise PreparedConfigProvenanceError(
                    "prepared run arguments differ from the frozen sample/model identity: "
                    + ", ".join(overrides)
                )
    except PreparedConfigProvenanceError as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        console.print("Refusing to spend with a changed prepared model config; prepare a new run.")
        RunMonitor(output_dir, module="sus", stage="generation").mark_failed(
            exc,
            status="failed_invalid",
            failure_stage="prepared_config_provenance",
            provenance_issues=list(exc.issues),
        )
        sys.exit(2)
    # Load models config
    if not models_path.exists():
        console.print(f"[red]ERROR:[/red] Models config not found: {models_path}")
        sys.exit(1)
    models_config = load_models_config(models_path)

    # Load scenarios
    scenarios_dir = pkg_root / "scenarios"
    if not scenarios_dir.exists():
        console.print(f"[red]ERROR:[/red] Scenarios directory not found: {scenarios_dir}")
        sys.exit(1)

    scenario_filter = None
    if args.scenarios:
        scenario_filter = [s.strip() for s in args.scenarios.split(",")]

    scenarios = []
    for yaml_file in sorted(scenarios_dir.glob("*.yaml")):
        scenario = load_scenario(yaml_file)
        scenario["_filename_stem"] = yaml_file.stem
        if scenario_filter is None or scenario["id"] in scenario_filter or yaml_file.stem in scenario_filter:
            scenarios.append(scenario)

    if not scenarios:
        console.print("[red]ERROR:[/red] No scenarios found or matched filter.")
        sys.exit(1)

    if prepared_config_receipt:
        frozen_identity = prepared_contract.get("identity") or {}
        frozen_sample = frozen_identity.get("sample_spec") or {}
        frozen_benchmark = frozen_identity.get("benchmark_spec") or {}
        actual_scenario_hashes = {
            scenario["id"]: stable_json_hash({
                key: value
                for key, value in scenario.items()
                if not str(key).startswith("_")
            })
            for scenario in scenarios
        }
        actual_prompt_hashes = {
            "extract": stable_json_hash(EXTRACT_PROMPT),
            "follow": stable_json_hash(FOLLOW_PROMPT),
            "adaptive_escalation": stable_json_hash(ADAPTIVE_ESCALATION_PROMPT),
            "post_analysis": stable_json_hash(POST_ANALYSIS_PROMPT),
        }
        if (
            actual_scenario_hashes != frozen_sample.get("scenario_hashes")
            or actual_prompt_hashes != frozen_benchmark.get("phase_prompts")
        ):
            exc = PreparedConfigProvenanceError(
                "prepared SUS scenario or instrument prompts differ from the frozen contract"
            )
            console.print(f"[red]ERROR:[/red] {exc}")
            console.print("Refusing to spend on a changed benchmark instrument; prepare a new run.")
            RunMonitor(output_dir, module="sus", stage="generation").mark_failed(
                exc,
                status="failed_invalid",
                failure_stage="prepared_config_provenance",
                provenance_issues=list(exc.issues),
            )
            sys.exit(2)

    try:
        preflight_admission = validate_preflight_receipt_for_prepared_config(
            output_dir,
            prepared_config_receipt,
        )
    except PreflightReceiptValidationError as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        console.print(
            "Refusing prepared generation without current exact-condition "
            "preflight evidence."
        )
        RunMonitor(output_dir, module="sus", stage="generation").mark_failed(
            exc,
            status="failed_invalid",
            failure_stage="preflight_receipt_admission",
            provenance_issues=list(exc.issues),
        )
        sys.exit(2)

    # Load environment only after the prepared config, sample, and instrument
    # have authenticated successfully.
    load_repo_env_files()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print(
            "[red]ERROR:[/red] OPENROUTER_API_KEY not set. "
            "Create a .env file or export the variable.",
        )
        sys.exit(1)

    temps = getattr(args, "temperature", None)
    reasoning = getattr(args, "reasoning", None)
    escalation_mode = getattr(args, "escalation_mode", "adaptive")
    console.print(f"[bold]sus-bench v{__version__}[/bold]")
    console.print(f"Models: {len(models_config['models']) if not args.model else 1}")
    console.print(f"Scenarios: {len(scenarios)}")
    console.print(f"Runs: {args.runs}")
    console.print(f"Escalation mode: {escalation_mode}")
    if temps:
        console.print(f"Temperatures: {', '.join(str(t) for t in temps)}")
    if reasoning:
        console.print(f"Reasoning efforts: {', '.join(reasoning)}")
    console.print()

    # Reset cost tracker, set API key for credit monitoring
    reset_cost_tracker()
    tracker = get_cost_tracker()
    tracker.set_api_key(api_key)

    # Check credit balance before starting
    remaining = tracker.check_credit_now()
    if remaining is not None:
        console.print(f"Credit remaining: [bold]${remaining:.2f}[/bold]")
    console.print()

    run_id = f"sus-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    score_inline = bool(getattr(args, "score_inline", False))
    run_stage = "run" if score_inline else "generation"
    monitor = RunMonitor(
        output_dir,
        module="sus",
        stage=run_stage,
        metadata={
            "run_id": run_id,
            "models": [m.get("id") for m in models_config.get("models", [])] if not args.model else [args.model],
            "scenarios": [s["id"] for s in scenarios],
            "runs": args.runs,
            "escalation_mode": escalation_mode,
            "score_inline": score_inline,
            "analyzer_model": args.analyzer_model or models_config.get("analyzer"),
            "judge_panel": models_config.get("judge_panel"),
        },
    )
    if prepared_config_receipt:
        monitor.record("prepared_config_verified", **prepared_config_receipt)
    if preflight_admission:
        monitor.record("preflight_receipt_admitted", **preflight_admission)
    contract_models = models_config.get("models", [])
    if args.model:
        matched = [model for model in contract_models if model.get("id") == args.model]
        contract_models = matched or [{"id": args.model, "label": args.model}]
    expected_models = [
        {
            "key": model.get("label") or model.get("id"),
            "label": model.get("label") or model.get("id"),
            "model_id": model.get("id"),
            "endpoint": model.get("base_url", "openrouter"),
            "source": str(models_path),
            **{
                field: model[field]
                for field in MODEL_CONDITION_METADATA_FIELDS
                if field in model
            },
        }
        for model in contract_models
    ]
    effective_analyzer_model = args.analyzer_model or models_config.get("analyzer")
    effective_judge_panel = models_config.get("judge_panel") or DEFAULT_JUDGE_PANEL
    contract_units = []
    for model in contract_models:
        model_condition_key = model.get("label") or model.get("id")
        model_condition_hash = stable_json_hash(
            {
                "model_id": model.get("id"),
                "label": model.get("label"),
                "endpoint": model.get("base_url"),
                "provider_api": model.get("provider_api"),
                "request_options": model.get("request_options"),
                **{
                    field: model[field]
                    for field in MODEL_CONDITION_METADATA_FIELDS
                    if field in model
                },
            }
        )[:12]
        for scenario in scenarios:
            for run_number in range(1, args.runs + 1):
                contract_units.append({
                    "unit_id": f"sus:{model_condition_hash}:{scenario['id']}:run{run_number}",
                    "model_key": model_condition_key,
                    "model_id": model.get("id"),
                    "model_condition_hash": model_condition_hash,
                    "scenario": scenario["id"],
                    "run_number": run_number,
                    "expected_summary_path": f"{run_id}.json",
                })
    identity = build_provenance_identity(
        benchmark_family_id="sus",
        benchmark_spec={
            "module": "sus",
            "module_version": __version__,
            "escalation_mode": escalation_mode,
            "phase_prompts": {
                "extract": stable_json_hash(EXTRACT_PROMPT),
                "follow": stable_json_hash(FOLLOW_PROMPT),
                "adaptive_escalation": stable_json_hash(ADAPTIVE_ESCALATION_PROMPT),
                "post_analysis": stable_json_hash(POST_ANALYSIS_PROMPT),
            },
            "score_dimensions": list(SUS_RELEASE_SCORE_DIMENSIONS),
            "scoring_contract": scoring_contract.as_benchmark_spec(),
            "score_component_weights": dict(SUS_RESPONSE_COMPONENT_WEIGHTS),
            "legacy_diagnostic_dimensions": list(SUS_LEGACY_DIAGNOSTIC_DIMENSIONS),
            "legacy_score_weights": dict(DEFAULT_WEIGHTS),
        },
        sample_spec={
            "scenario_ids": [scenario["id"] for scenario in scenarios],
            "scenario_hashes": {
                scenario["id"]: stable_json_hash(
                    {key: value for key, value in scenario.items() if not str(key).startswith("_")}
                )
                for scenario in scenarios
            },
            "runs": args.runs,
        },
        judge_panel={
            "analyzer": effective_analyzer_model,
            "panel": list(effective_judge_panel),
            "judge_prompt_hashes": {
                "post_analysis": stable_json_hash(POST_ANALYSIS_PROMPT),
            },
            "rubric_version": RUBRIC_VERSION,
            "rubric_source_ids": list(RUBRIC_SOURCE_IDS),
            "rubric_source_registry": RUBRIC_SOURCE_REGISTRY,
        },
        model_conditions=expected_models,
        execution={
            "run_id": run_id,
            "results_root": str(output_dir),
            "runner": "sus_bench.cli",
            "contract_scope": "module",
            "temperature": temps,
            "reasoning": reasoning,
            "escalation_mode": escalation_mode,
            "score_inline": score_inline,
        },
    )
    score_command = "\n".join(
        [
            f"cd {pkg_root}",
            "../venv/bin/python -m sus_bench score \\",
            f"  --input {output_dir.resolve()} \\",
            f"  --models {models_path.resolve()} \\",
            f"  --output {(output_dir / 'FINAL_RESULTS.json').resolve()}",
        ]
    )
    write_runtime_run_contract(
        output_dir,
        {
            "run_id": run_id,
            "source_command": " ".join(sys.argv),
            "score_command": None if score_inline else score_command,
            "results_root": str(output_dir),
            "contract_scope": "module",
            "identity": identity,
            "expected_models": expected_models,
            "expected_judges": [
                {"role": "analyzer", "model_id": effective_analyzer_model},
                *[
                    {"role": "panel", "model_id": judge}
                    for judge in effective_judge_panel
                ],
            ],
            "modules": [
                {
                    "module": "sus",
                    "stage": run_stage,
                    "output_dir": str(output_dir),
                    "escalation_mode": escalation_mode,
                    "score_inline": score_inline,
                    "expected_units": contract_units,
                    "expected_artifacts": [
                        {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "diagnostic"},
                        {"kind": "run_events", "path": "RUN_EVENTS.jsonl", "required_for": "diagnostic"},
                        {
                            "kind": "summary_json",
                            "path": f"{run_id}.json",
                            "required_for": "promotion" if score_inline else "diagnostic",
                        },
                        {
                            "kind": "conversations_json",
                            "path": f"{run_id}-conversations.json",
                            "required_for": "promotion" if score_inline else "scoring",
                        },
                        *(
                            []
                            if score_inline
                            else [
                                {
                                    "kind": "final_results",
                                    "path": "FINAL_RESULTS.json",
                                    "required_for": "promotion",
                                }
                            ]
                        ),
                    ],
                }
            ],
            "completion_gates": [
                "all expected SUS runs complete",
                "conversation JSON written",
                *(
                    ["status completed and score_ready"]
                    if score_inline
                    else [
                        "generation status completed and not_score_ready",
                        "score only after generation completes cleanly",
                        "scoring status completed and score_ready",
                    ]
                ),
            ],
        },
    )
    run_failed = None
    try:
        results = run_benchmark(
            models_config,
            scenarios,
            api_key,
            model_filter=args.model,
            scenario_filter=scenario_filter,
            runs=args.runs,
            analyzer_model=args.analyzer_model,
            delay=args.delay,
            temperatures=temps,
            reasoning_efforts=reasoning,
            parallel=not getattr(args, "no_parallel", False),
            monitor=monitor,
            control_dir=output_dir,
            escalation_mode=escalation_mode,
            score_inline=score_inline,
        )
    except BenchmarkRunError as e:
        if isinstance(e.__cause__, RunControlStopRequested):
            console.print(f"\n[bold yellow]RUN STOPPED: {e.__cause__}[/bold yellow]")
        else:
            console.print(f"\n[bold red]RUN FAILED: {e}[/bold red]")
        console.print("[yellow]Saving partial results that completed before the stop/failure...[/yellow]")
        results = e.partial_results
        run_failed = e
    except RunControlStopRequested as e:
        console.print(f"\n[bold yellow]RUN STOPPED: {e}[/bold yellow]")
        console.print("[yellow]Saving partial results...[/yellow]")
        results = []
        run_failed = e
    except CreditExhaustedError as e:
        console.print(f"\n[bold red]RUN STOPPED: {e}[/bold red]")
        console.print("[yellow]Saving partial results...[/yellow]")
        results = []
        run_failed = e

    # Aggregate scored results only. Generation-only runs intentionally defer
    # the blinded judge panel to the score command.
    aggregated = aggregate_runs(results) if score_inline else []
    console.print()
    if score_inline:
        print_table(aggregated)
    else:
        console.print("[bold]Generation complete.[/bold] Run the score command after reviewing transcripts.")

    json_path = output_dir / f"{run_id}.json"

    # Include cost summary in results
    cost_summary = get_cost_tracker().summary()
    write_json(results, aggregated, json_path, run_id=run_id, cost=cost_summary)

    console.print(f"\nResults saved: {json_path}")
    console.print(
        f"Cost: ${cost_summary['total_cost_usd']:.4f} "
        f"({cost_summary['total_calls']} calls, "
        f"{cost_summary['tokens_in']+cost_summary['tokens_out']:,} tokens)"
    )
    if cost_summary['cost_by_role']:
        parts = [f"{role}: ${c:.4f}" for role, c in cost_summary['cost_by_role'].items()]
        console.print(f"  Breakdown: {', '.join(parts)}")
    if cost_summary.get('credit_remaining_usd') is not None:
        remaining = cost_summary['credit_remaining_usd']
        color = "green" if remaining > 5 else "yellow" if remaining > 1 else "red"
        console.print(f"  Credit remaining: [{color}]${remaining:.2f}[/{color}]")

    if args.html and score_inline:
        html_path = output_dir / f"{run_id}.html"
        write_html(aggregated, html_path)
        console.print(f"Dashboard saved: {html_path}")

    if run_failed:
        control_cause = run_failed
        if isinstance(run_failed, BenchmarkRunError) and isinstance(run_failed.__cause__, RunControlStopRequested):
            control_cause = run_failed.__cause__
        exit_code = 2
        if isinstance(control_cause, RunControlStopRequested):
            monitor.mark_stopped(
                str(control_cause),
                control=control_cause.summary,
                partial_results=len(results),
                summary_path=str(json_path),
            )
            exit_code = 130
        else:
            monitor.mark_failed(
                run_failed,
                partial_results=len(results),
                summary_path=str(json_path),
            )
        sys.exit(exit_code)
    monitor.mark_completed(
        validity="score_ready" if score_inline else "not_score_ready",
        results=len(results),
        summary_path=str(json_path),
        score_command=None if score_inline else score_command,
    )


def _cmd_report(args: argparse.Namespace) -> None:
    """Generate reports from existing JSON result files."""
    from rich.console import Console

    from sus_bench.report import print_segmented_table, print_table, write_html
    from sus_bench.stats import aggregate_runs

    console = Console()

    all_results = []
    loaded_sources: set[Path] = set()
    for filepath in args.files:
        path = Path(filepath)
        try:
            results, source_path = _load_report_results(path)
        except ValueError as e:
            console.print(f"[red]ERROR:[/red] {e}")
            sys.exit(1)
        resolved_source = source_path.resolve()
        if resolved_source in loaded_sources:
            continue
        loaded_sources.add(resolved_source)
        all_results.extend(results)

    if not all_results:
        console.print("[red]ERROR:[/red] No results found in provided files.")
        sys.exit(1)

    aggregated = aggregate_runs(all_results)
    print_table(aggregated)
    print_segmented_table(all_results)

    if args.html:
        write_html(aggregated, Path(args.html))
        console.print(f"\nDashboard saved: {args.html}")


def _cmd_rescore(args: argparse.Namespace) -> None:
    """Re-score saved SUS conversations without re-running model-under-test calls."""
    from datetime import datetime

    from rich.console import Console

    from sus_bench.api import get_cost_tracker, reset_cost_tracker
    from sus_bench.classifier import classify_result
    from sus_bench.report import print_segmented_table, print_table, write_json
    from sus_bench.runner import load_models_config
    from sus_bench.scorer import (
        DEFAULT_JUDGE_PANEL,
        JudgePanelIncompleteError,
        POST_ANALYSIS_PROMPT,
        RUBRIC_SOURCE_IDS,
        RUBRIC_SOURCE_REGISTRY,
        RUBRIC_VERSION,
        compute_sus_score,
        run_post_analysis,
    )
    from sus_bench.stats import aggregate_runs
    from suite_tools.conversation_hygiene import blocking_issue_summaries
    from suite_tools.env import load_repo_env_files
    from suite_tools.request_receipts import (
        RequestConformanceError,
        require_request_conformance,
    )
    from suite_tools.run_monitor import RunMonitor, atomic_write_json, sanitize_error_message

    console = Console()
    pkg_root = _find_package_root()

    input_arg = getattr(args, "input", None)
    input_dir: Path | None = None
    input_files: list[Path]
    if input_arg:
        input_path = Path(input_arg)
        if input_path.is_dir():
            input_dir = input_path
            input_files = _discover_conversation_files(input_path)
            if not input_files:
                console.print(
                    "[red]ERROR:[/red] No SUS conversations JSON found in "
                    f"{input_path}"
                )
                sys.exit(1)
        else:
            input_files = [input_path]
    else:
        input_files = [Path(filepath) for filepath in getattr(args, "files", [])]

    models_path = Path(args.models) if args.models else pkg_root / "models.yaml"
    try:
        prepared_config_receipt = (
            validate_run_prepared_config_before_spend(input_dir, models_path)
            if input_dir is not None
            else False
        )
        if prepared_config_receipt and (
            getattr(args, "analyzer_model", None)
            or getattr(args, "judge_panel", None)
        ):
            raise PreparedConfigProvenanceError(
                "prepared scoring forbids analyzer/judge-panel overrides"
            )
    except PreparedConfigProvenanceError as exc:
        console.print(f"[red]ERROR:[/red] {sanitize_error_message(exc)}")
        console.print("Refusing to spend with changed prepared scoring configuration.")
        RunMonitor(input_dir, module="sus", stage="scoring").mark_failed(
            exc,
            status="failed_invalid",
            failure_stage="prepared_config_provenance",
            provenance_issues=list(exc.issues),
        )
        sys.exit(2)
    models_config = {}
    if args.models:
        models_config = load_models_config(models_path)
    elif (pkg_root / "models.yaml").exists():
        models_config = load_models_config(pkg_root / "models.yaml")

    analyzer_model = (
        args.analyzer_model
        or models_config.get("analyzer")
        or "google/gemini-3-flash-preview"
    )
    if args.judge_panel:
        judge_panel = [item.strip() for item in args.judge_panel.split(",") if item.strip()]
        judge_configs = None
    else:
        judge_panel = models_config.get("judge_panel") or DEFAULT_JUDGE_PANEL
        judge_configs = models_config.get("judge_configs")

    all_results = []
    result_sources = []
    loaded_sources: set[Path] = set()
    for filepath in input_files:
        results, source_path = _load_report_results(filepath)
        resolved_source = source_path.resolve()
        if resolved_source in loaded_sources:
            continue
        loaded_sources.add(resolved_source)
        for result in results:
            all_results.append(result)
            result_sources.append(source_path)

    if not all_results:
        console.print("[red]ERROR:[/red] No conversation-level results found.")
        sys.exit(1)

    output_dir = (
        Path(args.output).parent
        if getattr(args, "output", None)
        else input_dir
        if input_dir is not None
        else pkg_root / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_ledger_path = input_dir / "RUN_STATUS.json" if input_dir is not None else None
    source_ledger_status = _read_run_status(source_ledger_path) if source_ledger_path else {}
    existing_cost = _read_existing_status_cost(output_dir)
    monitor = RunMonitor(
        output_dir,
        module="sus",
        stage="scoring",
        metadata={
            "analyzer_model": analyzer_model,
            "judge_panel": judge_panel,
            "judge_configs": judge_configs,
            "source_files": [str(path) for path in sorted(loaded_sources)],
        },
    )
    if existing_cost:
        monitor.status["cost"] = {
            **existing_cost,
            **_merge_cost_summaries(existing_cost, {}),
        }
        atomic_write_json(monitor.status_path, monitor.status)
    if prepared_config_receipt:
        monitor.record("prepared_config_verified", **prepared_config_receipt)
    if input_dir is not None and (input_dir / "RUN_CONTRACT.json").is_file():
        try:
            artifact_identity = require_run_artifact_identity(input_dir)
        except ValueError as exc:
            console.print(f"[red]ERROR:[/red] {sanitize_error_message(exc)}")
            console.print(
                "Refusing to spend on judges for identity-invalid transcripts."
            )
            monitor.mark_failed(
                exc,
                status="failed_invalid",
                failure_stage="artifact_identity",
            )
            sys.exit(2)
        monitor.record(
            "artifact_identity_verified",
            checked_artifacts=artifact_identity["checked_artifacts"],
            checkable_artifacts=artifact_identity["checkable_artifacts"],
        )
        try:
            identity_normalization = _hydrate_score_input_identity_from_transcripts(
                all_results,
                input_dir,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            console.print(f"[red]ERROR:[/red] {sanitize_error_message(exc)}")
            console.print(
                "Refusing to spend on judges because the aggregate score input "
                "cannot be reconciled with its saved transcript artifacts."
            )
            monitor.mark_failed(
                exc,
                status="failed_invalid",
                failure_stage="artifact_identity",
            )
            sys.exit(2)
        if identity_normalization["restored_rows"]:
            monitor.record(
                "score_input_identity_normalized",
                method="restored_from_saved_transcript_artifact",
                restored_rows=identity_normalization["restored_rows"],
                restored_fields=identity_normalization["restored_fields"],
            )
    if args.models:
        try:
            _validate_rescore_condition_identity(all_results, models_config)
        except ValueError as exc:
            console.print(f"[red]ERROR:[/red] {sanitize_error_message(exc)}")
            console.print(
                "Refusing to spend on judges for transcripts whose saved condition "
                "identity does not match the rendered model configuration."
            )
            monitor.mark_failed(
                exc,
                status="failed_invalid",
                failure_stage="artifact_identity",
            )
            sys.exit(2)
    try:
        request_conformance = require_request_conformance(
            input_dir or output_dir,
            roles={"model_under_test"},
        )
    except RequestConformanceError as exc:
        console.print(f"[red]ERROR:[/red] {sanitize_error_message(exc)}")
        console.print(
            "Refusing to spend on judges for generation whose effective requests "
            "are unverified."
        )
        monitor.mark_failed(
            exc,
            status="failed_invalid",
            failure_stage="request_conformance",
            request_conformance=exc.result,
        )
        sys.exit(2)
    if request_conformance["requirement_count"]:
        monitor.record(
            "request_conformance_verified",
            roles=request_conformance["roles"],
            requirement_count=request_conformance["requirement_count"],
            receipt_count=request_conformance["receipt_count"],
            legacy_unverified_requirement_count=request_conformance[
                "legacy_unverified_requirement_count"
            ],
        )

    if input_dir is not None:
        contract = load_run_contract(input_dir)
        frozen_panel = ((contract.get("identity") or {}).get("judge_panel") or {})
        effective_configs = judge_configs or [
            {"model_id": judge, "provider_api": "openai_compatible"}
            for judge in judge_panel
        ]
        resolved_panel = {
            **frozen_panel,
            "analyzer": analyzer_model,
            "panel": list(judge_panel),
            "configs": effective_configs,
            "judge_prompt_hashes": {
                "post_analysis": stable_json_hash(POST_ANALYSIS_PROMPT),
            },
            "rubric_version": RUBRIC_VERSION,
            "rubric_source_ids": list(RUBRIC_SOURCE_IDS),
            "rubric_source_registry": RUBRIC_SOURCE_REGISTRY,
        }
        try:
            validated = validate_run_judge_provenance_before_spend(
                input_dir,
                resolved_panel,
            )
        except JudgeProvenanceError as exc:
            console.print(f"[red]ERROR:[/red] {sanitize_error_message(exc)}")
            console.print(
                "Refusing to spend on judges whose resolved identity differs "
                "from the run contract."
            )
            monitor.mark_failed(
                exc,
                status="failed_invalid",
                failure_stage="judge_provenance",
                drift_fields=list(exc.drift_fields),
            )
            sys.exit(2)
        if validated:
            monitor.record("judge_provenance_verified", judge_panel=list(judge_panel))

    hygiene_issues = []
    for index, (result, source_path) in enumerate(zip(all_results, result_sources), start=1):
        if is_score_excluded_result(result):
            continue
        hygiene_issues.extend(
            blocking_issue_summaries(
                result,
                source=source_path,
                record_index=index,
            )
        )
    if hygiene_issues:
        console.print("[red]ERROR:[/red] Refusing to score SUS conversations with blocking hygiene issues:")
        for issue in hygiene_issues:
            console.print(f"  - {issue}")
        console.print("Rerun or quarantine these transcripts before judge scoring.")
        monitor.mark_failed(
            "SUS transcripts are not scoreable",
            status="failed_incomplete",
            failure_stage="hygiene",
            transcript_hygiene_issues=hygiene_issues,
        )
        sys.exit(2)

    try:
        preflight_admission = validate_preflight_receipt_for_prepared_config(
            input_dir,
            prepared_config_receipt,
        )
    except PreflightReceiptValidationError as exc:
        console.print(f"[red]ERROR:[/red] {sanitize_error_message(exc)}")
        console.print(
            "Refusing prepared scoring without current exact-condition "
            "preflight evidence."
        )
        monitor.mark_failed(
            exc,
            status="failed_invalid",
            failure_stage="preflight_receipt_admission",
            provenance_issues=list(exc.issues),
        )
        sys.exit(2)
    if preflight_admission:
        monitor.record("preflight_receipt_admitted", **preflight_admission)

    load_repo_env_files()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]ERROR:[/red] OPENROUTER_API_KEY not set.")
        monitor.mark_failed(
            "OPENROUTER_API_KEY not set",
            status="failed_invalid",
            failure_stage="credentials",
        )
        sys.exit(1)
    reset_cost_tracker()
    tracker = get_cost_tracker()
    tracker.set_api_key(api_key)

    rescored_items = []
    score_failures = []
    score_parallelism = _configured_score_parallelism(getattr(args, "score_parallelism", None))
    scoreable_input_count = sum(1 for result in all_results if not is_score_excluded_result(result))
    source_files = [str(path) for path in sorted(loaded_sources)]

    def score_one_result(idx: int, result: dict):
        if is_score_excluded_result(result):
            monitor.record(
                "score_skipped",
                index=idx,
                model=result.get("model"),
                scenario=result.get("scenario"),
                reason=result.get("exclusion_reason") or result.get("score_state"),
            )
            return idx, dict(result), None, "provider refusal excluded"

        conversation = result.get("conversation")
        phases = result.get("phases")
        if not isinstance(conversation, list) or not isinstance(phases, dict):
            return idx, None, None, f"missing conversation/phases"

        first_user_message = ""
        if conversation and isinstance(conversation[0], dict):
            first_user_message = conversation[0].get("content", "")
        elicitation_prompt = phases.get("elicit", {}).get("prompt") or first_user_message
        model_id = result.get("model", "unknown")
        monitor.record(
            "score_started",
            index=idx,
            total=len(all_results),
            model=model_id,
            scenario=result.get("scenario"),
        )
        post_analysis_kwargs = {
            "model_id": model_id,
            "judge_panel": judge_panel,
            "call_context": {
                "unit_id": result.get("unit_id"),
                "condition_id": result.get("condition_id"),
                "scenario": result.get("scenario"),
                "phase": "post_analysis",
            },
            "monitor": monitor,
        }
        if judge_configs:
            post_analysis_kwargs["judge_configs"] = judge_configs
        try:
            post_analysis = run_post_analysis(
                conversation,
                elicitation_prompt,
                analyzer_model,
                api_key,
                **post_analysis_kwargs,
            )
        except JudgePanelIncompleteError as e:
            failure = {
                "index": idx,
                "model": model_id,
                "scenario": result.get("scenario"),
                "failure_reason": str(e),
                **e.to_status_payload(),
            }
            monitor.record(
                "score_failed",
                index=idx,
                model=model_id,
                scenario=result.get("scenario"),
                failure_reason=str(e),
                judge_panel_complete=False,
                missing_judges=e.missing_judges,
                judge_failures=e.judge_failures,
                rerun_recommended=True,
            )
            return idx, None, failure, None
        if post_analysis is None:
            failure = {
                "index": idx,
                "model": model_id,
                "scenario": result.get("scenario"),
                "failure_reason": "Judge panel returned no valid scores",
                "judge_panel_complete": False,
                "expected_judges": judge_panel,
                "successful_judges": [],
                "missing_judges": judge_panel,
                "judge_failures": [],
                "rerun_recommended": True,
            }
            monitor.record(
                "score_failed",
                index=idx,
                model=model_id,
                scenario=result.get("scenario"),
                failure_reason="Judge panel returned no valid scores",
                judge_panel_complete=False,
                rerun_recommended=True,
            )
            return idx, None, failure, None

        updated = dict(result)
        updated["post_analysis"] = post_analysis
        updated["score"] = compute_sus_score(phases, post_analysis)
        updated["score_state"] = "scored"
        updated["rescore_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "analyzer_model": analyzer_model,
            "judge_panel": judge_panel,
            "judge_configs": judge_configs,
            "source_files": source_files,
        }
        updated.update(classify_result(updated))
        monitor.record(
            "score_saved",
            index=idx,
            model=model_id,
            scenario=result.get("scenario"),
            sus=updated["score"].get("sus"),
            judges=(post_analysis or {}).get("num_judges"),
            unit_id=result.get("unit_id"),
            run_number=result.get("run_number"),
        )
        return idx, updated, None, None

    monitor.record(
        "score_batch_started",
        score_parallelism=score_parallelism,
        score_items=len(all_results),
    )
    try:
        work_items = list(enumerate(all_results, start=1))
        if score_parallelism <= 1:
            for idx, result in work_items:
                console.print(f"Scoring {idx}/{len(all_results)} {result.get('model', 'unknown')} {result.get('scenario', '')}...")
                score_idx, updated, failure, skipped = score_one_result(idx, result)
                if skipped:
                    console.print(f"[yellow]Skipping result {score_idx}: {skipped}[/yellow]")
                if failure:
                    score_failures.append(failure)
                    console.print(
                        f"[red]Judge panel failed for result {score_idx}; "
                        "rerun/rescore required.[/red]"
                    )
                if updated:
                    rescored_items.append((score_idx, updated))
        else:
            console.print(
                f"Scoring {len(work_items)} SUS result(s) with parallelism={score_parallelism}..."
            )
            with ThreadPoolExecutor(max_workers=score_parallelism) as executor:
                futures = {
                    executor.submit(score_one_result, idx, result): idx
                    for idx, result in work_items
                }
                for future in as_completed(futures):
                    score_idx, updated, failure, skipped = future.result()
                    if skipped:
                        console.print(f"[yellow]Skipping result {score_idx}: {skipped}[/yellow]")
                    if failure:
                        score_failures.append(failure)
                        console.print(
                            f"[red]Judge panel failed for result {score_idx}; "
                            "rerun/rescore required.[/red]"
                        )
                    if updated:
                        rescored_items.append((score_idx, updated))
    except Exception as e:
        monitor.mark_failed(e, status="failed_scoring")
        console.print(f"[red]ERROR:[/red] {sanitize_error_message(e)}")
        sys.exit(2)

    rescored = [
        result
        for _idx, result in sorted(rescored_items, key=lambda item: item[0])
    ]
    scored_result_count = sum(
        1 for result in rescored if not is_score_excluded_result(result)
    )
    excluded_result_count = len(rescored) - scored_result_count
    score_failures.sort(key=lambda item: int(item.get("index", 0)))

    out_path = (
        Path(args.output)
        if getattr(args, "output", None)
        else input_dir / "FINAL_RESULTS.json"
        if input_dir is not None
        else output_dir / f"sus-rescore-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    cost_summary = tracker.summary()
    combined_cost = _merge_cost_summaries(existing_cost, cost_summary)

    if score_failures:
        partial_path = None
        if rescored:
            partial_path = out_path.with_name(f"{out_path.stem}-partial{out_path.suffix}")
            write_json(
                rescored,
                aggregate_runs(rescored),
                partial_path,
                run_id=partial_path.stem,
                cost=combined_cost,
            )
            console.print(f"[yellow]Partial rescored results saved: {partial_path}[/yellow]")
        monitor.mark_failed(
            "SUS scoring incomplete; rerun/rescore required",
            status="failed_scoring",
            failure_stage="judge_panel",
            scored_results=len(rescored),
            expected_results=len(all_results),
            score_failures=score_failures,
            partial_results_path=str(partial_path) if partial_path else None,
            scoring_cost=cost_summary,
            rerun_recommended=True,
        )
        console.print("[red]ERROR:[/red] SUS scoring incomplete; rerun/rescore required.")
        sys.exit(2)

    if not scored_result_count:
        console.print("[red]ERROR:[/red] No results were rescored.")
        monitor.mark_failed("No results were rescored.", status="failed_scoring")
        sys.exit(1)

    aggregated = aggregate_runs(rescored)
    print_table(aggregated)
    print_segmented_table(rescored)

    write_json(
        rescored,
        aggregated,
        out_path,
        run_id=out_path.stem,
        cost=combined_cost,
    )
    console.print(f"\nRescored results saved: {out_path}")
    monitor.mark_completed(
        scored_results=scored_result_count,
        excluded_results=excluded_result_count,
        results_path=str(out_path),
        scoring_cost=cost_summary,
    )
    source_ledger_state = str(source_ledger_status.get("status", ""))
    output_ledger_path = output_dir / "RUN_STATUS.json"
    stale_source_ledger = (
        source_ledger_path is not None
        and source_ledger_state.startswith("failed_")
        and source_ledger_path.resolve() != output_ledger_path.resolve()
        and scored_result_count == scoreable_input_count
        and not score_failures
    )
    if stale_source_ledger:
        console.print(
            f"NOTE: {source_ledger_path} still records status={source_ledger_state} from a prior\n"
            f"scoring attempt. This rescore completed successfully in {output_dir}. If this\n"
            "output supersedes the module's scoring, update the module ledger by rerunning\n"
            "the score command in place, or record the promotion decision manually."
        )


def _read_run_status(status_path: Path) -> dict:
    try:
        with status_path.open() as handle:
            status = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return status if isinstance(status, dict) else {}


def _read_existing_status_cost(output_dir: Path) -> dict:
    status_path = output_dir / "RUN_STATUS.json"
    status = _read_run_status(status_path)
    cost = status.get("cost") if isinstance(status, dict) else None
    return dict(cost) if isinstance(cost, dict) else {}


def _merge_cost_summaries(existing: dict | None, stage: dict | None) -> dict:
    """Combine prior generation cost with the current scoring-stage cost."""
    from suite_tools.run_monitor import nonnegative_finite_number, nonnegative_integer

    existing = existing or {}
    stage = stage or {}
    invalid_money_values = 0

    def as_float(value) -> float:
        nonlocal invalid_money_values
        if value is None or value == "":
            return 0.0
        parsed, valid = nonnegative_finite_number(value)
        if not valid:
            invalid_money_values += 1
            return 0.0
        return parsed

    def as_int(value) -> int:
        parsed, valid = nonnegative_integer(value)
        return parsed if valid else 0

    def merge_money_map(key: str) -> dict:
        merged = {}
        for source in (existing.get(key) or {}, stage.get(key) or {}):
            if not isinstance(source, dict):
                continue
            for name, value in source.items():
                merged[str(name)] = round(as_float(merged.get(str(name))) + as_float(value), 8)
        return merged

    def merge_count_map(key: str) -> dict:
        merged = {}
        for source in (existing.get(key) or {}, stage.get(key) or {}):
            if not isinstance(source, dict):
                continue
            for name, value in source.items():
                merged[str(name)] = as_int(merged.get(str(name))) + as_int(value)
        return merged

    merged = {
        "total_cost_usd": round(
            as_float(existing.get("total_cost_usd")) + as_float(stage.get("total_cost_usd")),
            8,
        ),
        "total_calls": as_int(existing.get("total_calls")) + as_int(stage.get("total_calls")),
        "tokens_in": as_int(existing.get("tokens_in")) + as_int(stage.get("tokens_in")),
        "tokens_out": as_int(existing.get("tokens_out")) + as_int(stage.get("tokens_out")),
        "thinking_tokens_out": (
            as_int(existing.get("thinking_tokens_out")) + as_int(stage.get("thinking_tokens_out"))
        ),
        "cost_by_model": merge_money_map("cost_by_model"),
        "cost_by_role": merge_money_map("cost_by_role"),
        "reported_cost_usd": round(
            as_float(existing.get("reported_cost_usd"))
            + as_float(stage.get("reported_cost_usd")),
            8,
        ),
        "estimated_cost_usd": round(
            as_float(existing.get("estimated_cost_usd"))
            + as_float(stage.get("estimated_cost_usd")),
            8,
        ),
        "cost_by_source": merge_money_map("cost_by_source"),
        "unknown_cost_calls": (
            as_int(existing.get("unknown_cost_calls"))
            + as_int(stage.get("unknown_cost_calls"))
        ),
        "unknown_cost_by_model": merge_count_map("unknown_cost_by_model"),
    }
    if stage.get("credit_remaining_usd") is not None:
        merged["credit_remaining_usd"] = as_float(stage["credit_remaining_usd"])
    elif existing.get("credit_remaining_usd") is not None:
        merged["credit_remaining_usd"] = as_float(existing["credit_remaining_usd"])
    merged["usage_anomaly_count"] = (
        as_int(existing.get("usage_anomaly_count"))
        + as_int(stage.get("usage_anomaly_count"))
        + invalid_money_values
    )
    merged["invalid_usage_fields"] = merge_count_map("invalid_usage_fields")
    if invalid_money_values:
        merged["invalid_usage_fields"]["legacy_cost"] = (
            as_int(merged["invalid_usage_fields"].get("legacy_cost"))
            + invalid_money_values
        )
    return merged


def _discover_conversation_files(input_dir: Path) -> list[Path]:
    """Return generation conversation files, excluding prior score outputs."""
    candidates = []
    for path in sorted(input_dir.glob("*-conversations.json")):
        if path.name == "FINAL_RESULTS-conversations.json":
            continue
        if path.name.startswith("sus-rescore-"):
            continue
        candidates.append(path)
    if candidates:
        return candidates
    final_conversations = input_dir / "FINAL_RESULTS-conversations.json"
    return [final_conversations] if final_conversations.exists() else []


def _load_report_results(path: Path) -> tuple[list[dict], Path]:
    """Load conversation-level results for the report command.

    Summary files are intentionally small and contain only aggregate rows. When
    a summary file is passed, resolve its sibling `*-conversations.json` file.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [_normalize_report_result(r) for r in data], path

    if not isinstance(data, dict):
        return [], path

    results = data.get("results")
    if isinstance(results, list):
        return [_normalize_report_result(r) for r in results], path

    if "aggregated" in data or "cost" in data:
        conv_path = path.with_name(f"{path.stem}-conversations.json")
        if conv_path.exists():
            return _load_report_results(conv_path)
        raise ValueError(
            "Summary JSON does not contain conversation-level results. "
            f"Expected sibling file: {conv_path}"
        )

    return [], path


def _normalize_report_result(result: dict) -> dict:
    """Return a report-compatible copy of a result dict.

    Older artifacts used `score.sts` where higher meant safer. The current
    report pipeline expects `score.sus` where higher means worse.
    """
    if not isinstance(result, dict):
        return result
    score = result.get("score")
    if not isinstance(score, dict) or "sus" in score or "sts" not in score:
        return result

    sts = score.get("sts")
    if not isinstance(sts, (int, float)):
        return result

    normalized = dict(result)
    normalized_score = dict(score)
    normalized_score["legacy_sts"] = sts
    normalized_score["sus"] = 100 - float(sts)
    normalized["score"] = normalized_score
    return normalized


def _find_package_root() -> Path:
    """Find the package root directory (where models.yaml lives).

    Searches upward from CWD, then tries the package install location.
    """
    # Check CWD first
    cwd = Path.cwd()
    if (cwd / "models.yaml").exists():
        return cwd

    # Check parent directories (up to 3 levels)
    for parent in [cwd.parent, cwd.parent.parent, cwd.parent.parent.parent]:
        if (parent / "models.yaml").exists():
            return parent

    # Fall back to the package's own directory
    pkg_dir = Path(__file__).parent.parent
    if (pkg_dir / "models.yaml").exists():
        return pkg_dir

    # Last resort: CWD
    return cwd


if __name__ == "__main__":
    main()
