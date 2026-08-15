"""Durable, prompt-free coordination state for benchmark workflows.

The benchmark ledgers remain authoritative.  This module only remembers which
run directories belong to an operator workflow and derives a compact resume
receipt from their current contracts, statuses, and unit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from suite_tools.owed_units import owed_units
from suite_tools.run_monitor import atomic_write_json, utc_now


EVENT_SCHEMA_VERSION = "benchmark-companion-event-v1"
RESUME_SCHEMA_VERSION = "benchmark-companion-resume-v1"
ACTIVE_SCHEMA_VERSION = "benchmark-companion-active-v1"
CLAIM_SCHEMA_VERSION = "benchmark-companion-approval-claim-v1"
EVENTS_FILENAME = "EVENTS.jsonl"
RESUME_FILENAME = "RESUME.json"
ACTIVE_FILENAME = "ACTIVE.json"
CLAIMS_DIRNAME = "approval-claims"

WORKFLOW_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
GOALS = {
    "onboarding",
    "smoke",
    "collection",
    "recovery",
    "review",
    "evidence_package",
}
APPROVAL_STAGES = {"preflight", "generation", "scoring"}
CHOICES = {
    "connection_route": {
        "openrouter",
        "provider_direct",
        "openai_compatible",
        "bundled_adapter",
    },
    "target_provider": {
        "openrouter",
        "anthropic",
        "openai",
        "google",
        "custom",
        "local",
    },
    "first_module": {"sus", "aita", "epistemic", "suite"},
}


def default_state_root(start: Path | str | None = None) -> Path:
    """Return the companion dir for the benchmark checkout containing *start*."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "suite_models.yaml").is_file() and (
            candidate / "suite_tools"
        ).is_dir():
            return candidate / ".benchmark-companion"
    return current / ".benchmark-companion"


def _resolved_state_root(state_root: Path | str | None) -> Path:
    return Path(state_root) if state_root is not None else default_state_root()


def _validate_workflow_id(workflow_id: str) -> str:
    if not WORKFLOW_ID_RE.fullmatch(workflow_id):
        raise ValueError(
            "workflow_id must be 1-80 lowercase letters, digits, dots, dashes, or underscores"
        )
    return workflow_id


def _workflow_dir(state_root: Path | str, workflow_id: str) -> Path:
    return Path(state_root) / _validate_workflow_id(workflow_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_events(state_root: Path | str, workflow_id: str) -> list[dict[str, Any]]:
    path = _workflow_dir(state_root, workflow_id) / EVENTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Companion workflow not found: {workflow_id}")
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed companion event log: {path}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported companion event in {path}")
        events.append(value)
    return events


def _append_event(
    state_root: Path | str,
    workflow_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "recorded_at": utc_now(),
        "workflow_id": workflow_id,
        "event": event_type,
        **payload,
    }
    directory = _workflow_dir(state_root, workflow_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EVENTS_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def _contract_scope(run_dir: Path | str) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    contract_path = run_path / "RUN_CONTRACT.json"
    try:
        raw = contract_path.read_bytes()
        contract = json.loads(raw)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RUN_CONTRACT.json not found in {run_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed RUN_CONTRACT.json in {run_path}") from exc
    if not isinstance(contract, dict):
        raise ValueError(f"RUN_CONTRACT.json must contain an object: {contract_path}")

    expected_units = sum(
        len(module.get("expected_units") or [])
        for module in (contract.get("modules") or [])
        if isinstance(module, dict)
    )
    model_routes = sorted(
        {
            str(
                (model.get("condition_metadata") or {}).get("provider_route")
                or model.get("endpoint")
                or "unknown"
            )
            for model in (contract.get("expected_models") or [])
            if isinstance(model, dict)
        }
    )
    judge_routes = sorted(
        {
            str(
                ((judge.get("config") or {}).get("condition_metadata") or {}).get(
                    "provider_route"
                )
                or (judge.get("config") or {}).get("provider_api")
                or "unknown"
            )
            for judge in (contract.get("expected_judges") or [])
            if isinstance(judge, dict)
        }
    )
    route_payload = {
        "model_routes": model_routes,
        "judge_routes": judge_routes,
    }
    route_fingerprint = hashlib.sha256(
        json.dumps(route_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "run_dir": str(run_path),
        "run_id": str(contract.get("run_id") or run_path.name),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_units": expected_units,
        "model_routes": model_routes,
        "judge_routes": judge_routes,
        "route_fingerprint": route_fingerprint,
    }


def _attached_runs(events: list[dict[str, Any]]) -> list[str]:
    attached: list[str] = []
    for event in events:
        if event.get("event") not in {"workflow_started", "run_attached"}:
            continue
        values = event.get("run_dirs") or [event.get("run_dir")]
        for value in values:
            if value and str(value) not in attached:
                attached.append(str(value))
    return attached


def _run_phase(status: dict[str, Any], counts: dict[str, int]) -> tuple[str, str | None]:
    status_name = str(status.get("status") or "")
    stage = str(status.get("stage") or "")
    validity = str(status.get("validity") or "")
    owed = int(counts.get("owed") or 0)

    if not status_name or status_name == "prepared":
        return "prepared", None
    if status_name == "stopped" or status_name.startswith("failed"):
        return "attention", str(status.get("failure_reason") or status_name)
    if status_name == "running":
        return ("scoring" if stage == "scoring" else "generating"), None
    if status_name == "completed" and owed:
        return "ledger_conflict", "status is completed while contract units remain owed"
    if status_name == "completed" and validity == "score_ready":
        return "completed", None
    if (
        status_name == "completed"
        and validity == "not_score_ready"
        and stage == "generation"
    ):
        return "needs_scoring", None
    if status_name == "completed":
        return "attention", "completed status has an unrecognized stage or validity"
    return "attention", f"unrecognized RUN_STATUS state: {status_name or 'missing'}"


def _summarize_run(run_dir: Path | str) -> dict[str, Any]:
    try:
        scope = _contract_scope(run_dir)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "run_dir": str(Path(run_dir).resolve()),
            "phase": "missing_contract",
            "issue": str(exc),
        }

    run_path = Path(scope["run_dir"])
    status = _read_json(run_path / "RUN_STATUS.json")
    try:
        owed = owed_units(run_path)
        counts = {
            "done": int((owed.get("counts") or {}).get("done") or 0),
            "terminal_model_signal": int(
                (owed.get("counts") or {}).get("terminal_model_signal") or 0
            ),
            "owed": int((owed.get("counts") or {}).get("owed") or 0),
        }
        phase, issue = _run_phase(status, counts)
    except Exception as exc:  # A malformed scientific ledger must fail closed.
        counts = {"done": 0, "terminal_model_signal": 0, "owed": scope["expected_units"]}
        phase, issue = "ledger_conflict", str(exc)

    return {
        **scope,
        "phase": phase,
        "status": status.get("status"),
        "stage": status.get("stage"),
        "validity": status.get("validity"),
        "counts": counts,
        "issue": issue,
    }


def _aggregate_phase(runs: list[dict[str, Any]]) -> tuple[str, str]:
    phases = {str(run.get("phase")) for run in runs}
    if not runs:
        return "setup", "attach_or_prepare_a_run"
    if phases & {"attention", "ledger_conflict", "missing_contract"}:
        return "attention", "inspect_authoritative_ledgers"
    if phases & {"generating", "scoring"}:
        return "active", "observe_current_run"
    if "prepared" in phases:
        return "prepared", "preflight_or_generate"
    if "needs_scoring" in phases:
        return "needs_scoring", "review_generation_and_approve_scoring"
    if phases == {"completed"}:
        return "completed", "prepare_evidence_package"
    return "attention", "inspect_authoritative_ledgers"


def _approval_views(
    events: list[dict[str, Any]],
    runs_by_path: dict[str, dict[str, Any]],
    claimed_approval_ids: set[str],
) -> list[dict[str, Any]]:
    consumed = {
        str(event.get("approval_id"))
        for event in events
        if event.get("event") == "approval_consumed" and event.get("approval_id")
    }
    approvals: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "approval_granted":
            continue
        approval_id = str(event.get("approval_id"))
        run_dir = str(event.get("run_dir"))
        current = runs_by_path.get(run_dir)
        if approval_id in consumed or approval_id in claimed_approval_ids:
            state = "consumed"
        elif not current or not current.get("contract_sha256"):
            state = "stale_missing_run"
        elif current.get("contract_sha256") != event.get("contract_sha256"):
            state = "stale_contract"
        elif current.get("route_fingerprint") != event.get("route_fingerprint"):
            state = "stale_route"
        else:
            state = "active"
        approvals.append(
            {
                "approval_id": approval_id,
                "run_dir": run_dir,
                "stage": event.get("stage"),
                "contract_sha256": event.get("contract_sha256"),
                "route_fingerprint": event.get("route_fingerprint"),
                "expected_units": event.get("expected_units"),
                "granted_at": event.get("recorded_at"),
                "state": state,
            }
        )
    return approvals


def _claimed_approval_ids(state_root: Path | str, workflow_id: str) -> set[str]:
    directory = _workflow_dir(state_root, workflow_id) / CLAIMS_DIRNAME
    if not directory.exists():
        return set()
    return {path.stem for path in directory.glob("*.json") if path.is_file()}


def _claim_approval(
    state_root: Path | str,
    workflow_id: str,
    approval_id: str,
) -> None:
    directory = _workflow_dir(state_root, workflow_id) / CLAIMS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{approval_id}.json"
    payload = json.dumps(
        {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "approval_id": approval_id,
            "claimed_at": utc_now(),
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"approval already consumed: {approval_id}") from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _set_active(state_root: Path | str, workflow_id: str) -> None:
    atomic_write_json(
        Path(state_root) / ACTIVE_FILENAME,
        {
            "schema_version": ACTIVE_SCHEMA_VERSION,
            "workflow_id": workflow_id,
            "updated_at": utc_now(),
        },
    )


def active_workflow_id(state_root: Path | str | None = None) -> str:
    active = _read_json(_resolved_state_root(state_root) / ACTIVE_FILENAME)
    workflow_id = active.get("workflow_id")
    if not workflow_id:
        raise FileNotFoundError("No active companion workflow")
    return _validate_workflow_id(str(workflow_id))


def start_workflow(
    state_root: Path | str,
    *,
    workflow_id: str,
    goal: str,
    run_dirs: list[Path | str] | None = None,
) -> dict[str, Any]:
    _validate_workflow_id(workflow_id)
    if goal not in GOALS:
        raise ValueError(f"goal must be one of {sorted(GOALS)}")
    directory = _workflow_dir(state_root, workflow_id)
    if (directory / EVENTS_FILENAME).exists():
        raise ValueError(f"Companion workflow already exists: {workflow_id}")
    scopes = [_contract_scope(run_dir) for run_dir in (run_dirs or [])]
    _append_event(
        state_root,
        workflow_id,
        "workflow_started",
        {
            "goal": goal,
            "run_dirs": [scope["run_dir"] for scope in scopes],
        },
    )
    _set_active(state_root, workflow_id)
    return resume_workflow(state_root, workflow_id)


def attach_run(
    state_root: Path | str,
    workflow_id: str,
    *,
    run_dir: Path | str,
) -> dict[str, Any]:
    events = _read_events(state_root, workflow_id)
    scope = _contract_scope(run_dir)
    if scope["run_dir"] not in _attached_runs(events):
        _append_event(
            state_root,
            workflow_id,
            "run_attached",
            {"run_dir": scope["run_dir"]},
        )
    _set_active(state_root, workflow_id)
    return resume_workflow(state_root, workflow_id)


def record_choice(
    state_root: Path | str,
    workflow_id: str,
    *,
    key: str,
    value: str,
) -> dict[str, Any]:
    _read_events(state_root, workflow_id)
    if key not in CHOICES:
        raise ValueError(f"unsupported choice key: {key}")
    if value not in CHOICES[key]:
        raise ValueError(
            f"unsupported choice value for {key}: {value}; expected one of "
            f"{sorted(CHOICES[key])}"
        )
    _append_event(
        state_root,
        workflow_id,
        "choice_recorded",
        {"key": key, "value": value},
    )
    _set_active(state_root, workflow_id)
    return resume_workflow(state_root, workflow_id)


def grant_approval(
    state_root: Path | str,
    workflow_id: str,
    *,
    run_dir: Path | str,
    stage: str,
    confirmed_by_user: bool,
) -> dict[str, Any]:
    if not confirmed_by_user:
        raise ValueError("approval requires explicit user confirmation")
    if stage not in APPROVAL_STAGES:
        raise ValueError(f"stage must be one of {sorted(APPROVAL_STAGES)}")
    events = _read_events(state_root, workflow_id)
    scope = _contract_scope(run_dir)
    if scope["run_dir"] not in _attached_runs(events):
        raise ValueError("run must be attached to the workflow before approval")
    run_summary = _summarize_run(scope["run_dir"])
    run_phase = str(run_summary.get("phase") or "")
    if stage == "scoring" and not (
        run_phase == "needs_scoring"
        or (run_phase == "attention" and run_summary.get("stage") == "scoring")
    ):
        raise ValueError("scoring approval requires needs_scoring or failed scoring state")
    if stage == "generation" and run_phase not in {"prepared", "attention"}:
        raise ValueError("generation approval requires a prepared or attention state")
    if stage == "preflight" and run_phase not in {"prepared", "attention"}:
        raise ValueError("preflight approval requires a prepared or attention state")
    approval_id = str(uuid.uuid4())
    _append_event(
        state_root,
        workflow_id,
        "approval_granted",
        {
            "approval_id": approval_id,
            "run_dir": scope["run_dir"],
            "stage": stage,
            "contract_sha256": scope["contract_sha256"],
            "route_fingerprint": scope["route_fingerprint"],
            "expected_units": scope["expected_units"],
        },
    )
    _set_active(state_root, workflow_id)
    return {"approval_id": approval_id, "resume": resume_workflow(state_root, workflow_id)}


def consume_approval(
    state_root: Path | str,
    workflow_id: str,
    *,
    approval_id: str,
) -> dict[str, Any]:
    resume = resume_workflow(state_root, workflow_id)
    approval = next(
        (item for item in resume["approvals"] if item["approval_id"] == approval_id),
        None,
    )
    if approval is None:
        raise ValueError(f"approval not found: {approval_id}")
    if approval["state"] == "consumed":
        raise ValueError(f"approval already consumed: {approval_id}")
    if approval["state"] != "active":
        raise ValueError(f"approval is not active: {approval['state']}")
    _claim_approval(state_root, workflow_id, approval_id)
    _append_event(
        state_root,
        workflow_id,
        "approval_consumed",
        {"approval_id": approval_id},
    )
    return resume_workflow(state_root, workflow_id)


def _resume_workflow(
    state_root: Path | str | None = None,
    workflow_id: str | None = None,
    *,
    activate: bool,
) -> dict[str, Any]:
    state_root = _resolved_state_root(state_root)
    selected = workflow_id or active_workflow_id(state_root)
    events = _read_events(state_root, selected)
    started = next(
        (event for event in events if event.get("event") == "workflow_started"),
        {},
    )
    runs = [_summarize_run(path) for path in _attached_runs(events)]
    runs_by_path = {str(run.get("run_dir")): run for run in runs}
    phase, next_action = _aggregate_phase(runs)
    choices = {
        str(event["key"]): str(event["value"])
        for event in events
        if event.get("event") == "choice_recorded"
        and event.get("key") in CHOICES
        and event.get("value") in CHOICES[str(event.get("key"))]
    }
    approvals = _approval_views(
        events,
        runs_by_path,
        _claimed_approval_ids(state_root, selected),
    )
    snapshot = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "workflow_id": selected,
        "goal": started.get("goal"),
        "generated_at": utc_now(),
        "phase": phase,
        "next_action": next_action,
        "choices": choices,
        "runs": runs,
        "approvals": approvals,
        "active_approvals": [
            approval for approval in approvals if approval.get("state") == "active"
        ],
    }
    atomic_write_json(_workflow_dir(state_root, selected) / RESUME_FILENAME, snapshot)
    if activate:
        _set_active(state_root, selected)
    return snapshot


def resume_workflow(
    state_root: Path | str | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    return _resume_workflow(state_root, workflow_id, activate=True)


def list_workflows(state_root: Path | str | None = None) -> dict[str, Any]:
    root = _resolved_state_root(state_root)
    rows: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not (path / EVENTS_FILENAME).exists():
                continue
            try:
                snapshot = _resume_workflow(root, path.name, activate=False)
            except (FileNotFoundError, ValueError) as exc:
                rows.append({"workflow_id": path.name, "error": str(exc)})
            else:
                rows.append(
                    {
                        "workflow_id": path.name,
                        "goal": snapshot.get("goal"),
                        "phase": snapshot.get("phase"),
                        "next_action": snapshot.get("next_action"),
                    }
                )
    return {"schema_version": RESUME_SCHEMA_VERSION, "workflows": rows}


def _print_result(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    if payload.get("approval_id") and isinstance(payload.get("resume"), dict):
        print(f"approval: {payload['approval_id']}")
        payload = payload["resume"]
    if "workflows" in payload:
        for workflow in payload["workflows"]:
            print(
                f"{workflow.get('workflow_id')}: {workflow.get('phase', 'unknown')}"
                f" -> {workflow.get('next_action', workflow.get('error', 'inspect'))}"
            )
        return
    print(f"workflow: {payload.get('workflow_id')}")
    print(f"goal: {payload.get('goal')}")
    print(f"phase: {payload.get('phase')}")
    print(f"next action: {payload.get('next_action')}")
    if payload.get("choices"):
        print(f"choices: {json.dumps(payload['choices'], sort_keys=True)}")
    for run in payload.get("runs") or []:
        print(f"  {run.get('run_id', run.get('run_dir'))}: {run.get('phase')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m suite_tools.companion",
        description=(
            "Track prompt-free workflow coordination while deriving run state from "
            "authoritative benchmark ledgers. Makes no provider calls."
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=default_state_root(),
        help="Ignored local companion directory (default: .benchmark-companion).",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    start = sub.add_parser("start", help="Start a workflow and optionally attach runs.")
    start.add_argument("workflow_id")
    start.add_argument("--goal", required=True, choices=sorted(GOALS))
    start.add_argument("--run", dest="runs", action="append", type=Path, default=[])
    start.add_argument("--json", dest="as_json", action="store_true")

    attach = sub.add_parser("attach", help="Attach another immutable run contract.")
    attach.add_argument("workflow_id")
    attach.add_argument("--run", required=True, type=Path)
    attach.add_argument("--json", dest="as_json", action="store_true")

    choose = sub.add_parser("choose", help="Record one allowlisted onboarding choice.")
    choose.add_argument("workflow_id")
    choose.add_argument("--key", required=True, choices=sorted(CHOICES))
    choose.add_argument("--value", required=True)
    choose.add_argument("--json", dest="as_json", action="store_true")

    resume = sub.add_parser("resume", help="Rebuild a resume receipt from current ledgers.")
    resume.add_argument("workflow_id", nargs="?", default=None)
    resume.add_argument("--json", dest="as_json", action="store_true")

    approve = sub.add_parser("approve", help="Record exact, single-use user approval.")
    approve.add_argument("workflow_id")
    approve.add_argument("--run", required=True, type=Path)
    approve.add_argument("--stage", required=True, choices=sorted(APPROVAL_STAGES))
    approve.add_argument(
        "--confirmed-by-user",
        action="store_true",
        help="Required assertion that the user explicitly approved this exact scope.",
    )
    approve.add_argument("--json", dest="as_json", action="store_true")

    consume = sub.add_parser(
        "consume",
        help="Consume an approval immediately before its external operation.",
    )
    consume.add_argument("workflow_id")
    consume.add_argument("--approval", required=True)
    consume.add_argument("--json", dest="as_json", action="store_true")

    listing = sub.add_parser("list", help="List local companion workflows.")
    listing.add_argument("--json", dest="as_json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.verb == "start":
            result = start_workflow(
                args.state_root,
                workflow_id=args.workflow_id,
                goal=args.goal,
                run_dirs=args.runs,
            )
        elif args.verb == "attach":
            result = attach_run(
                args.state_root,
                args.workflow_id,
                run_dir=args.run,
            )
        elif args.verb == "choose":
            result = record_choice(
                args.state_root,
                args.workflow_id,
                key=args.key,
                value=args.value,
            )
        elif args.verb == "resume":
            result = resume_workflow(args.state_root, args.workflow_id)
        elif args.verb == "approve":
            result = grant_approval(
                args.state_root,
                args.workflow_id,
                run_dir=args.run,
                stage=args.stage,
                confirmed_by_user=args.confirmed_by_user,
            )
        elif args.verb == "consume":
            result = consume_approval(
                args.state_root,
                args.workflow_id,
                approval_id=args.approval,
            )
        else:
            result = list_workflows(args.state_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_result(result, as_json=getattr(args, "as_json", False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
