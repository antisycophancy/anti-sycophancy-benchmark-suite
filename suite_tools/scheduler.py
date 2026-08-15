"""Process-level scheduler for prepared benchmark run contracts.

This module is intentionally one layer above the benchmark runners. It does
not replace AITA/Epis/SUS concurrency internals. It coordinates those runners
through shared paid-call lease settings and durable process ledgers. Its job is
to make prepared runs operator-visible and agent-operable by writing durable
scheduler ledgers beside the normal ``RUN_CONTRACT.json`` /
``RUN_STATUS.json`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from suite_tools.run_contract import (
    CLEAR_CONTROL,
    CONTRACT_FILENAME,
    CONTROL_FILENAME,
    LEGACY_IDENTITY_PROJECTION_VERSION,
    STOP_BEFORE_NEXT_PAID_CALL,
    PreparedConfigProvenanceError,
    PreparedPricingProvenanceError,
    load_run_contract,
    load_run_control,
    provenance_hashes_for_version,
    summarize_contract,
    summarize_control,
    stable_json_hash,
    validate_run_pricing_before_spend,
    validate_run_prepared_config_before_spend,
    write_run_control,
)
from suite_tools.env import load_repo_env_files
from suite_tools.openrouter_preflight import fetch_key_info, sanitize_key_info
from suite_tools.preflight_conditions import (
    PREFLIGHT_RECEIPT_TTL_SECONDS,
    PreflightReceiptValidationError,
    validate_preflight_receipt_before_spend,
)
from suite_tools.run_monitor import (
    atomic_write_json,
    sanitize_error_message,
    sanitize_ledger_value,
    utc_now,
)
from suite_tools.progress_dedupe import (
    active_unit_count,
    completed_unit_keys,
    event_unit_key,
    SCORING_COMPLETED_EVENTS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEDULER_STATUS_FILENAME = "SCHEDULER_STATUS.json"
SCHEDULER_EVENTS_FILENAME = "SCHEDULER_EVENTS.jsonl"
SCHEDULER_LOCK_FILENAME = "SCHEDULER_LOCK.json"
SCHEDULER_SCHEMA_VERSION = "benchmark-scheduler-v1"
SCHEDULER_LOCK_SCHEMA_VERSION = "benchmark-scheduler-lock-v1"
DUPLICATE_SCHEDULER_EXIT_CODE = 75
CHILD_TERMINATION_GRACE_SECONDS = 5.0
ARBITRARY_COMMANDS_ENV = "BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS"
PREPARED_COMMANDS_SCHEMA_VERSION = "benchmark-prepared-commands-v1"

_MODULE_COMMAND_POLICY = {
    "sus": {
        "python_module": "sus_bench",
        "cwd": REPO_ROOT / "sus-bench",
        "execute": frozenset({"run"}),
        "score": frozenset({"score"}),
    },
    "aita": {
        "python_module": "aita_bench",
        "cwd": REPO_ROOT / "aita-bench",
        "execute": frozenset({"run"}),
        "score": frozenset({"score", "report"}),
    },
    "epistemic": {
        "python_module": "epis_bench",
        "cwd": REPO_ROOT / "epistemic-sycophancy-bench",
        "execute": frozenset({"run"}),
        "score": frozenset({"score", "report"}),
    },
}

DEFAULT_RUN_PACE = "normal"
DEFAULT_MAX_CONTRACT_WORKERS = 8
RUN_PACE_PRESETS: dict[str, dict[str, Any]] = {
    "cautious": {
        "max_active_calls": 2,
        "stagger_start_seconds": 2.0,
        "summary": "Small paid probes, expensive judges, new endpoints, or unknown provider quotas.",
    },
    "normal": {
        "max_active_calls": 4,
        "stagger_start_seconds": 1.0,
        "summary": "Default public run posture for comparable module-by-module execution.",
    },
    "fast": {
        "max_active_calls": 6,
        "stagger_start_seconds": 0.5,
        "summary": "Cheap models or already-proven provider quotas with the dashboard open.",
    },
    "full-speed": {
        "max_active_calls": 8,
        "stagger_start_seconds": 0.0,
        "summary": "Operator-monitored throughput test only; avoid as the default publication posture.",
    },
}

TERMINAL_SCHEDULER_STATES = {
    "queued",
    "completed",
    "score_ready",
    "needs_scoring",
    "attention",
    "stopped",
    "dry_run",
}


@dataclass(frozen=True)
class CommandStep:
    """One prepared command step."""

    cwd: Path
    argv: tuple[str, ...] | None = None
    shell_command: str | None = None

    @property
    def display(self) -> str:
        if self.argv is not None:
            return " ".join(str(item) for item in self.argv)
        return str(self.shell_command or "")

    @property
    def is_shell(self) -> bool:
        return self.argv is None


class SchedulerAlreadyRunning(RuntimeError):
    """Raised when another live scheduler owns this prepared contract."""

    def __init__(self, lock: dict[str, Any]) -> None:
        self.lock = lock
        pid = lock.get("pid")
        scheduler_id = lock.get("scheduler_id")
        detail = f"pid {pid}" if pid else "unknown pid"
        if scheduler_id:
            detail = f"{scheduler_id} ({detail})"
        super().__init__(f"Scheduler already running for this contract: {detail}")


class PreparedCommandProvenanceError(ValueError):
    """Raised when a prepared contract cannot authenticate a safe command."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(sorted(set(issues)))
        super().__init__(
            "prepared command provenance invalid: " + "; ".join(self.issues)
        )


def _path_for(path_or_dir: Path | str, filename: str) -> Path:
    path = Path(path_or_dir)
    return path if path.name == filename else path / filename


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def run_pace_payload() -> dict[str, Any]:
    return {
        "default": DEFAULT_RUN_PACE,
        "presets": {
            name: {
                "max_active_calls": preset["max_active_calls"],
                "stagger_start_seconds": preset["stagger_start_seconds"],
                "summary": preset["summary"],
            }
            for name, preset in RUN_PACE_PRESETS.items()
        },
    }


def print_run_paces(*, output_json: bool = False) -> None:
    payload = run_pace_payload()
    if output_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"Default run pace: {payload['default']}")
    for name, preset in payload["presets"].items():
        print(
            f"- {name}: max_active_calls={preset['max_active_calls']}, "
            f"stagger_start_seconds={preset['stagger_start_seconds']} - {preset['summary']}"
        )


def resolve_run_pace(
    *,
    run_pace: str | None,
    max_active_calls: int | None,
    stagger_start_seconds: float | None = None,
) -> tuple[str, int, float]:
    pace_name = run_pace or DEFAULT_RUN_PACE
    if pace_name not in RUN_PACE_PRESETS:
        valid = ", ".join(RUN_PACE_PRESETS)
        raise ValueError(f"Unknown run pace {pace_name!r}; expected one of: {valid}")
    preset = RUN_PACE_PRESETS[pace_name]
    resolved_max_active_calls = (
        max_active_calls
        if max_active_calls is not None
        else int(preset["max_active_calls"])
    )
    resolved_stagger = (
        stagger_start_seconds
        if stagger_start_seconds is not None
        else float(preset["stagger_start_seconds"])
    )
    return pace_name, resolved_max_active_calls, resolved_stagger


def _pid_is_running(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_is_live(lock: dict[str, Any]) -> bool:
    return (
        lock.get("schema_version") == SCHEDULER_LOCK_SCHEMA_VERSION
        and _pid_is_running(lock.get("pid"))
    )


def _lock_is_recent(path: Path, *, max_age_seconds: float = 60.0) -> bool:
    try:
        age_seconds = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age_seconds < max_age_seconds


def acquire_scheduler_lock(
    state_dir: Path,
    *,
    scheduler_id: str,
    contract_path: Path,
    command: str,
) -> Path:
    """Create an atomic local lease so one contract is not launched twice."""
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / SCHEDULER_LOCK_FILENAME
    payload = sanitize_ledger_value(
        {
            "schema_version": SCHEDULER_LOCK_SCHEMA_VERSION,
            "scheduler_id": scheduler_id,
            "pid": os.getpid(),
            "created_at": utc_now(),
            "contract_path": _display_path(contract_path),
            "command": command,
        }
    )
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _load_json(lock_path)
            if _lock_is_live(existing) or (not existing and _lock_is_recent(lock_path)):
                if not existing:
                    existing = {
                        "schema_version": SCHEDULER_LOCK_SCHEMA_VERSION,
                        "scheduler_id": "recent-unreadable-lock",
                        "pid": None,
                        "contract_path": _display_path(contract_path),
                    }
                raise SchedulerAlreadyRunning(existing)
            # Atomic takeover: claim the apparently-stale lock by renaming it
            # to a unique path. Only one challenger can win the rename, and no
            # challenger ever unlinks the shared path directly — unlinking it
            # could destroy a fresh lock another scheduler created after our
            # read (the old double-spend window).
            claim_path = lock_path.with_name(
                f"{lock_path.name}.claim-{uuid.uuid4().hex[:8]}"
            )
            try:
                os.rename(lock_path, claim_path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise SchedulerAlreadyRunning(existing) from exc
            claimed = _load_json(claim_path)
            if _lock_is_live(claimed):
                # Raced: the stale lock was replaced by a live one between our
                # read and our rename. Restore it without clobbering anything
                # newer (link fails if the path is occupied), then defer.
                try:
                    os.link(claim_path, lock_path)
                except OSError:
                    pass
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
                raise SchedulerAlreadyRunning(claimed)
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload, indent=2, default=str) + "\n")
        return lock_path


def release_scheduler_lock(lock_path: Path, *, scheduler_id: str) -> None:
    """Release a scheduler lease if it still belongs to this scheduler."""
    lock = _load_json(lock_path)
    if lock.get("scheduler_id") != scheduler_id:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def load_scheduler_status(path_or_dir: Path | str) -> dict[str, Any]:
    """Load ``SCHEDULER_STATUS.json`` from a file or containing directory."""
    return _load_json(_path_for(path_or_dir, SCHEDULER_STATUS_FILENAME))


def load_scheduler_events(path_or_dir: Path | str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load scheduler events from ``SCHEDULER_EVENTS.jsonl``."""
    path = _path_for(path_or_dir, SCHEDULER_EVENTS_FILENAME)
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    if limit is not None:
        lines = lines[-limit:]
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _display_path(path: Path | None, *, base: Path = REPO_ROOT) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _seconds_between(start: Any, end: Any) -> float | None:
    start_dt = _parse_time(start)
    end_dt = _parse_time(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("event") or "")
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _completed_units_from_events(events: list[dict[str, Any]]) -> int:
    """Count completed work units without double-counting stages or attempts.

    Generation events and scoring events describe the same unit at different
    stages, so the families are counted separately and the max is reported. A
    unit that completed generation and was then scored counts once.

    Within each family, units are deduplicated by canonical ``unit_id`` (or a
    fallback key) so that an attempt-2 reuse/completion of a unit already
    counted in attempt-1 does not inflate the total.
    """
    generated = len(completed_unit_keys(events))
    scored = len({
        event_unit_key(event)
        for event in events
        if isinstance(event, dict) and str(event.get("event") or "") in SCORING_COMPLETED_EVENTS
    })
    return max(generated, scored)


def _active_units_from_events(events: list[dict[str, Any]]) -> int:
    """Count units currently in flight (started but not yet finished).

    Delegates to ``active_unit_count`` which applies set-based dedup for events
    with ``unit_id`` (so a retried unit emitting a second ``conversation_started``
    does not inflate the count) while preserving count-based behaviour for
    legacy streams that have no identity fields.
    """
    return active_unit_count(events)


def _module_status_path(contract_path: Path) -> Path:
    return contract_path.parent / "RUN_STATUS.json"


def _module_events_path(contract_path: Path) -> Path:
    return contract_path.parent / "RUN_EVENTS.jsonl"


def _load_module_events(contract_path: Path) -> list[dict[str, Any]]:
    path = _module_events_path(contract_path)
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _resolve_step_cwd(value: Any) -> Path:
    if value in {None, ""}:
        return REPO_ROOT
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _argv_from_value(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    argv = tuple(str(item) for item in value)
    return argv if all(item for item in argv) else None


def _contract_module_key(contract: dict[str, Any]) -> str | None:
    modules = {
        str(module.get("module") or "").strip().lower()
        for module in contract.get("modules") or []
        if isinstance(module, dict) and module.get("module")
    }
    if len(modules) != 1:
        return None
    module = modules.pop()
    return "epistemic" if module == "epis" else module


def _prepared_command_binding(contract: dict[str, Any]) -> dict[str, Any] | None:
    identity = contract.get("identity")
    execution = identity.get("execution") if isinstance(identity, dict) else None
    binding = execution.get("prepared_commands") if isinstance(execution, dict) else None
    return binding if isinstance(binding, dict) else None


def _cli_options(argv: tuple[str, ...]) -> dict[str, list[str | None]]:
    options: dict[str, list[str | None]] = {}
    index = 4
    while index < len(argv):
        token = argv[index]
        if not token.startswith("-") or token == "-":
            index += 1
            continue
        if "=" in token:
            name, value = token.split("=", 1)
            options.setdefault(name, []).append(value)
            index += 1
            continue
        value = None
        if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            value = argv[index + 1]
            index += 1
        options.setdefault(token, []).append(value)
        index += 1
    return options


def _resolve_contract_cli_path(raw: str, *, label: str) -> tuple[Path | None, str | None]:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return None, f"{label} must be an absolute path"
    if ".." in path.parts:
        return None, f"{label} contains relative traversal"
    try:
        return path.resolve(), None
    except (OSError, RuntimeError, ValueError):
        return None, f"{label} is invalid"


def _expected_prepared_config_path(
    contract: dict[str, Any],
    contract_path: Path,
) -> Path | None:
    identity = contract.get("identity")
    execution = identity.get("execution") if isinstance(identity, dict) else None
    binding = execution.get("prepared_config") if isinstance(execution, dict) else None
    raw_path = binding.get("path") if isinstance(binding, dict) else None
    if not isinstance(raw_path, str) or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
        return None
    try:
        return (contract_path.parent.parent / raw_path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None


def _path_within_any_root(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _sealed_pack_binding(contract: dict[str, Any]) -> dict[str, Any] | None:
    for module in contract.get("modules") or []:
        if not isinstance(module, dict) or module.get("module") != "aita":
            continue
        dataset_manifest = module.get("dataset_manifest")
        if not isinstance(dataset_manifest, dict):
            continue
        if dataset_manifest.get("distribution_mode") != "sealed_public_pack":
            continue
        binding = dataset_manifest.get("sealed_pack")
        return binding if isinstance(binding, dict) else {}
    return None


def _sealed_pack_path_issues(
    raw: Any,
    *,
    label: str,
    frozen: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if not isinstance(raw, str) or not raw:
        return [f"{label} --sealed-pack path is missing"]
    envelope_path, error = _resolve_contract_cli_path(
        raw,
        label=f"{label} --sealed-pack",
    )
    if error:
        return [error]
    assert envelope_path is not None
    if envelope_path.is_symlink() or not envelope_path.is_file():
        return [f"{label} --sealed-pack must be a regular non-symlink file"]
    try:
        if envelope_path.stat().st_size > 128 * 1024:
            raise ValueError("envelope is too large")
        envelope = json.loads(
            envelope_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return [f"{label} sealed pack envelope is unreadable"]
    if not isinstance(envelope, dict):
        return [f"{label} sealed pack envelope must be a JSON object"]

    for envelope_key, frozen_key in (
        ("pack_id", "pack_id"),
        ("pack_version", "pack_version"),
        ("pair_count", "pair_count"),
        ("ciphertext_sha256", "ciphertext_sha256"),
        ("plaintext_identity_sha256", "plaintext_identity_sha256"),
        ("key_scheme", "key_scheme"),
    ):
        if envelope.get(envelope_key) != frozen.get(frozen_key):
            issues.append(
                f"{label} sealed pack {envelope_key} differs from the frozen dataset identity"
            )

    ciphertext_name = envelope.get("ciphertext_file")
    if (
        not isinstance(ciphertext_name, str)
        or not ciphertext_name
        or Path(ciphertext_name).name != ciphertext_name
        or ciphertext_name in {".", ".."}
        or any(character in ciphertext_name for character in "\\/\r\n\x00")
    ):
        issues.append(f"{label} sealed pack ciphertext_file is unsafe")
        return issues
    ciphertext_path = envelope_path.parent / ciphertext_name
    if ciphertext_path.is_symlink() or not ciphertext_path.is_file():
        issues.append(f"{label} sealed pack ciphertext must be a regular non-symlink file")
        return issues
    try:
        size = ciphertext_path.stat().st_size
        if size > 64 * 1024 * 1024:
            raise ValueError("ciphertext is too large")
        ciphertext = ciphertext_path.read_bytes()
    except (OSError, ValueError):
        issues.append(f"{label} sealed pack ciphertext is unreadable")
        return issues
    if envelope.get("ciphertext_bytes") != len(ciphertext):
        issues.append(f"{label} sealed pack ciphertext length mismatch")
    if hashlib.sha256(ciphertext).hexdigest() != envelope.get("ciphertext_sha256"):
        issues.append(f"{label} sealed pack ciphertext digest mismatch")
    return issues


def _command_path_issues(
    argv: tuple[str, ...],
    *,
    label: str,
    module_key: str,
    contract: dict[str, Any],
    contract_path: Path,
) -> list[str]:
    issues: list[str] = []
    command = argv[3]
    options = _cli_options(argv)
    module_dir = contract_path.parent.resolve()
    run_group_dir = module_dir.parent.resolve()
    trusted_data_roots = (REPO_ROOT.resolve(), run_group_dir)

    for name in options:
        if (
            (name.startswith("--output") and name != "--output")
            or (name.startswith("--input") and name != "--input")
            or (name.startswith("--config") and name != "--config")
        ):
            issues.append(f"{label} uses unsupported path option {name}")

    forbidden_identity_overrides = {
        "--api-key",
        "--base-url",
        "--judge-base-url",
        "--model",
        "--analyzer-model",
        "--judge-model",
        "--judge-panel",
        "--temperature",
        "--reasoning",
    }
    for name in sorted(forbidden_identity_overrides.intersection(options)):
        issues.append(f"{label} may not override frozen identity with {name}")

    def require_exact_path(
        names: tuple[str, ...],
        expected: Path,
        *,
        description: str,
        required: bool,
    ) -> None:
        supplied = [
            (name, raw)
            for name in names
            for raw in options.get(name, [])
        ]
        if not supplied:
            if required:
                issues.append(f"{label} is missing required {description} path")
            return
        if len(supplied) != 1:
            issues.append(f"{label} must contain exactly one {description} path")
            return
        name, raw = supplied[0]
        if not isinstance(raw, str) or not raw:
            issues.append(f"{label} {name} path is missing")
            return
        resolved, error = _resolve_contract_cli_path(raw, label=f"{label} {name}")
        if error:
            issues.append(error)
        elif resolved != expected:
            issues.append(f"{label} {name} must resolve exactly to {expected}")

    if command == "run":
        require_exact_path(
            ("--output", "-o"),
            module_dir,
            description="output",
            required=True,
        )
    elif command == "score":
        require_exact_path(
            ("--input", "-i"),
            module_dir,
            description="input",
            required=True,
        )
        if module_key == "sus":
            require_exact_path(
                ("--output", "-o"),
                module_dir / "FINAL_RESULTS.json",
                description="output",
                required=True,
            )
    elif command == "report" and module_key in {"aita", "epistemic"}:
        require_exact_path(
            ("--input", "-i"),
            module_dir,
            description="input",
            required=True,
        )

    expected_config = _expected_prepared_config_path(contract, contract_path)
    config_names = ("--models",) if module_key == "sus" else ("--config",)
    if command != "report" or module_key != "sus":
        if expected_config is None:
            issues.append(f"{label} cannot authenticate the prepared config path")
        else:
            require_exact_path(
                config_names,
                expected_config,
                description="config",
                required=True,
            )

    file_options = {
        "aita": ("--data", "--og-data", "--flip-data", "--paired-labels", "--item-selection"),
        "epistemic": ("--selection",),
    }.get(module_key, ())
    for name in file_options:
        for raw in options.get(name, []):
            if not isinstance(raw, str) or not raw:
                issues.append(f"{label} {name} path is missing")
                continue
            resolved, error = _resolve_contract_cli_path(raw, label=f"{label} {name}")
            if error:
                issues.append(error)
            elif not resolved.is_file():
                issues.append(f"{label} {name} must be a regular file")
            elif not _path_within_any_root(resolved, trusted_data_roots):
                issues.append(f"{label} {name} resolves outside trusted data roots")

    sealed_binding = _sealed_pack_binding(contract)
    sealed_values = options.get("--sealed-pack", [])
    if command == "run" and sealed_binding is not None:
        if len(sealed_values) != 1:
            issues.append(f"{label} must contain exactly one frozen --sealed-pack path")
        else:
            issues.extend(
                _sealed_pack_path_issues(
                    sealed_values[0],
                    label=label,
                    frozen=sealed_binding,
                )
            )
    elif sealed_values:
        issues.append(f"{label} may not supply an unfrozen --sealed-pack path")

    if module_key == "epistemic":
        for raw in options.get("--data-dir", []):
            if not isinstance(raw, str) or not raw:
                issues.append(f"{label} --data-dir path is missing")
                continue
            resolved, error = _resolve_contract_cli_path(
                raw,
                label=f"{label} --data-dir",
            )
            if error:
                issues.append(error)
            elif not resolved.is_dir():
                issues.append(f"{label} --data-dir must be a directory")
            elif not _path_within_any_root(resolved, trusted_data_roots):
                issues.append(f"{label} --data-dir resolves outside trusted data roots")
    return issues


def _command_step_issues(
    raw_steps: Any,
    *,
    prefix: str,
    policy: dict[str, Any],
    module_key: str,
    contract: dict[str, Any],
    contract_path: Path,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(raw_steps, list) or not raw_steps:
        return [f"{prefix}_steps missing or empty"]

    expected_executable = Path(sys.executable).resolve()
    expected_cwd = Path(policy["cwd"]).resolve()
    expected_module = str(policy["python_module"])
    allowed_commands = policy[prefix]
    for index, raw_step in enumerate(raw_steps):
        label = f"{prefix}_steps[{index}]"
        if not isinstance(raw_step, dict):
            issues.append(f"{label} is not an object")
            continue
        if set(raw_step) != {"cwd", "argv"}:
            issues.append(f"{label} must contain only cwd and argv")
        argv = _argv_from_value(raw_step.get("argv"))
        if argv is None:
            issues.append(f"{label}.argv is missing or invalid")
            continue
        if len(argv) < 4:
            issues.append(f"{label}.argv is not a benchmark module command")
            continue
        try:
            executable = Path(argv[0]).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            executable = Path(argv[0])
        if executable != expected_executable:
            issues.append(f"{label} executable is not the active Python interpreter")
        if argv[1] != "-m":
            issues.append(f"{label} must use Python -m")
        if argv[2] != expected_module:
            issues.append(f"{label} module must be {expected_module}")
        if argv[3] not in allowed_commands:
            issues.append(
                f"{label} subcommand {argv[3]!r} is not allowed for {prefix}"
            )
        else:
            issues.extend(
                _command_path_issues(
                    argv,
                    label=label,
                    module_key=module_key,
                    contract=contract,
                    contract_path=contract_path,
                )
            )
        try:
            cwd = _resolve_step_cwd(raw_step.get("cwd"))
        except (OSError, RuntimeError, ValueError):
            issues.append(f"{label}.cwd is invalid")
            continue
        if cwd != expected_cwd or not cwd.is_dir():
            issues.append(f"{label}.cwd is not the trusted {expected_module} source root")
    return issues


def validate_prepared_commands_before_execution(
    contract: dict[str, Any],
    contract_path: Path,
) -> bool:
    """Authenticate exact prepared steps and enforce the scheduler allowlist.

    Returns ``False`` only when the explicit unsafe compatibility override lets
    a legacy or test command continue. Current prepared contracts return True.
    """
    issues: list[str] = []
    binding = _prepared_command_binding(contract)
    if binding is None:
        issues.append("identity.execution.prepared_commands missing")
    else:
        if binding.get("schema_version") != PREPARED_COMMANDS_SCHEMA_VERSION:
            issues.append("prepared_commands schema_version is unsupported")

        stored_provenance = contract.get("provenance")
        if not isinstance(stored_provenance, dict) or not stored_provenance:
            issues.append("stored provenance missing")
        else:
            projection_version = str(
                stored_provenance.get("projection_version")
                or LEGACY_IDENTITY_PROJECTION_VERSION
            )
            try:
                expected_provenance = provenance_hashes_for_version(
                    contract,
                    projection_version,
                )
            except ValueError:
                issues.append("stored provenance projection is invalid")
            else:
                if expected_provenance != stored_provenance:
                    issues.append("stored provenance does not authenticate prepared_commands")

        for prefix in ("execute", "score"):
            frozen_steps = binding.get(f"{prefix}_steps")
            top_level_steps = contract.get(f"{prefix}_steps")
            if top_level_steps != frozen_steps:
                issues.append(f"top-level {prefix}_steps differ from prepared binding")
                continue
            if not isinstance(frozen_steps, list) or not frozen_steps:
                issues.append(f"prepared {prefix}_steps missing or empty")
                continue
            first_step = frozen_steps[0]
            if not isinstance(first_step, dict):
                issues.append(f"prepared {prefix}_steps[0] is not an object")
                continue
            if contract.get(f"{prefix}_cwd") != first_step.get("cwd"):
                issues.append(f"top-level {prefix}_cwd differs from prepared binding")
            if contract.get(f"{prefix}_argv") != first_step.get("argv"):
                issues.append(f"top-level {prefix}_argv differs from prepared binding")

        module_key = _contract_module_key(contract)
        policy = _MODULE_COMMAND_POLICY.get(str(module_key or ""))
        if policy is None:
            issues.append("contract must declare exactly one supported benchmark module")
        else:
            issues.extend(
                _command_step_issues(
                    binding.get("execute_steps"),
                    prefix="execute",
                    policy=policy,
                    module_key=str(module_key),
                    contract=contract,
                    contract_path=contract_path,
                )
            )
            issues.extend(
                _command_step_issues(
                    binding.get("score_steps"),
                    prefix="score",
                    policy=policy,
                    module_key=str(module_key),
                    contract=contract,
                    contract_path=contract_path,
                )
            )

    if not issues:
        return True
    if os.getenv(ARBITRARY_COMMANDS_ENV) == "1":
        print(
            "WARNING: unsafe arbitrary contract commands enabled via "
            f"{ARBITRARY_COMMANDS_ENV}=1: " + "; ".join(sorted(set(issues))),
            file=sys.stderr,
        )
        return False
    raise PreparedCommandProvenanceError(issues)


def _structured_steps_from_contract(contract: dict[str, Any], prefix: str) -> list[CommandStep]:
    raw_steps = contract.get(f"{prefix}_steps")
    steps: list[CommandStep] = []
    if isinstance(raw_steps, list):
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            argv = _argv_from_value(item.get("argv"))
            if argv is None:
                continue
            steps.append(CommandStep(cwd=_resolve_step_cwd(item.get("cwd")), argv=argv))
        if steps:
            return steps

    argv = _argv_from_value(contract.get(f"{prefix}_argv"))
    if argv is not None:
        return [CommandStep(cwd=_resolve_step_cwd(contract.get(f"{prefix}_cwd")), argv=argv)]
    return []


def _legacy_shell_steps_from_contract(contract: dict[str, Any], keys: tuple[str, ...]) -> list[CommandStep]:
    if os.getenv("BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS") != "1":
        return []
    for key in keys:
        value = contract.get(key)
        if isinstance(value, str) and value.strip():
            return [CommandStep(cwd=REPO_ROOT, shell_command=value.strip())]
    return []


def _execute_steps_from_contract(contract: dict[str, Any]) -> list[CommandStep]:
    return _structured_steps_from_contract(contract, "execute") or _legacy_shell_steps_from_contract(
        contract,
        ("execute_command",),
    )


def _score_steps_from_contract(contract: dict[str, Any], *, force: bool = False) -> list[CommandStep]:
    return _structured_steps_from_contract(contract, "score") or _legacy_shell_steps_from_contract(
        contract,
        ("score_command", "scoring_command"),
    )


def _display_steps(steps: list[CommandStep]) -> str:
    if not steps:
        return ""
    if len(steps) == 1 and steps[0].is_shell:
        return steps[0].display
    lines: list[str] = []
    current_cwd: Path | None = None
    for step in steps:
        if current_cwd != step.cwd:
            lines.append(f"cd {step.cwd}")
            current_cwd = step.cwd
        lines.append(step.display)
    return "\n".join(lines)


def classify_observed_state(status: dict[str, Any], *, returncode: int | None = None) -> str:
    """Classify runner ledger state for scheduler flow lanes."""
    status_text = str(status.get("status") or "")
    validity = str(status.get("validity") or "")
    stage = str(status.get("stage") or "")
    if status_text == "stopped" or returncode == 130:
        return "stopped"
    if status_text.startswith("failed"):
        return "attention"
    if status_text == "completed" and validity == "score_ready":
        return "score_ready"
    if status_text == "completed" and validity == "not_score_ready" and stage == "generation":
        return "needs_scoring"
    if status_text == "completed":
        return "completed"
    if returncode is not None and returncode != 0:
        return "attention"
    if status_text == "running":
        return "running"
    return "running" if returncode is None else "attention"


def build_progress_snapshot(
    *,
    contract_summary: dict[str, Any],
    module_status: dict[str, Any],
    module_events: list[dict[str, Any]],
    scheduler_started_at: str | None,
    max_active_calls: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build progress and ETA data from contract + runner ledgers."""
    now_dt = now or datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    expected_units = int(contract_summary.get("expected_units") or 0)
    event_completed = _completed_units_from_events(module_events)
    contract_completed = int(contract_summary.get("complete_units") or 0)
    observed_state = classify_observed_state(module_status)
    if observed_state in {"score_ready", "completed"} and expected_units:
        completed_units = expected_units
    else:
        completed_units = min(
            expected_units,
            max(contract_completed, event_completed),
        ) if expected_units else max(contract_completed, event_completed)

    remaining_units = max(0, expected_units - completed_units) if expected_units else None
    active_units = _active_units_from_events(module_events)
    if observed_state in {"needs_scoring", "score_ready", "completed"}:
        active_units = 0
    effective_parallelism = max(1, active_units or min(max_active_calls or 1, expected_units or 1))
    elapsed_seconds = _seconds_between(scheduler_started_at, now_iso)
    avg_completed_seconds = None
    eta_seconds = None
    if elapsed_seconds is not None and completed_units > 0:
        avg_completed_seconds = elapsed_seconds / completed_units
        if remaining_units is not None:
            eta_seconds = (avg_completed_seconds * remaining_units) / effective_parallelism

    percent = None
    if expected_units:
        percent = round(min(100.0, (completed_units / expected_units) * 100), 1)

    latest_event = module_events[-1] if module_events else None
    return {
        "expected_units": expected_units,
        "completed_units": completed_units,
        "remaining_units": remaining_units,
        "active_units": active_units,
        "effective_parallelism": effective_parallelism,
        "percent": percent,
        "average_completed_unit_seconds": (
            round(avg_completed_seconds, 3) if avg_completed_seconds is not None else None
        ),
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "eta_basis": (
            "completed-unit-average"
            if eta_seconds is not None
            else "pending-first-completed-unit"
        ),
        "latest_event": {
            "event": latest_event.get("event"),
            "timestamp": latest_event.get("timestamp"),
            "sequence": latest_event.get("sequence"),
        } if latest_event else None,
    }


class SchedulerLedger:
    """Append-only scheduler events plus latest status snapshot."""

    def __init__(
        self,
        state_dir: Path,
        *,
        contract_path: Path,
        contract: dict[str, Any],
        settings: dict[str, Any],
        scheduler_id: str | None = None,
        lock_path: Path | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.state_dir / SCHEDULER_STATUS_FILENAME
        self.events_path = self.state_dir / SCHEDULER_EVENTS_FILENAME
        self.scheduler_id = scheduler_id or f"scheduler-{uuid.uuid4().hex[:12]}"
        self.contract_path = contract_path
        self.contract = contract
        self.settings = settings
        self._sequence = self._existing_event_count()
        self.status: dict[str, Any] = {
            "schema_version": SCHEDULER_SCHEMA_VERSION,
            "scheduler_id": self.scheduler_id,
            "state": "queued",
            "run_id": contract.get("run_id") or contract_path.parent.name,
            "contract_path": _display_path(contract_path),
            "state_dir": _display_path(state_dir),
            "command": _display_steps(_execute_steps_from_contract(contract)),
            "score_command": _display_steps(_score_steps_from_contract(contract)),
            "settings": settings,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "progress": {},
            "process": {},
            "lock": {
                "path": _display_path(lock_path),
            } if lock_path else {},
        }
        self.refresh("queued")

    def _existing_event_count(self) -> int:
        if not self.events_path.exists():
            return 0
        try:
            return sum(1 for _ in self.events_path.open())
        except OSError:
            return 0

    def event(self, event: str, **fields: Any) -> None:
        self._sequence += 1
        payload = sanitize_ledger_value(
            {
                "schema_version": SCHEDULER_SCHEMA_VERSION,
                "sequence": self._sequence,
                "timestamp": utc_now(),
                "scheduler_id": self.scheduler_id,
                "event": event,
                **fields,
            }
        )
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def refresh(
        self,
        state: str | None = None,
        *,
        process: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        contract_summary = summarize_contract(
            self.contract,
            contract_path=self.contract_path,
            results_root=self.contract_path.parent,
        )
        module_status = _load_json(_module_status_path(self.contract_path))
        module_events = _load_module_events(self.contract_path)
        observed_state = classify_observed_state(module_status)
        next_state = state or ("attention" if observed_state == "attention" else observed_state)
        self.status.update(
            {
                "state": next_state,
                "updated_at": utc_now(),
                "contract": {
                    "expected_units": contract_summary.get("expected_units") or 0,
                    "complete_units": contract_summary.get("complete_units") or 0,
                    "progress_percent": contract_summary.get("progress_percent"),
                    "fingerprint": contract_summary.get("fingerprint"),
                    "attention": contract_summary.get("attention"),
                },
                "runner": {
                    "status": module_status.get("status"),
                    "stage": module_status.get("stage"),
                    "validity": module_status.get("validity"),
                    "updated_at": module_status.get("updated_at"),
                    "failure_reason": module_status.get("failure_reason"),
                },
                "progress": build_progress_snapshot(
                    contract_summary=contract_summary,
                    module_status=module_status,
                    module_events=module_events,
                    scheduler_started_at=self.status.get("started_at") or self.status.get("created_at"),
                    max_active_calls=self.settings.get("max_active_calls"),
                ),
                "control": summarize_control(
                    load_run_control(self.contract_path.parent / CONTROL_FILENAME),
                    control_path=self.contract_path.parent / CONTROL_FILENAME,
                ),
            }
        )
        if reason:
            self.status["reason"] = reason
        if process is not None:
            self.status["process"] = process
        if next_state in TERMINAL_SCHEDULER_STATES and next_state not in {"queued", "dry_run"}:
            self.status.setdefault("completed_at", utc_now())
        atomic_write_json(self.status_path, self.status)


def _resolve_contract(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / CONTRACT_FILENAME
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"RUN_CONTRACT.json not found: {path}")
    return path


def _contract_has_rendered_config(contract: dict[str, Any]) -> bool:
    has_artifact = any(
        isinstance(artifact, dict) and artifact.get("kind") == "rendered_models"
        for module in contract.get("modules") or []
        if isinstance(module, dict)
        for artifact in module.get("expected_artifacts") or []
    )
    identity = contract.get("identity")
    execution = identity.get("execution") if isinstance(identity, dict) else None
    has_binding = isinstance(execution, dict) and isinstance(
        execution.get("prepared_config"),
        dict,
    )
    return has_artifact or has_binding


def _preflight_receipt_policy(
    contract: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    if dry_run:
        return "not_enforced_dry_run"
    if _contract_has_rendered_config(contract):
        return "required_current_prepared"
    return "compatibility_bypass_legacy_or_runtime_owned"


def _admit_preflight_receipt(
    ledger: SchedulerLedger,
    *,
    emit_messages: bool,
) -> bool:
    """Gate a real spawn or record an explicit compatibility bypass."""
    if not _contract_has_rendered_config(ledger.contract):
        ledger.event(
            "preflight_receipt_compatibility_bypass",
            policy="legacy_or_runtime_owned_without_prepared_config",
            lifecycle_state=ledger.contract.get("lifecycle_state"),
        )
        return True
    try:
        admission = validate_preflight_receipt_before_spend(
            ledger.contract_path.parent,
            ttl_seconds=PREFLIGHT_RECEIPT_TTL_SECONDS,
        )
        loaded_provenance = ledger.contract.get("provenance")
        if (
            not isinstance(loaded_provenance, dict)
            or stable_json_hash(loaded_provenance)
            != admission["contract_provenance_fingerprint"]
        ):
            raise PreflightReceiptValidationError([
                "scheduler-loaded contract differs from current receipt provenance"
            ])
    except PreflightReceiptValidationError as exc:
        ledger.refresh("attention", reason=sanitize_error_message(exc))
        ledger.event(
            "scheduler_attention",
            reason="preflight_receipt_admission",
            provenance_issues=list(exc.issues),
        )
        if emit_messages:
            print(sanitize_error_message(exc), file=sys.stderr)
        return False
    ledger.event(
        "preflight_receipt_admitted",
        receipt_fingerprint=admission["receipt_fingerprint"],
        target_set_hash=admission["target_set_hash"],
        target_count=admission["target_count"],
        age_seconds=admission["age_seconds"],
        ttl_seconds=admission["ttl_seconds"],
    )
    return True


def _run_shell_command(
    command: str,
    *,
    cwd: Path,
    stream_output: bool = True,
    max_active_calls: int | None = None,
    benchmark_context: dict[str, Any] | None = None,
) -> subprocess.Popen:
    stdout = sys.stdout if stream_output else subprocess.DEVNULL
    stderr = sys.stderr if stream_output else subprocess.DEVNULL
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        # Legacy shell execution is reachable only through the explicit opt-in gate above.
        shell=True,  # nosec B602
        stdout=stdout,
        stderr=stderr,
        text=True,
        env=_child_env(max_active_calls=max_active_calls, benchmark_context=benchmark_context),
        start_new_session=True,
    )


def _child_env(
    *,
    max_active_calls: int | None = None,
    benchmark_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    lease_dir = env.get("BENCHMARK_PAID_CALL_LEASE_DIR")
    if lease_dir:
        lease_path = Path(lease_dir).expanduser()
        if not lease_path.is_absolute():
            lease_path = (REPO_ROOT / lease_path).resolve()
        env["BENCHMARK_PAID_CALL_LEASE_DIR"] = str(lease_path)
    if max_active_calls is not None:
        env["BENCHMARK_PAID_CALL_MAX_ACTIVE"] = str(max_active_calls)
        env["BENCHMARK_MAX_ACTIVE_CALLS"] = str(max_active_calls)
        env.setdefault("BENCHMARK_GENERATION_MAX_PARALLEL", str(max_active_calls))
        env.setdefault("BENCHMARK_SCORE_MAX_PARALLEL", str(max_active_calls))
    for key, value in (benchmark_context or {}).items():
        if value is None:
            continue
        text = str(value)
        if text:
            env[key] = text
    return env


def _run_command_step(
    step: CommandStep,
    *,
    stream_output: bool = True,
    max_active_calls: int | None = None,
    benchmark_context: dict[str, Any] | None = None,
) -> subprocess.Popen:
    if step.argv is None:
        return _run_shell_command(
            step.shell_command or "",
            cwd=step.cwd,
            stream_output=stream_output,
            max_active_calls=max_active_calls,
            benchmark_context=benchmark_context,
        )
    stdout = sys.stdout if stream_output else subprocess.DEVNULL
    stderr = sys.stderr if stream_output else subprocess.DEVNULL
    return subprocess.Popen(
        list(step.argv),
        cwd=str(step.cwd),
        shell=False,
        stdout=stdout,
        stderr=stderr,
        text=True,
        env=_child_env(max_active_calls=max_active_calls, benchmark_context=benchmark_context),
        start_new_session=True,
    )


def _signal_process_tree(process: subprocess.Popen, sig: signal.Signals) -> None:
    """Signal the isolated child process group, falling back to the child."""
    try:
        os.killpg(process.pid, sig)
    except (OSError, AttributeError):
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _terminate_process_tree(
    process: subprocess.Popen,
    *,
    grace_seconds: float = CHILD_TERMINATION_GRACE_SECONDS,
) -> None:
    """Stop an unfinished scheduler child and all descendants."""
    if process.poll() is not None:
        return
    _signal_process_tree(process, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_tree(process, signal.SIGKILL)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        return


def _benchmark_child_context(ledger: SchedulerLedger) -> dict[str, str]:
    """Return benchmark context env vars for child runner paid-call attribution."""
    modules = [module for module in ledger.contract.get("modules") or [] if isinstance(module, dict)]
    module = modules[0].get("module") if modules else None
    return {
        "BENCHMARK_RUN_ID": str(ledger.status.get("run_id") or ledger.contract.get("run_id") or ledger.contract_path.parent.name),
        "BENCHMARK_MODULE": str(module or ""),
        "BENCHMARK_OUTPUT_DIR": str(ledger.contract_path.parent.resolve()),
        "BENCHMARK_CONTRACT_PATH": str(ledger.contract_path.resolve()),
    }


def should_preflight_openrouter_key(command: str, contract: dict[str, Any]) -> bool:
    """Return true for prepared benchmark commands that will use OpenRouter by default."""
    command_text = str(command)
    if any(token in command_text for token in ("sus_bench", "aita_bench", "epis_bench")):
        return True
    return bool(contract.get("expected_judges") or contract.get("expected_models")) and (
        "OPENROUTER_API_KEY" in command_text
    )


def openrouter_key_limit_attention() -> str | None:
    """Return a scheduler blocker when the current OpenRouter key cap is exhausted."""
    load_repo_env_files()
    if not os.environ.get("OPENROUTER_API_KEY"):
        return None
    try:
        key_info = sanitize_key_info(fetch_key_info(timeout=10))
    except Exception:
        return None
    if not key_info.get("credit_limit_exhausted"):
        return None
    return (
        "OpenRouter key limit is exhausted before generation "
        f"(limit={key_info.get('limit')}, usage={key_info.get('usage')}, "
        f"remaining={key_info.get('limit_remaining')}). "
        "Increase or clear the key limit, or use a different key, then rerun."
    )


def _run_command_with_ledger(
    *,
    ledger: SchedulerLedger,
    steps: list[CommandStep],
    event_prefix: str,
    poll_seconds: float,
    stop_on_attention: bool,
    stream_command_output: bool,
    max_active_calls: int | None = None,
    ignore_attention_status: dict[str, Any] | None = None,
) -> int:
    if not steps:
        return 2

    control_requested = False
    for index, step in enumerate(steps):
        process = _run_command_step(
            step,
            stream_output=stream_command_output,
            max_active_calls=max_active_calls,
            benchmark_context=_benchmark_child_context(ledger),
        )
        try:
            ledger.status["started_at"] = ledger.status.get("started_at") or utc_now()
            start_event = f"{event_prefix}_started" if len(steps) == 1 else f"{event_prefix}_step_started"
            ledger.event(
                start_event,
                pid=process.pid,
                step_index=index + 1,
                step_count=len(steps),
                # This is ledger metadata, not a process-execution argument.
                shell=step.is_shell,  # nosec B604
                cwd=str(step.cwd),
            )
            ledger.refresh(
                "running" if event_prefix == "generation" else "scoring",
                process={"pid": process.pid, "returncode": None},
            )

            while True:
                returncode = process.poll()
                module_status = _load_json(_module_status_path(ledger.contract_path))
                observed = classify_observed_state(module_status, returncode=returncode)
                control = load_run_control(ledger.contract_path.parent / CONTROL_FILENAME)
                if (
                    stop_on_attention
                    and not control_requested
                    and returncode is None
                    and observed == "attention"
                    and module_status != ignore_attention_status
                ):
                    write_run_control(
                        ledger.contract_path.parent,
                        action=STOP_BEFORE_NEXT_PAID_CALL,
                        reason="scheduler stop_on_attention observed failed/not-scoreable runner state",
                        requested_by="scheduler",
                    )
                    ledger.event("control_requested", action=STOP_BEFORE_NEXT_PAID_CALL, reason="stop_on_attention")
                    control_requested = True
                if summarize_control(control).get("active") and not control_requested:
                    ledger.event("control_observed", action=control.get("action"), reason=control.get("reason"))
                    control_requested = True
                ledger.refresh(
                    "running" if event_prefix == "generation" and returncode is None else (
                        "scoring" if event_prefix == "scoring" and returncode is None else None
                    ),
                    process={"pid": process.pid, "returncode": returncode},
                )
                if returncode is not None:
                    completed_event = f"{event_prefix}_completed" if len(steps) == 1 else f"{event_prefix}_step_completed"
                    ledger.event(
                        completed_event,
                        returncode=returncode,
                        step_index=index + 1,
                        step_count=len(steps),
                    )
                    if returncode != 0:
                        return returncode
                    break
                time.sleep(max(0.1, poll_seconds))
        finally:
            if process.poll() is None:
                _terminate_process_tree(process)
                ledger.event(
                    f"{event_prefix}_interrupted",
                    pid=process.pid,
                    step_index=index + 1,
                    step_count=len(steps),
                )
                ledger.refresh(
                    "stopped",
                    process={"pid": process.pid, "returncode": process.poll()},
                    reason="scheduler interrupted while child process was active",
                )

    if len(steps) > 1:
        ledger.event(f"{event_prefix}_completed", returncode=0, step_count=len(steps))
    return 0


def _run_score_command_with_ledger(
    *,
    ledger: SchedulerLedger,
    score_steps: list[CommandStep],
    poll_seconds: float,
    stop_on_attention: bool,
    stream_command_output: bool,
    max_active_calls: int | None = None,
    ignore_attention_status: dict[str, Any] | None = None,
    emit_messages: bool = True,
) -> int:
    # Score commands can make fresh judge/analyzer calls. Re-check admission at
    # this separate spawn boundary because generation may have outlived the TTL.
    if not _admit_preflight_receipt(ledger, emit_messages=emit_messages):
        return 2
    score_returncode = _run_command_with_ledger(
        ledger=ledger,
        steps=score_steps,
        event_prefix="scoring",
        poll_seconds=poll_seconds,
        stop_on_attention=stop_on_attention,
        stream_command_output=stream_command_output,
        max_active_calls=max_active_calls,
        ignore_attention_status=ignore_attention_status,
    )
    final_status = _load_json(_module_status_path(ledger.contract_path))
    final_state = classify_observed_state(final_status, returncode=score_returncode)
    if final_state == "score_ready":
        ledger.refresh("score_ready", process={"returncode": score_returncode})
        return 0
    if final_state == "stopped":
        ledger.refresh("stopped", process={"returncode": score_returncode})
        return 130
    ledger.refresh("attention", process={"returncode": score_returncode})
    return 2


def run_contract(
    contract_path: Path,
    *,
    state_dir: Path | None = None,
    dry_run: bool = False,
    poll_seconds: float = 2.5,
    max_active_calls: int | None = None,
    stagger_start_seconds: float = 0.0,
    run_pace: str | None = None,
    stop_on_attention: bool = False,
    gate_after_generation: bool = True,
    auto_score_on_clean_generation: bool = False,
    emit_messages: bool = True,
    stream_command_output: bool = True,
) -> int:
    """Run a prepared contract's execute command and maintain scheduler ledgers."""
    contract = load_run_contract(contract_path)
    if not contract:
        raise ValueError(f"Invalid or empty contract: {contract_path}")
    target_state_dir = state_dir or contract_path.parent
    settings = {
        "dry_run": dry_run,
        "poll_seconds": poll_seconds,
        "max_active_calls": max_active_calls,
        "stagger_start_seconds": stagger_start_seconds,
        "run_pace": run_pace,
        "stop_on_attention": stop_on_attention,
        "gate_after_generation": gate_after_generation,
        "auto_score_on_clean_generation": auto_score_on_clean_generation,
        "preflight_receipt_policy": _preflight_receipt_policy(
            contract,
            dry_run=dry_run,
        ),
    }
    scheduler_id = f"scheduler-{uuid.uuid4().hex[:12]}"
    try:
        validate_prepared_commands_before_execution(contract, contract_path)
    except PreparedCommandProvenanceError as exc:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh("attention", reason=sanitize_error_message(exc))
        ledger.event(
            "scheduler_attention",
            reason="prepared_command_provenance",
            provenance_issues=list(exc.issues),
        )
        if emit_messages:
            print(sanitize_error_message(exc), file=sys.stderr)
        return 2
    try:
        validate_run_pricing_before_spend(contract_path)
    except PreparedPricingProvenanceError as exc:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh("attention", reason=sanitize_error_message(exc))
        ledger.event(
            "scheduler_attention",
            reason="prepared_pricing_provenance",
            provenance_issues=list(exc.issues),
        )
        if emit_messages:
            print(sanitize_error_message(exc), file=sys.stderr)
        return 2
    if _contract_has_rendered_config(contract):
        try:
            validate_run_prepared_config_before_spend(contract_path)
        except PreparedConfigProvenanceError as exc:
            ledger = SchedulerLedger(
                target_state_dir,
                contract_path=contract_path,
                contract=contract,
                settings=settings,
                scheduler_id=scheduler_id,
            )
            ledger.refresh("attention", reason=sanitize_error_message(exc))
            ledger.event(
                "scheduler_attention",
                reason="prepared_config_provenance",
                provenance_issues=list(exc.issues),
            )
            if emit_messages:
                print(sanitize_error_message(exc), file=sys.stderr)
            return 2
    execute_steps = _execute_steps_from_contract(contract)
    command = _display_steps(execute_steps)
    if not execute_steps:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh(
            "attention",
            reason=(
                "Contract does not contain structured execute argv. Legacy shell contracts "
                "require BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS=1."
            ),
        )
        ledger.event("scheduler_attention", reason="missing_structured_execute_argv")
        return 2
    if dry_run:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh("dry_run", reason="Dry run: command was not executed.")
        ledger.event(
            "dry_run_queued",
            command=command,
            preflight_receipt_policy="not_enforced_dry_run",
        )
        if emit_messages:
            print(f"Scheduler dry run queued: {_display_path(contract_path)}")
        return 0

    try:
        lock_path = acquire_scheduler_lock(
            target_state_dir,
            scheduler_id=scheduler_id,
            contract_path=contract_path,
            command=command,
        )
    except SchedulerAlreadyRunning as exc:
        if emit_messages:
            print(str(exc), file=sys.stderr)
        return DUPLICATE_SCHEDULER_EXIT_CODE

    ledger = SchedulerLedger(
        target_state_dir,
        contract_path=contract_path,
        contract=contract,
        settings=settings,
        scheduler_id=scheduler_id,
        lock_path=lock_path,
    )
    try:
        if not _admit_preflight_receipt(ledger, emit_messages=emit_messages):
            return 2
        observed_before = classify_observed_state(_load_json(_module_status_path(contract_path)))
        if observed_before == "score_ready":
            ledger.refresh("score_ready", reason="Run is already scored; no command was executed.")
            return 0
        if observed_before == "needs_scoring":
            score_steps = _score_steps_from_contract(contract)
            if auto_score_on_clean_generation and score_steps:
                return _run_score_command_with_ledger(
                    ledger=ledger,
                    score_steps=score_steps,
                    poll_seconds=poll_seconds,
                    stop_on_attention=stop_on_attention,
                    stream_command_output=stream_command_output,
                    max_active_calls=max_active_calls,
                    emit_messages=emit_messages,
                )
            ledger.refresh(
                "needs_scoring",
                process={"returncode": 0},
                reason="Generation already completed cleanly; scoring is gated.",
            )
            return 0

        if stagger_start_seconds > 0:
            ledger.event("stagger_sleep_started", seconds=stagger_start_seconds)
            time.sleep(stagger_start_seconds)
            ledger.event("stagger_sleep_completed", seconds=stagger_start_seconds)

        key_attention = (
            openrouter_key_limit_attention()
            if should_preflight_openrouter_key(command, contract)
            else None
        )
        if key_attention:
            ledger.refresh("attention", reason=key_attention, process={"returncode": None})
            ledger.event("scheduler_attention", reason=key_attention)
            return 2

        returncode = _run_command_with_ledger(
            ledger=ledger,
            steps=execute_steps,
            event_prefix="generation",
            poll_seconds=poll_seconds,
            stop_on_attention=stop_on_attention,
            stream_command_output=stream_command_output,
            max_active_calls=max_active_calls,
        )
        module_status = _load_json(_module_status_path(contract_path))
        observed = classify_observed_state(module_status, returncode=returncode)
        if observed == "stopped":
            ledger.refresh("stopped", process={"returncode": returncode})
            return 130
        if observed == "attention":
            ledger.refresh("attention", process={"returncode": returncode})
            return 2
        if observed == "needs_scoring":
            score_steps = _score_steps_from_contract(contract)
            if auto_score_on_clean_generation and score_steps:
                return _run_score_command_with_ledger(
                    ledger=ledger,
                    score_steps=score_steps,
                    poll_seconds=poll_seconds,
                    stop_on_attention=stop_on_attention,
                    stream_command_output=stream_command_output,
                    max_active_calls=max_active_calls,
                    emit_messages=emit_messages,
                )
            ledger.refresh(
                "needs_scoring",
                process={"returncode": returncode},
                reason=(
                    "Generation completed cleanly; scoring is gated."
                    if gate_after_generation or not score_steps
                    else "Generation completed cleanly; no score command was executed."
                ),
            )
            return 0
        ledger.refresh("score_ready" if observed == "score_ready" else "completed", process={"returncode": returncode})
        return 0
    finally:
        release_scheduler_lock(lock_path, scheduler_id=scheduler_id)


def run_contracts(
    contract_paths: list[Path],
    *,
    max_contract_workers: int = DEFAULT_MAX_CONTRACT_WORKERS,
    dry_run: bool = False,
    poll_seconds: float = 2.5,
    max_active_calls: int | None = None,
    stagger_start_seconds: float = 0.0,
    run_pace: str | None = None,
    stop_on_attention: bool = False,
    gate_after_generation: bool = True,
    auto_score_on_clean_generation: bool = True,
    emit_messages: bool = True,
    stream_command_output: bool = True,
) -> dict[Path, int]:
    """Run prepared contracts concurrently so independent stages can overlap."""
    paths = [Path(path) for path in contract_paths]
    if not paths:
        return {}
    workers = max(1, min(int(max_contract_workers), len(paths)))
    results: dict[Path, int] = {}

    def run_one(index: int, path: Path) -> int:
        return run_contract(
            path,
            dry_run=dry_run,
            poll_seconds=poll_seconds,
            max_active_calls=max_active_calls,
            stagger_start_seconds=index * max(0.0, stagger_start_seconds),
            run_pace=run_pace,
            stop_on_attention=stop_on_attention,
            gate_after_generation=gate_after_generation,
            auto_score_on_clean_generation=auto_score_on_clean_generation,
            emit_messages=emit_messages,
            stream_command_output=stream_command_output,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, index, path): path
            for index, path in enumerate(paths)
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception as exc:
                results[path] = 2
                if emit_messages:
                    print(
                        f"Scheduler worker failed for {_display_path(path)}: {sanitize_error_message(exc)}",
                        file=sys.stderr,
                    )
    return results


def score_contract(
    contract_path: Path,
    *,
    state_dir: Path | None = None,
    dry_run: bool = False,
    poll_seconds: float = 2.5,
    max_active_calls: int | None = None,
    run_pace: str | None = None,
    stop_on_attention: bool = False,
    force: bool = False,
    emit_messages: bool = True,
    stream_command_output: bool = True,
) -> int:
    """Run only a contract's score command from a clean generation gate."""
    contract = load_run_contract(contract_path)
    if not contract:
        raise ValueError(f"Invalid or empty contract: {contract_path}")
    target_state_dir = state_dir or contract_path.parent
    settings = {
        "dry_run": dry_run,
        "poll_seconds": poll_seconds,
        "max_active_calls": max_active_calls,
        "run_pace": run_pace,
        "stop_on_attention": stop_on_attention,
        "score_only": True,
        "force": force,
        "preflight_receipt_policy": _preflight_receipt_policy(
            contract,
            dry_run=dry_run,
        ),
    }
    scheduler_id = f"scheduler-{uuid.uuid4().hex[:12]}"
    try:
        validate_prepared_commands_before_execution(contract, contract_path)
    except PreparedCommandProvenanceError as exc:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh("attention", reason=sanitize_error_message(exc))
        ledger.event(
            "scheduler_attention",
            reason="prepared_command_provenance",
            provenance_issues=list(exc.issues),
        )
        if emit_messages:
            print(sanitize_error_message(exc), file=sys.stderr)
        return 2
    try:
        validate_run_pricing_before_spend(contract_path)
    except PreparedPricingProvenanceError as exc:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh("attention", reason=sanitize_error_message(exc))
        ledger.event(
            "scheduler_attention",
            reason="prepared_pricing_provenance",
            provenance_issues=list(exc.issues),
        )
        if emit_messages:
            print(sanitize_error_message(exc), file=sys.stderr)
        return 2
    if _contract_has_rendered_config(contract):
        try:
            validate_run_prepared_config_before_spend(contract_path)
        except PreparedConfigProvenanceError as exc:
            ledger = SchedulerLedger(
                target_state_dir,
                contract_path=contract_path,
                contract=contract,
                settings=settings,
                scheduler_id=scheduler_id,
            )
            ledger.refresh("attention", reason=sanitize_error_message(exc))
            ledger.event(
                "scheduler_attention",
                reason="prepared_config_provenance",
                provenance_issues=list(exc.issues),
            )
            if emit_messages:
                print(sanitize_error_message(exc), file=sys.stderr)
            return 2
    score_steps = _score_steps_from_contract(contract, force=force)
    score_command = _display_steps(score_steps)
    if not score_steps:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh(
            "attention",
            reason=(
                "Contract does not contain structured score argv. Legacy shell contracts "
                "require BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS=1."
            ),
        )
        ledger.event("scheduler_attention", reason="missing_structured_score_argv")
        return 2
    if dry_run:
        ledger = SchedulerLedger(
            target_state_dir,
            contract_path=contract_path,
            contract=contract,
            settings=settings,
            scheduler_id=scheduler_id,
        )
        ledger.refresh("dry_run", reason="Dry run: score command was not executed.")
        ledger.event(
            "dry_run_queued",
            command=score_command,
            score_only=True,
            preflight_receipt_policy="not_enforced_dry_run",
        )
        if emit_messages:
            print(f"Scheduler score dry run queued: {_display_path(contract_path)}")
        return 0

    try:
        lock_path = acquire_scheduler_lock(
            target_state_dir,
            scheduler_id=scheduler_id,
            contract_path=contract_path,
            command=score_command,
        )
    except SchedulerAlreadyRunning as exc:
        if emit_messages:
            print(str(exc), file=sys.stderr)
        return DUPLICATE_SCHEDULER_EXIT_CODE

    ledger = SchedulerLedger(
        target_state_dir,
        contract_path=contract_path,
        contract=contract,
        settings=settings,
        scheduler_id=scheduler_id,
        lock_path=lock_path,
    )
    try:
        module_status = _load_json(_module_status_path(contract_path))
        observed_before = classify_observed_state(module_status)
        if observed_before == "score_ready":
            ledger.refresh("score_ready", reason="Run is already scored; no score command was executed.")
            return 0
        force_allowed = (
            force
            and module_status.get("stage") == "scoring"
            and (
                module_status.get("status") in {"failed_scoring", "stopped"}
                or (
                    module_status.get("status") == "failed_invalid"
                    and module_status.get("failure_stage") == "artifact_identity"
                )
            )
        )
        if observed_before != "needs_scoring" and not force_allowed:
            ledger.refresh(
                "attention",
                reason=(
                    f"Score-only requires needs_scoring state; observed {observed_before}. "
                    "Use --force only for failed_scoring/stopped score ledgers or "
                    "a score-stage failed_invalid artifact-identity instrument "
                    "failure after clean generation."
                ),
            )
            ledger.event("scheduler_attention", reason="score_only_requires_needs_scoring", observed=observed_before)
            return 2
        if force_allowed:
            ledger.event(
                "force_score_retry_allowed",
                previous_status=module_status.get("status"),
                previous_stage=module_status.get("stage"),
                previous_failure_stage=module_status.get("failure_stage"),
            )
        return _run_score_command_with_ledger(
            ledger=ledger,
            score_steps=score_steps,
            poll_seconds=poll_seconds,
            stop_on_attention=stop_on_attention,
            stream_command_output=stream_command_output,
            max_active_calls=max_active_calls,
            ignore_attention_status=module_status if force_allowed else None,
            emit_messages=emit_messages,
        )
    finally:
        release_scheduler_lock(lock_path, scheduler_id=scheduler_id)


def stop_contract(contract_path: Path, *, reason: str, requested_by: str = "operator") -> Path:
    """Request cooperative stop before future paid calls for a contract."""
    return write_run_control(
        contract_path.parent,
        action=STOP_BEFORE_NEXT_PAID_CALL,
        reason=reason,
        requested_by=requested_by,
    )


def clear_contract_control(contract_path: Path, *, reason: str = "clear scheduler control") -> Path:
    """Clear cooperative scheduler control for a contract directory."""
    return write_run_control(
        contract_path.parent,
        action=CLEAR_CONTROL,
        reason=reason,
        requested_by="scheduler",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schedule prepared benchmark run contracts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run or dry-run a prepared contract.")
    run_parser.add_argument("--contract", required=True, help="Path to RUN_CONTRACT.json or its directory.")
    run_parser.add_argument("--state-dir", help="Directory for SCHEDULER_STATUS.json. Defaults beside contract.")
    run_parser.add_argument("--dry-run", action="store_true", help="Write scheduler ledgers without executing.")
    run_parser.add_argument("--poll-seconds", type=float, default=2.5)
    run_parser.add_argument(
        "--run-pace",
        choices=tuple(RUN_PACE_PRESETS),
        default=DEFAULT_RUN_PACE,
        help="Named paid-call concurrency posture. Explicit numeric flags override the preset.",
    )
    run_parser.add_argument("--max-active-calls", type=int)
    run_parser.add_argument("--stagger-start-seconds", type=float)
    run_parser.add_argument("--stop-on-attention", action="store_true")
    run_parser.add_argument("--gate-after-generation", action="store_true", default=True)
    run_parser.add_argument("--auto-score-on-clean-generation", action="store_true")
    run_parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print final scheduler status JSON. Suppresses child command stdout/stderr.",
    )

    many_parser = subparsers.add_parser(
        "run-many",
        help="Run prepared contracts concurrently and score each clean generation as it finishes.",
        description=(
            "Run prepared contracts concurrently and score each clean generation as it finishes. "
            "All contracts share the global paid-call lease."
        ),
    )
    many_parser.add_argument(
        "--contract",
        action="append",
        required=True,
        help="Path to a RUN_CONTRACT.json or its directory. Repeat for each module.",
    )
    many_parser.add_argument("--dry-run", action="store_true")
    many_parser.add_argument("--poll-seconds", type=float, default=2.5)
    many_parser.add_argument("--max-contract-workers", type=int, default=DEFAULT_MAX_CONTRACT_WORKERS)
    many_parser.add_argument(
        "--run-pace",
        choices=tuple(RUN_PACE_PRESETS),
        default=DEFAULT_RUN_PACE,
    )
    many_parser.add_argument(
        "--max-active-calls",
        type=int,
        help="Request the shared global paid-call lease ceiling; stricter limits still apply.",
    )
    many_parser.add_argument("--stagger-start-seconds", type=float)
    many_parser.add_argument(
        "--stop-on-attention",
        action="store_true",
        help="Affected contract only; does not stop sibling contracts.",
    )
    many_parser.add_argument(
        "--no-auto-score",
        action="store_false",
        dest="auto_score_on_clean_generation",
        help="Leave clean generation runs at the scoring gate instead of scoring immediately.",
    )
    many_parser.set_defaults(auto_score_on_clean_generation=True)
    many_parser.add_argument("--output-json", action="store_true")

    score_parser = subparsers.add_parser("score", help="Run only the prepared contract score command.")
    score_parser.add_argument("--contract", required=True, help="Path to RUN_CONTRACT.json or its directory.")
    score_parser.add_argument("--state-dir", help="Directory for SCHEDULER_STATUS.json. Defaults beside contract.")
    score_parser.add_argument("--dry-run", action="store_true", help="Write scheduler ledgers without scoring.")
    score_parser.add_argument("--poll-seconds", type=float, default=2.5)
    score_parser.add_argument(
        "--run-pace",
        choices=tuple(RUN_PACE_PRESETS),
        default=DEFAULT_RUN_PACE,
        help="Named paid-call concurrency posture. Explicit --max-active-calls overrides the preset.",
    )
    score_parser.add_argument("--max-active-calls", type=int)
    score_parser.add_argument("--stop-on-attention", action="store_true")
    score_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Retry scoring without rerunning generation after failed scoring or "
            "a score-stage artifact-identity instrument failure."
        ),
    )
    score_parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print final scheduler status JSON. Suppresses child command stdout/stderr.",
    )

    stop_parser = subparsers.add_parser("stop", help="Request cooperative stop for a contract.")
    stop_parser.add_argument("--contract", required=True)
    stop_parser.add_argument("--reason", default="operator requested scheduler stop")
    stop_parser.add_argument("--requested-by", default="operator")
    stop_parser.add_argument("--output-json", action="store_true", help="Print machine-readable JSON.")

    clear_parser = subparsers.add_parser("clear-control", help="Clear cooperative control for a contract.")
    clear_parser.add_argument("--contract", required=True)
    clear_parser.add_argument("--reason", default="operator cleared scheduler stop")
    clear_parser.add_argument("--output-json", action="store_true", help="Print machine-readable JSON.")

    status_parser = subparsers.add_parser("status", help="Print scheduler status JSON.")
    status_parser.add_argument("--state-dir", required=True)
    status_parser.add_argument(
        "--output-json",
        action="store_true",
        help="Accepted for CLI symmetry; scheduler status is always JSON.",
    )

    paces_parser = subparsers.add_parser("paces", help="List named scheduler run pace presets.")
    paces_parser.add_argument("--output-json", action="store_true", help="Print machine-readable JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print(json.dumps(load_scheduler_status(args.state_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "paces":
        print_run_paces(output_json=args.output_json)
        return 0
    if args.command == "run-many":
        contract_paths = [_resolve_contract(value) for value in args.contract]
        run_pace, max_active_calls, stagger_start_seconds = resolve_run_pace(
            run_pace=args.run_pace,
            max_active_calls=args.max_active_calls,
            stagger_start_seconds=args.stagger_start_seconds,
        )
        results = run_contracts(
            contract_paths,
            max_contract_workers=args.max_contract_workers,
            dry_run=args.dry_run,
            poll_seconds=args.poll_seconds,
            max_active_calls=max_active_calls,
            stagger_start_seconds=stagger_start_seconds,
            run_pace=run_pace,
            stop_on_attention=args.stop_on_attention,
            auto_score_on_clean_generation=args.auto_score_on_clean_generation,
            emit_messages=not args.output_json,
            stream_command_output=not args.output_json,
        )
        exit_code = 0 if all(code == 0 for code in results.values()) else 2
        if args.output_json:
            print(
                json.dumps(
                    {
                        "exit_code": exit_code,
                        "contracts": [
                            {
                                "contract_path": _display_path(path),
                                "exit_code": results[path],
                                **load_scheduler_status(path.parent),
                            }
                            for path in contract_paths
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return exit_code

    contract_path = _resolve_contract(args.contract)
    if args.command == "stop":
        path = stop_contract(contract_path, reason=args.reason, requested_by=args.requested_by)
        if args.output_json:
            print(
                json.dumps(
                    {
                        "action": STOP_BEFORE_NEXT_PAID_CALL,
                        "contract_path": _display_path(contract_path),
                        "control_path": _display_path(path),
                        "control": load_run_control(path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Stop requested: {_display_path(path)}")
        return 0
    if args.command == "clear-control":
        path = clear_contract_control(contract_path, reason=args.reason)
        if args.output_json:
            print(
                json.dumps(
                    {
                        "action": CLEAR_CONTROL,
                        "contract_path": _display_path(contract_path),
                        "control_path": _display_path(path),
                        "control": load_run_control(path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Control cleared: {_display_path(path)}")
        return 0
    if args.command == "run":
        state_dir = Path(args.state_dir) if args.state_dir else None
        run_pace, max_active_calls, stagger_start_seconds = resolve_run_pace(
            run_pace=args.run_pace,
            max_active_calls=args.max_active_calls,
            stagger_start_seconds=args.stagger_start_seconds,
        )
        exit_code = run_contract(
            contract_path,
            state_dir=state_dir,
            dry_run=args.dry_run,
            poll_seconds=args.poll_seconds,
            max_active_calls=max_active_calls,
            stagger_start_seconds=stagger_start_seconds,
            run_pace=run_pace,
            stop_on_attention=args.stop_on_attention,
            gate_after_generation=args.gate_after_generation,
            auto_score_on_clean_generation=args.auto_score_on_clean_generation,
            emit_messages=not args.output_json,
            stream_command_output=not args.output_json,
        )
        if args.output_json:
            print(
                json.dumps(
                    {
                        "exit_code": exit_code,
                        "status": load_scheduler_status(state_dir or contract_path.parent),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return exit_code
    if args.command == "score":
        state_dir = Path(args.state_dir) if args.state_dir else None
        run_pace, max_active_calls, _stagger_start_seconds = resolve_run_pace(
            run_pace=args.run_pace,
            max_active_calls=args.max_active_calls,
        )
        exit_code = score_contract(
            contract_path,
            state_dir=state_dir,
            dry_run=args.dry_run,
            poll_seconds=args.poll_seconds,
            max_active_calls=max_active_calls,
            run_pace=run_pace,
            stop_on_attention=args.stop_on_attention,
            force=args.force,
            emit_messages=not args.output_json,
            stream_command_output=not args.output_json,
        )
        if args.output_json:
            print(
                json.dumps(
                    {
                        "exit_code": exit_code,
                        "status": load_scheduler_status(state_dir or contract_path.parent),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return exit_code
    raise ValueError(f"Unknown scheduler command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
