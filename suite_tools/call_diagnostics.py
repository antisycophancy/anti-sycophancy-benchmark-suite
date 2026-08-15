"""Private, prompt-free lifecycle diagnostics for paid provider calls.

The canonical benchmark artifacts deliberately do not retain arbitrary provider
response bodies.  This journal fills the operational gap without changing the
request sent to a provider, the response returned to a runner, or any contract
or score artifact.  It is local-only and is not part of evidence bundles.

Each provider attempt has three durable states:

* ``intent_written``: the journal can identify the logical call before spend;
* ``provider_invocation_started``: the SDK/HTTP adapter is about to be invoked;
* ``closed``: the invocation returned or raised.

``provider_invocation_started`` is intentionally not named "dispatched".  The
OpenAI-compatible SDK boundary does not expose the instant at which a socket
write completes, so a process loss after this state has an ambiguous billing
outcome rather than a false claim that the provider received the request.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from suite_tools.provider_client import extract_raw_response, inspect_chat_completion_response
from suite_tools.provider_signals import classify_payload
from suite_tools.run_monitor import (
    classify_failure_status,
    response_usage_to_dict,
    sanitize_error_message,
)

SCHEMA_VERSION = "benchmark-call-diagnostic-v1"
DIAGNOSTICS_FILENAME = "CALL_DIAGNOSTICS.jsonl"
ROTATED_FILENAMES = tuple(f"{DIAGNOSTICS_FILENAME}.{index}" for index in range(1, 4))
LOCK_FILENAME = "CALL_DIAGNOSTICS.lock"

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
)
_SAFE_ERROR_FIELDS = (
    "type",
    "code",
    "status",
    "finish_reason",
    "native_finish_reason",
    "stop_reason",
)
_PROCESS_WRITER_ID = f"{os.getpid()}-{uuid.uuid4().hex}"
_THREAD_LOCK = threading.RLock()


class DiagnosticJournalError(RuntimeError):
    """A pre-invocation diagnostic write failed, so the paid call was stopped."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_sha256(output_dir: Path) -> str | None:
    configured = os.environ.get("BENCHMARK_CONTRACT_PATH")
    path = Path(configured) if configured else output_dir / "RUN_CONTRACT.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(value, 1)


def _safe_scalar(value: Any, *, limit: int = 1000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_error_message(value)[:limit]


def _safe_error_body(error: BaseException) -> dict[str, Any]:
    """Project a provider body to non-free-form fields plus an opaque digest."""
    raw = extract_raw_response(error)
    if not isinstance(raw, dict):
        return {}

    recorded_digest = raw.get("raw_body_sha256")
    result: dict[str, Any] = {
        "raw_body_sha256": (
            str(recorded_digest)
            if isinstance(recorded_digest, str) and len(recorded_digest) == 64
            else _stable_hash(raw)
        ),
    }
    error_obj = raw.get("error") if isinstance(raw.get("error"), dict) else {}
    safe_error = {
        field: _safe_scalar(error_obj[field])
        for field in _SAFE_ERROR_FIELDS
        if field in error_obj
    }
    if safe_error:
        result["provider_error"] = safe_error

    for field in (
        "finish_reason",
        "native_finish_reason",
        "stop_reason",
        "status",
        "response_shape",
    ):
        if field in raw:
            result[field] = _safe_scalar(raw[field])

    stop_details = raw.get("stop_details")
    if isinstance(stop_details, dict):
        safe_details = {
            key: _safe_scalar(stop_details[key])
            for key in ("category", "classifier")
            if key in stop_details
        }
        if safe_details:
            result["stop_details"] = safe_details

    prompt_feedback = raw.get("promptFeedback")
    if isinstance(prompt_feedback, dict):
        safe_feedback = {
            key: _safe_scalar(prompt_feedback[key])
            for key in ("blockReason",)
            if key in prompt_feedback
        }
        if safe_feedback:
            result["prompt_feedback"] = safe_feedback

    candidates = raw.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        first = candidates[0]
        candidate = {
            key: _safe_scalar(first[key])
            for key in ("finishReason",)
            if key in first
        }
        ratings = first.get("safetyRatings")
        if isinstance(ratings, list):
            candidate["safety_ratings"] = [
                {
                    key: _safe_scalar(rating[key])
                    for key in ("category", "probability", "blocked")
                    if key in rating
                }
                for rating in ratings
                if isinstance(rating, dict)
            ][:20]
        if candidate:
            result["candidate"] = candidate
    return result


def _journal_paths(path: Path) -> list[Path]:
    rotated = [path.with_name(name) for name in reversed(ROTATED_FILENAMES)]
    return [candidate for candidate in rotated + [path] if candidate.exists()]


def load_call_diagnostics(output_dir: Path | str) -> dict[str, Any]:
    """Load deduplicated records and report malformed/torn journal lines."""
    path = Path(output_dir) / DIAGNOSTICS_FILENAME
    records: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    torn_tail = False
    seen: set[str] = set()

    for journal_path in _journal_paths(path):
        try:
            lines = journal_path.read_bytes().splitlines(keepends=True)
        except OSError as exc:
            malformed.append({"path": str(journal_path), "error": str(exc)})
            continue
        for index, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                is_last = index == len(lines) and journal_path == path
                if is_last:
                    torn_tail = True
                else:
                    malformed.append(
                        {"path": str(journal_path), "line": index, "error": str(exc)}
                    )
                continue
            if not isinstance(value, dict):
                malformed.append(
                    {"path": str(journal_path), "line": index, "error": "record is not an object"}
                )
                continue
            event_id = str(value.get("event_id") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            records.append(value)

    records.sort(key=lambda item: (str(item.get("timestamp") or ""), int(item.get("sequence") or 0)))
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "records": records,
        "malformed": malformed,
        "torn_tail": torn_tail,
    }


class CallDiagnosticJournal:
    """Durable local JSONL writer with per-logical-call attempt allocation."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / DIAGNOSTICS_FILENAME
        self.writer_id = _PROCESS_WRITER_ID

    def _rotate_locked(self) -> None:
        max_bytes = _positive_int_env("BENCHMARK_DIAGNOSTIC_MAX_BYTES", 32 * 1024 * 1024)
        try:
            active_size = self.path.stat().st_size
        except OSError:
            return
        if active_size < max_bytes:
            return
        for index in range(len(ROTATED_FILENAMES), 1, -1):
            older = self.path.with_name(f"{DIAGNOSTICS_FILENAME}.{index - 1}")
            newer = self.path.with_name(f"{DIAGNOSTICS_FILENAME}.{index}")
            if older.exists():
                os.replace(older, newer)
        if self.path.exists():
            os.replace(self.path, self.path.with_name(ROTATED_FILENAMES[0]))

    @staticmethod
    def _decode_records(data: bytes) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for raw_line in data.splitlines():
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def _append(self, fields: dict[str, Any], *, allocate_attempt: bool = False) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / LOCK_FILENAME
        with _THREAD_LOCK:
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.chmod(lock_path, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                existing: list[dict[str, Any]] = []
                for journal_path in _journal_paths(self.path):
                    try:
                        existing.extend(self._decode_records(journal_path.read_bytes()))
                    except OSError:
                        continue
                sequence = max((int(row.get("sequence") or 0) for row in existing), default=0) + 1
                payload = dict(fields)
                if allocate_attempt:
                    logical_id = str(payload["logical_call_id"])
                    ordinal = max(
                        (
                            int(row.get("attempt_ordinal") or 0)
                            for row in existing
                            if row.get("logical_call_id") == logical_id
                        ),
                        default=0,
                    ) + 1
                    payload["attempt_ordinal"] = ordinal
                    payload["attempt_id"] = _stable_hash(
                        {"logical_call_id": logical_id, "attempt_ordinal": ordinal}
                    )
                event_id = _stable_hash(
                    {
                        "attempt_id": payload.get("attempt_id"),
                        "state": payload.get("state"),
                    }
                )
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": sequence,
                    "timestamp": _utc_now(),
                    "event_id": event_id,
                    "writer_id": self.writer_id,
                    **payload,
                }
                encoded = (json.dumps(payload, sort_keys=True, default=str) + "\n").encode("utf-8")
                self._rotate_locked()
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.chmod(self.path, 0o600)
                    os.write(fd, encoded)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return payload
            except OSError as exc:
                raise DiagnosticJournalError(f"could not persist provider-call diagnostic: {exc}") from exc
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    def begin(self, identity: dict[str, Any]) -> "CallAttempt":
        intent = self._append({**identity, "state": "intent_written"}, allocate_attempt=True)
        return CallAttempt(self, intent)


@dataclass
class CallAttempt:
    journal: CallDiagnosticJournal
    intent: dict[str, Any]
    invocation_started: bool = False
    closed: bool = False

    @property
    def logical_call_id(self) -> str:
        return str(self.intent["logical_call_id"])

    @property
    def attempt_id(self) -> str:
        return str(self.intent["attempt_id"])

    def _base(self) -> dict[str, Any]:
        excluded = {"schema_version", "sequence", "timestamp", "event_id", "writer_id", "state"}
        return {key: value for key, value in self.intent.items() if key not in excluded}

    def mark_provider_invocation_started(self) -> None:
        if self.invocation_started:
            return
        self.journal._append({**self._base(), "state": "provider_invocation_started"})
        self.invocation_started = True

    def close_success(self, response: Any) -> None:
        if self.closed:
            return
        usage = response_usage_to_dict(response)
        inspection = inspect_chat_completion_response(response)
        signal_evidence = classify_payload(inspection.signal_payload)
        if signal_evidence is not None:
            outcome = "provider_signal"
        elif inspection.response_shape is not None:
            outcome = "malformed_response"
        else:
            outcome = "provider_response"
        fields: dict[str, Any] = {
            **self._base(),
            "state": "closed",
            "outcome": outcome,
            "billing_state": "confirmed" if usage else "unknown",
            "normalized_response_sha256": _stable_hash(inspection.raw_response),
            "usage": {
                key: usage[key]
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "thoughts_tokens",
                    "cost",
                    "cost_source",
                )
                if key in usage
            },
        }
        if inspection.response_shape is not None:
            fields["response_shape"] = inspection.response_shape
            if signal_evidence is None:
                fields["failure_status"] = "failed_provider"
                fields["error_type"] = "ProviderMalformedResponseError"
        if signal_evidence is not None:
            for key in (
                "evidence_class",
                "category",
                "provider",
                "provider_code",
                "signal_source",
                "retry_policy",
                "stochastic",
            ):
                if key in signal_evidence:
                    fields[key] = signal_evidence[key]
        for key in ("finish_reason", "native_finish_reason"):
            value = inspection.signal_payload.get(key)
            if value is not None:
                fields[key] = _safe_scalar(value)
        self.journal._append(fields)
        self.closed = True

    def close_error(self, error: BaseException) -> None:
        if self.closed:
            return
        usage = getattr(error, "usage", None)
        usage_dict = dict(usage) if isinstance(usage, dict) else {}
        snapshot = _safe_error_body(error)
        status_code = getattr(error, "status_code", None)
        error_message_digest = _stable_hash(
            {
                "error_type": type(error).__name__,
                "message": sanitize_error_message(error),
            }
        )
        self.journal._append(
            {
                **self._base(),
                "state": "closed",
                "outcome": "error",
                "billing_state": (
                    "confirmed" if usage_dict else
                    "likely" if self.invocation_started else
                    "unbilled_likely"
                ),
                "failure_status": classify_failure_status(error),
                "error_type": type(error).__name__,
                "error_message_sha256": error_message_digest,
                "http_status": (
                    int(status_code)
                    if isinstance(status_code, int) or str(status_code).isdigit()
                    else None
                ),
                "usage": {
                    key: usage_dict[key]
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "thoughts_tokens",
                        "cost",
                        "cost_source",
                    )
                    if key in usage_dict
                },
                **snapshot,
            }
        )
        self.closed = True


class NullCallAttempt:
    """No-op attempt used when no run output directory is available."""

    logical_call_id = ""
    attempt_id = ""

    def mark_provider_invocation_started(self) -> None:
        return None

    def close_success(self, response: Any) -> None:
        return None

    def close_error(self, error: BaseException) -> None:
        return None


def begin_provider_attempt(
    *,
    monitor: Any = None,
    output_dir: Path | str | None = None,
    module: str | None = None,
    stage: str | None = None,
    role: str,
    model: str,
    provider: str = "unknown",
    provider_api: str | None = None,
    context: dict[str, Any] | None = None,
) -> CallAttempt | NullCallAttempt:
    """Persist one prompt-free call intent and return its lifecycle handle."""
    if output_dir is None and monitor is not None:
        output_dir = getattr(monitor, "output_dir", None)
    if output_dir is None:
        output_dir = os.environ.get("BENCHMARK_OUTPUT_DIR")
    if output_dir is None:
        return NullCallAttempt()
    if not isinstance(output_dir, (str, Path)):
        return NullCallAttempt()

    path = Path(output_dir)
    module = module or getattr(monitor, "module", None) or os.environ.get("BENCHMARK_MODULE") or "unknown"
    stage = stage or getattr(monitor, "stage", None) or os.environ.get("BENCHMARK_STAGE") or role
    run_id = os.environ.get("BENCHMARK_RUN_ID") or path.name
    safe_context = {
        key: _safe_scalar(value)
        for key in _CONTEXT_FIELDS
        if (value := (context or {}).get(key)) is not None
    }
    identity = {
        "run_id": run_id,
        "contract_sha256": _contract_sha256(path),
        "module": str(module),
        "stage": str(stage),
        "role": str(role),
        "model": str(model),
        "provider": str(provider),
        "provider_api": str(provider_api) if provider_api else None,
        "context": safe_context,
    }
    identity["logical_call_id"] = _stable_hash(identity)
    return CallDiagnosticJournal(path).begin(identity)


def close_success_best_effort(attempt: CallAttempt | NullCallAttempt, response: Any, monitor: Any = None) -> None:
    """Close after provider return without changing the runner's success path."""
    try:
        attempt.close_success(response)
    except Exception as exc:
        _record_diagnostic_warning(attempt, exc, monitor)


def close_error_best_effort(attempt: CallAttempt | NullCallAttempt, error: BaseException, monitor: Any = None) -> None:
    """Close after provider error without hiding the provider's exception."""
    try:
        attempt.close_error(error)
    except Exception as exc:
        _record_diagnostic_warning(attempt, exc, monitor)


def _record_diagnostic_warning(attempt: Any, error: BaseException, monitor: Any) -> None:
    if monitor is None:
        return
    try:
        monitor.record(
            "diagnostic_write_failed",
            logical_call_id=getattr(attempt, "logical_call_id", None),
            attempt_id=getattr(attempt, "attempt_id", None),
            error_type=type(error).__name__,
            error=sanitize_error_message(error)[:500],
        )
    except Exception:
        pass


def diagnose_call_journal(output_dir: Path | str) -> dict[str, Any]:
    """Summarize lifecycle gaps and provider failures for one run directory."""
    loaded = load_call_diagnostics(output_dir)
    attempts: dict[str, dict[str, Any]] = {}
    for record in loaded["records"]:
        attempt_id = str(record.get("attempt_id") or "")
        if not attempt_id:
            continue
        row = attempts.setdefault(
            attempt_id,
            {
                "attempt_id": attempt_id,
                "logical_call_id": record.get("logical_call_id"),
                "run_id": record.get("run_id"),
                "module": record.get("module"),
                "stage": record.get("stage"),
                "role": record.get("role"),
                "model": record.get("model"),
                "provider": record.get("provider"),
                "context": record.get("context") or {},
                "states": [],
            },
        )
        state = record.get("state")
        if state not in row["states"]:
            row["states"].append(state)
        if state == "closed":
            for key in (
                "outcome",
                "billing_state",
                "failure_status",
                "error_type",
                "error_message_sha256",
                "http_status",
                "response_shape",
                "provider_error",
                "raw_body_sha256",
                "normalized_response_sha256",
                "evidence_class",
                "category",
                "provider_code",
                "signal_source",
                "retry_policy",
                "stochastic",
            ):
                if key in record:
                    row[key] = record[key]

    ordered = sorted(
        attempts.values(),
        key=lambda item: (str(item.get("module")), str(item.get("model")), str(item.get("attempt_id"))),
    )
    unresolved = [row for row in ordered if "closed" not in row["states"]]
    failures = [
        row
        for row in ordered
        if row.get("outcome") in {"error", "malformed_response"}
    ]
    provider_signals = [row for row in ordered if row.get("outcome") == "provider_signal"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(Path(output_dir)),
        "attempt_count": len(ordered),
        "closed_count": len(ordered) - len(unresolved),
        "unresolved_count": len(unresolved),
        "failure_count": len(failures),
        "provider_signal_count": len(provider_signals),
        "torn_tail": loaded["torn_tail"],
        "malformed": loaded["malformed"],
        "unresolved": unresolved,
        "failures": failures,
        "provider_signals": provider_signals,
        "attempts": ordered,
    }
