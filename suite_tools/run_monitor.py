"""Run status and event ledger utilities for paid benchmark collection.

The benchmark suite intentionally writes conversations and scores as they are
created, because paid frontier runs are too expensive to treat as disposable.
This module gives each output directory two small machine-readable files:

* ``RUN_STATUS.json``: latest stage, counters, validity, and failure reason.
* ``RUN_EVENTS.jsonl``: append-only event stream for turns, scores, and aborts.

The ledger avoids prompt/response text by default. Conversation content stays
in the transcript artifacts; the ledger records paths, identifiers, and failure
classes so reviewers can tell whether a score is model behavior or harness
infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from suite_tools.provider_signals import classify_payload

SCHEMA_VERSION = "benchmark-run-ledger-v1"
ATTEMPTS_FILENAME = "ATTEMPTS.jsonl"
ATTEMPT_SCHEMA_VERSION = "benchmark-run-attempt-v1"
_ATTEMPT_LOCK_FILENAME = "ATTEMPTS.lock"
BLOCKS_FILENAME = "BLOCKS.jsonl"
BLOCK_SCHEMA_VERSION = "benchmark-block-v2"


def nonnegative_finite_number(value: Any) -> tuple[float, bool]:
    """Return a ledger-safe amount and whether the source value was valid."""
    if value is None or isinstance(value, bool):
        return 0.0, False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not math.isfinite(parsed) or parsed < 0:
        return 0.0, False
    return parsed, True


def nonnegative_integer(value: Any) -> tuple[int, bool]:
    """Return a ledger-safe token count and whether the source value was valid."""
    if value is None or isinstance(value, bool):
        return 0, False
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0, False
    if not math.isfinite(numeric) or numeric != parsed or parsed < 0:
        return 0, False
    return parsed, True


def load_attempts(output_dir: "Path | str") -> list[dict[str, Any]]:
    """Read archived attempt snapshots (oldest first). Missing file -> []."""
    path = Path(output_dir) / ATTEMPTS_FILENAME
    if not path.exists():
        return []
    attempts: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            attempts.append(record)
    return attempts


class RunMonitorConflictError(RuntimeError):
    """A second monitor tried to attach to a directory owned by a live one."""


def _monitor_stale_seconds() -> float:
    try:
        return float(os.environ.get("BENCHMARK_MONITOR_STALE_SECONDS", "120"))
    except ValueError:
        return 120.0


def _attempt_lock_deadline_seconds() -> float:
    try:
        return float(os.environ.get("BENCHMARK_ATTEMPT_LOCK_DEADLINE_SECONDS", "30.0"))
    except ValueError:
        return 30.0


_STALE_LOCK_AGE_SECONDS = 60.0


def _status_age_seconds(status: dict[str, Any]) -> float:
    """Seconds since the status's updated_at; +inf when unparsable (treat as stale)."""
    raw = status.get("updated_at")
    try:
        updated = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return float("inf")
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated).total_seconds()


def _max_attempt_for_stage(output_dir: "Path | str", module: str, stage: str) -> int:
    """Highest attempt_number recorded for (module, stage) across the chain; 0 when none."""
    best = 0
    for record in load_attempts(output_dir):
        status = record.get("status") or {}
        if status.get("module") == module and status.get("stage") == stage:
            try:
                best = max(best, int(status.get("attempt_number", 1)))
            except (TypeError, ValueError):
                best = max(best, 1)
    return best


TERMINAL_STATUSES = {
    "completed",
    "stopped",
    "failed_auth",
    "failed_billing",
    "failed_incomplete",
    "failed_invalid",
    "failed_provider",
    "failed_rate_limited",
    "failed_scoring",
    "failed_timeout",
}


def sanitize_error_message(error: object) -> str:
    """Redact provider account identifiers and token-shaped secrets from errors."""
    text = str(error)
    text = re.sub(
        r"(['\"]?user_id['\"]?\s*:\s*)['\"]user_[A-Za-z0-9_-]+['\"]",
        r"\1'<redacted>'",
        text,
    )
    text = re.sub(r"user_[A-Za-z0-9_-]{12,}", "user_<redacted>", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{30,}", "AIza<redacted>", text)
    text = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "sk-or-v1-<redacted>", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{16,}", "sk-<redacted>", text)
    text = re.sub(
        r"(x-goog-api-key['\"]?\s*[:=]\s*)['\"]?[A-Za-z0-9._\-]{8,}",
        r"\1<redacted>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(openrouter\.ai/workspaces/[^/\s]+/keys/)[A-Za-z0-9_-]+",
        r"\1<redacted>",
        text,
    )
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer <redacted>", text)
    text = re.sub(
        r"(['\"]?(?:api_key|authorization)['\"]?\s*:\s*)['\"][^'\"]+['\"]",
        r"\1'<redacted>'",
        text,
        flags=re.IGNORECASE,
    )
    return text


def sanitize_ledger_value(value: Any) -> Any:
    """Recursively sanitize strings before they are written to run ledgers."""
    if isinstance(value, str):
        return sanitize_error_message(value)
    if isinstance(value, list):
        return [sanitize_ledger_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_ledger_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_ledger_value(item) for key, item in value.items()}
    return value


def _extract_raw_body_text(raw_error: object) -> str | None:
    """Return a canonical JSON serialization of a provider error body, or None.

    Priority: raw_error.raw_response (dict or str) → raw_error.body (dict or str).
    Returns None when no structured body is available.  Never raises.

    The returned string is used for two purposes:
    1. sha256 provenance digest (pre-truncation, pre-sanitization).
    2. Excerpt prefix (first 2000 chars, then sanitized by the caller).
    """
    raw = getattr(raw_error, "raw_response", None)
    if isinstance(raw, dict):
        return json.dumps(raw, separators=(",", ":"), sort_keys=True)
    if isinstance(raw, str):
        return raw
    body = getattr(raw_error, "body", None)
    if isinstance(body, dict):
        return json.dumps(body, separators=(",", ":"), sort_keys=True)
    if isinstance(body, str):
        return body
    return None


def make_evidence_snapshot(
    evidence: dict[str, Any],
    raw_error: object | None = None,
    billed_attempts: int | None = None,
) -> dict[str, Any]:
    """Build optional BLOCKS v2 / classified-event snapshot fields.

    Returns a dict of fields safe to merge into a BLOCKS record or an
    ``attempt_failure_classified`` event.  All fields are optional; the dict
    may be empty when no information is available.

    ``raw_body_sha256`` is the sha256 of the FULL raw body bytes (utf-8,
    pre-truncation, pre-sanitization) — a provenance digest.  The full body is
    NOT retained; the digest proves which body the excerpt derived from and
    enables dedupe/correlation across records (plan 020 D2).

    ``raw_body_excerpt`` is the FIRST 2000 chars of the serialized raw body,
    passed through ``sanitize_error_message`` to redact secrets (plan 020 D2).
    Raw bodies can echo prompt content; local ledgers keep this field; bundles
    project it out (plan 020 D2).

    Callers (runners) that have the raw exception in scope should pass it as
    ``raw_error`` so body-derived fields are populated.
    """
    snap: dict[str, Any] = {}

    # Fields derivable from the evidence dict (from classify_evidence / classify_payload)
    for field in ("signal_source", "stochastic", "native_finish_reason",
                  "provider", "provider_code"):
        if field in evidence:
            snap[field] = evidence[field]

    retry_policy = evidence.get("retry_policy")
    if isinstance(retry_policy, dict) and "kind" in retry_policy:
        snap["retry_policy_kind"] = retry_policy["kind"]

    # Fields derived from the raw error body
    if raw_error is not None:
        body_text = _extract_raw_body_text(raw_error)
        if body_text is not None:
            raw_bytes = body_text.encode("utf-8")
            # sha256 over FULL raw body bytes, pre-truncation, pre-sanitization.
            # Provenance digest: full body is not retained; digest enables dedupe
            # and correlation across records (plan 020 D2).
            snap["raw_body_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
            # Excerpt: first 2000 chars, sanitized (secrets redacted, plan 020 D2).
            snap["raw_body_excerpt"] = sanitize_error_message(body_text)[:2000]

            # native_finish_reason from body if not already provided by evidence
            if "native_finish_reason" not in snap:
                body_dict = getattr(raw_error, "raw_response", None)
                if not isinstance(body_dict, dict):
                    body_dict = getattr(raw_error, "body", None)
                if isinstance(body_dict, dict):
                    nfr = body_dict.get("native_finish_reason") or body_dict.get("finish_reason")
                    if nfr:
                        snap["native_finish_reason"] = str(nfr)

    if billed_attempts is not None:
        snap["billed_attempts"] = int(billed_attempts)

    return snap


def usage_to_dict(usage: Any) -> dict[str, Any]:
    """Normalize OpenAI/OpenRouter SDK usage payloads to a plain dict."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        try:
            return dict(model_dump())
        except TypeError:
            return dict(model_dump(mode="json"))
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost",
        "estimated_cost",
        "cost_source",
        "thoughts_tokens",
        "reasoning_tokens",
        "billable_completion_tokens",
        "completion_tokens_details",
        "output_tokens_details",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result


def response_usage_to_dict(response: Any) -> dict[str, Any]:
    """Extract normalized usage from an OpenAI/OpenRouter SDK response."""
    if isinstance(response, dict):
        return usage_to_dict(response.get("usage"))
    return usage_to_dict(getattr(response, "usage", None))


def record_provider_call_error_usage(
    monitor: "RunMonitor | None",
    model: str,
    error: BaseException,
    *,
    role: str = "unknown",
    provider: str = "unknown",
) -> bool:
    """Record one failed physical provider call unless an owner already did."""
    if monitor is None or getattr(error, "usage_recorded", False):
        return False
    monitor.record_usage(
        str(model),
        getattr(error, "usage", None),
        role=role,
        provider=provider,
        allow_empty=True,
    )
    try:
        error.usage_recorded = True
    except (AttributeError, TypeError):
        pass
    return True


class MonitoredOpenAIClient:
    """Tiny wrapper that records OpenRouter usage from OpenAI SDK responses."""

    def __init__(self, client: Any, monitor: "RunMonitor | None", *, role: str = "unknown") -> None:
        self._client = client
        self._monitor = monitor
        self._role = role
        # Real OpenAI SDK clients expose base_url on the client object, not on
        # chat.completions; capture it here so the wrapped completions resource
        # can attribute paid calls to the correct provider.
        base_url = getattr(client, "base_url", None)
        self.chat = _MonitoredChat(client.chat, monitor, role=role, base_url=base_url)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _MonitoredChat:
    def __init__(self, chat: Any, monitor: "RunMonitor | None", *, role: str, base_url: Any = None) -> None:
        self._chat = chat
        self.completions = _MonitoredCompletions(chat.completions, monitor, role=role, base_url=base_url)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _MonitoredCompletions:
    def __init__(self, completions: Any, monitor: "RunMonitor | None", *, role: str, base_url: Any = None) -> None:
        self._completions = completions
        self._monitor = monitor
        self._role = role
        self._base_url = base_url

    def create(self, *args: Any, **kwargs: Any) -> Any:
        from suite_tools.call_diagnostics import (
            begin_provider_attempt,
            close_error_best_effort,
            close_success_best_effort,
        )
        from suite_tools.paid_call_lease import paid_call_lease, provider_from_base_url
        from suite_tools.request_receipts import record_effective_request

        diagnostic_context = kwargs.pop("_benchmark_request_context", None)
        if not isinstance(diagnostic_context, dict):
            diagnostic_context = {}
        model = kwargs.get("model") or "unknown"
        output_dir = getattr(self._monitor, "output_dir", None) if self._monitor is not None else None
        module = getattr(self._monitor, "module", None) if self._monitor is not None else None
        run_id = os.environ.get("BENCHMARK_RUN_ID") or (Path(output_dir).name if output_dir is not None else None)
        contract_path = os.environ.get("BENCHMARK_CONTRACT_PATH")
        base_url = self._base_url
        if base_url is None:
            base_url = getattr(self._completions, "base_url", None)
        provider = provider_from_base_url(str(base_url) if base_url is not None else None)
        receipt: dict[str, Any] = {}
        if self._monitor is not None:
            receipt_context = {
                key: value
                for key, value in diagnostic_context.items()
                if key not in {"provider", "role"}
            }
            receipt = record_effective_request(
                self._monitor,
                kwargs,
                base_url=str(base_url) if base_url is not None else None,
                role=self._role,
                provider=provider,
                **receipt_context,
            )
        context = {**receipt, **diagnostic_context}
        attempt = begin_provider_attempt(
            monitor=self._monitor,
            output_dir=output_dir,
            module=module,
            role=self._role,
            model=str(model),
            provider=provider,
            provider_api=context.get("provider_api"),
            context=context,
        )
        provider_invocation_started = False
        try:
            with paid_call_lease(
                provider=provider,
                model=str(model),
                role=self._role,
                module=module,
                output_dir=output_dir,
                run_id=run_id,
                contract_path=contract_path,
            ):
                attempt.mark_provider_invocation_started()
                provider_invocation_started = True
                response = self._completions.create(*args, **kwargs)
        except Exception as exc:
            close_error_best_effort(attempt, exc, self._monitor)
            if provider_invocation_started:
                record_provider_call_error_usage(
                    self._monitor,
                    str(model),
                    exc,
                    role=self._role,
                    provider=provider,
                )
            raise
        close_success_best_effort(attempt, response, self._monitor)
        if self._monitor is not None:
            self._monitor.record_usage(
                str(model),
                response_usage_to_dict(response),
                role=self._role,
                provider=provider,
                allow_empty=True,
            )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with timezone."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with ``os.replace`` so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def classify_failure_status(error: object) -> str:
    """Map provider/adapter errors to run statuses used by all modules."""
    text = str(error).lower()
    status_code = getattr(error, "status_code", None)
    payload = classify_payload(error)
    if (
        payload
        and payload.get("evidence_class") == "environment"
        and payload.get("category") == "billing"
    ):
        return "failed_billing"
    if status_code in (401, 403) or "401" in text or "403" in text or "unauthorized" in text:
        return "failed_auth"
    if (
        status_code == 402
        or "402" in text
        or "insufficient credits" in text
        or "insufficient quota" in text
        or "insufficient_quota" in text
        or "billing_error" in text
        or "credit" in text and "exhaust" in text
    ):
        return "failed_billing"
    if "adapter rejected" in text or "incomplete artifact" in text or "not score-ready" in text:
        return "failed_invalid"
    if status_code in (400, 404) or "not a valid model id" in text or "invalid model" in text:
        return "failed_invalid"
    if status_code == 429 or "429" in text or "rate limit" in text or "too-many-requests" in text:
        return "failed_rate_limited"
    if "missing" in text and "score" in text:
        return "failed_scoring"
    if isinstance(error, httpx.TimeoutException) or re.search(
        r"\b(?:timeout|timed\s+out|deadline\s+exceeded)\b",
        text,
    ):
        return "failed_timeout"
    return "failed_provider"


def is_auth_or_billing_error(error: object) -> bool:
    """Return true when retrying would waste calls or time."""
    return classify_failure_status(error) in {"failed_auth", "failed_billing"}


def is_non_retryable_provider_error(error: object) -> bool:
    """Return true for provider/config errors that retries cannot repair."""
    return classify_failure_status(error) in {
        "failed_auth",
        "failed_billing",
        "failed_invalid",
        "failed_timeout",
    }


class RunMonitor:
    """Append-only run ledger plus latest status snapshot."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        module: str,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.module = module
        self.stage = stage
        self.status_path = self.output_dir / "RUN_STATUS.json"
        self.events_path = self.output_dir / "RUN_EVENTS.jsonl"
        self._lock = threading.RLock()
        self._event_seq = self._existing_event_count()
        lock_path = self.output_dir / _ATTEMPT_LOCK_FILENAME
        lock_fd = None
        deadline = time.monotonic() + _attempt_lock_deadline_seconds()
        while True:
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                # Check whether the lock is stale (left behind by a hard-killed process).
                try:
                    lock_mtime = os.stat(str(lock_path)).st_mtime
                except FileNotFoundError:
                    continue  # disappeared between attempts; retry acquisition
                if time.time() - lock_mtime > _STALE_LOCK_AGE_SECONDS:
                    try:
                        os.unlink(str(lock_path))
                    except FileNotFoundError:
                        pass  # another waiter already cleared it
                    continue  # retry acquisition immediately; don't consume deadline
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"attempt lock stuck at {lock_path}; "
                        "a live monitor may hold it — wait for it to finish "
                        "or remove the lock file if the process has died."
                    )
                time.sleep(0.05)
        try:
            previous_status = self._existing_status()
            takeover = os.environ.get("BENCHMARK_MONITOR_TAKEOVER", "").strip().lower() in ("1", "true", "yes")
            if (
                previous_status.get("status") == "running"
                and previous_status.get("module") == module
                and previous_status.get("stage") == stage
                and not takeover
                and _status_age_seconds(previous_status) < _monitor_stale_seconds()
            ):
                raise RunMonitorConflictError(
                    f"A live monitor owns {self.status_path} "
                    f"(status running, updated {previous_status.get('updated_at')}); "
                    "set BENCHMARK_MONITOR_TAKEOVER=1 to take over."
                )
            best_archived = _max_attempt_for_stage(self.output_dir, module, stage)
            if previous_status:
                previous_status.setdefault("attempt_number", 1)
                self._archive_attempt(previous_status)
                if previous_status.get("module") == module and previous_status.get("stage") == stage:
                    try:
                        best_archived = max(best_archived, int(previous_status["attempt_number"]))
                    except (TypeError, ValueError):
                        best_archived = max(best_archived, 1)
            self.attempt_number = best_archived + 1
            self.status: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "module": module,
                "stage": stage,
                "status": "running",
                "validity": "not_score_ready",
                "output_dir": str(self.output_dir),
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "metadata": metadata or {},
                "counters": {},
                "attempt_number": self.attempt_number,
            }
            if isinstance(previous_status.get("cost"), dict):
                self.status["cost"] = previous_status["cost"]
            if previous_status.get("run_started_at"):
                self.status["run_started_at"] = previous_status["run_started_at"]
            elif previous_status.get("started_at"):
                self.status["run_started_at"] = previous_status["started_at"]
            atomic_write_json(self.status_path, self.status)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
                lock_path.unlink(missing_ok=True)
        self.record(
            "stage_started",
            metadata=metadata or {},
            request_receipt_schema_version="benchmark-effective-request-v1",
        )

    def _existing_event_count(self) -> int:
        if not self.events_path.exists():
            return 0
        try:
            return sum(1 for _ in self.events_path.open())
        except OSError:
            return 0

    def _existing_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            with self.status_path.open() as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _archive_attempt(self, previous_status: dict[str, Any]) -> None:
        """Append the superseded RUN_STATUS snapshot to ATTEMPTS.jsonl."""
        record = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "archived_at": utc_now(),
            "status": previous_status,
        }
        with (self.output_dir / ATTEMPTS_FILENAME).open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _bump(self, key: str, amount: int = 1) -> None:
        counters = self.status.setdefault("counters", {})
        counters[key] = int(counters.get(key, 0)) + amount

    def record(self, event: str, **fields: Any) -> None:
        """Append a ledger event and refresh ``RUN_STATUS.json`` counters.

        ``attempt_failure_classified`` events automatically gain an ``event_id``
        (uuid4 hex) for fact identity (plan 020 D4/D10 — additive; old-shape
        events without event_id remain valid and are tolerated by all readers).
        """
        with self._lock:
            self._event_seq += 1
            sanitized_fields = {
                key: sanitize_ledger_value(value)
                for key, value in fields.items()
            }
            payload = {
                "schema_version": SCHEMA_VERSION,
                "sequence": self._event_seq,
                "timestamp": utc_now(),
                "module": self.module,
                "stage": self.stage,
                "event": event,
                "attempt_number": getattr(self, "attempt_number", 1),
                **sanitized_fields,
            }
            # Stamp event_id on attempt_failure_classified events (plan 020 D4/D10).
            # Additive: event readers that do not know about event_id tolerate absence.
            if event == "attempt_failure_classified" and "event_id" not in payload:
                payload["event_id"] = uuid.uuid4().hex
            with self.events_path.open("a") as f:
                f.write(json.dumps(payload, default=str) + "\n")
            self._bump(f"events.{event}")
            self.status["updated_at"] = payload["timestamp"]
            atomic_write_json(self.status_path, self.status)

    def record_usage(
        self,
        model: str,
        usage: Any,
        *,
        role: str = "unknown",
        provider: str = "unknown",
        allow_empty: bool = False,
    ) -> None:
        """Accumulate paid-call usage/cost in ``RUN_STATUS.json``."""
        normalized = usage_to_dict(usage)
        if not normalized and not allow_empty:
            return

        def existing_float(value: Any) -> float:
            parsed, valid = nonnegative_finite_number(value)
            return parsed if valid else 0.0

        def existing_int(value: Any) -> int:
            parsed, valid = nonnegative_integer(value)
            return parsed if valid else 0

        raw_reported_cost = normalized.get("cost")
        raw_estimated_cost = normalized.get("estimated_cost")
        source_hint = str(normalized.get("cost_source") or "").lower()
        reported_value, reported_valid = nonnegative_finite_number(raw_reported_cost)
        estimated_value, estimated_valid = nonnegative_finite_number(raw_estimated_cost)
        cost_field_is_estimate = reported_valid and (
            "estimate" in source_hint or source_hint == "pricing_snapshot"
        )
        has_reported_cost = reported_valid and not cost_field_is_estimate
        has_estimated_cost = not has_reported_cost and (
            estimated_valid or cost_field_is_estimate
        )
        cost_usd = (
            reported_value
            if has_reported_cost
            else estimated_value
            if estimated_valid
            else reported_value
        )
        source = str(normalized.get("cost_source") or (
            "provider_reported" if has_reported_cost else
            "pricing_snapshot" if has_estimated_cost else
            "unknown"
        ))
        raw_prompt_tokens = normalized.get("prompt_tokens")
        prompt_field = "prompt_tokens"
        if raw_prompt_tokens is None and "input_tokens" in normalized:
            raw_prompt_tokens = normalized.get("input_tokens")
            prompt_field = "input_tokens"
        raw_completion_tokens = normalized.get("completion_tokens")
        completion_field = "completion_tokens"
        if raw_completion_tokens is None and "output_tokens" in normalized:
            raw_completion_tokens = normalized.get("output_tokens")
            completion_field = "output_tokens"
        prompt_tokens, prompt_valid = nonnegative_integer(raw_prompt_tokens)
        completion_tokens, completion_valid = nonnegative_integer(raw_completion_tokens)

        completion_details = normalized.get("completion_tokens_details")
        completion_details = completion_details if isinstance(completion_details, dict) else {}
        output_details = normalized.get("output_tokens_details")
        output_details = output_details if isinstance(output_details, dict) else {}
        thinking_field = None
        raw_thinking_tokens = None
        for candidate_field, candidate_value in (
            ("thoughts_tokens", normalized.get("thoughts_tokens")),
            ("reasoning_tokens", normalized.get("reasoning_tokens")),
            (
                "completion_tokens_details.reasoning_tokens",
                completion_details.get("reasoning_tokens"),
            ),
            (
                "output_tokens_details.reasoning_tokens",
                output_details.get("reasoning_tokens"),
            ),
        ):
            if candidate_value is not None:
                thinking_field = candidate_field
                raw_thinking_tokens = candidate_value
                break
        thinking_tokens, thinking_valid = nonnegative_integer(raw_thinking_tokens)
        if raw_thinking_tokens is None:
            thinking_tokens = 0
            thinking_valid = True

        raw_billable_tokens = normalized.get("billable_completion_tokens")
        explicit_billable_tokens, explicit_billable_valid = nonnegative_integer(
            raw_billable_tokens
        )
        native_thinking_is_separate = normalized.get("thoughts_tokens") is not None
        if native_thinking_is_separate:
            visible_tokens_out = completion_tokens
            derived_billable_tokens = completion_tokens + thinking_tokens
        else:
            derived_billable_tokens = completion_tokens
            visible_tokens_out = max(completion_tokens - thinking_tokens, 0)
        billable_tokens_out = (
            explicit_billable_tokens
            if explicit_billable_valid
            else derived_billable_tokens
        )
        invalid_fields = []
        if raw_reported_cost is not None and not reported_valid:
            invalid_fields.append("cost")
        if raw_estimated_cost is not None and not estimated_valid:
            invalid_fields.append("estimated_cost")
        if has_reported_cost and estimated_valid:
            invalid_fields.append("conflicting_cost_sources")
        if raw_prompt_tokens is not None and not prompt_valid:
            invalid_fields.append(prompt_field)
        if raw_completion_tokens is not None and not completion_valid:
            invalid_fields.append(completion_field)
        if raw_thinking_tokens is not None and not thinking_valid:
            invalid_fields.append(str(thinking_field or "thinking_tokens"))
        if raw_billable_tokens is not None and not explicit_billable_valid:
            invalid_fields.append("billable_completion_tokens")
        if (
            thinking_valid
            and completion_valid
            and not native_thinking_is_separate
            and thinking_tokens > completion_tokens
        ):
            invalid_fields.append("reasoning_tokens_exceed_output")
        if (
            explicit_billable_valid
            and native_thinking_is_separate
            and explicit_billable_tokens != derived_billable_tokens
        ):
            invalid_fields.append("billable_completion_tokens_inconsistent")
        with self._lock:
            cost = self.status.setdefault(
                "cost",
                {
                    "total_cost_usd": 0.0,
                    "total_calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "thinking_tokens_out": 0,
                    "billable_tokens_out": 0,
                    "cost_by_model": {},
                    "cost_by_role": {},
                    "cost_by_stage": {},
                    "cost_by_provider": {},
                    "cost_by_source": {},
                    "usage_by_stage": {},
                    "usage_by_role": {},
                    "usage_by_provider": {},
                    "usage_by_model": {},
                    "usage_by_source": {},
                },
            )
            cost["total_cost_usd"] = round(existing_float(cost.get("total_cost_usd")) + cost_usd, 8)
            cost["total_calls"] = existing_int(cost.get("total_calls")) + 1
            cost["tokens_in"] = existing_int(cost.get("tokens_in")) + prompt_tokens
            cost["tokens_out"] = existing_int(cost.get("tokens_out")) + visible_tokens_out
            cost["thinking_tokens_out"] = existing_int(
                cost.get("thinking_tokens_out")
            ) + thinking_tokens
            cost["billable_tokens_out"] = existing_int(
                cost.get("billable_tokens_out")
            ) + billable_tokens_out
            cost["reported_cost_usd"] = round(
                existing_float(cost.get("reported_cost_usd")) + (cost_usd if has_reported_cost else 0),
                8,
            )
            cost["estimated_cost_usd"] = round(
                existing_float(cost.get("estimated_cost_usd")) + (cost_usd if has_estimated_cost else 0),
                8,
            )
            cost["unknown_cost_calls"] = existing_int(cost.get("unknown_cost_calls")) + (
                0 if has_reported_cost or has_estimated_cost else 1
            )
            if invalid_fields:
                cost["usage_anomaly_count"] = existing_int(
                    cost.get("usage_anomaly_count")
                ) + len(invalid_fields)
                anomaly_map = cost.setdefault("invalid_usage_fields", {})
                for field in invalid_fields:
                    anomaly_map[field] = existing_int(anomaly_map.get(field)) + 1
            by_model = cost.setdefault("cost_by_model", {})
            by_role = cost.setdefault("cost_by_role", {})
            by_model[str(model)] = round(existing_float(by_model.get(str(model))) + cost_usd, 8)
            by_role[str(role)] = round(existing_float(by_role.get(str(role))) + cost_usd, 8)

            dimensions = {
                "stage": self.stage,
                "provider": provider,
                "source": source,
            }
            for dimension, key in dimensions.items():
                mapping = cost.setdefault(f"cost_by_{dimension}", {})
                mapping[str(key)] = round(existing_float(mapping.get(str(key))) + cost_usd, 8)

            usage_dimensions = {
                "stage": self.stage,
                "role": role,
                "provider": provider,
                "model": model,
                "source": source,
            }
            for dimension, key in usage_dimensions.items():
                mapping = cost.setdefault(f"usage_by_{dimension}", {})
                bucket = mapping.setdefault(
                    str(key),
                    {
                        "calls": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "thinking_tokens_out": 0,
                        "billable_tokens_out": 0,
                        "cost_usd": 0.0,
                        "reported_calls": 0,
                        "estimated_calls": 0,
                        "unknown_cost_calls": 0,
                    },
                )
                bucket["calls"] = existing_int(bucket.get("calls")) + 1
                bucket["tokens_in"] = existing_int(bucket.get("tokens_in")) + prompt_tokens
                bucket["tokens_out"] = existing_int(
                    bucket.get("tokens_out")
                ) + visible_tokens_out
                bucket["thinking_tokens_out"] = existing_int(
                    bucket.get("thinking_tokens_out")
                ) + thinking_tokens
                bucket["billable_tokens_out"] = existing_int(
                    bucket.get("billable_tokens_out")
                ) + billable_tokens_out
                bucket["cost_usd"] = round(existing_float(bucket.get("cost_usd")) + cost_usd, 8)
                bucket["reported_calls"] = existing_int(bucket.get("reported_calls")) + int(has_reported_cost)
                bucket["estimated_calls"] = existing_int(bucket.get("estimated_calls")) + int(has_estimated_cost)
                bucket["unknown_cost_calls"] = existing_int(bucket.get("unknown_cost_calls")) + int(
                    not has_reported_cost and not has_estimated_cost
                )
                known_calls = bucket["reported_calls"] + bucket["estimated_calls"]
                bucket["cost_state"] = (
                    "unknown" if known_calls == 0 else
                    "mixed" if bucket["unknown_cost_calls"] else
                    "known"
                )
            self.status["updated_at"] = utc_now()
            atomic_write_json(self.status_path, self.status)

    def record_block(
        self,
        *,
        unit: dict[str, Any],
        evidence: dict[str, Any],
        model: str,
        evidence_pointer: str | None = None,
        unit_id: str | None = None,
        raw_error: object | None = None,
        billed_attempts: int | None = None,
    ) -> None:
        """Append a model_signal block to BLOCKS.jsonl (spec 015 §4, plan 020 D3).

        evidence_pointer names the transcript/artifact file (relative to the
        run dir) that holds the raw evidence for this block.  ``unit_id`` is the
        canonical prepared-contract unit identity; all three benchmark runners
        (aita, epis, sus) pass it.

        New in BLOCKS v2 (plan 020 D3, D10):
        - ``block_id``: uuid4 hex, collision-resistant without cross-process
          coordination (the ATTEMPTS.lock is init-only; runtime writes hold only
          an instance RLock so counters cannot be trusted across processes).
        - ``raw_error``: original provider exception or dict; when present the
          block gains ``raw_body_sha256`` / ``raw_body_excerpt`` / snapshot fields.
        - ``billed_attempts``: explicit caller-supplied count of paid provider
          calls for this block (the loop knows; the exception does not).
        - Plus optional fields from the evidence dict (signal_source, stochastic,
          retry_policy_kind, native_finish_reason, provider, provider_code).

        All new fields are optional; v1 readers that use .get() tolerate them.
        """
        with self._lock:
            # block_id: uuid4 hex — collision-resistant with NO cross-process
            # coordination (plan 020 D3; the ATTEMPTS.lock is init-only, runtime
            # writes hold only an instance RLock so counters cannot be trusted).
            block_id = uuid.uuid4().hex
            snap = make_evidence_snapshot(evidence, raw_error, billed_attempts)
            block = {
                "schema_version": BLOCK_SCHEMA_VERSION,
                "block_id": block_id,
                "timestamp": utc_now(),
                "module": self.module,
                "stage": self.stage,
                "attempt_number": getattr(self, "attempt_number", 1),
                "model": str(model),
                "unit": sanitize_ledger_value(unit),
                "evidence_class": evidence.get("evidence_class"),
                "category": evidence.get("category"),
                "evidence_pointer": evidence_pointer,
            }
            if unit_id is not None:
                block["unit_id"] = unit_id
            # Merge optional snapshot fields (all absent when not computable)
            block.update(snap)
            with (self.output_dir / BLOCKS_FILENAME).open("a") as f:
                f.write(json.dumps(block, default=str) + "\n")
            event_fields = {
                "model": str(model),
                "evidence_class": block["evidence_class"],
                "category": block["category"],
                "evidence_pointer": evidence_pointer,
                **{k: v for k, v in (unit or {}).items() if k in ("item_idx", "side", "scenario", "test_type")},
            }
            if unit_id is not None:
                event_fields["unit_id"] = unit_id
            self.record("block_recorded", **event_fields)

    def mark_completed(self, **fields: Any) -> None:
        """Mark the current stage complete and scored unless overridden."""
        with self._lock:
            sanitized_fields = {
                key: sanitize_ledger_value(value)
                for key, value in fields.items()
            }
            requested_validity = sanitized_fields.pop("validity", "score_ready")
            request_conformance = None
            if requested_validity == "score_ready":
                # Import lazily to keep the ledger primitives independent of
                # contract parsing during module import.
                from suite_tools.request_receipts import evaluate_request_conformance

                request_conformance = evaluate_request_conformance(self.output_dir)
                if request_conformance["requirement_count"]:
                    sanitized_fields["request_conformance"] = request_conformance
                if not request_conformance["conformant"]:
                    requested_validity = "not_score_ready"
            self.status.update(
                {
                    "status": "completed",
                    "validity": requested_validity,
                    "completed_at": utc_now(),
                    "updated_at": utc_now(),
                }
            )
            self.status.update(sanitized_fields)
            atomic_write_json(self.status_path, self.status)
            if request_conformance is not None and not request_conformance["conformant"]:
                self.record(
                    "request_conformance_failed",
                    requirement_count=request_conformance["requirement_count"],
                    receipt_count=request_conformance["receipt_count"],
                    issues=request_conformance["issues"],
                )
            self.record("stage_completed", status=self.status["status"], validity=self.status["validity"])

    def mark_failed(self, error: object, *, status: str | None = None, **fields: Any) -> None:
        """Mark a terminal failure while preserving all artifacts already written."""
        failure_status = status or classify_failure_status(error)
        if failure_status not in TERMINAL_STATUSES:
            failure_status = "failed_provider"
        with self._lock:
            sanitized_fields = {
                key: sanitize_ledger_value(value)
                for key, value in fields.items()
            }
            self.status.update(
                {
                    "status": failure_status,
                    "validity": "not_score_ready",
                    "failed_at": utc_now(),
                    "updated_at": utc_now(),
                    "failure_reason": sanitize_error_message(error),
                }
            )
            self.status.update(sanitized_fields)
            atomic_write_json(self.status_path, self.status)
            self.record(
                "stage_failed",
                status=failure_status,
                failure_reason=sanitize_error_message(error),
                **sanitized_fields,
            )

    def mark_stopped(self, reason: object, **fields: Any) -> None:
        """Mark a cooperative operator stop without treating it as model evidence."""
        with self._lock:
            sanitized_fields = {
                key: sanitize_ledger_value(value)
                for key, value in fields.items()
            }
            self.status.update(
                {
                    "status": "stopped",
                    "validity": "not_score_ready",
                    "stopped_at": utc_now(),
                    "updated_at": utc_now(),
                    "failure_reason": sanitize_error_message(reason),
                }
            )
            self.status.update(sanitized_fields)
            atomic_write_json(self.status_path, self.status)
            self.record(
                "stage_stopped",
                status="stopped",
                reason=sanitize_error_message(reason),
                **sanitized_fields,
            )
