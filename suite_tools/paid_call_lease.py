"""Global paid-call leases for benchmark API calls.

This module is deliberately small and file-backed. Benchmark runners can run
their own work units in parallel, but every paid provider call should briefly
hold one lease so separate processes and agents share one local concurrency
budget.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from suite_tools.run_monitor import atomic_write_json, sanitize_ledger_value, utc_now

REPO_ROOT = Path(__file__).resolve().parents[1]

LEASE_SCHEMA_VERSION = "benchmark-paid-call-lease-v1"
POLICY_SCHEMA_VERSION = "benchmark-paid-call-policy-v1"
PAID_CALL_LIMIT_ENV_NAMES = (
    "BENCHMARK_PAID_CALL_MAX_ACTIVE",
    "BENCHMARK_MAX_ACTIVE_CALLS",
)
LEASE_STATUS_FILENAME = "PAID_CALL_LEASES.json"
LEASE_EVENTS_FILENAME = "PAID_CALL_LEASE_EVENTS.jsonl"
LEASE_LOCK_FILENAME = "PAID_CALL_LEASE_LOCK.json"
POLICY_FILENAME = "PAID_CALL_POLICY.json"

DEFAULT_MAX_ACTIVE_CALLS = 2
DEFAULT_WAIT_TIMEOUT_SECONDS = 600.0
DEFAULT_POLL_SECONDS = 0.2
DEFAULT_STALE_SECONDS = 30 * 60.0
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 30.0
DEFAULT_MAX_RATE_LIMIT_COOLDOWN_SECONDS = 5 * 60.0
DEFAULT_EVENT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_EVENT_SEGMENTS = 3
LEASE_LOCK_ACQUIRE_TIMEOUT_SECONDS = 120.0
WAITER_HEARTBEAT_STALE_SECONDS = 10.0

_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_CONDITIONS: dict[str, threading.Condition] = {}


def _process_lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _process_condition_for(path: Path) -> threading.Condition:
    key = str(path.resolve())
    with _PROCESS_LOCKS_GUARD:
        condition = _PROCESS_CONDITIONS.get(key)
        if condition is None:
            condition = threading.Condition()
            _PROCESS_CONDITIONS[key] = condition
        return condition


class PaidCallLeaseTimeout(RuntimeError):
    """Raised when the global paid-call budget does not free in time."""


@dataclass(frozen=True)
class PaidCallLease:
    """One active paid-call lease."""

    lease_id: str
    provider: str
    model: str
    role: str
    acquired_at: str
    lease_dir: Path
    enabled: bool = True


def default_lease_dir() -> Path:
    """Return the process-wide lease directory."""
    override = os.environ.get("BENCHMARK_PAID_CALL_LEASE_DIR")
    if override:
        return Path(override).expanduser()
    return REPO_ROOT / "results" / "_runtime" / "paid_call_leases"


def _positive_integer(value: Any, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _configured_environment_limit_from(
    environment: Mapping[str, str],
) -> tuple[int | None, str | None]:
    for name in PAID_CALL_LIMIT_ENV_NAMES:
        if name in environment:
            return _positive_integer(environment[name], label=name), name
    return None, None


def _configured_environment_limit() -> int | None:
    return _configured_environment_limit_from(os.environ)[0]


def configured_max_active_calls() -> int:
    """Return the configured environment cap or the conservative default."""
    return _configured_environment_limit() or DEFAULT_MAX_ACTIVE_CALLS


def paid_call_leasing_disabled() -> bool:
    """Return whether cross-process paid-call leasing was explicitly disabled."""
    raw = os.environ.get("BENCHMARK_PAID_CALL_LEASE_DISABLED")
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "BENCHMARK_PAID_CALL_LEASE_DISABLED must be one of 1/0, true/false, yes/no, or on/off"
    )


def configured_wait_timeout_seconds() -> float:
    """Return how long a caller should wait for a lease by default."""
    raw = os.environ.get("BENCHMARK_PAID_CALL_LEASE_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_WAIT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_WAIT_TIMEOUT_SECONDS


def configured_rate_limit_cooldown_seconds() -> float:
    """Return the fallback cooldown after a 429 without a parseable reset time."""
    raw = os.environ.get("BENCHMARK_RATE_LIMIT_COOLDOWN_SECONDS")
    if raw is None:
        return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS


def configured_rate_limit_max_cooldown_seconds() -> float:
    """Return the maximum local pause honored from provider reset metadata."""
    raw = os.environ.get("BENCHMARK_RATE_LIMIT_MAX_COOLDOWN_SECONDS")
    if raw is None:
        return DEFAULT_MAX_RATE_LIMIT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RATE_LIMIT_COOLDOWN_SECONDS


def configured_event_max_bytes() -> int:
    """Return the maximum size of the active diagnostic event segment."""
    raw = os.environ.get("BENCHMARK_PAID_CALL_EVENT_MAX_BYTES")
    if raw is None:
        return DEFAULT_EVENT_MAX_BYTES
    return _positive_limit(raw, DEFAULT_EVENT_MAX_BYTES)


def provider_from_base_url(base_url: str | None) -> str:
    """Return a stable provider label from an OpenAI-compatible base URL."""
    if not base_url:
        return "openrouter"
    hostname = (urlsplit(str(base_url)).hostname or "").lower().rstrip(".")
    if hostname == "openrouter.ai":
        return "openrouter"
    if hostname == "localhost" or hostname == "127.0.0.1" or hostname == "::1":
        return "local"
    if hostname == "api.openai.com":
        return "openai"
    if hostname == "api.anthropic.com":
        return "anthropic"
    if hostname == "generativelanguage.googleapis.com":
        return "google"
    return "openai_compatible"


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


def _age_seconds(value: Any) -> float | None:
    parsed = _parse_time(value)
    if parsed is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _existing_max_active(status_path: Path, fallback: int) -> int:
    state = _load_json(status_path)
    try:
        value = int(state.get("max_active_calls"))
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _positive_limit(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def load_paid_call_policy(lease_dir: Path | str | None = None) -> dict[str, Any]:
    """Read the effective paid-call policy without locking or writing files."""
    manager = PaidCallLeaseManager(lease_dir)
    return manager._load_policy(persist=False)


def paid_call_capacity_report(
    lease_dir: Path | str | None = None,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Describe the effective limit and the policy or environment value binding it."""
    manager = PaidCallLeaseManager(lease_dir)
    stored = _load_json(manager.policy_path)
    environment_limit, environment_variable = _configured_environment_limit_from(
        os.environ if environment is None else environment
    )

    policy_limit: int | None = None
    policy_updated_by: str | None = None
    if stored.get("schema_version") == POLICY_SCHEMA_VERSION:
        try:
            candidate = int(stored.get("global_limit"))
        except (TypeError, ValueError):
            candidate = 0
        if candidate > 0:
            policy_limit = candidate
            policy_updated_by = str(stored.get("updated_by") or "operator")

    if policy_limit is None:
        effective_limit = environment_limit or DEFAULT_MAX_ACTIVE_CALLS
        source = (
            f"environment:{environment_variable}"
            if environment_limit is not None
            else "default"
        )
    elif environment_limit is not None and environment_limit <= policy_limit:
        effective_limit = environment_limit
        source = f"environment:{environment_variable}"
    else:
        effective_limit = policy_limit
        source = f"policy:{policy_updated_by}"

    return {
        "effective_limit": effective_limit,
        "effective_limit_source": source,
        "policy_limit": policy_limit,
        "policy_updated_by": policy_updated_by,
        "environment_limit": environment_limit,
        "environment_variable": environment_variable,
    }


def set_paid_call_policy(
    global_limit: int,
    *,
    lease_dir: Path | str | None = None,
    updated_by: str = "operator",
) -> dict[str, Any]:
    """Explicitly replace the authoritative local paid-call limit."""
    try:
        parsed_limit = int(global_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("global paid-call limit must be a positive integer") from exc
    if parsed_limit <= 0:
        raise ValueError("global paid-call limit must be a positive integer")

    manager = PaidCallLeaseManager(lease_dir)
    with manager._locked():
        previous = manager._load_policy(persist=False)
        policy = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "global_limit": parsed_limit,
            "updated_at": utc_now(),
            "updated_by": str(updated_by or "operator"),
        }
        atomic_write_json(manager.policy_path, policy)
        manager._append_event(
            "capacity_policy_changed",
            {
                "old_global_limit": previous.get("global_limit"),
                "new_global_limit": parsed_limit,
                "updated_by": policy["updated_by"],
            },
        )
        return manager._load_policy(persist=False)


def effective_paid_call_parallelism(
    requested: int | str | None,
    *,
    planned_work: int | None = None,
    lease_dir: Path | str | None = None,
) -> int:
    """Return a bounded local worker count under the authoritative policy."""
    global_limit = int(load_paid_call_policy(lease_dir)["global_limit"])
    requested_limit = (
        global_limit
        if requested is None
        else _positive_integer(requested, label="requested paid-call parallelism")
    )
    limits = [global_limit, requested_limit]
    if planned_work is not None:
        limits.append(_positive_integer(planned_work, label="planned paid-call work"))
    return max(1, min(limits))


def load_paid_call_lease_status(lease_dir: Path | str | None = None) -> dict[str, Any]:
    """Read the latest lease status without taking the writer lock."""
    path = Path(lease_dir) if lease_dir is not None else default_lease_dir()
    manager = PaidCallLeaseManager(path)
    return manager._load_state(manager._global_limit())


def record_rate_limit_cooldown(
    *,
    provider: str = "openrouter",
    model: str = "unknown",
    role: str = "unknown",
    module: str | None = None,
    run_id: str | None = None,
    unit_id: str | None = None,
    error: BaseException | object | None = None,
    headers: dict[str, Any] | None = None,
    lease_dir: Path | str | None = None,
    scope: str = "provider",
) -> dict[str, Any]:
    """Publish a shared rate-limit cooldown without acquiring a lease."""
    manager = PaidCallLeaseManager(lease_dir)
    return manager.record_rate_limit_cooldown(
        provider=provider,
        model=model,
        role=role,
        module=module,
        run_id=run_id,
        unit_id=unit_id,
        error=error,
        headers=headers,
        scope=scope,
    )


def _header_lookup(headers: dict[str, Any], name: str) -> Any:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return value
    return None


def _headers_from_error(error: BaseException | object | None) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if error is None:
        return headers

    for source in (getattr(error, "headers", None), getattr(getattr(error, "response", None), "headers", None)):
        if source is None:
            continue
        try:
            headers.update(dict(source))
        except (TypeError, ValueError):
            pass

    text = str(error)
    for name in ("Retry-After", "X-RateLimit-Reset"):
        pattern = rf"{re.escape(name)}[\"']?\s*:\s*[\"']?([^\"',}}\s]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and name not in headers:
            headers[name] = match.group(1)
    return headers


def is_rate_limit_error(error: BaseException | object | None) -> bool:
    """Return whether an exception or response-like object represents a 429."""
    if error is None:
        return False
    for source in (error, getattr(error, "response", None)):
        status = getattr(source, "status_code", None)
        try:
            if int(status) == 429:
                return True
        except (TypeError, ValueError):
            pass
    text = str(error).lower()
    return "429" in text and ("rate limit" in text or "too-many-requests" in text)


def _parse_reset_delay_seconds(value: Any, *, now: float) -> float | None:
    if value is None:
        return None
    raw = str(value).strip().strip("'\"")
    if not raw:
        return None
    try:
        numeric = float(raw)
    except ValueError:
        parsed = _parse_time(raw)
        if parsed is None:
            return None
        return max(0.0, parsed.timestamp() - now)

    if numeric > 1_000_000_000_000:
        return max(0.0, numeric / 1000.0 - now)
    if numeric > 1_000_000_000:
        return max(0.0, numeric - now)
    return max(0.0, numeric)


def rate_limit_delay_seconds(
    headers: dict[str, Any] | None = None,
    *,
    error: BaseException | object | None = None,
    now: float | None = None,
    default_seconds: float | None = None,
    max_seconds: float | None = None,
) -> float:
    """Return a bounded local cooldown from provider rate-limit metadata."""
    headers = {**(headers or {}), **_headers_from_error(error)}
    current = time.time() if now is None else now
    default = configured_rate_limit_cooldown_seconds() if default_seconds is None else default_seconds
    maximum = configured_rate_limit_max_cooldown_seconds() if max_seconds is None else max_seconds

    for header in ("Retry-After", "X-RateLimit-Reset"):
        delay = _parse_reset_delay_seconds(_header_lookup(headers, header), now=current)
        if delay is not None and delay > 0:
            return min(maximum, delay)
    return min(maximum, max(0.0, default))


class PaidCallLeaseManager:
    """File-backed local paid-call concurrency budget."""

    def __init__(self, lease_dir: Path | str | None = None) -> None:
        self.lease_dir = Path(lease_dir) if lease_dir is not None else default_lease_dir()
        self.status_path = self.lease_dir / LEASE_STATUS_FILENAME
        self.events_path = self.lease_dir / LEASE_EVENTS_FILENAME
        self.lock_path = self.lease_dir / LEASE_LOCK_FILENAME
        self.policy_path = self.lease_dir / POLICY_FILENAME
        self._process_lock = _process_lock_for(self.lock_path)
        self._process_condition = _process_condition_for(self.lock_path)

    def _load_policy(self, *, persist: bool) -> dict[str, Any]:
        policy = _load_json(self.policy_path)
        environment_limit = _configured_environment_limit()
        if policy.get("schema_version") == POLICY_SCHEMA_VERSION:
            try:
                global_limit = int(policy.get("global_limit"))
            except (TypeError, ValueError):
                global_limit = 0
            if global_limit > 0:
                effective_limit = min(global_limit, environment_limit) if environment_limit else global_limit
                normalized = {**policy, "global_limit": effective_limit}
                if effective_limit != global_limit:
                    normalized["operator_global_limit"] = global_limit
                    normalized["environment_limit"] = environment_limit
                return normalized
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "global_limit": environment_limit or DEFAULT_MAX_ACTIVE_CALLS,
            "updated_at": None,
            "updated_by": "environment_default",
        }

    def _global_limit(self) -> int:
        return int(self._load_policy(persist=False)["global_limit"])

    def acquire(
        self,
        *,
        provider: str = "openrouter",
        model: str = "unknown",
        role: str = "unknown",
        module: str | None = None,
        run_id: str | None = None,
        unit_id: str | None = None,
        output_dir: Path | str | None = None,
        contract_path: Path | str | None = None,
        max_active_calls: int | None = None,
        timeout_seconds: float | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> PaidCallLease:
        """Acquire a lease, waiting until the global cap has room.

        ``max_active_calls`` is a diagnostic/probe ceiling compared with the
        total active lease count, not a private reservation for this caller. A
        strict-cap waiter can therefore wait out its timeout under sustained
        higher-cap traffic. That work-conserving behavior is intentional;
        benchmark runners use the shared policy and
        ``effective_paid_call_parallelism`` instead.
        """
        if paid_call_leasing_disabled():
            return PaidCallLease(
                lease_id="lease-disabled",
                provider=provider,
                model=model,
                role=role,
                acquired_at=utc_now(),
                lease_dir=self.lease_dir,
                enabled=False,
            )
        requested_limit = (
            None
            if max_active_calls is None
            else _positive_integer(max_active_calls, label="max_active_calls")
        )

        wait_timeout = configured_wait_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
        if wait_timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.monotonic() + wait_timeout
        expires_at = datetime.fromtimestamp(
            time.time() + wait_timeout,
            tz=timezone.utc,
        ).isoformat()
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        cooldown_wait_logged: set[str] = set()
        acquired = False
        try:
            while True:
                cooldown_sleep_seconds: float | None = None
                with self._locked():
                    global_limit = self._global_limit()
                    max_active = min(global_limit, requested_limit) if requested_limit is not None else global_limit
                    state = self._load_state(global_limit)
                    active = self._active_leases(state)
                    waiters = self._active_waiters(state)
                    current_waiter = next(
                        (item for item in waiters if item.get("lease_id") == lease_id),
                        None,
                    )
                    state_changed = False
                    cooldown = self._matching_cooldown(state, provider=provider, model=model)
                    should_wait = cooldown is not None or len(active) >= max_active or bool(waiters)
                    if current_waiter is None and should_wait:
                        now = utc_now()
                        ticket = int(state.get("next_waiter_ticket") or 1)
                        current_waiter = sanitize_ledger_value({
                            "lease_id": lease_id,
                            "ticket": ticket,
                            "provider": provider,
                            "model": model,
                            "role": role,
                            "module": module,
                            "run_id": run_id,
                            "unit_id": unit_id,
                            "output_dir": str(output_dir) if output_dir is not None else None,
                            "contract_path": str(contract_path) if contract_path is not None else None,
                            "requested_limit": requested_limit,
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "thread_id": threading.get_ident(),
                            "enqueued_at": now,
                            "last_polled_at": now,
                            "expires_at": expires_at,
                        })
                        waiters.append(current_waiter)
                        waiters.sort(key=lambda item: int(item.get("ticket") or 0))
                        state["next_waiter_ticket"] = ticket + 1
                        state_changed = True
                        self._append_event("lease_waiting", {
                            **{key: current_waiter.get(key) for key in (
                                "lease_id", "provider", "model", "role", "module", "run_id", "unit_id",
                            )},
                            "active_count": len(active),
                            "max_active_calls": max_active,
                            "ticket": ticket,
                        })
                    elif current_waiter is not None:
                        heartbeat_age = _age_seconds(current_waiter.get("last_polled_at"))
                        if heartbeat_age is None or heartbeat_age >= 2.0:
                            current_waiter["last_polled_at"] = utc_now()
                            state_changed = True

                    if cooldown is not None:
                        cooldown_sleep_seconds = self._cooldown_remaining_seconds(cooldown)
                        cooldown_id = str(cooldown.get("cooldown_id") or cooldown.get("until") or "")
                        if cooldown_id not in cooldown_wait_logged:
                            self._append_event("rate_limit_waiting", {
                                "lease_id": lease_id,
                                "provider": provider,
                                "model": model,
                                "role": role,
                                "module": module,
                                "run_id": run_id,
                                "unit_id": unit_id,
                                "cooldown_id": cooldown.get("cooldown_id"),
                                "until": cooldown.get("until"),
                                "wait_seconds": cooldown_sleep_seconds,
                            })
                            cooldown_wait_logged.add(cooldown_id)
                    else:
                        eligible_head = next(
                            (
                                item for item in waiters
                                if self._matching_cooldown(
                                    state,
                                    provider=str(item.get("provider") or "openrouter"),
                                    model=str(item.get("model") or "unknown"),
                                ) is None
                                and len(active) < min(
                                    global_limit,
                                    _positive_limit(item.get("requested_limit"), global_limit),
                                )
                            ),
                            None,
                        )
                        current_has_turn = (
                            eligible_head is None
                            or current_waiter is not None and eligible_head.get("lease_id") == lease_id
                        )
                        if len(active) < max_active and current_has_turn:
                            now = utc_now()
                            lease = sanitize_ledger_value({
                                "lease_id": lease_id,
                                "provider": provider,
                                "model": model,
                                "role": role,
                                "module": module,
                                "run_id": run_id,
                                "unit_id": unit_id,
                                "output_dir": str(output_dir) if output_dir is not None else None,
                                "contract_path": str(contract_path) if contract_path is not None else None,
                                "pid": os.getpid(),
                                "host": socket.gethostname(),
                                "thread_id": threading.get_ident(),
                                "acquired_at": now,
                            })
                            active.append(lease)
                            waiters = [item for item in waiters if item.get("lease_id") != lease_id]
                            state["active_leases"] = active
                            state["active_count"] = len(active)
                            state["waiting_leases"] = waiters
                            state["waiting_count"] = len(waiters)
                            state["updated_at"] = now
                            atomic_write_json(self.status_path, state)
                            self._append_event("lease_acquired", lease)
                            acquired = True
                            return PaidCallLease(
                                lease_id=lease_id,
                                provider=provider,
                                model=model,
                                role=role,
                                acquired_at=now,
                                lease_dir=self.lease_dir,
                            )

                    if state_changed:
                        state["active_leases"] = active
                        state["active_count"] = len(active)
                        state["waiting_leases"] = waiters
                        state["waiting_count"] = len(waiters)
                        state["updated_at"] = utc_now()
                        atomic_write_json(self.status_path, state)

                if time.monotonic() >= deadline:
                    self._discard_waiter(lease_id)
                    with self._locked():
                        self._append_event("lease_timeout", {
                            "lease_id": lease_id,
                            "provider": provider,
                            "model": model,
                            "role": role,
                            "module": module,
                            "run_id": run_id,
                            "unit_id": unit_id,
                            "max_active_calls": max_active,
                        })
                    raise PaidCallLeaseTimeout(
                        f"Timed out waiting for paid-call lease "
                        f"({provider}, {model}, role={role}, max_active={max_active})"
                    )
                wait_seconds = max(0.01, poll_seconds)
                if cooldown_sleep_seconds is not None:
                    wait_seconds = min(wait_seconds, max(0.01, cooldown_sleep_seconds))
                self._wait_for_change(wait_seconds)
        finally:
            if not acquired:
                try:
                    self._discard_waiter(lease_id)
                except Exception:
                    # Heartbeat pruning prevents an unclean waiter from blocking the queue.
                    pass

    def _wait_for_change(self, timeout_seconds: float) -> None:
        with self._process_condition:
            self._process_condition.wait(timeout=timeout_seconds)

    def _notify_waiters(self) -> None:
        with self._process_condition:
            self._process_condition.notify_all()

    def _discard_waiter(self, lease_id: str) -> bool:
        with self._locked():
            global_limit = self._global_limit()
            state = self._load_state(global_limit)
            waiters = self._active_waiters(state)
            kept = [item for item in waiters if item.get("lease_id") != lease_id]
            if len(kept) == len(waiters):
                return False
            state["waiting_leases"] = kept
            state["waiting_count"] = len(kept)
            state["updated_at"] = utc_now()
            atomic_write_json(self.status_path, state)
        self._notify_waiters()
        return True

    def release(
        self,
        lease: PaidCallLease,
        *,
        status: str = "completed",
        error: object | None = None,
    ) -> None:
        """Release a previously acquired lease."""
        if not lease.enabled:
            return
        with self._locked():
            max_active = self._global_limit()
            state = self._load_state(max_active)
            active = self._active_leases(state)
            released = None
            kept = []
            for item in active:
                if item.get("lease_id") == lease.lease_id:
                    released = item
                    continue
                kept.append(item)
            state["active_leases"] = kept
            state["active_count"] = len(kept)
            state["updated_at"] = utc_now()
            atomic_write_json(self.status_path, state)
            event = {
                "lease_id": lease.lease_id,
                "provider": lease.provider,
                "model": lease.model,
                "role": lease.role,
                "status": status,
                "duration_seconds": _duration_seconds(lease.acquired_at),
            }
            if released is not None:
                for key in ("module", "run_id", "unit_id", "output_dir", "contract_path", "pid", "thread_id"):
                    event[key] = released.get(key)
            if error is not None:
                event["error"] = str(error)
            if released is None:
                event["missing"] = True
            self._append_event("lease_released", event)
        self._notify_waiters()

    def record_rate_limit_cooldown(
        self,
        *,
        provider: str,
        model: str,
        role: str = "unknown",
        module: str | None = None,
        run_id: str | None = None,
        unit_id: str | None = None,
        error: BaseException | object | None = None,
        headers: dict[str, Any] | None = None,
        scope: str = "provider",
    ) -> dict[str, Any]:
        """Publish a shared cooldown so sibling paid calls pause after a 429."""
        delay_seconds = rate_limit_delay_seconds(headers, error=error)
        now_ts = time.time()
        until_dt = datetime.fromtimestamp(now_ts + delay_seconds, tz=timezone.utc)
        cooldown_model = "*" if scope == "provider" else model
        cooldown = sanitize_ledger_value(
            {
                "cooldown_id": f"cooldown-{uuid.uuid4().hex[:12]}",
                "provider": provider,
                "model": cooldown_model,
                "source_model": model,
                "role": role,
                "module": module,
                "run_id": run_id,
                "unit_id": unit_id,
                "scope": scope,
                "reason": "rate_limit_429",
                "delay_seconds": round(delay_seconds, 3),
                "until": until_dt.isoformat(),
                "created_at": utc_now(),
                "headers": headers or _headers_from_error(error),
                "error": str(error)[:500] if error is not None else None,
            }
        )
        with self._locked():
            max_active = self._global_limit()
            state = self._load_state(max_active)
            cooldowns = [
                item for item in self._active_cooldowns(state)
                if not (
                    item.get("provider") == cooldown["provider"]
                    and item.get("model") == cooldown["model"]
                    and item.get("scope") == cooldown["scope"]
                )
            ]
            cooldowns.append(cooldown)
            state["rate_limit_cooldowns"] = cooldowns
            state["updated_at"] = utc_now()
            atomic_write_json(self.status_path, state)
            self._append_event("rate_limit_cooldown_started", cooldown)
        self._notify_waiters()
        return cooldown

    def _load_state(self, max_active_calls: int) -> dict[str, Any]:
        state = _load_json(self.status_path)
        if state.get("schema_version") != LEASE_SCHEMA_VERSION:
            state = {
                "schema_version": LEASE_SCHEMA_VERSION,
                "max_active_calls": max_active_calls,
                "active_count": 0,
                "active_leases": [],
                "waiting_count": 0,
                "waiting_leases": [],
                "next_waiter_ticket": 1,
                "updated_at": utc_now(),
            }
        state["max_active_calls"] = max_active_calls
        state["active_leases"] = self._active_leases(state)
        state["active_count"] = len(state["active_leases"])
        state["waiting_leases"] = self._active_waiters(state)
        state["waiting_count"] = len(state["waiting_leases"])
        state["next_waiter_ticket"] = max(1, int(state.get("next_waiter_ticket") or 1))
        state["rate_limit_cooldowns"] = self._active_cooldowns(state)
        state["rate_limit_cooldown_count"] = len(state["rate_limit_cooldowns"])
        return state

    def _active_leases(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        leases = state.get("active_leases")
        if not isinstance(leases, list):
            return []
        local_host = socket.gethostname()
        kept = []
        for item in leases:
            if not isinstance(item, dict):
                continue
            pid = item.get("pid")
            age = _age_seconds(item.get("acquired_at"))
            if age is not None and age > DEFAULT_STALE_SECONDS:
                continue
            host = item.get("host")
            if host in (None, "", local_host):
                # Local lease (missing host = pre-host legacy entry, also
                # local): the PID check is authoritative, so a dead holder's
                # lease is reclaimed immediately instead of starving the
                # budget until the stale ceiling.
                if _pid_is_running(pid):
                    kept.append(item)
                continue
            # Foreign-host lease (shared filesystem): the local PID table
            # says nothing about it, so honor it until the stale ceiling.
            if age is not None:
                kept.append(item)
        return kept

    def _active_waiters(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        waiters = state.get("waiting_leases")
        if not isinstance(waiters, list):
            return []
        local_host = socket.gethostname()
        kept: list[dict[str, Any]] = []
        for item in waiters:
            if not isinstance(item, dict):
                continue
            expires_at = _parse_time(item.get("expires_at"))
            if expires_at is not None and expires_at.timestamp() <= time.time():
                continue
            heartbeat_age = _age_seconds(item.get("last_polled_at"))
            if heartbeat_age is not None and heartbeat_age > WAITER_HEARTBEAT_STALE_SECONDS:
                continue
            host = item.get("host")
            if host in (None, "", local_host) and not _pid_is_running(item.get("pid")):
                continue
            kept.append(item)
        return sorted(kept, key=lambda item: int(item.get("ticket") or 0))

    def _active_cooldowns(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        cooldowns = state.get("rate_limit_cooldowns")
        if not isinstance(cooldowns, list):
            return []
        kept: list[dict[str, Any]] = []
        for item in cooldowns:
            if not isinstance(item, dict):
                continue
            remaining = self._cooldown_remaining_seconds(item)
            if remaining is not None and remaining > 0:
                copy = dict(item)
                copy["remaining_seconds"] = round(remaining, 3)
                kept.append(copy)
        return kept

    def _cooldown_remaining_seconds(self, cooldown: dict[str, Any]) -> float | None:
        until = _parse_time(cooldown.get("until"))
        if until is None:
            return None
        return max(0.0, until.timestamp() - time.time())

    def _matching_cooldown(
        self,
        state: dict[str, Any],
        *,
        provider: str,
        model: str,
    ) -> dict[str, Any] | None:
        matches: list[dict[str, Any]] = []
        for cooldown in self._active_cooldowns(state):
            if cooldown.get("provider") != provider:
                continue
            cooldown_model = cooldown.get("model")
            if cooldown_model not in {"*", model}:
                continue
            matches.append(cooldown)
        if not matches:
            return None
        return max(matches, key=lambda item: item.get("until") or "")

    def _append_event(self, event: str, fields: dict[str, Any]) -> None:
        self.lease_dir.mkdir(parents=True, exist_ok=True)
        payload = sanitize_ledger_value(
            {
                "schema_version": LEASE_SCHEMA_VERSION,
                "timestamp": utc_now(),
                "event": event,
                **fields,
            }
        )
        line = json.dumps(payload, default=str) + "\n"
        max_bytes = configured_event_max_bytes()
        try:
            current_bytes = self.events_path.stat().st_size
        except OSError:
            current_bytes = 0
        if current_bytes and current_bytes + len(line.encode("utf-8")) > max_bytes:
            oldest = Path(f"{self.events_path}.{DEFAULT_EVENT_SEGMENTS}")
            try:
                oldest.unlink()
            except FileNotFoundError:
                pass
            for index in range(DEFAULT_EVENT_SEGMENTS - 1, 0, -1):
                source = Path(f"{self.events_path}.{index}")
                if source.exists():
                    source.replace(Path(f"{self.events_path}.{index + 1}"))
            self.events_path.replace(Path(f"{self.events_path}.1"))
        with self.events_path.open("a") as handle:
            handle.write(line)

    @contextmanager
    def _locked(
        self,
        *,
        acquire_timeout_seconds: float = LEASE_LOCK_ACQUIRE_TIMEOUT_SECONDS,
    ) -> Iterator[None]:
        self.lease_dir.mkdir(parents=True, exist_ok=True)
        with self._process_lock:
            lock_fd = None
            started_at = time.monotonic()

            def raise_if_timed_out(existing: dict[str, Any]) -> None:
                if time.monotonic() - started_at < acquire_timeout_seconds:
                    return
                pid = existing.get("pid") if existing else None
                host = existing.get("host") if existing else None
                created_at = existing.get("created_at") if existing else None
                message = (
                    "paid-call lease lock held too long; "
                    f"holder pid={pid if pid is not None else '<unknown>'} "
                    f"host={host if host is not None else '<unknown>'}"
                )
                if created_at:
                    message += f" created_at={created_at}"
                raise TimeoutError(message)

            while lock_fd is None:
                try:
                    lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    existing = _load_json(self.lock_path)
                    if not existing.get("pid"):
                        try:
                            age = time.time() - self.lock_path.stat().st_mtime
                        except OSError:
                            age = 0
                        if age <= 10:
                            raise_if_timed_out(existing)
                            time.sleep(0.02)
                            continue
                    if not _pid_is_running(existing.get("pid")):
                        claim_path = self.lock_path.with_suffix(
                            f".claim-{os.getpid()}-{time.monotonic_ns()}"
                        )
                        try:
                            os.rename(self.lock_path, claim_path)
                        except FileNotFoundError:
                            continue
                        except OSError:
                            raise_if_timed_out(existing)
                            time.sleep(0.02)
                            continue
                        claimed = _load_json(claim_path)
                        if _pid_is_running(claimed.get("pid")):
                            try:
                                os.link(claim_path, self.lock_path)
                            except OSError:
                                pass
                            try:
                                claim_path.unlink()
                            except FileNotFoundError:
                                pass
                            raise_if_timed_out(claimed)
                            time.sleep(0.02)
                            continue
                        try:
                            claim_path.unlink()
                        except FileNotFoundError:
                            pass
                        continue
                    raise_if_timed_out(existing)
                    time.sleep(0.02)
            try:
                with os.fdopen(lock_fd, "w") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "pid": os.getpid(),
                                "host": socket.gethostname(),
                                "created_at": utc_now(),
                            }
                        )
                        + "\n"
                    )
                yield
            finally:
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass


def _duration_seconds(started_at: str) -> float | None:
    started = _parse_time(started_at)
    if started is None:
        return None
    return round((datetime.now(timezone.utc) - started).total_seconds(), 3)


@contextmanager
def paid_call_lease(
    *,
    provider: str = "openrouter",
    model: str = "unknown",
    role: str = "unknown",
    module: str | None = None,
    run_id: str | None = None,
    unit_id: str | None = None,
    output_dir: Path | str | None = None,
    contract_path: Path | str | None = None,
    max_active_calls: int | None = None,
    timeout_seconds: float | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    lease_dir: Path | str | None = None,
) -> Iterator[PaidCallLease]:
    """Hold one paid-call lease around a provider request."""
    manager = PaidCallLeaseManager(lease_dir)
    lease = manager.acquire(
        provider=provider,
        model=model,
        role=role,
        module=module,
        run_id=run_id,
        unit_id=unit_id,
        output_dir=output_dir,
        contract_path=contract_path,
        max_active_calls=max_active_calls,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    try:
        yield lease
    except BaseException as exc:
        status = "failed"
        if is_rate_limit_error(exc):
            status = "rate_limited"
            manager.record_rate_limit_cooldown(
                provider=provider,
                model=model,
                role=role,
                module=module,
                run_id=run_id,
                unit_id=unit_id,
                error=exc,
            )
        manager.release(lease, status=status, error=exc)
        raise
    else:
        manager.release(lease, status="completed")
