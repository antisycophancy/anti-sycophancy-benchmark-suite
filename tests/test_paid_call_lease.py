import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from suite_tools.paid_call_lease import (
    LEASE_EVENTS_FILENAME,
    LEASE_SCHEMA_VERSION,
    LEASE_STATUS_FILENAME,
    PaidCallLeaseManager,
    PaidCallLeaseTimeout,
    is_rate_limit_error,
    load_paid_call_policy,
    load_paid_call_lease_status,
    paid_call_lease,
    provider_from_base_url,
    rate_limit_delay_seconds,
    record_rate_limit_cooldown,
    set_paid_call_policy,
)


def test_provider_label_requires_exact_hostname():
    assert provider_from_base_url("https://openrouter.ai/api/v1") == "openrouter"
    assert provider_from_base_url("http://127.0.0.1:8000/v1") == "local"
    assert (
        provider_from_base_url("https://openrouter.ai.attacker.example/api/v1")
        == "openai_compatible"
    )


def test_paid_call_lease_records_acquire_and_release(tmp_path):
    with paid_call_lease(
        lease_dir=tmp_path,
        provider="openrouter",
        model="google/gemini-flash",
        role="model_under_test",
        module="aita",
        run_id="run-1",
        max_active_calls=2,
    ) as lease:
        assert lease.enabled
        status = load_paid_call_lease_status(tmp_path)
        assert status["active_count"] == 1
        assert status["active_leases"][0]["model"] == "google/gemini-flash"
        assert status["active_leases"][0]["role"] == "model_under_test"

    status = json.loads((tmp_path / LEASE_STATUS_FILENAME).read_text())
    assert status["active_count"] == 0
    events = [
        json.loads(line)
        for line in (tmp_path / LEASE_EVENTS_FILENAME).read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["lease_acquired", "lease_released"]
    assert events[1]["module"] == "aita"
    assert events[1]["run_id"] == "run-1"


def test_paid_call_event_log_rotates_at_configured_size(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_EVENT_MAX_BYTES", "400")
    manager = PaidCallLeaseManager(tmp_path)

    for index in range(12):
        manager._append_event("test_event", {"index": index, "detail": "x" * 80})

    rotated = tmp_path / f"{LEASE_EVENTS_FILENAME}.1"
    assert rotated.exists()
    assert manager.events_path.stat().st_size <= 400
    assert '"index": 11' in manager.events_path.read_text()


def test_loading_unchanged_lease_status_does_not_rewrite_registry(tmp_path):
    with paid_call_lease(lease_dir=tmp_path, model="m", max_active_calls=2):
        status_path = tmp_path / LEASE_STATUS_FILENAME
        before = status_path.stat().st_mtime_ns
        load_paid_call_lease_status(tmp_path)
        after = status_path.stat().st_mtime_ns

    assert after == before


def test_paid_call_lease_enforces_global_cap(tmp_path):
    first = PaidCallLeaseManager(tmp_path)
    second = PaidCallLeaseManager(tmp_path)
    lease = first.acquire(model="m1", max_active_calls=1)
    try:
        with pytest.raises(PaidCallLeaseTimeout):
            second.acquire(
                model="m2",
                max_active_calls=1,
                timeout_seconds=0.03,
                poll_seconds=0.01,
            )
    finally:
        first.release(lease)

    status = load_paid_call_lease_status(tmp_path)
    assert status["active_count"] == 0
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / LEASE_EVENTS_FILENAME).read_text().splitlines()
    ]
    assert "lease_waiting" in events
    assert "lease_timeout" in events


def test_waiter_is_removed_when_wait_loop_dies_before_timeout(tmp_path, monkeypatch):
    set_paid_call_policy(1, lease_dir=tmp_path)
    blocker_manager = PaidCallLeaseManager(tmp_path)
    blocker = blocker_manager.acquire(model="blocker")
    waiter = PaidCallLeaseManager(tmp_path)
    monkeypatch.setattr(
        waiter,
        "_wait_for_change",
        lambda _seconds: (_ for _ in ()).throw(RuntimeError("interrupted waiter")),
    )
    try:
        with pytest.raises(RuntimeError, match="interrupted waiter"):
            waiter.acquire(model="waiting", timeout_seconds=1, poll_seconds=0.01)
        assert load_paid_call_lease_status(tmp_path)["waiting_count"] == 0
    finally:
        blocker_manager.release(blocker)


def test_fifo_skips_waiter_blocked_only_by_its_local_cap(tmp_path):
    set_paid_call_policy(4, lease_dir=tmp_path)
    blocker_manager = PaidCallLeaseManager(tmp_path)
    blocker = blocker_manager.acquire(model="blocker")
    acquired = []
    errors = []

    def locally_blocked_waiter():
        manager = PaidCallLeaseManager(tmp_path)
        try:
            lease = manager.acquire(
                model="local-cap-one",
                max_active_calls=1,
                timeout_seconds=2,
                poll_seconds=0.01,
            )
            acquired.append("limited")
            manager.release(lease)
        except Exception as exc:
            errors.append(exc)

    waiting = threading.Thread(target=locally_blocked_waiter)
    waiting.start()
    deadline = time.monotonic() + 1
    while load_paid_call_lease_status(tmp_path)["waiting_count"] != 1:
        if time.monotonic() >= deadline:
            raise AssertionError("local-cap waiter was not enqueued")
        time.sleep(0.01)

    available_manager = PaidCallLeaseManager(tmp_path)
    available = available_manager.acquire(
        model="global-cap-four",
        timeout_seconds=0.5,
        poll_seconds=0.01,
    )
    acquired.append("available")
    available_manager.release(available)
    blocker_manager.release(blocker)
    waiting.join(timeout=2)

    assert errors == []
    assert acquired == ["available", "limited"]


def test_zero_local_cap_is_an_error_not_a_lease_bypass(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        PaidCallLeaseManager(tmp_path).acquire(model="m", max_active_calls=0)


def test_explicit_disabled_flag_is_the_only_lease_bypass(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DISABLED", "1")

    lease = PaidCallLeaseManager(tmp_path).acquire(model="m")

    assert lease.enabled is False
    assert not (tmp_path / LEASE_STATUS_FILENAME).exists()


def test_observer_reads_are_lock_free_and_write_free(tmp_path, monkeypatch):
    set_paid_call_policy(3, lease_dir=tmp_path)
    with paid_call_lease(lease_dir=tmp_path, model="m"):
        pass
    policy_path = tmp_path / "PAID_CALL_POLICY.json"
    status_path = tmp_path / LEASE_STATUS_FILENAME
    mtimes = (policy_path.stat().st_mtime_ns, status_path.stat().st_mtime_ns)

    def fail_lock(*args, **kwargs):
        raise AssertionError("observer attempted to acquire the coordination lock")

    monkeypatch.setattr(PaidCallLeaseManager, "_locked", fail_lock)

    assert load_paid_call_policy(tmp_path)["global_limit"] == 3
    assert load_paid_call_lease_status(tmp_path)["active_count"] == 0
    assert mtimes == (policy_path.stat().st_mtime_ns, status_path.stat().st_mtime_ns)


def test_observer_reads_do_not_create_runtime_directory(tmp_path):
    lease_dir = tmp_path / "missing-runtime"

    assert load_paid_call_policy(lease_dir)["global_limit"] > 0
    assert load_paid_call_lease_status(lease_dir)["active_count"] == 0
    assert not lease_dir.exists()


def test_environment_cap_can_only_lower_operator_policy(tmp_path, monkeypatch):
    set_paid_call_policy(8, lease_dir=tmp_path)
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "3")
    assert load_paid_call_policy(tmp_path)["global_limit"] == 3

    set_paid_call_policy(1, lease_dir=tmp_path)
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "8")
    assert load_paid_call_policy(tmp_path)["global_limit"] == 1


def test_explicit_global_policy_cannot_be_raised_by_caller_request(tmp_path):
    set_paid_call_policy(1, lease_dir=tmp_path)
    first = PaidCallLeaseManager(tmp_path)
    second = PaidCallLeaseManager(tmp_path)
    lease = first.acquire(model="m1", max_active_calls=1)
    try:
        with pytest.raises(PaidCallLeaseTimeout):
            second.acquire(
                model="m2",
                max_active_calls=2,
                timeout_seconds=0.03,
                poll_seconds=0.01,
            )
    finally:
        first.release(lease)

    status = load_paid_call_lease_status(tmp_path)
    assert status["max_active_calls"] == 1


def test_waiting_leases_are_visible_and_admitted_fifo(tmp_path):
    set_paid_call_policy(1, lease_dir=tmp_path)
    blocker_manager = PaidCallLeaseManager(tmp_path)
    blocker = blocker_manager.acquire(model="blocker")
    acquired_order = []

    def wait_for_waiters(expected):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if load_paid_call_lease_status(tmp_path).get("waiting_count") == expected:
                return
            time.sleep(0.01)
        raise AssertionError(f"expected {expected} visible waiters")

    def worker(run_id):
        manager = PaidCallLeaseManager(tmp_path)
        lease = manager.acquire(
            model="m",
            run_id=run_id,
            timeout_seconds=1,
            poll_seconds=0.01,
        )
        acquired_order.append(run_id)
        time.sleep(0.02)
        manager.release(lease)

    first = threading.Thread(target=worker, args=("run-a",))
    second = threading.Thread(target=worker, args=("run-b",))
    first.start()
    wait_for_waiters(1)
    second.start()
    wait_for_waiters(2)
    blocker_manager.release(blocker)
    first.join()
    second.join()

    assert acquired_order == ["run-a", "run-b"]
    assert load_paid_call_lease_status(tmp_path)["waiting_count"] == 0


def test_rate_limited_waiter_does_not_block_other_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_RATE_LIMIT_MAX_COOLDOWN_SECONDS", "10")
    monkeypatch.setenv("BENCHMARK_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    set_paid_call_policy(1, lease_dir=tmp_path)
    blocker_manager = PaidCallLeaseManager(tmp_path)
    blocker = blocker_manager.acquire(provider="local", model="blocker")
    acquired_order = []
    available_acquired = threading.Event()

    def wait_for_waiters(expected):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if load_paid_call_lease_status(tmp_path).get("waiting_count") == expected:
                return
            time.sleep(0.01)
        raise AssertionError(f"expected {expected} visible waiters")

    def worker(provider, run_id):
        manager = PaidCallLeaseManager(tmp_path)
        lease = manager.acquire(
            provider=provider,
            model=f"{provider}/model",
            run_id=run_id,
            timeout_seconds=10,
            poll_seconds=0.01,
        )
        acquired_order.append(run_id)
        if run_id == "available":
            available_acquired.set()
        time.sleep(0.02)
        manager.release(lease)

    limited = threading.Thread(target=worker, args=("openrouter", "limited"))
    available = threading.Thread(target=worker, args=("google", "available"))
    limited.start()
    wait_for_waiters(1)
    available.start()
    wait_for_waiters(2)
    record_rate_limit_cooldown(
        lease_dir=tmp_path,
        provider="openrouter",
        model="limited/model",
        headers={"Retry-After": "5"},
    )
    blocker_manager.release(blocker)

    assert available_acquired.wait(timeout=5)
    assert acquired_order == ["available"]

    # Replace the provider cooldown with an immediate reset so the limited
    # worker can finish without making this test sleep for five seconds.
    record_rate_limit_cooldown(
        lease_dir=tmp_path,
        provider="openrouter",
        model="limited/model",
        headers={"Retry-After": "0"},
    )
    limited.join(timeout=5)
    available.join(timeout=5)

    assert acquired_order == ["available", "limited"]
    assert not limited.is_alive()
    assert not available.is_alive()


def test_one_hundred_workers_never_overshoot_global_limit(tmp_path):
    set_paid_call_policy(8, lease_dir=tmp_path)
    active = 0
    max_seen = 0
    completed = []
    errors = []
    guard = threading.Lock()

    def worker(index):
        nonlocal active, max_seen
        try:
            with paid_call_lease(
                lease_dir=tmp_path,
                provider="local",
                model="fake/model",
                run_id=f"run-{index % 2}",
                unit_id=str(index),
                # This test asserts global exclusion, not scheduler speed.
                # Shared CI hosts can take more than five seconds to start and
                # drain one hundred filesystem-coordinated worker threads.
                timeout_seconds=30,
                poll_seconds=0.005,
            ):
                with guard:
                    active += 1
                    max_seen = max(max_seen, active)
                time.sleep(0.005)
                with guard:
                    active -= 1
                    completed.append(index)
        except Exception as exc:  # The assertion below reports worker failures cleanly.
            with guard:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    status = load_paid_call_lease_status(tmp_path)
    assert errors == []
    assert len(completed) == 100
    assert max_seen <= 8
    assert status["active_count"] == 0
    assert status["waiting_count"] == 0


def test_paid_call_lease_serializes_across_threads(tmp_path):
    active = 0
    max_seen = 0
    lock = threading.Lock()

    def worker():
        nonlocal active, max_seen
        with paid_call_lease(lease_dir=tmp_path, model="m", max_active_calls=1):
            with lock:
                active += 1
                max_seen = max(max_seen, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_seen == 1
    assert load_paid_call_lease_status(tmp_path)["active_count"] == 0


def test_paid_call_lease_lock_stale_dead_pid_takeover_race_serializes_once(tmp_path, monkeypatch):
    manager = PaidCallLeaseManager(tmp_path)
    manager.lock_path.write_text(
        json.dumps(
            {
                "pid": _dead_pid(),
                "host": socket.gethostname(),
                "created_at": _now_iso(),
            }
        )
        + "\n"
    )
    active = 0
    max_seen = 0
    entered = 0
    guard = threading.Lock()
    successful_rename_claims = 0
    real_rename = os.rename

    def counting_rename(src, dst):
        nonlocal successful_rename_claims
        result = real_rename(src, dst)
        if src == manager.lock_path:
            with guard:
                successful_rename_claims += 1
        return result

    monkeypatch.setattr(os, "rename", counting_rename)

    def worker():
        nonlocal active, max_seen, entered
        local_manager = PaidCallLeaseManager(tmp_path)
        with local_manager._locked():
            with guard:
                active += 1
                entered += 1
                max_seen = max(max_seen, active)
            time.sleep(0.03)
            with guard:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert entered == 2
    assert max_seen == 1
    assert successful_rename_claims == 1
    assert not manager.lock_path.exists()
    assert list(tmp_path.glob("PAID_CALL_LEASE_LOCK.claim-*")) == []


def test_paid_call_lease_lock_live_pid_timeout_propagates(tmp_path):
    manager = PaidCallLeaseManager(tmp_path)
    manager.lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at": _now_iso(),
            }
        )
        + "\n"
    )

    with pytest.raises(TimeoutError) as exc_info:
        with manager._locked(acquire_timeout_seconds=0.03):
            pass

    message = str(exc_info.value)
    assert "paid-call lease lock held too long" in message
    assert f"holder pid={os.getpid()}" in message
    assert f"host={socket.gethostname()}" in message


def _dead_pid() -> int:
    proc = subprocess.Popen(["sleep", "0"])
    proc.wait()
    return proc.pid


def _write_lease_state(tmp_path, leases) -> None:
    state = {
        "schema_version": LEASE_SCHEMA_VERSION,
        "max_active_calls": 2,
        "active_count": len(leases),
        "active_leases": leases,
        "updated_at": "2026-06-10T00:00:00+00:00",
    }
    (tmp_path / LEASE_STATUS_FILENAME).write_text(json.dumps(state))


def _lease_entry(*, pid, acquired_at, host=None) -> dict:
    entry = {
        "lease_id": "lease-deadbeef0000",
        "provider": "openrouter",
        "model": "m",
        "role": "model_under_test",
        "pid": pid,
        "acquired_at": acquired_at,
    }
    if host is not None:
        entry["host"] = host
    return entry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_paid_call_lease_reclaims_dead_local_pid_immediately(tmp_path):
    _write_lease_state(
        tmp_path,
        [_lease_entry(pid=_dead_pid(), acquired_at=_now_iso(), host=socket.gethostname())],
    )

    status = load_paid_call_lease_status(tmp_path)

    assert status["active_count"] == 0
    assert status["active_leases"] == []


def test_paid_call_lease_reclaims_legacy_dead_pid_lease_without_host(tmp_path):
    _write_lease_state(
        tmp_path,
        [_lease_entry(pid=_dead_pid(), acquired_at=_now_iso())],
    )

    status = load_paid_call_lease_status(tmp_path)

    assert status["active_count"] == 0


def test_paid_call_lease_keeps_foreign_host_lease_until_stale(tmp_path):
    _write_lease_state(
        tmp_path,
        [_lease_entry(pid=_dead_pid(), acquired_at=_now_iso(), host="some-other-host")],
    )

    status = load_paid_call_lease_status(tmp_path)

    assert status["active_count"] == 1


def test_paid_call_lease_drops_stale_foreign_host_lease(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _write_lease_state(
        tmp_path,
        [_lease_entry(pid=_dead_pid(), acquired_at=old, host="some-other-host")],
    )

    status = load_paid_call_lease_status(tmp_path)

    assert status["active_count"] == 0


def test_paid_call_lease_records_host_on_acquire(tmp_path):
    with paid_call_lease(lease_dir=tmp_path, model="m", max_active_calls=2):
        status = load_paid_call_lease_status(tmp_path)
        assert status["active_leases"][0]["host"] == socket.gethostname()


def test_rate_limit_delay_parses_openrouter_reset_epoch_ms():
    reset_ms = str(int((1_700_000_010.0) * 1000))

    assert rate_limit_delay_seconds(
        {"X-RateLimit-Reset": reset_ms},
        now=1_700_000_000.0,
        default_seconds=30,
        max_seconds=60,
    ) == 10


def test_paid_call_lease_records_cooldown_on_429_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_RATE_LIMIT_MAX_COOLDOWN_SECONDS", "1")

    class RateLimitError(RuntimeError):
        status_code = 429
        headers = {"Retry-After": "0.25"}

    assert is_rate_limit_error(RateLimitError("rate limit exceeded"))

    with pytest.raises(RateLimitError):
        with paid_call_lease(
            lease_dir=tmp_path,
            provider="openrouter",
            model="google/gemini-flash",
            role="model_under_test",
            max_active_calls=2,
        ):
            raise RateLimitError("rate limit exceeded")

    status = load_paid_call_lease_status(tmp_path)
    assert status["active_count"] == 0
    assert status["rate_limit_cooldown_count"] == 1
    assert status["rate_limit_cooldowns"][0]["provider"] == "openrouter"
    assert status["rate_limit_cooldowns"][0]["model"] == "*"
    events = [
        json.loads(line)
        for line in (tmp_path / LEASE_EVENTS_FILENAME).read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "lease_acquired",
        "rate_limit_cooldown_started",
        "lease_released",
    ]
    assert events[-1]["status"] == "rate_limited"


def test_paid_call_lease_waits_for_shared_rate_limit_cooldown(tmp_path):
    record_rate_limit_cooldown(
        lease_dir=tmp_path,
        provider="openrouter",
        model="google/gemini-flash",
        role="model_under_test",
        headers={"Retry-After": "0.04"},
    )

    start = time.monotonic()
    with paid_call_lease(
        lease_dir=tmp_path,
        provider="openrouter",
        model="anthropic/claude-sonnet",
        role="model_under_test",
        max_active_calls=1,
        timeout_seconds=1,
        poll_seconds=0.01,
    ):
        pass

    assert time.monotonic() - start >= 0.03
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / LEASE_EVENTS_FILENAME).read_text().splitlines()
    ]
    assert "rate_limit_waiting" in events
    assert events[-2:] == ["lease_acquired", "lease_released"]
