from pathlib import Path
import threading
import time

from suite_tools.dashboard_store import DashboardStore


def test_dashboard_store_serves_last_good_snapshot_when_refresh_fails(tmp_path):
    revision = ["r1"]
    should_fail = [False]

    def build_summary():
        if should_fail[0]:
            raise ValueError("malformed status")
        return {"groups": [{"id": revision[0]}]}

    store = DashboardStore(
        results_root=Path(tmp_path),
        build_summary=build_summary,
        source_revision=lambda: revision[0],
    )
    store.refresh()
    first = store.summary()

    revision[0] = "r2"
    should_fail[0] = True
    store.refresh()
    failed = store.summary()

    assert first["source_revision"] == "r1"
    assert failed["source_revision"] == "r1"
    assert failed["groups"] == [{"id": "r1"}]
    assert failed["refresh_error"] == "malformed status"


def test_dashboard_store_initial_failure_still_serves_a_safe_empty_summary(tmp_path):
    def build_summary():
        raise ValueError("initial build failed")

    store = DashboardStore(
        results_root=Path(tmp_path),
        build_summary=build_summary,
        source_revision=lambda: "r1",
    )

    assert store.refresh() is False
    summary = store.summary()

    assert summary["groups"] == []
    assert summary["contracts"] == []
    assert summary["plans"] == []
    assert summary["schedulers"] == []
    assert summary["flow"] == {"lanes": []}
    assert summary["operational_queue"] == {"stages": []}
    assert summary["summary"] == {}
    assert summary["refresh_error"] == "initial build failed"


def test_dashboard_store_refreshes_in_background_without_blocking_readers(tmp_path):
    revision = ["r1"]
    entered = threading.Event()
    release = threading.Event()

    def build_summary():
        if revision[0] == "r2":
            entered.set()
            release.wait(timeout=2)
        return {"groups": [{"id": revision[0]}]}

    store = DashboardStore(
        results_root=Path(tmp_path),
        build_summary=build_summary,
        source_revision=lambda: revision[0],
    )
    store.refresh()
    revision[0] = "r2"

    started = time.monotonic()
    assert store.refresh_async() is True
    assert time.monotonic() - started < 0.05
    assert entered.wait(timeout=1)
    during = store.summary()

    assert during["refreshing"] is True
    assert during["groups"] == [{"id": "r1"}]

    release.set()
    deadline = time.monotonic() + 2
    while store.summary()["source_revision"] != "r2" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.summary()["groups"] == [{"id": "r2"}]


def test_dashboard_store_background_loop_discovers_source_changes(tmp_path):
    revision = ["r1"]
    store = DashboardStore(
        results_root=Path(tmp_path),
        build_summary=lambda: {"groups": [{"id": revision[0]}]},
        source_revision=lambda: revision[0],
    )

    store.start(refresh_interval_seconds=0.02)
    try:
        assert store.summary()["source_revision"] == "r1"
        revision[0] = "r2"
        deadline = time.monotonic() + 1
        while store.summary()["source_revision"] != "r2" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.summary()["source_revision"] == "r2"
    finally:
        store.close()
