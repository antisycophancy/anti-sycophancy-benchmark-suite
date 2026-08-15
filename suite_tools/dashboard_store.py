"""Thread-safe stale-while-refresh snapshot store for the local dashboard."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


EMPTY_DASHBOARD_SUMMARY: dict[str, Any] = {
    "schema_version": "benchmark-live-dashboard-v1",
    "groups": [],
    "contracts": [],
    "plans": [],
    "schedulers": [],
    "evidence_feed": [],
    "latest_events": [],
    "flow": {"lanes": []},
    "operational_queue": {"stages": []},
    "summary": {},
}


class DashboardStore:
    """Keep the last good dashboard summary available during refresh work."""

    def __init__(
        self,
        *,
        results_root: Path,
        build_summary: Callable[[], dict[str, Any]],
        source_revision: Callable[[], str],
    ) -> None:
        self.results_root = Path(results_root)
        self._build_summary = build_summary
        self._source_revision = source_revision
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] | None = None
        self._revision: str | None = None
        self._refreshed_at_monotonic: float | None = None
        self._refreshing = False
        self._refresh_error: str | None = None
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None

    def _claim_refresh(self) -> bool:
        with self._lock:
            if self._refreshing:
                return False
            self._refreshing = True
            return True

    def _refresh_claimed(self, *, force: bool) -> bool:
        try:
            revision = self._source_revision()
            with self._lock:
                if not force and self._snapshot is not None and revision == self._revision:
                    self._refresh_error = None
                    return False
            snapshot = self._build_summary()
            if not isinstance(snapshot, dict):
                raise TypeError("dashboard summary builder must return an object")
            with self._lock:
                self._snapshot = snapshot
                self._revision = revision
                self._refreshed_at_monotonic = time.monotonic()
                self._refresh_error = None
            return True
        except Exception as exc:
            with self._lock:
                self._refresh_error = str(exc)[:500]
            return False
        finally:
            with self._lock:
                self._refreshing = False

    def refresh(self, *, force: bool = False) -> bool:
        """Refresh once; retain the prior snapshot when the builder fails."""
        if not self._claim_refresh():
            return False
        return self._refresh_claimed(force=force)

    def refresh_async(self, *, force: bool = False) -> bool:
        """Start a refresh without making the caller wait for filesystem work."""
        if not self._claim_refresh():
            return False
        thread = threading.Thread(
            target=self._refresh_claimed,
            kwargs={"force": force},
            name="dashboard-refresh",
            daemon=True,
        )
        thread.start()
        return True

    def start(self, *, refresh_interval_seconds: float = 2.5) -> None:
        """Build the initial snapshot, then watch for changes in the background."""
        if refresh_interval_seconds <= 0:
            raise ValueError("refresh_interval_seconds must be positive")
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        self.refresh(force=True)

        def loop() -> None:
            while not self._stop_event.wait(refresh_interval_seconds):
                self.refresh()

        self._loop_thread = threading.Thread(
            target=loop,
            name="dashboard-store-loop",
            daemon=True,
        )
        self._loop_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2)
        self._loop_thread = None

    def summary(self) -> dict[str, Any]:
        """Return the last good summary plus refresh-health metadata."""
        with self._lock:
            snapshot = dict(self._snapshot or EMPTY_DASHBOARD_SUMMARY)
            refreshed = self._refreshed_at_monotonic
            snapshot.update(
                {
                    "source_revision": self._revision,
                    "refreshing": self._refreshing,
                    "stale_seconds": (
                        round(max(0.0, time.monotonic() - refreshed), 3)
                        if refreshed is not None
                        else None
                    ),
                    "refresh_error": self._refresh_error,
                    "snapshot_checked_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            return snapshot
