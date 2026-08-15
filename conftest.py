"""Repository-wide test isolation for paid-call coordination state."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_paid_call_runtime(tmp_path, monkeypatch):
    """Keep unit tests away from the operator's machine-global lease registry."""
    monkeypatch.setenv(
        "BENCHMARK_PAID_CALL_LEASE_DIR",
        str(tmp_path / "_paid_call_runtime"),
    )
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "100")
    monkeypatch.delenv("BENCHMARK_MAX_ACTIVE_CALLS", raising=False)
    monkeypatch.delenv("BENCHMARK_PAID_CALL_LEASE_DISABLED", raising=False)
