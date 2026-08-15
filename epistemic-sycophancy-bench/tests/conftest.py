"""Shared fixtures for epistemic benchmark tests."""

import pytest


@pytest.fixture(autouse=True)
def allow_test_requested_parallelism(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "_paid_call_runtime"))
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "100")
