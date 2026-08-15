import json
import subprocess
import sys

import pytest

from suite_tools.paid_call_lease import (
    POLICY_FILENAME,
    POLICY_SCHEMA_VERSION,
    effective_paid_call_parallelism,
    load_paid_call_policy,
    paid_call_capacity_report,
    set_paid_call_policy,
)


def test_capacity_cli_sets_and_reads_authoritative_global_limit(tmp_path):
    set_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suite_tools.capacity",
            "--lease-dir",
            str(tmp_path),
            "--json",
            "set",
            "--global",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert set_result.returncode == 0, set_result.stderr
    assert json.loads(set_result.stdout)["global_limit"] == 3

    get_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suite_tools.capacity",
            "--lease-dir",
            str(tmp_path),
            "--json",
            "get",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert get_result.returncode == 0, get_result.stderr
    assert json.loads(get_result.stdout)["global_limit"] == 3


def test_effective_parallelism_respects_global_requested_and_planned_limits(tmp_path):
    set_paid_call_policy(3, lease_dir=tmp_path)

    assert effective_paid_call_parallelism(10, planned_work=8, lease_dir=tmp_path) == 3
    assert effective_paid_call_parallelism(2, planned_work=8, lease_dir=tmp_path) == 2
    assert effective_paid_call_parallelism(10, planned_work=1, lease_dir=tmp_path) == 1


def test_capacity_report_names_environment_floor_and_authoritative_policy(tmp_path, monkeypatch):
    set_paid_call_policy(64, lease_dir=tmp_path)
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "3")

    report = paid_call_capacity_report(tmp_path)

    assert report == {
        "effective_limit": 3,
        "effective_limit_source": "environment:BENCHMARK_PAID_CALL_MAX_ACTIVE",
        "policy_limit": 64,
        "policy_updated_by": "operator",
        "environment_limit": 3,
        "environment_variable": "BENCHMARK_PAID_CALL_MAX_ACTIVE",
    }


def test_policy_loader_normalizes_numeric_string_without_resetting_limit(tmp_path):
    (tmp_path / POLICY_FILENAME).write_text(json.dumps({
        "schema_version": POLICY_SCHEMA_VERSION,
        "global_limit": "7",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "updated_by": "operator",
    }))

    policy = load_paid_call_policy(tmp_path)

    assert policy["global_limit"] == 7
    assert policy["updated_by"] == "operator"
    assert json.loads((tmp_path / POLICY_FILENAME).read_text())["global_limit"] == "7"


def test_zero_and_malformed_limits_fail_loudly(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="positive integer"):
        effective_paid_call_parallelism(0, lease_dir=tmp_path)

    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "0")
    with pytest.raises(ValueError, match="BENCHMARK_PAID_CALL_MAX_ACTIVE"):
        load_paid_call_policy(tmp_path)

    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "many")
    with pytest.raises(ValueError, match="BENCHMARK_PAID_CALL_MAX_ACTIVE"):
        load_paid_call_policy(tmp_path)
