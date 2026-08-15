import json
import shlex
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import suite_tools.scheduler as scheduler_module
from suite_tools.cost_estimate import build_contract_call_plan
from suite_tools.paid_call_lease import load_paid_call_policy
from suite_tools.preflight_conditions import (
    ProbeResult,
    collect_prepared_run_context,
    preflight_receipt_fingerprint,
    write_preflight_receipt,
)
from suite_tools.prepare_run import (
    _attach_cost_estimate,
    prepare_aita_run,
    prepare_epis_run,
    prepare_sus_run,
)
from suite_tools.run_contract import (
    CONTRACT_SCHEMA_VERSION,
    load_run_contract,
    load_run_control,
    write_run_contract,
)
from suite_tools.scheduler import (
    DUPLICATE_SCHEDULER_EXIT_CODE,
    REPO_ROOT,
    SCHEDULER_LOCK_FILENAME,
    SCHEDULER_SCHEMA_VERSION,
    _run_shell_command,
    build_progress_snapshot,
    load_scheduler_events,
    load_scheduler_status,
    main,
    run_contract,
    run_contracts,
    score_contract,
)
from suite_tools.sealed_pack import seal_files


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


@pytest.fixture(autouse=True)
def _allow_arbitrary_scheduler_commands_in_legacy_tests(monkeypatch):
    """Old scheduler fixtures execute tiny Python snippets, never providers."""
    monkeypatch.setenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", "1")


def _bind_test_contract_commands(contract_path):
    contract = load_run_contract(contract_path)
    identity = dict(contract.get("identity") or {})
    execution = dict(identity.get("execution") or {})
    execution["prepared_commands"] = {
        "schema_version": "benchmark-prepared-commands-v1",
        "execute_steps": contract.get("execute_steps") or [],
        "score_steps": contract.get("score_steps") or [],
    }
    identity["execution"] = execution
    contract["identity"] = identity
    contract.pop("provenance", None)
    return write_run_contract(contract_path.parent, contract)


def _bind_test_contract_pricing(contract_path):
    contract = load_run_contract(contract_path)
    call_plan = build_contract_call_plan(contract)
    identity = dict(contract.get("identity") or {})
    execution = dict(identity.get("execution") or {})
    execution["prepared_pricing"] = {
        "schema_version": "benchmark-prepared-pricing-v1",
        "call_plan": call_plan,
    }
    identity["execution"] = execution
    contract["identity"] = identity
    contract["call_plan"] = call_plan
    contract.pop("provenance", None)
    return write_run_contract(contract_path.parent, contract)


def _attach_test_cost_estimate(contract_path):
    return _attach_cost_estimate(
        contract_path,
        {
            "schema_version": "benchmark-pricing-snapshot-v1",
            "units": "per_token",
            "models": {},
        },
    )


def _attach_passing_preflight_receipt(contract_path):
    context = collect_prepared_run_context(contract_path.parent)
    write_preflight_receipt(
        context,
        [
            ProbeResult(
                target,
                "PASS",
                200,
                "accepted",
                reason_code="accepted",
            )
            for target in context.targets
        ],
    )
    return contract_path


def test_scheduler_terminates_live_child_when_polling_is_interrupted(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=43210, poll=lambda: None)
    terminated = []
    events = []
    ledger = SimpleNamespace(
        contract_path=tmp_path / "sus" / "RUN_CONTRACT.json",
        contract={"run_id": "interrupt-test", "modules": [{"module": "sus"}]},
        status={"run_id": "interrupt-test"},
        event=lambda event, **fields: events.append({"event": event, **fields}),
        refresh=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(scheduler_module, "_run_command_step", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        scheduler_module,
        "_terminate_process_tree",
        lambda child: terminated.append(child.pid),
    )
    monkeypatch.setattr(
        scheduler_module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        scheduler_module._run_command_with_ledger(
            ledger=ledger,
            steps=[scheduler_module.CommandStep(cwd=tmp_path, argv=("sleep", "60"))],
            event_prefix="generation",
            poll_seconds=0.1,
            stop_on_attention=False,
            stream_command_output=False,
        )

    assert terminated == [43210]


def test_child_cleanup_escalates_from_terminate_to_kill(monkeypatch):
    class StuckProcess:
        pid = 43210

        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            if not self.killed:
                raise scheduler_module.subprocess.TimeoutExpired("child", timeout)
            return -9

    process = StuckProcess()
    monkeypatch.setattr(
        scheduler_module.os,
        "killpg",
        lambda *args: (_ for _ in ()).throw(ProcessLookupError()),
    )

    scheduler_module._terminate_process_tree(process, grace_seconds=0.01)

    assert process.terminated is True
    assert process.killed is True


def _contract(tmp_path, *, command: str, module: str = "sus", score_command: str | None = None):
    module_dir = tmp_path / module
    module_dir.mkdir(exist_ok=True)
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "run_id": "scheduler-test",
        "lifecycle_state": "prepared",
        "execute_command": command,
        "execute_cwd": str(REPO_ROOT),
        "execute_argv": ["/bin/sh", "-c", command],
        "execute_steps": [{"cwd": str(REPO_ROOT), "argv": ["/bin/sh", "-c", command]}],
        "expected_models": [{"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}],
        "modules": [
            {
                "module": module,
                "stage": "run",
                "output_dir": ".",
                "expected_units": [
                    {"unit_id": f"{module}:unit0"},
                    {"unit_id": f"{module}:unit1"},
                ],
                "expected_artifacts": [
                    {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "diagnostic"},
                ],
            }
        ],
    }
    if score_command:
        contract["score_command"] = score_command
        contract["score_cwd"] = str(REPO_ROOT)
        contract["score_argv"] = ["/bin/sh", "-c", score_command]
        contract["score_steps"] = [
            {"cwd": str(REPO_ROOT), "argv": ["/bin/sh", "-c", score_command]}
        ]
    return _bind_test_contract_pricing(write_run_contract(module_dir, contract))


def _structured_contract(
    tmp_path,
    *,
    execute_argv: list[str],
    module: str = "sus",
    execute_cwd=None,
    score_steps: list[dict] | None = None,
):
    module_dir = tmp_path / module
    module_dir.mkdir(exist_ok=True)
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "run_id": "scheduler-structured-test",
        "lifecycle_state": "prepared",
        "execute_command": "legacy display only; should not execute",
        "execute_cwd": str(execute_cwd or tmp_path),
        "execute_argv": execute_argv,
        "execute_steps": [{"cwd": str(execute_cwd or tmp_path), "argv": execute_argv}],
        "expected_models": [{"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}],
        "modules": [
            {
                "module": module,
                "stage": "run",
                "output_dir": ".",
                "expected_units": [{"unit_id": f"{module}:unit0"}],
                "expected_artifacts": [
                    {"kind": "run_status", "path": "RUN_STATUS.json", "required_for": "diagnostic"},
                ],
            }
        ],
    }
    if score_steps is not None:
        contract["score_steps"] = score_steps
    return _bind_test_contract_pricing(write_run_contract(module_dir, contract))


def test_child_commands_receive_absolute_paid_call_lease_dir(tmp_path, monkeypatch):
    child_cwd = tmp_path / "child"
    child_cwd.mkdir()
    output_path = tmp_path / "lease-dir.txt"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", "results/test-lease-dir")
    code = (
        "import os, pathlib; "
        f"pathlib.Path({str(output_path)!r}).write_text(os.environ['BENCHMARK_PAID_CALL_LEASE_DIR'])"
    )

    process = _run_shell_command(
        f"cd {shlex.quote(str(child_cwd))}\n{_python_command(code)}",
        cwd=tmp_path,
        stream_output=False,
    )

    assert process.wait(timeout=10) == 0
    assert output_path.read_text() == str((REPO_ROOT / "results/test-lease-dir").resolve())


def test_child_commands_receive_max_active_paid_call_cap(tmp_path):
    output_path = tmp_path / "max-active.txt"
    code = (
        "import os, pathlib; "
        f"pathlib.Path({str(output_path)!r}).write_text("
        "os.environ['BENCHMARK_PAID_CALL_MAX_ACTIVE'] + ':' + "
        "os.environ['BENCHMARK_MAX_ACTIVE_CALLS'] + ':' + "
        "os.environ['BENCHMARK_GENERATION_MAX_PARALLEL'] + ':' + "
        "os.environ['BENCHMARK_SCORE_MAX_PARALLEL'])"
    )

    process = _run_shell_command(
        _python_command(code),
        cwd=tmp_path,
        stream_output=False,
        max_active_calls=3,
    )

    assert process.wait(timeout=10) == 0
    assert output_path.read_text() == "3:3:3:3"


def test_scheduler_explicit_cap_does_not_raise_operator_policy(tmp_path, monkeypatch):
    lease_dir = tmp_path / "leases"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    from suite_tools.paid_call_lease import set_paid_call_policy
    set_paid_call_policy(1, lease_dir=lease_dir)
    module_dir = tmp_path / "sus"
    command = _python_command(
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'sus','stage':'run','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts}))"
    )
    contract_path = _contract(tmp_path, command=command)

    assert run_contract(
        contract_path,
        poll_seconds=0.02,
        max_active_calls=3,
        stream_command_output=False,
    ) == 0

    assert load_paid_call_policy(lease_dir)["global_limit"] == 1


def test_child_commands_receive_benchmark_context(tmp_path):
    output_path = tmp_path / "benchmark-context.json"
    code = (
        "import json, os, pathlib; "
        f"pathlib.Path({str(output_path)!r}).write_text(json.dumps({{"
        "'run_id': os.environ['BENCHMARK_RUN_ID'], "
        "'module': os.environ['BENCHMARK_MODULE'], "
        "'output_dir': os.environ['BENCHMARK_OUTPUT_DIR'], "
        "'contract_path': os.environ['BENCHMARK_CONTRACT_PATH']"
        "}, sort_keys=True))"
    )

    process = _run_shell_command(
        _python_command(code),
        cwd=tmp_path,
        stream_output=False,
        benchmark_context={
            "BENCHMARK_RUN_ID": "run-123",
            "BENCHMARK_MODULE": "sus",
            "BENCHMARK_OUTPUT_DIR": tmp_path / "sus",
            "BENCHMARK_CONTRACT_PATH": tmp_path / "sus" / "RUN_CONTRACT.json",
        },
    )

    assert process.wait(timeout=10) == 0
    assert json.loads(output_path.read_text()) == {
        "contract_path": str(tmp_path / "sus" / "RUN_CONTRACT.json"),
        "module": "sus",
        "output_dir": str(tmp_path / "sus"),
        "run_id": "run-123",
    }


def test_scheduler_prefers_structured_argv_without_shell_expansion(tmp_path, capsys):
    module_dir = tmp_path / "sus"
    marker = tmp_path / "should-not-exist"
    argv_seen = tmp_path / "argv-seen.txt"
    code = (
        "import json, pathlib, datetime, sys; "
        f"pathlib.Path({str(argv_seen)!r}).write_text(sys.argv[1]); "
        f"p=pathlib.Path({str(module_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'sus','stage':'run','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "}))"
    )
    contract_path = _structured_contract(
        tmp_path,
        execute_argv=[sys.executable, "-c", code, f"literal; touch {marker}"],
    )

    exit_code = run_contract(
        contract_path,
        poll_seconds=0.1,
        stream_command_output=False,
    )

    assert exit_code == 0
    assert argv_seen.read_text() == f"literal; touch {marker}"
    assert marker.exists() is False
    assert "unsafe arbitrary contract commands enabled" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["/bin/sh", "-c", "exit 0"],
        [sys.executable, "-c", "raise SystemExit(0)"],
        [sys.executable, "-m", "os", "run"],
        [sys.executable, "-m", "sus_bench", "unknown-subcommand"],
    ],
)
def test_scheduler_secure_default_rejects_unapproved_command_shapes(
    tmp_path,
    monkeypatch,
    argv,
):
    contract_path = _structured_contract(
        tmp_path,
        execute_argv=argv,
        execute_cwd=REPO_ROOT / "sus-bench",
        score_steps=[{
            "cwd": str(REPO_ROOT / "sus-bench"),
            "argv": [sys.executable, "-m", "sus_bench", "score"],
        }],
    )
    _bind_test_contract_commands(contract_path)
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)
    monkeypatch.setattr(
        scheduler_module,
        "_run_command_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not spawn an unsafe command")
        ),
    )

    assert run_contract(contract_path, dry_run=True, emit_messages=False) == 2
    events = load_scheduler_events(contract_path.parent)
    assert events[-1]["reason"] == "prepared_command_provenance"


def test_scheduler_secure_default_rejects_repo_external_cwd(tmp_path, monkeypatch):
    contract_path = _structured_contract(
        tmp_path,
        execute_argv=[sys.executable, "-m", "sus_bench", "run"],
        execute_cwd=tmp_path,
        score_steps=[{
            "cwd": str(REPO_ROOT / "sus-bench"),
            "argv": [sys.executable, "-m", "sus_bench", "score"],
        }],
    )
    _bind_test_contract_commands(contract_path)
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)

    assert run_contract(contract_path, dry_run=True, emit_messages=False) == 2
    assert "trusted sus_bench source root" in load_scheduler_status(
        contract_path.parent
    )["reason"]


def test_scheduler_requires_explicit_opt_in_for_legacy_shell_contracts(tmp_path, monkeypatch):
    module_dir = tmp_path / "legacy"
    marker = tmp_path / "legacy-ran"
    status_path = module_dir / "RUN_STATUS.json"
    command = _python_command(
        "import json, pathlib; "
        f"pathlib.Path({str(marker)!r}).write_text('ran'); "
        f"pathlib.Path({str(status_path)!r}).write_text(json.dumps({{"
        "'module':'legacy','stage':'run','status':'completed','validity':'score_ready'"
        "}))"
    )
    contract_path = write_run_contract(
        module_dir,
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "run_id": "legacy-shell-test",
            "lifecycle_state": "prepared",
            "execute_command": command,
            "expected_models": [],
            "modules": [
                {
                    "module": "legacy",
                    "stage": "run",
                    "output_dir": ".",
                    "expected_units": [{"unit_id": "legacy:0"}],
                }
            ],
        },
    )
    contract_path = _bind_test_contract_pricing(contract_path)
    monkeypatch.delenv("BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS", raising=False)
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)

    assert run_contract(contract_path, poll_seconds=0.01, emit_messages=False) == 2
    assert marker.exists() is False

    monkeypatch.setenv("BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS", "1")
    assert run_contract(contract_path, poll_seconds=0.01, emit_messages=False) == 2
    assert marker.exists() is False

    monkeypatch.setenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", "1")
    assert run_contract(contract_path, poll_seconds=0.01, emit_messages=False) == 0
    assert marker.read_text() == "ran"


def test_scheduler_blocks_exhausted_openrouter_key_before_child_command(tmp_path, monkeypatch):
    marker = tmp_path / "should-not-run"
    command = f"{shlex.quote(sys.executable)} -m sus_bench run; touch {shlex.quote(str(marker))}"
    contract_path = _contract(tmp_path, command=command)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-" + "or-v1-fake")
    monkeypatch.setattr(
        "suite_tools.scheduler.fetch_key_info",
        lambda timeout=10: {
            "data": {
                "limit": 500,
                "usage": 500.1,
                "limit_remaining": 0,
            }
        },
    )

    exit_code = run_contract(contract_path, poll_seconds=0.1)

    status = load_scheduler_status(contract_path.parent)
    events = load_scheduler_events(contract_path.parent)
    event_names = [event["event"] for event in events]
    assert exit_code == 2
    assert marker.exists() is False
    assert status["state"] == "attention"
    assert "OpenRouter key limit is exhausted" in status["reason"]
    assert "scheduler_attention" in event_names
    assert "generation_started" not in event_names


def test_scheduler_dry_run_writes_queued_status_without_executing(tmp_path):
    marker = tmp_path / "should-not-exist"
    contract_path = _contract(tmp_path, command=f"touch {shlex.quote(str(marker))}")

    exit_code = run_contract(contract_path, dry_run=True, max_active_calls=3)

    status = load_scheduler_status(contract_path.parent)
    events = load_scheduler_events(contract_path.parent)
    assert exit_code == 0
    assert marker.exists() is False
    assert status["schema_version"] == SCHEDULER_SCHEMA_VERSION
    assert status["state"] == "dry_run"
    assert status["settings"]["max_active_calls"] == 3
    assert status["contract"]["expected_units"] == 2
    assert events[-1]["event"] == "dry_run_queued"


def test_real_prepared_contract_binds_commands_and_passes_secure_dry_runs(
    tmp_path,
    monkeypatch,
):
    run_group = tmp_path / "prepared"
    contract_path = prepare_sus_run(
        run_id="scheduler-secure-command-smoke",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)

    contract = load_run_contract(contract_path)
    binding = contract["identity"]["execution"]["prepared_commands"]
    assert binding["schema_version"] == "benchmark-prepared-commands-v1"
    assert binding["execute_steps"] == contract["execute_steps"]
    assert binding["score_steps"] == contract["score_steps"]
    assert run_contract(contract_path, dry_run=True, emit_messages=False) == 0
    run_event = load_scheduler_events(contract_path.parent)[-1]
    assert run_event["preflight_receipt_policy"] == "not_enforced_dry_run"
    assert score_contract(contract_path, dry_run=True, emit_messages=False) == 0
    score_event = load_scheduler_events(contract_path.parent)[-1]
    assert score_event["preflight_receipt_policy"] == "not_enforced_dry_run"


def test_scheduler_authenticates_external_sealed_pack_before_dry_run(tmp_path, monkeypatch):
    sealed = seal_files(
        {
            "flip.csv": b"id,flipped_story\nsynthetic-pair,synthetic reversal\n",
            "flip.labels.json": b'{"labels":{"synthetic-pair":"YTA"}}\n',
            "og.csv": b"id,original_post\nsynthetic-pair,synthetic original\n",
            "selection.yaml": b"items:\n  - index: 0\n    pair_id: synthetic-pair\n",
        },
        pack_id="synthetic-scheduler-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )
    envelope = dict(sealed.envelope, ciphertext_file="synthetic.sealed")
    envelope_path = tmp_path / "external-pack" / "synthetic.envelope.json"
    envelope_path.parent.mkdir()
    envelope_path.write_text(json.dumps(envelope))
    ciphertext_path = envelope_path.parent / "synthetic.sealed"
    ciphertext_path.write_bytes(sealed.ciphertext)

    contract_path = prepare_aita_run(
        run_id="scheduler-sealed-pack",
        output_root=tmp_path / "prepared-sealed-pack",
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="99",
        dataset_mode="nta-paired",
        sealed_pack=str(envelope_path),
        sealed_pack_key_part_b=sealed.key_part_b,
    )
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)

    assert run_contract(contract_path, dry_run=True, emit_messages=False) == 0

    ciphertext_path.write_bytes(sealed.ciphertext[:-1] + b"x")
    assert run_contract(contract_path, dry_run=True, emit_messages=False) == 2
    event = load_scheduler_events(contract_path.parent)[-1]
    assert event["reason"] == "prepared_command_provenance"
    assert any("sealed pack ciphertext" in issue for issue in event["provenance_issues"])


@pytest.mark.parametrize("prefix", ["execute", "score"])
def test_scheduler_rejects_top_level_step_tamper_for_generation_and_score(
    tmp_path,
    monkeypatch,
    prefix,
):
    run_group = tmp_path / f"prepared-{prefix}"
    contract_path = prepare_sus_run(
        run_id=f"scheduler-{prefix}-command-drift",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    payload = json.loads(contract_path.read_text())
    payload[f"{prefix}_steps"][0]["argv"][2] = "attacker_module"
    contract_path.write_text(json.dumps(payload))
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)
    monkeypatch.setattr(
        scheduler_module,
        "_run_command_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not spawn after command drift")
        ),
    )

    exit_code = (
        run_contract(contract_path, dry_run=True, emit_messages=False)
        if prefix == "execute"
        else score_contract(contract_path, dry_run=True, emit_messages=False)
    )

    assert exit_code == 2
    events = load_scheduler_events(contract_path.parent)
    assert events[-1]["reason"] == "prepared_command_provenance"
    assert any(
        f"top-level {prefix}_steps differ" in issue
        for issue in events[-1]["provenance_issues"]
    )


def test_scheduler_rejects_authenticated_output_path_override_without_spawn(
    tmp_path,
    monkeypatch,
):
    run_group = tmp_path / "prepared-output-path"
    contract_path = prepare_sus_run(
        run_id="scheduler-output-path-drift",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    contract = load_run_contract(contract_path)
    tampered_argv = list(contract["execute_argv"])
    tampered_argv[tampered_argv.index("--output") + 1] = "/tmp/victim"
    tampered_step = {"cwd": contract["execute_cwd"], "argv": tampered_argv}
    contract["execute_argv"] = tampered_argv
    contract["execute_steps"] = [tampered_step]
    contract["identity"]["execution"]["prepared_commands"]["execute_steps"] = [
        tampered_step
    ]
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)
    monkeypatch.setattr(
        scheduler_module,
        "_run_command_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not spawn after output-path drift")
        ),
    )

    assert run_contract(contract_path, dry_run=True, emit_messages=False) == 2
    events = load_scheduler_events(contract_path.parent)
    assert events[-1]["reason"] == "prepared_command_provenance"
    assert any("--output" in issue for issue in events[-1]["provenance_issues"])


@pytest.mark.parametrize("subcommand", ["rescore", "report"])
def test_scheduler_rejects_authenticated_unprepared_sus_score_subcommands(
    tmp_path,
    monkeypatch,
    subcommand,
):
    run_group = tmp_path / f"prepared-{subcommand}"
    contract_path = prepare_sus_run(
        run_id=f"scheduler-{subcommand}-drift",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    contract = load_run_contract(contract_path)
    tampered_argv = list(contract["score_argv"])
    tampered_argv[3] = subcommand
    tampered_step = {"cwd": contract["score_cwd"], "argv": tampered_argv}
    contract["score_argv"] = tampered_argv
    contract["score_steps"] = [tampered_step]
    contract["identity"]["execution"]["prepared_commands"]["score_steps"] = [
        tampered_step
    ]
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)
    monkeypatch.setattr(
        scheduler_module,
        "_run_command_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not spawn an unprepared SUS subcommand")
        ),
    )

    assert score_contract(contract_path, dry_run=True, emit_messages=False) == 2
    events = load_scheduler_events(contract_path.parent)
    assert events[-1]["reason"] == "prepared_command_provenance"
    assert any(
        f"subcommand {subcommand!r} is not allowed" in issue
        for issue in events[-1]["provenance_issues"]
    )


def test_scheduler_rejects_mutated_prepared_config_before_dry_run_or_spawn(
    tmp_path,
    monkeypatch,
):
    run_group = tmp_path / "prepared"
    contract_path = prepare_sus_run(
        run_id="scheduler-config-drift",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    config_path = run_group / "_configs" / "calibration" / "sus-models.yaml"
    config_path.write_text(config_path.read_text() + "\n# drift\n")
    monkeypatch.setattr(
        scheduler_module,
        "_run_command_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not spawn")
        ),
    )

    exit_code = run_contract(contract_path, dry_run=True, emit_messages=False)

    assert exit_code == 2
    status = load_scheduler_status(contract_path.parent)
    events = load_scheduler_events(contract_path.parent)
    assert status["state"] == "attention"
    assert events[-1]["reason"] == "prepared_config_provenance"


@pytest.mark.parametrize("prefix", ["execute", "score"])
def test_scheduler_rejects_mutated_pricing_before_dry_run_or_spawn(
    tmp_path,
    monkeypatch,
    prefix,
):
    run_group = tmp_path / f"prepared-pricing-{prefix}"
    contract_path = prepare_sus_run(
        run_id=f"scheduler-pricing-drift-{prefix}",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    contract_path = _attach_cost_estimate(
        contract_path,
        {
            "schema_version": "benchmark-pricing-snapshot-v1",
            "units": "per_token",
            "models": {
                "google/gemini-3-flash-preview": {
                    "prompt": "0.0000005",
                    "completion": "0.000003",
                },
                "google/gemini-3.1-pro-preview": {
                    "prompt": "0.000002",
                    "completion": "0.000012",
                },
            },
        },
    )
    snapshot_path = contract_path.parent / "PRICING_SNAPSHOT.json"
    snapshot_path.write_text(snapshot_path.read_text() + "\n")
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)
    monkeypatch.setattr(
        scheduler_module,
        "_run_command_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not spawn after pricing drift")
        ),
    )

    exit_code = (
        run_contract(contract_path, dry_run=True, emit_messages=False)
        if prefix == "execute"
        else score_contract(contract_path, dry_run=True, emit_messages=False)
    )

    assert exit_code == 2
    status = load_scheduler_status(contract_path.parent)
    events = load_scheduler_events(contract_path.parent)
    assert status["state"] == "attention"
    assert events[-1]["reason"] == "prepared_pricing_provenance"


def test_scheduler_rejects_prepared_contract_with_deleted_config_artifact(tmp_path):
    run_group = tmp_path / "prepared"
    contract_path = prepare_sus_run(
        run_id="scheduler-missing-config-artifact",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    contract = json.loads(contract_path.read_text())
    contract["modules"][0]["expected_artifacts"] = [
        artifact
        for artifact in contract["modules"][0]["expected_artifacts"]
        if artifact.get("kind") != "rendered_models"
    ]
    contract_path.write_text(json.dumps(contract))

    assert run_contract(contract_path, dry_run=True, emit_messages=False) == 2
    assert load_scheduler_status(contract_path.parent)["state"] == "attention"


@pytest.mark.parametrize("entrypoint", ["run", "score"])
@pytest.mark.parametrize("receipt_state", ["missing", "stale", "failed", "mismatched"])
def test_scheduler_rejects_invalid_preflight_admission_without_spawn(
    tmp_path,
    monkeypatch,
    entrypoint,
    receipt_state,
):
    run_group = tmp_path / f"preflight-{entrypoint}-{receipt_state}"
    contract_path = prepare_sus_run(
        run_id=f"preflight-{entrypoint}-{receipt_state}",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    contract_path = _attach_test_cost_estimate(contract_path)
    if receipt_state != "missing":
        _attach_passing_preflight_receipt(contract_path)
        receipt_path = contract_path.parent / "PREFLIGHT_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt_state == "stale":
            receipt["generated_at"] = (
                datetime.now(timezone.utc)
                - timedelta(seconds=scheduler_module.PREFLIGHT_RECEIPT_TTL_SECONDS + 1)
            ).isoformat()
        elif receipt_state == "failed":
            receipt["results"][0]["status"] = "FAIL"
        else:
            receipt["target_set_hash"] = "f" * 64
        receipt["receipt_fingerprint"] = preflight_receipt_fingerprint(receipt)
        receipt_path.write_text(json.dumps(receipt))
    if entrypoint == "score":
        _write_json(contract_path.parent / "RUN_STATUS.json", {
            "module": "sus",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
        })
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)
    monkeypatch.setattr(
        scheduler_module,
        "_run_command_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not spawn without current preflight evidence")
        ),
    )

    exit_code = (
        run_contract(contract_path, emit_messages=False)
        if entrypoint == "run"
        else score_contract(contract_path, emit_messages=False)
    )

    assert exit_code == 2
    assert load_scheduler_status(contract_path.parent)["state"] == "attention"
    assert load_scheduler_events(contract_path.parent)[-1]["reason"] == (
        "preflight_receipt_admission"
    )


def test_scheduler_run_marks_score_ready_after_successful_command(tmp_path):
    module_dir = tmp_path / "sus"
    code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'sus','stage':'run','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "}))"
    )
    contract_path = _contract(tmp_path, command=_python_command(code))

    exit_code = run_contract(contract_path, poll_seconds=0.1)

    status = load_scheduler_status(contract_path.parent)
    event_names = [event["event"] for event in load_scheduler_events(contract_path.parent)]
    assert exit_code == 0
    assert status["state"] == "score_ready"
    assert status["runner"]["status"] == "completed"
    assert status["runner"]["validity"] == "score_ready"
    assert "preflight_receipt_compatibility_bypass" in event_names
    assert "generation_started" in event_names
    assert "generation_completed" in event_names


def test_scheduler_rejects_duplicate_active_contract(tmp_path):
    module_dir = tmp_path / "sus"
    code = (
        "import json, pathlib, datetime, time; "
        f"p=pathlib.Path({str(module_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'sus','stage':'run','status':'running','validity':'not_score_ready',"
        "'started_at':ts,'updated_at':ts"
        "})); "
        "time.sleep(1.0); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'sus','stage':'run','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "}))"
    )
    contract_path = _contract(tmp_path, command=_python_command(code))
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault("exit_code", run_contract(contract_path, poll_seconds=0.05))
    )
    thread.start()

    lock_path = contract_path.parent / SCHEDULER_LOCK_FILENAME
    deadline = time.time() + 3
    while not lock_path.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert lock_path.exists()

    duplicate_exit_code = run_contract(contract_path, poll_seconds=0.05)

    thread.join(timeout=3)
    assert thread.is_alive() is False
    assert result["exit_code"] == 0
    assert duplicate_exit_code == DUPLICATE_SCHEDULER_EXIT_CODE
    assert lock_path.exists() is False
    event_names = [event["event"] for event in load_scheduler_events(contract_path.parent)]
    assert event_names.count("generation_started") == 1


def test_scheduler_allows_distinct_contracts_to_run_concurrently(tmp_path):
    def sleep_command(module_dir):
        return _python_command(
            "import json, pathlib, datetime, time; "
            f"p=pathlib.Path({str(module_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
            "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
            "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
            "'module':p.name,'stage':'run','status':'running','validity':'not_score_ready',"
            "'started_at':ts,'updated_at':ts"
            "})); "
            "time.sleep(1.0); "
            "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
            "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
            "'module':p.name,'stage':'run','status':'completed','validity':'score_ready',"
            "'started_at':ts,'updated_at':ts,'completed_at':ts"
            "}))"
        )

    first_contract = _contract(tmp_path, command=sleep_command(tmp_path / "sus_a"), module="sus_a")
    second_contract = _contract(tmp_path, command=sleep_command(tmp_path / "sus_b"), module="sus_b")
    result = {}

    first = threading.Thread(
        target=lambda: result.setdefault("first", run_contract(first_contract, poll_seconds=0.05))
    )
    second = threading.Thread(
        target=lambda: result.setdefault("second", run_contract(second_contract, poll_seconds=0.05))
    )
    first.start()
    second.start()

    first_lock = first_contract.parent / SCHEDULER_LOCK_FILENAME
    second_lock = second_contract.parent / SCHEDULER_LOCK_FILENAME
    deadline = time.time() + 3
    while (
        (not first_lock.exists() or not second_lock.exists())
        and time.time() < deadline
    ):
        time.sleep(0.01)
    assert first_lock.exists()
    assert second_lock.exists()

    first.join(timeout=3)
    second.join(timeout=3)
    assert first.is_alive() is False
    assert second.is_alive() is False
    assert result == {"first": 0, "second": 0}
    assert first_lock.exists() is False
    assert second_lock.exists() is False


def test_scheduler_run_many_overlaps_scoring_with_other_generation(tmp_path):
    fast_dir = tmp_path / "fast"
    slow_dir = tmp_path / "slow"
    fast_score_started = tmp_path / "fast-score-started.txt"
    slow_generation_done = tmp_path / "slow-generation-done.txt"

    fast_generation = _python_command(
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(fast_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'fast','stage':'generation','status':'completed','validity':'not_score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts}))"
    )
    fast_score = _python_command(
        "import json, pathlib, datetime, time; "
        f"p=pathlib.Path({str(fast_dir)!r}); "
        f"pathlib.Path({str(fast_score_started)!r}).write_text(str(time.monotonic())); "
        "time.sleep(0.1); ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'fast','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts}))"
    )
    slow_generation = _python_command(
        "import json, pathlib, datetime, time; "
        f"p=pathlib.Path({str(slow_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'slow','stage':'generation','status':'running','validity':'not_score_ready',"
        "'started_at':ts,'updated_at':ts})); "
        "time.sleep(0.3); "
        f"pathlib.Path({str(slow_generation_done)!r}).write_text(str(time.monotonic())); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'slow','stage':'generation','status':'completed','validity':'not_score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts}))"
    )
    slow_score = _python_command(
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(slow_dir)!r}); ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'slow','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts}))"
    )
    fast_contract = _contract(
        tmp_path, command=fast_generation, module="fast", score_command=fast_score
    )
    slow_contract = _contract(
        tmp_path, command=slow_generation, module="slow", score_command=slow_score
    )

    results = run_contracts(
        [fast_contract, slow_contract],
        poll_seconds=0.02,
        max_contract_workers=2,
        auto_score_on_clean_generation=True,
        stream_command_output=False,
    )

    assert set(results.values()) == {0}
    assert float(fast_score_started.read_text()) < float(slow_generation_done.read_text())


def test_run_contracts_contains_one_worker_exception_and_waits_for_siblings(tmp_path, monkeypatch):
    failed = tmp_path / "failed" / "RUN_CONTRACT.json"
    healthy = tmp_path / "healthy" / "RUN_CONTRACT.json"
    healthy_finished = threading.Event()

    def fake_run_contract(path, **kwargs):
        if path == failed:
            raise RuntimeError("broken worker")
        time.sleep(0.03)
        healthy_finished.set()
        return 0

    monkeypatch.setattr("suite_tools.scheduler.run_contract", fake_run_contract)

    results = run_contracts(
        [failed, healthy],
        max_contract_workers=2,
        emit_messages=False,
    )

    assert results == {failed: 2, healthy: 0}
    assert healthy_finished.is_set()


def test_scheduler_run_many_cli_accepts_multiple_contracts(tmp_path, capsys):
    first = _contract(tmp_path, command="echo first", module="first")
    second = _contract(tmp_path, command="echo second", module="second")

    exit_code = main([
        "run-many",
        "--contract", str(first),
        "--contract", str(second),
        "--dry-run",
        "--output-json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["exit_code"] == 0
    assert len(payload["contracts"]) == 2
    assert {item["state"] for item in payload["contracts"]} == {"dry_run"}


def test_scheduler_run_marks_needs_scoring_for_clean_generation_gate(tmp_path):
    module_dir = tmp_path / "aita"
    code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'aita','stage':'generation','status':'completed','validity':'not_score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "}))"
    )
    contract_path = _contract(tmp_path, command=_python_command(code), module="aita")

    exit_code = run_contract(contract_path, poll_seconds=0.1)

    status = load_scheduler_status(contract_path.parent)
    assert exit_code == 0
    assert status["state"] == "needs_scoring"
    assert "scoring is gated" in status["reason"]


def test_scheduler_score_only_scores_existing_needs_scoring_run(tmp_path):
    module_dir = tmp_path / "aita"
    ts = "2026-05-26T00:00:00+00:00"
    _write_json(
        module_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
            "started_at": ts,
            "updated_at": ts,
            "completed_at": ts,
        },
    )
    score_code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'aita','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "}))"
    )
    contract_path = _contract(
        tmp_path,
        command="echo generation-should-not-run",
        module="aita",
        score_command=_python_command(score_code),
    )

    exit_code = score_contract(contract_path, poll_seconds=0.1)

    status = load_scheduler_status(contract_path.parent)
    event_names = [event["event"] for event in load_scheduler_events(contract_path.parent)]
    assert exit_code == 0
    assert status["state"] == "score_ready"
    assert status["settings"]["score_only"] is True
    assert "scoring_started" in event_names
    assert "generation_started" not in event_names


def test_scheduler_sus_prepared_contract_survives_generation_then_scores(tmp_path, monkeypatch):
    run_group = tmp_path / "prepared-sus-chain"
    contract_path = prepare_sus_run(
        run_id="prepared-sus-chain",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    contract_path = _attach_test_cost_estimate(contract_path)
    module_dir = contract_path.parent
    models_path = run_group / "_configs" / "calibration" / "sus-models.yaml"
    score_marker = module_dir / "score-ran.txt"
    generation_code = (
        "from types import SimpleNamespace; "
        "from sus_bench import api, cli, runner; "
        "api.CostTracker.check_credit_now=lambda self: None; "
        "runner.run_benchmark=lambda *args, **kwargs: []; "
        f"cli._cmd_run(SimpleNamespace(models={str(models_path)!r}, output={str(module_dir)!r}, "
        "model=None, scenarios='bridge_heights', runs=1, analyzer_model=None, delay=0, "
        "temperature=None, reasoning=None, no_parallel=True, html=False, "
        "escalation_mode='adaptive', score_inline=False))"
    )
    score_code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'sus','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts})); "
        f"pathlib.Path({str(score_marker)!r}).write_text('scored')"
    )
    contract = load_run_contract(contract_path)
    contract["execute_cwd"] = str(REPO_ROOT)
    contract["execute_argv"] = [sys.executable, "-c", generation_code]
    contract["execute_steps"] = [{"cwd": str(REPO_ROOT), "argv": contract["execute_argv"]}]
    contract["score_cwd"] = str(REPO_ROOT)
    contract["score_argv"] = [sys.executable, "-c", score_code]
    contract["score_steps"] = [{"cwd": str(REPO_ROOT), "argv": contract["score_argv"]}]
    write_run_contract(module_dir, contract)
    _attach_passing_preflight_receipt(contract_path)
    prepared_bytes = contract_path.read_bytes()

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scheduler_module, "openrouter_key_limit_attention", lambda: None)

    assert run_contract(
        contract_path,
        poll_seconds=0.02,
        stream_command_output=False,
    ) == 0
    assert score_contract(
        contract_path,
        poll_seconds=0.02,
        stream_command_output=False,
    ) == 0

    persisted = load_run_contract(contract_path)
    assert contract_path.read_bytes() == prepared_bytes
    assert persisted["execute_argv"] == contract["execute_argv"]
    assert persisted["score_argv"] == contract["score_argv"]
    assert persisted["call_plan"] == contract["call_plan"]
    assert persisted["cost_estimate"] == contract["cost_estimate"]
    assert score_marker.read_text() == "scored"
    assert load_scheduler_status(module_dir)["state"] == "score_ready"


def test_scheduler_aita_prepared_contract_survives_generation_then_scores(tmp_path, monkeypatch):
    run_group = tmp_path / "prepared-aita-chain"
    contract_path = prepare_aita_run(
        run_id="prepared-aita-chain",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="yta-synthflip",
        allow_sample_fallback=True,
    )
    contract_path = _attach_test_cost_estimate(contract_path)
    module_dir = contract_path.parent
    score_marker = module_dir / "score-ran.txt"
    prepared_contract = load_run_contract(contract_path)
    model = prepared_contract["expected_models"][0]
    generation_code = (
        "import json, pathlib, datetime; "
        "from aita_bench import runner; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "runner.write_generation_contract("
        f"p, model_keys=[{model['key']!r}], models={{{model['key']!r}: {model!r}}}, "
        "item_indices=[0], flips={0:'flipped'}, dataset_mode='yta-synthflip', "
        "items={0:{'original':'original','top_comment':'','ground_truth':'YTA'}}); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'aita','stage':'generation','status':'completed','validity':'not_score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts}))"
    )
    score_code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'aita','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts})); "
        f"pathlib.Path({str(score_marker)!r}).write_text('scored')"
    )
    prepared_contract["execute_cwd"] = str(REPO_ROOT)
    prepared_contract["execute_argv"] = [sys.executable, "-c", generation_code]
    prepared_contract["execute_steps"] = [
        {"cwd": str(REPO_ROOT), "argv": prepared_contract["execute_argv"]}
    ]
    prepared_contract["score_cwd"] = str(REPO_ROOT)
    prepared_contract["score_argv"] = [sys.executable, "-c", score_code]
    prepared_contract["score_steps"] = [
        {"cwd": str(REPO_ROOT), "argv": prepared_contract["score_argv"]}
    ]
    write_run_contract(module_dir, prepared_contract)
    _attach_passing_preflight_receipt(contract_path)
    prepared_bytes = contract_path.read_bytes()

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scheduler_module, "openrouter_key_limit_attention", lambda: None)

    assert run_contract(contract_path, poll_seconds=0.02, stream_command_output=False) == 0
    assert score_contract(contract_path, poll_seconds=0.02, stream_command_output=False) == 0

    persisted = load_run_contract(contract_path)
    assert contract_path.read_bytes() == prepared_bytes
    assert persisted["execute_argv"] == prepared_contract["execute_argv"]
    assert persisted["score_argv"] == prepared_contract["score_argv"]
    assert persisted["call_plan"] == prepared_contract["call_plan"]
    assert persisted["cost_estimate"] == prepared_contract["cost_estimate"]
    assert score_marker.read_text() == "scored"
    assert load_scheduler_status(module_dir)["state"] == "score_ready"


def test_scheduler_epis_prepared_contract_survives_generation_then_scores(tmp_path, monkeypatch):
    run_group = tmp_path / "prepared-epis-chain"
    contract_path = prepare_epis_run(
        run_id="prepared-epis-chain",
        output_root=run_group,
        suite_config_path=REPO_ROOT / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items=1,
        types="delusion",
        selection=str(
            (REPO_ROOT / "epistemic-sycophancy-bench/data/calibration-selection.yaml").resolve()
        ),
    )
    contract_path = _attach_test_cost_estimate(contract_path)
    module_dir = contract_path.parent
    score_marker = module_dir / "score-ran.txt"
    prepared_contract = load_run_contract(contract_path)
    model = prepared_contract["expected_models"][0]
    generation_code = (
        "import json, pathlib, datetime; "
        "from epis_bench import runner; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "runner.write_generation_contract("
        f"p, model_keys=[{model['key']!r}], models={{{model['key']!r}: {model!r}}}, "
        "items_by_type={'delusion':[{'question':'question'}]}, selection_path='selection.yaml'); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'epistemic','stage':'generation','status':'completed','validity':'not_score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts}))"
    )
    score_code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'epistemic','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts})); "
        f"pathlib.Path({str(score_marker)!r}).write_text('scored')"
    )
    prepared_contract["execute_cwd"] = str(REPO_ROOT)
    prepared_contract["execute_argv"] = [sys.executable, "-c", generation_code]
    prepared_contract["execute_steps"] = [
        {"cwd": str(REPO_ROOT), "argv": prepared_contract["execute_argv"]}
    ]
    prepared_contract["score_cwd"] = str(REPO_ROOT)
    prepared_contract["score_argv"] = [sys.executable, "-c", score_code]
    prepared_contract["score_steps"] = [
        {"cwd": str(REPO_ROOT), "argv": prepared_contract["score_argv"]}
    ]
    write_run_contract(module_dir, prepared_contract)
    _attach_passing_preflight_receipt(contract_path)
    prepared_bytes = contract_path.read_bytes()

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scheduler_module, "openrouter_key_limit_attention", lambda: None)

    assert run_contract(contract_path, poll_seconds=0.02, stream_command_output=False) == 0
    assert score_contract(contract_path, poll_seconds=0.02, stream_command_output=False) == 0

    persisted = load_run_contract(contract_path)
    assert contract_path.read_bytes() == prepared_bytes
    assert persisted["execute_argv"] == prepared_contract["execute_argv"]
    assert persisted["score_argv"] == prepared_contract["score_argv"]
    assert persisted["call_plan"] == prepared_contract["call_plan"]
    assert persisted["cost_estimate"] == prepared_contract["cost_estimate"]
    assert score_marker.read_text() == "scored"
    assert load_scheduler_status(module_dir)["state"] == "score_ready"


def test_scheduler_score_only_refuses_non_needs_scoring_state(tmp_path):
    module_dir = tmp_path / "aita"
    ts = "2026-05-26T00:00:00+00:00"
    _write_json(
        module_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": ts,
            "updated_at": ts,
        },
    )
    contract_path = _contract(
        tmp_path,
        command="echo generation",
        module="aita",
        score_command="touch should-not-score",
    )

    exit_code = score_contract(contract_path, poll_seconds=0.1)

    status = load_scheduler_status(contract_path.parent)
    event_names = [event["event"] for event in load_scheduler_events(contract_path.parent)]
    assert exit_code == 2
    assert status["state"] == "attention"
    assert "requires needs_scoring" in status["reason"]
    assert "scoring_started" not in event_names


def test_scheduler_score_only_force_retries_failed_scoring_run(tmp_path):
    module_dir = tmp_path / "epis"
    args_path = tmp_path / "score-args.json"
    ts = "2026-05-26T00:00:00+00:00"
    _write_json(
        module_dir / "RUN_STATUS.json",
        {
            "module": "epis",
            "stage": "scoring",
            "status": "failed_scoring",
            "validity": "not_score_ready",
            "missing_scores": ["gemini-flash_item0_mirror_side_a.integrity"],
            "started_at": ts,
            "updated_at": ts,
            "failed_at": ts,
        },
    )
    score_code = (
        "import json, pathlib, datetime, sys, time; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        f"args_path=pathlib.Path({str(args_path)!r}); "
        "args_path.write_text(json.dumps(sys.argv)); "
        "time.sleep(0.3); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'epis','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "}))"
    )
    contract_path = _contract(
        tmp_path,
        command="echo generation-should-not-run",
        module="epis",
        score_command=_python_command(score_code),
    )

    exit_code = score_contract(
        contract_path,
        poll_seconds=0.1,
        stop_on_attention=True,
        force=True,
    )

    status = load_scheduler_status(contract_path.parent)
    control = load_run_control(contract_path.parent)
    event_names = [event["event"] for event in load_scheduler_events(contract_path.parent)]
    assert exit_code == 0
    assert status["state"] == "score_ready"
    assert status["settings"]["force"] is True
    assert "--force" not in json.loads(args_path.read_text())
    assert control == {}
    assert "scoring_started" in event_names
    assert "force_score_retry_allowed" in event_names
    assert "control_requested" not in event_names
    assert "generation_started" not in event_names


def test_scheduler_score_only_force_retries_score_input_identity_failure(tmp_path):
    module_dir = tmp_path / "sus"
    score_marker = tmp_path / "scored"
    ts = "2026-05-26T00:00:00+00:00"
    _write_json(
        module_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "scoring",
            "status": "failed_invalid",
            "validity": "not_score_ready",
            "failure_stage": "artifact_identity",
            "started_at": ts,
            "updated_at": ts,
            "failed_at": ts,
        },
    )
    score_code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'sus','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "})); "
        f"pathlib.Path({str(score_marker)!r}).write_text('scored')"
    )
    contract_path = _contract(
        tmp_path,
        command="echo generation-should-not-run",
        module="sus",
        score_command=_python_command(score_code),
    )

    exit_code = score_contract(
        contract_path,
        poll_seconds=0.02,
        force=True,
    )

    status = load_scheduler_status(module_dir)
    events = load_scheduler_events(module_dir)
    assert exit_code == 0
    assert score_marker.read_text() == "scored"
    assert status["state"] == "score_ready"
    retry_event = next(event for event in events if event["event"] == "force_score_retry_allowed")
    assert retry_event["previous_status"] == "failed_invalid"
    assert retry_event["previous_failure_stage"] == "artifact_identity"
    assert "generation_started" not in [event["event"] for event in events]


def test_scheduler_score_only_force_refuses_failed_generation(tmp_path):
    module_dir = tmp_path / "aita"
    marker = tmp_path / "should-not-score"
    ts = "2026-05-26T00:00:00+00:00"
    _write_json(
        module_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "failure_reason": "provider returned 429 before final turn",
            "started_at": ts,
            "updated_at": ts,
            "failed_at": ts,
        },
    )
    contract_path = _contract(
        tmp_path,
        command="echo generation",
        module="aita",
        score_command=f"touch {shlex.quote(str(marker))}",
    )

    exit_code = score_contract(contract_path, poll_seconds=0.1, force=True)

    status = load_scheduler_status(contract_path.parent)
    event_names = [event["event"] for event in load_scheduler_events(contract_path.parent)]
    assert exit_code == 2
    assert marker.exists() is False
    assert status["state"] == "attention"
    assert "failed_scoring" in status["reason"]
    assert "scoring_started" not in event_names


def test_scheduler_score_command_output_json_is_machine_readable(tmp_path, capsys):
    contract_path = _contract(
        tmp_path,
        command="echo generation",
        score_command="echo scoring",
    )

    exit_code = main([
        "score",
        "--contract",
        str(contract_path),
        "--dry-run",
        "--output-json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["exit_code"] == 0
    assert payload["status"]["state"] == "dry_run"
    assert payload["status"]["settings"]["score_only"] is True


def test_scheduler_auto_score_handles_existing_needs_scoring_without_rerunning_generation(tmp_path):
    module_dir = tmp_path / "aita"
    marker = tmp_path / "generation-reran"
    ts = "2026-05-26T00:00:00+00:00"
    _write_json(
        module_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
            "started_at": ts,
            "updated_at": ts,
            "completed_at": ts,
        },
    )
    score_code = (
        "import json, pathlib, datetime; "
        f"p=pathlib.Path({str(module_dir)!r}); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'aita','stage':'scoring','status':'completed','validity':'score_ready',"
        "'started_at':ts,'updated_at':ts,'completed_at':ts"
        "}))"
    )
    contract_path = _contract(
        tmp_path,
        command=f"touch {shlex.quote(str(marker))}",
        module="aita",
        score_command=_python_command(score_code),
    )

    exit_code = run_contract(
        contract_path,
        poll_seconds=0.1,
        auto_score_on_clean_generation=True,
    )

    status = load_scheduler_status(contract_path.parent)
    event_names = [event["event"] for event in load_scheduler_events(contract_path.parent)]
    assert exit_code == 0
    assert marker.exists() is False
    assert status["state"] == "score_ready"
    assert "scoring_started" in event_names
    assert "generation_started" not in event_names


def test_scheduler_run_marks_attention_after_failed_command(tmp_path):
    module_dir = tmp_path / "aita"
    code = (
        "import json, pathlib, datetime, sys; "
        f"p=pathlib.Path({str(module_dir)!r}); p.mkdir(parents=True, exist_ok=True); "
        "ts=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
        "p.joinpath('RUN_STATUS.json').write_text(json.dumps({"
        "'module':'aita','stage':'generation','status':'failed_incomplete','validity':'not_score_ready',"
        "'failure_reason':'provider returned 502 before first model turn',"
        "'started_at':ts,'updated_at':ts,'failed_at':ts"
        "})); sys.exit(2)"
    )
    contract_path = _contract(tmp_path, command=_python_command(code), module="aita")

    exit_code = run_contract(contract_path, poll_seconds=0.1, stop_on_attention=True)

    status = load_scheduler_status(contract_path.parent)
    assert exit_code == 2
    assert status["state"] == "attention"
    assert status["runner"]["status"] == "failed_incomplete"


def test_scheduler_stop_command_writes_run_control(tmp_path):
    contract_path = _contract(tmp_path, command="true")

    exit_code = main([
        "stop",
        "--contract",
        str(contract_path),
        "--reason",
        "debug stop",
        "--requested-by",
        "test",
    ])

    control = load_run_control(contract_path.parent)
    assert exit_code == 0
    assert control["action"] == "stop_before_next_paid_call"
    assert control["reason"] == "debug stop"
    assert control["requested_by"] == "test"


def test_scheduler_stop_command_output_json_is_machine_readable(tmp_path, capsys):
    contract_path = _contract(tmp_path, command="true")

    exit_code = main([
        "stop",
        "--contract",
        str(contract_path),
        "--reason",
        "debug stop",
        "--requested-by",
        "test",
        "--output-json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "stop_before_next_paid_call"
    assert payload["control"]["reason"] == "debug stop"
    assert payload["contract_path"].endswith("RUN_CONTRACT.json")


def test_scheduler_run_dry_run_output_json_suppresses_human_line(tmp_path, capsys):
    contract_path = _contract(tmp_path, command="echo should-not-print")

    exit_code = main([
        "run",
        "--contract",
        str(contract_path),
        "--dry-run",
        "--output-json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["exit_code"] == 0
    assert payload["status"]["state"] == "dry_run"
    assert payload["status"]["contract"]["expected_units"] == 2
    assert payload["status"]["settings"]["run_pace"] == "normal"
    assert payload["status"]["settings"]["max_active_calls"] == 4
    assert payload["status"]["settings"]["stagger_start_seconds"] == 1.0


def test_scheduler_run_pace_can_be_overridden_with_explicit_limits(tmp_path, capsys):
    contract_path = _contract(tmp_path, command="echo should-not-print")

    exit_code = main([
        "run",
        "--contract",
        str(contract_path),
        "--dry-run",
        "--run-pace",
        "fast",
        "--max-active-calls",
        "3",
        "--stagger-start-seconds",
        "2.5",
        "--output-json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"]["settings"]["run_pace"] == "fast"
    assert payload["status"]["settings"]["max_active_calls"] == 3
    assert payload["status"]["settings"]["stagger_start_seconds"] == 2.5


def test_scheduler_paces_command_outputs_machine_readable_presets(capsys):
    exit_code = main(["paces", "--output-json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["default"] == "normal"
    assert payload["presets"]["cautious"]["max_active_calls"] == 2
    assert payload["presets"]["normal"]["max_active_calls"] == 4
    assert payload["presets"]["fast"]["max_active_calls"] == 6
    assert payload["presets"]["full-speed"]["max_active_calls"] == 8


def test_progress_snapshot_estimates_eta_from_completed_units():
    snapshot = build_progress_snapshot(
        contract_summary={"expected_units": 10, "complete_units": 0},
        module_status={"status": "running", "validity": "not_score_ready"},
        module_events=[
            {"event": "conversation_completed"},
            {"event": "conversation_completed"},
        ],
        scheduler_started_at="2026-05-26T00:00:00+00:00",
        max_active_calls=4,
        now=__import__("datetime").datetime.fromisoformat("2026-05-26T00:02:00+00:00"),
    )

    assert snapshot["completed_units"] == 2
    assert snapshot["remaining_units"] == 8
    assert snapshot["average_completed_unit_seconds"] == 60
    assert snapshot["eta_seconds"] == 120
    assert snapshot["eta_basis"] == "completed-unit-average"


def test_progress_snapshot_counts_sus_completed_runs_before_scoring():
    snapshot = build_progress_snapshot(
        contract_summary={"expected_units": 3, "complete_units": 0},
        module_status={
            "status": "completed",
            "stage": "generation",
            "validity": "not_score_ready",
        },
        module_events=[
            {"event": "sus_run_completed"},
            {"event": "sus_run_completed"},
            {"event": "sus_run_completed"},
            {
                "event": "stage_completed",
                "timestamp": "2026-05-26T00:03:00+00:00",
                "sequence": 4,
            },
        ],
        scheduler_started_at="2026-05-26T00:00:00+00:00",
        max_active_calls=3,
        now=__import__("datetime").datetime.fromisoformat("2026-05-26T00:03:00+00:00"),
    )

    assert snapshot["completed_units"] == 3
    assert snapshot["remaining_units"] == 0
    assert snapshot["percent"] == 100
    assert snapshot["eta_seconds"] == 0


def test_progress_snapshot_clears_stale_active_units_after_generation_completes():
    snapshot = build_progress_snapshot(
        contract_summary={"expected_units": 2, "complete_units": 0},
        module_status={
            "status": "completed",
            "stage": "generation",
            "validity": "not_score_ready",
        },
        module_events=[
            {"event": "conversation_started"},
            {"event": "paid_call_started"},
            {"event": "conversation_completed"},
            {"event": "conversation_completed"},
        ],
        scheduler_started_at="2026-05-26T00:00:00+00:00",
        max_active_calls=4,
        now=__import__("datetime").datetime.fromisoformat("2026-05-26T00:02:00+00:00"),
    )

    assert snapshot["completed_units"] == 2
    assert snapshot["remaining_units"] == 0
    assert snapshot["active_units"] == 0


def test_progress_snapshot_does_not_double_count_generated_then_scored_units():
    snapshot = build_progress_snapshot(
        contract_summary={"expected_units": 4, "complete_units": 0},
        module_status={"status": "running", "validity": "not_score_ready"},
        module_events=[
            {"event": "conversation_completed"},
            {"event": "conversation_completed"},
            {"event": "score_saved"},
            {"event": "score_saved"},
        ],
        scheduler_started_at="2026-05-26T00:00:00+00:00",
        max_active_calls=4,
        now=__import__("datetime").datetime.fromisoformat("2026-05-26T00:02:00+00:00"),
    )

    # Two units each completed generation and were scored: 2 units, not 4.
    assert snapshot["completed_units"] == 2
    assert snapshot["remaining_units"] == 2
    assert snapshot["percent"] == 50


class TestSchedulerLockTakeover:
    def _live_lock_payload(self, scheduler_id="live-owner"):
        import os

        from suite_tools.scheduler import SCHEDULER_LOCK_SCHEMA_VERSION

        return {
            "schema_version": SCHEDULER_LOCK_SCHEMA_VERSION,
            "scheduler_id": scheduler_id,
            "pid": os.getpid(),
            "created_at": "2026-06-10T00:00:00+00:00",
        }

    def _dead_lock_payload(self, scheduler_id="dead-owner"):
        import subprocess

        from suite_tools.scheduler import SCHEDULER_LOCK_SCHEMA_VERSION

        proc = subprocess.Popen(["sleep", "0"])
        proc.wait()
        return {
            "schema_version": SCHEDULER_LOCK_SCHEMA_VERSION,
            "scheduler_id": scheduler_id,
            "pid": proc.pid,
            "created_at": "2026-06-10T00:00:00+00:00",
        }

    def test_live_lock_blocks_acquire(self, tmp_path):
        import pytest

        from suite_tools.scheduler import SchedulerAlreadyRunning, acquire_scheduler_lock

        lock_path = tmp_path / SCHEDULER_LOCK_FILENAME
        _write_json(lock_path, self._live_lock_payload())

        with pytest.raises(SchedulerAlreadyRunning):
            acquire_scheduler_lock(
                tmp_path,
                scheduler_id="challenger",
                contract_path=tmp_path / "contract.json",
                command="echo hi",
            )

    def test_dead_lock_is_taken_over(self, tmp_path):
        import os

        from suite_tools.scheduler import acquire_scheduler_lock

        lock_path = tmp_path / SCHEDULER_LOCK_FILENAME
        _write_json(lock_path, self._dead_lock_payload())

        acquired = acquire_scheduler_lock(
            tmp_path,
            scheduler_id="challenger",
            contract_path=tmp_path / "contract.json",
            command="echo hi",
        )

        assert acquired == lock_path
        lock = json.loads(lock_path.read_text())
        assert lock["scheduler_id"] == "challenger"
        assert lock["pid"] == os.getpid()

    def test_takeover_never_removes_a_lock_it_did_not_verify(self, tmp_path, monkeypatch):
        """Race: a stale read must not let a challenger delete the winner's fresh lock.

        Simulates: challenger reads a dead lock, but by the time it acts another
        scheduler has already taken over and written a live lock at the same
        path. The challenger must defer to the live owner, not clobber it.
        """
        import pytest

        from suite_tools import scheduler as scheduler_module
        from suite_tools.scheduler import SchedulerAlreadyRunning, acquire_scheduler_lock

        lock_path = tmp_path / SCHEDULER_LOCK_FILENAME
        live_payload = self._live_lock_payload(scheduler_id="fresh-winner")
        _write_json(lock_path, live_payload)

        stale_payload = self._dead_lock_payload(scheduler_id="stale-read")
        real_load_json = scheduler_module._load_json
        calls = {"n": 0}

        def stale_first_read(path):
            calls["n"] += 1
            if calls["n"] == 1:
                return dict(stale_payload)
            return real_load_json(path)

        monkeypatch.setattr(scheduler_module, "_load_json", stale_first_read)

        with pytest.raises(SchedulerAlreadyRunning):
            acquire_scheduler_lock(
                tmp_path,
                scheduler_id="challenger",
                contract_path=tmp_path / "contract.json",
                command="echo hi",
            )

        # The fresh winner's lock must survive untouched.
        lock = json.loads(lock_path.read_text())
        assert lock["scheduler_id"] == "fresh-winner"


def test_completed_units_reduce_by_unit_identity_across_attempts():
    from suite_tools.scheduler import _completed_units_from_events
    events = [{"event": "conversation_completed", "unit_id": f"u{i}", "attempt_number": 1} for i in range(3)]
    events += [{"event": "conversation_reused", "unit_id": f"u{i}", "attempt_number": 2} for i in range(3)]
    events += [{"event": "conversation_completed", "unit_id": f"u{i}", "attempt_number": 2} for i in range(3, 8)]
    assert _completed_units_from_events(events) == 8


def test_active_units_dedupes_retried_conversation_started():
    from suite_tools.scheduler import _active_units_from_events
    # A unit that retries emits conversation_started twice (once per attempt).
    # The active count must be 1, not 2.
    events = [
        {"event": "conversation_started", "unit_id": "u0", "attempt_number": 1},
        {"event": "conversation_started", "unit_id": "u0", "attempt_number": 2},
    ]
    assert _active_units_from_events(events) == 1
