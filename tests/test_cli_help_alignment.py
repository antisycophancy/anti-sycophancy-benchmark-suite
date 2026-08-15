from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def cli_help(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, *args, "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def normalized_help(*args: str) -> str:
    return " ".join(cli_help(*args).split())


def test_prepare_help_explains_model_selectors_and_local_adapter_group():
    help_text = cli_help("-m", "suite_tools.prepare_run")

    assert "group:<name>" in help_text
    assert "local_endpoint_smoke" in help_text


def test_preflight_help_distinguishes_network_probe_from_provider_spend():
    help_text = cli_help("-m", "suite_tools.preflight_conditions")

    assert "network request" in help_text
    assert "reference adapter probes are free" in help_text
    assert "--run-dir" in help_text


def test_run_many_help_explains_shared_lease_and_attention_scope():
    help_text = normalized_help("-m", "suite_tools.scheduler", "run-many")

    assert "shared global paid-call lease" in help_text
    assert "does not stop sibling contracts" in help_text


def test_adapter_smoke_help_points_to_full_guide_and_paid_proxy_gate():
    help_text = cli_help("adapter/smoke.py")

    assert "adapter/README.md" in help_text
    assert "proxy mode" in help_text
    assert "--allow-proxy-call" in help_text
    assert "--api-key-env" in help_text


def test_subset_help_explains_repeated_model_exclusions():
    help_text = normalized_help("-m", "suite_tools.materialize_subset")

    assert "Repeat to exclude multiple model keys" in help_text
