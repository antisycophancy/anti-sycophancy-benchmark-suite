import os
import subprocess
import sys
from pathlib import Path

from suite_tools.suite_registry import suite_root


ROOT = Path(__file__).resolve().parents[1]


def _run_without_pythonpath(cwd: Path, code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_aita_runner_imports_suite_tools_from_nested_cwd():
    result = _run_without_pythonpath(
        suite_root("aita"),
        "import aita_bench.runner",
    )

    assert result.returncode == 0, result.stderr


def test_epis_runner_imports_suite_tools_from_nested_cwd():
    result = _run_without_pythonpath(
        suite_root("epistemic"),
        "import epis_bench.runner",
    )

    assert result.returncode == 0, result.stderr


def test_sus_runner_imports_suite_tools_from_nested_cwd():
    result = _run_without_pythonpath(
        suite_root("sus"),
        "import sus_bench.runner",
    )

    assert result.returncode == 0, result.stderr
