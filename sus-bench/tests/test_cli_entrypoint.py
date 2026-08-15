import subprocess
import sys

from sus_bench import __version__


def test_python_module_entrypoint_prints_version():
    result = subprocess.run(
        [sys.executable, "-m", "sus_bench.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == f"sus-bench {__version__}"

