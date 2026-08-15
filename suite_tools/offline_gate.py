"""Run the benchmark offline test gate with isolated pytest roots.

The benchmark repo contains several nested benchmark packages, each with its
own ``tests`` package and, in some cases, its own ``conftest.py``. Running one
large pytest collection from the repository root imports those separate suites
under colliding module names. This gate keeps the validation command simple
while still running each first-party suite from the root it expects.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from suite_tools.suite_registry import REPO_ROOT, suite_root


@dataclass(frozen=True)
class GateCheck:
    """One isolated offline test command."""

    name: str
    cwd: Path
    args: tuple[str, ...]


def build_checks(*, python_executable: str = sys.executable) -> list[GateCheck]:
    """Return first-party offline checks in dependency-light order."""
    pytest_cmd = (python_executable, "-m", "pytest", "-q", "-rs")
    return [
        GateCheck(
            name="shared",
            cwd=REPO_ROOT,
            args=(*pytest_cmd, "tests", "unified_profile/tests"),
        ),
        GateCheck(
            name="aita",
            cwd=suite_root("aita"),
            args=pytest_cmd,
        ),
        GateCheck(
            name="epis",
            cwd=suite_root("epistemic"),
            args=pytest_cmd,
        ),
        GateCheck(
            name="sus",
            cwd=suite_root("sus"),
            args=pytest_cmd,
        ),
    ]


def _env_without_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def run_check(check: GateCheck) -> subprocess.CompletedProcess[str]:
    """Run a single isolated check and capture combined output."""
    return subprocess.run(
        list(check.args),
        cwd=check.cwd,
        env=_env_without_pythonpath(),
        text=True,
        capture_output=True,
        check=False,
    )


def run_checks(checks: Iterable[GateCheck]) -> int:
    """Run checks sequentially and print a compact pass/fail transcript."""
    failures: list[str] = []
    for check in checks:
        print(f"== {check.name} ==")
        print(f"$ {' '.join(check.args)}")
        print(f"cwd: {check.cwd}")
        result = run_check(check)
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if output:
            print(output.rstrip())
        if result.returncode == 0:
            print(f"PASS {check.name}\n")
        else:
            print(f"FAIL {check.name} ({result.returncode})\n")
            failures.append(check.name)

    if failures:
        print(f"Offline gate failed: {', '.join(failures)}")
        return 1
    print("Offline gate passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated offline benchmark test suites.")
    parser.add_argument(
        "--suite",
        choices=("all", "shared", "aita", "epis", "sus"),
        default="all",
        help="Limit the gate to one suite.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = build_checks()
    if args.suite != "all":
        checks = [check for check in checks if check.name == args.suite]
    return run_checks(checks)


if __name__ == "__main__":
    raise SystemExit(main())
