from pathlib import Path

from suite_tools.offline_gate import REPO_ROOT, build_checks
from suite_tools.suite_registry import suite_root


def test_offline_gate_runs_first_party_suites_in_isolated_roots():
    checks = build_checks(python_executable="/python")

    assert [check.name for check in checks] == ["shared", "aita", "epis", "sus"]
    assert checks[0].cwd == REPO_ROOT
    assert checks[0].args == (
        "/python",
        "-m",
        "pytest",
        "-q",
        "-rs",
        "tests",
        "unified_profile/tests",
    )
    assert checks[1].cwd == suite_root("aita")
    assert checks[2].cwd == suite_root("epistemic")
    assert checks[3].cwd == suite_root("sus")


def test_offline_gate_does_not_collect_vendored_or_duplicate_bench_repos():
    check_roots = {check.cwd.relative_to(REPO_ROOT) for check in build_checks()}

    assert Path("repos") not in check_roots
    assert Path("syco-bench") not in check_roots
