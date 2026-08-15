from pathlib import Path
import tomllib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]

_RELEASE_PLAN_PATH = ROOT / "plans" / "014-release-stabilization-and-scale.md"


def _project(path: str) -> dict:
    return tomllib.loads((ROOT / path / "pyproject.toml").read_text())["project"]


def test_release_packages_publish_one_supported_python_and_pandas_matrix():
    projects = [
        tomllib.loads((ROOT / "pyproject.toml").read_text())["project"],
        _project("sus-bench"),
        _project("aita-bench"),
        _project("epistemic-sycophancy-bench"),
    ]

    assert {project["requires-python"] for project in projects} == {">=3.11,<3.14"}
    for project in projects[2:]:
        assert "pandas>=2.2,<3.1" in project["dependencies"]

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    matrix = workflow["jobs"]["offline-gate"]["strategy"]["matrix"]["python-version"]
    assert matrix == ["3.11", "3.12", "3.13"]

    readme = (ROOT / "README.md").read_text()
    assert "Python 3.11, 3.12, or 3.13" in readme
    assert "single local POSIX machine" in readme


def test_release_distinguishes_code_license_from_third_party_dataset_rights():
    readme = (ROOT / "README.md").read_text()
    policy = (ROOT / "docs" / "DATA_RIGHTS_AND_PRIVACY.md").read_text()
    pack_readme = (
        ROOT
        / "aita-bench"
        / "data"
        / "curated"
        / "aita_reversed_n20_v1"
        / "PACK.md"
    ).read_text()

    assert "docs/DATA_RIGHTS_AND_PRIVACY.md" in readme
    assert "does not grant rights" in policy
    assert "Reddit-derived prompt text" in policy
    assert "separately signed sealed data pack" in policy
    assert "personal information" in policy
    assert "removal" in policy.lower()
    assert "DATA_RIGHTS_AND_PRIVACY.md" in pack_readme
    assert "separately signed sealed data-pack" in pack_readme
    assert "not confidentiality" in pack_readme
    assert not (ROOT / "aita-bench" / "data" / "curated" / "aita_reversed_n20_v1" / "og.csv").exists()


def test_documented_clean_install_includes_the_tested_reference_adapter():
    root_project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    adapter_dependencies = root_project["optional-dependencies"]["adapter"]

    assert "fastapi>=0.115,<1" in adapter_dependencies
    assert "uvicorn>=0.32,<1" in adapter_dependencies

    readme = (ROOT / "README.md").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    bootstrap = (ROOT / "scripts" / "bootstrap").read_text()
    lock = (ROOT / "requirements-release.txt").read_text()
    assert "scripts/bootstrap" in readme
    assert "--require-hashes" in bootstrap
    assert "fastapi==" in lock
    assert "uvicorn==" in lock
    assert "scripts/bootstrap" in workflow
    assert "pip-audit --local --skip-editable" in workflow


def test_clean_install_upgrades_to_the_audited_pip_constraint():
    constraints = (ROOT / "constraints.txt").read_text()
    readme = (ROOT / "README.md").read_text()
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "pip==26.1.2" in constraints
    bootstrap = (ROOT / "scripts" / "bootstrap").read_text()
    lock = (ROOT / "requirements-release.txt").read_text()
    assert "--require-hashes" in bootstrap
    assert "pip==26.1.2" in lock
    assert "constraints.txt" not in bootstrap
    assert "scripts/bootstrap" in readme
    assert "scripts/bootstrap" in workflow


def test_benchmark_packages_declare_the_shared_runtime_without_a_bare_aita_dependency():
    root_project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "aita-bench" not in root_project["dependencies"]

    for package in ("sus-bench", "aita-bench", "epistemic-sycophancy-bench"):
        assert "antisycophancy-suite-tools==1.0.0" in _project(package)["dependencies"]


def test_every_benchmark_package_installs_a_console_command():
    root_project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert root_project["scripts"]["bench"] == "suite_tools.bench:main"
    expected = {
        "sus-bench": ("sus-bench", "sus_bench.cli:main"),
        "aita-bench": ("aita-bench", "aita_bench.cli:main"),
        "epistemic-sycophancy-bench": ("epis-bench", "epis_bench.cli:main"),
    }

    for package, (command, entrypoint) in expected.items():
        assert _project(package)["scripts"][command] == entrypoint


def test_readme_has_no_missing_local_images():
    import re

    text = (ROOT / "README.md").read_text()
    references = re.findall(r'<img\s+[^>]*src="([^"]+)"', text)
    references.extend(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))
    local_references = [path for path in references if "://" not in path]

    assert local_references
    assert [path for path in local_references if not (ROOT / path).is_file()] == []


@pytest.mark.skipif(
    not _RELEASE_PLAN_PATH.exists(),
    reason=(
        "Internal plans file absent (plans/014-release-stabilization-and-scale.md)."
        " This file is not part of the public repository tree and is only present"
        " in maintainer environments. Skip is expected on fresh clones."
    ),
)
def test_release_plan_records_capacity_fairness_and_measures_utilization_at_ramp_time():
    readme = (ROOT / "README.md").read_text()
    plan = _RELEASE_PLAN_PATH.read_text()
    runbook = (ROOT / "RUNBOOK.md").read_text()
    c5 = plan.split("### C5.", 1)[1].split("## Workstream D", 1)[0]
    milestone5 = plan.split("### Milestone 5:", 1)[1].split("### Milestone 6:", 1)[0]

    assert "strict per-run cap" in readme
    assert "starve" in readme
    assert "90%" not in c5
    assert "90%" in milestone5
    assert "1,250 / 1,250" in milestone5
    assert "canary-c64-c100" in milestone5
    assert "official-run paid-call cap of 64" in milestone5
    assert "capacity set --global 64" in runbook
    assert "BENCHMARK_PAID_CALL_MAX_ACTIVE=64" in runbook
    assert "Effective paid-call limit" in runbook
