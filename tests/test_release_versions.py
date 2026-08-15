from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_all_public_version_identities_match():
    versions = {}
    for relative in (
        "pyproject.toml",
        "sus-bench/pyproject.toml",
        "aita-bench/pyproject.toml",
        "epistemic-sycophancy-bench/pyproject.toml",
    ):
        versions[relative] = tomllib.loads((ROOT / relative).read_text())["project"]["version"]
    for relative in (
        "sus-bench/sus_bench/__init__.py",
        "aita-bench/aita_bench/__init__.py",
        "epistemic-sycophancy-bench/epis_bench/__init__.py",
        "unified_profile/__init__.py",
    ):
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', (ROOT / relative).read_text(), re.MULTILINE)
        assert match
        versions[relative] = match.group(1)
    citation = re.search(
        r'^version:\s*"([^"]+)"',
        (ROOT / "CITATION.cff").read_text(),
        re.MULTILINE,
    )
    assert citation
    versions["CITATION.cff"] = citation.group(1)
    assert set(versions.values()) == {"1.0.0"}, versions


def test_public_packages_advertise_only_the_supported_python_matrix():
    for relative in (
        "pyproject.toml",
        "sus-bench/pyproject.toml",
        "aita-bench/pyproject.toml",
        "epistemic-sycophancy-bench/pyproject.toml",
    ):
        project = tomllib.loads((ROOT / relative).read_text())["project"]
        assert project["requires-python"] == ">=3.11,<3.14", relative


def test_public_packages_advertise_the_1_0_stable_release_status():
    for relative in (
        "pyproject.toml",
        "sus-bench/pyproject.toml",
        "aita-bench/pyproject.toml",
        "epistemic-sycophancy-bench/pyproject.toml",
    ):
        project = tomllib.loads((ROOT / relative).read_text())["project"]
        classifiers = project.get("classifiers", [])
        assert "Development Status :: 5 - Production/Stable" in classifiers, relative
        assert not any("Alpha" in classifier for classifier in classifiers), relative
