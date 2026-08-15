import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-release.txt"
INPUT = ROOT / "requirements-release.in"
PYPROJECTS = (
    ROOT / "pyproject.toml",
    ROOT / "sus-bench" / "pyproject.toml",
    ROOT / "aita-bench" / "pyproject.toml",
    ROOT / "epistemic-sycophancy-bench" / "pyproject.toml",
)


def _locked_requirements() -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    current: list[str] = []
    for raw_line in LOCK.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current.append(line.rstrip("\\"))
        if line.endswith("\\"):
            continue
        requirement = " ".join(current)
        current = []
        name = requirement.split("==", 1)[0].strip().lower().replace("_", "-")
        parsed[name] = requirement.split()
    assert not current
    return parsed


def test_release_requirements_are_fully_hashed_and_cover_release_inputs():
    requirements = _locked_requirements()

    assert requirements
    for name, tokens in requirements.items():
        assert any(token.startswith("--hash=sha256:") for token in tokens), name
    for name in (
        "httpx",
        "openai",
        "pyyaml",
        "fastapi",
        "python-dotenv",
        "uvicorn",
        "pip",
        "setuptools",
        "pytest",
        "jinja2",
        "rich",
        "pandas",
        "hatchling",
        "editables",
        "pip-audit",
        "bandit",
        "cryptography",
    ):
        assert name in requirements

    assert "cryptography==50.0.0" in INPUT.read_text().splitlines()
    root_dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert "cryptography>=50,<51" in root_dependencies


def test_release_lock_inputs_are_exact_and_document_the_generator():
    text = INPUT.read_text()

    assert "pip-compile" in text
    assert "--generate-hashes" in text
    direct_requirements = [
        line for line in text.splitlines() if line and not line.startswith("#")
    ]
    assert direct_requirements
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", line) for line in direct_requirements)
    requirements = _locked_requirements()
    assert {
        line.split("==", 1)[0].lower().replace("_", "-")
        for line in direct_requirements
    }.issubset(requirements)


def test_build_isolation_uses_the_hatchling_locked_for_the_release():
    requirements = _locked_requirements()
    hatchling_version = requirements["hatchling"][0].split("==", 1)[1]

    assert all(
        tomllib.loads(path.read_text())["build-system"]["requires"]
        == [f"hatchling=={hatchling_version}"]
        for path in PYPROJECTS
    )


def test_ci_token_is_read_only_and_checkout_does_not_persist_credentials():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "permissions:\n  contents: read\n" in workflow
    assert "persist-credentials: false" in workflow
    assert "contents: write" not in workflow
