import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_signed_tag_git(bin_dir: Path) -> Path:
    git = bin_dir / "git"
    git.parent.mkdir(parents=True, exist_ok=True)
    git.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *'rev-parse --show-toplevel'*) printf '%s\\n' \"$FAKE_GIT_ROOT\" ;;\n"
        "  *'describe --tags --exact-match HEAD'*) printf '%s\\n' 'v1.0.0' ;;\n"
        "  *'verify-tag v1.0.0'*) [ \"${FAKE_GIT_UNSIGNED:-0}\" = 0 ] ;;\n"
        "  *diff*--quiet*) exit 0 ;;\n"
        "  *'ls-files --others --exclude-standard'*) exit 0 ;;\n"
        "  *'ls-files -s'*) printf '%s\\n' '100755 deadbeef 0 scripts/verify-release-source' ;;\n"
        "  *) printf '%s\\n' \"unexpected fake git invocation: $*\" >&2; exit 90 ;;\n"
        "esac\n"
    )
    git.chmod(0o755)
    return git


def test_bootstrap_is_guarded_and_installs_only_locked_third_party_packages():
    text = (ROOT / "scripts" / "bootstrap").read_text()

    assert "3, 11" in text
    assert "3, 13" in text
    assert 'sys.implementation.name != "cpython"' in text
    assert "--require-hashes" in text
    assert "--only-binary=:all:" in text
    assert "--no-deps" in text
    assert "--no-build-isolation" in text
    assert "-e . -e sus-bench -e aita-bench -e epistemic-sycophancy-bench" in text
    assert "pip check" in text
    assert "suite_tools.model_config --validate" in text
    assert "suite_tools.offline_gate" in text
    assert "cleanup_failed_bootstrap" in text
    assert "shutil.rmtree" in text
    assert "trap - EXIT" in text
    assert ".env" not in text
    assert ".claude" not in text


def test_ci_uses_a_clean_export_and_pinned_actions_for_every_supported_python():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow
    assert "scripts/export-release" in workflow
    assert "--release-version" in workflow
    assert "scripts/bootstrap" in workflow
    assert "PYTHONPATH: \"\"" in workflow
    assert "actions/checkout@" in workflow
    assert "actions/setup-python@" in workflow
    assert (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1"
        in workflow
    )
    assert (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0"
        in workflow
    )
    for action in ("actions/checkout", "actions/setup-python"):
        line = next(line for line in workflow.splitlines() if action in line)
        assert len(line.rsplit("@", 1)[1].strip()) >= 40


def test_ci_exercises_hashed_crypto_install_on_linux_and_macos():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "runs-on: ubuntu-latest" in workflow
    assert "runs-on: macos-14" in workflow
    assert "macos-release-smoke" in workflow
    assert "tests/test_sealed_pack.py" in workflow


def test_failed_bootstrap_removes_only_the_environment_it_created(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bootstrap = scripts / "bootstrap"
    bootstrap.write_text((ROOT / "scripts" / "bootstrap").read_text())
    bootstrap.chmod(0o755)
    verifier = scripts / "verify-release-source"
    verifier.write_text((ROOT / "scripts" / "verify-release-source").read_text())
    verifier.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-\" ] && [ -z \"${2:-}\" ]; then cat >/dev/null; exit 0; fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n"
        "  mkdir -p \"$3/bin\"\n"
        "  cp \"$0\" \"$3/bin/python\"\n"
        "  chmod +x \"$3/bin/python\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then exit 42; fi\n"
        f"exec {sys.executable!s} \"$@\"\n"
    )
    fake_python.chmod(0o755)
    sentinel = tmp_path / "keep-me.txt"
    sentinel.write_text("preserve")
    manifest_paths = (bootstrap, verifier, fake_python, sentinel)
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path).as_posix()}\n"
            for path in sorted(manifest_paths)
        )
    )

    result = subprocess.run(
        [str(bootstrap)],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 42
    assert not (tmp_path / "venv").exists()
    assert sentinel.read_text() == "preserve"
    assert "removed incomplete environment" in result.stderr


def test_bootstrap_rejects_unlisted_source_before_starting_python(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bootstrap = scripts / "bootstrap"
    verifier = scripts / "verify-release-source"
    bootstrap.write_text((ROOT / "scripts" / "bootstrap").read_text())
    verifier.write_text((ROOT / "scripts" / "verify-release-source").read_text())
    bootstrap.chmod(0o755)
    verifier.chmod(0o755)
    marker = tmp_path / "python-started"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    fake_python.chmod(0o755)
    manifest_paths = (bootstrap, verifier, fake_python)
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path).as_posix()}\n"
            for path in sorted(manifest_paths)
        )
    )
    (tmp_path / "sitecustomize.py").write_text("raise SystemExit('must not run')\n")

    result = subprocess.run(
        [str(bootstrap)],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert "source inventory differs" in result.stderr


def test_source_verifier_reports_malformed_manifest_before_hashing(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    verifier = scripts / "verify-release-source"
    verifier.write_text((ROOT / "scripts" / "verify-release-source").read_text())
    verifier.chmod(0o755)
    digest = hashlib.sha256(verifier.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  scripts/verify-release-source\n"
        "this trailing line is malformed\n"
    )

    result = subprocess.run([str(verifier)], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode != 0
    assert "malformed checksum manifest" in result.stderr


def test_source_verifier_accepts_clean_exact_signed_tag_without_manifest(tmp_path):
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    verifier = scripts / "verify-release-source"
    verifier.write_text((ROOT / "scripts" / "verify-release-source").read_text())
    verifier.chmod(0o755)
    fake_bin = tmp_path / "bin"
    _write_fake_signed_tag_git(fake_bin)

    result = subprocess.run(
        [str(verifier)],
        cwd=source,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GIT_ROOT": str(source),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "signed Git tag 'v1.0.0' verified" in result.stdout


def test_source_verifier_rejects_unsigned_exact_tag_without_manifest(tmp_path):
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    verifier = scripts / "verify-release-source"
    verifier.write_text((ROOT / "scripts" / "verify-release-source").read_text())
    verifier.chmod(0o755)
    fake_bin = tmp_path / "bin"
    _write_fake_signed_tag_git(fake_bin)

    result = subprocess.run(
        [str(verifier)],
        cwd=source,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GIT_ROOT": str(source),
            "FAKE_GIT_UNSIGNED": "1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "no valid cryptographic signature" in result.stderr
