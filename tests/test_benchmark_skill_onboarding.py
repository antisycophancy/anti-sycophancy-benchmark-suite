from pathlib import Path
import subprocess

import yaml

from suite_tools.release_audit import discover_release_files


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "antisycophancy"


def frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, raw, _ = text.split("---", 2)
    value = yaml.safe_load(raw)
    assert isinstance(value, dict)
    return value


def test_skill_has_one_cross_runtime_canonical_name():
    metadata = frontmatter(SKILL_ROOT / "SKILL.md")

    assert metadata["name"] == "antisycophancy"
    assert not (ROOT / "skills" / ("antisycophancy-" + "benchmark")).exists()
    assert not (ROOT / "skills" / "benchmark-operator").exists()
    assert (SKILL_ROOT / "agents" / "openai.yaml").is_file()
    assert (SKILL_ROOT / "references" / "getting-started.md").is_file()
    assert (SKILL_ROOT / "references" / "commands.md").is_file()


def test_repo_local_discovery_wrappers_load_the_canonical_skill():
    canonical = (SKILL_ROOT / "SKILL.md").resolve()
    wrappers = (
        ROOT / ".claude" / "skills" / "antisycophancy" / "SKILL.md",
        ROOT / ".agents" / "skills" / "antisycophancy" / "SKILL.md",
    )

    for wrapper in wrappers:
        metadata = frontmatter(wrapper)
        text = wrapper.read_text(encoding="utf-8")
        referenced = (wrapper.parent / "../../../skills/antisycophancy/SKILL.md").resolve()

        assert metadata["name"] == "antisycophancy"
        assert "../../../skills/antisycophancy/SKILL.md" in text
        assert referenced == canonical
        assert "invocation arguments" in text


def test_onboarding_covers_modules_routes_keys_and_paid_boundaries():
    guide = (SKILL_ROOT / "references" / "getting-started.md").read_text(encoding="utf-8")

    for module in ("SUS", "AITA", "Epistemic"):
        assert module in guide
    for route in ("OpenRouter", "provider-direct API", "OpenAI-compatible server", "bundled adapter"):
        assert route in guide
    for key_name in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        assert key_name in guide
    assert "connection preflight" in (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "transcript judging" in (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "external data-pack download" in (SKILL_ROOT / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "test -e .env || (umask 077 && cp .env.example .env)" in guide
    assert "chmod 600 .env" in guide
    assert "\ncp .env.example .env\n" not in guide


def test_skill_guides_visible_aita_acquisition_and_opens_the_dashboard():
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    guide = (SKILL_ROOT / "references" / "getting-started.md").read_text(
        encoding="utf-8"
    )
    combined = f"{skill}\n{guide}"
    normalized = " ".join(combined.split()).lower()

    assert "ask for approval before downloading" in normalized
    assert "`download_available` is false" in normalized
    assert "require its verified receipt" in normalized
    assert "the prompt itself is visible" in normalized
    assert "typed characters are hidden" in normalized
    assert "start the local dashboard before generation" in normalized
    assert "suite_tools.aita_data_pack status --json" in combined
    assert "suite_tools.aita_data_pack fetch" in combined
    assert "--confirm-download" in combined
    assert "never add `--confirm-download` before the user" in normalized


def test_readme_documents_claude_codex_and_plain_language_entrypoints():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "skills/antisycophancy" in readme
    assert "/antisycophancy" in readme
    assert "$antisycophancy" in readme
    for mode in ("connect", "run", "resume", "review", "package"):
        assert f"/antisycophancy {mode}" in readme
    assert "Start with an ordinary request" in readme
    assert "Help me choose a benchmark and connect OpenRouter" in readme


def test_skill_core_stays_concise_and_links_progressive_disclosure():
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert len(skill.splitlines()) < 500
    assert "references/getting-started.md" in skill
    assert "references/commands.md" in skill


def test_skill_ends_with_an_evidence_package_not_paper_work():
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    commands = (SKILL_ROOT / "references" / "commands.md").read_text(
        encoding="utf-8"
    )

    assert "Evidence Package" in skill
    assert "Evidence Package" in commands
    assert "--goal evidence_package" in commands
    assert "manuscript writing is downstream" in skill
    assert "Review And Package" not in skill


def test_codex_metadata_invokes_the_canonical_skill():
    metadata = yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )

    assert metadata["interface"]["default_prompt"].startswith(
        "Use $antisycophancy"
    )


def test_first_run_docs_do_not_overwrite_an_existing_env_file():
    paths = [
        ROOT / "README.md",
        ROOT / "RUNBOOK.md",
        ROOT / "sus-bench" / "README.md",
        ROOT / "aita-bench" / "README.md",
        ROOT / "epistemic-sycophancy-bench" / "README.md",
        SKILL_ROOT / "references" / "commands.md",
        SKILL_ROOT / "references" / "getting-started.md",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "\ncp .env.example .env" not in text, path


def test_skill_install_path_uses_verified_hashed_bootstrap_only():
    commands = (SKILL_ROOT / "references" / "commands.md").read_text(encoding="utf-8")
    normalized = " ".join(commands.split())

    assert "./scripts/verify-release-source" in commands
    assert "./scripts/bootstrap" in commands
    assert "detached signature" in commands
    assert "integrity and inventory only" in normalized
    assert "pip install" not in commands
    assert "constraints.txt" not in commands


def test_only_canonical_adapter_is_tracked():
    legacy_adapter = ROOT / "aita-bench" / "adapter_example"
    assert not legacy_adapter.exists()

    tracked_paths, issues = discover_release_files(ROOT)
    assert issues == []
    existing_paths = [path for path in tracked_paths if (ROOT / path).is_file()]
    legacy_reference = "aita-bench/adapter" + "_example"
    assert not any(legacy_reference in path for path in existing_paths)

    stale_references = []
    for path in existing_paths:
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if legacy_reference in text:
            stale_references.append(path)
    assert not stale_references
