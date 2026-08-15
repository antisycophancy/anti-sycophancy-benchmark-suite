from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _docs() -> str:
    return "\n".join(
        (ROOT / relative).read_text()
        for relative in (
            "README.md",
            "RUNBOOK.md",
            "docs/SCORING_MECHANICS.md",
            "aita-bench/README.md",
            "sus-bench/README.md",
            "epistemic-sycophancy-bench/README.md",
        )
    )


def test_score_directions_are_not_presented_as_universal():
    docs = _docs()
    assert "Higher = worse, everywhere" not in docs
    assert "AITA" in docs and "higher is better" in docs
    assert "higher susceptibility and capitulation\nare worse" in docs
    assert "mixed directions" in docs
    assert "on all of them a higher score is worse" not in docs


def test_cost_guidance_uses_token_math_and_unknown_pricing():
    docs = _docs().lower()
    assert "uncached input tokens" in docs
    assert "cached input tokens" in docs
    assert "output tokens" in docs
    assert "pricing snapshot" in docs
    assert "unknown pricing remains unknown" in docs
    assert "provider account" in docs
    assert "roughly $1/model" not in docs
    assert "~$1/model" not in docs
    assert "first run (a few cents)" not in docs


def test_distribution_boundary_is_source_only():
    readme = (ROOT / "README.md").read_text()
    normalized = " ".join(readme.lower().split())
    assert "signed git release tag" in normalized
    assert "signed release archive" in normalized
    assert "`sha256sums` file" in normalized
    assert "standalone pypi packages and wheels are not supported" in normalized
    assert "verify file integrity and inventory" in normalized
    assert "do not identify who published an archive" in normalized
    assert "scripts/bootstrap" in readme


def test_key_file_onboarding_uses_owner_only_permissions():
    docs = [
        ROOT / "README.md",
        ROOT / "RUNBOOK.md",
        ROOT / "adapter" / "README.md",
        ROOT / "aita-bench" / "README.md",
        ROOT / "sus-bench" / "README.md",
        ROOT / "epistemic-sycophancy-bench" / "README.md",
        ROOT / "skills" / "antisycophancy" / "references" / "commands.md",
        ROOT / "skills" / "antisycophancy" / "references" / "getting-started.md",
    ]

    for path in docs:
        text = path.read_text()
        assert "umask 077" in text, path
        assert "chmod 600" in text, path
        assert "|| cp .env.example .env" not in text, path


def test_public_install_docs_use_the_verified_bootstrap_only():
    docs = [
        ROOT / "README.md",
        ROOT / "RUNBOOK.md",
        ROOT / "adapter" / "README.md",
        ROOT / "aita-bench" / "README.md",
        ROOT / "sus-bench" / "README.md",
        ROOT / "epistemic-sycophancy-bench" / "README.md",
        ROOT / "skills" / "antisycophancy" / "references" / "commands.md",
        ROOT / "skills" / "antisycophancy" / "references" / "getting-started.md",
    ]

    for path in docs:
        text = path.read_text()
        assert "constraints.txt" not in text, path
        assert "pip install" not in text, path
    assert "./scripts/bootstrap" in (ROOT / "adapter" / "README.md").read_text()
    assert "./scripts/bootstrap" in (ROOT / "aita-bench" / "README.md").read_text()


def test_public_docs_describe_the_external_sealed_aita_pack_without_plaintext_paths():
    docs = _docs()
    assert "--sealed-pack" in docs
    assert "anti-indexing friction" in docs
    assert "not confidentiality" in docs
    assert "aita-sealed-pack-v1.json" in docs

    stale_plaintext_commands = (
        "--og-data aita-bench/data/curated/aita_reversed_n20_v1/og.csv",
        "--flip-data aita-bench/data/curated/aita_reversed_n20_v1/flip.csv",
        "--item-selection aita-bench/data/curated/aita_reversed_n20_v1/selection.yaml",
    )
    assert all(command not in docs for command in stale_plaintext_commands)


def test_readme_first_smoke_is_one_complete_ordered_workflow():
    readme = (ROOT / "README.md").read_text()
    smoke = readme.split("### Prepare an OpenRouter smoke run", 1)[1].split(
        "## Connect your model", 1
    )[0]
    commands = (
        "suite_tools.openrouter_preflight",
        "suite_tools.prepare_run",
        "suite_tools.preflight_conditions",
        "suite_tools.live_dashboard",
        "suite_tools.scheduler run",
        "suite_tools.bench verify",
        "suite_tools.hygiene_gate",
        "suite_tools.scheduler score",
    )

    positions = [smoke.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "--gate-after-generation" in smoke
    assert "--run-dir results/prepared/first-smoke/sus" in smoke
    assert "live_dashboard" in smoke and " &" not in smoke


def test_repository_agent_entrypoints_route_to_current_safe_workflow():
    claude = (ROOT / "CLAUDE.md").read_text()
    agents = (ROOT / "AGENTS.md").read_text()
    runbook = (ROOT / "docs" / "AGENT_RUNBOOK.md").read_text()

    for entrypoint in (claude, agents):
        assert "skills/antisycophancy/SKILL.md" in entrypoint
        assert "docs/AGENT_RUNBOOK.md" in entrypoint
        assert "suite_tools.companion resume --json" in entrypoint

    for command in (
        "suite_tools.prepare_run",
        "suite_tools.preflight_conditions",
        "suite_tools.live_dashboard",
        "suite_tools.scheduler run",
        "suite_tools.bench verify",
        "suite_tools.hygiene_gate",
        "suite_tools.scheduler score",
    ):
        assert command in runbook

    assert "http://127.0.0.1:8765" in claude
    assert "http://127.0.0.1:8765" in agents
    assert "http://127.0.0.1:8765" in runbook


def test_operator_copy_is_plain_bounded_and_avoids_em_dashes():
    paths = (
        ROOT / "README.md",
        ROOT / "RUNBOOK.md",
        ROOT / "CLAUDE.md",
        ROOT / "AGENTS.md",
        ROOT / "docs" / "AGENT_RUNBOOK.md",
    )
    text = "\n".join(path.read_text() for path in paths)

    assert "—" not in text
    assert "What that reveals is consistent" not in text
    assert "the way sycophancy actually happens" not in text
    assert "The easiest way to seem helpful is to agree" not in text
    assert "when the user keeps pushing" not in text
    assert len((ROOT / "README.md").read_text().splitlines()) < 650


def test_readme_hero_explains_the_adaptive_mechanism():
    readme = (ROOT / "README.md").read_text()
    hero = readme.split(
        "## Three adaptive benchmarks for different forms of sycophancy", 1
    )[0]
    hero = " ".join(hero.split())

    assert "suite of adaptive, multi-turn benchmarks" in hero
    assert "what the model actually says" in hero
    assert "reframes, pressure, and perspective shifts" in hero
    assert "included agent skill or CLI" in hero
    assert "local dashboard" in hero


def test_readme_defines_sycophancy_and_explains_the_shared_runner():
    readme = (ROOT / "README.md").read_text()
    section = readme.split(
        "## Three adaptive benchmarks for different forms of sycophancy", 1
    )[1].split("## The runner, agent skill, and dashboard", 1)[0]

    assert "shifts its answer toward what a user seems to want" in section
    assert "separate seeker adapts the next user turn" in section
    assert "shared runner" in section
    assert "underlying benchmark condition stays fixed" in section


def test_readme_puts_requirements_and_key_storage_before_the_quick_starts():
    readme = (ROOT / "README.md").read_text()
    requirements = readme.split("## What you need", 1)[1].split(
        "## Quick start with the agent skill", 1
    )[0]

    assert "Python 3.11, 3.12, or 3.13" in requirements
    assert "OpenRouter API key" in requirements
    assert "direct provider API key" in requirements
    assert "OpenAI-compatible endpoint" in requirements
    assert "OPENROUTER_API_KEY" in requirements
    assert "ignored `.env`" in requirements
    assert "synthetic smoke data" in requirements


def test_readme_explains_release_verification_without_implying_obfuscation():
    readme = (ROOT / "README.md").read_text()
    install = readme.split("### Install the release", 1)[1].split(
        "### Prepare an OpenRouter smoke run", 1
    )[0]
    install = " ".join(install.split())

    assert "ordinary readable Python and text" in install
    assert "does not encrypt or obfuscate" in install
    assert "What the signature and checksums verify" in install


def test_readme_describes_aita_as_a_separate_text_data_add_on():
    readme = (ROOT / "README.md").read_text()
    aita_pack = readme.split("## AITA data add-on", 1)[1].split(
        "## Watch a run", 1
    )[0]
    aita_pack = " ".join(aita_pack.split())

    assert "separate add-on" in aita_pack
    assert "readable JSON envelope" in aita_pack
    assert "adjacent `.sealed` file" in aita_pack
    assert "opens the pack in memory" in aita_pack
    assert "CSV, JSON, YAML, and text files" in aita_pack
    assert "synthetic smoke fixtures" in aita_pack
    assert "less likely to appear in search results" in aita_pack
    assert "not confidentiality" in aita_pack


def test_readme_presents_agent_skill_and_cli_as_product_interfaces():
    readme = (ROOT / "README.md").read_text()
    section = readme.split("## Quick start with the agent skill", 1)[1].split(
        "## Quick start from the CLI", 1
    )[0]

    assert "/antisycophancy" in section
    assert "$antisycophancy" in section
    assert "### Prefer direct control?" in section
    assert "CLI" in section
    assert "open the dashboard" in section
    assert "CLAUDE.md" in section
    assert "AGENTS.md" in section
    assert "docs/AGENT_RUNBOOK.md" in section
    assert "RUN_CONTRACT.json" not in section


def test_readme_agent_prompts_cover_full_suite_adapter_and_dashboard_workflows():
    readme = (ROOT / "README.md").read_text()
    section = readme.split("## Quick start with the agent skill", 1)[1].split(
        "### Prefer direct control?", 1
    )[0]
    normalized = " ".join(section.split())

    assert "Run all three benchmarks against MODEL_KEY through OpenRouter" in normalized
    assert "full AITA N=20 add-on" in normalized
    assert "open the dashboard before generation" in normalized
    assert "Inspect my local model API" in normalized
    assert "build only the adapter translation" in normalized
    assert "Do not make a paid call" in normalized


def test_readme_explains_aita_download_consent_and_hidden_input_plainly():
    readme = (ROOT / "README.md").read_text()
    section = readme.split("## AITA data add-on", 1)[1].split(
        "## Watch a run", 1
    )[0]
    normalized = " ".join(section.split())

    assert "ask before downloading" in normalized
    assert "direct asset URLs" in normalized
    assert "The prompt itself is visible" in normalized
    assert "characters you type are hidden" in normalized
    assert "preparation and generation" in normalized.lower()
    assert "does not mean the download happens silently" in normalized
    assert "Release verification separately confirms" in normalized
    assert "frozen size and SHA-256" in normalized
    assert "suite_tools.aita_data_pack status --json" in section
    assert "suite_tools.aita_data_pack fetch" in section
    assert "--confirm-download" in section
    assert "suite_tools.aita_data_pack verify" in section
    assert "never asks for Part B" in normalized


def test_readme_dashboard_image_is_a_real_local_png():
    readme = (ROOT / "README.md").read_text()
    image_path = ROOT / "docs" / "assets" / "dashboard-cockpit.png"

    assert "![Local benchmark dashboard](docs/assets/dashboard-cockpit.png)" in readme
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_readme_starts_with_the_branded_suite_header():
    readme = (ROOT / "README.md").read_text()
    header = ROOT / "docs" / "assets" / "anti-sycophancy-suite-header.png"

    assert readme.startswith(
        "![Anti-Sycophancy Benchmark Suite header]"
        "(docs/assets/anti-sycophancy-suite-header.png)"
    )
    assert header.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_repository_launch_images_have_the_exact_public_dimensions():
    import struct

    expected = {
        "anti-sycophancy-suite-header.png": (1600, 480),
        "anti-sycophancy-suite-social-preview.png": (1280, 640),
    }
    for filename, dimensions in expected.items():
        image = ROOT / "docs" / "assets" / filename
        payload = image.read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert struct.unpack(">II", payload[16:24]) == dimensions
        assert image.stat().st_size < 1_000_000
