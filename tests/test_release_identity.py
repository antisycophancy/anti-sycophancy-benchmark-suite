from pathlib import Path

from suite_tools.release_audit import discover_release_files


ROOT = Path(__file__).resolve().parents[1]
FUTURE_REPOSITORY = "github.com/antisycophancy/antisycophancy-" + "benchmark-suite"
STALE_SKILL = "antisycophancy-" + "benchmark"


def _tracked_text() -> dict[str, str]:
    paths, issues = discover_release_files(ROOT)
    assert issues == []
    result = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            result[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pass
    return result


def test_unpublished_canonical_repository_is_not_claimed():
    hits = [path for path, text in _tracked_text().items() if FUTURE_REPOSITORY in text]
    assert hits == []


def test_stale_skill_name_is_absent_from_public_source():
    hits = [path for path, text in _tracked_text().items() if STALE_SKILL in text]
    assert hits == []


def test_openrouter_requests_do_not_claim_an_unpublished_referer():
    text = (ROOT / "suite_tools" / "throughput_probe.py").read_text()
    assert "HTTP-Referer" not in text
