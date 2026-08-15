"""Central registry for first-party benchmark suite identity.

Keep suite folder names, import packages, and path markers in one place so
renames and public/private artifact policy have a small interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SuiteSpec:
    """Stable identity for one benchmark suite."""

    key: str
    root_name: str
    import_package: str
    family_id: str
    aliases: tuple[str, ...] = ()
    path_markers: tuple[str, ...] = ()
    test_types: tuple[str, ...] = ()

    @property
    def root(self) -> Path:
        return REPO_ROOT / self.root_name

    @property
    def results_root(self) -> Path:
        return self.root / "results"

    def matches_text(self, text: str) -> bool:
        normalized = text.lower()
        markers = (self.root_name, self.import_package, self.key, *self.aliases, *self.path_markers)
        return any(marker and marker.lower() in normalized for marker in markers)


SUS = SuiteSpec(
    key="sus",
    root_name="sus-bench",
    import_package="sus_bench",
    family_id="sus",
    path_markers=("sus-bench-",),
)
AITA = SuiteSpec(
    key="aita",
    root_name="aita-bench",
    import_package="aita_bench",
    family_id="aita",
)
EPISTEMIC = SuiteSpec(
    key="epistemic",
    root_name="epistemic-sycophancy-bench",
    import_package="epis_bench",
    family_id="epistemic",
    aliases=("epis",),
    test_types=("delusion", "pickside", "mirror"),
)

FIRST_PARTY_SUITES: tuple[SuiteSpec, ...] = (AITA, EPISTEMIC, SUS)
SUITES_BY_KEY = {suite.key: suite for suite in FIRST_PARTY_SUITES}
SUITES_BY_ALIAS = {
    alias: suite
    for suite in FIRST_PARTY_SUITES
    for alias in (suite.key, suite.family_id, suite.root_name, suite.import_package, *suite.aliases)
}


def get_suite(key: str) -> SuiteSpec:
    """Return a suite spec by key, alias, root folder, or import package."""
    normalized = str(key).strip().lower()
    try:
        return SUITES_BY_ALIAS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark suite: {key}") from exc


# Canonical short form used by adapters/predicates for each suite.
# SUS and AITA use the suite key directly; EPISTEMIC collapses to "epis"
# because adapter dicts, unit_state predicates, and scoring contracts are
# keyed to the short alias rather than the full "epistemic" key.
_ADAPTER_CANONICAL: dict[str, str] = {
    "sus": "sus",
    "aita": "aita",
    "epistemic": "epis",
}


def normalize_module_name(name: str) -> str:
    """Return the canonical module name ("aita", "epis", or "sus").

    Accepts any alias registered for a first-party suite, including "epistemic"
    which maps to "epis".  Raises ``ValueError`` for names that are not
    recognised by the suite registry.

    Parameters
    ----------
    name:
        Raw module name as it appears in a contract or CLI argument
        (e.g. "epistemic", "epis", "aita", "sus").

    Returns
    -------
    str
        One of "aita", "epis", or "sus".

    Raises
    ------
    ValueError
        When *name* is not an alias of any first-party suite.
    """
    try:
        suite = get_suite(str(name).strip())
    except KeyError:
        raise ValueError(
            f"Unknown module name {name!r}; expected aita, epis/epistemic, or sus"
        ) from None
    return _ADAPTER_CANONICAL[suite.key]


def suite_root(key: str) -> Path:
    """Return the repository-local root for a first-party suite."""
    return get_suite(key).root


def module_key_for_record(record: dict[str, Any] | None, source: str | Path = "") -> str:
    """Infer a benchmark module key from a record and/or source path."""
    record = record or {}
    if record.get("module_hint"):
        return str(record["module_hint"])

    explicit = record.get("module")
    if explicit in SUITES_BY_ALIAS:
        return get_suite(str(explicit)).key

    test_type = record.get("test_type")
    if test_type in EPISTEMIC.test_types:
        return EPISTEMIC.key

    text = str(source).lower()
    for suite in FIRST_PARTY_SUITES:
        if suite.matches_text(text):
            return suite.key
    return str(explicit or "generic")
