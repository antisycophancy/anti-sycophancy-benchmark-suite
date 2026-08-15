"""Gate for tests that assert against real, on-disk prepared contracts.

Why this module exists
----------------------
The real-contract tests used to be gated with a bare
``@pytest.mark.skipif(not REFERENCE.exists())``.  That is a *silent* gate.
``results/`` is gitignored, so it is absent in every git worktree and in CI —
and the tests simply vanished from the run.  The suite went green while
asserting nothing.

That blindness is not hypothetical: commit ``3392139`` (2026-07-18) narrowed the
nested judge-config identity projection and moved every frozen judge-panel
constant.  The only tests that would have caught it were skipping in every
environment where they ran, so the break went unnoticed for six days.  Skips
must therefore be reserved for the one case where they are honest — a checkout
that has no benchmark data at all.

The discriminator
-----------------
``results/prepared/`` is created only by ``suite_tools.prepare_run``.  Its
presence means "this checkout carries prepared benchmark data", which is
exactly the condition under which the frozen reference contracts are expected
on disk.

* ``results/prepared/`` **absent** → genuine fresh clone, no benchmark data at
  all.  There is nothing to assert against; SKIP is honest.
* ``results/prepared/`` **present** → data-bearing checkout.  A missing frozen
  reference contract is then a real defect (deleted, renamed, partially synced,
  or a reference path that drifted), not an absence of data.  FAIL loudly.

Rejected alternatives, and why:

* *"Does ``results/`` exist?"* — too coarse.  ``results/`` also holds report
  markdown and PDFs, so a docs-only checkout would trip the fail branch.
  ``results/prepared/`` is specifically the machine-written contract tree.
* *"Probe git history / branch name to detect the main checkout."* — fragile in
  shallow clones and CI detached HEADs, and it would still skip in worktrees,
  which is precisely the environment that hid this bug.
* *"Always fail."* — breaks genuine fresh clones, which is the one case where
  having no data is normal.

Escape hatch
------------
Someone who clones this repo and prepares their *own* unrelated run will have a
populated ``results/prepared/`` without our frozen references.  Setting
``BENCH_ALLOW_MISSING_REFERENCE_CONTRACTS=1`` restores skip behaviour for them.
It is opt-in and must be set deliberately, so it cannot silently re-introduce
the regression class described above.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ALLOW_MISSING_ENV = "BENCH_ALLOW_MISSING_REFERENCE_CONTRACTS"

RUN = "run"
SKIP = "skip"
FAIL = "fail"


def reference_gate_decision(
    prepared_root: Path,
    contract_path: Path,
    *,
    allow_missing: bool = False,
) -> tuple[str, str]:
    """Decide how a real-contract test should behave.  Pure; no pytest calls.

    Returns ``(decision, reason)`` where ``decision`` is one of ``"run"``,
    ``"skip"`` or ``"fail"``.  See the module docstring for the rationale.
    """
    if contract_path.exists():
        return RUN, f"reference contract present: {contract_path}"

    if not prepared_root.exists():
        return SKIP, (
            f"no benchmark data in this checkout ({prepared_root} does not exist) — "
            "genuine fresh clone, nothing to assert against"
        )

    if allow_missing:
        return SKIP, (
            f"{prepared_root} is populated but {contract_path} is missing; "
            f"skipping because {ALLOW_MISSING_ENV} is set"
        )

    return FAIL, (
        f"This checkout is data-bearing ({prepared_root} exists), but the frozen "
        f"reference contract is missing:\n"
        f"    {contract_path}\n"
        f"That is a real defect, not an absence of data — the contract was "
        f"deleted, renamed, or only partially synced.\n"
        f"Fix the data (re-sync or re-prepare the reference run), or, if this is "
        f"a third-party checkout carrying only your own unrelated runs, set "
        f"{ALLOW_MISSING_ENV}=1 to opt out of this check."
    )


def _allow_missing() -> bool:
    return os.environ.get(ALLOW_MISSING_ENV, "").strip() not in ("", "0", "false", "False")


def require_reference_contract(prepared_root: Path, contract_path: Path) -> None:
    """Skip (fresh clone) or fail (data-bearing checkout) when the contract is absent."""
    decision, reason = reference_gate_decision(
        prepared_root, contract_path, allow_missing=_allow_missing()
    )
    if decision == SKIP:
        pytest.skip(reason)
    elif decision == FAIL:
        pytest.fail(reason, pytrace=False)


def require_reference_contracts(prepared_root: Path, contract_paths: list[Path]) -> None:
    """``require_reference_contract`` for a set of contracts that must all be present."""
    for path in contract_paths:
        require_reference_contract(prepared_root, path)
