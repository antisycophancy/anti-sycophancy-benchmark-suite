"""Assert that prepared contracts share the frozen judge-panel hashes from the
v1.0.1 fable frontier era.

Two assertion modes:

1. judge_panel_hash mode (default / original):
   Hashes are RECOMPUTED from identity.judge_panel using the fixed
   _judge_panel_for_comparison function (which excludes transport-config keys
   such as primary_config).  This verifies that the hash logic produces the right
   value regardless of what was stored in provenance at prepare time.

2. Component-wise comparison-identity mode (--component-wise):
   For each new gpt56-*-20260716 contract, asserts ALL of:
   (a) recomputed judge_panel_hash equals the frozen per-module value exactly;
   (b) recomputed benchmark_spec_hash equals the reference contract value;
   (c) identity.sample_spec['dataset_manifest'] deep-equal to reference;
   (d) identity.sample_spec['dataset_mode'] equal to reference;
   (e) item universe equal to reference — old flat item_indices vs new
       items[].item_idx describe the same index set (AITA); item_hash sets
       per test_type (EPIS); scenario_hashes dict (SUS).
   Also records/prints the GPT-5.6-era comparison_spec_hash anchor values.

Usage:
    python -m suite_tools.assert_hash_panel
    python -m suite_tools.assert_hash_panel --component-wise

    Or target specific contracts:
    python -m suite_tools.assert_hash_panel CONTRACT_PATH [CONTRACT_PATH ...]

Exits 0 if every asserted component matches, 1 if any mismatch is found.

NOTE: the frozen constants below are bound to a specific identity projection
(``CONSTANTS_BLESSED_FOR_PROJECTION_VERSION``).  They were re-blessed on
2026-07-24 for ``benchmark-identity-projection-v3`` after commit 3392139
narrowed the nested judge-config projection; see the HISTORY comment blocks and
``PREREG_FREEZE_GPT56_20260716.json`` amendment P1.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

from suite_tools.run_contract import (
    LEGACY_V3_IDENTITY_PROJECTION_VERSION,
    legacy_v3_provenance_hashes,
    stable_json_hash,
)
from suite_tools.suite_registry import suite_root as _suite_root

_EPIS_ROOT = _suite_root("epistemic")


_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "prepared"

# ===========================================================================
# IDENTITY-PROJECTION BINDING FOR THE FROZEN CONSTANTS BELOW
# ===========================================================================
# Every constant in this block is the output of hashing an identity *through a
# particular projection*.  A projection change re-values them all even when the
# underlying content is untouched, so the constants are only meaningful when
# paired with the projection that produced them.
#
# If suite_tools.run_contract.IDENTITY_PROJECTION_VERSION moves away from the
# value recorded here, these constants are stale by construction.  The pairing
# is asserted by assert_constants_blessed_for_current_projection() below (called
# from main()) and by tests/test_assert_hash_panel.py, so the next projection
# change fails loudly instead of silently.
CONSTANTS_BLESSED_FOR_PROJECTION_VERSION = "benchmark-identity-projection-v3"

# ---------------------------------------------------------------------------
# Frozen judge-panel targets  (re-blessed 2026-07-24 for projection v3)
# ---------------------------------------------------------------------------
# HISTORY — pre-3392139 values, retained so the transition is traceable:
#
#     FROZEN_AITA_JUDGE_PANEL_HASH  73b62d298053d804a476ec024966e1101d0718ed0bbd2b48b28424c21e6e3ce9
#     FROZEN_EPIS_JUDGE_PANEL_HASH  9ecc2251c3fbcebcc0d1da187f4bef3081b68b565d497b7f7f205e67e6bc1df6
#     FROZEN_SUS_JUDGE_PANEL_HASH   84e507ba76da7c9fafb32b48d2a0456783a09e02f0499e0196f6fd58b9d58e19
#
# Those were correct under the pre-v2 judge projection, which passed each entry
# of judge_panel.configs[] into the hash whole.  Commit 3392139 (2026-07-18,
# "feat(provenance): whitelist identity projections (v2) + projection_version/
# artifact stamps") narrowed the nested per-judge projection to
# JUDGE_CONFIG_IDENTITY_KEYS = {model_id, provider_api, condition_metadata},
# dropping three transport/operator fields: api_key_env, base_url, label.
#
# WHY THE VALUES MOVED BUT THE SCIENCE DID NOT:
#   * The dropped fields are transport/operator concerns, not judge identity.
#     Route identity survives via condition_metadata.provider_route.
#   * The hashed *content* is unchanged — same three-model panel, same
#     rubric_version, same judge_prompt_hashes, same rubric_source_ids/registry.
#     Re-running the old projection over today's on-disk contracts reproduces
#     the three values above byte-exactly, which proves the projection is the
#     sole mover and the contracts did not drift.
#   * Live comparisons are unaffected: compare_provenance recomputes BOTH sides
#     under one projection and never trusts a stored string, so
#     judge_panel_hash is True for fable-frontier vs gpt56-sol.
#
# See PREREG_FREEZE_GPT56_20260716.json amendment P1 for the auditable record.
FROZEN_AITA_JUDGE_PANEL_HASH = (
    "23617dd5aa1aecb4dcb2ec7b5f5946892b3e349d4ddc1ea37027c568f2161b67"
)
FROZEN_EPIS_JUDGE_PANEL_HASH = (
    "dfb77df3169963bf2c38c73ba030ef81c7b9cc694e2b5bf22624b4465d0b9c58"
)

# SUS canary (gpt56-luna-low-canary-20260715) recomputed with fixed function
FROZEN_SUS_JUDGE_PANEL_HASH = (
    "5790916fae62b147461386248da39a1c690855b0209af6fbee4156f9e2f1f3e0"
)

# Old 1-judge SUS run (sus-fable-5-native-effort-n20-20260701-142614) — expected
# to differ from the 3-judge frontier panel; included for the record only.
#
# NOT re-blessed: this value is unchanged by commit 3392139.  That panel carries
# no per-judge ``configs`` list, so the nested projection had nothing to drop and
# the hash is identical under both the old and the v3 projection.  It is a useful
# control — it confirms the three constants above moved specifically because of
# the nested judge-config narrowing, and not because of any broader hash change.
OLD_1JUDGE_SUS_JUDGE_PANEL_HASH = (
    "6133ad9464cb258b181d3736d1af412e4d56a239f7749eb8741ec4e02824e89d"
)

# ---------------------------------------------------------------------------
# Reference contract paths for component-wise comparison-identity checks
# ---------------------------------------------------------------------------
_AITA_REFERENCE_PATH = (
    _RESULTS_ROOT
    / "fable-5-native-suite-n20-frontier-20260702-142711-frontier"
    / "aita"
    / "RUN_CONTRACT.json"
)
_EPIS_REFERENCE_PATH = (
    _RESULTS_ROOT
    / "fable-5-native-suite-n20-frontier-20260702-142711-frontier"
    / "epis"
    / "RUN_CONTRACT.json"
)
_SUS_REFERENCE_PATH = (
    _RESULTS_ROOT
    / "gpt56-luna-low-canary-20260715"
    / "sus"
    / "RUN_CONTRACT.json"
)

# ---------------------------------------------------------------------------
# GPT-5.6-era comparison_spec_hash anchors
# (per-module, recorded for the prereg freeze; re-blessed 2026-07-24 for v3)
# ---------------------------------------------------------------------------
# These anchors differ from the v1.0.1 values because sample_spec serialization
# was enriched (flat index lists → per-item hash objects) while the underlying
# dataset content is provably identical.
#
# HISTORY — pre-3392139 values, retained so the transition is traceable:
#
#     GPT56_AITA_COMPARISON_SPEC_HASH  39d5bcfb208ef81e3fe5f079e87fd57ed0fb6e12dbe201c116527305f0cef24f
#     GPT56_EPIS_COMPARISON_SPEC_HASH  2bf6f965175017796e57f0327b8c65563fa7afc59821c50ef93d398afed9f9f3
#     GPT56_SUS_COMPARISON_SPEC_HASH   ed1fc3eadb589ead1159111b230e41e644344292b65f7aa1450a67c0fc8c668f
#
# comparison_spec_hash = H({benchmark_spec_hash, sample_hash, judge_panel_hash}),
# so it inherits the judge_panel_hash move above and nothing else.  Recomputing
# it from today's benchmark_spec_hash + sample_hash + the OLD judge projection
# reproduces the three values above exactly, which isolates the cause: the same
# three dropped transport fields, propagated one level up.  The v3 item_hash
# sample-axis bump contributed nothing to this particular transition.
GPT56_AITA_COMPARISON_SPEC_HASH = (
    "a164fe78f149c4098f2b610b50fde33a36b8182ac14a09fd22357bf5445e0a6b"
)
GPT56_EPIS_COMPARISON_SPEC_HASH = (
    "5d5e10b60ff350cf3606dd934669761dfa69f042530d517b45fb477676ddf7f6"
)
GPT56_SUS_COMPARISON_SPEC_HASH = (
    "573cdc86ca785ae006cbe2aa5f8cfde54f9212ec0280f7d98c0c76c07afd3806"
)


class StaleFrozenConstantsError(RuntimeError):
    """Raised when the frozen constants predate the active identity projection."""


def assert_constants_blessed_for_current_projection(
    active_projection_version: str = LEGACY_V3_IDENTITY_PROJECTION_VERSION,
) -> None:
    """Fail loudly when the identity projection has moved past the blessed one.

    The frozen constants in this module are projection-bound: they are hashes of
    identities as filtered by a specific projection.  Bumping
    ``IDENTITY_PROJECTION_VERSION`` silently re-values all of them.  This guard
    turns that silent staleness into an explicit, actionable failure.
    """
    if active_projection_version == CONSTANTS_BLESSED_FOR_PROJECTION_VERSION:
        return
    raise StaleFrozenConstantsError(
        "Frozen identity constants in suite_tools/assert_hash_panel.py are stale.\n"
        f"  blessed for projection: {CONSTANTS_BLESSED_FOR_PROJECTION_VERSION}\n"
        f"  active projection:      {active_projection_version}\n"
        "\n"
        "The identity projection changed, so every FROZEN_*_JUDGE_PANEL_HASH and\n"
        "GPT56_*_COMPARISON_SPEC_HASH below it is now the hash of a projection that\n"
        "is no longer in use.  Do NOT simply paste in new numbers.\n"
        "\n"
        "To re-bless:\n"
        "  1. Confirm the hashed CONTENT is unchanged (same panel, rubric_version,\n"
        "     judge_prompt_hashes, rubric_source_ids/registry).  If the content\n"
        "     changed, this is a real provenance break, not a re-blessing.\n"
        "  2. Recompute each constant from the stored contracts:\n"
        "       python -m suite_tools.assert_hash_panel --component-wise\n"
        "  3. Replace the constants, keeping the previous values in the adjacent\n"
        "     HISTORY comment block with the commit that moved them and why.\n"
        "  4. Set CONSTANTS_BLESSED_FOR_PROJECTION_VERSION to the new projection.\n"
        "  5. Add a dated amendment to PREREG_FREEZE_GPT56_20260716.json recording\n"
        "     the old values, the new values, the causing commit, and the evidence\n"
        "     that the scientific content is unaffected.  This is a preregistration\n"
        "     record: amend it, never rewrite it.\n"
    )


def _load_contract(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ERROR loading {path}: {exc}", file=sys.stderr)
        return {}


def _recompute_judge_panel_hash(contract: dict) -> str:
    """Recompute judge_panel_hash from identity.judge_panel using the fixed
    comparison function (excludes primary_config and other transport keys)."""
    identity = contract.get("identity")
    if not identity:
        return ""
    return legacy_v3_provenance_hashes(identity).get("judge_panel_hash") or ""


# ---------------------------------------------------------------------------
# Component-wise item-universe extractors
# ---------------------------------------------------------------------------

def _extract_item_universe_aita(sample_spec: dict) -> set:
    """Return the set of item indices from either old-flat or new per-item format.

    Old format (fable-era): sample_spec.item_indices = [0, 1, ...]
    New format (gpt56-era): sample_spec.items = [{item_idx: 0, ...}, ...]
    """
    if "item_indices" in sample_spec:
        return set(sample_spec["item_indices"])
    items = sample_spec.get("items")
    if isinstance(items, list):
        return {item["item_idx"] for item in items if isinstance(item, dict) and "item_idx" in item}
    return set()


def _extract_item_universe_epis(sample_spec: dict) -> dict:
    """Return {test_type: frozenset(item_hash)} for EPIS sample_spec.items."""
    items = sample_spec.get("items", {})
    if not isinstance(items, dict):
        return {}
    return {
        tt: frozenset(
            item["item_hash"]
            for item in item_list
            if isinstance(item, dict) and "item_hash" in item
        )
        for tt, item_list in items.items()
        if isinstance(item_list, list)
    }


def _extract_scenario_universe_sus(sample_spec: dict) -> dict:
    """Return the scenario_hashes dict for SUS sample_spec."""
    sh = sample_spec.get("scenario_hashes", {})
    return dict(sh) if isinstance(sh, dict) else {}


# ---------------------------------------------------------------------------
# Pure item-universe checker (Sol finding 4)
# ---------------------------------------------------------------------------


def item_universe_report(contract_a: dict, contract_b: dict) -> dict:
    """Compare item universes of two contracts.  Structured return, no printing.

    Parameters
    ----------
    contract_a, contract_b:
        Raw contract dicts (each carrying ``modules[0].module`` and
        ``identity.sample_spec``).

    Returns
    -------
    ``{"module": str, "match": bool, "detail": dict}``

    * If the modules differ: ``{"match": False, "detail": {"reason":
      "module_mismatch"}}``.
    * If the module is unknown: ``{"match": False, "detail": {"reason":
      "unknown_module"}}``.
    * Otherwise, uses the module-specific extractor to compare item sets:

      - ``aita``: set of item indices (supports both old ``item_indices`` and
        new ``items[].item_idx`` formats).
      - ``epistemic``: ``{test_type: frozenset(item_hash)}`` equality.
      - ``sus``: ``scenario_hashes`` dict equality.
    """
    modules_a = contract_a.get("modules") or [{}]
    modules_b = contract_b.get("modules") or [{}]
    module_a = (modules_a[0] if modules_a else {}).get("module") or ""
    module_b = (modules_b[0] if modules_b else {}).get("module") or ""

    if module_a != module_b:
        return {
            "module": module_a or module_b,
            "match": False,
            "detail": {"reason": "module_mismatch"},
        }

    module = module_a

    identity_a = (contract_a.get("identity") or {})
    identity_b = (contract_b.get("identity") or {})
    sample_spec_a = identity_a.get("sample_spec") or {}
    sample_spec_b = identity_b.get("sample_spec") or {}

    if module == "aita":
        univ_a = _extract_item_universe_aita(sample_spec_a)
        univ_b = _extract_item_universe_aita(sample_spec_b)
        match = univ_a == univ_b
        return {
            "module": module,
            "match": match,
            "detail": {"a_size": len(univ_a), "b_size": len(univ_b)},
        }

    if module == "epistemic":
        univ_a = _extract_item_universe_epis(sample_spec_a)
        univ_b = _extract_item_universe_epis(sample_spec_b)
        match = univ_a == univ_b
        return {
            "module": module,
            "match": match,
            "detail": {"test_types_a": sorted(univ_a.keys()), "test_types_b": sorted(univ_b.keys())},
        }

    if module == "sus":
        univ_a = _extract_scenario_universe_sus(sample_spec_a)
        univ_b = _extract_scenario_universe_sus(sample_spec_b)
        match = univ_a == univ_b
        return {
            "module": module,
            "match": match,
            "detail": {"scenario_count_a": len(univ_a), "scenario_count_b": len(univ_b)},
        }

    # Unknown module — cannot compare
    return {
        "module": module,
        "match": False,
        "detail": {"reason": "unknown_module"},
    }


# ---------------------------------------------------------------------------
# EPIS initial_formatter prompt-hash scheme helpers
#
# The initial_formatter prompt hash changed between fable-era and gpt56-era
# contracts solely because the fingerprint *scheme* changed, not the content:
#
#   Old scheme (runner.py, pre-fix):
#       stable_json_hash("epis_bench.prompts.format_initial_prompt")
#       → 458bce7956add32faffaf8f562556e32eb65aede789beb4948ee79503385983f
#
#   New scheme (prepare_run.py / fixed runner.py):
#       stable_json_hash(inspect.getsource(format_initial_prompt))
#       → a052f11c9494a7292c5ee727bc0c3ade9a924ba31d2699aa81b08e085d76a0d9
#
# The format_initial_prompt source was last changed 2026-06-10, before the
# July-2 Fable prepare.  Both values are recorded here and verified live by
# assert_comparison_identity so that the audit proves identical source content
# under both schemes rather than reporting a phantom benchmark change.
# ---------------------------------------------------------------------------

EPIS_INITIAL_FORMATTER_OLD_HASH = (
    "458bce7956add32faffaf8f562556e32eb65aede789beb4948ee79503385983f"
)
EPIS_INITIAL_FORMATTER_NEW_HASH = (
    "a052f11c9494a7292c5ee727bc0c3ade9a924ba31d2699aa81b08e085d76a0d9"
)


def _epis_benchmark_spec_content_equal(new_bs: dict, ref_bs: dict) -> tuple[bool, str]:
    """Assert EPIS benchmark_spec content equality across fingerprint schemes.

    For ``initial_formatter`` specifically:
      - new contract's value must == stable_json_hash(inspect.getsource(format_initial_prompt))
      - old contract's value must == stable_json_hash("epis_bench.prompts.format_initial_prompt")
    Both passing proves same source content under both schemes.

    All other prompt_hashes keys are compared directly (strict equality).

    Returns (ok, note_string).
    """
    if str(_EPIS_ROOT) not in sys.path:
        sys.path.insert(0, str(_EPIS_ROOT))
    from epis_bench import prompts as _epis_prompts  # noqa: PLC0415

    live_new = stable_json_hash(inspect.getsource(_epis_prompts.format_initial_prompt))
    live_old = stable_json_hash("epis_bench.prompts.format_initial_prompt")

    new_ph = (new_bs.get("prompt_hashes") or {})
    ref_ph = (ref_bs.get("prompt_hashes") or {})

    new_if = new_ph.get("initial_formatter", "")
    ref_if = ref_ph.get("initial_formatter", "")

    # New contract must use the source-hash scheme; ref must use the constant-string scheme.
    new_scheme_ok = new_if == live_new
    ref_scheme_ok = ref_if == live_old

    if not (new_scheme_ok and ref_scheme_ok):
        note = (
            f"initial_formatter scheme mismatch — "
            f"new={new_if[:16]}… expected {live_new[:16]}… ({new_scheme_ok}); "
            f"ref={ref_if[:16]}… expected {live_old[:16]}… ({ref_scheme_ok})"
        )
        return False, note

    # All other prompt_hashes keys must be strictly equal.
    other_keys = (set(new_ph) | set(ref_ph)) - {"initial_formatter"}
    for key in sorted(other_keys):
        if new_ph.get(key) != ref_ph.get(key):
            return False, f"prompt_hashes[{key!r}] differs: {new_ph.get(key)} vs {ref_ph.get(key)}"

    # Remaining benchmark_spec fields (other than prompt_hashes) must also match.
    for field in set(new_bs) | set(ref_bs):
        if field == "prompt_hashes":
            continue
        if new_bs.get(field) != ref_bs.get(field):
            return False, f"benchmark_spec[{field!r}] differs"

    return True, "same source content; fingerprint scheme old→new (constant-string→source-hash)"


# ---------------------------------------------------------------------------
# Component-wise comparison-identity checker
# ---------------------------------------------------------------------------

def assert_comparison_identity(
    contract_paths: list[Path],
    *,
    aita_reference_path: Path = _AITA_REFERENCE_PATH,
    epis_reference_path: Path = _EPIS_REFERENCE_PATH,
    sus_reference_path: Path = _SUS_REFERENCE_PATH,
) -> int:
    """Assert comparison-identity components for new gpt56 contracts.

    For each contract asserts:
      (a) judge_panel_hash    — vs frozen per-module constant
      (b) benchmark_spec_hash — vs recomputed from reference contract
      (c) dataset_manifest    — deep-equal to reference sample_spec value
      (d) dataset_mode        — equal to reference sample_spec value
      (e) item_universe       — module-specific set/dict equality

    Prints the GPT-5.6-era comparison_spec_hash for each contract as anchors.
    Returns 0 if all components pass, 1 on any failure.
    """
    _FROZEN_JPH = {
        "aita": FROZEN_AITA_JUDGE_PANEL_HASH,
        "epistemic": FROZEN_EPIS_JUDGE_PANEL_HASH,
        "sus": FROZEN_SUS_JUDGE_PANEL_HASH,
    }
    _REF_PATHS = {
        "aita": aita_reference_path,
        "epistemic": epis_reference_path,
        "sus": sus_reference_path,
    }

    # Pre-load references and compute their provenance hashes once.
    _refs: dict[str, dict] = {}
    _ref_hashes: dict[str, dict] = {}
    for mod, rpath in _REF_PATHS.items():
        ref_contract = _load_contract(rpath)
        if not ref_contract:
            print(f"  WARNING: reference not found for {mod}: {rpath}", file=sys.stderr)
        _refs[mod] = ref_contract
        identity = ref_contract.get("identity", {}) if ref_contract else {}
        _ref_hashes[mod] = legacy_v3_provenance_hashes(identity) if identity else {}

    failures = 0
    checked = 0
    gpt56_anchors: dict[str, str] = {}

    for path in contract_paths:
        contract = _load_contract(path)
        if not contract:
            print(f"SKIP  {path} (unreadable)")
            continue

        modules = contract.get("modules") or [{}]
        raw_module = modules[0].get("module") or "unknown" if modules else "unknown"
        # Normalise "epistemic" alias used in some older contracts
        module = raw_module if raw_module in _FROZEN_JPH else raw_module
        run_id = contract.get("run_id") or path.parent.name

        if module not in _FROZEN_JPH:
            print(f"SKIP  [{run_id}/{module}] (unknown module)")
            continue

        identity = contract.get("identity", {}) or {}
        new_hashes = legacy_v3_provenance_hashes(identity)
        new_ss = identity.get("sample_spec", {}) or {}

        ref_hashes = _ref_hashes[module]
        ref_ss = (_refs[module].get("identity", {}) or {}).get("sample_spec", {}) or {}

        def _status(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        def _record(label: str, ok: bool) -> None:
            nonlocal failures
            status = _status(ok)
            print(f"  {status}  {label:25s} [{run_id}/{module}]")
            if not ok:
                failures += 1

        # (a) judge_panel_hash
        jph = new_hashes.get("judge_panel_hash", "")
        _record("judge_panel_hash", jph == _FROZEN_JPH[module])

        # (b) benchmark_spec_hash — for EPIS, use scheme-aware content equality
        #     because initial_formatter changed fingerprint scheme (not content).
        if module == "epistemic":
            new_bs = (identity.get("benchmark_spec") or {})
            ref_identity_data = (_refs[module].get("identity", {}) or {})
            ref_bs = (ref_identity_data.get("benchmark_spec") or {})
            bsh_ok, bsh_note = _epis_benchmark_spec_content_equal(new_bs, ref_bs)
            _record("benchmark_spec_hash", bsh_ok)
            print(f"       note: {bsh_note}")
        else:
            new_bsh = new_hashes.get("benchmark_spec_hash", "")
            ref_bsh = ref_hashes.get("benchmark_spec_hash", "")
            bsh_ok = bool(new_bsh) and new_bsh == ref_bsh
            _record("benchmark_spec_hash", bsh_ok)
            if not bsh_ok and new_bsh and ref_bsh:
                print(f"       ref: {ref_bsh}")
                print(f"       new: {new_bsh}")

        # (c) dataset_manifest deep-equal
        new_dm = new_ss.get("dataset_manifest", "")
        ref_dm = ref_ss.get("dataset_manifest", "")
        _record("dataset_manifest", new_dm == ref_dm)

        # (d) dataset_mode equal
        new_mode = new_ss.get("dataset_mode") or ""
        ref_mode = ref_ss.get("dataset_mode") or ""
        _record("dataset_mode", new_mode == ref_mode)

        # (e) item universe (module-specific) — delegate to item_universe_report
        iu_report = item_universe_report(contract, _refs[module])
        _record("item_universe", iu_report["match"])

        # Record GPT-5.6-era comparison_spec_hash anchor
        csh = new_hashes.get("comparison_spec_hash", "")
        gpt56_anchors[module] = csh
        print(f"  INFO  comparison_spec_hash (gpt56-era anchor): {csh}")
        print()

        checked += 1

    # Print anchor summary
    print("GPT-5.6-era comparison_spec_hash anchors:")
    _ANCHOR_LABELS = {
        "aita": "AITA",
        "epistemic": "EPIS",
        "sus": "SUS",
    }
    for mod in ("aita", "epistemic", "sus"):
        if mod in gpt56_anchors:
            print(f"  {_ANCHOR_LABELS[mod]:6s}  {gpt56_anchors[mod]}")
    print()
    print(f"Checked {checked} contract(s).  Failures: {failures}")
    return 0 if failures == 0 else 1


def assert_contracts(
    contract_paths: list[Path],
    *,
    frozen_aita_hash: str = FROZEN_AITA_JUDGE_PANEL_HASH,
    frozen_epis_hash: str = FROZEN_EPIS_JUDGE_PANEL_HASH,
    frozen_sus_hash: str = FROZEN_SUS_JUDGE_PANEL_HASH,
) -> int:
    """Return 0 if all assertions pass, 1 if any fail."""
    failures = 0
    total_checked = 0

    for path in contract_paths:
        contract = _load_contract(path)
        if not contract:
            print(f"SKIP  {path} (unreadable)")
            continue

        modules = contract.get("modules") or [{}]
        module = modules[0].get("module") or "unknown"
        run_id = contract.get("run_id") or path.parent.name

        actual = _recompute_judge_panel_hash(contract)

        if module in ("aita",):
            target = frozen_aita_hash
            label = "AITA"
        elif module in ("epistemic",):
            target = frozen_epis_hash
            label = "EPIS"
        elif module in ("sus",):
            target = frozen_sus_hash
            label = "SUS"
        else:
            print(f"SKIP  judge_panel_hash [{run_id}/{module}] (unknown module)")
            continue

        total_checked += 1
        ok = actual == target
        status = "PASS" if ok else "FAIL"
        print(f"{status}  judge_panel_hash [{label}] [{run_id}/{module}]")
        if not ok:
            print(f"      expected: {target}")
            print(f"      actual:   {actual}")
            failures += 1

    print()
    print(f"Checked {total_checked} contract(s).  Failures: {failures}")
    return 0 if failures == 0 else 1


def run_component_wise_audit() -> int:
    """Run component-wise comparison-identity check on all new gpt56-*-20260716 contracts.

    Checks all 10 new gpt56 contracts (sol/terra/luna × aita/epis/sus + none/sus)
    against their frozen references, per component.  Prints GPT-5.6-era
    comparison_spec_hash anchors for each module.
    """
    print("=" * 70)
    print("assert_hash_panel: component-wise comparison-identity audit")
    print("=" * 70)
    print()

    new_contract_dirs = [
        "gpt56-sol-suite-n20-frontier-20260716/aita",
        "gpt56-sol-suite-n20-frontier-20260716/epis",
        "gpt56-sol-suite-n20-frontier-20260716/sus",
        "gpt56-terra-suite-n20-frontier-20260716/aita",
        "gpt56-terra-suite-n20-frontier-20260716/epis",
        "gpt56-terra-suite-n20-frontier-20260716/sus",
        "gpt56-luna-suite-n20-frontier-20260716/aita",
        "gpt56-luna-suite-n20-frontier-20260716/epis",
        "gpt56-luna-suite-n20-frontier-20260716/sus",
        "gpt56-none-sus-n20-frontier-20260716/sus",
    ]
    contract_paths = []
    for subpath in new_contract_dirs:
        p = _RESULTS_ROOT / subpath / "RUN_CONTRACT.json"
        if p.exists():
            contract_paths.append(p)
        else:
            print(f"  WARNING: not found: {p}")

    print()
    return assert_comparison_identity(contract_paths)


def run_full_audit() -> int:
    """Run the canonical audit against all 4 new GPT-5.6 prepared dirs and the
    Fable frontier dirs.  Reports canary-SUS and old-1judge-SUS hashes for the
    record.
    """
    print("=" * 70)
    print("assert_hash_panel: full audit (recomputing from identity)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Report informational hashes
    # -----------------------------------------------------------------------
    canary_path = _RESULTS_ROOT / "gpt56-luna-low-canary-20260715" / "sus" / "RUN_CONTRACT.json"
    canary = _load_contract(canary_path)
    canary_sus_hash = _recompute_judge_panel_hash(canary) if canary else "(missing)"
    print(f"\nCanary SUS recomputed (gpt56-luna-low-canary-20260715/sus):")
    print(f"  {canary_sus_hash}")
    print(f"  frozen target: {FROZEN_SUS_JUDGE_PANEL_HASH}")
    print(f"  match: {canary_sus_hash == FROZEN_SUS_JUDGE_PANEL_HASH}")

    old1j_path = (
        _RESULTS_ROOT
        / "sus-fable-5-native-effort-n20-20260701-142614"
        / "sus"
        / "RUN_CONTRACT.json"
    )
    old1j = _load_contract(old1j_path)
    old1j_hash = _recompute_judge_panel_hash(old1j) if old1j else "(missing)"
    print(f"\nOld 1-judge SUS recomputed (sus-fable-5-native-effort-n20-20260701-142614/sus):")
    print(f"  {old1j_hash}")
    print(f"  NOTE: expected to DIFFER from frontier panel (different panel); not a failure")
    print(f"  reference stored in OLD_1JUDGE_SUS_JUDGE_PANEL_HASH = {OLD_1JUDGE_SUS_JUDGE_PANEL_HASH}")

    # -----------------------------------------------------------------------
    # Collect contract paths for new GPT-5.6 dirs and fable frontier dirs
    # -----------------------------------------------------------------------
    aita_contract_dirs = [
        "fable-5-native-suite-n20-frontier-20260702-142711-frontier/aita",
        "gpt56-sol-suite-n20-frontier-20260716/aita",
        "gpt56-terra-suite-n20-frontier-20260716/aita",
        "gpt56-luna-suite-n20-frontier-20260716/aita",
    ]
    epis_contract_dirs = [
        "fable-5-native-suite-n20-frontier-20260702-142711-frontier/epis",
        "gpt56-sol-suite-n20-frontier-20260716/epis",
        "gpt56-terra-suite-n20-frontier-20260716/epis",
        "gpt56-luna-suite-n20-frontier-20260716/epis",
    ]
    sus_contract_dirs = [
        "gpt56-none-sus-n20-frontier-20260716/sus",
        "gpt56-sol-suite-n20-frontier-20260716/sus",
        "gpt56-terra-suite-n20-frontier-20260716/sus",
        "gpt56-luna-suite-n20-frontier-20260716/sus",
    ]

    contract_paths = []
    for subpath in aita_contract_dirs + epis_contract_dirs + sus_contract_dirs:
        p = _RESULTS_ROOT / subpath / "RUN_CONTRACT.json"
        if p.exists():
            contract_paths.append(p)
        else:
            print(f"  WARNING: not found: {p}")

    print()
    return assert_contracts(contract_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert frozen hash identity of prepared contracts (recomputes from identity)."
    )
    parser.add_argument(
        "contracts",
        nargs="*",
        metavar="CONTRACT_PATH",
        help="Paths to RUN_CONTRACT.json files. Omit to run the full canonical audit.",
    )
    parser.add_argument(
        "--frozen-aita-hash",
        default=FROZEN_AITA_JUDGE_PANEL_HASH,
        help="Expected AITA judge_panel_hash.",
    )
    parser.add_argument(
        "--frozen-epis-hash",
        default=FROZEN_EPIS_JUDGE_PANEL_HASH,
        help="Expected EPIS judge_panel_hash.",
    )
    parser.add_argument(
        "--frozen-sus-hash",
        default=FROZEN_SUS_JUDGE_PANEL_HASH,
        help="Expected SUS judge_panel_hash.",
    )
    parser.add_argument(
        "--component-wise",
        action="store_true",
        help=(
            "Run component-wise comparison-identity check on all new gpt56-*-20260716 "
            "contracts.  Asserts judge_panel_hash, benchmark_spec_hash, "
            "dataset_manifest, dataset_mode, and item_universe individually rather "
            "than the composite comparison_spec_hash."
        ),
    )
    args = parser.parse_args(argv)

    # Projection-bound constants: refuse to audit against constants that predate
    # the active identity projection rather than reporting phantom mismatches.
    assert_constants_blessed_for_current_projection()

    if args.component_wise:
        return run_component_wise_audit()

    if not args.contracts:
        return run_full_audit()

    return assert_contracts(
        [Path(p) for p in args.contracts],
        frozen_aita_hash=args.frozen_aita_hash,
        frozen_epis_hash=args.frozen_epis_hash,
        frozen_sus_hash=args.frozen_sus_hash,
    )


if __name__ == "__main__":
    raise SystemExit(main())
