import json
from pathlib import Path

import pytest

from suite_tools import experiment


def _member_run(root: Path, name: str, unit_ids: list[str], *, started_at: str,
                attempt: int = 1) -> Path:
    d = root / name
    d.mkdir(parents=True)
    units = [{"unit_id": u, "expected_score_path": f"{u}.json"} for u in unit_ids]
    (d / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1", "run_id": name,
        "modules": [{"module": "aita", "expected_units": units}],
        "identity": {"model_conditions": [{"key": "m", "condition_id": "cond-m"}]},
    }))
    (d / "RUN_STATUS.json").write_text(json.dumps({
        "attempt_number": attempt, "started_at": started_at}))
    for u in unit_ids:                                 # mark each done
        (d / f"{u}.json").write_text(json.dumps({"completed": True}))
    return d


def _experiment_with(members: list[tuple[Path, str]], exp_dir: Path) -> Path:
    exp_dir.mkdir(parents=True)
    (exp_dir / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": "exp1", "title": "t",
        "instrument": {"modules": ["aita"], "hashes": {}},
        "conditions": [], "target": {"n_items": 2},
        "members": [{"path": str((d / "RUN_CONTRACT.json").resolve()),
                     "role": role} for d, role in members],
    }))
    return exp_dir


def test_union_picks_latest_started_at_not_manifest_order(tmp_path):
    runs = tmp_path / "runs"
    # pilot listed FIRST but started LATER -> pilot must win the collision
    later = _member_run(runs, "pilot", ["aita:m:item0:a"],
                        started_at="2026-07-20T12:00:00Z")
    earlier = _member_run(runs, "expansion", ["aita:m:item0:a", "aita:m:item1:a"],
                          started_at="2026-07-19T09:00:00Z")
    exp = _experiment_with([(later, "pilot"), (earlier, "expansion")],
                           tmp_path / "exp")
    out = experiment.union(exp)
    dup = next(u for u in out["units"] if u["unit_id"] == "aita:m:item0:a")
    assert dup["chosen_member"].endswith("pilot/RUN_CONTRACT.json")
    assert dup["reason"] == "latest_started_at"
    assert {c["member"].split("/")[-2] for c in dup["candidates"]} == {"pilot", "expansion"}
    # the non-colliding unit resolves to its sole provider
    sole = next(u for u in out["units"] if u["unit_id"] == "aita:m:item1:a")
    assert sole["reason"] == "sole_provider"
    assert len(out["units"]) == 2                      # union of item0 + item1


def test_status_winner_now_matches_union(tmp_path):
    runs = tmp_path / "runs"
    later = _member_run(runs, "pilot", ["aita:m:item0:a"],
                        started_at="2026-07-20T12:00:00Z")
    earlier = _member_run(runs, "expansion", ["aita:m:item0:a"],
                          started_at="2026-07-19T09:00:00Z")
    exp = _experiment_with([(later, "pilot"), (earlier, "expansion")],
                           tmp_path / "exp")
    st = experiment.status(exp)
    coll = next(c for c in st["collisions"] if c["unit_id"] == "aita:m:item0:a")
    assert coll["kept_member"].endswith("pilot/RUN_CONTRACT.json")  # not providers[-1]


def test_attempt_number_does_not_govern_latest_started_at_wins(tmp_path):
    """Spec §5.2: latest started_at wins; attempt_number is irrelevant to ordering.

    Member X has attempt_number=2 but an OLDER started_at than member Y
    (attempt_number=1).  Y must win because spec §5.2 mandates the member
    with the latest started_at, not the highest attempt number.
    """
    runs = tmp_path / "runs"
    # Y: attempt 1, newer started_at → must win
    y = _member_run(runs, "y_run", ["aita:m:item0:a"],
                    started_at="2026-07-20T10:00:00Z", attempt=1)
    # X: attempt 2, older started_at → must lose despite higher attempt number
    x = _member_run(runs, "x_run", ["aita:m:item0:a"],
                    started_at="2026-07-19T08:00:00Z", attempt=2)
    exp = _experiment_with([(y, "pilot"), (x, "expansion")], tmp_path / "exp")
    out = experiment.union(exp)
    dup = next(u for u in out["units"] if u["unit_id"] == "aita:m:item0:a")
    assert dup["chosen_member"].endswith("y_run/RUN_CONTRACT.json"), (
        "Y (attempt 1, newer started_at) must win over X (attempt 2, older started_at) "
        "— spec §5.2: latest started_at governs, attempt_number does not"
    )


def test_mixed_tz_formats_equal_instants_deterministic_tiebreak(tmp_path):
    """Spec §5.2 tz-safe: 'Z' and '+00:00' equal instants → stable manifest-order tiebreak.

    When two members share the same logical instant expressed in different
    timezone formats, _resolve_winner must compare them as equal datetimes
    (not as different strings) and fall back to manifest_index tiebreak.
    """
    runs = tmp_path / "runs"
    # Both represent the same UTC instant in different formats
    first = _member_run(runs, "first", ["aita:m:item0:a"],
                        started_at="2026-07-20T12:00:00Z")
    second = _member_run(runs, "second", ["aita:m:item0:a"],
                         started_at="2026-07-20T12:00:00+00:00")
    exp = _experiment_with([(first, "pilot"), (second, "expansion")], tmp_path / "exp")
    out = experiment.union(exp)
    dup = next(u for u in out["units"] if u["unit_id"] == "aita:m:item0:a")
    # Equal instants → manifest_index tiebreak (lower index = first in manifest wins)
    assert dup["chosen_member"].endswith("first/RUN_CONTRACT.json"), (
        "Equal instants in 'Z' vs '+00:00' formats must resolve deterministically "
        "by manifest order, not string comparison"
    )


def test_union_skips_superseded_members(tmp_path):
    runs = tmp_path / "runs"
    keep = _member_run(runs, "keep", ["aita:m:item0:a"],
                       started_at="2026-07-18T00:00:00Z")
    dead = _member_run(runs, "dead", ["aita:m:item0:a"],
                       started_at="2026-07-21T00:00:00Z")  # later, but superseded
    exp = _experiment_with([(keep, "pilot"), (dead, "expansion")], tmp_path / "exp")
    m = json.loads((exp / "EXPERIMENT.json").read_text())
    m["members"][1]["superseded_by"] = m["members"][0]["path"]
    (exp / "EXPERIMENT.json").write_text(json.dumps(m))
    out = experiment.union(exp)
    dup = next(u for u in out["units"] if u["unit_id"] == "aita:m:item0:a")
    assert dup["chosen_member"].endswith("keep/RUN_CONTRACT.json")
    assert all("dead" not in c["member"] for c in dup["candidates"])
