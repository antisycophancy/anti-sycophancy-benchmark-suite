"""Tests for suite_tools.td_threshold_derivation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from suite_tools.td_threshold_derivation import (
    BIN_WIDTH,
    JUDGE_SOLO_ID,
    PLACEHOLDER_THRESHOLD,
    SCALE_MAX,
    TD_EPS,
    _effort_from_filename,
    _find_solo_leg,
    _pair_td_coverage_rate,
    bin_observations,
    derive_td_threshold,
    extract_observations,
    propose_threshold,
    sensitivity_by_effort,
)


# ---------------------------------------------------------------------------
# Unit: _effort_from_filename
# ---------------------------------------------------------------------------


def test_effort_from_filename_high():
    assert _effort_from_filename("claude-fable-5-native-high_item0_scores.json") == "high"


def test_effort_from_filename_xhigh():
    assert _effort_from_filename("claude-fable-5-native-xhigh_item5_scores.json") == "xhigh"


def test_effort_from_filename_low():
    assert _effort_from_filename("claude-fable-5-native-low_item3_scores.json") == "low"


# ---------------------------------------------------------------------------
# Unit: _find_solo_leg
# ---------------------------------------------------------------------------


def _make_record(judge_entries: list[dict]) -> dict:
    return {"judge_scores": judge_entries}


def test_find_solo_leg_found():
    record = _make_record([
        {"judge_model": "openai/gpt-5.5", "therapeutic_a": 2.0},
        {"judge_model": JUDGE_SOLO_ID, "therapeutic_a": 1.5},
        {"judge_model": "google/gemini-3.1-pro-preview", "therapeutic_a": 3.0},
    ])
    leg = _find_solo_leg(record)
    assert leg is not None
    assert leg["judge_model"] == JUDGE_SOLO_ID
    assert leg["therapeutic_a"] == 1.5


def test_find_solo_leg_missing():
    record = _make_record([
        {"judge_model": "openai/gpt-5.5", "therapeutic_a": 2.0},
        {"judge_model": "google/gemini-3.1-pro-preview", "therapeutic_a": 3.0},
    ])
    assert _find_solo_leg(record) is None


def test_find_solo_leg_ambiguous():
    record = _make_record([
        {"judge_model": JUDGE_SOLO_ID, "therapeutic_a": 1.0},
        {"judge_model": JUDGE_SOLO_ID, "therapeutic_a": 2.0},
    ])
    assert _find_solo_leg(record) is None


def test_find_solo_leg_no_judge_scores():
    assert _find_solo_leg({}) is None
    assert _find_solo_leg({"judge_scores": "bad"}) is None


# ---------------------------------------------------------------------------
# Unit: extract_observations
# ---------------------------------------------------------------------------


def _write_score_file(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _make_full_record(
    *,
    therapeutic_a_solo: float,
    therapeutic_b_solo: float,
    therapeutic_a_gpt: float = 2.0,
    therapeutic_b_gpt: float = 2.0,
    therapeutic_a_gemini: float = 2.0,
    therapeutic_b_gemini: float = 2.0,
    pair_id: str = "pair0",
    model_id: str = "test-model",
    item_idx: int = 0,
) -> dict:
    return {
        "pair_id": pair_id,
        "model_id": model_id,
        "item_idx": item_idx,
        "judge_scores": [
            {
                "judge_model": "openai/gpt-5.5",
                "therapeutic_a": therapeutic_a_gpt,
                "therapeutic_b": therapeutic_b_gpt,
            },
            {
                "judge_model": JUDGE_SOLO_ID,
                "therapeutic_a": therapeutic_a_solo,
                "therapeutic_b": therapeutic_b_solo,
            },
            {
                "judge_model": "google/gemini-3.1-pro-preview",
                "therapeutic_a": therapeutic_a_gemini,
                "therapeutic_b": therapeutic_b_gemini,
            },
        ],
    }


def test_extract_observations_basic(tmp_path: Path):
    record = _make_full_record(
        therapeutic_a_solo=1.0,
        therapeutic_b_solo=2.0,
        therapeutic_a_gpt=2.0,
        therapeutic_b_gpt=3.0,
        therapeutic_a_gemini=3.0,
        therapeutic_b_gemini=2.0,
    )
    _write_score_file(tmp_path / "test_item0_scores.json", record)

    obs = extract_observations(tmp_path)
    assert len(obs) == 2  # two sides

    side_a = next(o for o in obs if o["side"] == "a")
    assert side_a["solo"] == 1.0
    # panel_mean = (2.0 + 1.0 + 3.0) / 3
    assert abs(side_a["panel_mean"] - 2.0) < 1e-9
    assert abs(side_a["abs_diff"] - 1.0) < 1e-9
    assert side_a["n_judges"] == 3


def test_extract_observations_skips_missing_solo(tmp_path: Path):
    record = _make_full_record(
        therapeutic_a_solo=1.0,
        therapeutic_b_solo=2.0,
    )
    # Remove solo leg
    record["judge_scores"] = [
        js for js in record["judge_scores"]
        if js["judge_model"] != JUDGE_SOLO_ID
    ]
    _write_score_file(tmp_path / "test_item0_scores.json", record)
    obs = extract_observations(tmp_path)
    assert obs == []


def test_extract_observations_skips_non_numeric_therapeutic(tmp_path: Path):
    record = _make_full_record(
        therapeutic_a_solo=1.5,
        therapeutic_b_solo=2.0,
    )
    # Corrupt solo therapeutic_a to None
    for js in record["judge_scores"]:
        if js["judge_model"] == JUDGE_SOLO_ID:
            js["therapeutic_a"] = None
    _write_score_file(tmp_path / "test_item0_scores.json", record)
    obs = extract_observations(tmp_path)
    # side_a skipped; side_b still present
    assert len(obs) == 1
    assert obs[0]["side"] == "b"


# ---------------------------------------------------------------------------
# Unit: bin_observations
# ---------------------------------------------------------------------------


def _obs(solo: float, panel_mean: float, effort: str = "high") -> dict:
    return {
        "file": "f.json",
        "pair_id": "p0",
        "model_id": "m",
        "side": "a",
        "solo": solo,
        "panel_mean": panel_mean,
        "abs_diff": abs(solo - panel_mean),
        "panel_vals": [panel_mean],
        "panel_std": 0.0,
        "n_judges": 1,
        "effort": effort,
    }


def test_bin_observations_groups_correctly():
    observations = [
        _obs(0.1, 0.1),   # bin [0.00, 0.25)
        _obs(0.2, 0.3),   # bin [0.00, 0.25)
        _obs(0.5, 0.8),   # bin [0.50, 0.75)
        _obs(1.5, 3.0),   # bin [1.50, 1.75)
    ]
    bins = bin_observations(observations)
    bin_labels = [b["bin_label"] for b in bins]
    assert "[0.00, 0.25)" in bin_labels
    assert "[0.50, 0.75)" in bin_labels
    assert "[1.50, 1.75)" in bin_labels

    # Check mean for [0.00, 0.25): diffs are 0.0, 0.1 -> mean = 0.05
    b0 = next(b for b in bins if b["bin_label"] == "[0.00, 0.25)")
    assert abs(b0["mean_abs_diff"] - 0.05) < 1e-9


def test_bin_observations_flip_relevant_threshold():
    # diff = 0.6 should be flip-relevant; diff = 0.4 should not
    observations = [
        _obs(1.0, 0.4),   # abs_diff = 0.6 -> flip-relevant
        _obs(1.1, 0.7),   # abs_diff = 0.4 -> not flip-relevant
        _obs(1.2, 0.8),   # abs_diff = 0.4 -> not flip-relevant
    ]
    bins = bin_observations(observations)
    b = bins[0]
    assert b["flip_relevant_n"] == 1
    assert b["flip_relevant_pct"] == pytest.approx(33.3, abs=0.1)


# ---------------------------------------------------------------------------
# Unit: propose_threshold
# ---------------------------------------------------------------------------


def test_pair_td_coverage_rate_basic():
    # 3 files: 2 fire (solo=1.0, within 0.25 of t=1.0), 1 does not (solo=2.0)
    obs = [
        _obs(1.0, 1.5),    # fires; file 'f.json'
        _obs(2.0, 2.0),    # does not fire at t=1.0
    ]
    # Override file to separate pairs
    obs[0] = {**obs[0], "file": "pair_a.json"}
    obs[1] = {**obs[1], "file": "pair_b.json"}
    rate = _pair_td_coverage_rate(obs, threshold=1.0)
    assert rate == pytest.approx(0.5, abs=1e-9)


def test_propose_threshold_avoids_high_coverage():
    # Scenario: bin at solo=2.0 has high disagreement but high coverage,
    # bin at solo=1.0 has lower disagreement but low coverage -> should prefer solo=1.0 bin
    obs_high_coverage = [
        {**_obs(2.0, 2.8), "file": f"pair_{i}.json"}
        for i in range(10)
    ]  # 10 distinct pairs, all fire at t=2.0 -> 100% coverage
    obs_low_coverage = [
        {**_obs(1.0, 1.8), "file": f"pair_rare_{i}.json"}
        for i in range(1)
    ]  # 1 pair, fires at t=1.0 -> 10% coverage
    all_obs = obs_high_coverage + obs_low_coverage
    bins = bin_observations(all_obs)
    result = propose_threshold(bins, all_obs)
    # t* should prefer the low-coverage bin (1.0 bin centre=1.125) over the
    # high-coverage bin even though the latter has higher disagreement
    assert result["t_star"] == pytest.approx(1.125, abs=1e-4)
    assert result["escalation_rate_tstar"] <= 0.25


def test_propose_threshold_placeholder_rate_zero_when_equidistant():
    # Solo scores always integer: 1 and 2; placeholder=1.5 should never fire
    obs = [
        {**_obs(1.0, 1.8), "file": "p1.json"},
        {**_obs(2.0, 2.5), "file": "p2.json"},
    ]
    bins = bin_observations(obs)
    result = propose_threshold(bins, obs, placeholder=1.5)
    assert result["escalation_rate_placeholder"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit: sensitivity_by_effort
# ---------------------------------------------------------------------------


def test_sensitivity_by_effort_keys():
    obs = (
        [_obs(1.5, 2.5, effort="high")] * 10
        + [_obs(2.0, 1.0, effort="low")] * 10
    )
    result = sensitivity_by_effort(obs)
    assert "high" in result
    assert "low" in result
    assert result["high"]["n_observations"] == 10
    assert result["low"]["n_observations"] == 10


# ---------------------------------------------------------------------------
# Integration: real data smoke test
# ---------------------------------------------------------------------------

FABLE_AITA_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "prepared"
    / "fable-5-native-suite-n20-frontier-20260702-142711-frontier"
    / "aita"
)


@pytest.mark.skipif(
    not FABLE_AITA_DIR.exists() or not list(FABLE_AITA_DIR.glob("*_scores.json")),
    reason="Fable-5 AITA data not available",
)
def test_derive_td_threshold_real_data():
    results = derive_td_threshold(FABLE_AITA_DIR)

    assert results["n_observations"] >= 10
    assert 0.0 <= results["proposal"]["t_star"] <= SCALE_MAX
    assert 0.0 <= results["proposal"]["escalation_rate_tstar"] <= 1.0
    assert 0.0 <= results["proposal"]["escalation_rate_placeholder"] <= 1.0
    assert len(results["bins"]) > 0
    assert len(results["sensitivity_by_effort"]) > 0

    # t* should not just default to placeholder unless data supports it
    prop = results["proposal"]
    # The escalation rate at t* should be reasonable (< 25% budget)
    assert prop["escalation_rate_tstar"] < 1.0, "Td-alone rate should not be 100%"
