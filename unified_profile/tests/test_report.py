from pathlib import Path

from unified_profile.profile import build_all_profiles
from unified_profile.report import generate_unified_report


def _profiles():
    sus = {
        "m1": {
            "model_id": "m1",
            "source_model_key": "m1",
            "label": "Model One",
            "n_items": 2,
            "sycophancy_score": 50.0,
            "raw": {
                "sd": 5.0,
                "grade_distribution": {"D": 2},
                "failure_class_distribution": {"no-safety": 2},
                "failure_mode_distribution": {"immediate": 2},
                "classifier_judge_conflicts": 0,
            },
        }
    }
    aita = {
        "m1": {
            "model_id": "m1",
            "source_model_key": "m1",
            "label": "Model One",
            "n_items": 3,
            "sycophancy_score": 10.0,
            "raw": {
                "paired_verdict_alignment": 1.0,
                "verdict_alignment": 1.0,
                "outcome": 2.0,
                "resistance": 1.0,
                "therapeutic": 3.0,
                "consistency": 1.0,
            },
        },
        "m2": {
            "model_id": "m2",
            "source_model_key": "m2",
            "label": "Model Two",
            "n_items": 1,
            "sycophancy_score": 80.0,
            "raw": {"outcome": 0.0},
        },
    }
    return build_all_profiles(sus, aita, {})


def test_report_contains_all_models(tmp_path):
    report = generate_unified_report(_profiles(), tmp_path)

    assert "Model One" in report
    assert "Model Two" in report


def test_report_score_direction_legend(tmp_path):
    report = generate_unified_report(_profiles(), tmp_path)

    assert "higher = more sycophancy detected (worse)" in report


def test_report_missing_module_shows_no_data(tmp_path):
    report = generate_unified_report(_profiles(), tmp_path)

    assert "[no data]" in report
    assert "missing Safety (SUS)" in report


def test_report_output_file(tmp_path):
    report = generate_unified_report(_profiles(), tmp_path)

    assert (tmp_path / "REPORT.md").read_text() == report
