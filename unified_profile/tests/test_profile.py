from unified_profile.profile import build_all_profiles, build_profile


SUS = {
    "model_id": "m",
    "source_model_key": "source-sus",
    "label": "Model",
    "n_items": 2,
    "sycophancy_score": 60.0,
    "raw": {
        "failure_class_distribution": {"self-coaching": 2},
        "failure_mode_distribution": {"passphrase": 2},
        "classifier_judge_conflicts": 1,
    },
}
AITA = {
    "model_id": "m",
    "source_model_key": "source-aita",
    "label": "Model",
    "n_items": 3,
    "sycophancy_score": 30.0,
    "raw": {"outcome": 1.5},
}
EPIS = {
    "model_id": "m",
    "source_model_key": "source-epis",
    "label": "Model",
    "n_items": 4,
    "sycophancy_score": 0.0,
    "raw": {"persistence": 1},
}


def test_build_profile_all_modules():
    profile = build_profile("m", SUS, AITA, EPIS)

    assert profile.composite_score == 30.0
    assert profile.safety_score == 60.0
    assert profile.moral_score == 30.0
    assert profile.epistemic_score == 0.0


def test_build_profile_missing_module():
    profile = build_profile("m", SUS, AITA, None)

    assert profile.composite_score == 45.0
    assert profile.epistemic_score is None
    assert profile.epis_n == 0


def test_build_profile_only_sus():
    profile = build_profile("m", SUS, None, None)

    assert profile.composite_score == 60.0
    assert profile.moral_score is None
    assert profile.epistemic_score is None


def test_composite_ordering():
    profiles = build_all_profiles(
        {"bad": {**SUS, "model_id": "bad", "sycophancy_score": 80.0}},
        {"good": {**AITA, "model_id": "good", "sycophancy_score": 10.0}},
        {},
    )

    assert [profile.model_id for profile in profiles] == ["bad", "good"]


def test_failure_class_from_sus():
    profile = build_profile("m", SUS, None, EPIS)

    assert profile.failure_class == "self-coaching"
    assert profile.failure_mode_distribution == {"passphrase": 2}
