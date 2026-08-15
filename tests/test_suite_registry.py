from suite_tools.suite_registry import module_key_for_record, suite_root


def test_suite_registry_owns_current_suite_roots():
    assert suite_root("sus").name == "sus-bench"
    assert suite_root("aita").name == "aita-bench"
    assert suite_root("epis").name == "epistemic-sycophancy-bench"


def test_module_key_for_record_uses_suite_markers_and_epistemic_types():
    assert module_key_for_record({}, "sus-bench/results/run.json") == "sus"
    assert module_key_for_record({}, "aita-bench/results/run.json") == "aita"
    assert module_key_for_record({"test_type": "mirror"}, "scores.json") == "epistemic"
    assert module_key_for_record({"module": "sus"}, "x.json") == "sus"
