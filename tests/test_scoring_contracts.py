from suite_tools.scoring_contracts import get_scoring_contract


def _walk_keys(node):
    """Yield every mapping key anywhere in a nested dict/list structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_keys(item)


def test_statistic_kind_never_enters_identity_serialization():
    # statistic_kind is rendering metadata: it must NOT appear anywhere in the
    # serialized benchmark spec, or adding it to a dimension would shift
    # benchmark_spec_hash for an otherwise-unchanged instrument. Report code
    # reads it from the registry objects directly instead.
    for suite in ("sus", "aita", "epistemic"):
        contract = get_scoring_contract(suite)
        spec = contract.as_benchmark_spec()
        assert "statistic_kind" not in set(_walk_keys(spec)), suite
        # ...but the field is still available on the in-memory dataclass.
        assert contract.dimension(
            contract.release_score_dimensions[0]
        ).statistic_kind in {"wilson", "bootstrap"}


def test_aita_scoring_contract_separates_primary_alignment_from_diagnostics():
    contract = get_scoring_contract("aita")
    spec = contract.as_benchmark_spec()

    assert contract.score_dimensions[0] == "verdict_alignment_a"
    assert spec["primary_outcome_dimensions"] == [
        "verdict_alignment_a",
        "verdict_alignment_b",
        "paired_verdict_alignment",
    ]
    assert "therapeutic_a" in spec["diagnostic_dimensions"]
    assert contract.dimension("therapeutic_a").scale == "0-3"
    assert contract.dimension("verdict_alignment_a").direction == "higher=better"


def test_epistemic_scoring_contract_names_derived_primary_failure():
    contract = get_scoring_contract("epis")
    spec = contract.as_benchmark_spec()

    assert spec["score_dimensions"] == ["persistence", "integrity", "consistency", "amplification"]
    assert spec["primary_outcome_dimensions"] == ["primary_failure"]
    assert "endpoint_shift" in spec["mechanism_dimensions"]
    assert contract.dimension("amplification").direction == "higher=worse"


def test_sus_scoring_contract_keeps_binary_cap_separate_from_severity():
    contract = get_scoring_contract("sus")
    spec = contract.as_benchmark_spec()

    assert spec["primary_outcome_dimensions"] == ["cap_outcome"]
    assert "sus_response_score" in spec["diagnostic_dimensions"]
    assert contract.dimension("cap_outcome").scale == "cap|no_cap"
    assert contract.dimension("sus_response_score").direction == "higher=worse"

