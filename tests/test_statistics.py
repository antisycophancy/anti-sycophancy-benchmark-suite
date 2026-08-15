import pytest

from suite_tools.statistics import (
    binary_rate_summary,
    bootstrap_ci,
    confidence_interval,
    exact_sign_test_p_value,
    paired_binary_delta_summary,
    wilson_interval,
)


def test_wilson_interval_handles_small_extreme_rates():
    low, high = wilson_interval(0, 5)

    assert low == 0.0
    assert 0.0 < high < 1.0


def test_binary_rate_summary_reports_percent_and_proportion():
    summary = binary_rate_summary(3, 4)

    assert summary["count"] == 3
    assert summary["denominator"] == 4
    assert summary["rate"] == 0.75
    assert summary["rate_percent"] == 75.0
    assert summary["wilson_95_ci_low"] < 0.75 < summary["wilson_95_ci_high"]
    assert summary["wilson_95_ci_low_percent"] < 75.0 < summary["wilson_95_ci_high_percent"]


def test_confidence_interval_and_bootstrap_ci_match_legacy_shapes():
    assert confidence_interval([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([42.0]) == (42.0, 42.0, 42.0)

    mean, low, high = bootstrap_ci([70.0, 75.0, 80.0], seed=42)
    assert mean == 75.0
    assert low <= mean <= high


def test_confidence_interval_uses_t_for_df_n_minus_1():
    from suite_tools.statistics import t_value_95

    # df = N - 1: N=2 must use t(1)=12.706, not t(2)=4.303.
    scores = [10.0, 20.0]
    stdev = 7.0710678118654755
    mean, low, high = confidence_interval(scores)
    assert mean == 15.0
    assert low == round(15.0 - 12.706 * (stdev / 2**0.5), 2)
    assert high == round(15.0 + 12.706 * (stdev / 2**0.5), 2)

    # Former z-fallback gaps (N=12 -> df=11) must use the exact t-value.
    assert t_value_95(11) == 2.201
    # Between sparse rows beyond df=30, use the next smaller df (conservative).
    assert t_value_95(45) == 2.021  # falls back to df=40
    assert t_value_95(200) == 1.96
    assert t_value_95(0) == float("inf")


def test_exact_sign_test_over_discordant_pairs():
    assert exact_sign_test_p_value(0, 0) is None
    assert exact_sign_test_p_value(3, 0) == pytest.approx(0.25)


def test_paired_binary_delta_summary_keeps_unit_at_pair_level():
    summary = paired_binary_delta_summary(
        [True, True, False, False, None],
        [True, False, True, False, True],
    )

    assert summary["n_pairs"] == 4
    assert summary["a_successes"] == 2
    assert summary["b_successes"] == 2
    assert summary["both_success"] == 1
    assert summary["both_failure"] == 1
    assert summary["a_only"] == 1
    assert summary["b_only"] == 1
    assert summary["delta"] == 0.0


def test_paired_binary_delta_summary_rejects_unequal_lengths():
    with pytest.raises(
        ValueError,
        match=r"paired arrays must be equal length \(got 2 vs 1\)",
    ):
        paired_binary_delta_summary([True, False], [True])
