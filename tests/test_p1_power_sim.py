"""Tests for the P1 paired max-contrast power simulation (prereq A freeze gate).

Written test-first. The simulation estimates statistical power for the prereg's
P1a (SUS unsafe-delivery sensitivity, detection gate) and P1b (AITA paired
accuracy / EPIS primary-failure, equivalence gate) sub-predictions using the
item-level correlation structure observed in the Fable-5 native effort sweep.
"""

from __future__ import annotations

import math

import pytest

from suite_tools.p1_power_sim import (
    EFFORTS,
    column_rates,
    joint_power_at_least_k_of_n,
    load_fable_matrices,
    max_contrast_simultaneous_bound,
    pairwise_contrasts,
    simulate_power,
)

FABLE_SUITE_DIR = (
    "results/prepared/fable-5-native-suite-n20-frontier-20260702-142711-frontier"
)
FABLE_SUS_DIR = "results/prepared/sus-fable-5-native-effort-n20-20260701-142614/sus"


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_column_rates_computes_per_effort_means():
    rows = [[1, 0], [1, 0], [1, 1], [0, 1]]
    assert column_rates(rows) == [0.75, 0.5]


def test_pairwise_contrasts_covers_all_effort_pairs():
    rates = [0.7, 0.8, 0.85, 0.6, 0.25]
    contrasts = pairwise_contrasts(rates)
    # 5 efforts -> C(5,2) = 10 pairs
    assert len(contrasts) == 10
    # every entry: ((i, j), signed_value)
    for (i, j), value in contrasts:
        assert 0 <= i < j < 5
        assert value == pytest.approx(rates[i] - rates[j])
    # the largest absolute contrast is high(0.85) vs max(0.25) = 0.60
    top = max(contrasts, key=lambda c: abs(c[1]))
    assert abs(top[1]) == pytest.approx(0.60)


def test_max_contrast_bound_detects_large_separation():
    # column 0 always unsafe, column 1 never -> deterministic 100pp contrast.
    rows = [[1, 0] for _ in range(20)]
    result = max_contrast_simultaneous_bound(rows, n_boot=200, seed=1)
    assert result["point"] == pytest.approx(1.0)
    # deterministic separation -> band collapses onto the point estimate.
    assert result["lower_supt"] == pytest.approx(1.0)
    assert result["upper_supt"] == pytest.approx(1.0)


def test_max_contrast_bound_flat_data_has_zero_contrast():
    # identical columns -> zero contrast, tight equivalence bound.
    rows = [[1, 1] for _ in range(10)] + [[0, 0] for _ in range(10)]
    result = max_contrast_simultaneous_bound(rows, n_boot=200, seed=1)
    assert result["point"] == pytest.approx(0.0)
    assert result["upper_supt"] == pytest.approx(0.0)


def test_max_contrast_bound_is_reproducible():
    rows = [[1, 0, 1, 0, 1] for _ in range(10)] + [[0, 1, 0, 1, 0] for _ in range(10)]
    a = max_contrast_simultaneous_bound(rows, n_boot=300, seed=7)
    b = max_contrast_simultaneous_bound(rows, n_boot=300, seed=7)
    assert a == b


# --------------------------------------------------------------------------- #
# Power simulation
# --------------------------------------------------------------------------- #
def test_simulate_power_detection_strong_effect_is_high():
    # A population with a 100pp true separation must be detected at 15pp floor.
    population = [[1, 1, 1, 0, 0] for _ in range(10)] + [
        [1, 1, 1, 0, 0] for _ in range(10)
    ]
    out = simulate_power(
        population,
        gate="detect",
        margin=0.15,
        n_sims=60,
        n_boot=200,
        seed=3,
    )
    assert out["power"] >= 0.95


def test_simulate_power_equivalence_flat_effect_is_high():
    # A perfectly flat population must satisfy the <=25pp equivalence bound.
    population = [[1, 1, 1, 1, 1] for _ in range(15)] + [
        [0, 0, 0, 0, 0] for _ in range(5)
    ]
    out = simulate_power(
        population,
        gate="equivalence",
        margin=0.25,
        n_sims=60,
        n_boot=200,
        seed=3,
    )
    assert out["power"] >= 0.95


def test_simulate_power_detection_flat_effect_is_low():
    # No true effect -> detection gate (>=15pp) should almost never fire.
    population = [[1, 1, 1, 1, 1] for _ in range(14)] + [
        [0, 0, 0, 0, 0] for _ in range(6)
    ]
    out = simulate_power(
        population,
        gate="detect",
        margin=0.15,
        n_sims=60,
        n_boot=200,
        seed=5,
    )
    assert out["power"] <= 0.20


def test_simulate_power_is_reproducible():
    population = [[1, 0, 1, 0, 0] for _ in range(12)] + [
        [0, 1, 0, 1, 1] for _ in range(8)
    ]
    kwargs = dict(gate="detect", margin=0.15, n_sims=40, n_boot=150, seed=11)
    assert simulate_power(population, **kwargs) == simulate_power(population, **kwargs)


def test_joint_power_at_least_two_of_three():
    # Analytic binomial helper for the P1a ">=2 of 3 tiers" aggregate.
    assert joint_power_at_least_k_of_n(1.0, 2, 3) == pytest.approx(1.0)
    assert joint_power_at_least_k_of_n(0.0, 2, 3) == pytest.approx(0.0)
    # p=0.8 -> p^3 + 3 p^2 (1-p) = 0.512 + 0.384 = 0.896
    assert joint_power_at_least_k_of_n(0.8, 2, 3) == pytest.approx(0.896)
    # all-of-3 (k=n): p^3
    assert joint_power_at_least_k_of_n(0.9, 3, 3) == pytest.approx(0.729)


# --------------------------------------------------------------------------- #
# Real Fable data loading (integration; skips if data absent)
# --------------------------------------------------------------------------- #
def _fable_available() -> bool:
    import os

    return os.path.isdir(FABLE_SUITE_DIR) and os.path.isdir(FABLE_SUS_DIR)


@pytest.mark.skipif(not _fable_available(), reason="Fable prepared data not present")
def test_load_fable_matrices_recovers_prereg_anchor_marginals():
    matrices = load_fable_matrices(FABLE_SUITE_DIR, FABLE_SUS_DIR)

    # SUS unsafe-delivery ITT rate must match the prereg anchor 70/80/85/60/25.
    sus = matrices["sus"]
    assert len(sus["rows"]) == 20
    assert sus["efforts"] == list(EFFORTS)
    sus_rates = column_rates(sus["rows"])
    assert [round(r, 2) for r in sus_rates] == [0.70, 0.80, 0.85, 0.60, 0.25]

    # AITA paired accuracy near ceiling (anchor ~95/95/90/95/95).
    aita = matrices["aita"]
    assert len(aita["rows"]) == 20
    aita_rates = column_rates(aita["rows"])
    assert min(aita_rates) >= 0.85
    assert max(aita_rates) <= 1.0

    # EPIS primary-failure near floor.
    epis = matrices["epis"]
    assert len(epis["rows"]) >= 10
    epis_rates = column_rates(epis["rows"])
    assert max(epis_rates) <= 0.15


@pytest.mark.skipif(not _fable_available(), reason="Fable prepared data not present")
def test_sus_max_contrast_point_is_sixty_pp():
    matrices = load_fable_matrices(FABLE_SUITE_DIR, FABLE_SUS_DIR)
    bound = max_contrast_simultaneous_bound(
        matrices["sus"]["rows"], n_boot=500, seed=42
    )
    assert bound["point"] == pytest.approx(0.60, abs=1e-9)
    assert not math.isnan(bound["lower_supt"])
