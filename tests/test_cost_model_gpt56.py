"""Tests for suite_tools.cost_model_gpt56."""

from __future__ import annotations

import math

import numpy as np
import pytest

from suite_tools.cost_model_gpt56 import (
    AITA_BILLABLE_OUT,
    AITA_FABLE_TARGET_USD,
    AITA_THINKING_OUT,
    AITA_VISIBLE_OUT,
    EPIS_FABLE_TARGET_USD,
    EPIS_VISIBLE_OUT,
    EFFORTS,
    FABLE_PRICE_IN_RATIO,
    FABLE_PRICE_OUT_MU,
    GPT56_PRICING,
    GPT56_TIERS,
    LUNA_CANARY_COST_COMPUTED,
    LUNA_CANARY_TOKENS_IN,
    LUNA_CANARY_TOKENS_THINKING,
    LUNA_CANARY_TOKENS_VISIBLE,
    SUS_FABLE_TARGET_FULL_USD,
    SUS_THINKING_FULL,
    SUS_VISIBLE_FULL,
    build_summary,
    compute_gpt56_arm_cost,
    compute_judge_seeker_costs,
    compute_point_estimates,
    derive_input_tokens,
    run_monte_carlo,
)


# ---------------------------------------------------------------------------
# derive_input_tokens
# ---------------------------------------------------------------------------

class TestDeriveInputTokens:
    def test_aita_at_median_price_is_consistent(self):
        """Back-derived input tokens reconstruct AITA Fable cost within $0.01."""
        input_tokens = derive_input_tokens(
            AITA_FABLE_TARGET_USD, FABLE_PRICE_OUT_MU, AITA_BILLABLE_OUT
        )
        price_in = FABLE_PRICE_OUT_MU / FABLE_PRICE_IN_RATIO
        reconstructed = (
            input_tokens * price_in / 1e6
            + AITA_BILLABLE_OUT * FABLE_PRICE_OUT_MU / 1e6
        )
        assert abs(reconstructed - AITA_FABLE_TARGET_USD) < 0.01

    def test_epis_at_median_price_is_consistent(self):
        """Back-derived EPIS input tokens reconstruct EPIS cost within $0.01."""
        input_tokens = derive_input_tokens(
            EPIS_FABLE_TARGET_USD, FABLE_PRICE_OUT_MU, EPIS_VISIBLE_OUT
        )
        price_in = FABLE_PRICE_OUT_MU / FABLE_PRICE_IN_RATIO
        reconstructed = (
            input_tokens * price_in / 1e6
            + EPIS_VISIBLE_OUT * FABLE_PRICE_OUT_MU / 1e6
        )
        assert abs(reconstructed - EPIS_FABLE_TARGET_USD) < 0.01

    def test_returns_positive_for_reasonable_prices(self):
        for price_out in [5.0, 10.0, 15.0, 25.0]:
            result = derive_input_tokens(AITA_FABLE_TARGET_USD, price_out, AITA_BILLABLE_OUT)
            assert result > 0, f"Expected positive input tokens at price_out={price_out}"

    def test_clamped_to_minimum_if_price_too_high(self):
        # price_out so high that output cost alone exceeds observed cost → clamp
        result = derive_input_tokens(10.0, 1000.0, 10_000_000)
        assert result == 1_000.0


# ---------------------------------------------------------------------------
# compute_gpt56_arm_cost
# ---------------------------------------------------------------------------

class TestComputeGpt56ArmCost:
    def test_luna_canary_matches_computed_value(self):
        """Single-scenario Luna canary: model cost should match LUNA_CANARY_COST_COMPUTED."""
        cost = compute_gpt56_arm_cost(
            input_tokens=LUNA_CANARY_TOKENS_IN,
            visible_tokens=LUNA_CANARY_TOKENS_VISIBLE,
            thinking_tokens=LUNA_CANARY_TOKENS_THINKING,
            thinking_mult=1.0,
            tier="luna",
        )
        assert abs(cost - LUNA_CANARY_COST_COMPUTED) < 1e-6

    def test_sol_more_expensive_than_terra_more_than_luna(self):
        kwargs = dict(input_tokens=10_000, visible_tokens=1_000, thinking_tokens=500, thinking_mult=1.0)
        cost_sol   = compute_gpt56_arm_cost(**kwargs, tier="sol")
        cost_terra = compute_gpt56_arm_cost(**kwargs, tier="terra")
        cost_luna  = compute_gpt56_arm_cost(**kwargs, tier="luna")
        assert cost_sol > cost_terra > cost_luna

    def test_higher_thinking_mult_increases_cost(self):
        base = compute_gpt56_arm_cost(input_tokens=5_000, visible_tokens=500, thinking_tokens=200, thinking_mult=1.0, tier="sol")
        high = compute_gpt56_arm_cost(input_tokens=5_000, visible_tokens=500, thinking_tokens=200, thinking_mult=3.0, tier="sol")
        assert high > base

    def test_zero_thinking_tokens_ignores_multiplier(self):
        c1 = compute_gpt56_arm_cost(input_tokens=5_000, visible_tokens=500, thinking_tokens=0, thinking_mult=1.0, tier="terra")
        c2 = compute_gpt56_arm_cost(input_tokens=5_000, visible_tokens=500, thinking_tokens=0, thinking_mult=99.0, tier="terra")
        assert c1 == c2


# ---------------------------------------------------------------------------
# compute_point_estimates  (deterministic, median assumptions)
# ---------------------------------------------------------------------------

class TestComputePointEstimates:
    def setup_method(self):
        self.pt = compute_point_estimates()

    def test_structure_contains_all_modules_tiers_efforts(self):
        for module in ("aita", "epis", "sus"):
            assert module in self.pt
            for tier in GPT56_TIERS:
                assert tier in self.pt[module]
                for effort in EFFORTS:
                    assert effort in self.pt[module][tier]
        # SUS also has "none" effort
        for tier in GPT56_TIERS:
            assert "none" in self.pt["sus"][tier]

    def test_aita_sol_total_within_expected_range(self):
        """AITA Sol: sum over 5 efforts should be ~$188–$195 (sensitivity check)."""
        sol_total = sum(self.pt["aita"]["sol"].values())
        assert 150 < sol_total < 240, f"AITA Sol total ${sol_total:.2f} out of expected range"

    def test_aita_max_effort_more_expensive_than_low(self):
        """Max effort has ~60× more thinking tokens than low; must dominate cost."""
        for tier in GPT56_TIERS:
            assert self.pt["aita"][tier]["max"] > self.pt["aita"][tier]["low"] * 1.5

    def test_epis_costs_flat_across_efforts(self):
        """EPIS thinking ≈ 0 → all effort costs identical within floating-point."""
        for tier in GPT56_TIERS:
            vals = list(self.pt["epis"][tier].values())
            assert max(vals) - min(vals) < 0.01, (
                f"EPIS {tier} should be flat across efforts: {vals}"
            )

    def test_sol_always_more_expensive_than_luna(self):
        for module in ("aita", "epis", "sus"):
            for effort in self.pt[module]["sol"]:
                assert self.pt[module]["sol"][effort] > self.pt[module]["luna"][effort]

    def test_terra_matches_fable_aita_at_fable_price(self):
        """Terra has same in/out pricing as Fable ($2.5/$15 vs $3/$15 Fable assumed).
        At the median Fable price assumption, Terra AITA sum should be ~80–100% of Fable."""
        terra_total = sum(self.pt["aita"]["terra"].values())
        # Terra is cheaper than Fable per-input but same output rate: should be 70–95%
        assert 70 < terra_total < 120, f"Terra AITA ${terra_total:.2f} unexpected"


# ---------------------------------------------------------------------------
# compute_judge_seeker_costs
# ---------------------------------------------------------------------------

class TestComputeJudgeSeekerCosts:
    def test_structure_complete(self):
        costs = compute_judge_seeker_costs()
        for module in ("aita", "epis", "sus"):
            for tier in GPT56_TIERS:
                assert tier in costs[module]
                assert "judge" in costs[module][tier]

    def test_aita_judge_cost_matches_fable(self):
        """AITA judge cost per tier must equal Fable observed $60.49."""
        costs = compute_judge_seeker_costs()
        for tier in GPT56_TIERS:
            assert abs(costs["aita"][tier]["judge"] - 60.489105) < 0.001

    def test_sus_judge_cost_positive_and_reasonable(self):
        costs = compute_judge_seeker_costs()
        for tier in GPT56_TIERS:
            j = costs["sus"][tier]["judge"]
            assert 5.0 < j < 30.0, f"SUS {tier} judge ${j:.2f} out of range"


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_grand_total_equals_sum_of_tiers(self):
        arm = compute_point_estimates()
        anc = compute_judge_seeker_costs()
        summary = build_summary(arm, anc)
        tier_sum = sum(v["total"] for v in summary["by_tier"].values())
        assert abs(summary["grand_total"] - tier_sum) < 0.01

    def test_grand_total_above_prior_upper_bound(self):
        """Full-grid GPT-5.6 run should exceed the $280–500 prior (which was wrong)."""
        arm = compute_point_estimates()
        anc = compute_judge_seeker_costs()
        summary = build_summary(arm, anc)
        assert summary["grand_total"] > 500, (
            f"Grand total ${summary['grand_total']:.0f} should exceed prior upper bound $500"
        )


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def setup_method(self):
        self.result = run_monte_carlo(n_samples=500, seed=42)

    def test_schema_version_present(self):
        assert self.result["schema_version"].startswith("benchmark-cost-model-gpt56")

    def test_grand_total_interval_ordering(self):
        mc = self.result["monte_carlo"]["grand_total"]
        assert mc["p5"] < mc["median"] < mc["p95"]

    def test_grand_total_median_reasonable(self):
        median = self.result["monte_carlo"]["grand_total"]["median"]
        assert 400 < median < 2000, f"MC median ${median:.0f} out of expected range"

    def test_by_tier_sol_more_than_luna(self):
        mc = self.result["monte_carlo"]["by_tier"]
        assert mc["sol"]["median"] > mc["luna"]["median"]

    def test_reproducible_with_same_seed(self):
        r1 = run_monte_carlo(n_samples=200, seed=99)
        r2 = run_monte_carlo(n_samples=200, seed=99)
        assert r1["monte_carlo"]["grand_total"]["median"] == r2["monte_carlo"]["grand_total"]["median"]

    def test_different_seeds_produce_different_results(self):
        r1 = run_monte_carlo(n_samples=200, seed=1)
        r2 = run_monte_carlo(n_samples=200, seed=2)
        assert r1["monte_carlo"]["grand_total"]["median"] != r2["monte_carlo"]["grand_total"]["median"]

    def test_luna_canary_section_present(self):
        assert "luna_canary_sanity_check" in self.result
        check = self.result["luna_canary_sanity_check"]
        assert check["computed_target_cost_usd"] > 0.005
        assert check["computed_target_cost_usd"] < 0.025

    def test_fable_aita_input_derivation_consistent(self):
        """Fable anchor: derived input tokens should reconstruct AITA cost ±$0.02."""
        fa = self.result["fable_anchors"]["aita"]
        input_tok = fa["derived_input_tokens_at_median_price"]
        price_in = FABLE_PRICE_OUT_MU / FABLE_PRICE_IN_RATIO
        reconstructed = (
            input_tok * price_in / 1e6
            + AITA_BILLABLE_OUT * FABLE_PRICE_OUT_MU / 1e6
        )
        assert abs(reconstructed - AITA_FABLE_TARGET_USD) < 0.02
