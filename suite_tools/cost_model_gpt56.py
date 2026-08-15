"""
GPT-5.6 cost prediction model for the full 3-tier × 5-effort × 3-module benchmark run.

Method: token-level cost model anchored on Fable 5 run ledgers, scaled to GPT-5.6
list prices.  Monte Carlo uncertainty quantifies two primary unknowns:

  1. Fable base output price ($/Mtok) — affects the number of input tokens
     back-derived from the observed target-model cost.  Log-normal, median $15/Mtok.

  2. GPT-5.6 thinking-token multiplier relative to Fable per effort level —
     higher effort = larger multiplier uncertainty.  Log-normal centred on 1.0.

Run structure projected:
  • 3 tiers  × 5 efforts × 3 modules (AITA / EPIS / SUS), N=20 per condition
  • SUS-only "none"-effort arm × 3 tiers, N=20
  • 3-judge panel  (OpenRouter, same models as Fable)

Empirical anchors (Fable 5 native-suite run ledgers):
  AITA  target=$109.60  billable_out=1,760k  visible_out=651k  (see cost design doc)
  EPIS  target=$25.71   visible_out~160k      thinking~0
  SUS   target=$10.13   for 65 of 100 planned completions; thinking_out=24,384 (all models)
  Luna canary: 1 scenario, target tokens_in=3,205  out_visible=713  thinking=234

GPT-5.6 pricing (per Mtok in / out):
  Sol $5 / $30    Terra $2.5 / $15    Luna $1 / $6
  Judge costs: same OpenRouter models as Fable → observed per-item costs applied directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "benchmark-cost-model-gpt56-v1"

# ---------------------------------------------------------------------------
# Fable empirical anchors  (read-only ledger data)
# ---------------------------------------------------------------------------

# Target-model (model_under_test) costs from Fable RUN_STATUS ledgers
AITA_FABLE_TARGET_USD: float = 109.59934   # 5 efforts × 40 convs = 200 convs
EPIS_FABLE_TARGET_USD: float = 25.70922    # 5 efforts × 20 items (multi-turn)
SUS_FABLE_TARGET_USD:  float = 10.1319     # 65 of 100 planned completions

# Judge + seeker costs from Fable (applied unchanged to GPT-5.6 run)
AITA_JUDGE_PER_TIER_USD:    float = 60.489105   # 3-judge panel, 100 conditions
AITA_SEEKER_PER_TIER_USD:   float = 1.798031
EPIS_JUDGE_PER_TIER_USD:    float = 8.115508    # 3-judge panel, 100 conditions
EPIS_SEEKER_PER_TIER_USD:   float = 0.2707985
SUS_JUDGE_PER_ITEM_USD:     float = 5.6811 / 64  # $0.0888 per scored item (3-judge)
SUS_ANALYZER_PER_ITEM_USD:  float = 0.2505 / 65  # $0.00385 per completion

# Token volumes (AITA)
# "~1.76M billable vs 651k visible output" — aita-cost-design-decision-2026-07-05.md
AITA_BILLABLE_OUT:  int = 1_760_000   # visible + thinking, all 5 efforts × 200 convs
AITA_VISIBLE_OUT:   int =   651_000   # from doc
AITA_THINKING_OUT:  int = 1_109_000   # derived: 1,760k − 651k

# Per-effort visible output (chars-per-4 text analysis, rescaled to sum=651k)
AITA_VISIBLE_PER_EFFORT: list[int] = [116_333, 128_063, 129_812, 133_101, 143_892]

# Per-effort thinking (fitted exponential r≈2.8, c≈11,700; sum ≈ 1,109k)
AITA_THINKING_PER_EFFORT: list[int] = [11_700, 32_700, 91_500, 256_300, 717_500]

EFFORTS: list[str] = ["low", "medium", "high", "xhigh", "max"]

# Token volumes (EPIS) — essentially flat output, thinking ≈ 0
EPIS_VISIBLE_OUT:  int = 160_000   # ~32k/effort × 5 efforts (flat across efforts)
EPIS_THINKING_OUT: int = 0

# Token volumes (SUS) — partial run, scale to 100 planned completions
SUS_OBSERVED_COMPLETIONS:  int = 65    # events.score_started in ledger
SUS_PLANNED_COMPLETIONS:   int = 100   # 5 efforts × 20 scenarios
SUS_SCALE: float = SUS_PLANNED_COMPLETIONS / SUS_OBSERVED_COMPLETIONS

# Fable thinking_out=24,384 for all models; attribute ~63% to target (cost ratio)
_SUS_TARGET_COST_FRAC: float = SUS_FABLE_TARGET_USD / 16.0635   # ≈ 0.631
SUS_THINKING_FULL: int = round(24_384 * _SUS_TARGET_COST_FRAC * SUS_SCALE)  # ≈ 23,600

# Visible output estimated from token allocation:
#   total_out=368,510 − thinking=24,384 − judge_out≈96k − analyzer_out≈13k ≈ 235,126
#   target fraction (63%) ≈ 148,229 for 65 comps → 100: 228,122 ≈ 228k
SUS_VISIBLE_FULL: int = 228_000   # for 100 completions
SUS_FABLE_TARGET_FULL_USD: float = SUS_FABLE_TARGET_USD * SUS_SCALE  # ≈ $15.59

# ---------------------------------------------------------------------------
# GPT-5.6 pricing
# ---------------------------------------------------------------------------

GPT56_TIERS: list[str] = ["sol", "terra", "luna"]

GPT56_PRICING: dict[str, dict[str, float]] = {
    "sol":   {"in": 5.0,  "out": 30.0},
    "terra": {"in": 2.5,  "out": 15.0},
    "luna":  {"in": 1.0,  "out":  6.0},
}

# Luna canary: real observed data (1 scenario, target model)
LUNA_CANARY_TOKENS_IN:      int = 3_205  # 11,453 total − 8,248 scoring
LUNA_CANARY_TOKENS_VISIBLE: int =   713  # 947 target out − 234 thinking
LUNA_CANARY_TOKENS_THINKING: int =  234
LUNA_CANARY_COST_OBS: float = 0.0   # rounded to 0 in ledger; computed below

# Canary cost at Luna pricing (what the model should charge for 1 scenario)
LUNA_CANARY_COST_COMPUTED: float = (
    LUNA_CANARY_TOKENS_IN * GPT56_PRICING["luna"]["in"] / 1e6
    + (LUNA_CANARY_TOKENS_VISIBLE + LUNA_CANARY_TOKENS_THINKING)
    * GPT56_PRICING["luna"]["out"] / 1e6
)  # ≈ $0.0103

# ---------------------------------------------------------------------------
# Monte Carlo parameters
# ---------------------------------------------------------------------------

MC_N_SAMPLES:   int = 10_000
MC_SEED:        int = 42
MC_PERCENTILES: tuple[float, float] = (5.0, 95.0)

# Fable output price uncertainty: log-normal centred on $15/Mtok, σ=0.3
FABLE_PRICE_OUT_MU:    float = 15.0   # $/Mtok, median
FABLE_PRICE_OUT_SIGMA: float = 0.3    # log-scale σ → ~90% CI $8.4–$26.8
FABLE_PRICE_IN_RATIO:  float = 5.0    # price_out / price_in (Anthropic 1:5 ratio)

# Thinking multiplier σ by effort (log-normal, centred on 1.0)
THINKING_SIGMA: dict[str, float] = {
    "low":    0.30,
    "medium": 0.40,
    "high":   0.50,
    "xhigh":  0.60,
    "max":    0.70,
    "none":   0.40,   # treated as medium for SUS none-effort arm
}


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def derive_input_tokens(
    fable_target_usd: float,
    fable_price_out: float,
    total_output_tokens: int,
) -> float:
    """Back-derive target-model input token count from observed Fable cost.

    Uses:  cost = input_tokens × price_in + output_tokens × price_out
           price_in = price_out / FABLE_PRICE_IN_RATIO
    Returns input tokens (float, may be clipped to a minimum of 0).
    """
    price_in = fable_price_out / FABLE_PRICE_IN_RATIO
    output_cost = total_output_tokens * fable_price_out / 1e6
    input_cost = fable_target_usd - output_cost
    if input_cost <= 0:
        # price_out too high given observed cost; clamp input at token minimum
        return 1_000.0
    return input_cost / price_in * 1e6


def compute_gpt56_arm_cost(
    *,
    input_tokens: float,
    visible_tokens: float,
    thinking_tokens: float,
    thinking_mult: float,
    tier: str,
) -> float:
    """Cost for one arm (one module × one effort × one tier)."""
    p = GPT56_PRICING[tier]
    return (
        input_tokens * p["in"] / 1e6
        + visible_tokens * p["out"] / 1e6
        + thinking_tokens * thinking_mult * p["out"] / 1e6
    )


def compute_point_estimates(
    fable_price_out: float = FABLE_PRICE_OUT_MU,
    thinking_mult: float = 1.0,
) -> dict[str, Any]:
    """Point estimates at median price and thinking multiplier assumptions.

    Returns a nested dict: {module: {tier: {effort: cost_usd}}}.
    """
    results: dict[str, Any] = {}

    # --- AITA ---
    aita_input_total = derive_input_tokens(
        AITA_FABLE_TARGET_USD, fable_price_out, AITA_BILLABLE_OUT
    )
    aita_input_per_effort = aita_input_total / len(EFFORTS)
    results["aita"] = {}
    for tier in GPT56_TIERS:
        results["aita"][tier] = {}
        for i, effort in enumerate(EFFORTS):
            cost = compute_gpt56_arm_cost(
                input_tokens=aita_input_per_effort,
                visible_tokens=AITA_VISIBLE_PER_EFFORT[i],
                thinking_tokens=AITA_THINKING_PER_EFFORT[i],
                thinking_mult=thinking_mult,
                tier=tier,
            )
            results["aita"][tier][effort] = cost

    # --- EPIS ---
    epis_input_total = derive_input_tokens(
        EPIS_FABLE_TARGET_USD, fable_price_out, EPIS_VISIBLE_OUT
    )
    epis_input_per_effort = epis_input_total / len(EFFORTS)
    results["epis"] = {}
    for tier in GPT56_TIERS:
        results["epis"][tier] = {}
        for effort in EFFORTS:
            cost = compute_gpt56_arm_cost(
                input_tokens=epis_input_per_effort,
                visible_tokens=EPIS_VISIBLE_OUT / len(EFFORTS),
                thinking_tokens=0.0,
                thinking_mult=1.0,
                tier=tier,
            )
            results["epis"][tier][effort] = cost

    # --- SUS (5 efforts + none-effort arm) ---
    sus_all_efforts = EFFORTS + ["none"]
    sus_input_total = derive_input_tokens(
        SUS_FABLE_TARGET_FULL_USD, fable_price_out, SUS_VISIBLE_FULL + SUS_THINKING_FULL
    )
    sus_input_per_effort = sus_input_total / len(EFFORTS)
    sus_visible_per_effort = SUS_VISIBLE_FULL / len(EFFORTS)
    sus_thinking_per_effort = SUS_THINKING_FULL / len(EFFORTS)

    results["sus"] = {}
    for tier in GPT56_TIERS:
        results["sus"][tier] = {}
        for effort in sus_all_efforts:
            # "none" effort: same token profile as "medium"
            v = sus_visible_per_effort
            t = sus_thinking_per_effort
            cost = compute_gpt56_arm_cost(
                input_tokens=sus_input_per_effort,
                visible_tokens=v,
                thinking_tokens=t,
                thinking_mult=thinking_mult,
                tier=tier,
            )
            results["sus"][tier][effort] = cost

    return results


def compute_judge_seeker_costs() -> dict[str, Any]:
    """Fixed ancillary costs (judges + seeker/analyzer).

    These use observed Fable per-item rates applied to GPT-5.6 run structure.
    No uncertainty applied (same OpenRouter models, same transcripts structure).
    """
    # SUS: 5 efforts + none-effort = 6 × 20 = 120 items per tier
    sus_items_per_tier = (len(EFFORTS) + 1) * 20
    return {
        "aita": {
            tier: {
                "judge": AITA_JUDGE_PER_TIER_USD,
                "seeker": AITA_SEEKER_PER_TIER_USD,
            }
            for tier in GPT56_TIERS
        },
        "epis": {
            tier: {
                "judge": EPIS_JUDGE_PER_TIER_USD,
                "seeker": EPIS_SEEKER_PER_TIER_USD,
            }
            for tier in GPT56_TIERS
        },
        "sus": {
            tier: {
                "judge": SUS_JUDGE_PER_ITEM_USD * sus_items_per_tier,
                "analyzer": SUS_ANALYZER_PER_ITEM_USD * sus_items_per_tier,
            }
            for tier in GPT56_TIERS
        },
    }


def _arm_total(arm_costs: dict[str, Any]) -> float:
    """Sum all effort-level costs in a single tier-module arm dict."""
    return sum(arm_costs.values())


def build_summary(
    arm_costs: dict[str, Any],
    ancillary: dict[str, Any],
) -> dict[str, Any]:
    """Build per-tier, per-module, and grand-total summary."""
    summary: dict[str, Any] = {"by_tier": {}, "by_module": {}, "grand_total": 0.0}

    for tier in GPT56_TIERS:
        tier_target = sum(_arm_total(arm_costs[m][tier]) for m in ("aita", "epis", "sus"))
        tier_judge = sum(
            ancillary[m][tier].get("judge", 0) + ancillary[m][tier].get("seeker", 0)
            + ancillary[m][tier].get("analyzer", 0)
            for m in ("aita", "epis", "sus")
        )
        summary["by_tier"][tier] = {
            "target_generation": tier_target,
            "judge_seeker": tier_judge,
            "total": tier_target + tier_judge,
        }

    for module in ("aita", "epis", "sus"):
        mod_target = sum(_arm_total(arm_costs[module][tier]) for tier in GPT56_TIERS)
        mod_judge = sum(
            ancillary[module][tier].get("judge", 0)
            + ancillary[module][tier].get("seeker", 0)
            + ancillary[module][tier].get("analyzer", 0)
            for tier in GPT56_TIERS
        )
        summary["by_module"][module] = {
            "target_generation": mod_target,
            "judge_seeker": mod_judge,
            "total": mod_target + mod_judge,
        }

    summary["grand_total"] = sum(v["total"] for v in summary["by_tier"].values())
    return summary


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(
    n_samples: int = MC_N_SAMPLES,
    seed: int = MC_SEED,
) -> dict[str, Any]:
    """Monte Carlo prediction intervals for the full GPT-5.6 run.

    Uncertainty sources:
      1. Fable output price: LogNormal(log(15), 0.3)
         → determines back-derived input token counts for each module
      2. Per-effort thinking multiplier (GPT-5.6 vs Fable): LogNormal(0, sigma_effort)
         → applied to AITA thinking tokens (dominant source of variance for high efforts)
      3. SUS thinking multiplier: LogNormal(0, 0.5) (uniform across efforts)

    Returns dict with point estimates (median), p5, p95, and per-arm breakdowns.
    """
    rng = np.random.default_rng(seed)

    # Sample Fable output price (log-normal)
    log_mu = math.log(FABLE_PRICE_OUT_MU)
    fable_price_out_samples = rng.lognormal(log_mu, FABLE_PRICE_OUT_SIGMA, n_samples)

    # Sample per-effort thinking multipliers for AITA (log-normal, mean=1)
    aita_mult_samples: dict[str, np.ndarray] = {
        effort: rng.lognormal(0.0, THINKING_SIGMA[effort], n_samples)
        for effort in EFFORTS
    }

    # SUS thinking multiplier (uniform across efforts)
    sus_mult_samples = rng.lognormal(0.0, THINKING_SIGMA["medium"], n_samples)

    # Per-sample totals by tier and module
    totals: dict[str, list[float]] = {tier: [] for tier in GPT56_TIERS}
    aita_totals: dict[str, list[float]] = {tier: [] for tier in GPT56_TIERS}
    epis_totals: dict[str, list[float]] = {tier: [] for tier in GPT56_TIERS}
    sus_totals:  dict[str, list[float]] = {tier: [] for tier in GPT56_TIERS}
    grand_totals: list[float] = []

    ancillary = compute_judge_seeker_costs()
    ancillary_sum: dict[str, float] = {
        tier: sum(
            ancillary[m][tier].get("judge", 0)
            + ancillary[m][tier].get("seeker", 0)
            + ancillary[m][tier].get("analyzer", 0)
            for m in ("aita", "epis", "sus")
        )
        for tier in GPT56_TIERS
    }
    ancillary_grand = sum(ancillary_sum.values())

    for s in range(n_samples):
        po = float(fable_price_out_samples[s])

        # AITA
        aita_in_total = derive_input_tokens(AITA_FABLE_TARGET_USD, po, AITA_BILLABLE_OUT)
        aita_in_per_effort = aita_in_total / len(EFFORTS)
        aita_tier_cost: dict[str, float] = {tier: 0.0 for tier in GPT56_TIERS}
        for i, effort in enumerate(EFFORTS):
            mult = float(aita_mult_samples[effort][s])
            for tier in GPT56_TIERS:
                aita_tier_cost[tier] += compute_gpt56_arm_cost(
                    input_tokens=aita_in_per_effort,
                    visible_tokens=AITA_VISIBLE_PER_EFFORT[i],
                    thinking_tokens=AITA_THINKING_PER_EFFORT[i],
                    thinking_mult=mult,
                    tier=tier,
                )

        # EPIS
        epis_in_total = derive_input_tokens(EPIS_FABLE_TARGET_USD, po, EPIS_VISIBLE_OUT)
        epis_in_per_effort = epis_in_total / len(EFFORTS)
        epis_tier_cost: dict[str, float] = {}
        for tier in GPT56_TIERS:
            p = GPT56_PRICING[tier]
            epis_tier_cost[tier] = (
                epis_in_total * p["in"] / 1e6
                + EPIS_VISIBLE_OUT * p["out"] / 1e6
            )

        # SUS (6 efforts: 5 + none)
        sus_out_total = SUS_VISIBLE_FULL + SUS_THINKING_FULL
        sus_in_total = derive_input_tokens(SUS_FABLE_TARGET_FULL_USD, po, sus_out_total)
        sus_in_per_effort = sus_in_total / len(EFFORTS)  # none shares same profile
        n_sus_efforts = len(EFFORTS) + 1  # 6 total
        mult_sus = float(sus_mult_samples[s])
        sus_tier_cost: dict[str, float] = {}
        for tier in GPT56_TIERS:
            p = GPT56_PRICING[tier]
            sus_tier_cost[tier] = n_sus_efforts * (
                sus_in_per_effort * p["in"] / 1e6
                + (SUS_VISIBLE_FULL / len(EFFORTS)) * p["out"] / 1e6
                + (SUS_THINKING_FULL / len(EFFORTS)) * mult_sus * p["out"] / 1e6
            )

        # Accumulate
        sample_grand = ancillary_grand
        for tier in GPT56_TIERS:
            target = aita_tier_cost[tier] + epis_tier_cost[tier] + sus_tier_cost[tier]
            total = target + ancillary_sum[tier]
            totals[tier].append(total)
            aita_totals[tier].append(aita_tier_cost[tier])
            epis_totals[tier].append(epis_tier_cost[tier])
            sus_totals[tier].append(sus_tier_cost[tier])
            sample_grand += target
        grand_totals.append(sample_grand)

    def _stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        p5, p95 = np.percentile(a, list(MC_PERCENTILES))
        return {"median": float(np.median(a)), "p5": float(p5), "p95": float(p95)}

    # Point estimates at median price
    point = compute_point_estimates()
    ancillary_costs = compute_judge_seeker_costs()
    point_summary = build_summary(point, ancillary_costs)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "description": (
                "Token-level cost model from Fable 5 run ledgers scaled by GPT-5.6 "
                "list prices. Monte Carlo over (1) Fable output price uncertainty and "
                "(2) per-effort thinking-token multiplier uncertainty."
            ),
            "n_samples": n_samples,
            "seed": seed,
            "interval": "p5–p95",
            "fable_price_out_mu": FABLE_PRICE_OUT_MU,
            "fable_price_out_sigma_lognormal": FABLE_PRICE_OUT_SIGMA,
            "thinking_sigma_by_effort": THINKING_SIGMA,
        },
        "run_structure": {
            "tiers": GPT56_TIERS,
            "efforts_per_module": EFFORTS,
            "sus_extra_arm": "none (modelled as medium effort)",
            "n_per_condition": 20,
            "judge_panel_size": 3,
        },
        "point_estimates": {
            "by_tier": {
                tier: {
                    "target_generation": point_summary["by_tier"][tier]["target_generation"],
                    "judge_seeker": point_summary["by_tier"][tier]["judge_seeker"],
                    "total": point_summary["by_tier"][tier]["total"],
                }
                for tier in GPT56_TIERS
            },
            "by_module": {
                module: {
                    "target_generation": point_summary["by_module"][module]["target_generation"],
                    "judge_seeker": point_summary["by_module"][module]["judge_seeker"],
                    "total": point_summary["by_module"][module]["total"],
                }
                for module in ("aita", "epis", "sus")
            },
            "grand_total": point_summary["grand_total"],
        },
        "monte_carlo": {
            "grand_total": _stats(grand_totals),
            "by_tier": {
                tier: _stats(totals[tier]) for tier in GPT56_TIERS
            },
            "by_module_target_only": {
                "aita": _stats([sum(aita_totals[t][i] for t in GPT56_TIERS) for i in range(n_samples)]),
                "epis": _stats([sum(epis_totals[t][i] for t in GPT56_TIERS) for i in range(n_samples)]),
                "sus":  _stats([sum(sus_totals[t][i]  for t in GPT56_TIERS) for i in range(n_samples)]),
            },
        },
        "prior_estimate": {
            "source": "prereg-gpt56-family-edition-2026-07-15-DRAFT.md §7 placeholder",
            "grand_total_range": "$280–$500",
            "note": "Price-ratio anchor only; did not account for high-effort AITA thinking costs",
        },
        "luna_canary_sanity_check": {
            "observed_total_usd": 0.08,
            "observed_target_cost_usd": 0.0,  # rounds to 0 in ledger
            "computed_target_cost_usd": round(LUNA_CANARY_COST_COMPUTED, 5),
            "model_projection_per_20_scenarios": round(
                compute_point_estimates()["sus"]["luna"]["medium"], 2
            ),
            "note": (
                "Canary = single minimal-turn scenario; production SUS has ~11× larger "
                "input context (system prompt + scenario setup). Canary validates Luna "
                "pricing but not token volumes. Model projects from Fable production "
                "token volumes, not canary."
            ),
        },
        "fable_anchors": {
            "aita": {
                "fable_target_usd": AITA_FABLE_TARGET_USD,
                "billable_out_tokens": AITA_BILLABLE_OUT,
                "visible_out_tokens": AITA_VISIBLE_OUT,
                "thinking_out_tokens": AITA_THINKING_OUT,
                "derived_input_tokens_at_median_price": round(
                    derive_input_tokens(AITA_FABLE_TARGET_USD, FABLE_PRICE_OUT_MU, AITA_BILLABLE_OUT)
                ),
            },
            "epis": {
                "fable_target_usd": EPIS_FABLE_TARGET_USD,
                "visible_out_tokens": EPIS_VISIBLE_OUT,
                "derived_input_tokens_at_median_price": round(
                    derive_input_tokens(EPIS_FABLE_TARGET_USD, FABLE_PRICE_OUT_MU, EPIS_VISIBLE_OUT)
                ),
            },
            "sus": {
                "fable_target_usd": SUS_FABLE_TARGET_USD,
                "observed_completions": SUS_OBSERVED_COMPLETIONS,
                "planned_completions": SUS_PLANNED_COMPLETIONS,
                "fable_target_full_usd": round(SUS_FABLE_TARGET_FULL_USD, 4),
                "thinking_full_tokens": SUS_THINKING_FULL,
                "visible_full_tokens": SUS_VISIBLE_FULL,
                "derived_input_tokens_at_median_price": round(
                    derive_input_tokens(
                        SUS_FABLE_TARGET_FULL_USD,
                        FABLE_PRICE_OUT_MU,
                        SUS_VISIBLE_FULL + SUS_THINKING_FULL,
                    )
                ),
            },
        },
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GPT-5.6 cost prediction model from Fable ledger anchors."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this path (default: stdout).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=MC_N_SAMPLES,
        help=f"Monte Carlo samples (default {MC_N_SAMPLES}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=MC_SEED,
        help=f"RNG seed (default {MC_SEED}).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_monte_carlo(n_samples=args.n_samples, seed=args.seed)
    payload = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(payload)
        print(f"Written to {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
