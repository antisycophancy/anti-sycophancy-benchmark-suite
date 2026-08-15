"""Shared statistical helpers for benchmark reports.

The benchmark packages run in lightweight environments, so these helpers stay
stdlib-only. They cover the common release needs: Wilson intervals for binary
outcomes, bootstrap intervals for bounded judge scores, and paired binary
summaries for matched item comparisons.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from math import comb
from typing import Any


# Two-tailed 95% CI critical t-values keyed by degrees of freedom (df = N - 1).
# Complete for df 1-30; sparse beyond. Lookups between entries use the next
# smaller df (wider interval, conservative).
T_VALUES_BY_DF: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t_value_95(df: int) -> float:
    """Return the two-tailed 95% critical t-value for the given df.

    Exact for df 1-30 and the standard 40/60/120 rows; between table rows the
    next smaller df is used so intervals never narrow below the exact value.
    """
    if df <= 0:
        return float("inf")
    if df in T_VALUES_BY_DF:
        return T_VALUES_BY_DF[df]
    floor_df = max((d for d in T_VALUES_BY_DF if d <= df), default=None)
    if floor_df is None:
        return T_VALUES_BY_DF[1]
    if df > 120:
        return 1.96
    return T_VALUES_BY_DF[floor_df]


def wilson_interval(
    successes: int,
    n: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return Wilson score interval bounds for a binary proportion."""
    if n <= 0:
        return (0.0, 0.0)
    successes = max(0, min(int(successes), int(n)))
    p_hat = successes / n
    denom = 1.0 + (z * z / n)
    center = (p_hat + z * z / (2 * n)) / denom
    half_width = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (round(max(0.0, center - half_width), 6), round(min(1.0, center + half_width), 6))


def binary_rate_summary(successes: int, n: int) -> dict[str, Any]:
    """Return count/rate/Wilson metadata for a binary outcome."""
    n = max(0, int(n))
    successes = max(0, min(int(successes), n))
    low, high = wilson_interval(successes, n)
    rate = successes / n if n else 0.0
    return {
        "count": successes,
        "denominator": n,
        "rate": round(rate, 6),
        "rate_percent": round(rate * 100, 1) if n else None,
        "wilson_95_ci_low": low,
        "wilson_95_ci_high": high,
        "wilson_95_ci_low_percent": round(low * 100, 1) if n else None,
        "wilson_95_ci_high_percent": round(high * 100, 1) if n else None,
    }


def confidence_interval(scores: list[float]) -> tuple[float, float, float]:
    """Compute mean and 95% t-style confidence interval for numeric scores."""
    n = len(scores)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (scores[0], scores[0], scores[0])

    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    if stdev == 0:
        return (mean, mean, mean)

    t = t_value_95(n - 1)
    margin = t * (stdev / math.sqrt(n))
    return (round(mean, 2), round(mean - margin, 2), round(mean + margin, 2))


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
    *,
    statistic_fn: Callable[[Sequence[float]], float] | None = None,
) -> tuple[float, float, float]:
    """Compute a percentile bootstrap interval for a numeric statistic."""
    n = len(scores)
    if statistic_fn is None:
        statistic_fn = statistics.mean
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = statistic_fn(scores)
    if n == 1:
        return (point, point, point)

    if seed is None:
        seed = 42
    rng = random.Random(seed)

    bootstrap_values = []
    for _ in range(n_bootstrap):
        sample = rng.choices(scores, k=n)
        bootstrap_values.append(statistic_fn(sample))

    bootstrap_values.sort()
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1
    return (
        round(point, 2),
        round(bootstrap_values[lo_idx], 2),
        round(bootstrap_values[hi_idx], 2),
    )


def exact_sign_test_p_value(a_only: int, b_only: int) -> float | None:
    """Two-sided exact sign-test p-value over discordant paired outcomes."""
    discordant = int(a_only) + int(b_only)
    if discordant <= 0:
        return None
    tail = min(int(a_only), int(b_only))
    probability = sum(comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return round(min(1.0, 2 * probability), 6)


def paired_binary_delta_summary(
    a_values: Sequence[bool | int | None],
    b_values: Sequence[bool | int | None],
) -> dict[str, Any]:
    """Summarize matched binary outcomes without treating pairs as independent."""
    if len(a_values) != len(b_values):
        raise ValueError(
            f"paired arrays must be equal length (got {len(a_values)} vs {len(b_values)})"
        )

    both_success = both_failure = a_only = b_only = 0
    observed = 0

    for a_value, b_value in zip(a_values, b_values):
        if a_value is None or b_value is None:
            continue
        a = bool(a_value)
        b = bool(b_value)
        observed += 1
        if a and b:
            both_success += 1
        elif not a and not b:
            both_failure += 1
        elif a:
            a_only += 1
        else:
            b_only += 1

    a_successes = both_success + a_only
    b_successes = both_success + b_only
    a_rate = a_successes / observed if observed else 0.0
    b_rate = b_successes / observed if observed else 0.0
    return {
        "n_pairs": observed,
        "a_successes": a_successes,
        "b_successes": b_successes,
        "a_rate": round(a_rate, 6) if observed else None,
        "b_rate": round(b_rate, 6) if observed else None,
        "delta": round(a_rate - b_rate, 6) if observed else None,
        "both_success": both_success,
        "both_failure": both_failure,
        "a_only": a_only,
        "b_only": b_only,
        "discordant": a_only + b_only,
        "exact_sign_test_p": exact_sign_test_p_value(a_only, b_only),
    }
