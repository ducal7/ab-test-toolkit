"""Power and sample-size calculators.

All formulas use the normal approximation. Sample-size functions return the
required size *per group* assuming a balanced (1:1) design and a two-sided test.

References
----------
* Two proportions:
    n = ( z_{a/2} * sqrt(2*pbar*(1-pbar)) + z_b * sqrt(p1(1-p1)+p2(1-p2)) )^2 / (p1-p2)^2
* Two means (equal variance):
    n = 2 * (z_{a/2} + z_b)^2 * sigma^2 / delta^2
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _z_two_sided(alpha: float) -> float:
    return norm.ppf(1.0 - alpha / 2.0)


def _z_power(power: float) -> float:
    return norm.ppf(power)


def sample_size_two_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Required sample size per group to detect a difference between two proportions.

    Parameters
    ----------
    p1, p2 : baseline and variant conversion rates (must differ).
    alpha  : two-sided significance level.
    power  : desired statistical power (1 - beta).

    Returns
    -------
    int : sample size per group (rounded up).
    """
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1 and p2 must be in (0, 1).")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ to define an effect.")

    z_a = _z_two_sided(alpha)
    z_b = _z_power(power)
    p_bar = (p1 + p2) / 2.0
    pooled = z_a * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
    unpooled = z_b * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    n = (pooled + unpooled) ** 2 / (p1 - p2) ** 2
    return math.ceil(n)


def power_two_proportions(
    p1: float,
    p2: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """Achieved power of a two-proportion test for a given per-group sample size."""
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")
    z_a = _z_two_sided(alpha)
    p_bar = (p1 + p2) / 2.0
    se_null = math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n_per_group)
    se_alt = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n_per_group)
    effect = abs(p1 - p2)
    z = (effect - z_a * se_null) / se_alt
    return float(norm.cdf(z))


def sample_size_two_means(
    delta: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Required sample size per group to detect a mean difference `delta`.

    Parameters
    ----------
    delta : true difference in means to detect (must be non-zero).
    sigma : common standard deviation of the outcome.
    alpha : two-sided significance level.
    power : desired statistical power.

    Returns
    -------
    int : sample size per group (rounded up).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if delta == 0:
        raise ValueError("delta must be non-zero to define an effect.")
    z_a = _z_two_sided(alpha)
    z_b = _z_power(power)
    n = 2.0 * (z_a + z_b) ** 2 * sigma**2 / delta**2
    return math.ceil(n)


def power_two_means(
    delta: float,
    sigma: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """Achieved power of a two-sample means test for a given per-group sample size."""
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    z_a = _z_two_sided(alpha)
    se = sigma * math.sqrt(2.0 / n_per_group)
    z = abs(delta) / se - z_a
    return float(norm.cdf(z))
