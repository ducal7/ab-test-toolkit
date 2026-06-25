"""Power and sample-size math against hand-computed reference values."""

import math

from ab.power import (
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
)


def test_sample_size_two_proportions_reference():
    # p1=0.5, p2=0.6, alpha=0.05, power=0.80 -> 388 per group.
    # Hand check: (1.959964*sqrt(0.495) + 0.841621*0.7)^2 / 0.01 = 387.33 -> ceil 388.
    assert sample_size_two_proportions(0.5, 0.6, alpha=0.05, power=0.80) == 388


def test_sample_size_two_means_reference():
    # Cohen d = delta/sigma = 0.5, alpha=0.05, power=0.80 (normal approx).
    # n = 2*(1.959964+0.841621)^2 / 0.25 = 62.79 -> ceil 63 per group.
    assert sample_size_two_means(delta=0.5, sigma=1.0, alpha=0.05, power=0.80) == 63


def test_power_roundtrip_proportions():
    # At the computed sample size, achieved power should be >= the target.
    n = sample_size_two_proportions(0.20, 0.25, power=0.80)
    achieved = power_two_proportions(0.20, 0.25, n)
    assert achieved >= 0.80
    assert achieved < 0.83  # just above target, not wildly over.


def test_power_roundtrip_means():
    n = sample_size_two_means(delta=0.5, sigma=1.0, power=0.80)
    achieved = power_two_means(delta=0.5, sigma=1.0, n_per_group=n)
    assert achieved >= 0.80
    assert achieved < 0.82


def test_power_monotonic_in_n():
    low = power_two_means(0.3, 1.0, 50)
    high = power_two_means(0.3, 1.0, 500)
    assert high > low


def test_sample_size_scales_with_effect():
    # Halving the effect roughly quadruples the required sample size.
    small = sample_size_two_means(delta=0.25, sigma=1.0)
    big = sample_size_two_means(delta=0.50, sigma=1.0)
    assert math.isclose(small / big, 4.0, rel_tol=0.05)
