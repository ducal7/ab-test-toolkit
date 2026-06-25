"""Multiple-comparison corrections and the peeking guardrail."""

import numpy as np

from ab.corrections import (
    benjamini_hochberg,
    bonferroni,
    peeking_false_positive_rate,
    pocock_alpha,
)


def test_bonferroni_threshold():
    pvals = np.array([0.001, 0.02, 0.04, 0.5])
    # alpha/m = 0.05/4 = 0.0125; only 0.001 passes.
    rejected = bonferroni(pvals, alpha=0.05)
    assert rejected.tolist() == [True, False, False, False]


def test_benjamini_hochberg_basic():
    # m=4, BH thresholds = 0.05 * [1,2,3,4]/4 = [0.0125, 0.025, 0.0375, 0.05].
    # Sorted p = [0.001, 0.008, 0.030, 0.5]; largest passing index is the 3rd.
    pvals = np.array([0.001, 0.008, 0.030, 0.5])
    rejected = benjamini_hochberg(pvals, alpha=0.05)
    assert rejected.tolist() == [True, True, True, False]


def test_bh_at_least_as_powerful_as_bonferroni():
    rng = np.random.default_rng(0)
    pvals = np.concatenate([rng.uniform(0, 0.01, 5), rng.uniform(0, 1, 20)])
    bh = benjamini_hochberg(pvals)
    bon = bonferroni(pvals)
    assert bh.sum() >= bon.sum()


def test_pocock_alpha_is_stricter_for_more_looks():
    assert pocock_alpha(1) == 0.05
    assert pocock_alpha(5) < pocock_alpha(2) < pocock_alpha(1)


def test_peeking_inflates_false_positives():
    single = peeking_false_positive_rate(n_looks=1, n_sims=1500, seed=0)
    many = peeking_false_positive_rate(n_looks=10, n_sims=1500, seed=0)
    # A single look is ~nominal; ten naive looks inflate well above 0.05.
    assert single < 0.08
    assert many > 0.10
    assert many > single
