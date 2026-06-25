"""Pitfall guardrails: multiple comparisons and sequential peeking.

Multiple comparisons
--------------------
Running many tests at level ``alpha`` inflates the family-wise false-positive
rate. With ``m`` independent true nulls the chance of at least one false positive
is ``1 - (1 - alpha) ** m`` (e.g. ~0.40 for m=10 at alpha=0.05). Use
``bonferroni`` to control the family-wise error rate, or ``benjamini_hochberg``
to control the (less conservative) false-discovery rate.

Sequential peeking
------------------
Repeatedly checking a fixed-horizon test and stopping the first time ``p < alpha``
inflates the type-I error far above the nominal level, because each peek is an
extra chance to cross the threshold. ``peeking_false_positive_rate`` quantifies
this by simulation. The principled fixes are group-sequential alpha-spending
boundaries (e.g. Pocock or O'Brien-Fleming) or always-valid sequential tests.
``pocock_alpha`` returns the constant per-look significance level that keeps the
overall type-I error at ``alpha`` across ``k`` equally-spaced looks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

# Pocock constant per-look two-sided nominal alpha for an overall alpha of 0.05,
# for k = 1..10 equally-spaced interim analyses (standard tabulated values).
_POCOCK_05 = {
    1: 0.0500,
    2: 0.0294,
    3: 0.0221,
    4: 0.0182,
    5: 0.0158,
    6: 0.0142,
    7: 0.0130,
    8: 0.0120,
    9: 0.0112,
    10: 0.0106,
}


def bonferroni(pvalues: ArrayLike, alpha: float = 0.05) -> np.ndarray:
    """Bonferroni family-wise error correction.

    Returns a boolean array marking which hypotheses are rejected at level
    ``alpha`` after dividing by the number of tests.
    """
    p = np.asarray(pvalues, dtype=float)
    return p < (alpha / p.size)


def benjamini_hochberg(pvalues: ArrayLike, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate procedure.

    Returns a boolean array marking rejected hypotheses (in the original order).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.max(np.flatnonzero(passed))
        rejected[order[: k + 1]] = True
    return rejected


def pocock_alpha(k: int) -> float:
    """Per-look nominal alpha for a Pocock boundary with ``k`` interim looks.

    Keeps the overall two-sided type-I error at 0.05. Valid for k in 1..10.
    """
    if k not in _POCOCK_05:
        raise ValueError("k must be an integer in 1..10.")
    return _POCOCK_05[k]


def peeking_false_positive_rate(
    n_looks: int,
    n_per_look: int = 200,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int = 0,
) -> float:
    """Estimate the inflated type-I error from naive sequential peeking.

    Simulates an A/A test (no true effect): two streams of standard-normal data
    grow in ``n_looks`` increments of ``n_per_look``. After each increment a
    z-test is run and the experiment "stops" the first time ``p < alpha``. The
    returned value is the fraction of simulations that falsely reject at *any*
    look, which exceeds ``alpha`` and grows with ``n_looks``.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be >= 1.")
    rng = np.random.default_rng(seed)
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    false_positives = 0

    for _ in range(n_sims):
        a = np.empty(0)
        b = np.empty(0)
        rejected = False
        for _ in range(n_looks):
            a = np.concatenate([a, rng.standard_normal(n_per_look)])
            b = np.concatenate([b, rng.standard_normal(n_per_look)])
            diff = b.mean() - a.mean()
            se = np.sqrt(b.var(ddof=1) / b.size + a.var(ddof=1) / a.size)
            if se > 0 and abs(diff / se) > z_crit:
                rejected = True
                break
        false_positives += int(rejected)

    return false_positives / n_sims
