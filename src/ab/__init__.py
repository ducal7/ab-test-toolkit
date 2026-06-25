"""ab: a reusable A/B test analysis toolkit.

Modules
-------
power        : power and sample-size calculators (two-proportion, two-sample means).
stats_tests  : frequentist tests (two-proportion z-test, Welch t-test) with CIs.
cuped        : CUPED variance reduction.
corrections  : multiple-comparison corrections and sequential-peeking guardrails.
data         : seeded synthetic experiment generator (configurable true effect).
report       : end-to-end worked report.
"""

from ab.corrections import (
    benjamini_hochberg,
    bonferroni,
    peeking_false_positive_rate,
    pocock_alpha,
)
from ab.cuped import CupedResult, apply_cuped, cuped_theta
from ab.power import (
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
)
from ab.stats_tests import TestResult, two_proportion_ztest, welch_ttest

__all__ = [
    "CupedResult",
    "TestResult",
    "apply_cuped",
    "benjamini_hochberg",
    "bonferroni",
    "cuped_theta",
    "peeking_false_positive_rate",
    "pocock_alpha",
    "power_two_means",
    "power_two_proportions",
    "sample_size_two_means",
    "sample_size_two_proportions",
    "two_proportion_ztest",
    "welch_ttest",
]

__version__ = "0.1.0"
