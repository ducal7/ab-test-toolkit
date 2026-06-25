"""Frequentist two-sample tests with confidence intervals.

Provides:
* two_proportion_ztest : pooled z-test for a difference in conversion rates.
* welch_ttest          : unequal-variance t-test for a difference in means.

Each returns a `TestResult` carrying the point estimate, test statistic,
two-sided p-value, and a confidence interval for the difference (treatment - control).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, t


@dataclass(frozen=True)
class TestResult:
    """Outcome of a two-sample test.

    Attributes
    ----------
    estimate   : point estimate of the difference (treatment - control).
    statistic  : test statistic (z or t).
    pvalue     : two-sided p-value.
    ci_low     : lower bound of the confidence interval for the difference.
    ci_high    : upper bound of the confidence interval for the difference.
    alpha      : significance level used for the interval.
    df         : degrees of freedom (None for the z-test).
    name       : human-readable test name.
    """

    estimate: float
    statistic: float
    pvalue: float
    ci_low: float
    ci_high: float
    alpha: float
    df: float | None
    name: str

    @property
    def significant(self) -> bool:
        return self.pvalue < self.alpha


def two_proportion_ztest(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> TestResult:
    """Pooled two-proportion z-test for treatment - control conversion rate.

    Parameters
    ----------
    x_control, x_treatment : number of successes (conversions) in each group.
    n_control, n_treatment : group sizes.
    alpha                  : two-sided significance level (drives the CI width).

    Notes
    -----
    The test statistic uses the *pooled* standard error (valid under H0), while
    the confidence interval uses the *unpooled* standard error (valid under the
    observed estimate). This is the standard, internally-consistent convention.
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Group sizes must be positive.")
    if not (0 <= x_control <= n_control and 0 <= x_treatment <= n_treatment):
        raise ValueError("Successes must be within [0, n] for each group.")

    p_c = x_control / n_control
    p_t = x_treatment / n_treatment
    estimate = p_t - p_c

    p_pool = (x_control + x_treatment) / (n_control + n_treatment)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_control + 1.0 / n_treatment))
    z = estimate / se_pool if se_pool > 0 else 0.0
    pvalue = 2.0 * norm.sf(abs(z))

    se_unpooled = math.sqrt(
        p_c * (1.0 - p_c) / n_control + p_t * (1.0 - p_t) / n_treatment
    )
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    half = z_crit * se_unpooled

    return TestResult(
        estimate=float(estimate),
        statistic=float(z),
        pvalue=float(pvalue),
        ci_low=float(estimate - half),
        ci_high=float(estimate + half),
        alpha=alpha,
        df=None,
        name="two-proportion z-test",
    )


def welch_ttest(
    control: ArrayLike,
    treatment: ArrayLike,
    alpha: float = 0.05,
) -> TestResult:
    """Welch's unequal-variance t-test for treatment - control mean difference.

    Parameters
    ----------
    control, treatment : 1-D arrays of outcome values for each group.
    alpha              : two-sided significance level.
    """
    c = np.asarray(control, dtype=float)
    t_arr = np.asarray(treatment, dtype=float)
    if c.size < 2 or t_arr.size < 2:
        raise ValueError("Each group needs at least 2 observations.")

    n_c, n_t = c.size, t_arr.size
    m_c, m_t = c.mean(), t_arr.mean()
    v_c = c.var(ddof=1)
    v_t = t_arr.var(ddof=1)
    estimate = m_t - m_c

    se = math.sqrt(v_c / n_c + v_t / n_t)
    stat = estimate / se if se > 0 else 0.0

    # Welch-Satterthwaite degrees of freedom.
    num = (v_c / n_c + v_t / n_t) ** 2
    den = (v_c / n_c) ** 2 / (n_c - 1) + (v_t / n_t) ** 2 / (n_t - 1)
    df = num / den if den > 0 else float(n_c + n_t - 2)

    pvalue = 2.0 * t.sf(abs(stat), df)
    t_crit = t.ppf(1.0 - alpha / 2.0, df)
    half = t_crit * se

    return TestResult(
        estimate=float(estimate),
        statistic=float(stat),
        pvalue=float(pvalue),
        ci_low=float(estimate - half),
        ci_high=float(estimate + half),
        alpha=alpha,
        df=float(df),
        name="Welch t-test",
    )
