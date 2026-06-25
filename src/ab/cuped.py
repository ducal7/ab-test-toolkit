"""CUPED: Controlled-experiment Using Pre-Experiment Data.

CUPED (Deng, Xu, Kohavi & Walker, 2013) reduces the variance of an experiment's
outcome metric by regressing out a pre-experiment covariate `X` that is correlated
with the outcome `Y` but, by construction, unaffected by the treatment.

The adjusted metric is

    Y_cuped = Y - theta * (X - mean(X)),   theta = cov(Y, X) / var(X)

`theta` is estimated on the *pooled* (control + treatment) data so the adjustment
is independent of treatment assignment and therefore unbiased for the treatment
effect. The variance of the adjusted metric is reduced by a factor of (1 - rho^2),
where rho = corr(Y, X). Subtracting the mean keeps the metric on its original scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CupedResult:
    """Result of a CUPED adjustment.

    Attributes
    ----------
    y_adjusted        : CUPED-adjusted outcome (same length/scale as input `y`).
    theta             : estimated regression coefficient cov(Y, X) / var(X).
    rho               : correlation between Y and X (on the pooled data).
    var_before        : variance of the raw outcome.
    var_after         : variance of the adjusted outcome.
    variance_reduction: fractional reduction in variance, 1 - var_after / var_before.
    """

    y_adjusted: np.ndarray
    theta: float
    rho: float
    var_before: float
    var_after: float
    variance_reduction: float


def cuped_theta(y: ArrayLike, x: ArrayLike) -> float:
    """Estimate the CUPED coefficient theta = cov(Y, X) / var(X) on pooled data."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.shape != x_arr.shape:
        raise ValueError("y and x must have the same shape.")
    var_x = x_arr.var(ddof=1)
    if var_x == 0:
        raise ValueError("Covariate x has zero variance; CUPED is undefined.")
    cov_xy = np.cov(y_arr, x_arr, ddof=1)[0, 1]
    return float(cov_xy / var_x)


def apply_cuped(y: ArrayLike, x: ArrayLike, theta: float | None = None) -> CupedResult:
    """Apply the CUPED adjustment to outcome `y` using covariate `x`.

    Parameters
    ----------
    y     : outcome metric (1-D).
    x     : pre-experiment covariate (1-D), same length as y.
    theta : optional pre-computed coefficient; if None it is estimated on `(y, x)`.

    Returns
    -------
    CupedResult
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.shape != x_arr.shape:
        raise ValueError("y and x must have the same shape.")

    if theta is None:
        theta = cuped_theta(y_arr, x_arr)

    y_adj = y_arr - theta * (x_arr - x_arr.mean())

    var_before = float(y_arr.var(ddof=1))
    var_after = float(y_adj.var(ddof=1))
    rho = float(np.corrcoef(y_arr, x_arr)[0, 1])
    reduction = 1.0 - var_after / var_before if var_before > 0 else 0.0

    return CupedResult(
        y_adjusted=y_adj,
        theta=float(theta),
        rho=rho,
        var_before=var_before,
        var_after=var_after,
        variance_reduction=reduction,
    )
