"""CUPED actually reduces variance versus the naive estimator, without bias."""

import numpy as np

from ab.cuped import apply_cuped, cuped_theta
from ab.data import ExperimentConfig, generate_experiment
from ab.stats_tests import welch_ttest


def test_cuped_reduces_variance():
    df = generate_experiment(ExperimentConfig(n_per_group=5000, seed=11))
    res = apply_cuped(df["metric"].to_numpy(), df["pre_metric"].to_numpy())
    assert res.var_after < res.var_before
    assert res.variance_reduction > 0.30  # strong covariate => sizeable reduction.


def test_cuped_variance_reduction_matches_rho_squared():
    df = generate_experiment(ExperimentConfig(n_per_group=8000, seed=12))
    res = apply_cuped(df["metric"].to_numpy(), df["pre_metric"].to_numpy())
    # Variance reduction should be approximately rho^2.
    assert np.isclose(res.variance_reduction, res.rho**2, atol=0.01)


def test_cuped_preserves_effect_estimate():
    cfg = ExperimentConfig(n_per_group=10000, true_effect=0.2, seed=13)
    df = generate_experiment(cfg)
    res = apply_cuped(df["metric"].to_numpy(), df["pre_metric"].to_numpy())
    df = df.assign(metric_cuped=res.y_adjusted)

    c = df[df["group"] == "control"]
    t = df[df["group"] == "treatment"]
    naive = welch_ttest(c["metric"].to_numpy(), t["metric"].to_numpy())
    cuped = welch_ttest(c["metric_cuped"].to_numpy(), t["metric_cuped"].to_numpy())

    # Same (unbiased) point estimate, but a tighter interval after CUPED.
    assert np.isclose(naive.estimate, cuped.estimate, atol=0.02)
    cuped_half = (cuped.ci_high - cuped.ci_low) / 2
    naive_half = (naive.ci_high - naive.ci_low) / 2
    assert cuped_half < naive_half


def test_theta_zero_when_uncorrelated():
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    x = rng.normal(size=5000)  # independent of y.
    theta = cuped_theta(y, x)
    assert abs(theta) < 0.05
