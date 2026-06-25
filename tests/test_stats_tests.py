"""Frequentist tests: recover significance on synthetic data with a known effect,
and agree with scipy on the means test."""

import numpy as np
from scipy import stats

from ab.data import ExperimentConfig, generate_experiment
from ab.stats_tests import two_proportion_ztest, welch_ttest


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=300)
    b = rng.normal(0.4, 1.2, size=280)
    res = welch_ttest(a, b)
    sp = stats.ttest_ind(b, a, equal_var=False)
    assert np.isclose(res.statistic, sp.statistic, rtol=1e-10)
    assert np.isclose(res.pvalue, sp.pvalue, rtol=1e-10)


def test_two_proportion_matches_manual():
    # 120/1000 vs 150/1000.
    res = two_proportion_ztest(120, 1000, 150, 1000)
    p_pool = (120 + 150) / 2000
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / 1000 + 1 / 1000))
    z = (0.15 - 0.12) / se
    assert np.isclose(res.statistic, z, rtol=1e-10)
    assert np.isclose(res.estimate, 0.03, atol=1e-12)


def test_recovers_significance_means():
    # Known true effect of 0.2 with large n should be detected.
    df = generate_experiment(ExperimentConfig(n_per_group=5000, true_effect=0.2, seed=1))
    c = df.loc[df["group"] == "control", "metric"].to_numpy()
    t = df.loc[df["group"] == "treatment", "metric"].to_numpy()
    res = welch_ttest(c, t)
    assert res.significant
    assert res.ci_low < 0.2 < res.ci_high  # CI covers the true effect.


def test_recovers_significance_proportions():
    df = generate_experiment(
        ExperimentConfig(n_per_group=20000, conversion_lift=0.03, seed=2)
    )
    c = df[df["group"] == "control"]
    t = df[df["group"] == "treatment"]
    res = two_proportion_ztest(
        int(c["converted"].sum()), len(c), int(t["converted"].sum()), len(t)
    )
    assert res.significant
    assert res.estimate > 0


def test_no_effect_usually_not_significant():
    df = generate_experiment(ExperimentConfig(n_per_group=4000, true_effect=0.0, seed=3))
    c = df.loc[df["group"] == "control", "metric"].to_numpy()
    t = df.loc[df["group"] == "treatment", "metric"].to_numpy()
    res = welch_ttest(c, t)
    assert not res.significant
