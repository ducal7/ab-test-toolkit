"""Generator determinism and structural guarantees."""

import numpy as np
import pandas as pd

from ab.data import ExperimentConfig, generate_experiment


def test_generator_is_deterministic():
    a = generate_experiment(ExperimentConfig(seed=42))
    b = generate_experiment(ExperimentConfig(seed=42))
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate_experiment(ExperimentConfig(seed=1))
    b = generate_experiment(ExperimentConfig(seed=2))
    assert not a["metric"].equals(b["metric"])


def test_balanced_groups_and_columns():
    df = generate_experiment(ExperimentConfig(n_per_group=1000))
    assert set(df.columns) == {"user_id", "group", "pre_metric", "metric", "converted"}
    counts = df["group"].value_counts()
    assert counts["control"] == 1000
    assert counts["treatment"] == 1000


def test_true_effect_shows_up_in_means():
    df = generate_experiment(ExperimentConfig(n_per_group=20000, true_effect=0.5, seed=5))
    means = df.groupby("group")["metric"].mean()
    observed = means["treatment"] - means["control"]
    assert np.isclose(observed, 0.5, atol=0.05)


def test_pre_metric_balanced_across_groups():
    # The covariate must not be affected by assignment (precondition for CUPED).
    df = generate_experiment(ExperimentConfig(n_per_group=20000, seed=6))
    means = df.groupby("group")["pre_metric"].mean()
    assert np.isclose(means["treatment"], means["control"], atol=0.05)
