"""Seeded synthetic experiment generator.

Produces a balanced two-arm experiment with:

* ``pre_metric``  : a pre-period covariate X (e.g. last week's revenue per user),
                    unaffected by treatment -- this is what CUPED exploits.
* ``metric``      : the continuous outcome Y, correlated with ``pre_metric`` and
                    shifted by a CONFIGURABLE ``true_effect`` in the treatment arm.
* ``converted``   : a binary conversion outcome with a configurable lift.

The model for the continuous outcome is

    Y = base + corr_strength * (X - mean(X)) + true_effect * 1[treatment] + noise

so X explains a tunable share of Y's variance, letting CUPED demonstrably reduce
variance. Generation is fully deterministic given ``seed``.

Run as a module to (re)generate the dataset on disk::

    python -m ab.data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_PATH = DATA_DIR / "experiment.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a synthetic experiment."""

    n_per_group: int = 5000
    true_effect: float = 0.20  # additive lift on the continuous outcome (treatment).
    base_mean: float = 10.0  # baseline level of the continuous outcome.
    corr_strength: float = 0.80  # loading of pre_metric onto the outcome.
    noise_sd: float = 1.0  # idiosyncratic noise SD on the outcome.
    pre_mean: float = 10.0  # mean of the pre-period covariate.
    pre_sd: float = 1.0  # SD of the pre-period covariate.
    base_rate: float = 0.20  # control conversion rate.
    conversion_lift: float = 0.04  # additive lift on conversion (treatment).
    seed: int = 42


def generate_experiment(config: ExperimentConfig | None = None, **overrides) -> pd.DataFrame:
    """Generate a deterministic synthetic experiment as a tidy DataFrame.

    Parameters
    ----------
    config    : an ExperimentConfig; if None, defaults are used.
    overrides : keyword overrides applied on top of ``config`` (e.g. ``true_effect=0.5``).

    Returns
    -------
    pandas.DataFrame with columns:
        user_id, group, pre_metric, metric, converted.
    """
    if config is None:
        config = ExperimentConfig()
    if overrides:
        config = ExperimentConfig(**{**config.__dict__, **overrides})

    rng = np.random.default_rng(config.seed)
    n = config.n_per_group
    total = 2 * n

    # Balanced assignment: first n control, next n treatment, then shuffle order.
    group = np.array(["control"] * n + ["treatment"] * n)
    is_treat = (group == "treatment").astype(float)

    # Pre-period covariate, independent of assignment.
    pre = rng.normal(config.pre_mean, config.pre_sd, size=total)
    pre_centered = pre - config.pre_mean

    # Continuous outcome correlated with the covariate plus the treatment effect.
    noise = rng.normal(0.0, config.noise_sd, size=total)
    metric = (
        config.base_mean
        + config.corr_strength * pre_centered
        + config.true_effect * is_treat
        + noise
    )

    # Binary conversion outcome with an additive treatment lift.
    prob = np.clip(config.base_rate + config.conversion_lift * is_treat, 0.0, 1.0)
    converted = (rng.random(total) < prob).astype(int)

    df = pd.DataFrame(
        {
            "user_id": np.arange(total),
            "group": group,
            "pre_metric": pre,
            "metric": metric,
            "converted": converted,
        }
    )
    # Deterministic shuffle so rows are not ordered by group.
    perm = rng.permutation(total)
    return df.iloc[perm].reset_index(drop=True)


def write_experiment(path: Path = DEFAULT_PATH, config: ExperimentConfig | None = None) -> Path:
    """Generate and persist the experiment to CSV; returns the output path."""
    df = generate_experiment(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic A/B experiment data.")
    p.add_argument("--out", type=Path, default=DEFAULT_PATH, help="output CSV path")
    p.add_argument("--n-per-group", type=int, default=ExperimentConfig.n_per_group)
    p.add_argument("--true-effect", type=float, default=ExperimentConfig.true_effect)
    p.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = ExperimentConfig(
        n_per_group=args.n_per_group,
        true_effect=args.true_effect,
        seed=args.seed,
    )
    out = write_experiment(args.out, config)
    df = pd.read_csv(out)
    print(f"Wrote {len(df)} rows to {out}")
    print(df.groupby("group")[["pre_metric", "metric", "converted"]].mean())


if __name__ == "__main__":
    main()
