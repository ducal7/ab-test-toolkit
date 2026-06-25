"""End-to-end worked A/B test report.

Run with::

    python -m ab.report

Steps
-----
1. Generate a seeded synthetic experiment with a known true effect.
2. Run an a-priori power / sample-size analysis.
3. Run the frequentist tests (two-proportion z-test and Welch t-test) with CIs.
4. Apply CUPED and quantify the variance reduction.
5. Demonstrate the sequential-peeking pitfall.
6. Write plots and a metrics table into ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ab.corrections import peeking_false_positive_rate, pocock_alpha  # noqa: E402
from ab.cuped import apply_cuped  # noqa: E402
from ab.data import ExperimentConfig, generate_experiment  # noqa: E402
from ab.power import sample_size_two_means, sample_size_two_proportions  # noqa: E402
from ab.stats_tests import two_proportion_ztest, welch_ttest  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _fmt(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def run_report(results_dir: Path = RESULTS_DIR) -> dict:
    """Run the full analysis, write artifacts, and return a metrics dict."""
    results_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    df = generate_experiment(config)

    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    # ---- 1. Power / sample-size analysis (a priori) --------------------------
    n_means = sample_size_two_means(
        delta=config.true_effect, sigma=config.noise_sd, alpha=0.05, power=0.80
    )
    n_prop = sample_size_two_proportions(
        p1=config.base_rate,
        p2=config.base_rate + config.conversion_lift,
        alpha=0.05,
        power=0.80,
    )

    # ---- 2. Frequentist tests ------------------------------------------------
    prop_res = two_proportion_ztest(
        x_control=int(control["converted"].sum()),
        n_control=len(control),
        x_treatment=int(treatment["converted"].sum()),
        n_treatment=len(treatment),
    )
    mean_res = welch_ttest(control["metric"].to_numpy(), treatment["metric"].to_numpy())

    # ---- 3. CUPED on the continuous outcome ----------------------------------
    cuped = apply_cuped(df["metric"].to_numpy(), df["pre_metric"].to_numpy())
    df_adj = df.assign(metric_cuped=cuped.y_adjusted)
    mean_res_cuped = welch_ttest(
        df_adj.loc[df_adj["group"] == "control", "metric_cuped"].to_numpy(),
        df_adj.loc[df_adj["group"] == "treatment", "metric_cuped"].to_numpy(),
    )

    # ---- 4. Sequential-peeking pitfall ---------------------------------------
    peek_fpr = peeking_false_positive_rate(n_looks=10, n_per_look=200, n_sims=1000, seed=7)

    metrics = {
        "config": config.__dict__,
        "power_analysis": {
            "n_per_group_means_for_true_effect": n_means,
            "n_per_group_proportions_for_lift": n_prop,
            "actual_n_per_group": config.n_per_group,
        },
        "proportion_test": {
            "control_rate": _fmt(control["converted"].mean()),
            "treatment_rate": _fmt(treatment["converted"].mean()),
            "estimate": _fmt(prop_res.estimate),
            "z": _fmt(prop_res.statistic),
            "pvalue": _fmt(prop_res.pvalue, 6),
            "ci": [_fmt(prop_res.ci_low), _fmt(prop_res.ci_high)],
            "significant": prop_res.significant,
        },
        "means_test_naive": {
            "estimate": _fmt(mean_res.estimate),
            "t": _fmt(mean_res.statistic),
            "pvalue": _fmt(mean_res.pvalue, 6),
            "ci": [_fmt(mean_res.ci_low), _fmt(mean_res.ci_high)],
            "ci_halfwidth": _fmt((mean_res.ci_high - mean_res.ci_low) / 2),
            "significant": mean_res.significant,
        },
        "means_test_cuped": {
            "estimate": _fmt(mean_res_cuped.estimate),
            "t": _fmt(mean_res_cuped.statistic),
            "pvalue": _fmt(mean_res_cuped.pvalue, 6),
            "ci": [_fmt(mean_res_cuped.ci_low), _fmt(mean_res_cuped.ci_high)],
            "ci_halfwidth": _fmt((mean_res_cuped.ci_high - mean_res_cuped.ci_low) / 2),
            "significant": mean_res_cuped.significant,
        },
        "cuped": {
            "theta": _fmt(cuped.theta),
            "rho": _fmt(cuped.rho),
            "var_before": _fmt(cuped.var_before),
            "var_after": _fmt(cuped.var_after),
            "variance_reduction_pct": _fmt(100.0 * cuped.variance_reduction, 2),
        },
        "peeking": {
            "naive_false_positive_rate_10_looks": peek_fpr,
            "nominal_alpha": 0.05,
            "pocock_per_look_alpha_10_looks": pocock_alpha(10),
        },
    }

    _write_plots(df, cuped, results_dir)
    _write_table(metrics, results_dir)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _write_plots(df, cuped, results_dir: Path) -> None:
    # Plot 1: outcome distribution before vs after CUPED.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    axes[0].hist(df["metric"], bins=50, color="#4C72B0", alpha=0.85)
    axes[0].set_title(f"Raw outcome (var = {cuped.var_before:.3f})")
    axes[0].set_xlabel("metric")
    axes[0].set_ylabel("count")
    axes[1].hist(cuped.y_adjusted, bins=50, color="#55A868", alpha=0.85)
    axes[1].set_title(f"CUPED-adjusted (var = {cuped.var_after:.3f})")
    axes[1].set_xlabel("metric (CUPED)")
    fig.suptitle(
        f"CUPED variance reduction: {100 * cuped.variance_reduction:.1f}%  "
        f"(rho = {cuped.rho:.2f})"
    )
    fig.tight_layout()
    fig.savefig(results_dir / "cuped_variance_reduction.png", dpi=120)
    plt.close(fig)

    # Plot 2: covariate vs outcome scatter showing the exploitable correlation.
    fig, ax = plt.subplots(figsize=(6, 5))
    sample = df.sample(n=min(1500, len(df)), random_state=0)
    ax.scatter(sample["pre_metric"], sample["metric"], s=8, alpha=0.35, color="#4C72B0")
    coeffs = np.polyfit(df["pre_metric"], df["metric"], 1)
    xs = np.linspace(df["pre_metric"].min(), df["pre_metric"].max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="#C44E52", lw=2, label="OLS fit")
    ax.set_xlabel("pre_metric (pre-period covariate X)")
    ax.set_ylabel("metric (outcome Y)")
    ax.set_title(f"Pre-period covariate vs outcome (rho = {cuped.rho:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "covariate_vs_outcome.png", dpi=120)
    plt.close(fig)

    # Plot 3: peeking false-positive rate vs number of looks.
    looks = list(range(1, 11))
    fprs = [
        peeking_false_positive_rate(n_looks=k, n_per_look=200, n_sims=600, seed=7)
        for k in looks
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(looks, fprs, marker="o", color="#C44E52", label="naive peeking")
    ax.axhline(0.05, ls="--", color="#555", label="nominal alpha = 0.05")
    ax.set_xlabel("number of peeks (interim looks)")
    ax.set_ylabel("type-I error (A/A false-positive rate)")
    ax.set_title("Naive sequential peeking inflates false positives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "peeking_false_positive_rate.png", dpi=120)
    plt.close(fig)


def _write_table(metrics: dict, results_dir: Path) -> None:
    p = metrics["proportion_test"]
    mn = metrics["means_test_naive"]
    mc = metrics["means_test_cuped"]
    cu = metrics["cuped"]
    lines = [
        "# Worked example results\n",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| True continuous effect (configured) | {metrics['config']['true_effect']} |",
        f"| Welch t-test estimate (naive) | {mn['estimate']} |",
        f"| Welch t-test p-value (naive) | {mn['pvalue']} |",
        f"| 95% CI (naive) | [{mn['ci'][0]}, {mn['ci'][1]}] |",
        f"| Welch t-test estimate (CUPED) | {mc['estimate']} |",
        f"| Welch t-test p-value (CUPED) | {mc['pvalue']} |",
        f"| 95% CI (CUPED) | [{mc['ci'][0]}, {mc['ci'][1]}] |",
        f"| CUPED theta | {cu['theta']} |",
        f"| corr(Y, X) | {cu['rho']} |",
        f"| Variance before / after | {cu['var_before']} / {cu['var_after']} |",
        f"| **CUPED variance reduction** | **{cu['variance_reduction_pct']}%** |",
        f"| Conversion control / treatment | {p['control_rate']} / {p['treatment_rate']} |",
        f"| Two-proportion z p-value | {p['pvalue']} |",
        f"| Conversion 95% CI | [{p['ci'][0]}, {p['ci'][1]}] |",
        "",
    ]
    (results_dir / "results_table.md").write_text("\n".join(lines))


def main() -> None:
    metrics = run_report()
    print(json.dumps(metrics, indent=2))
    cu = metrics["cuped"]
    mn = metrics["means_test_naive"]
    mc = metrics["means_test_cuped"]
    print("\n=== Worked example summary ===")
    print(f"Naive Welch t-test : effect={mn['estimate']}, p={mn['pvalue']}, CI={mn['ci']}")
    print(f"CUPED Welch t-test : effect={mc['estimate']}, p={mc['pvalue']}, CI={mc['ci']}")
    print(f"CUPED variance reduction: {cu['variance_reduction_pct']}%")


if __name__ == "__main__":
    main()
