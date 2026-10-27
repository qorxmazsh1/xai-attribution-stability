
"""Estimate bootstrap confidence intervals for SHAP stability metrics."""

import json
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "stability"

INPUT_FILE = RESULTS_DIR / "model_comparison_stability.csv"
OUTPUT_FILE = RESULTS_DIR / "bootstrap_confidence_intervals.csv"
METADATA_FILE = RESULTS_DIR / "bootstrap_metadata.json"

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
CONFIDENCE_LEVEL = 0.95

METRICS = {
    "prediction_unchanged": "prediction_unchanged_rate",
    "absolute_probability_change": "mean_abs_probability_change",
    "shap_cosine_similarity": "mean_shap_cosine",
    "absolute_shap_rank_correlation": (
        "mean_abs_shap_rank_correlation"
    ),
    "relative_shap_l1_change": "mean_relative_shap_l1_change",
}

def wilson_interval(
    successes: int,
    n: int,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Calculate a Wilson confidence interval for a proportion."""
    if n <= 0:
        return float("nan"), float("nan")

    z = NormalDist().inv_cdf(
        1 - (1 - confidence_level) / 2
    )
    proportion = successes / n
    z_squared = z**2

    denominator = 1 + z_squared / n

    center = (
        proportion + z_squared / (2 * n)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / n
            + z_squared / (4 * n**2)
        )
        / denominator
    )

    return (
        float(max(0.0, center - margin)),
        float(min(1.0, center + margin)),
    )

def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float]:
    """Return the observed mean and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    estimate = float(np.mean(values))

    if values.size == 1:
        return estimate, float("nan"), float("nan")

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=values.size,
            replace=True,
        )
        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence_level

    lower = float(np.quantile(bootstrap_means, alpha / 2))
    upper = float(np.quantile(bootstrap_means, 1 - alpha / 2))

    return estimate, lower, upper


def main() -> None:
    """Calculate confidence intervals for each model and perturbation."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(INPUT_FILE)

    required_columns = {
        "model",
        "perturbation",
        *METRICS.keys(),
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    grouped = results.groupby(
        ["model", "perturbation"],
        sort=True,
    )

    for (model_name, perturbation), group in grouped:
        print(
            f"Analyzing {model_name} | {perturbation} "
            f"(n={len(group)})"
        )

        for source_column, metric_name in METRICS.items():
            values = pd.to_numeric(
                group[source_column],
                errors="coerce",
            ).to_numpy()

            if source_column == "prediction_unchanged":
                valid_values = values[np.isfinite(values)]
                n_valid = len(valid_values)
                successes = int(np.sum(valid_values))

                estimate = float(successes / n_valid) if n_valid > 0 else float("nan")

                lower, upper = wilson_interval(successes, n_valid)
        else:
            estimate, lower, upper = bootstrap_mean_ci(values, rng)

            rows.append(
                {
                    "model": model_name,
                    "perturbation": perturbation,
                    "metric": metric_name,
                    "n": int(np.isfinite(values).sum()),
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": CONFIDENCE_LEVEL,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_FILE, index=False)

    metadata = {
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "random_state": RANDOM_STATE,
        "n_bootstrap": N_BOOTSTRAP,
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "Percentile bootstrap of observations within each group",
        "interpretation": (
            "Intervals quantify observation-level uncertainty conditional "
            "on the existing fitted models, test split, background sample, "
            "and perturbation definitions. They do not capture variation "
            "from retraining models or resampling the SHAP background."
        ),
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nBootstrap confidence intervals:")
    print(output.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")
    print(f"Saved metadata to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
