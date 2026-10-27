
"""Compare SHAP stability between models across random seeds."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "results" / "stability" / "multi_seed_stability.csv"
OUTPUT_DIR = ROOT / "results" / "stability"

SEED_COLUMN = "seed"
MODEL_COLUMN = "model"
PERTURBATION_COLUMN = "perturbation"
METRIC_COLUMN = "relative_shap_l1_change"

MODEL_NAMES = ["logistic_regression", "random_forest"]


def exact_sign_flip_test(differences: list[float]) -> float:
    """Calculate a two-sided exact sign-flip permutation p-value."""
    values = [float(value) for value in differences]
    observed = abs(sum(values) / len(values))

    permuted_means = []
    for signs in product([-1, 1], repeat=len(values)):
        mean_difference = sum(
            sign * value for sign, value in zip(signs, values)
        ) / len(values)
        permuted_means.append(abs(mean_difference))

    return sum(
        value >= observed - 1e-12 for value in permuted_means
    ) / len(permuted_means)


def main() -> None:
    """Run paired seed-level stability comparisons and create plots."""
    data = pd.read_csv(INPUT_PATH)

    required = {
        SEED_COLUMN,
        MODEL_COLUMN,
        PERTURBATION_COLUMN,
        METRIC_COLUMN,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            f"Available columns: {data.columns.tolist()}"
        )

    data = data.dropna(subset=[METRIC_COLUMN])

    # First average observations within each seed, model and scenario.
    per_seed = (
        data.groupby(
            [SEED_COLUMN, MODEL_COLUMN, PERTURBATION_COLUMN],
            as_index=False,
        )[METRIC_COLUMN]
        .mean()
    )

    wide = per_seed.pivot(
        index=[SEED_COLUMN, PERTURBATION_COLUMN],
        columns=MODEL_COLUMN,
        values=METRIC_COLUMN,
    ).reset_index()

    for model in MODEL_NAMES:
        if model not in wide.columns:
            raise ValueError(
                f"Expected model '{model}'. "
                f"Found: {wide.columns.tolist()}"
            )

    # Positive difference means Random Forest changes more;
    # therefore Logistic Regression has lower relative SHAP change.
    wide["rf_minus_lr"] = (
        wide["random_forest"] - wide["logistic_regression"]
    )

    wide.to_csv(
        OUTPUT_DIR / "paired_seed_stability_differences.csv",
        index=False,
    )

    summary = (
        per_seed.groupby([MODEL_COLUMN, PERTURBATION_COLUMN])
        [METRIC_COLUMN]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(
        OUTPUT_DIR / "stability_by_seed_summary.csv",
        index=False,
    )

    test_rows = []
    for perturbation, group in wide.groupby(PERTURBATION_COLUMN):
        differences = group["rf_minus_lr"].dropna().tolist()

        test_rows.append(
            {
                "perturbation": perturbation,
                "n_seeds": len(differences),
                "mean_rf_minus_lr": (
                    sum(differences) / len(differences)
                    if differences else float("nan")
                ),
                "exact_sign_flip_p": (
                    exact_sign_flip_test(differences)
                    if differences else float("nan")
                ),
                "lr_lower_change_all_seeds": (
                    all(value > 0 for value in differences)
                    if differences else False
                ),
            }
        )

    tests = pd.DataFrame(test_rows)
    tests.to_csv(
        OUTPUT_DIR / "paired_seed_statistical_tests.csv",
        index=False,
    )

    # Plot seed-level means with standard deviation across seeds.
    plot_data = summary.pivot(
        index=PERTURBATION_COLUMN,
        columns=MODEL_COLUMN,
        values="mean",
    )
    plot_data = plot_data.rename(
    columns={
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }
)
    plot_data = plot_data.rename(
    columns={
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }
    )
    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6),
        rot=25,
    )
    ax.set_title("Mean Relative SHAP Change Across Seeds")
    ax.set_xlabel("Perturbation scenario")
    ax.set_ylabel("Relative SHAP L1 change")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multi_seed_stability_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\n=== Paired seed-level differences ===")
    print(wide.to_string(index=False))
    print("\n=== Exact sign-flip tests ===")
    print(tests.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
