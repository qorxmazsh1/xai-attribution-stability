
"""Evaluate SHAP stability across multiple random seeds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xai_stability.data.load_data import load_adult_dataset
from xai_stability.data.preprocess import prepare_data
from xai_stability.explainability.shap_explainer import (
    explain_linear_model,
    explain_tree_model,
)
from xai_stability.experiments.perturbations import perturb_feature
from xai_stability.models.train import (
    train_logistic_regression,
    train_random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "stability"

SEEDS = [42, 123, 2026, 7, 21]
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 300
POSITIVE_CLASS = ">50K"

PERTURBATIONS = {
    "age_plus_one_year": "age",
    "hours_per_week_plus_one": "hours_per_week",
    "education_num_plus_one": "education_num",
    "hours_per_week_plus_five": "hours_per_week",
    "capital_gain_plus_ten_percent": "capital_gain",
}


def to_dense(matrix) -> np.ndarray:
    """Convert sparse or dense matrices to NumPy arrays."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def transform_raw(preprocessor, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw features using the fitted preprocessor."""
    X_clean = X.replace({pd.NA: np.nan})
    transformed = to_dense(preprocessor.transform(X_clean))

    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )


def calculate_similarity(
    original: np.ndarray,
    perturbed: np.ndarray,
) -> tuple[float, float]:
    """Calculate SHAP cosine similarity and absolute-value rank correlation."""
    eps = 1e-12

    original_norm = np.linalg.norm(original)
    perturbed_norm = np.linalg.norm(perturbed)

    if original_norm < eps and perturbed_norm < eps:
        cosine = 1.0
    elif original_norm < eps or perturbed_norm < eps:
        cosine = 0.0
    else:
        cosine = float(
            np.dot(original, perturbed)
            / (original_norm * perturbed_norm)
        )

    original_ranks = pd.Series(np.abs(original)).rank()
    perturbed_ranks = pd.Series(np.abs(perturbed)).rank()

    if (
        original_ranks.nunique() < 2
        or perturbed_ranks.nunique() < 2
    ):
        rank_correlation = float("nan")
    else:
        rank_correlation = float(
            original_ranks.corr(
                perturbed_ranks,
                method="pearson",
            )
        )

    return cosine, rank_correlation


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    positive_class: str,
) -> dict:
    """Calculate baseline predictive performance."""
    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(positive_class)
    probabilities = model.predict_proba(X_test)[:, positive_index]
    binary_targets = (y_test.to_numpy() == positive_class).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                pos_label=positive_class,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                pos_label=positive_class,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                pos_label=positive_class,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(binary_targets, probabilities)
        ),
    }


def main() -> None:
    """Run perturbation experiments over multiple random seeds."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataset()
    all_results = []
    baseline_results = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"Running seed: {seed}")
        print(f"{'=' * 60}")

        splits = prepare_data(df, random_state=seed)

        rng = np.random.default_rng(seed)

        background_size = min(
            BACKGROUND_SIZE,
            splits.X_train_processed.shape[0],
        )
        explain_size = min(
            EXPLAIN_SIZE,
            splits.X_test_processed.shape[0],
        )

        background_indices = rng.choice(
            splits.X_train_processed.shape[0],
            size=background_size,
            replace=False,
        )

        feature_names = list(splits.feature_names)

        background = pd.DataFrame(
            to_dense(
                splits.X_train_processed[background_indices]
            ),
            columns=feature_names,
        )

        # Use a reproducible random test sample rather than the first rows.
        X_original_raw = splits.X_test.sample(
            n=explain_size,
            random_state=seed,
        ).copy()

        y_explain = splits.y_test.loc[X_original_raw.index]

        X_original = transform_raw(
            splits.preprocessor,
            X_original_raw,
        )

        models = {
            "random_forest": (
                train_random_forest(
                    splits.X_train_processed,
                    splits.y_train,
                    random_state=seed,
                ),
                explain_tree_model,
            ),
            "logistic_regression": (
                train_logistic_regression(
                    splits.X_train_processed,
                    splits.y_train,
                    random_state=seed,
                ),
                explain_linear_model,
            ),
        }

        for model_name, (model, explain_function) in models.items():
            print(f"\nModel: {model_name}")

            metrics = evaluate_model(
                model,
                splits.X_test_processed,
                splits.y_test,
                POSITIVE_CLASS,
            )

            baseline_results.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "train_size": len(splits.X_train),
                    "test_size": len(splits.X_test),
                    **metrics,
                }
            )

            original_shap, _ = explain_function(
                model,
                background,
                X_original,
                positive_class=POSITIVE_CLASS,
            )

            positive_index = list(model.classes_).index(
                POSITIVE_CLASS
            )

            original_array = X_original.to_numpy()
            original_probabilities = model.predict_proba(
                original_array
            )[:, positive_index]
            original_predictions = model.predict(original_array)

            for perturbation_name, feature_name in PERTURBATIONS.items():
                print(f"  Perturbation: {perturbation_name}")

                X_perturbed_raw, _ = perturb_feature(
                    X_original_raw,
                    perturbation_name,
                )

                if X_perturbed_raw.empty:
                    print("    Skipped: no valid rows.")
                    continue

                X_perturbed = transform_raw(
                    splits.preprocessor,
                    X_perturbed_raw,
                )

                perturbed_shap, _ = explain_function(
                    model,
                    background,
                    X_perturbed,
                    positive_class=POSITIVE_CLASS,
                )

                perturbed_array = X_perturbed.to_numpy()
                perturbed_probabilities = model.predict_proba(
                    perturbed_array
                )[:, positive_index]
                perturbed_predictions = model.predict(
                    perturbed_array
                )

                # Align every perturbed row with its original row.
                original_positions = X_original_raw.index.get_indexer(
                    X_perturbed_raw.index
                )

                for j, original_position in enumerate(
                    original_positions
                ):
                    original_vector = original_shap[original_position]
                    perturbed_vector = perturbed_shap[j]

                    cosine, rank_corr = calculate_similarity(
                        original_vector,
                        perturbed_vector,
                    )

                    relative_l1 = float(
                        np.abs(
                            perturbed_vector - original_vector
                        ).sum()
                        / (
                            np.abs(original_vector).sum()
                            + 1e-12
                        )
                    )

                    probability_change = float(
                        abs(
                            perturbed_probabilities[j]
                            - original_probabilities[original_position]
                        )
                    )

                    all_results.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "perturbation": perturbation_name,
                            "feature_changed": feature_name,
                            "sample_index": str(
                                X_perturbed_raw.index[j]
                            ),
                            "prediction_unchanged": bool(
                                original_predictions[original_position]
                                == perturbed_predictions[j]
                            ),
                            "absolute_probability_change": (
                                probability_change
                            ),
                            "shap_cosine_similarity": cosine,
                            "absolute_shap_rank_correlation": rank_corr,
                            "relative_shap_l1_change": relative_l1,
                        }
                    )

                print(f"    Valid rows: {len(X_perturbed_raw)}")

    results = pd.DataFrame(all_results)
    baseline_df = pd.DataFrame(baseline_results)

    if results.empty:
        raise RuntimeError("No perturbation results were generated.")

    # Save observation-level results.
    results.to_csv(
        RESULTS_DIR / "multi_seed_stability.csv",
        index=False,
    )

    # Summarize each seed independently.
    per_seed_summary = (
        results.groupby(
            ["seed", "model", "perturbation"],
            as_index=False,
        )
        .agg(
            n=("sample_index", "count"),
            prediction_unchanged_rate=(
                "prediction_unchanged",
                "mean",
            ),
            mean_abs_probability_change=(
                "absolute_probability_change",
                "mean",
            ),
            mean_shap_cosine=(
                "shap_cosine_similarity",
                "mean",
            ),
            mean_abs_shap_rank_correlation=(
                "absolute_shap_rank_correlation",
                "mean",
            ),
            mean_relative_shap_l1_change=(
                "relative_shap_l1_change",
                "mean",
            ),
        )
    )

    per_seed_summary.to_csv(
        RESULTS_DIR / "multi_seed_per_seed_summary.csv",
        index=False,
    )

    # Summarize how the per-seed metrics vary across seeds.
    metrics = [
        "prediction_unchanged_rate",
        "mean_abs_probability_change",
        "mean_shap_cosine",
        "mean_abs_shap_rank_correlation",
        "mean_relative_shap_l1_change",
    ]

    aggregation = {
        metric: ["mean", "std", "min", "max"]
        for metric in metrics
    }

    across_seed_summary = (
        per_seed_summary.groupby(
            ["model", "perturbation"]
        )
        .agg(aggregation)
    )

    across_seed_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in across_seed_summary.columns
    ]

    across_seed_summary = across_seed_summary.reset_index()
    across_seed_summary.to_csv(
        RESULTS_DIR / "multi_seed_across_seed_summary.csv",
        index=False,
    )

    baseline_df.to_csv(
        RESULTS_DIR / "multi_seed_baseline_metrics.csv",
        index=False,
    )

    metadata = {
        "seeds": SEEDS,
        "background_size": BACKGROUND_SIZE,
        "explain_size": EXPLAIN_SIZE,
        "positive_class": POSITIVE_CLASS,
        "models": list(models),
        "perturbations": list(PERTURBATIONS),
        "shap_output_scale": {
            "random_forest": "probability",
            "logistic_regression": "log_odds",
        },
        "notes": [
            "Each seed creates a new stratified train/test split.",
            "Preprocessing is fitted only on each seed's training data.",
            "Both models use the same split within a given seed.",
            "The background sample and explanation sample are seed-specific.",
            "Models are retrained for each seed, not for each perturbation.",
            "Across-seed standard deviations describe variation across five runs.",
            "SHAP output scales differ between the two models.",
            "Capital-gain perturbations may have smaller sample sizes.",
        ],
    }

    with (
        RESULTS_DIR / "multi_seed_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nAcross-seed stability summary:")
    print(across_seed_summary.to_string(index=False))

    print("\nBaseline predictive metrics:")
    print(baseline_df.to_string(index=False))

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
