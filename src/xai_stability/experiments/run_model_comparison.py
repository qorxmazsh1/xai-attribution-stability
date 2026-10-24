
"""Compare SHAP stability across baseline models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xai_stability.data.load_data import load_adult_dataset
from xai_stability.data.preprocess import prepare_data
from xai_stability.explainability.shap_explainer import (
    explain_linear_model,
    explain_tree_model,
)
from xai_stability.models.train import (
    train_logistic_regression,
    train_random_forest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "stability"

RANDOM_STATE = 42
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 300
POSITIVE_CLASS = ">50K"

PERTURBATIONS = {
    "age_plus_one_year": ("age", 90),
    "hours_per_week_plus_one": ("hours_per_week", 99),
}


def to_dense(matrix) -> np.ndarray:
    """Convert sparse or dense matrices to NumPy arrays."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def transform_raw(preprocessor, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw data with the fitted preprocessor."""
    X_clean = X.replace({pd.NA: np.nan})
    transformed = preprocessor.transform(X_clean)
    transformed = to_dense(transformed)

    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )


def calculate_similarity(
    original: np.ndarray,
    perturbed: np.ndarray,
) -> tuple[float, float]:
    """Calculate cosine similarity and absolute-SHAP rank correlation."""
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


def main() -> None:
    """Run the same perturbations against both fitted models."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataset()
    splits = prepare_data(df)

    rng = np.random.default_rng(RANDOM_STATE)

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

    X_original_raw = splits.X_test.iloc[:explain_size].copy()

    X_original = pd.DataFrame(
        to_dense(splits.X_test_processed[:explain_size]),
        columns=feature_names,
        index=X_original_raw.index,
    )

    models = {
        "random_forest": (
            train_random_forest(
                splits.X_train_processed,
                splits.y_train,
                random_state=RANDOM_STATE,
            ),
            explain_tree_model,
        ),
        "logistic_regression": (
            train_logistic_regression(
                splits.X_train_processed,
                splits.y_train,
                random_state=RANDOM_STATE,
            ),
            explain_linear_model,
        ),
    }

    all_results = []

    for model_name, (model, explain_function) in models.items():
        print(f"\nRunning model: {model_name}")

        original_shap, _ = explain_function(
            model,
            background,
            X_original,
            positive_class=POSITIVE_CLASS,
        )

        classes = list(model.classes_)
        positive_index = classes.index(POSITIVE_CLASS)

        original_probabilities = model.predict_proba(
            X_original.to_numpy()
        )[:, positive_index]

        original_predictions = model.predict(
            X_original.to_numpy()
        )

        for perturbation_name, (column, upper_bound) in (
            PERTURBATIONS.items()
        ):
            values = pd.to_numeric(X_original_raw[column])
            valid_mask = values < upper_bound

            X_perturbed_raw = X_original_raw.loc[
                valid_mask
            ].copy()

            if X_perturbed_raw.empty:
                continue

            X_perturbed_raw[column] = (
                values.loc[valid_mask] + 1
            )

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

            X_perturbed_array = X_perturbed.to_numpy()

            perturbed_probabilities = model.predict_proba(
                X_perturbed_array
            )[:, positive_index]

            perturbed_predictions = model.predict(
                X_perturbed_array
            )

            original_positions = np.flatnonzero(
                valid_mask.to_numpy()
            )

            for j, original_position in enumerate(
                original_positions
            ):
                cosine, rank_corr = calculate_similarity(
                    original_shap[original_position],
                    perturbed_shap[j],
                )

                original_vector = original_shap[
                    original_position
                ]
                perturbed_vector = perturbed_shap[j]

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
                        - original_probabilities[
                            original_position
                        ]
                    )
                )

                all_results.append(
                    {
                        "model": model_name,
                        "perturbation": perturbation_name,
                        "feature_changed": column,
                        "sample_index": str(
                            X_original_raw.index[
                                original_position
                            ]
                        ),
                        "prediction_unchanged": bool(
                            original_predictions[
                                original_position
                            ]
                            == perturbed_predictions[j]
                        ),
                        "absolute_probability_change": (
                            probability_change
                        ),
                        "shap_cosine_similarity": cosine,
                        "absolute_shap_rank_correlation": (
                            rank_corr
                        ),
                        "relative_shap_l1_change": relative_l1,
                    }
                )

    results = pd.DataFrame(all_results)

    if results.empty:
        raise RuntimeError(
            "No perturbation results were generated."
        )

    results.to_csv(
        RESULTS_DIR / "model_comparison_stability.csv",
        index=False,
    )

    summary = (
        results.groupby(["model", "perturbation"])
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
        .reset_index()
    )

    summary.to_csv(
        RESULTS_DIR / "model_comparison_summary.csv",
        index=False,
    )

    metadata = {
        "random_state": RANDOM_STATE,
        "background_size": background_size,
        "explain_size": explain_size,
        "positive_class": POSITIVE_CLASS,
        "model_refitted_during_perturbations": False,
        "models": list(models),
        "perturbations": list(PERTURBATIONS),
        "shap_output_scale": {
            "random_forest": "probability",
            "logistic_regression": "log_odds",
        },
    }

    (
        RESULTS_DIR / "model_comparison_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nModel comparison summary:")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
