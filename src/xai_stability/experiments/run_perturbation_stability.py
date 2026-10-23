
"""Measure SHAP stability under small input perturbations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xai_stability.data.load_data import load_adult_dataset
from xai_stability.data.preprocess import prepare_data
from xai_stability.models.train import train_random_forest
from xai_stability.explainability.shap_explainer import (
    explain_tree_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "stability"

RANDOM_STATE = 42
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 300


def to_dense(matrix) -> np.ndarray:
    """Convert sparse or dense matrices to NumPy arrays."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def transform_raw(preprocessor, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw data using the already-fitted preprocessor."""
    X_clean = X.replace({pd.NA: np.nan})
    transformed = preprocessor.transform(X_clean)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )


def calculate_similarity(
    original: np.ndarray,
    perturbed: np.ndarray,
) -> tuple[float, float, float]:
    """Compare two attribution vectors."""
    eps = 1e-12

    cosine = float(
        np.dot(original, perturbed)
        / (
            np.linalg.norm(original)
            * np.linalg.norm(perturbed)
            + eps
        )
    )

    original_ranks = pd.Series(np.abs(original)).rank().to_numpy()
    perturbed_ranks = pd.Series(np.abs(perturbed)).rank().to_numpy()

    rank_correlation = float(
        pd.Series(original_ranks).corr(
            pd.Series(perturbed_ranks),
            method="pearson",
        )
    )

    relative_l1_change = float(
        np.abs(perturbed - original).sum()
        / (np.abs(original).sum() + eps)
    )

    return cosine, rank_correlation, relative_l1_change


def main() -> None:
    """Run controlled perturbation experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataset()
    splits = prepare_data(df)

    # Train once. Keep this exact fitted model fixed for all comparisons.
    model = train_random_forest(
        splits.X_train_processed,
        splits.y_train,
        random_state=RANDOM_STATE,
    )

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
        to_dense(splits.X_train_processed[background_indices]),
        columns=feature_names,
    )

    X_original_raw = splits.X_test.iloc[:explain_size].copy()
    X_original = pd.DataFrame(
        to_dense(splits.X_test_processed[:explain_size]),
        columns=feature_names,
        index=X_original_raw.index,
    )

    # Explain the original examples once.
    original_shap, _ = explain_tree_model(
        model,
        background,
        X_original,
    )

    positive_index = list(model.classes_).index(">50K")
    original_probabilities = model.predict_proba(
        splits.X_test_processed[:explain_size]
    )[:, positive_index]
    original_predictions = model.predict(
        splits.X_test_processed[:explain_size]
    )

    perturbation_results = []

    perturbations = {
        "age_plus_one_year": ("age", 90),
        "hours_per_week_plus_one": ("hours_per_week", 99),
    }

    for perturbation_name, (column, upper_bound) in perturbations.items():
        # Apply the positive perturbation only where it stays in bounds.
        values = pd.to_numeric(X_original_raw[column])
        valid_mask = values < upper_bound

        X_perturbed_raw = X_original_raw.loc[valid_mask].copy()
        if X_perturbed_raw.empty:
            continue

        X_perturbed_raw[column] = values.loc[valid_mask] + 1

        X_perturbed = transform_raw(
            splits.preprocessor,
            X_perturbed_raw,
        )

        perturbed_shap, _ = explain_tree_model(
            model,
            background,
            X_perturbed,
        )

        transformed_array = X_perturbed.to_numpy()
        perturbed_probabilities = model.predict_proba(
            transformed_array
        )[:, positive_index]
        perturbed_predictions = model.predict(transformed_array)

        original_positions = np.flatnonzero(valid_mask.to_numpy())

        for j, original_position in enumerate(original_positions):
            cosine, rank_corr, relative_l1 = calculate_similarity(
                original_shap[original_position],
                perturbed_shap[j],
            )

            perturbation_results.append(
                {
                    "sample_index": str(
                        X_original_raw.index[original_position]
                    ),
                    "perturbation": perturbation_name,
                    "feature_changed": column,
                    "original_prediction": str(
                        original_predictions[original_position]
                    ),
                    "perturbed_prediction": str(perturbed_predictions[j]),
                    "prediction_unchanged": bool(
                        original_predictions[original_position]
                        == perturbed_predictions[j]
                    ),
                    "original_probability": float(
                        original_probabilities[original_position]
                    ),
                    "perturbed_probability": float(
                        perturbed_probabilities[j]
                    ),
                    "absolute_probability_change": float(
                        abs(
                            perturbed_probabilities[j]
                            - original_probabilities[original_position]
                        )
                    ),
                    "shap_cosine_similarity": cosine,
                    "absolute_shap_rank_correlation": rank_corr,
                    "relative_shap_l1_change": relative_l1,
                }
            )

    results = pd.DataFrame(perturbation_results)

    results = pd.DataFrame(perturbation_results)

    results.to_csv(
        RESULTS_DIR / "perturbation_stability.csv",
        index=False,
    )

    summary = (
        results.groupby("perturbation")
        .agg(
            n=("sample_index", "count"),
            prediction_unchanged_rate=("prediction_unchanged", "mean"),
            mean_abs_probability_change=(
                "absolute_probability_change",
                "mean",
            ),
            mean_shap_cosine=("shap_cosine_similarity", "mean"),
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
        RESULTS_DIR / "perturbation_summary.csv",
        index=False,
    )

    metadata = {
        "model": "random_forest",
        "random_state": RANDOM_STATE,
        "background_size": background_size,
        "explain_size": explain_size,
        "model_refitted_during_perturbations": False,
        "perturbations": list(perturbations),
    }

    (RESULTS_DIR / "perturbation_metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nPerturbation stability summary:")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
