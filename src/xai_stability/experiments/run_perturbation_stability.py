
"""Measure SHAP stability under controlled input perturbations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xai_stability.data.load_data import load_adult_dataset
from xai_stability.data.preprocess import prepare_data
from xai_stability.explainability.shap_explainer import (
    explain_tree_model,
)
from xai_stability.experiments.perturbations import perturb_feature
from xai_stability.models.train import train_random_forest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "stability"

RANDOM_STATE = 42
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 300

PERTURBATION_NAMES = [
    "age_plus_one_year",
    "hours_per_week_plus_one",
    "education_num_plus_one",
    "hours_per_week_plus_five",
    "capital_gain_plus_ten_percent",
]


def to_dense(matrix) -> np.ndarray:
    """Convert sparse or dense matrices to NumPy arrays."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def transform_raw(
    preprocessor,
    X: pd.DataFrame,
) -> pd.DataFrame:
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
    """Compare original and perturbed SHAP attribution vectors."""
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

    original_l1 = np.abs(original).sum()
    relative_l1_change = float(
        np.abs(perturbed - original).sum()
        / (original_l1 + eps)
    )

    return cosine, rank_correlation, relative_l1_change


def main() -> None:
    """Run controlled perturbation experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataset()
    splits = prepare_data(df)

    # Train once; keep the fitted model fixed for every perturbation.
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

    # Calculate original explanations and predictions once.
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

    # Map original row labels to their positions in the explanation arrays.
    original_position_by_index = {
        index: position
        for position, index in enumerate(X_original_raw.index)
    }

    perturbation_results = []

    for perturbation_name in PERTURBATION_NAMES:
        X_perturbed_raw, valid_mask = perturb_feature(
            X_original_raw,
            perturbation_name,
        )

        if X_perturbed_raw.empty:
            print(
                f"Skipping {perturbation_name}: "
                "no eligible rows."
            )
            continue

        X_perturbed = transform_raw(
            splits.preprocessor,
            X_perturbed_raw,
        )

        # Explain the perturbed rows without retraining the model.
        perturbed_shap, _ = explain_tree_model(
            model,
            background,
            X_perturbed,
        )

        transformed_array = X_perturbed.to_numpy()

        perturbed_probabilities = model.predict_proba(
            transformed_array
        )[:, positive_index]

        perturbed_predictions = model.predict(
            transformed_array
        )

        # Preserve exact alignment with the original SHAP arrays.
        original_positions = np.array(
            [
                original_position_by_index[index]
                for index in X_perturbed_raw.index
            ],
            dtype=int,
        )

        if len(original_positions) != len(X_perturbed_raw):
            raise RuntimeError(
                "Could not align perturbed rows with original rows."
            )

        if len(valid_mask) != len(X_original_raw):
            raise RuntimeError(
                "Perturbation mask length does not match input data."
            )

        changed_columns = {
            "age_plus_one_year": "age",
            "hours_per_week_plus_one": "hours_per_week",
            "education_num_plus_one": "education_num",
            "hours_per_week_plus_five": "hours_per_week",
            "capital_gain_plus_ten_percent": "capital_gain",
        }

        feature_changed = changed_columns[perturbation_name]

        for j, original_position in enumerate(original_positions):
            cosine, rank_corr, relative_l1 = calculate_similarity(
                original_shap[original_position],
                perturbed_shap[j],
            )

            original_prediction = original_predictions[
                original_position
            ]
            perturbed_prediction = perturbed_predictions[j]

            original_probability = original_probabilities[
                original_position
            ]
            perturbed_probability = perturbed_probabilities[j]

            perturbation_results.append(
                {
                    "sample_index": str(
                        X_original_raw.index[original_position]
                    ),
                    "model": "random_forest",
                    "perturbation": perturbation_name,
                    "feature_changed": feature_changed,
                    "original_prediction": str(
                        original_prediction
                    ),
                    "perturbed_prediction": str(
                        perturbed_prediction
                    ),
                    "prediction_unchanged": bool(
                        original_prediction == perturbed_prediction
                    ),
                    "original_probability": float(
                        original_probability
                    ),
                    "perturbed_probability": float(
                        perturbed_probability
                    ),
                    "absolute_probability_change": float(
                        abs(
                            perturbed_probability
                            - original_probability
                        )
                    ),
                    "shap_cosine_similarity": cosine,
                    "absolute_shap_rank_correlation": rank_corr,
                    "relative_shap_l1_change": relative_l1,
                }
            )

    if not perturbation_results:
        raise RuntimeError(
            "No perturbation results were generated."
        )

    results = pd.DataFrame(perturbation_results)

    results.to_csv(
        RESULTS_DIR / "perturbation_stability.csv",
        index=False,
    )

    summary = (
        results.groupby("perturbation")
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
        RESULTS_DIR / "perturbation_summary.csv",
        index=False,
    )

    metadata = {
        "model": "random_forest",
        "random_state": RANDOM_STATE,
        "background_size": background_size,
        "explain_size": explain_size,
        "model_refitted_during_perturbations": False,
        "shap_output_scale": "probability",
        "perturbations": PERTURBATION_NAMES,
        "valid_rows_per_perturbation": {
            name: int(
                (results["perturbation"] == name).sum()
            )
            for name in PERTURBATION_NAMES
        },
    }

    (
        RESULTS_DIR / "perturbation_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nPerturbation stability summary:")
    print(summary.to_string(index=False))

    print(f"\nDetailed results: {RESULTS_DIR / 'perturbation_stability.csv'}")
    print(f"Summary: {RESULTS_DIR / 'perturbation_summary.csv'}")
    print(f"Metadata: {RESULTS_DIR / 'perturbation_metadata.json'}")


if __name__ == "__main__":
    main()
