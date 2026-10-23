
"""Run and save baseline SHAP explanations for Random Forest."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from xai_stability.data.load_data import load_adult_dataset
from xai_stability.data.preprocess import prepare_data
from xai_stability.models.train import train_random_forest
from xai_stability.explainability.shap_explainer import (
    explain_tree_model,
    get_mean_absolute_shap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "explanations"

RANDOM_STATE = 42
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 300


def to_dense(matrix) -> np.ndarray:
    """Convert a matrix to a dense NumPy array when necessary."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


def main() -> None:
    """Train the model and save baseline SHAP results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataset()
    splits = prepare_data(df)

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

    background = to_dense(
        splits.X_train_processed[background_indices]
    )
    X_explain = to_dense(
        splits.X_test_processed[:explain_size]
    )

    feature_names = list(splits.feature_names)

    background_df = pd.DataFrame(
        background,
        columns=feature_names,
    )
    explain_df = pd.DataFrame(
        X_explain,
        columns=feature_names,
    )

    shap_values, names = explain_tree_model(
        model=model,
        X_background=background_df,
        X_explain=explain_df,
    )

    importance = get_mean_absolute_shap(
        shap_values,
        names,
    )

    importance.to_csv(
        RESULTS_DIR / "random_forest_shap_importance.csv",
        index=False,
    )

    pd.DataFrame(
        shap_values,
        columns=names,
    ).to_csv(
        RESULTS_DIR / "random_forest_shap_values.csv",
        index=False,
    )

    metadata = {
        "model": "random_forest",
        "random_state": RANDOM_STATE,
        "background_size": background_size,
        "explain_size": explain_size,
        "n_features": len(names),
        "positive_class": ">50K",
        "model_output": "probability",
        "shap_version": shap.__version__,
    }

    (RESULTS_DIR / "shap_metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nTop 15 features by mean absolute SHAP:")
    print(importance.head(15).to_string(index=False))
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
