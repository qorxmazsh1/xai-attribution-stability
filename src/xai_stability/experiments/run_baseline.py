
"""Run baseline model experiments on the Adult dataset."""

import json
from pathlib import Path

import pandas as pd

from xai_stability.data.load_data import load_adult_dataset
from xai_stability.data.preprocess import prepare_data
from xai_stability.models.evaluate import evaluate_model
from xai_stability.models.train import (
    train_logistic_regression,
    train_random_forest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


def main() -> None:
    """Train and evaluate baseline classifiers."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataset()
    splits = prepare_data(df)

    models = {
        "logistic_regression": train_logistic_regression(
            splits.X_train_processed,
            splits.y_train,
        ),
        "random_forest": train_random_forest(
            splits.X_train_processed,
            splits.y_train,
        ),
    }

    results = {}

    for name, model in models.items():
        metrics = evaluate_model(
            model,
            splits.X_test_processed,
            splits.y_test,
        )
        results[name] = metrics

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    output_path = RESULTS_DIR / "baseline_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    pd.DataFrame(results).T.to_csv(
        RESULTS_DIR / "baseline_metrics.csv",
        index=True,
    )

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
