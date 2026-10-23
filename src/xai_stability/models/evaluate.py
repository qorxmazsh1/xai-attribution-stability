
"""Evaluation utilities for binary classification models."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Evaluate a binary classifier using standard metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=">50K",
                            zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=">50K",
                         zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, pos_label=">50K",
                     zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(
                (y_test == ">50K").astype(int),
                probabilities,
            )
        ),
    }
