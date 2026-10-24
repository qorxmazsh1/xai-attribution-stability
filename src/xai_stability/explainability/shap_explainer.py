
"""SHAP explanation utilities for tree-based classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def explain_tree_model(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    positive_class: str = ">50K",
) -> tuple[np.ndarray, list[str]]:
    """Calculate SHAP values for a selected classification class.

    Args:
        model: A fitted tree-based classifier.
        X_background: Background samples for SHAP.
        X_explain: Samples to explain.
        positive_class: Class whose predicted probability is explained.

    Returns:
        SHAP values with shape (samples, features), and feature names.
    """
    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError(
            f"Class {positive_class!r} not found in {classes!r}."
        )

    positive_index = classes.index(positive_class)

    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional",
        model_output="probability",
    )

    explanation = explainer(X_explain)
    values = np.asarray(explanation.values)

    if values.ndim == 3:
        values = values[:, :, positive_index]
    elif values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if values.shape[1] != X_explain.shape[1]:
        raise ValueError(
            "SHAP feature count does not match input feature count."
        )

    return values, list(X_explain.columns)


def explain_linear_model(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    positive_class: str = ">50K",
) -> tuple[np.ndarray, list[str]]:
    """Calculate SHAP values for a linear classifier.

    For binary Logistic Regression, SHAP values explain the
    decision score (log-odds), not the predicted probability.
    """
    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError(
            f"Class {positive_class!r} not found in {classes!r}."
        )

    positive_index = classes.index(positive_class)

    explainer = shap.LinearExplainer(
        model,
        X_background,
    )

    explanation = explainer(X_explain)
    values = np.asarray(explanation.values)

    if values.ndim == 3:
        if (
            values.shape[1] == X_explain.shape[1]
            and values.shape[2] == len(classes)
        ):
            values = values[:, :, positive_index]
        elif (
            values.shape[1] == len(classes)
            and values.shape[2] == X_explain.shape[1]
        ):
            values = values[:, positive_index, :]
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    elif values.ndim == 2:
        # Binary sklearn Logistic Regression explains the
        # decision score for classes_[1] by default.
        if len(classes) == 2 and positive_index == 0:
            values = -values
        elif len(classes) != 2:
            raise ValueError(
                "Unexpected two-dimensional SHAP output "
                "for a multiclass model."
            )
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if values.shape[1] != X_explain.shape[1]:
        raise ValueError(
            "SHAP feature count does not match input feature count."
        )

    return values, list(X_explain.columns)


def get_mean_absolute_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value."""
    importance = np.abs(shap_values).mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": importance,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
