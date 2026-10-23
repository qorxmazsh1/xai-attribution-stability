
"""Validation utilities for tabular classification datasets."""

import pandas as pd


EXPECTED_COLUMNS = {
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
}


def validate_dataset(
    df: pd.DataFrame,
    target_column: str = "income",
) -> dict[str, object]:
    """Validate dataset structure and return a diagnostic report.

    Args:
        df: Dataset containing features and target.
        target_column: Name of the classification target.

    Returns:
        A dictionary with row counts, missing values and target distribution.

    Raises:
        ValueError: If essential structural or target checks fail.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    if not df.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    missing_columns = EXPECTED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    if df[target_column].isna().any():
        raise ValueError("Target contains missing values.")

    if df[target_column].nunique() < 2:
        raise ValueError("Classification target must contain at least 2 classes.")

    features = df.drop(columns=[target_column])

    report: dict[str, object] = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "target_distribution": (
            df[target_column].value_counts(dropna=False).to_dict()
        ),
        "numeric_columns": features.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": features.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist(),
    }

    return report
