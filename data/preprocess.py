
"""Preprocessing utilities for tabular classification."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataSplits:
    """Hold raw train/test splits and a fitted preprocessor."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    X_train_processed: object
    X_test_processed: object
    feature_names: list[str]


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing transformations based on training columns.

    Args:
        X_train: Training features only.

    Returns:
        An unfitted ColumnTransformer.
    """
    numeric_columns = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def prepare_data(
    df: pd.DataFrame,
    target_column: str = "income",
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplits:
    """Split data and fit preprocessing using only training data.

    Args:
        df: Validated dataset.
        target_column: Name of the classification target.
        test_size: Proportion reserved for testing.
        random_state: Seed for reproducible splitting.

    Returns:
        Raw splits, processed matrices, and the fitted preprocessor.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    return DataSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        feature_names=feature_names,
    )
