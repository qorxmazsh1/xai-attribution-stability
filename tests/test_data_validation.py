
"""Tests for dataset validation."""

import pandas as pd
import pytest

from xai_stability.data.validate_data import validate_dataset


@pytest.fixture
def valid_dataset() -> pd.DataFrame:
    """Create a small dataset with the expected columns."""
    return pd.DataFrame(
        {
            "age": [25, 40, 35],
            "workclass": ["Private", "Private", "Self-emp"],
            "fnlwgt": [100, 200, 300],
            "education": ["Bachelors", "HS-grad", "Masters"],
            "education_num": [13, 9, 14],
            "marital_status": [
                "Never-married",
                "Married",
                "Divorced",
            ],
            "occupation": ["Tech", "Sales", "Tech"],
            "relationship": ["Not-in-family", "Husband", "Unmarried"],
            "race": ["White", "White", "Black"],
            "sex": ["Male", "Female", "Male"],
            "capital_gain": [0, 100, 0],
            "capital_loss": [0, 0, 50],
            "hours_per_week": [40, 45, 35],
            "native_country": ["United-States"] * 3,
            "income": ["<=50K", ">50K", "<=50K"],
        }
    )


def test_valid_dataset_passes(valid_dataset: pd.DataFrame) -> None:
    """Valid data should produce a validation report."""
    report = validate_dataset(valid_dataset)

    assert report["rows"] == 3
    assert report["columns"] == 15
    assert report["duplicate_rows"] == 0


def test_empty_dataset_fails() -> None:
    """An empty dataset should raise ValueError."""
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset(pd.DataFrame())


def test_missing_expected_column_fails(
    valid_dataset: pd.DataFrame,
) -> None:
    """Missing required columns should raise ValueError."""
    dataset = valid_dataset.drop(columns=["age"])

    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_dataset(dataset)


def test_missing_target_value_fails(
    valid_dataset: pd.DataFrame,
) -> None:
    """Missing target values should raise ValueError."""
    dataset = valid_dataset.copy()
    dataset.loc[0, "income"] = None

    with pytest.raises(ValueError, match="Target contains missing values"):
        validate_dataset(dataset)


def test_single_class_target_fails(
    valid_dataset: pd.DataFrame,
) -> None:
    """A target with one class should raise ValueError."""
    dataset = valid_dataset.copy()
    dataset["income"] = "<=50K"

    with pytest.raises(ValueError, match="at least 2 classes"):
        validate_dataset(dataset)
