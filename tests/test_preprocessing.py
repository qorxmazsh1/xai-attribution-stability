
"""Tests for preprocessing and train/test splitting."""

import pandas as pd
import pytest

from xai_stability.data.preprocess import prepare_data


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create a dataset suitable for classification preprocessing."""
    rows = 20

    return pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "workclass": ["Private", "Self-emp"] * 10,
            "education_num": list(range(1, 21)),
            "income": ["<=50K", ">50K"] * 10,
        }
    )


def test_train_test_split_sizes(
    sample_dataset: pd.DataFrame,
) -> None:
    """The split should approximately follow the requested ratio."""
    splits = prepare_data(sample_dataset, test_size=0.2)

    assert len(splits.X_train) == 16
    assert len(splits.X_test) == 4
    assert splits.X_train_processed.shape[0] == 16
    assert splits.X_test_processed.shape[0] == 4


def test_preprocessing_feature_counts_match(
    sample_dataset: pd.DataFrame,
) -> None:
    """Train and test data should have the same encoded features."""
    splits = prepare_data(sample_dataset)

    assert (
        splits.X_train_processed.shape[1]
        == splits.X_test_processed.shape[1]
    )
    assert (
        splits.X_train_processed.shape[1]
        == len(splits.feature_names)
    )


def test_invalid_test_size_fails(
    sample_dataset: pd.DataFrame,
) -> None:
    """Invalid test proportions should raise ValueError."""
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        prepare_data(sample_dataset, test_size=1.0)


def test_missing_target_column_fails(
    sample_dataset: pd.DataFrame,
) -> None:
    """A missing target column should raise a clear error."""
    dataset = sample_dataset.drop(columns=["income"])

    with pytest.raises(ValueError):
        prepare_data(dataset)
