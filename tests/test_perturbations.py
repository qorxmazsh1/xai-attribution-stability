
"""Tests for controlled feature perturbations."""

import pandas as pd
import pytest

from xai_stability.experiments.perturbations import perturb_feature


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create rows for testing perturbation rules."""
    return pd.DataFrame(
        {
            "age": [30, 90],
            "hours_per_week": [40, 98],
            "education_num": [10, 16],
            "capital_gain": [0, 1000],
        }
    )


def test_age_increases_by_one(sample_data: pd.DataFrame) -> None:
    """Age should increase by one without exceeding 90."""
    result, mask = perturb_feature(sample_data, "age_plus_one_year")

    assert result["age"].tolist() == [31]
    assert mask.tolist() == [True, False]



def test_hours_increase_by_five(sample_data: pd.DataFrame) -> None:
    """Hours should increase by five without exceeding 99."""
    result, mask = perturb_feature(
        sample_data,
        "hours_per_week_plus_five",
    )

    assert result["hours_per_week"].tolist() == [45]
    assert mask.tolist() == [True, False]


def test_education_num_increases_by_one(
    sample_data: pd.DataFrame,
) -> None:
    """Education number should not exceed 16."""
    result, mask = perturb_feature(
        sample_data,
        "education_num_plus_one",
    )

    assert result["education_num"].tolist() == [11]
    assert mask.tolist() == [True, False]


def test_capital_gain_increases_by_ten_percent(
    sample_data: pd.DataFrame,
) -> None:
    """Only positive capital gains should be increased."""
    result, mask = perturb_feature(
        sample_data,
        "capital_gain_plus_ten_percent",
    )

    assert result["capital_gain"].tolist() == [1100]
    assert mask.tolist() == [False, True]


def test_unknown_perturbation_fails(
    sample_data: pd.DataFrame,
) -> None:
    """Unknown perturbation names should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturb_feature(sample_data, "unknown_feature")
