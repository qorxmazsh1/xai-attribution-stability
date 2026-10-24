
"""Controlled input perturbations for tabular ML experiments."""

import pandas as pd


def perturb_feature(
    X: pd.DataFrame,
    perturbation_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply a named perturbation and return changed rows' validity mask."""
    perturbations = {
        "age_plus_one_year": ("age", 1, 90),
        "hours_per_week_plus_one": ("hours_per_week", 1, 99),
        "education_num_plus_one": ("education_num", 1, 16),
        "hours_per_week_plus_five": ("hours_per_week", 5, 99),
    }

    
    if perturbation_name == "capital_gain_plus_ten_percent":
        X_perturbed = X.copy()
        values = pd.to_numeric(X["capital_gain"])
        valid_mask = values > 0

        X_perturbed["capital_gain"] = values.astype(float)
        X_perturbed.loc[valid_mask, "capital_gain"] = (
            values.loc[valid_mask].astype(float) * 1.10
        )

        return X_perturbed.loc[valid_mask].copy(), valid_mask


    if perturbation_name not in perturbations:
        raise ValueError(f"Unknown perturbation: {perturbation_name}")

    column, increment, upper_bound = perturbations[perturbation_name]
    values = pd.to_numeric(X[column])
    valid_mask = values <= upper_bound - increment

    X_perturbed = X.loc[valid_mask].copy()
    X_perturbed[column] = values.loc[valid_mask] + increment

    return X_perturbed, valid_mask
