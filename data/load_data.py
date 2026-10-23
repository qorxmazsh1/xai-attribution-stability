
"""Load and cache the UCI Adult dataset."""

from pathlib import Path
import re

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PATH = RAW_DATA_DIR / "adult.csv"


def normalize_column_name(name: str) -> str:
    """Normalize a column name to lowercase snake_case."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and missing-value markers."""
    df = df.copy()
    df.columns = [normalize_column_name(str(col)) for col in df.columns]

    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns after normalization.")

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace("?", pd.NA)

    if "income" in df.columns:
        df["income"] = df["income"].str.rstrip(".")

    return df


def load_adult_dataset(force_download: bool = False) -> pd.DataFrame:
    """Load the Adult dataset from a local CSV cache or UCI."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists() and not force_download:
        return clean_dataset(pd.read_csv(DATA_PATH))

    adult = fetch_ucirepo(id=2)
    features = adult.data.features
    targets = adult.data.targets

    if features is None or targets is None:
        raise RuntimeError("UCI did not return both features and targets.")

    df = pd.concat(
        [features.reset_index(drop=True), targets.reset_index(drop=True)],
        axis=1,
    )
    df = clean_dataset(df)

    if "income" not in df.columns:
        raise ValueError(
            f"Expected target 'income'. Found: {df.columns.tolist()}"
        )

    df.to_csv(DATA_PATH, index=False)
    return df
