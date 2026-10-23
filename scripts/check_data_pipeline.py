
"""Run basic checks for the data pipeline."""

from xai_stability.data.load_data import load_adult_dataset
from xai_stability.data.preprocess import prepare_data
from xai_stability.data.validate_data import validate_dataset


def main() -> None:
    """Load, validate, and preprocess the Adult dataset."""
    df = load_adult_dataset()

    report = validate_dataset(df)

    print("Dataset validation passed.")
    print(f"Rows: {report['rows']}")
    print(f"Columns: {report['columns']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print("Target distribution:")
    print(report["target_distribution"])

    splits = prepare_data(df)

    print(f"\nTraining rows: {len(splits.X_train)}")
    print(f"Testing rows: {len(splits.X_test)}")
    print(f"Processed training shape: {splits.X_train_processed.shape}")
    print(f"Processed testing shape: {splits.X_test_processed.shape}")
    print(f"Encoded feature count: {len(splits.feature_names)}")


if __name__ == "__main__":
    main()
