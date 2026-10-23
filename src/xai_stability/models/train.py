
"""Model training utilities for baseline experiments."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train,
    y_train,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a Logistic Regression baseline."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train,
    y_train,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest baseline."""
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
