"""
Load the Oxford Parkinson's Disease Detection Dataset and prepare it for
modeling.

Key design decision: each patient contributes ~6 recordings. If we split
randomly by ROW, recordings from the same patient can end up in both the
train and test sets, which leaks patient identity into the split and
inflates test accuracy. Instead we extract a `patient_id` from the `name`
column (e.g. "phon_R01_S01_1" -> "S01") and split by GROUP, so a given
patient's recordings are entirely in train or entirely in test.
"""

import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "data/parkinsons.data"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["patient_id"] = df["name"].apply(_extract_patient_id)
    return df


def _extract_patient_id(name: str) -> str:
    """'phon_R01_S01_1' -> 'S01' (drops the recording-number suffix)."""
    match = re.search(r"(S\d+)", name)
    return match.group(1) if match else name


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"name", "status", "patient_id"}
    return [c for c in df.columns if c not in exclude]


def grouped_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split so all recordings from one patient stay on one side."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["status"]
    groups = df["patient_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx],
    )


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} recordings from {df['patient_id'].nunique()} patients")
    print(df["status"].value_counts().rename({0: "healthy", 1: "PD"}))

    X_train, X_test, y_train, y_test, g_train, g_test = grouped_train_test_split(df)
    print(f"\nTrain: {len(X_train)} recordings, {g_train.nunique()} patients")
    print(f"Test:  {len(X_test)} recordings, {g_test.nunique()} patients")
    overlap = set(g_train) & set(g_test)
    print(f"Patient overlap between train/test: {len(overlap)} (should be 0)")
