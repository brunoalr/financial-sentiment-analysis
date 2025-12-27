"""
Data loading functions
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from src.config import DATA_DIR, LABEL_MAPPING


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets from CSV files.

    Returns:
        tuple: (df_train, df_val, df_test) DataFrames with loaded data
    """
    df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
    df_val = pd.read_csv(f"{DATA_DIR}/val.csv")
    df_test = pd.read_csv(f"{DATA_DIR}/test.csv")

    return df_train, df_val, df_test


def encode_labels(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series] | Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Encode string labels to numeric values.

    Args:
        df_train: Training DataFrame with 'label' column
        df_val: Validation DataFrame with 'label' column
        df_test: Optional test DataFrame with 'label' column

    Returns:
        tuple: (y_train, y_val, y_test) encoded labels

    Raises:
        ValueError: If any labels cannot be mapped to numeric values
    """
    df_train["label_encoded"] = df_train["label"].map(LABEL_MAPPING)
    df_val["label_encoded"] = df_val["label"].map(LABEL_MAPPING)

    # Check for NaN values (unmapped labels)
    train_nan_count = df_train["label_encoded"].isna().sum()
    val_nan_count = df_val["label_encoded"].isna().sum()

    if train_nan_count > 0:
        unmapped_labels = df_train[df_train["label_encoded"].isna()]["label"].unique()
        raise ValueError(
            f"Found {train_nan_count} unmapped label(s) in training set: "
            f"{unmapped_labels}. Expected labels: "
            f"{list(LABEL_MAPPING.keys())}"
        )

    if val_nan_count > 0:
        unmapped_labels = df_val[df_val["label_encoded"].isna()]["label"].unique()
        raise ValueError(
            f"Found {val_nan_count} unmapped label(s) in validation set: "
            f"{unmapped_labels}. Expected labels: "
            f"{list(LABEL_MAPPING.keys())}"
        )

    y_train = df_train["label_encoded"]
    y_val = df_val["label_encoded"]

    if df_test is not None and "label" in df_test.columns:
        df_test["label_encoded"] = df_test["label"].map(LABEL_MAPPING)
        test_nan_count = df_test["label_encoded"].isna().sum()

        if test_nan_count > 0:
            unmapped_labels = df_test[df_test["label_encoded"].isna()]["label"].unique()
            raise ValueError(
                f"Found {test_nan_count} unmapped label(s) in test set: "
                f"{unmapped_labels}. Expected labels: "
                f"{list(LABEL_MAPPING.keys())}"
            )

        y_test = df_test["label_encoded"]
        return y_train, y_val, y_test

    return y_train, y_val
