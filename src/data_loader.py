"""
Data loading functions
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd
from datasets import load_dataset

from src.config import DATA_DIR, LABEL_MAPPING, REVERSE_LABEL_MAPPING


def _is_huggingface_dataset(data_source: str) -> bool:
    """
    Determine if the data_source string is a Hugging Face dataset name.

    Args:
        data_source: String that could be a directory path or HF dataset name

    Returns:
        bool: True if it appears to be a Hugging Face dataset name,
            False otherwise
    """
    # Check if it's a valid local directory path first
    if os.path.exists(data_source) and os.path.isdir(data_source):
        return False

    # Hugging Face dataset names typically contain "/"
    # (e.g., "username/dataset-name") and don't exist as local paths
    if "/" in data_source and not os.path.exists(data_source):
        return True

    # If it doesn't exist as a path and contains "/", assume it's a HF dataset
    return "/" in data_source


def load_datasets(
    data_source: Optional[str] = None,
    label_mapping: Optional[Dict[int, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets from CSV files or
    Hugging Face datasets.

    Args:
        data_source: Optional string that can be either:
            - A directory path containing train.csv, val.csv, and test.csv
            - A Hugging Face dataset name (e.g., "username/dataset-name")
            - None (default): uses DATA_DIR from config
        label_mapping: Optional dictionary mapping numeric labels to string
            labels (e.g., {0: "neutral", 1: "positive", 2: "negative"}).
            If None, uses REVERSE_LABEL_MAPPING from config. Only used when
            loading Hugging Face datasets with numeric labels.

    Returns:
        tuple: (df_train, df_val, df_test) DataFrames with loaded data

    Raises:
        ValueError: If data_source is provided but is neither a valid
            directory nor a valid Hugging Face dataset, or if required
            splits are missing
        FileNotFoundError: If CSV files are missing from the specified
            directory
    """
    # Default to DATA_DIR if no data_source provided
    if data_source is None:
        data_source = DATA_DIR

    # Default to REVERSE_LABEL_MAPPING if no label_mapping provided
    if label_mapping is None:
        label_mapping = REVERSE_LABEL_MAPPING

    # Determine if it's a Hugging Face dataset or a local directory
    if _is_huggingface_dataset(data_source):
        return _load_huggingface_dataset(data_source, label_mapping)
    else:
        return _load_csv_datasets(data_source)


def _load_csv_datasets(
    data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load datasets from CSV files in a directory.

    Args:
        data_dir: Directory path containing train.csv, val.csv, and test.csv

    Returns:
        tuple: (df_train, df_val, df_test) DataFrames with loaded data

    Raises:
        FileNotFoundError: If any of the required CSV files are missing
        ValueError: If data_dir is not a valid directory
    """
    if not os.path.isdir(data_dir):
        raise ValueError(
            f"'{data_dir}' is not a valid directory. "
            f"Please provide a valid directory path or a Hugging Face "
            f"dataset name."
        )

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Check if files exist
    missing_files = []
    if not os.path.exists(train_path):
        missing_files.append("train.csv")
    if not os.path.exists(val_path):
        missing_files.append("val.csv")
    if not os.path.exists(test_path):
        missing_files.append("test.csv")

    if missing_files:
        raise FileNotFoundError(
            f"Missing required CSV files in '{data_dir}': "
            f"{', '.join(missing_files)}"
        )

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_val, df_test


def _load_huggingface_dataset(
    dataset_name: str,
    label_mapping: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load datasets from a Hugging Face dataset.

    Args:
        dataset_name: Hugging Face dataset name (e.g., "username/dataset-name")
        label_mapping: Dictionary mapping numeric labels to string labels
            (e.g., {0: "neutral", 1: "positive", 2: "negative"})

    Returns:
        tuple: (df_train, df_val, df_test) DataFrames with loaded data

    Raises:
        ValueError: If dataset cannot be loaded or required splits are missing
    """
    try:
        # Load the dataset
        dataset_dict = load_dataset(dataset_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load Hugging Face dataset '{dataset_name}': {str(e)}"
        ) from e

    # Get available splits
    available_splits = list(dataset_dict.keys())

    # Try to find train, validation/val, and test splits
    train_split = None
    val_split = None
    test_split = None

    # Look for train split
    if "train" in available_splits:
        train_split = "train"
    elif len(available_splits) == 1:
        # If only one split exists, assume it's the train split
        train_split = available_splits[0]
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' does not have a 'train' split. "
            f"Available splits: {available_splits}"
        )

    # Look for validation split (try both 'validation' and 'val')
    if "validation" in available_splits:
        val_split = "validation"
    elif "val" in available_splits:
        val_split = "val"

    # Look for test split
    if "test" in available_splits:
        test_split = "test"

    # If we only have train split, we need to split it
    if val_split is None or test_split is None:
        if len(dataset_dict[train_split]) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' train split is empty. "
                f"Cannot create train/val/test splits."
            )

        # Split the train dataset into train/val/test (80/10/10)
        train_dataset = dataset_dict[train_split]
        split_dataset = train_dataset.train_test_split(
            test_size=0.2, seed=42, shuffle=True
        )
        train_data = split_dataset["train"]
        remaining_data = split_dataset["test"]

        # Split remaining into val and test (50/50 of the 20%, so 10% each)
        val_test_split = remaining_data.train_test_split(
            test_size=0.5, seed=42, shuffle=True
        )
        val_data = val_test_split["train"]
        test_data = val_test_split["test"]

        df_train = train_data.to_pandas()
        df_val = val_data.to_pandas()
        df_test = test_data.to_pandas()
    else:
        # Use existing splits
        df_train = dataset_dict[train_split].to_pandas()
        df_val = dataset_dict[val_split].to_pandas()
        df_test = dataset_dict[test_split].to_pandas()

    # Ensure column names match expected format (text, label)
    # Hugging Face datasets might use different column names
    # We'll keep the original columns but note that downstream code expects
    # 'text' and 'label'. If the dataset uses different names, the user will
    # need to rename them or we could add automatic mapping, but for now we'll
    # keep it simple and let the user handle it

    # Convert numeric labels to string labels if needed
    for df in [df_train, df_val, df_test]:
        if "label" in df.columns:
            # Check if labels are numeric
            if pd.api.types.is_numeric_dtype(df["label"]):
                # Store original values for error reporting
                original_labels = df["label"].copy()
                # Convert numeric labels to string labels
                df["label"] = df["label"].map(label_mapping)
                # Check for any unmapped values
                if df["label"].isna().any():
                    unmapped = original_labels[df["label"].isna()].unique()
                    raise ValueError(
                        f"Found unmapped numeric label(s): {unmapped}. "
                        f"Expected values: "
                        f"{list(label_mapping.keys())}"
                    )

    return df_train, df_val, df_test


def encode_labels(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series] | Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Encode string labels to numeric values, or use numeric labels directly.

    Handles both string labels (e.g., 'neutral', 'positive', 'negative') and
    numeric labels (e.g., 0, 1, 2). If labels are already numeric and in the
    valid range [0, 1, 2], they are used directly. Otherwise, string labels
    are mapped using LABEL_MAPPING.

    Args:
        df_train: Training DataFrame with 'label' column
        df_val: Validation DataFrame with 'label' column
        df_test: Optional test DataFrame with 'label' column

    Returns:
        tuple: (y_train, y_val, y_test) encoded labels as numeric values

    Raises:
        ValueError: If any labels cannot be mapped to numeric values or are
            outside the valid range [0, 1, 2]
    """
    # Check if labels are already numeric
    train_labels = df_train["label"]
    val_labels = df_val["label"]

    # Determine if labels are numeric
    train_is_numeric = pd.api.types.is_numeric_dtype(train_labels)
    val_is_numeric = pd.api.types.is_numeric_dtype(val_labels)

    # If labels are numeric, validate and use directly
    if train_is_numeric:
        train_unique = set(train_labels.unique())
        valid_numeric = {0, 1, 2}
        if not train_unique.issubset(valid_numeric):
            invalid = train_unique - valid_numeric
            raise ValueError(
                f"Found invalid numeric label(s) in training set: {invalid}. "
                f"Expected values: {valid_numeric}"
            )
        df_train["label_encoded"] = train_labels.astype(int)
    else:
        # Map string labels to numeric
        df_train["label_encoded"] = df_train["label"].map(LABEL_MAPPING)
        train_nan_count = df_train["label_encoded"].isna().sum()
        if train_nan_count > 0:
            unmapped_labels = (
                df_train[df_train["label_encoded"].isna()]["label"].unique()
            )
            raise ValueError(
                f"Found {train_nan_count} unmapped label(s) in training set: "
                f"{unmapped_labels}. Expected labels: "
                f"{list(LABEL_MAPPING.keys())}"
            )

    if val_is_numeric:
        val_unique = set(val_labels.unique())
        valid_numeric = {0, 1, 2}
        if not val_unique.issubset(valid_numeric):
            invalid = val_unique - valid_numeric
            raise ValueError(
                f"Found invalid numeric label(s) in validation set: "
                f"{invalid}. Expected values: {valid_numeric}"
            )
        df_val["label_encoded"] = val_labels.astype(int)
    else:
        # Map string labels to numeric
        df_val["label_encoded"] = df_val["label"].map(LABEL_MAPPING)
        val_nan_count = df_val["label_encoded"].isna().sum()
        if val_nan_count > 0:
            unmapped_labels = (
                df_val[df_val["label_encoded"].isna()]["label"].unique()
            )
            raise ValueError(
                f"Found {val_nan_count} unmapped label(s) in validation set: "
                f"{unmapped_labels}. Expected labels: "
                f"{list(LABEL_MAPPING.keys())}"
            )

    y_train = df_train["label_encoded"]
    y_val = df_val["label_encoded"]

    if df_test is not None:
        if "label" not in df_test.columns:
            # Populate with dummy values (zeros) if label column is missing
            y_test = pd.Series([0] * len(df_test), index=df_test.index)
            return y_train, y_val, y_test

        test_labels = df_test["label"]
        test_is_numeric = pd.api.types.is_numeric_dtype(test_labels)

        if test_is_numeric:
            test_unique = set(test_labels.unique())
            valid_numeric = {0, 1, 2}
            if not test_unique.issubset(valid_numeric):
                invalid = test_unique - valid_numeric
                raise ValueError(
                    f"Found invalid numeric label(s) in test set: {invalid}. "
                    f"Expected values: {valid_numeric}"
                )
            df_test["label_encoded"] = test_labels.astype(int)
        else:
            df_test["label_encoded"] = df_test["label"].map(LABEL_MAPPING)
            test_nan_count = df_test["label_encoded"].isna().sum()
            if test_nan_count > 0:
                unmapped_labels = (
                    df_test[df_test["label_encoded"].isna()]["label"].unique()
                )
                raise ValueError(
                    f"Found {test_nan_count} unmapped label(s) in test set: "
                    f"{unmapped_labels}. Expected labels: "
                    f"{list(LABEL_MAPPING.keys())}"
                )

        y_test = df_test["label_encoded"]
        return y_train, y_val, y_test

    return y_train, y_val
