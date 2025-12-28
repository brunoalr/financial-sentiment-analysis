"""
Data loading functions
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd
from datasets import load_dataset

from src.config import DATA_DIR, LABEL_MAPPING, REVERSE_LABEL_MAPPING

# Valid numeric label values
VALID_NUMERIC_LABELS = {0, 1, 2}


def _is_huggingface_dataset(data_source: str) -> bool:
    """
    Determine if the data_source string is a Hugging Face dataset name.

    Args:
        data_source: String that could be a directory path or HF dataset name

    Returns:
        bool: True if it appears to be a Hugging Face dataset name,
            False otherwise
    """
    # If it's a valid local directory path, it's not a HF dataset
    if os.path.exists(data_source) and os.path.isdir(data_source):
        return False

    # Hugging Face dataset names typically contain "/"
    # (e.g., "username/dataset-name") and don't exist as local paths
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

    # else
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


def _convert_numeric_labels_to_string(
    df: pd.DataFrame, label_mapping: Dict[int, str]
) -> None:
    """
    Convert numeric labels to string labels in-place if needed.

    Args:
        df: DataFrame with 'label' column to convert
        label_mapping: Dictionary mapping numeric labels to string labels

    Raises:
        ValueError: If any numeric labels cannot be mapped
    """
    if "label" not in df.columns:
        return

    if not pd.api.types.is_numeric_dtype(df["label"]):
        return

    # Store original values for error reporting
    original_labels = df["label"].copy()
    # Convert numeric labels to string labels
    df["label"] = df["label"].map(label_mapping)
    # Check for any unmapped values
    if df["label"].isna().any():
        unmapped = original_labels[df["label"].isna()].unique()
        raise ValueError(
            f"Found unmapped numeric label(s): {unmapped}. "
            f"Expected values: {list(label_mapping.keys())}"
        )


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

    # Find train split
    if "train" in available_splits:
        train_split = "train"
    elif len(available_splits) == 1:
        train_split = available_splits[0]
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' does not have a 'train' split. "
            f"Available splits: {available_splits}"
        )

    # Find validation and test splits (try preferred names first)
    val_split = next(
        (s for s in ["validation", "val"] if s in available_splits), None
    )
    test_split = "test" if "test" in available_splits else None

    # Split dataset if needed (80/10/10)
    if val_split is None or test_split is None:
        train_dataset = dataset_dict[train_split]
        if len(train_dataset) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' train split is empty. "
                f"Cannot create train/val/test splits."
            )

        # Split train into train/remaining (80/20)
        split_1 = train_dataset.train_test_split(
            test_size=0.2, seed=42, shuffle=True
        )
        # Split remaining into val/test (50/50 of 20%, so 10% each)
        split_2 = split_1["test"].train_test_split(
            test_size=0.5, seed=42, shuffle=True
        )

        df_train = split_1["train"].to_pandas()
        df_val = split_2["train"].to_pandas()
        df_test = split_2["test"].to_pandas()
    else:
        # Use existing splits
        df_train = dataset_dict[train_split].to_pandas()
        df_val = dataset_dict[val_split].to_pandas()
        df_test = dataset_dict[test_split].to_pandas()

    # Convert numeric labels to string labels if needed
    for df in [df_train, df_val, df_test]:
        _convert_numeric_labels_to_string(df, label_mapping)

    return df_train, df_val, df_test


def _encode_single_label_column(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    label_col: str = "label",
) -> pd.Series:
    """
    Encode a single label column to numeric values.

    Handles both string labels (e.g., 'neutral', 'positive', 'negative') and
    numeric labels (e.g., 0, 1, 2). If labels are already numeric and in the
    valid range [0, 1, 2], they are used directly. Otherwise, string labels
    are mapped using LABEL_MAPPING.

    Args:
        df: DataFrame with label column to encode
        dataset_name: Name of the dataset (for error messages)
        label_col: Name of the label column (default: "label")

    Returns:
        pd.Series: Encoded labels as numeric values

    Raises:
        ValueError: If any labels cannot be mapped to numeric values or are
            outside the valid range [0, 1, 2]
    """
    labels = df[label_col]

    # Check if labels are numeric
    if pd.api.types.is_numeric_dtype(labels):
        # Validate numeric labels are in valid range
        unique_labels = set(labels.unique())
        if not unique_labels.issubset(VALID_NUMERIC_LABELS):
            invalid = unique_labels - VALID_NUMERIC_LABELS
            raise ValueError(
                f"Found invalid numeric label(s) in {dataset_name}: "
                f"{invalid}. Expected values: {VALID_NUMERIC_LABELS}"
            )
        return labels.astype(int)

    # else
    # Map string labels to numeric
    encoded = df[label_col].map(LABEL_MAPPING)
    nan_count = encoded.isna().sum()
    if nan_count > 0:
        unmapped_labels = df[encoded.isna()][label_col].unique()
        raise ValueError(
            f"Found {nan_count} unmapped label(s) in {dataset_name}: "
            f"{unmapped_labels}. Expected labels: "
            f"{list(LABEL_MAPPING.keys())}"
        )
    return encoded


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
    # Encode train and validation labels
    df_train["label_encoded"] = _encode_single_label_column(
        df_train, dataset_name="training set"
    )
    df_val["label_encoded"] = _encode_single_label_column(
        df_val, dataset_name="validation set"
    )

    y_train = df_train["label_encoded"]
    y_val = df_val["label_encoded"]

    if df_test is not None:
        if "label" not in df_test.columns:
            # Populate with dummy values (zeros) if label column is missing
            y_test = pd.Series([0] * len(df_test), index=df_test.index)
            return y_train, y_val, y_test

        df_test["label_encoded"] = _encode_single_label_column(
            df_test, dataset_name="test set"
        )
        y_test = df_test["label_encoded"]
        return y_train, y_val, y_test

    return y_train, y_val
