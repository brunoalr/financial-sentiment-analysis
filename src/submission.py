"""
Submission file generation functions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import SUBMISSIONS_DIR

if TYPE_CHECKING:
    from transformers import Trainer


def generate_submission(
    predictions: Union[np.ndarray, list[int]],
    df_test: pd.DataFrame,
    model_name: str,
) -> str:
    """
    Generate submission CSV file.

    Args:
        predictions: Array of predictions
        df_test: Test DataFrame with 'ID' column
        model_name: Name of the model for filename

    Returns:
        str: Path to saved submission file
    """
    submission = pd.DataFrame({"ID": df_test["ID"], "TARGET": predictions})
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    submission_filename = f"{SUBMISSIONS_DIR}/submission_{model_name_safe}.csv"
    submission.to_csv(submission_filename, index=False)
    print(f"File '{submission_filename}' generated successfully!")
    return submission_filename


def generate_submission_transformers(
    trainer: Trainer,
    model_name: str,
    test_dataset: Dataset,
    df_test: pd.DataFrame,
) -> str:
    """
    Generate submission file for transformer models.

    Args:
        trainer: Hugging Face transformers.Trainer instance
        model_name: Name of the model
        test_dataset: PyTorch Dataset instance (test dataset)
        df_test: Test DataFrame with 'ID' column

    Returns:
        str: Path to saved submission file
    """
    # BERT models need specific tokenization
    print(f"Using {model_name} for prediction...")
    predictions = trainer.predict(test_dataset)

    # Handle format when output_attentions=True
    logits = predictions.predictions
    if isinstance(logits, (list, tuple)) and len(logits) > 0:
        logits = logits[0]
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if len(logits.shape) > 2:
        logits = logits.reshape(-1, logits.shape[-1])

    final_pred = np.argmax(logits, axis=1)

    # Create submission DataFrame
    submission = pd.DataFrame({"ID": df_test["ID"], "TARGET": final_pred})
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    submission_filename = f"{SUBMISSIONS_DIR}/submission_{model_name_safe}.csv"
    submission.to_csv(submission_filename, index=False)
    print(f"File '{submission_filename}' generated successfully!")
    return submission_filename
