"""
Base classes and utilities for models
"""

from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import TRANSFORMER_MAX_LEN


from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)


class FinancialNewsDataset(Dataset):
    """
    PyTorch Dataset class for financial news sentiment analysis.
    """

    def __init__(
        self,
        texts: Union[list[str], np.ndarray],
        labels: Union[list[int], np.ndarray],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = TRANSFORMER_MAX_LEN,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            texts: Array of text strings
            labels: Array of numeric labels
            tokenizer: Hugging Face tokenizer
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = (
            np.asarray(labels) if not isinstance(labels, np.ndarray) else labels
        )
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.texts)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            item: Index of the item to retrieve

        Returns:
            Dictionary containing 'input_ids', 'attention_mask', and 'labels'
        """
        text = str(self.texts[item])
        label = int(self.labels[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
