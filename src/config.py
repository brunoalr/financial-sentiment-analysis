"""
Configuration constants and default settings
"""

from __future__ import annotations

import os
import random

from transformers import TrainingArguments

# Random seed for reproducibility
SEED: int = 42

# Set random seeds for reproducibility
# Note: numpy, tensorflow, and torch seeds are set here
# if those libraries are available
try:
    import numpy as np
    np.random.seed(SEED)
except ImportError:
    pass

try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except ImportError:
    pass

try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

# Directory paths
DATA_DIR: str = "data"
MODELS_DIR: str = "models"
SUBMISSIONS_DIR: str = "submissions"
LOGS_DIR: str = "./logs"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Default training arguments for transformers
DEFAULT_TRAINING_ARGS = TrainingArguments(
    num_train_epochs=8,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-6,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=LOGS_DIR,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)

# LSTM parameters
MAX_NB_WORDS: int = 5000
MAX_LEN: int = 100
EMBEDDING_DIM: int = 100

# TF-IDF parameters
TFIDF_MAX_FEATURES: int = 5000

# Transformer parameters
TRANSFORMER_MAX_LEN: int = 64

# Default SFT training parameters
DEFAULT_SFT_LEARNING_RATE: float = 2e-5
DEFAULT_SFT_BATCH_SIZE: int = 8
DEFAULT_SFT_NUM_EPOCHS: int = 3
DEFAULT_SFT_MAX_SEQ_LENGTH: int = 64

# Label mapping
LABEL_MAPPING: dict[str, int] = {"neutral": 0, "positive": 1, "negative": 2}
REVERSE_LABEL_MAPPING: dict[int, str] = {
    0: "neutral",
    1: "positive",
    2: "negative",
}
