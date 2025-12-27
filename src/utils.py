"""
Utility functions for device detection, plotting, and history management
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import hashlib
import json
import os

import matplotlib.axes
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import torch
from tqdm import tqdm
from wordcloud import WordCloud

from src.config import MODELS_DIR, SEED

if TYPE_CHECKING:
    from transformers import Trainer
    import tensorflow as tf


def get_device() -> str:
    """
    Helper function to determine device (CUDA, MPS or CPU).

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def plot_loss(
    history: Union[tf.keras.callbacks.History, List[Dict[str, Union[int, float, str]]]],
    model_name: str = "Transformers",
) -> None:
    """
    Plot training history (Loss).
    Supports both Keras and Hugging Face Transformers history.

    Args:
        history: Training history (tf.keras.callbacks.History object or
            list of dicts from Transformers with values of type int, float, or str)
        model_name: Name of the model for plot title
    """
    # Check if it's Keras history (has .history attribute)
    if hasattr(history, "history"):
        history_dict = history.history
        train_loss = history_dict.get("loss", [])
        val_loss = history_dict.get("val_loss", [])
        epochs = range(1, len(train_loss) + 1)
        val_epochs = range(1, len(val_loss) + 1) if val_loss else []
    # Otherwise, assume it's a list of dictionaries from Transformers
    else:
        train_loss = []
        train_epochs = []
        val_loss = []
        val_epochs = []

        for log in history:
            # Training loss
            if "loss" in log and "epoch" in log and "eval_loss" not in log:
                train_loss.append(log["loss"])
                train_epochs.append(log["epoch"])
            # Validation loss
            if "eval_loss" in log:
                val_loss.append(log["eval_loss"])
                val_epochs.append(log["epoch"])

        epochs = train_epochs if train_epochs else range(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        train_loss,
        "b-",
        label="Training Loss",
        marker="o",
        alpha=0.6,
        linewidth=1,
    )
    if val_loss:
        if val_epochs:
            plt.plot(
                val_epochs,
                val_loss,
                "r-",
                label="Validation Loss",
                marker="s",
                linewidth=2,
            )
        else:
            plt.plot(
                range(1, len(val_loss) + 1),
                val_loss,
                "r-",
                label="Validation Loss",
                marker="s",
                linewidth=2,
            )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_training_history(trainer: Trainer, model_name: str) -> None:
    """
    Save training history using the Trainer class logic.
    Uses trainer.state.log_history to get the training history.

    Args:
        trainer: Hugging Face Trainer instance
        model_name: Name of the model
    """
    # Get log_history from TrainerState (contains all registered metrics)
    log_history = trainer.state.log_history

    # Create safe filename
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    if model_name_safe.startswith("models/"):
        model_name_safe = model_name_safe.replace("models/", "")

    # File path
    history_path = os.path.join(MODELS_DIR, f"{model_name_safe}_training_history.json")

    # Create structure with log_history and additional state information
    training_state = {
        "log_history": log_history,
        "best_metric": getattr(trainer.state, "best_metric", None),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "epoch": getattr(trainer.state, "epoch", None),
        "global_step": getattr(trainer.state, "global_step", None),
        "max_steps": getattr(trainer.state, "max_steps", None),
    }

    # Save state as JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_state, f, indent=2, ensure_ascii=False)

    print(f"Training history saved to: {history_path}")


def load_training_history(
    model_name: str,
) -> Optional[List[Dict[str, Union[int, float, str]]]]:
    """
    Load training history from a JSON file if available.
    Returns log_history if the file exists, None otherwise.

    Args:
        model_name: Name of the model

    Returns:
        Optional list of dictionaries with string keys and values of type
            int, float, or str. Returns None if file doesn't exist.
    """
    # Create safe filename
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    if model_name_safe.startswith("models/"):
        model_name_safe = model_name_safe.replace("models/", "")

    # File path
    history_path = os.path.join(MODELS_DIR, f"{model_name_safe}_training_history.json")

    # Check if file exists
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                training_state = json.load(f)

            # Extract log_history from state
            log_history = training_state.get("log_history", [])
            if log_history:
                print(f"Training history loaded from: {history_path}")
                return log_history
            else:
                print(f"History file found but is empty: {history_path}")
                return None
        except Exception as e:
            print(f"Error loading training history: {e}")
            return None
    else:
        return None


def plot_results_comparison(
    model_type: str,
    results_val_dict: Dict[str, Dict[str, float]],
    results_test_dict: Dict[str, Dict[str, float]],
) -> None:
    """
    Compare validation and test results.
    Based on https://python-graph-gallery.com/11-grouped-barplot/

    Args:
        model_type: Type/category of models to compare
        results_val_dict: Dictionary of validation results per model type
        results_test_dict: Dictionary of test results per model type
    """
    model_names = list(results_test_dict[model_type].keys())

    rows = []
    for m in model_names:
        rows.append(
            {
                "Model": m,
                "Set": "Validation",
                "Balanced Accuracy": results_val_dict[model_type][m],
            }
        )
        rows.append(
            {
                "Model": m,
                "Set": "Test",
                "Balanced Accuracy": results_test_dict[model_type][m],
            }
        )

    df_plot = pd.DataFrame(rows)

    # Sort by TEST metric (desc) and fix this order in the plot
    order = (
        df_plot[df_plot["Set"] == "Test"]
        .sort_values("Balanced Accuracy", ascending=False)["Model"]
        .tolist()
    )

    plt.figure(figsize=(14, max(7, 0.6 * len(order))))
    ax = sns.barplot(
        data=df_plot,
        y="Model",
        x="Balanced Accuracy",
        hue="Set",
        order=order,
        palette="viridis",
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.5f", fontsize=12, padding=2)

    ax.set_xlabel("Balanced Accuracy", fontsize=16, fontweight="bold")
    ax.set_ylabel("Models", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)

    ax.set_title(
        f"{model_type}: Validation VS Test Results",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlim(0, df_plot["Balanced Accuracy"].max() * 1.1)
    ax.legend(
        title="",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fontsize=14,
    )
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", zorder=1, color="white")
    ax.set_axisbelow(False)
    plt.tight_layout()
    plt.show()


def generate_word_cloud(
    text_series: pd.Series, title: str, ax: matplotlib.axes.Axes
) -> None:
    """
    Generate and plot a word cloud from a text series.

    Args:
        text_series: pandas Series containing text data
        title: Title for the word cloud plot
        ax: Matplotlib axes object to plot on
    """
    # Join all texts in the series into a single long string
    text_corpus = " ".join(text_series.astype(str))

    # Create WordCloud object
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100,  # Limit to 100 words for clarity
        min_font_size=10,
        random_state=SEED,
    ).generate(text_corpus)

    # Plot the word cloud
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=15)


def calculate_sha256(file_path: Union[str, Path]) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        str: SHA256 hash in hexadecimal format
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def check_sha256(file_path: Union[str, Path], expected_hash: str) -> bool:
    """
    Check if file's SHA256 hash matches expected hash.

    Args:
        file_path: Path to the file
        expected_hash: Expected SHA256 hash (hexadecimal string)

    Returns:
        bool: True if hash matches, False otherwise
    """
    actual_sha256 = calculate_sha256(file_path)
    return actual_sha256.lower() == expected_hash.lower()


def download_file_with_checksum(
    url: str,
    dest_path: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> None:
    """
    Download a file from URL with optional SHA256 checksum verification.

    Args:
        url: URL to download from
        dest_path: Destination path (Path object or string)
        expected_sha256: Optional expected SHA256 hash for verification

    Raises:
        ValueError: If checksum verification fails
        requests.RequestException: If download fails
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and hash matches
    if dest_path.exists():
        if expected_sha256 and check_sha256(dest_path, expected_sha256):
            print(f"File already exists and checksum verified: {dest_path}")
            return
        else:
            print("File exists but checksum doesn't match. Re-downloading...")
            dest_path.unlink()

    print(f"Downloading from {url} to {dest_path}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Verify SHA256 checksum if provided
    if expected_sha256:
        print("Verifying SHA256 checksum...")
        actual_sha256 = calculate_sha256(dest_path)
        if actual_sha256.lower() != expected_sha256.lower():
            print("ERROR: SHA256 checksum does not match!")
            print(f"Expected: {expected_sha256}")
            print(f"Got:      {actual_sha256}")
            dest_path.unlink()  # Remove corrupted file
            raise ValueError("Download failed: SHA256 checksum does not match!")
        else:
            print("SHA256 checksum verified successfully!")


def download_finbert_trc2_model() -> None:
    """
    Download FinBERT model trained on TRC2 dataset.

    Downloads the PyTorch model file and verifies its SHA256 checksum.
    """
    project_root = Path.cwd()
    model_dir = project_root / MODELS_DIR / "finbertTRC2"
    model_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://prosus-public.s3-eu-west-1.amazonaws.com/"
        "finbert/language-model/pytorch_model.bin"
    )
    dest_path = model_dir / "pytorch_model.bin"

    # Expected SHA256 hash
    expected_sha256 = "a051f129ec0534f527d20987f3a619b7a97663b89e59bcc83a076b3a2826a0af"

    download_file_with_checksum(url, dest_path, expected_sha256)
