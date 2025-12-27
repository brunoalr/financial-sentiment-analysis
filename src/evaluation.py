"""
Evaluation metrics and confusion matrix plotting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from transformers import EvalPrediction

SHAP_ERROR_MSG = "SHAP is not installed. Install it with: pip install shap"

if TYPE_CHECKING:
    import shap
    from shap import Explanation, Explainer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
else:
    try:
        import shap
    except ImportError:
        shap = None


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Define metric for Trainer (common for Transformer models) -
    version with output_attentions.

    Args:
        eval_pred: EvalPrediction object from Transformers

    Returns:
        dict: Dictionary with balanced_accuracy metric
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # If logits is list/tuple (can happen with output_attentions),
    # get first element
    if isinstance(logits, (list, tuple)) and len(logits) > 0:
        logits = logits[0]

    # Convert to numpy if necessary
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Ensure logits is 2D
    if len(logits.shape) > 2:
        logits = logits.reshape(-1, logits.shape[-1])

    # Get predictions
    predictions = np.argmax(logits, axis=-1)
    labels = np.array(labels).flatten()
    return {"balanced_accuracy": balanced_accuracy_score(labels, predictions)}


def compute_metrics_without_attentions(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Define metric for Trainer (common for Transformer models) -
    version without output_attentions.

    Args:
        eval_pred: EvalPrediction object from Transformers

    Returns:
        dict: Dictionary with balanced_accuracy metric
    """
    # eval_pred is an EvalPrediction object
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Convert to numpy if necessary
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get predictions
    predictions = np.argmax(logits, axis=-1)

    # Calculate balanced accuracy
    return {"balanced_accuracy": balanced_accuracy_score(labels, predictions)}


def plot_confusion_matrix(
    labels: Union[np.ndarray, list[int]],
    predictions: Union[np.ndarray, list[int]],
    title: str,
) -> None:
    """
    Plot confusion matrix with normalized values.

    Args:
        labels: True labels
        predictions: Predicted labels
        title: Title for the plot
    """
    class_names = ["neutral", "positive", "negative"]

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Normalized confusion matrix
    # Protect against division by zero
    row_sums = cm.astype(float).sum(axis=1, keepdims=True)
    # Replace zero sums with 1 to avoid division by zero (will result in 0/1 = 0)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cmn = cm / row_sums
    balanced_acc = np.trace(cmn) / 3

    plt.figure(figsize=(11, 9))
    plt.suptitle(f"{title} - Balanced accuracy = {balanced_acc:.5f}")
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 16},
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.show()


def create_shap_predictor(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: Optional[Union[str, torch.device]] = None,
) -> Callable[[Union[str, list[str], np.ndarray, tuple]], np.ndarray]:
    """
    Create a prediction function wrapper for SHAP explainability.

    Args:
        model: PyTorch model
        tokenizer: PreTrainedTokenizerBase tokenizer for the model
        device: Device to run on (defaults to model's device)

    Returns:
        function: Prediction function that takes texts and returns
            probabilities
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    def predict_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            if texts.ndim == 0:
                texts = [str(texts.item())]
            else:
                texts = texts.tolist()
        elif isinstance(texts, tuple):
            texts = list(texts)

        texts = [str(t) if not isinstance(t, str) else t for t in texts]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    return predict_fn


def create_shap_explainer(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: Optional[Union[str, torch.device]] = None,
    output_names: Optional[list[str]] = None,
) -> Explainer:
    """
    Create a SHAP explainer for the model.

    Args:
        model: PyTorch model
        tokenizer: PreTrainedTokenizerBase tokenizer for the model
        device: Device to run on (defaults to model's device)
        output_names: Optional list of output class names

    Returns:
        shap.Explainer: SHAP explainer instance
    """
    if shap is None:
        raise ImportError(SHAP_ERROR_MSG)
    predict_fn = create_shap_predictor(model, tokenizer, device)
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker, output_names=output_names)
    return explainer


def compute_shap_values(
    explainer: Explainer,
    texts: Union[pd.Series, str, list[str]],
    max_evals: Optional[int] = None,
    batch_size: int = 2,
) -> Explanation:
    """
    Compute SHAP values for given texts.

    Args:
        explainer: SHAP explainer instance
        texts: Texts to explain (can be pd.Series, str, or list[str])
        max_evals: Optional maximum evaluations
        batch_size: Batch size for computation

    Returns:
        Explanation: SHAP Explanation object containing SHAP values
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    elif isinstance(texts, str):
        texts = [texts]

    if max_evals is None:
        shap_values = explainer(texts, fixed_context=1, batch_size=batch_size)
    else:
        shap_values = explainer(
            texts, max_evals=max_evals, fixed_context=1, batch_size=batch_size
        )

    return shap_values


def plot_shap_summary(
    shap_values: Explanation,
    class_names: Optional[list[str]] = None,
    max_display: int = 15,
) -> None:
    """
    Plot SHAP summary for all classes.

    Args:
        shap_values: SHAP Explanation object from compute_shap_values
        class_names: Optional list of class names (defaults to Neutral,
            Positive, Negative)
        max_display: Maximum number of features to display
    """
    if shap is None:
        raise ImportError(SHAP_ERROR_MSG)
    if class_names is None:
        class_names = ["Neutral", "Positive", "Negative"]
    for idx, name in enumerate(class_names):
        print(f"\n=== SHAP: Summary for {name} ===")
        shap.plots.bar(shap_values[:, :, idx].mean(0), max_display=max_display)


def plot_shap_waterfall(
    shap_values: Explanation,
    text_idx: int,
    class_idx: int,
    class_names: Optional[list[str]] = None,
    max_display: int = 15,
) -> None:
    """
    Plot SHAP waterfall for a specific text and class.

    Args:
        shap_values: SHAP Explanation object from compute_shap_values
        text_idx: Index of the text to plot
        class_idx: Index of the class to plot
        class_names: Optional list of class names (defaults to Neutral,
            Positive, Negative)
        max_display: Maximum number of features to display
    """
    if shap is None:
        raise ImportError(SHAP_ERROR_MSG)
    if class_names is None:
        class_names = ["Neutral", "Positive", "Negative"]
    class_name = class_names[class_idx]
    print(f"\n=== Waterfall Plot for {class_name} ===")
    shap.plots.waterfall(
        shap_values[text_idx, :, class_idx], max_display=max_display
    )


def plot_shap_waterfall_all_classes(
    shap_values: Explanation,
    text_idx: int,
    class_names: Optional[list[str]] = None,
    max_display: int = 15,
) -> None:
    """
    Plot SHAP waterfall for all classes for a specific text.

    Args:
        shap_values: SHAP Explanation object from compute_shap_values
        text_idx: Index of the text to plot
        class_names: Optional list of class names (defaults to Neutral,
            Positive, Negative)
        max_display: Maximum number of features to display
    """
    if class_names is None:
        class_names = ["Neutral", "Positive", "Negative"]
    for idx in range(len(class_names)):
        plot_shap_waterfall(
            shap_values, text_idx, idx, class_names, max_display
        )


def analyse_sample(
    idx: int,
    sample_texts: list[str],
    predictions: dict[str, dict[str, Union[int, str]]],
    shap_values: Explanation,
) -> None:
    """
    Analyze a single sample with SHAP visualizations.

    Args:
        idx: Index of the sample to analyze
        sample_texts: List of text samples
        predictions: Dictionary mapping text to prediction info with keys:
            'true_label' and 'predicted_label'
        shap_values: SHAP Explanation object from compute_shap_values
    """
    if shap is None:
        raise ImportError(SHAP_ERROR_MSG)

    text = sample_texts[idx]
    print(f"\n=== Sample {idx} ===")
    print(f"Text: {text}")
    print(f"True Label: {predictions[text]['true_label']}")
    print(f"Predicted: {predictions[text]['predicted_label']}")
    shap.plots.text(shap_values[idx])
    plot_shap_waterfall_all_classes(shap_values, idx)
