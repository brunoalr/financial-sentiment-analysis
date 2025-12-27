"""
Transformer models for financial sentiment analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.config import DEFAULT_TRAINING_ARGS, MODELS_DIR, SEED
from src.evaluation import compute_metrics, plot_confusion_matrix
from src.models.dataset import FinancialNewsDataset
from src.submission import (
    generate_submission,
    generate_submission_transformers,
)
import src.utils as utils
from src.utils import get_device

if TYPE_CHECKING:
    from transformers.pipelines import Pipeline


# Suppress warnings de pin_memory - false positive for macbooks
warnings.filterwarnings("ignore", message=".*pin_memory.*")


def _prepare_model_path(model_name: str) -> str:
    """
    Prepare model save path from model name.

    Args:
        model_name: Name of the model

    Returns:
        str: Model save path
    """
    # If model_name starts with "models/", extract only the base name
    if model_name.startswith("models/"):
        base_name = model_name.replace("models/", "").replace("/", "_")
    else:
        base_name = model_name.replace("/", "_").replace("-", "_")

    model_name_safe = base_name.replace("/", "_").replace("-", "_")
    model_save_path = os.path.join(MODELS_DIR, f"{model_name_safe}_fine_tuned")

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    return model_save_path


def _check_model_exists(model_save_path: str) -> bool:
    """
    Check if a fine-tuned model already exists.

    Args:
        model_save_path: Path to model directory

    Returns:
        bool: True if model exists, False otherwise
    """
    model_exists = False
    if os.path.exists(model_save_path):
        # Check if it contains essential model files
        config_exists = os.path.exists(os.path.join(model_save_path, "config.json"))
        model_file_exists = os.path.exists(
            os.path.join(model_save_path, "pytorch_model.bin")
        ) or os.path.exists(os.path.join(model_save_path, "model.safetensors"))
        if config_exists and model_file_exists:
            model_exists = True
            print(f"\nModel found at {model_save_path}. Loading saved model...")

    return model_exists


def _load_tokenizer(
    model_name: str,
    model_save_path: str,
    model_exists: bool,
    tokenizer_name: Optional[str] = None,
) -> AutoTokenizer:
    """
    Load tokenizer for a model.

    Args:
        model_name: Name of the model
        model_save_path: Path to saved model
        model_exists: Whether model exists
        tokenizer_name: Optional tokenizer name (if different from model_name)

    Returns:
        AutoTokenizer: Tokenizer instance from transformers
    """
    if model_exists:
        # Load tokenizer from saved model or use tokenizer_name if provided
        if tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    else:
        tokenizer = (
            AutoTokenizer.from_pretrained(model_name)
            if tokenizer_name is None
            else AutoTokenizer.from_pretrained(tokenizer_name)
        )

    return tokenizer


def _create_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: Union[np.ndarray, pd.Series],
    y_val: Union[np.ndarray, pd.Series],
    tokenizer: AutoTokenizer,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
) -> tuple[FinancialNewsDataset, FinancialNewsDataset, FinancialNewsDataset]:
    """
    Create PyTorch datasets for training, validation, and test.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        y_train: Training labels
        y_val: Validation labels
        tokenizer: AutoTokenizer instance from transformers
        y_test: Optional test labels. If None, dummy labels are used

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)

    Raises:
        ValueError: If y_test length doesn't match df_test length
    """
    train_dataset = FinancialNewsDataset(
        df_train["text"].values, y_train, tokenizer
    )
    val_dataset = FinancialNewsDataset(
        df_val["text"].values, y_val, tokenizer
    )
    # Use provided test labels or dummy labels
    if y_test is not None:
        if len(y_test) != len(df_test):
            raise ValueError(
                f"Length mismatch: y_test has {len(y_test)} samples, "
                f"but df_test has {len(df_test)} samples. "
                f"They must have the same length."
            )
        test_labels = y_test
    else:
        test_labels = np.zeros(len(df_test))

    test_dataset = FinancialNewsDataset(
        df_test["text"].values, test_labels, tokenizer
    )

    return train_dataset, val_dataset, test_dataset


def _print_model_info(model: torch.nn.Module) -> None:
    """
    Print model parameter information.

    Args:
        model: PyTorch model
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )


def _load_existing_model(
    model_save_path: str,
    model_name: str,
    val_dataset: FinancialNewsDataset,
    compute_metrics_func: Callable[[EvalPrediction], Dict[str, float]],
) -> Trainer:
    """
    Load an existing fine-tuned model.

    Args:
        model_save_path: Path to saved model
        model_name: Name of the model
        val_dataset: Validation dataset
        compute_metrics_func: Function to compute metrics

    Returns:
        Trainer: Trainer instance from transformers
    """
    model_ft = AutoModelForSequenceClassification.from_pretrained(
        model_save_path, num_labels=3, output_attentions=False
    )

    _print_model_info(model_ft)

    # Create Trainer only for evaluation (without training)
    training_args = TrainingArguments(
        output_dir=None,
        per_device_eval_batch_size=32,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    trainer_ft = Trainer(
        model=model_ft,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_func,
    )

    print("Model loaded. Skipping training and going directly to evaluation...")

    # Try to load and plot training history if available
    loaded_history = utils.load_training_history(model_name)
    if loaded_history is not None:
        print("Plotting loaded training history...")
        utils.plot_loss(loaded_history, model_name)

    return trainer_ft


def _train_new_model(
    model_name: str,
    model_save_path: str,
    train_dataset: FinancialNewsDataset,
    val_dataset: FinancialNewsDataset,
    tokenizer: AutoTokenizer,
    compute_metrics_func: Callable[[EvalPrediction], Dict[str, float]],
    training_args: TrainingArguments = DEFAULT_TRAINING_ARGS,
) -> Trainer:
    """
    Train a new transformer model.

    Args:
        model_name: Name of the base model
        model_save_path: Path to save fine-tuned model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: AutoTokenizer instance from transformers
        compute_metrics_func: Function to compute metrics
        training_args: Training arguments

    Returns:
        Trainer: Trainer instance from transformers
    """
    # Load base model for training
    model_ft = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, output_attentions=False
    )

    _print_model_info(model_ft)

    # Train
    trainer_ft = Trainer(
        model=model_ft,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_func,
    )

    print("Starting training...")
    trainer_ft.train()

    # Save model after training
    print(f"\nSaving fine-tuned model to {model_save_path}...")
    trainer_ft.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved successfully to {model_save_path}!")

    utils.plot_loss(trainer_ft.state.log_history, model_name)
    utils.save_training_history(trainer_ft, model_name)

    return trainer_ft


def _plot_confusion_matrix_from_trainer(
    trainer_ft: Trainer,
    dataset: FinancialNewsDataset,
    title: str,
    batch_size: Optional[int] = None,
) -> None:
    """
    Plot confusion matrix from trainer predictions.

    Args:
        trainer_ft: Trainer instance from transformers
        dataset: FinancialNewsDataset to evaluate
        title: Title for the plot
        batch_size: Optional batch size for prediction. If None, uses
            trainer's default batch size. Use smaller batch size for
            large datasets to avoid memory issues.
    """
    np.set_printoptions(precision=4, suppress=True)

    # Temporarily update batch size if specified
    original_batch_size = None
    if batch_size is not None:
        original_batch_size = trainer_ft.args.per_device_eval_batch_size
        trainer_ft.args.per_device_eval_batch_size = batch_size

    try:
        # Get predictions (logits)
        predictions_obj = trainer_ft.predict(dataset)

        # Extract logits from predictions
        logits = predictions_obj.predictions
        if isinstance(logits, (list, tuple)) and len(logits) > 0:
            logits = logits[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if len(logits.shape) > 2:
            logits = logits.reshape(-1, logits.shape[-1])

        # Convert logits to classes
        predictions = np.argmax(logits, axis=1)
        labels = dataset.labels
        plot_confusion_matrix(labels, predictions, title)
    finally:
        # Restore original batch size
        if original_batch_size is not None:
            trainer_ft.args.per_device_eval_batch_size = original_batch_size


def _evaluate_and_submit(
    trainer_ft: Trainer,
    model_name: str,
    val_dataset: FinancialNewsDataset,
    test_dataset: FinancialNewsDataset,
    results_dict: Dict[str, float],
    type_name: str,
    results_per_type: Dict[str, Dict[str, float]],
    df_test: pd.DataFrame,
) -> float:
    """
    Evaluate model and generate submission file.

    Args:
        trainer_ft: Trainer instance from transformers
        model_name: Name of the model
        val_dataset: Validation FinancialNewsDataset
        test_dataset: Test FinancialNewsDataset
        results_dict: Dictionary to store results
        type_name: Type/category of the model
        results_per_type: Dictionary to store results per type
        df_test: Test DataFrame

    Returns:
        float: Balanced accuracy
    """
    # Evaluate model (with fine-tuning)
    eval_result = trainer_ft.evaluate(eval_dataset=val_dataset)
    ft_bal_acc = eval_result["eval_balanced_accuracy"]
    print("\n" + "-" * 100)
    print(f"{model_name} Balanced Accuracy: {ft_bal_acc:.5f}")

    print("\n" + "-" * 100)
    generate_submission_transformers(
        trainer_ft, f"{model_name}_fine_tuned", test_dataset, df_test
    )
    print("-" * 100)

    print("\n" + "-" * 100)
    name = f"{model_name} (fine-tuned)"
    results_dict[name] = ft_bal_acc
    results_per_type[type_name][name] = ft_bal_acc

    print("=" * 100)

    return ft_bal_acc


def train_and_evaluate_model(
    model_name: str,
    type_name: str,
    training_args: TrainingArguments = DEFAULT_TRAINING_ARGS,
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
    y_train: Optional[Union[np.ndarray, pd.Series]] = None,
    y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
    tokenizer_name: Optional[str] = None,
    results_val: Optional[Dict[str, float]] = None,
    results_val_per_type: Optional[Dict[str, Dict[str, float]]] = None,
) -> Trainer:
    """
    Train and evaluate a transformer model.

    Args:
        model_name: Name of the base model
        type_name: Type/category of the model
        training_args: Training arguments
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        y_train: Training labels
        y_val: Validation labels
        y_test: Optional test labels. If None, dummy labels are used
        tokenizer_name: Optional tokenizer name
        results_val: Dictionary to store validation results
        results_val_per_type: Dictionary to store results per type

    Returns:
        Trainer: Trainer instance from transformers
    """
    print("=" * 100)
    print(f"Evaluating {model_name} on validation set...")
    print("=" * 100)
    print("-" * 100)

    print("\n" + "-" * 100)
    print(f"{model_name} (fine-tuned)")
    print("-" * 100)

    # Prepare model path
    model_save_path = _prepare_model_path(model_name)

    # Check if model already exists
    model_exists = _check_model_exists(model_save_path)

    # Load tokenizer
    tokenizer = _load_tokenizer(
        model_name, model_save_path, model_exists, tokenizer_name
    )

    # Create datasets
    train_dataset, val_dataset, test_dataset = _create_datasets(
        df_train, df_val, df_test, y_train, y_val, tokenizer, y_test
    )

    # Load existing model or train new one
    if model_exists:
        trainer_ft = _load_existing_model(
            model_save_path, model_name, val_dataset, compute_metrics
        )
    else:
        trainer_ft = _train_new_model(
            model_name,
            model_save_path,
            train_dataset,
            val_dataset,
            tokenizer,
            compute_metrics,
            training_args,
        )

    _plot_confusion_matrix_from_trainer(
        trainer_ft, val_dataset, f"{model_name} - Fine-Tuned"
    )

    # Plot test confusion matrix if test labels are available
    # Use smaller batch size for test dataset to avoid memory issues
    if y_test is not None:
        _plot_confusion_matrix_from_trainer(
            trainer_ft,
            test_dataset,
            f"{model_name} - Test",
            batch_size=8,  # Smaller batch size to avoid memory issues
        )

    # Evaluate and generate submission
    if results_val is not None and results_val_per_type is not None:
        _evaluate_and_submit(
            trainer_ft,
            model_name,
            val_dataset,
            test_dataset,
            results_val,
            type_name,
            results_val_per_type,
            df_test,
        )

    return trainer_ft


def get_zero_shot_predictions(
    classifier: Pipeline, df: pd.DataFrame, batch_size: int
) -> list[int]:
    """
    Get zero-shot predictions from a classifier pipeline.

    Args:
        classifier: transformers.Pipeline for zero-shot classification
        df: DataFrame with 'text' column
        batch_size: Batch size for processing

    Returns:
        list: Predicted labels
    """
    label_mapping = {"positive": 1, "neutral": 0, "negative": 2}
    candidate_labels = ["positive", "neutral", "negative"]
    y_pred = []
    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].iloc[i : i + batch_size].tolist()
        batch_preds = classifier(batch_texts, candidate_labels)
        for pred in batch_preds:
            # Get label with highest score
            predicted_label = pred["labels"][0]
            y_pred.append(label_mapping.get(predicted_label, 0))
    return y_pred


def evaluate_zero_shot_model(
    model_name: str,
    model_type: str,
    df_val: pd.DataFrame,
    y_val: Union[np.ndarray, pd.Series],
    df_test: pd.DataFrame,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
    results_val: Optional[Dict[str, float]] = None,
    results_val_per_type: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Evaluate a zero-shot transformer model.

    Args:
        model_name: Name of the model
        model_type: Type/category of the model
        df_val: Validation DataFrame
        y_val: Validation labels
        df_test: Test DataFrame
        y_test: Optional test labels. If provided, confusion matrix will be
            plotted for test set
        results_val: Dictionary to store validation results
        results_val_per_type: Dictionary to store results per type
    """
    print(f"\n{'-' * 100}")
    print(f"Evaluating: {model_name} - zero-shot classification (pre-trained)")
    print(f"{'-' * 100}")

    classifier = pipeline(
        "zero-shot-classification", model=model_name, device=get_device()
    )

    batch_size = 16

    y_pred = get_zero_shot_predictions(classifier, df_val, batch_size)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.5f}")
    name_pre_trained = f"{model_name} - zero-shot (pre-trained)"

    if results_val is not None:
        results_val[name_pre_trained] = bal_acc
    if results_val_per_type is not None:
        results_val_per_type[model_type][name_pre_trained] = bal_acc

    plot_confusion_matrix(y_val, y_pred, name_pre_trained)

    # Generate predictions for test set
    y_pred_test = get_zero_shot_predictions(classifier, df_test, batch_size)

    # Plot test confusion matrix if test labels are available
    if y_test is not None:
        plot_confusion_matrix(
            y_test,
            y_pred_test,
            f"{model_name} - zero-shot (pre-trained) - Test",
        )

    # Generate submission file (will skip if ID column is missing)
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    generate_submission(
        y_pred_test,
        df_test,
        f"{model_name_safe}_zero_shot_pre_trained",
    )


def get_sentiment_predictions(
    classifier: Pipeline, df: pd.DataFrame, batch_size: int
) -> list[int]:
    """
    Get sentiment predictions from a sentiment analysis pipeline.

    Args:
        classifier: transformers.Pipeline for sentiment analysis
        df: DataFrame with 'text' column
        batch_size: Batch size for processing

    Returns:
        list: Predicted labels
    """
    predictions = []
    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].iloc[i : i + batch_size].tolist()
        batch_preds = classifier(
            batch_texts, truncation=True, max_length=512, padding=True
        )
        predictions.extend(batch_preds)

    label_mapping = {
        "POSITIVE": 1,
        "positive": 1,
        "LABEL_1": 1,
        "POS": 1,
        "NEUTRAL": 0,
        "neutral": 0,
        "LABEL_0": 0,
        "NEU": 0,
        "NEGATIVE": 2,
        "negative": 2,
        "LABEL_2": 2,
        "NEG": 2,
    }

    # Converter para labels numéricos
    y_pred = []
    for pred in predictions:
        label_str = str(pred["label"]).upper().strip()
        mapped_label = label_mapping.get(label_str, None)
        # Default to neutral (0) if label mapping fails
        if mapped_label is None:
            print(
                f"Warning: Unknown label '{label_str}' not found in "
                f"mapping. Defaulting to neutral (0)."
            )
            mapped_label = 0
        y_pred.append(mapped_label)
    return y_pred


def evaluate_sentiment_model(
    model_name: str,
    model_type: str,
    df_val: pd.DataFrame,
    y_val: Union[np.ndarray, pd.Series],
    df_test: pd.DataFrame,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
    save_result: bool = True,
    results_val: Optional[Dict[str, float]] = None,
    results_val_per_type: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Evaluate a pre-trained sentiment analysis model.

    Args:
        model_name: Name of the model
        model_type: Type/category of the model
        df_val: Validation DataFrame
        y_val: Validation labels
        df_test: Test DataFrame
        y_test: Optional test labels. If provided, confusion matrix will be
            plotted for test set
        save_result: Whether to save results
        results_val: Dictionary to store validation results
        results_val_per_type: Dictionary to store results per type
    """
    print(f"\n{'-' * 100}")
    print(f"Evaluating: {model_name} - sentiment-analysis (pre-trained)")
    print(f"{'-' * 100}")

    classifier = pipeline("sentiment-analysis", model=model_name, device=get_device())

    batch_size = 32

    y_pred = get_sentiment_predictions(classifier, df_val, batch_size)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.5f}")
    name_pre_trained = f"{model_name} - sentiment (pre-trained)"

    if save_result:
        if results_val is not None:
            results_val[name_pre_trained] = bal_acc
        if results_val_per_type is not None:
            results_val_per_type[model_type][name_pre_trained] = bal_acc

    plot_confusion_matrix(y_val, y_pred, name_pre_trained)

    y_pred_test = get_sentiment_predictions(classifier, df_test, batch_size)

    # Plot test confusion matrix if test labels are available
    if y_test is not None:
        plot_confusion_matrix(
            y_test,
            y_pred_test,
            f"{model_name} - sentiment (pre-trained) - Test",
        )

    # Generate submission file (will skip if ID column is missing)
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    generate_submission(
        y_pred_test,
        df_test,
        f"{model_name_safe}_sentiment_pre_trained",
    )
