"""
LLM (Large Language Model) evaluation functions
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from datasets import Dataset
from trl import SFTTrainer

from src.config import MODELS_DIR, SUBMISSIONS_DIR
from src.evaluation import plot_confusion_matrix
from src.submission import generate_submission
from src.utils import get_device

# Constants
DEFAULT_NEUTRAL_LABEL = "   Defaulting to neutral (0)"

# Default SFT training parameters
DEFAULT_SFT_LEARNING_RATE = 2e-5
DEFAULT_SFT_BATCH_SIZE = 4
DEFAULT_SFT_NUM_EPOCHS = 3
DEFAULT_SFT_MAX_SEQ_LENGTH = 512


def _get_sentiment_instruction() -> str:
    """Get the standardized instruction text for sentiment classification."""
    return (
        "Classify the financial sentiment as positive, neutral, or negative. "
        "Respond with only one word: positive, neutral, or negative."
    )


def _map_response_to_label(response: str) -> int:
    """
    Map text response to numeric label.

    Args:
        response: Text response from model

    Returns:
        int: Numeric label (0=neutral, 1=positive, 2=negative)
    """
    response_lower = response.lower()
    if "positive" in response_lower:
        return 1
    elif "negative" in response_lower:
        return 2
    else:
        return 0


def _process_single_text_llm(
    text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> int:
    """
    Process a single text with LLM model.

    Args:
        text: Text to classify
        model: Loaded LLM model
        tokenizer: Loaded tokenizer

    Returns:
        int: Numeric label (0=neutral, 1=positive, 2=negative)
    """
    try:
        system_msg = _get_sentiment_instruction()
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": str(text)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer([prompt], return_tensors="pt").to(get_device())
        pad_token = tokenizer.eos_token_id or tokenizer.pad_token_id
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=pad_token,
        )

        input_len = inputs["input_ids"].shape[1]
        response = (
            tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            .strip()
            .lower()
        )

        return _map_response_to_label(response)
    except Exception as e:
        print(f"\nWarning: Error processing text '{text[:50]}...': {e}")
        print(DEFAULT_NEUTRAL_LABEL)
        return 0


def _process_texts_llm(
    texts: np.ndarray,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    desc: str = "Processing texts",
) -> np.ndarray:
    """
    Process multiple texts with LLM model.

    Args:
        texts: Array of texts to process
        model: Loaded LLM model
        tokenizer: Loaded tokenizer
        desc: Description for progress bar

    Returns:
        np.ndarray: Array of numeric labels
    """
    predictions = []
    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
            pred_label = _process_single_text_llm(text, model, tokenizer)
            predictions.append(pred_label)
    return np.array(predictions)


def _process_single_text_chatgpt(
    text: str, client: OpenAI, model_name: str, max_retries: int = 10
) -> int:
    """
    Process a single text with ChatGPT API.

    Args:
        text: Text to classify
        client: OpenAI client instance
        model_name: Name of the ChatGPT model
        max_retries: Maximum number of retry attempts

    Returns:
        int: Numeric label (0=neutral, 1=positive, 2=negative)
    """
    retry_count = 0
    instructions_text = _get_sentiment_instruction()

    while retry_count <= max_retries:
        try:
            response = client.responses.create(
                model=model_name,
                instructions=instructions_text,
                input=str(text),
                temperature=0.0,
            )
            response_text = response.output_text
            return _map_response_to_label(response_text)

        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(
                    f"\nError: Max retries ({max_retries}) "
                    f"exceeded for text '{text[:50]}...'"
                )
                print(DEFAULT_NEUTRAL_LABEL)
                return 0
            else:
                wait_time = min(2**retry_count, 60)
                print(f"\nAPI call error (attempt {retry_count}): {e}")
                print(f"   Waiting {wait_time} seconds before trying again...")
                time.sleep(wait_time)

    return 0


def _process_texts_chatgpt(
    texts: np.ndarray,
    client: OpenAI,
    model_name: str,
    desc: str = "Processing texts",
    max_retries: int = 10,
) -> np.ndarray:
    """
    Process multiple texts with ChatGPT API.

    Args:
        texts: Array of texts to process
        client: OpenAI client instance
        model_name: Name of the ChatGPT model
        desc: Description for progress bar
        max_retries: Maximum number of retry attempts per text

    Returns:
        np.ndarray: Array of numeric labels
    """
    predictions = []
    for text in tqdm(texts, desc=desc):
        pred_label = _process_single_text_chatgpt(
            text, client, model_name, max_retries
        )
        predictions.append(pred_label)
    return np.array(predictions)


def _create_submission_file(
    predictions: np.ndarray,
    df_test: pd.DataFrame,
    model_name: str,
    prefix: str = "",
) -> None:
    """
    Create submission CSV file for test predictions.

    Args:
        predictions: Array of predictions
        df_test: Test DataFrame with 'ID' column
        model_name: Name of the model
        prefix: Optional prefix for submission name
    """
    if prefix:
        submission_name = f"{prefix}_{model_name}"
    else:
        submission_name = model_name

    generate_submission(predictions, df_test, submission_name)


def _create_submission_file_direct(
    predictions: np.ndarray, df_test: pd.DataFrame, model_name: str
) -> None:
    """
    Create submission CSV file directly (for LLM models).

    Args:
        predictions: Array of predictions
        df_test: Test DataFrame with 'ID' column
        model_name: Name of the model
    """
    submission_df = pd.DataFrame({"ID": df_test["ID"], "TARGET": predictions})
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    filename = f"{SUBMISSIONS_DIR}/submission_{model_name_safe}.csv"
    submission_df.to_csv(filename, index=False)
    print(f"File 'submission_{model_name_safe}.csv' generated successfully!")


def _prepare_sft_model_path(model_name: str) -> str:
    """
    Prepare model save path for SFT fine-tuned models.

    Args:
        model_name: Name of the model

    Returns:
        str: Model save path
    """
    if model_name.startswith("models/"):
        base_name = model_name.replace("models/", "").replace("/", "_")
    else:
        base_name = model_name.replace("/", "_").replace("-", "_")

    model_name_safe = base_name.replace("/", "_").replace("-", "_")
    model_save_path = os.path.join(MODELS_DIR, f"{model_name_safe}_sft")

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    return model_save_path


def _convert_to_sft_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Dataset:
    """
    Convert DataFrame to SFT conversational dataset format.

    Args:
        df: DataFrame with 'text' and 'label' columns
        text_col: Name of the text column (default: "text")
        label_col: Name of the label column (default: "label")

    Returns:
        Dataset: HuggingFace Dataset with 'messages' format
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame")

    system_msg = _get_sentiment_instruction()

    def create_messages(row):
        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": str(row[text_col])},
                {"role": "assistant", "content": str(row[label_col])},
            ]
        }

    messages_list = df.apply(create_messages, axis=1).tolist()
    return Dataset.from_list(messages_list)


def train_llm_sft(
    model_name: str,
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    training_args: Optional[TrainingArguments] = None,
    **sft_config_kwargs,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Train an LLM model using Supervised Fine-Tuning (SFT).

    Args:
        model_name: Name/path of the base LLM model
        df_train: Training DataFrame with 'text' and 'label' columns
        df_val: Optional validation DataFrame with 'text' and 'label' columns
        output_dir: Optional output directory for saving the model.
            If None, uses MODELS_DIR/{model_name}_sft
        training_args: Optional TrainingArguments. If None, uses defaults.
        max_seq_length: Maximum sequence length for training
        **sft_config_kwargs: Additional arguments to pass to SFTTrainer

    Returns:
        tuple: (trained_model, tokenizer)

    Raises:
        ImportError: If TRL is not installed
        ValueError: If required columns are missing from DataFrames
    """
    if df_train is None or df_train.empty:
        raise ValueError("df_train cannot be None or empty")
    if "text" not in df_train.columns or "label" not in df_train.columns:
        raise ValueError("df_train must contain 'text' and 'label' columns")

    if df_val is not None:
        if "text" not in df_val.columns or "label" not in df_val.columns:
            raise ValueError("df_val must contain 'text' and 'label' columns")

    # Prepare output directory
    if output_dir is None:
        output_dir = _prepare_sft_model_path(model_name)

    print(f"Training model: {model_name}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert datasets to SFT format
    train_dataset = _convert_to_sft_dataset(df_train)
    eval_dataset = (
        _convert_to_sft_dataset(df_val) if df_val is not None else None
    )

    # Prepare training arguments
    if training_args is None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=DEFAULT_SFT_BATCH_SIZE,
            per_device_eval_batch_size=DEFAULT_SFT_BATCH_SIZE,
            num_train_epochs=DEFAULT_SFT_NUM_EPOCHS,
            learning_rate=DEFAULT_SFT_LEARNING_RATE,
            logging_steps=10,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset is not None else False,
            report_to="none",
            seed=42,
            data_seed=42,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # Create trainer with SFT-specific parameters
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        **sft_config_kwargs,
    )

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")

    return model, tokenizer


def evaluate_llm(
    model_name: str,
    df: pd.DataFrame,
    y: Optional[Union[np.ndarray, pd.Series, list[int]]] = None,
    train_sft: bool = False,
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    training_args: Optional[TrainingArguments] = None,
    **sft_config_kwargs,
) -> Tuple[Optional[float], np.ndarray]:
    """
    Evaluate an LLM model on a dataset, optionally with SFT training.

    Args:
        model_name: Name/path of the LLM model
        df: DataFrame with 'text' column for evaluation
        y: Optional labels for evaluation
        train_sft: If True, fine-tune the model using SFT before evaluation
        df_train: Training DataFrame with 'text' and 'label' columns.
            Required if train_sft=True
        df_val: Optional validation DataFrame with 'text' and 'label' columns
            for SFT training
        output_dir: Optional output directory for saving the SFT model.
            If None and train_sft=True, uses MODELS_DIR/{model_name}_sft
        training_args: Optional TrainingArguments for SFT training
        **sft_config_kwargs: Additional arguments to pass to SFTTrainer

    Returns:
        tuple: (balanced_accuracy, predictions) or (None, predictions)
            if y is None

    Raises:
        ValueError: If train_sft=True but df_train is not provided
    """
    # Validate inputs
    if df is None or df.empty:
        raise ValueError("df cannot be None or empty")
    if "text" not in df.columns:
        raise ValueError("df must contain a 'text' column")

    if train_sft:
        if df_train is None or df_train.empty:
            raise ValueError(
                "df_train is required when train_sft=True. "
                "Provide a DataFrame with 'text' and 'label' columns."
            )
        # Train the model first
        model, tokenizer = train_llm_sft(
            model_name=model_name,
            df_train=df_train,
            df_val=df_val,
            output_dir=output_dir,
            training_args=training_args,
            **sft_config_kwargs,
        )
        model.eval()
    else:
        # Check if a fine-tuned model exists, otherwise load base model
        if output_dir is None:
            potential_sft_path = _prepare_sft_model_path(model_name)
        else:
            potential_sft_path = output_dir

        # Try to load fine-tuned model if it exists
        safetensors_path = os.path.join(
            potential_sft_path, "model.safetensors"
        )
        pytorch_model_path = os.path.join(
            potential_sft_path, "pytorch_model.bin"
        )
        model_file_exists = (
            os.path.exists(safetensors_path)
            or os.path.exists(pytorch_model_path)
        )
        if os.path.exists(potential_sft_path) and model_file_exists:
            print(f"Loading fine-tuned model from: {potential_sft_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                potential_sft_path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                potential_sft_path,
                device_map="auto",
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
        else:
            # Load base model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
        model.eval()

    texts = df["text"].values
    model_display_name = (
        f"{model_name} (SFT)" if train_sft else model_name
    )
    print(f"Evaluating model: {model_display_name}")
    print(f"Evaluating {len(texts)} samples...")

    # Evaluate if labels provided
    if y is not None:
        # Process validation set
        desc = "Processing validation texts"
        predictions = _process_texts_llm(texts, model, tokenizer, desc)
        balanced_acc = balanced_accuracy_score(y, predictions)
        plot_confusion_matrix(y, predictions, f"{model_display_name} (LLM)")
        return balanced_acc, predictions

    # Process test set if provided (without validation labels)
    test_texts = df["text"].values
    desc = "Processing test texts"
    test_predictions = _process_texts_llm(test_texts, model, tokenizer, desc)
    submission_model_name = (
        f"{model_name}_sft" if train_sft else model_name
    )
    _create_submission_file_direct(
        test_predictions, df, submission_model_name
    )

    return None, test_predictions


def evaluate_chatgpt(
    model_name: str,
    df: pd.DataFrame,
    y: Optional[Union[np.ndarray, pd.Series, list[int]]] = None,
) -> Tuple[Optional[float], np.ndarray]:
    """
    Evaluate ChatGPT model using OpenAI API.

    Args:
        model_name: Name of the ChatGPT model (e.g., "gpt-5.2")
        df: DataFrame with 'text' column
        y: Optional labels

    Returns:
        tuple: (balanced_accuracy, predictions) or (None, predictions)
            if y is None

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    # Validate inputs
    if df is None or df.empty:
        raise ValueError("df_val cannot be None or empty")
    if "text" not in df.columns:
        raise ValueError("df must contain a 'text' column")

    texts = df["text"].values
    print(f"Evaluating model: {model_name}")
    print(f"Evaluating {len(texts)} samples...")

    # Check if API key is configured
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Configure the OPENAI_API_KEY "
            "environment variable."
        )

    # Initialize OpenAI client with error handling
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        msg = f"Failed to initialize OpenAI client: {e}"
        raise RuntimeError(msg) from e

    # Evaluate if labels provided
    if y is not None:
        # Process validation set
        desc = "Processing validation texts"
        predictions = _process_texts_chatgpt(texts, client, model_name, desc)
        balanced_acc = balanced_accuracy_score(y, predictions)
        plot_confusion_matrix(y, predictions, f"{model_name} (ChatGPT)")
        return balanced_acc, predictions

    # Process test set if provided (without validation labels)
    test_texts = df["text"].values
    desc = "Processing test texts"
    test_predictions = _process_texts_chatgpt(
        test_texts, client, model_name, desc
    )
    _create_submission_file(test_predictions, df, model_name, "chatgpt")

    return None, test_predictions
