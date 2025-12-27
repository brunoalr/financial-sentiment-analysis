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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import SUBMISSIONS_DIR
from src.evaluation import plot_confusion_matrix
from src.submission import generate_submission
from src.utils import get_device

# Constants
DEFAULT_NEUTRAL_LABEL = "   Defaulting to neutral (0)"


def _get_sentiment_instruction() -> str:
    """Get the standardized instruction text for sentiment classification."""
    return "Classify the financial sentiment as positive, neutral, or negative. Respond with only one word: positive, neutral, or negative."


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
        pred_label = _process_single_text_chatgpt(text, client, model_name, max_retries)
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


def evaluate_llm(
    model_name: str,
    df: pd.DataFrame,
    y: Optional[Union[np.ndarray, pd.Series, list[int]]] = None,
) -> Tuple[Optional[float], np.ndarray]:
    """
    Evaluate an LLM model on a dataset.

    Args:
        model_name: Name/path of the LLM model
        df: DataFrame with 'text' column
        y: Optional labels

    Returns:
        tuple: (balanced_accuracy, predictions) or (None, predictions)
            if y is None
    """
    # Validate inputs
    if df is None or df.empty:
        raise ValueError("df_val cannot be None or empty")
    if "text" not in df.columns:
        raise ValueError("df must contain a 'text' column")

    texts = df["text"].values
    print(f"Evaluating model: {model_name}")
    print(f"Evaluating {len(texts)} samples...")

    # Load model with error handling
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model.eval()

    # Evaluate if labels provided
    if y is not None:
        # Process validation set
        desc = "Processing validation texts"
        predictions = _process_texts_llm(texts, model, tokenizer, desc)
        balanced_acc = balanced_accuracy_score(y, predictions)
        plot_confusion_matrix(y, predictions, f"{model_name} (LLM)")
        return balanced_acc, predictions

    # Process test set if provided (without validation labels)
    test_texts = df["text"].values
    desc = "Processing test texts"
    test_predictions = _process_texts_llm(test_texts, model, tokenizer, desc)
    _create_submission_file_direct(test_predictions, df, model_name)

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
    test_predictions = _process_texts_chatgpt(test_texts, client, model_name, desc)
    _create_submission_file(test_predictions, df, model_name, "chatgpt")

    return None, test_predictions
