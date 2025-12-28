"""
LLM (Large Language Model) evaluation functions
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import REVERSE_LABEL_MAPPING, SEED, SUBMISSIONS_DIR
from src.evaluation import plot_confusion_matrix
from src.submission import generate_submission
from src.utils import get_device

# Constants
DEFAULT_NEUTRAL_LABEL = "   Defaulting to neutral (0)"


def _get_sentiment_instruction() -> str:
    """Get the standardized instruction text for sentiment classification."""
    return (
        "Classify the financial sentiment as positive, neutral, or negative. Respond with only one word: positive, neutral, or negative."
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


def _label_to_string(label: int) -> str:
    """
    Convert numeric label to string label.

    Args:
        label: Numeric label (0=neutral, 1=positive, 2=negative)

    Returns:
        str: String label ("neutral", "positive", or "negative")
    """
    return REVERSE_LABEL_MAPPING.get(label, "neutral")


def _select_few_shot_examples(
    texts: Union[np.ndarray, list[str]],
    labels: Union[np.ndarray, list[int], pd.Series],
    num_per_class: int = 2,
) -> List[Tuple[str, str]]:
    """
    Auto-select balanced examples from data for few-shot learning.

    Args:
        texts: Array of text strings
        labels: Array of numeric labels (0=neutral, 1=positive, 2=negative)
        num_per_class: Number of examples to select per class (default: 2)

    Returns:
        List of tuples (text, label_str) where label_str is "positive",
            "neutral", or "negative"
    """
    # Convert to numpy arrays for easier indexing
    if not isinstance(texts, np.ndarray):
        texts_array = np.array(texts)
    else:
        texts_array = texts
    if not isinstance(labels, np.ndarray):
        labels_array = np.array(labels)
    else:
        labels_array = labels

    examples = []
    for label_value in [0, 1, 2]:  # neutral, positive, negative
        # Find indices of examples with this label
        label_indices = np.nonzero(labels_array == label_value)[0]

        if len(label_indices) == 0:
            continue  # Skip if no examples of this class

        # Randomly select up to num_per_class examples
        num_to_select = min(num_per_class, len(label_indices))
        # Use seed for reproducibility
        random.seed(SEED)
        selected_indices = random.sample(list(label_indices), num_to_select)

        # Add selected examples
        label_str = _label_to_string(label_value)
        for idx in selected_indices:
            examples.append((str(texts_array[idx]), label_str))

    return examples


def select_misclassification_examples(
    texts: Union[np.ndarray, list[str], pd.Series],
    predictions: Union[np.ndarray, list[int], pd.Series],
    actual_labels: Union[np.ndarray, list[int], pd.Series],
    num_per_pair: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Select few-shot examples from misclassifications (wrong predictions).

    Selects num_per_pair examples for each unique (predicted_label,
    actual_label) combination where the prediction was wrong. This is useful
    for few-shot learning when you want to use hard examples that another
    model struggled with.

    Args:
        texts: Array of text strings
        predictions: Array of predicted labels (numeric: 0, 1, or 2)
        actual_labels: Array of actual/correct labels (numeric: 0, 1, or 2)
        num_per_pair: Number of examples to select per (predicted, actual)
            pair. If None, defaults to 1. If a pair has fewer examples
            available, all available examples will be selected.

    Returns:
        List of tuples (text, actual_label_str) where actual_label_str is
            the correct label string ("neutral", "positive", or "negative").
            Returns up to 6 * num_per_pair examples (num_per_pair per wrong
            combination). If some combinations don't exist, returns only
            existing combinations.
    """
    # Set default value if None
    if num_per_pair is None:
        num_per_pair = 1

    # Convert to numpy arrays for easier indexing
    if isinstance(texts, pd.Series):
        texts_array = texts.values
    elif not isinstance(texts, np.ndarray):
        texts_array = np.array(texts)
    else:
        texts_array = texts

    if isinstance(predictions, pd.Series):
        predictions_array = predictions.values
    elif not isinstance(predictions, np.ndarray):
        predictions_array = np.array(predictions)
    else:
        predictions_array = predictions

    if isinstance(actual_labels, pd.Series):
        actual_labels_array = actual_labels.values
    elif not isinstance(actual_labels, np.ndarray):
        actual_labels_array = np.array(actual_labels)
    else:
        actual_labels_array = actual_labels

    # Find misclassified examples (where prediction != actual)
    misclassified_mask = predictions_array != actual_labels_array
    misclassified_indices = np.nonzero(misclassified_mask)[0]

    if len(misclassified_indices) == 0:
        return []  # No misclassifications found

    # Group misclassified examples by (predicted_label, actual_label) pairs
    examples_by_pair = {}
    for idx in misclassified_indices:
        predicted = int(predictions_array[idx])
        actual = int(actual_labels_array[idx])
        pair = (predicted, actual)

        if pair not in examples_by_pair:
            examples_by_pair[pair] = []
        examples_by_pair[pair].append(idx)

    # Select num_per_pair examples from each (predicted, actual) combination
    selected_examples = []
    random.seed(SEED)  # Set seed once before the loop
    for (_, actual_label), indices in examples_by_pair.items():
        # Randomly select num_per_pair examples from this combination
        # (or all available if fewer than num_per_pair)
        num_to_select = min(num_per_pair, len(indices))
        selected_indices = random.sample(indices, num_to_select)

        # Use the actual (correct) label for the example
        actual_label_str = _label_to_string(actual_label)
        for selected_idx in selected_indices:
            text = str(texts_array[selected_idx])
            selected_examples.append((text, actual_label_str))

    return selected_examples


def _format_few_shot_prompt(
    examples: List[Tuple[str, str]], instruction: str, query_text: str
) -> List[dict[str, str]]:
    """
    Format few-shot examples into chat messages with examples in system msg.

    Args:
        examples: List of (text, label) tuples for few-shot examples
        instruction: System instruction text
        query_text: The text to classify

    Returns:
        List of message dictionaries formatted for chat template
    """
    # Build system message with instruction and examples
    system_parts = [instruction]
    system_parts.append("\n\nExamples with output separated by |:")
    for i, (example_text, example_label) in enumerate(examples):
        system_parts.append(f'Example {i+1}: Input: "{example_text}" | Output: {example_label}')
    system_parts.append("\nNow classify the input text following the examples above.")

    system_content = "\n".join(system_parts)

    # Create messages: system with examples, then user with query
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query_text},
    ]

    return messages


def _format_chatgpt_instructions_with_examples(
    examples: List[Tuple[str, str]], base_instruction: str
) -> str:
    """
    Format few-shot examples into ChatGPT instructions text.

    Args:
        examples: List of (text, label) tuples for few-shot examples
        base_instruction: Base instruction text

    Returns:
        Formatted instruction string with examples
    """
    instruction_parts = [base_instruction]
    instruction_parts.append(
        "\n\nExamples with output separated by |:"
    )
    for example_text, example_label in examples:
        instruction_parts.append(
            f'Input: "{example_text}" | Output: {example_label}'
        )
    instruction_parts.append(
        "\nNow classify the input text following the examples above."
    )

    return "\n".join(instruction_parts)


def _process_single_text_llm(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> int:
    """
    Process a single text with LLM model.

    Args:
        text: Text to classify
        model: Loaded LLM model
        tokenizer: Loaded tokenizer
        few_shot_examples: Optional list of (text, label) tuples for few-shot
            learning. If None, uses zero-shot learning.

    Returns:
        int: Numeric label (0=neutral, 1=positive, 2=negative)
    """
    try:
        system_msg = _get_sentiment_instruction()

        if few_shot_examples:
            messages = _format_few_shot_prompt(
                few_shot_examples, system_msg, str(text)
            )
        else:
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
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> np.ndarray:
    """
    Process multiple texts with LLM model.

    Args:
        texts: Array of texts to process
        model: Loaded LLM model
        tokenizer: Loaded tokenizer
        desc: Description for progress bar
        few_shot_examples: Optional list of (text, label) tuples for few-shot
            learning. If None, uses zero-shot learning. Examples are reused
            across all texts.

    Returns:
        np.ndarray: Array of numeric labels
    """
    predictions = []
    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
            pred_label = _process_single_text_llm(
                text, model, tokenizer, few_shot_examples
            )
            predictions.append(pred_label)
    return np.array(predictions)


def _process_single_text_chatgpt(
    text: str,
    client: OpenAI,
    model_name: str,
    max_retries: int = 10,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> int:
    """
    Process a single text with ChatGPT API.

    Args:
        text: Text to classify
        client: OpenAI client instance
        model_name: Name of the ChatGPT model
        max_retries: Maximum number of retry attempts
        few_shot_examples: Optional list of (text, label) tuples for few-shot
            learning. If None, uses zero-shot learning.

    Returns:
        int: Numeric label (0=neutral, 1=positive, 2=negative)
    """
    retry_count = 0
    base_instruction = _get_sentiment_instruction()

    if few_shot_examples:
        instructions_text = _format_chatgpt_instructions_with_examples(
            few_shot_examples, base_instruction
        )
    else:
        instructions_text = base_instruction

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
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> np.ndarray:
    """
    Process multiple texts with ChatGPT API.

    Args:
        texts: Array of texts to process
        client: OpenAI client instance
        model_name: Name of the ChatGPT model
        desc: Description for progress bar
        max_retries: Maximum number of retry attempts per text
        few_shot_examples: Optional list of (text, label) tuples for few-shot
            learning. If None, uses zero-shot learning. Examples are reused
            across all texts.

    Returns:
        np.ndarray: Array of numeric labels
    """
    predictions = []
    for text in tqdm(texts, desc=desc):
        pred_label = _process_single_text_chatgpt(
            text, client, model_name, max_retries, few_shot_examples
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


def evaluate_llm(
    model_name: str,
    df: pd.DataFrame,
    y: Optional[Union[np.ndarray, pd.Series, list[int]]] = None,
    df_train: Optional[pd.DataFrame] = None,
    y_train: Optional[Union[np.ndarray, pd.Series, list[int]]] = None,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
    num_few_shot_per_class: int = 2,
) -> Tuple[Optional[float], np.ndarray]:
    """
    Evaluate an LLM model on a dataset.

    Args:
        model_name: Name/path of the LLM model
        df: DataFrame with 'text' column
        y: Optional labels
        df_train: Optional training DataFrame for auto-selecting
            few-shot examples
        y_train: Optional training labels for auto-selecting
            few-shot examples
        few_shot_examples: Optional list of (text, label) tuples for few-shot
            learning. If provided, takes precedence over auto-selection.
        num_few_shot_per_class: Number of examples per class for auto-selection
            (default: 2). Only used if few_shot_examples is None and df_train
            and y_train are provided.

    Returns:
        tuple: (balanced_accuracy, predictions) or (None, predictions)
            if y is None
    """
    # Validate inputs
    if df is None or df.empty:
        raise ValueError("df_val cannot be None or empty")
    if "text" not in df.columns:
        raise ValueError("df must contain a 'text' column")

    # Determine few-shot examples
    selected_examples = None
    if few_shot_examples is not None:
        selected_examples = few_shot_examples
    elif df_train is not None and y_train is not None:
        if "text" not in df_train.columns:
            raise ValueError("df_train must contain a 'text' column")
        train_texts = df_train["text"].values
        selected_examples = _select_few_shot_examples(
            train_texts, y_train, num_few_shot_per_class
        )
        if selected_examples:
            print(
                f"Using {len(selected_examples)} few-shot examples "
                f"({num_few_shot_per_class} per class)"
            )

    texts = df["text"].values
    print(f"Evaluating model: {model_name}")
    print(f"Evaluating {len(texts)} samples...")

    # Load model with error handling
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

    # Evaluate if labels provided
    if y is not None:
        # Process validation set
        desc = "Processing validation texts"
        predictions = _process_texts_llm(
            texts, model, tokenizer, desc, selected_examples
        )
        balanced_acc = balanced_accuracy_score(y, predictions)
        plot_confusion_matrix(y, predictions, f"{model_name} (LLM)")
        return balanced_acc, predictions

    # Process test set if provided (without validation labels)
    test_texts = df["text"].values
    desc = "Processing test texts"
    test_predictions = _process_texts_llm(
        test_texts, model, tokenizer, desc, selected_examples
    )
    _create_submission_file_direct(test_predictions, df, model_name)

    return None, test_predictions


def evaluate_chatgpt(
    model_name: str,
    df: pd.DataFrame,
    y: Optional[Union[np.ndarray, pd.Series, list[int]]] = None,
    df_train: Optional[pd.DataFrame] = None,
    y_train: Optional[Union[np.ndarray, pd.Series, list[int]]] = None,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
    num_few_shot_per_class: int = 2,
) -> Tuple[Optional[float], np.ndarray]:
    """
    Evaluate ChatGPT model using OpenAI API.

    Args:
        model_name: Name of the ChatGPT model (e.g., "gpt-5.2")
        df: DataFrame with 'text' column
        y: Optional labels
        df_train: Optional training DataFrame for auto-selecting
            few-shot examples
        y_train: Optional training labels for auto-selecting
            few-shot examples
        few_shot_examples: Optional list of (text, label) tuples for few-shot
            learning. If provided, takes precedence over auto-selection.
        num_few_shot_per_class: Number of examples per class for auto-selection
            (default: 2). Only used if few_shot_examples is None and df_train
            and y_train are provided.

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

    # Determine few-shot examples
    selected_examples = None
    if few_shot_examples is not None:
        selected_examples = few_shot_examples
    elif df_train is not None and y_train is not None:
        if "text" not in df_train.columns:
            raise ValueError("df_train must contain a 'text' column")
        train_texts = df_train["text"].values
        selected_examples = _select_few_shot_examples(
            train_texts, y_train, num_few_shot_per_class
        )
        if selected_examples:
            print(
                f"Using {len(selected_examples)} few-shot examples "
                f"({num_few_shot_per_class} per class)"
            )

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
        predictions = _process_texts_chatgpt(
            texts,
            client,
            model_name,
            desc,
            max_retries=10,
            few_shot_examples=selected_examples,
        )
        balanced_acc = balanced_accuracy_score(y, predictions)
        plot_confusion_matrix(y, predictions, f"{model_name} (ChatGPT)")
        return balanced_acc, predictions

    # Process test set if provided (without validation labels)
    test_texts = df["text"].values
    desc = "Processing test texts"
    test_predictions = _process_texts_chatgpt(
        test_texts,
        client,
        model_name,
        desc,
        max_retries=10,
        few_shot_examples=selected_examples,
    )
    _create_submission_file(test_predictions, df, model_name, "chatgpt")

    return None, test_predictions
