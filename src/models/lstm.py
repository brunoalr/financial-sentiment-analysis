"""
LSTM model implementation for financial sentiment analysis
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import keras
from sklearn.metrics import balanced_accuracy_score
from keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    SpatialDropout1D,
)
from keras.models import Sequential
from keras.utils import to_categorical

from src import utils
from src.config import EMBEDDING_DIM, MAX_NB_WORDS
from src.evaluation import plot_confusion_matrix


def create_lstm_model(
    max_nb_words: int = MAX_NB_WORDS,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
) -> Sequential:
    """
    Create a bidirectional LSTM model.

    Args:
        max_nb_words: Maximum number of words in vocabulary
        embedding_dim: Embedding dimension
        lstm_units: Number of LSTM units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential(
        [
            Embedding(max_nb_words, embedding_dim),
            SpatialDropout1D(dropout_rate),
            Bidirectional(
                LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)
            ),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_rate),
            Dense(3, activation="softmax"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def train_lstm(
    model: Sequential,
    X_train_seq: np.ndarray,
    y_train_cat: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_cat: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    verbose: int = 1,
) -> keras.callbacks.History:
    """
    Train an LSTM model.

    Args:
        model: Keras Sequential model
        X_train_seq: Training sequences
        y_train_cat: Training labels (one-hot encoded)
        X_val_seq: Validation sequences
        y_val_cat: Validation labels (one-hot encoded)
        epochs: Number of training epochs
        batch_size: Batch size
        verbose: Verbosity level

    Returns:
        keras.callbacks.History: Training history object
    """
    history = model.fit(
        X_train_seq,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_cat),
        verbose=verbose,
    )

    return history


def evaluate_lstm(
    model: Sequential,
    X_val_seq: np.ndarray,
    y_val: Union[np.ndarray, list[int]],
    name: str = "Bidirectional LSTM",
    results_dict: Optional[Dict[str, float]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate an LSTM model.

    Args:
        model: Trained Keras Sequential model
        X_val_seq: Validation sequences
        y_val: Validation labels (numeric)
        name: Name of the model
        results_dict: Optional dictionary to store results

    Returns:
        tuple: (balanced_accuracy, predictions)
    """
    y_pred_prob = model.predict(X_val_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = balanced_accuracy_score(y_val, y_pred)

    if results_dict is not None:
        results_dict[name] = acc

    print(f"> {name} Acc: {acc:.5f}")
    plot_confusion_matrix(y_val, y_pred, name)

    return acc, y_pred


def train_and_evaluate_lstm(
    X_train_seq: np.ndarray,
    y_train: Union[np.ndarray, list[int]],
    X_val_seq: np.ndarray,
    y_val: Union[np.ndarray, list[int]],
    epochs: int = 10,
    batch_size: int = 64,
    results_dict: Optional[Dict[str, float]] = None,
    max_nb_words: int = MAX_NB_WORDS,
    embedding_dim: int = EMBEDDING_DIM,
) -> Tuple[Sequential, keras.callbacks.History, float, np.ndarray]:
    """
    Train and evaluate an LSTM model end-to-end.

    Args:
        X_train_seq: Training sequences
        y_train: Training labels (numeric)
        X_val_seq: Validation sequences
        y_val: Validation labels (numeric)
        epochs: Number of training epochs
        batch_size: Batch size
        results_dict: Optional dictionary to store results
        max_nb_words: Maximum number of words in vocabulary
        embedding_dim: Embedding dimension

    Returns:
        tuple: (trained_model: Sequential, history: keras.callbacks.History,
            balanced_accuracy: float, predictions: np.ndarray)
    """
    # One-hot encoding of targets
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)

    # Create and train model
    model = create_lstm_model(max_nb_words=max_nb_words, embedding_dim=embedding_dim)
    print(model.summary())

    history = train_lstm(
        model,
        X_train_seq,
        y_train_cat,
        X_val_seq,
        y_val_cat,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Evaluate
    acc, y_pred = evaluate_lstm(
        model,
        X_val_seq,
        y_val,
        name="Bidirectional LSTM",
        results_dict=results_dict,
    )

    utils.plot_loss(history, "LSTM")

    return model, history, acc, y_pred
