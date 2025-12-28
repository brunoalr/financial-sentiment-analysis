"""
Text preprocessing and vectorization functions
"""

from __future__ import annotations

import re
from typing import Optional, Set, Tuple

import numpy as np
import nltk
import scipy.sparse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src import config

# Download stopwords (if necessary)
try:
    nltk.download("stopwords", quiet=True)
    stop_words: Set[str] = set(stopwords.words("english"))
except Exception as e:
    print(
        f"Warning: Could not download stopwords. Using empty or default list. Error: {e}"
    )
    stop_words: Set[str] = set()


def clean_text(text: str, stop_words_set: Optional[Set[str]] = None) -> str:
    """
    Clean text by converting to lowercase, removing special characters,
    and removing stopwords.

    Args:
        text: Input text string
        stop_words_set: Set of stopwords to remove (defaults to English stopwords)

    Returns:
        Cleaned text string
    """
    if stop_words_set is None:
        stop_words_set = stop_words

    text = str(text).lower()  # Lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Letters only
    tokens = [w for w in text.split() if w not in stop_words_set]  # Remove stopwords
    return " ".join(tokens)


def create_tfidf_vectorizer(max_features: Optional[int] = None) -> TfidfVectorizer:
    """
    Create and return a TF-IDF vectorizer.

    Args:
        max_features: Maximum number of features to use. If None, uses TFIDF_MAX_FEATURES from config.

    Returns:
        TfidfVectorizer instance
    """
    max_features = max_features if max_features is not None else config.TFIDF_MAX_FEATURES
    return TfidfVectorizer(max_features=max_features)


def fit_transform_tfidf(
    vectorizer: TfidfVectorizer,
    train_texts: list[str],
    val_texts: Optional[list[str]] = None,
    test_texts: Optional[list[str]] = None,
) -> Tuple[
    scipy.sparse.csr_matrix,
    Optional[scipy.sparse.csr_matrix],
    Optional[scipy.sparse.csr_matrix],
]:
    """
    Fit TF-IDF vectorizer on training data and transform all datasets.

    Args:
        vectorizer: TfidfVectorizer instance
        train_texts: Training text data
        val_texts: Optional validation text data
        test_texts: Optional test text data

    Returns:
        tuple: (X_train_tfidf, X_val_tfidf, X_test_tfidf) transformed data
    """
    X_train_tfidf = vectorizer.fit_transform(train_texts)

    X_val_tfidf = None
    X_test_tfidf = None

    if val_texts is not None:
        X_val_tfidf = vectorizer.transform(val_texts)

    if test_texts is not None:
        X_test_tfidf = vectorizer.transform(test_texts)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf


def create_lstm_tokenizer(max_nb_words: Optional[int] = None) -> Tokenizer:
    """
    Create and return an LSTM tokenizer.

    Args:
        max_nb_words: Maximum number of words to keep. If None, uses MAX_NB_WORDS from config.

    Returns:
        Tokenizer instance
    """
    max_nb_words = max_nb_words if max_nb_words is not None else config.MAX_NB_WORDS
    return Tokenizer(num_words=max_nb_words, lower=True)


def fit_transform_lstm(
    tokenizer: Tokenizer,
    train_texts: list[str],
    val_texts: Optional[list[str]] = None,
    test_texts: Optional[list[str]] = None,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit LSTM tokenizer on training data and transform all datasets.

    Args:
        tokenizer: Tokenizer instance
        train_texts: Training text data
        val_texts: Optional validation text data
        test_texts: Optional test text data
        max_len: Maximum sequence length. If None, uses MAX_LEN from config.

    Returns:
        tuple: (X_train_seq, X_val_seq, X_test_seq) padded sequences
    """
    max_len = max_len if max_len is not None else config.MAX_LEN
    tokenizer.fit_on_texts(train_texts)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(train_texts), maxlen=max_len
    )

    X_val_seq = None
    X_test_seq = None

    if val_texts is not None:
        X_val_seq = pad_sequences(
            tokenizer.texts_to_sequences(val_texts), maxlen=max_len
        )

    if test_texts is not None:
        X_test_seq = pad_sequences(
            tokenizer.texts_to_sequences(test_texts), maxlen=max_len
        )

    return X_train_seq, X_val_seq, X_test_seq
