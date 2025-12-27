"""
Classical Machine Learning models (Logistic Regression, Random Forest, SVM, XGBoost)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.config import SEED
from src.evaluation import plot_confusion_matrix


def train_and_eval(
    model: Union[
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        xgb.XGBClassifier,
        VotingClassifier,
    ],
    name: str,
    X_tr: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_tr: Union[np.ndarray, list[int]],
    X_v: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_v: Union[np.ndarray, list[int]],
    sample_weights: Optional[np.ndarray] = None,
    results_dict: Optional[Dict[str, float]] = None,
) -> Union[
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    xgb.XGBClassifier,
    VotingClassifier,
]:
    """
    Train and evaluate a classical ML model.

    Args:
        model: Scikit-learn model instance
        name: Name of the model
        X_tr: Training features
        y_tr: Training labels
        X_v: Validation features
        y_v: Validation labels
        sample_weights: Optional sample weights
        results_dict: Optional dictionary to store results

    Returns:
        Trained model
    """
    print(f"\n--- Training {name} ---")
    if sample_weights is not None:
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
    else:
        model.fit(X_tr, y_tr)

    y_pred = model.predict(X_v)
    acc = balanced_accuracy_score(y_v, y_pred)

    if results_dict is not None:
        results_dict[name] = acc

    print(f"> {name} Balanced Accuracy: {acc:.5f}")
    plot_confusion_matrix(y_v, y_pred, name)
    return model


def train_logistic_regression(
    X_train: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_train: Union[np.ndarray, list[int]],
    X_val: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_val: Union[np.ndarray, list[int]],
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
    results_dict: Optional[Dict[str, float]] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        class_weight: Class weight strategy ('balanced' or None)
        results_dict: Optional dictionary to store results
        sample_weights: Optional sample weights

    Returns:
        Trained LogisticRegression model
    """
    name = (
        "Logistic Regression"
        if class_weight == "balanced"
        else "Logistic Regression without class balancing"
    )
    model = LogisticRegression(
        class_weight=class_weight, max_iter=1000, random_state=SEED
    )
    return train_and_eval(
        model,
        name,
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weights=sample_weights,
        results_dict=results_dict,
    )


def train_xgboost(
    X_train: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_train: Union[np.ndarray, list[int]],
    X_val: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_val: Union[np.ndarray, list[int]],
    results_dict: Optional[Dict[str, float]] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        results_dict: Optional dictionary to store results
        sample_weights: Optional sample weights

    Returns:
        Trained XGBClassifier model
    """
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=SEED,
        eval_metric="mlogloss",
    )
    return train_and_eval(
        model,
        "XGBoost",
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weights=sample_weights,
        results_dict=results_dict,
    )


def train_random_forest(
    X_train: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_train: Union[np.ndarray, list[int]],
    X_val: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_val: Union[np.ndarray, list[int]],
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
    results_dict: Optional[Dict[str, float]] = None,
    cv: int = 3,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Train a Random Forest model with Grid Search.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        class_weight: Class weight strategy ('balanced' or None)
        results_dict: Optional dictionary to store results
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs

    Returns:
        Best RandomForestClassifier from grid search
    """
    name = (
        "Random Forest Tuned"
        if class_weight == "balanced"
        else "Random Forest Tuned without class balancing"
    )

    print("\n--- Optimizing Random Forest ---")
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [class_weight],
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED),
        rf_params,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=n_jobs,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    acc = balanced_accuracy_score(y_val, best_rf.predict(X_val))
    if results_dict is not None:
        results_dict[name] = acc

    plot_confusion_matrix(y_val, best_rf.predict(X_val), name)
    print("Evaluating on validation set...")
    print(f"> Best RF Params: {rf_grid.best_params_}")
    print(f"> RF Tuned Acc: {acc:.5f}")

    return best_rf


def train_svm(
    X_train: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_train: Union[np.ndarray, list[int]],
    X_val: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_val: Union[np.ndarray, list[int]],
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
    results_dict: Optional[Dict[str, float]] = None,
    cv: int = 3,
    n_jobs: int = -1,
) -> SVC:
    """
    Train an SVM model with Grid Search.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        class_weight: Class weight strategy ('balanced' or None)
        results_dict: Optional dictionary to store results
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs

    Returns:
        Best SVC from grid search
    """
    name = (
        "SVM Tuned"
        if class_weight == "balanced"
        else "SVM Tuned without class balancing"
    )

    print("\n--- Optimizing SVM ---")
    svm_params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "class_weight": [class_weight],
    }

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=SEED),
        svm_params,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=n_jobs,
    )
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_

    acc = balanced_accuracy_score(y_val, best_svm.predict(X_val))
    if results_dict is not None:
        results_dict[name] = acc

    plot_confusion_matrix(y_val, best_svm.predict(X_val), name)
    print("Evaluating on validation set...")
    print(f"> Best SVM Params: {svm_grid.best_params_}")
    print(f"> SVM Tuned Acc: {acc:.5f}")

    return best_svm


def train_voting_classifier(
    estimators: List[
        Tuple[
            str,
            Union[LogisticRegression, RandomForestClassifier, SVC, xgb.XGBClassifier],
        ]
    ],
    X_train: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_train: Union[np.ndarray, list[int]],
    X_val: Union[np.ndarray, scipy.sparse.csr_matrix],
    y_val: Union[np.ndarray, list[int]],
    results_dict: Optional[Dict[str, float]] = None,
    sample_weights: Optional[np.ndarray] = None,
    voting: str = "soft",
    name: Optional[str] = None,
    plot_cm: bool = True,
) -> VotingClassifier:
    """
    Train a Voting Classifier ensemble.

    Args:
        estimators: List of (name, model) tuples
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        results_dict: Optional dictionary to store results
        sample_weights: Optional sample weights
        voting: Voting strategy ('soft' or 'hard')
        name: Optional custom name for the model (default: "Voting Ensemble")
        plot_cm: Whether to plot confusion matrix (default: True)

    Returns:
        Trained VotingClassifier
    """
    voting_clf = VotingClassifier(estimators=estimators, voting=voting)
    voting_clf.fit(X_train, y_train, sample_weight=sample_weights)

    acc = balanced_accuracy_score(y_val, voting_clf.predict(X_val))
    if name is None:
        name = "Voting Ensemble"

    if results_dict is not None and len(results_dict) > 0:
        results_dict[name] = acc

    print(f"> {name} Acc: {acc:.5f}")
    if plot_cm:
        plot_confusion_matrix(y_val, voting_clf.predict(X_val), name)

    return voting_clf
