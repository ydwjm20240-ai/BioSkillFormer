"""
Evaluation Metrics for Multimodal Ordinal Motor Skill Assessment.
Computes Accuracy, Macro-F1, Balanced Accuracy, Ordinal MAE, and Quadratic Weighted Kappa.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)


def compute_ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE across ordinal discrete levels: MAE = (1 / N) * sum(|y_true - y_pred|)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    ordinal_mae = compute_ordinal_mae(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "ordinal_mae": float(ordinal_mae),
        "qwk": float(qwk),
        "confusion_matrix": cm
    }


def ordinal_prediction_from_thresholds(ordinal_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Derives discrete rank predictions from threshold probabilities q_k = P(y > k)."""
    decisions = (ordinal_probs > threshold).astype(int)
    return np.sum(decisions, axis=1)