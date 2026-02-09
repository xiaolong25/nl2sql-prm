import numpy as np
from sklearn.metrics import roc_auc_score

# evaluation/metrics.py

from typing import List, Optional


def find_first_error_from_labels(labels: List[int]) -> Optional[int]:
    """
    Find the index of the first error step from ground-truth labels.

    labels:
        List[int], where
        - 1 means correct step
        - 0 means error step

    Returns:
        index of first error step, or None if no error exists
    """
    for i, label in enumerate(labels):
        if label == 0:
            return i
    return None


def find_first_error_from_scores(
    scores: List[float],
    threshold: float = 0.5
) -> Optional[int]:
    """
    Find the index of the first predicted error step from model scores.

    scores:
        List[float], higher means more confident correct step

    threshold:
        score < threshold is considered an error

    Returns:
        index of first predicted error step, or None if no error exists
    """
    for i, score in enumerate(scores):
        if score < threshold:
            return i
    return None


def step_auc(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)

def step_accuracy(scores, labels, threshold=0.5):
    scores = np.array(scores)
    labels = np.array(labels)
    preds = (scores >= threshold).astype(int)
    return (preds == labels).mean()

def false_negative_rate(scores, labels, threshold=0.5):
    scores = np.array(scores)
    labels = np.array(labels)
    preds = (scores >= threshold).astype(int)

    fn = ((preds == 1) & (labels == 0)).sum()
    total_neg = (labels == 0).sum()
    return fn / max(total_neg, 1)

def first_error_accuracy(step_scores, step_labels):
    """
    step_scores: List[List[float]]
    step_labels: List[List[int]]
    """
    correct = 0
    total = len(step_scores)

    for scores, labels in zip(step_scores, step_labels):
        pred_idx = next((i for i, s in enumerate(scores) if s < 0.5), None)
        gt_idx = next((i for i, l in enumerate(labels) if l == 0), None)

        if pred_idx == gt_idx:
            correct += 1

    return correct / max(total, 1)
