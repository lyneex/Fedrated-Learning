"""
src/evaluate.py -- Module 6: Evaluation metrics for binary cardiovascular risk prediction.
"""
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

import src.config as config


def collect_predictions(
    model: torch.nn.Module,
    dataset: TensorDataset,
    batch_size: int = 512,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return probabilities, labels, and mean unweighted BCE loss."""
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss()

    all_logits: list = []
    all_labels: list = []
    total_loss  = 0.0
    n_batches   = 0

    with torch.no_grad():
        for batch in loader:
            if config.DATA_MODE == "multimodal":
                # Multimodal: (x_ehr, x_vitals, x_img, mask, y)
                x_ehr, x_vit, x_img, mask, y_batch = [b.to(device) for b in batch]
                logits = model(x_ehr, x_vit, x_img, mask)
            else:
                # Legacy: (X, y)
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)
                logits = model(X_batch)

            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            n_batches  += 1

            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)

    probs = 1.0 / (1.0 + np.exp(-logits_np))
    return probs, labels_np, total_loss / max(n_batches, 1)


def find_best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation threshold that maximizes binary F1 score."""
    thresholds = np.arange(
        config.THRESHOLD_START,
        config.THRESHOLD_STOP + config.THRESHOLD_STEP / 2,
        config.THRESHOLD_STEP,
    )
    scores = [
        f1_score(labels, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def metrics_from_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    loss: float,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate binary metrics for probabilities at a fixed threshold."""
    preds = (probabilities >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")

    return {
        "loss":             loss,
        "accuracy":         float(accuracy_score(labels, preds)),
        "precision":        float(precision_score(labels, preds, zero_division=0)),
        "recall":           float(recall_score(labels, preds, zero_division=0)),
        "f1":               float(f1_score(labels, preds, zero_division=0)),
        "roc_auc":          auc,
        "threshold":        threshold,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: TensorDataset,
    batch_size: int = 512,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a model using a fixed decision threshold."""
    probabilities, labels, loss = collect_predictions(model, dataset, batch_size, device)
    return metrics_from_predictions(probabilities, labels, loss, threshold)


def print_report(metrics: Dict[str, Any], title: str = "Evaluation Report") -> None:
    """Pretty-print a metrics dict to the console."""
    cm = metrics.get("confusion_matrix", None)
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Loss       : {metrics['loss']:.4f}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1-Score   : {metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Threshold  : {metrics.get('threshold', 0.5):.2f}")
    if cm is not None:
        print(f"  Confusion Matrix (rows=Actual, cols=Predicted):")
        print(f"               Pred 0    Pred 1")
        print(f"    Actual 0   {cm[0][0]:>6,}    {cm[0][1]:>6,}  (TN / FP)")
        print(f"    Actual 1   {cm[1][0]:>6,}    {cm[1][1]:>6,}  (FN / TP)")
    print(sep + "\n")


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Persist metrics as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Metrics saved -> {path}")
