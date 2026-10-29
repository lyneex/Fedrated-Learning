"""
src/evaluate.py -- Module 6: Evaluation metrics for binary cardiovascular risk prediction.
"""
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

import src.config as config


def evaluate_model(
    model: torch.nn.Module,
    dataset: TensorDataset,
    batch_size: int = 512,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate model on a TensorDataset.

    Supports both legacy 2-tuple (X, y) and multimodal 5-tuple
    (x_ehr, x_vitals, x_img, mask, y) dataset formats.

    Returns
    -------
    dict with keys:
        loss, accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
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

    probs = 1.0 / (1.0 + np.exp(-logits_np))   # sigmoid
    preds = (probs >= 0.5).astype(int)

    return {
        "loss":             total_loss / max(n_batches, 1),
        "accuracy":         float(accuracy_score(labels_np, preds)),
        "precision":        float(precision_score(labels_np, preds, zero_division=0)),
        "recall":           float(recall_score(labels_np, preds, zero_division=0)),
        "f1":               float(f1_score(labels_np, preds, zero_division=0)),
        "roc_auc":          float(roc_auc_score(labels_np, probs)),
        "confusion_matrix": confusion_matrix(labels_np, preds).tolist(),
    }


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
