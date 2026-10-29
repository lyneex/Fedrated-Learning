"""
baseline.py -- Phase 1: Centralized Baseline Training.

Trains one model on the entire training set (all hospital data combined) to
establish a performance ceiling before federated learning.

Selects tabular, image, or real multimodal training through config.DATA_MODE.

Usage
-----
    python baseline.py

Outputs (under runs/baseline/)
-------------------------------
    baseline_model.pt        -- trained model weights
    metrics.json             -- accuracy, precision, recall, F1, ROC-AUC, etc.
    training_history.png     -- loss & accuracy curves per epoch
    confusion_matrix.png     -- predicted vs actual heatmap
"""
import os
import sys
import copy

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import src.config as config
from src.data_utils import prepare_selected_data, split_client_validation_sets, training_pos_weight
from src.model import CardioMLP, save_checkpoint, build_model
from src.evaluate import collect_predictions, evaluate_model, find_best_threshold, print_report, save_metrics
from src.visualize import plot_baseline_history, plot_confusion_matrix


# -- Training loop -------------------------------------------------------------

def train_centralized(
    model,
    train_dataset,
    validation_dataset,
    n_epochs: int,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> dict:
    """
    Train model on the full training dataset for n_epochs.
    Supports both legacy 2-tuple and multimodal 5-tuple datasets.

    Returns
    -------
    history : dict with keys 'loss' and 'accuracy' (one value per epoch)
    """
    loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"loss": [], "accuracy": [], "val_roc_auc": []}
    best_auc = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    remaining_patience = config.EARLY_STOPPING_PATIENCE

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for batch in loader:
            if config.DATA_MODE == "multimodal":
                # Multimodal: (x_ehr, x_vitals, x_img, mask, y)
                x_ehr, x_vit, x_img, mask, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()
                logits = model(x_ehr, x_vit, x_img, mask)
            else:
                # Legacy: (X, y)
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)
                optimizer.zero_grad()
                logits = model(X_batch)

            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds       = (torch.sigmoid(logits) >= 0.5).long()
            correct    += (preds == y_batch.long()).sum().item()
            total      += len(y_batch)

        epoch_loss = total_loss / total
        epoch_acc  = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        val_probs, val_labels, _ = collect_predictions(model, validation_dataset, device=str(device))
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = float(roc_auc_score(val_labels, val_probs))
        except ValueError:
            val_auc = float("nan")
        history["val_roc_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = config.EARLY_STOPPING_PATIENCE
        else:
            remaining_patience -= 1

        print(
            f"  Epoch {epoch:02d}/{n_epochs} | "
            f"loss={epoch_loss:.4f} | acc={epoch_acc:.4f} | val_auc={val_auc:.4f}"
        )
        if remaining_patience == 0:
            print(f"  Early stopping after {config.EARLY_STOPPING_PATIENCE} epochs without validation AUC improvement.")
            break

    model.load_state_dict(best_state)
    history["best_val_roc_auc"] = best_auc
    return history


# -- Entry point ---------------------------------------------------------------

def main() -> dict:
    print("\n" + "=" * 60)
    print("  FEDERATED CARDIO -- Centralized Baseline")
    print("=" * 60)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Baseline] Device     : {device}")
    print(f"[Baseline] Epochs     : {config.BASELINE_EPOCHS}")
    print(f"[Baseline] Batch      : {config.BATCH_SIZE}")
    print(f"[Baseline] LR         : {config.LEARNING_RATE}")
    print(f"[Baseline] Data mode  : {config.DATA_MODE}")

    # -- Data -----------------------------------------------------------------
    hospital_datasets, test_dataset, scaler, n_features = prepare_selected_data()
    train_clients, validation_dataset = split_client_validation_sets(hospital_datasets)

    # Combine all hospital partitions -> single training set for centralized run
    full_train = ConcatDataset(train_clients)
    print(f"\n[Baseline] Total training samples : {len(full_train):,}")
    print(f"[Baseline] Test samples           : {len(test_dataset):,}")
    print(f"[Baseline] Validation samples     : {len(validation_dataset):,}")
    print(f"[Baseline] Feature count          : {n_features}")

    # -- Model -----------------------------------------------------------------
    model = build_model(n_features).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Baseline] Model: {type(model).__name__}")
    print(f"[Baseline] Total parameters: {total_params:,}")

    # -- Train -----------------------------------------------------------------
    print(f"\n[Baseline] Training for {config.BASELINE_EPOCHS} epochs ...")
    pos_weight = training_pos_weight(train_clients, device)
    history = train_centralized(
        model, full_train, validation_dataset, config.BASELINE_EPOCHS, device, pos_weight
    )

    # -- Persist model ---------------------------------------------------------
    os.makedirs(config.BASELINE_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.BASELINE_DIR, "baseline_model.pt")
    save_checkpoint(model, ckpt_path)
    print(f"\n[Baseline] Model saved -> {ckpt_path}")

    # -- Evaluate --------------------------------------------------------------
    val_probs, val_labels, _ = collect_predictions(model, validation_dataset, device=str(device))
    threshold = find_best_threshold(val_labels, val_probs)
    print(f"[Baseline] Validation-selected F1 threshold: {threshold:.2f}")
    metrics = evaluate_model(model, test_dataset, device=str(device), threshold=threshold)
    metrics["best_validation_roc_auc"] = history["best_val_roc_auc"]
    print_report(metrics.copy(), title="Centralized Baseline -- Test Set")
    save_metrics(metrics, os.path.join(config.BASELINE_DIR, "metrics.json"))

    # -- Plots -----------------------------------------------------------------
    plot_baseline_history(
        history,
        os.path.join(config.BASELINE_DIR, "training_history.png"),
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(config.BASELINE_DIR, "confusion_matrix.png"),
        title="Centralized Baseline -- Confusion Matrix",
    )

    print(f"\n[Baseline] All results saved to: {config.BASELINE_DIR}")
    return metrics


if __name__ == "__main__":
    main()
