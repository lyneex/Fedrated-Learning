"""
federated_main.py - Phase 4: Federated Learning Simulation (pure Python, no Ray).

Implements a manual FedAvg loop that replicates Flower's simulation behaviour
without requiring the Ray distributed computing backend.  Each round:
  1. Server broadcasts global weights to all hospital clients.
  2. Each client trains locally and returns updated weights + sample count.
  3. Server aggregates via weighted FedAvg.
  4. Server evaluates the new global model on the shared test set.
  5. Server saves a per-round checkpoint.

Selects tabular, image, or real multimodal training through config.DATA_MODE.

Usage
-----
    python federated_main.py

Outputs (under runs/federated/)
---------------------------------
    final_metrics.json           - held-out test scores for the best validation model
    rounds_log.json              - per-round validation accuracy / loss / F1 / AUC
    accuracy_curve.png           - global accuracy over rounds
    loss_curve.png               - global BCE loss over rounds
    confusion_matrix.png         - final global model confusion matrix
    model_comparison.png         - bar chart: centralized vs local vs federated
    local_hospital_metrics.json  - per-hospital evaluation of the global model

    runs/checkpoints/
        round_01.pt ... round_10.pt  - global model after each round
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.config as config
from src.data_utils import prepare_selected_data, split_client_validation_sets, training_pos_weight
from src.model import CardioMLP, get_parameters, set_parameters, save_checkpoint, load_checkpoint, build_model
from src.evaluate import collect_predictions, evaluate_model, find_best_threshold, print_report, save_metrics
from src.visualize import (
    plot_accuracy_curve,
    plot_loss_curve,
    plot_confusion_matrix,
    plot_client_comparison,
)


# -- FedAvg helpers ------------------------------------------------------------

def fedavg_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes:   List[int],
) -> List[np.ndarray]:
    """
    Weighted average of client model parameters.
    Each client's contribution is proportional to its dataset size.
    """
    total = sum(client_sizes)
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = sum(
            client_weights[c][layer_idx] * (client_sizes[c] / total)
            for c in range(len(client_weights))
        )
        aggregated.append(layer_avg)
    return aggregated


def train_local(
    model,
    dataset,
    device: torch.device,
    local_epochs: int = config.LOCAL_EPOCHS,
    batch_size: int   = config.BATCH_SIZE,
    lr: float         = config.LEARNING_RATE,
    cid: str          = "?",
    fl_round: int     = 0,
    pos_weight: torch.Tensor | None = None,
) -> tuple:
    """
    Train model on local dataset for local_epochs.
    Supports both legacy 2-tuple and multimodal 5-tuple datasets.
    Returns (updated_weights, n_samples, avg_loss).
    """
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    total_loss = 0.0
    n_batches  = 0

    for _ in range(local_epochs):
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
            total_loss += loss.item()
            n_batches  += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(
        f"    [Hospital {cid}] Round {fl_round:02d} | "
        f"local_loss={avg_loss:.4f} | samples={len(dataset):,}"
    )
    return get_parameters(model), len(dataset), avg_loss


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  FEDERATED CARDIO - Federated Learning Simulation")
    print("=" * 60)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FL] Device      : {device}")
    print(f"[FL] Clients     : {config.NUM_CLIENTS}")
    print(f"[FL] Rounds      : {config.NUM_ROUNDS}")
    print(f"[FL] Local epochs: {config.LOCAL_EPOCHS}")
    print(f"[FL] Non-IID     : {config.NON_IID}")
    print(f"[FL] Data mode   : {config.DATA_MODE}")

    # -- Prepare data ----------------------------------------------------------
    hospital_datasets, test_dataset, scaler, n_features = prepare_selected_data()
    hospital_datasets, validation_dataset = split_client_validation_sets(hospital_datasets)
    pos_weight = training_pos_weight(hospital_datasets, device)
    print(f"[FL] Validation samples: {len(validation_dataset):,}")

    # -- Global model ----------------------------------------------------------
    global_model = build_model(n_features).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"\n[FL] Model: {type(global_model).__name__}")
    print(f"[FL] Total parameters: {total_params:,}")

    # -- Output dirs -----------------------------------------------------------
    os.makedirs(config.FEDERATED_DIR,  exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    rounds_log: List[Dict] = []
    best_validation_auc = float("-inf")
    best_round = 0
    remaining_patience = config.EARLY_STOPPING_PATIENCE
    best_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_validation_model.pt")

    print(f"\n[FL] Starting pure-Python FedAvg simulation ...\n")

    # ======================================================================
    # FEDERATED ROUNDS
    # ======================================================================
    for fl_round in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'-' * 50}")
        print(f"  Round {fl_round}/{config.NUM_ROUNDS}")
        print(f"{'-' * 50}")

        # Global weights to broadcast
        global_weights = get_parameters(global_model)

        client_weights_list: List[List[np.ndarray]] = []
        client_sizes:        List[int]               = []

        # -- Local training on each hospital ------------------------------
        for cid, h_dataset in enumerate(hospital_datasets):
            # Give each client a fresh copy of the global model
            local_model = copy.deepcopy(global_model)
            set_parameters(local_model, global_weights)

            weights, n_samples, local_loss = train_local(
                model=local_model,
                dataset=h_dataset,
                device=device,
                local_epochs=config.LOCAL_EPOCHS,
                batch_size=config.BATCH_SIZE,
                lr=config.LEARNING_RATE,
                cid=str(cid),
                fl_round=fl_round,
                pos_weight=pos_weight,
            )
            client_weights_list.append(weights)
            client_sizes.append(n_samples)

        # -- FedAvg aggregation --------------------------------------------
        aggregated_weights = fedavg_aggregate(client_weights_list, client_sizes)
        set_parameters(global_model, aggregated_weights)

        # -- Server-side validation; the test set remains untouched here. ---
        metrics = evaluate_model(global_model, validation_dataset, device=str(device))

        rounds_log.append({
            "round":     fl_round,
            "validation_loss":      metrics["loss"],
            "validation_accuracy":  metrics["accuracy"],
            "validation_precision": metrics["precision"],
            "validation_recall":    metrics["recall"],
            "validation_f1":        metrics["f1"],
            "validation_roc_auc":   metrics["roc_auc"],
            # Compatibility with existing plotting helpers.
            "loss": metrics["loss"], "accuracy": metrics["accuracy"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "f1": metrics["f1"], "roc_auc": metrics["roc_auc"],
        })

        print(
            f"\n  [Server] Round {fl_round:02d} | "
            f"validation_accuracy={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

        # Save checkpoint
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"round_{fl_round:02d}.pt")
        save_checkpoint(global_model, ckpt_path)
        if best_round == 0 or metrics["roc_auc"] > best_validation_auc:
            best_validation_auc = metrics["roc_auc"]
            best_round = fl_round
            remaining_patience = config.EARLY_STOPPING_PATIENCE
            save_checkpoint(global_model, best_ckpt)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                print(f"[FL] Early stopping after {config.EARLY_STOPPING_PATIENCE} rounds without validation AUC improvement.")
                break

    # ======================================================================
    # POST-TRAINING
    # ======================================================================

    # Persist rounds log
    rounds_log_path = os.path.join(config.FEDERATED_DIR, "rounds_log.json")
    with open(rounds_log_path, "w") as f:
        json.dump(rounds_log, f, indent=2)
    print(f"\n[FL] Rounds log saved -> {rounds_log_path}")

    # Final evaluation uses the best validation checkpoint exactly once.
    load_checkpoint(global_model, best_ckpt)
    print(f"[FL] Loaded best validation checkpoint from round {best_round} -> {best_ckpt}")

    val_probs, val_labels, _ = collect_predictions(global_model, validation_dataset, device=str(device))
    threshold = find_best_threshold(val_labels, val_probs)
    print(f"[FL] Validation-selected F1 threshold: {threshold:.2f}")
    final_metrics = evaluate_model(global_model, test_dataset, device=str(device), threshold=threshold)
    final_metrics["best_validation_round"] = best_round
    final_metrics["best_validation_roc_auc"] = best_validation_auc
    print_report(final_metrics.copy(), title="Federated Global Model - Final Test Evaluation")
    save_metrics(final_metrics, os.path.join(config.FEDERATED_DIR, "final_metrics.json"))

    # Plots: training curves
    if rounds_log:
        plot_accuracy_curve(
            rounds_log, os.path.join(config.FEDERATED_DIR, "accuracy_curve.png")
        )
        plot_loss_curve(
            rounds_log, os.path.join(config.FEDERATED_DIR, "loss_curve.png")
        )
    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        os.path.join(config.FEDERATED_DIR, "confusion_matrix.png"),
        title="Federated Global Model - Confusion Matrix",
    )

    # Per-hospital evaluation of the final global model
    comparison_metrics: Dict = {}

    baseline_metrics_path = os.path.join(config.BASELINE_DIR, "metrics.json")
    if os.path.exists(baseline_metrics_path):
        with open(baseline_metrics_path) as f:
            comparison_metrics["Centralized"] = json.load(f)
        print("[FL] Centralized baseline metrics loaded for comparison.")
    else:
        print(
            "[FL] Centralized baseline not found - run baseline.py first "
            "to include it in the comparison chart."
        )

    hospital_labels = ["H0 (Older >=55)", "H1 (Hypertensive)", "H2 (Balanced)"]
    local_hospital_metrics: Dict = {}

    for i, h_dataset in enumerate(hospital_datasets):
        label = hospital_labels[i] if config.NON_IID else f"Hospital {i}"
        h_metrics = evaluate_model(global_model, h_dataset, device=str(device), threshold=threshold)
        comparison_metrics[label] = h_metrics
        local_hospital_metrics[label] = h_metrics
        print(
            f"[FL] {label:25s} global-model eval -> "
            f"acc={h_metrics['accuracy']:.4f}  "
            f"f1={h_metrics['f1']:.4f}"
        )

    local_path = os.path.join(config.FEDERATED_DIR, "local_hospital_metrics.json")
    with open(local_path, "w") as f:
        json.dump(local_hospital_metrics, f, indent=2)
    print(f"[FL] Per-hospital metrics saved -> {local_path}")

    comparison_metrics["Federated (Global)"] = final_metrics

    plot_client_comparison(
        comparison_metrics,
        os.path.join(config.FEDERATED_DIR, "model_comparison.png"),
    )

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Federated Accuracy  : {final_metrics['accuracy']:.4f}")
    print(f"  Federated F1-Score  : {final_metrics['f1']:.4f}")
    print(f"  Federated ROC-AUC   : {final_metrics['roc_auc']:.4f}")

    if "Centralized" in comparison_metrics:
        base = comparison_metrics["Centralized"]
        gap  = final_metrics["accuracy"] - base["accuracy"]
        print(f"\n  Centralized Accuracy: {base['accuracy']:.4f}")
        print(f"  Accuracy gap (Federated - Centralized): {gap:+.4f}")

    print(f"\n[FL] All outputs saved to: {config.FEDERATED_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
