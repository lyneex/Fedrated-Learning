"""
src/visualize.py -- Module 7: Visualization helpers.

All functions save plots to disk (Agg / non-interactive backend).
Functions
---------
plot_baseline_history(history, save_path)
plot_accuracy_curve(rounds_log, save_path)
plot_loss_curve(rounds_log, save_path)
plot_confusion_matrix(cm, save_path, title)
plot_client_comparison(model_metrics, save_path)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # non-interactive, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# -- Style setup ---------------------------------------------------------------
# Try modern seaborn style name first (matplotlib >= 3.6), fall back gracefully 
for _style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
FIGSIZE_WIDE = (9, 4)
FIGSIZE_SQ   = (5, 4)
DPI = 150


def _savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved -> {path}")


# -- 1. Centralized baseline training history ----------------------------------

def plot_baseline_history(history: Dict[str, List[float]], save_path: str) -> None:
    """Two-panel plot: BCE loss and accuracy per epoch for the centralized baseline."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["loss"], marker="o", color=PALETTE[1], linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("BCE Loss", fontsize=11)
    ax1.set_title("Centralized Training Loss", fontsize=13, fontweight="bold")

    ax2.plot(epochs, history["accuracy"], marker="o", color=PALETTE[0], linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Centralized Training Accuracy", fontsize=13, fontweight="bold")
    ax2.set_ylim(0.5, 1.0)

    fig.suptitle("Centralized Baseline -- Training History", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _savefig(fig, save_path)


# -- 2. Federated accuracy per round ------------------------------------------

def plot_accuracy_curve(rounds_log: List[Dict], save_path: str) -> None:
    """Line chart of global model accuracy over federated rounds."""
    rounds = [r["round"] for r in rounds_log]
    accs   = [r["accuracy"] for r in rounds_log]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(rounds, accs, marker="o", color=PALETTE[0], linewidth=2.5, label="Global Accuracy")
    ax.axhline(y=max(accs), color=PALETTE[0], linestyle="--", alpha=0.4,
               label=f"Best: {max(accs):.4f}")
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Global Model Accuracy per Federated Round", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(rounds)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, save_path)


# -- 3. Federated loss per round -----------------------------------------------

def plot_loss_curve(rounds_log: List[Dict], save_path: str) -> None:
    """Line chart of global model BCE loss over federated rounds."""
    rounds = [r["round"] for r in rounds_log]
    losses = [r["loss"] for r in rounds_log]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(rounds, losses, marker="s", color=PALETTE[1], linewidth=2.5, label="Global BCE Loss")
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("BCE Loss", fontsize=11)
    ax.set_title("Global Model Loss per Federated Round", fontsize=13, fontweight="bold")
    ax.set_xticks(rounds)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, save_path)


# -- 4. Confusion matrix heatmap -----------------------------------------------

def plot_confusion_matrix(
    cm: List[List[int]],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Annotated heatmap of a 2×2 confusion matrix."""
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    sns.heatmap(
        cm_arr,
        annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred: No CVD (0)", "Pred: CVD (1)"],
        yticklabels=["True: No CVD (0)", "True: CVD (1)"],
        linewidths=0.5, linecolor="gray",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    fig.tight_layout()
    _savefig(fig, save_path)


# -- 5. Cross-model comparison bar chart --------------------------------------

def plot_client_comparison(
    model_metrics: Dict[str, Dict[str, Any]],
    save_path: str,
) -> None:
    """
    Grouped bar chart comparing Accuracy, F1, and ROC-AUC
    across centralized / local-hospital / federated models.

    Parameters
    ----------
    model_metrics : {model_name: metrics_dict, ...}
    """
    names = list(model_metrics.keys())
    accs  = [model_metrics[n]["accuracy"] for n in names]
    f1s   = [model_metrics[n]["f1"]       for n in names]
    aucs  = [model_metrics[n]["roc_auc"]  for n in names]

    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2.5), 5))

    bars1 = ax.bar(x - width, accs, width, label="Accuracy",  color=PALETTE[0], alpha=0.90)
    bars2 = ax.bar(x,         f1s,  width, label="F1-Score",  color=PALETTE[1], alpha=0.90)
    bars3 = ax.bar(x + width, aucs, width, label="ROC-AUC",   color=PALETTE[2], alpha=0.90)

    # Value annotations
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Model Comparison: Centralized vs Local Hospitals vs Federated Global",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, save_path)
