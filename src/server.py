"""
src/server.py -- Module 5: Federated Server setup.

Provides:
  build_strategy()      -- constructs a FedAvg strategy with server-side evaluation
  get_evaluate_fn()     -- returns the evaluate callback (called after each round)
  get_fit_config_fn()   -- returns the per-round fit-config callback
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import TensorDataset

import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server.strategy import FedAvg

import src.config as config
from src.model import CardioMLP, get_parameters, set_parameters, save_checkpoint
from src.evaluate import evaluate_model


# -- Server-side evaluation callback ------------------------------------------

def get_evaluate_fn(
    model: CardioMLP,
    test_dataset: TensorDataset,
    rounds_log: List[Dict],
):
    """
    Return a Flower-compatible server-side evaluation function.

    This function is called by FedAvg after every aggregation round.
    It evaluates the freshly aggregated global model on the shared
    held-out test set and appends the results to `rounds_log`.

    Flower signature: (server_round, parameters, config) -> (loss, metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config_dict: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        # Load aggregated weights into the evaluation model
        set_parameters(model, parameters)
        model.eval()

        metrics = evaluate_model(model, test_dataset, device=str(device))

        # Store for later plotting
        rounds_log.append({
            "round":     server_round,
            "loss":      metrics["loss"],
            "accuracy":  metrics["accuracy"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1":        metrics["f1"],
            "roc_auc":   metrics["roc_auc"],
        })

        print(
            f"\n  [Server] Round {server_round:02d} | "
            f"test_accuracy={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

        # Save a per-round checkpoint
        ckpt_path = os.path.join(
            config.CHECKPOINT_DIR, f"round_{server_round:02d}.pt"
        )
        save_checkpoint(model, ckpt_path)

        return metrics["loss"], {
            "accuracy": metrics["accuracy"],
            "f1":       metrics["f1"],
            "roc_auc":  metrics["roc_auc"],
        }

    return evaluate_fn


# -- Per-round fit-config callback ---------------------------------------------

def get_fit_config_fn():
    """
    Return a function that supplies training hyperparameters to each client
    before every round.  Flower calls this once per round.
    """
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "round":        server_round,
            "local_epochs": config.LOCAL_EPOCHS,
            "batch_size":   config.BATCH_SIZE,
            "lr":           config.LEARNING_RATE,
        }
    return fit_config


# -- Strategy builder ----------------------------------------------------------

def build_strategy(
    model: CardioMLP,
    test_dataset: TensorDataset,
    rounds_log: List[Dict],
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
) -> FedAvg:
    """
    Build a FedAvg strategy with:
    - all clients participating in every round (fraction_fit = 1.0)
    - server-side test-set evaluation after each round
    - per-round checkpoint saving
    - initial parameters seeded from the global model
    """
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(model))

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=config.NUM_CLIENTS,
        min_evaluate_clients=config.NUM_CLIENTS,
        min_available_clients=config.NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(model, test_dataset, rounds_log),
        on_fit_config_fn=get_fit_config_fn(),
        initial_parameters=initial_params,
    )

    return strategy
