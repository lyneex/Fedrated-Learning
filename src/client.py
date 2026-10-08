"""
src/client.py -- Module 4: Flower Federated Learning Client.

Each FlowerClient represents one simulated hospital.
It trains on its own private TensorDataset and never exposes raw data --
only model weight arrays are passed to the server.

Supports both legacy 2-tuple (X, y) and multimodal 5-tuple
(x_ehr, x_vitals, x_img, mask, y) dataset formats.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

import src.config as config
from src.model import CardioMLP, get_parameters, set_parameters, build_model
from src.evaluate import evaluate_model


class FlowerClient(fl.client.NumPyClient):
    """
    A federated learning client representing one hospital node.

    Behaviour per round
    -------------------
    1. Receive global model weights from the server  (fit / evaluate).
    2. Train locally on its private dataset           (fit).
    3. Return updated weights to the server           (fit).
    4. Report local validation metrics                (evaluate).
    """

    def __init__(
        self,
        cid: str,
        n_features: int,
        train_dataset: TensorDataset,
        val_dataset: Optional[TensorDataset] = None,
    ):
        self.cid           = cid
        self.n_features    = n_features
        self.train_dataset = train_dataset
        # Use train_dataset for local evaluation if no separate val set given
        self.val_dataset   = val_dataset if val_dataset is not None else train_dataset
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model         = build_model(n_features).to(self.device)
        self.criterion     = nn.BCEWithLogitsLoss()

    # -- Flower interface ------------------------------------------------------

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model weights."""
        return get_parameters(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global weights -> train locally -> return updated weights.
        """
        # 1. Sync with global model
        set_parameters(self.model, parameters)

        # 2. Read per-round hyperparams sent by the server
        local_epochs = int(config.get("local_epochs", 3))
        batch_size   = int(config.get("batch_size",   64))
        lr           = float(config.get("lr",         1e-3))
        fl_round     = int(config.get("round",        0))

        # 3. Local training loop
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,   # avoid single-sample batches (LayerNorm is safe,
        )                      # but drop_last keeps batch stats consistent)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for _ in range(local_epochs):
            for batch in loader:
                if len(batch) == 5:
                    # Multimodal: (x_ehr, x_vitals, x_img, mask, y)
                    x_ehr, x_vit, x_img, mask, y_batch = [b.to(self.device) for b in batch]
                    optimizer.zero_grad()
                    logits = self.model(x_ehr, x_vit, x_img, mask)
                else:
                    # Legacy: (X, y)
                    X_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(X_batch)

                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

        avg_loss  = total_loss / max(n_batches, 1)
        n_samples = len(self.train_dataset)

        print(
            f"    [Hospital {self.cid}] Round {fl_round:02d} | "
            f"local_loss={avg_loss:.4f} | samples={n_samples:,}"
        )

        return get_parameters(self.model), n_samples, {"loss": float(avg_loss)}

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Receive global weights -> evaluate on local data -> return metrics.
        """
        set_parameters(self.model, parameters)
        metrics = evaluate_model(
            self.model, self.val_dataset, device=str(self.device)
        )
        return (
            float(metrics["loss"]),
            len(self.val_dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1":       float(metrics["f1"]),
            },
        )
