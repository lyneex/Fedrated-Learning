"""
src/model.py -- Module 3: MLP Classifier for binary cardiovascular risk prediction.

Architecture:
    Linear(n_features -> 64) -> LayerNorm -> ReLU -> Dropout
    Linear(64 -> 32)          -> LayerNorm -> ReLU -> Dropout
    Linear(32 -> 1)           -> (raw logit; use BCEWithLogitsLoss externally)

LayerNorm is used instead of BatchNorm to avoid issues with batch size = 1
at inference time or with small tail batches during local training.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
import numpy as np
import torch
import torch.nn as nn

import src.config as config


# -- Model ---------------------------------------------------------------------

class CardioMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary tabular classification.
    Outputs raw logits -- pair with BCEWithLogitsLoss during training
    or apply sigmoid to get probabilities at inference.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: List[int] = config.HIDDEN_DIMS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))   # logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)         # shape: (batch_size,)


# -- Parameter helpers (used by Flower client/server) --------------------------

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Restore model weights from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# -- Checkpoint helpers ---------------------------------------------------------

def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str) -> None:
    model.load_state_dict(torch.load(path, map_location="cpu"))


# -- Model factory --------------------------------------------------------------

def build_model(n_features: int):
    """
    Factory: returns the model selected by config.DATA_MODE.
    """
    if config.DATA_MODE in {"image", "ecg_image"}:
        from src.multimodal_model import ImageClassifier
        return ImageClassifier()
    if config.DATA_MODE == "multimodal":
        from src.multimodal_model import MultimodalFusionModel
        return MultimodalFusionModel(n_ehr_features=n_features)
    if config.DATA_MODE == "tabular":
        return CardioMLP(n_features=n_features)
    raise ValueError(
        f"Unsupported DATA_MODE={config.DATA_MODE!r}. "
        "Choose 'tabular', 'image', or 'multimodal'."
    )

