"""Separate clinical-tabular and ECG-image experiments for the Kaggle data.

The source has 4,327 clinical rows and 928 ECG images but no record-to-image
mapping.  These loaders deliberately prevent an invalid multimodal pairing.
Each task is binary: Normal=0 and any other diagnosis=1.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Subset, TensorDataset

import src.config as config
from src.data_utils import ImageManifestDataset


def _split_clients(indices: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
    """Create reproducible IID client partitions with preserved class balance."""
    splitter = StratifiedKFold(
        n_splits=config.NUM_CLIENTS, shuffle=True, random_state=config.SEED
    )
    client_indices = []
    for _, client_positions in splitter.split(indices, labels[indices]):
        client_indices.append(indices[client_positions])
    return client_indices


def _print_client_balance(hospitals: List[TensorDataset | Subset]) -> None:
    for cid, dataset in enumerate(hospitals):
        if isinstance(dataset, TensorDataset):
            labels = dataset.tensors[-1].numpy()
        else:
            labels = dataset.dataset.labels[np.asarray(dataset.indices, dtype=int)].numpy()
        print(f"[Kaggle] Client {cid}: {len(dataset):,} samples | abnormal rate {labels.mean():.3f}")


def prepare_kaggle_tabular_data() -> Tuple[List[TensorDataset], TensorDataset, StandardScaler, int]:
    """Load the clinical CSV without using its unresolvable ECG image path."""
    if not os.path.isfile(config.KAGGLE_CARDIO_CSV):
        raise FileNotFoundError(f"Kaggle clinical CSV not found: {config.KAGGLE_CARDIO_CSV}")
    frame = pd.read_csv(config.KAGGLE_CARDIO_CSV)
    required = {"Gender", "ECG_Image_Path", "Target"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Kaggle clinical CSV missing columns: {sorted(missing)}")
    labels = frame["Target"].astype(str).ne("Normal").astype(np.float32).to_numpy()
    features = frame.drop(columns=["Target", "ECG_Image_Path"]).copy()
    features["Gender"] = features["Gender"].astype(str).str.lower().map({"female": 0.0, "male": 1.0})
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=config.TEST_SIZE, stratify=labels, random_state=config.SEED)
    scaler = StandardScaler().fit(features.iloc[train_idx])
    values = scaler.transform(features).astype(np.float32)
    hospitals = [
        TensorDataset(torch.tensor(values[idx]), torch.tensor(labels[idx]))
        for idx in _split_clients(train_idx, labels)
    ]
    test_dataset = TensorDataset(torch.tensor(values[test_idx]), torch.tensor(labels[test_idx]))
    print(f"[Kaggle] Clinical: {len(train_idx):,} train | {len(test_idx):,} test | {values.shape[1]} features")
    _print_client_balance(hospitals)
    return hospitals, test_dataset, scaler, values.shape[1]


def prepare_kaggle_ecg_image_data() -> Tuple[List[Subset], Subset, None, int]:
    """Load ECG image folders, classifying normal versus any abnormal ECG."""
    rows = []
    for class_name in ("normal", "abnormal_heartbeat", "myocardial_infarction", "post_mi_history"):
        class_dir = os.path.join(config.KAGGLE_ECG_IMAGE_ROOT, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected ECG class folder not found: {class_dir}")
        label = 0.0 if class_name == "normal" else 1.0
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append((os.path.join(class_dir, filename), label))
    paths, labels = zip(*rows)
    labels_array = np.asarray(labels, dtype=np.float32)
    indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(indices, test_size=config.TEST_SIZE, stratify=labels_array, random_state=config.SEED)
    dataset = ImageManifestDataset(list(paths), labels_array)
    hospitals = [Subset(dataset, idx.tolist()) for idx in _split_clients(train_idx, labels_array)]
    test_dataset = Subset(dataset, test_idx.tolist())
    print(f"[Kaggle] ECG images: {len(train_idx):,} train | {len(test_idx):,} test | Normal=0, abnormal=1")
    _print_client_balance(hospitals)
    return hospitals, test_dataset, None, 0
