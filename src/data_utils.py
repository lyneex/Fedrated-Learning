"""
src/data_utils.py -- Modules 1 & 2: Data Ingestion, Cleaning, and Feature Engineering.

Pipeline
--------
load_raw()           -> raw DataFrame (semicolon-delimited Kaggle CSV)
clean()              -> remove invalid BP / height / weight rows + duplicates
engineer_features()  -> add age_years, bmi, pulse_pressure, hypertension_flag; recode gender
prepare_data()       -> full pipeline: returns hospital TensorDatasets + test TensorDataset

Non-IID hospital split (when NON_IID=True, NUM_CLIENTS=3)
-----------------------------------------------------------
Hospital 0 -- older patients (age >= 55 years) overrepresented
Hospital 1 -- hypertensive patients (ap_hi >= 140 or ap_lo >= 90) overrepresented
Hospital 2 -- balanced / residual mix
"""
import os
import sys
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

import src.config as config

# -- Feature lists --------------------------------------------------------------
# Numeric features that will be scaled by StandardScaler
NUMERIC_FEATURES = [
    "age_years", "height", "weight",
    "ap_hi", "ap_lo",
    "bmi", "pulse_pressure",
]

# Categorical / binary features kept as-is (no scaling needed)
CATEGORICAL_FEATURES = [
    "gender",       # 0 = female, 1 = male
    "cholesterol",  # 1 = normal, 2 = above, 3 = well above
    "gluc",         # 1 = normal, 2 = above, 3 = well above
    "smoke",        # 0/1
    "alco",         # 0/1
    "active",       # 0/1
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cardio"


# -- Step 1: Load ---------------------------------------------------------------

def load_raw(path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load the cardio CSV (semicolon-delimited Kaggle format)."""
    df = pd.read_csv(path, sep=";")
    print(f"[Data] Loaded  {len(df):,} rows, {df.shape[1]} columns")
    return df


# -- Step 2: Clean --------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and physiologically impossible values."""
    df = df.copy()

    # Drop the 'id' surrogate key if present
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)

    # Blood pressure: physically impossible values
    mask = (
        (df["ap_hi"] >= 50)  & (df["ap_hi"] <= 250) &
        (df["ap_lo"] >= 30)  & (df["ap_lo"] <= 200) &
        (df["ap_hi"] > df["ap_lo"])   # systolic must exceed diastolic
    )
    df = df[mask]

    # Implausible anthropometrics
    mask2 = (
        (df["height"] >= 100) & (df["height"] <= 250) &
        (df["weight"] >= 30)  & (df["weight"] <= 200)
    )
    df = df[mask2]

    df.reset_index(drop=True, inplace=True)
    print(f"[Data] Cleaned {len(df):,} rows  (removed {before - len(df):,} invalid rows)")
    return df


# -- Step 3: Feature Engineering ------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived medical features and recode categorical columns."""
    df = df.copy()

    # Age: original column is in days -> convert to years
    df["age_years"] = df["age"] / 365.25

    # BMI = weight(kg) / height(m)^2
    df["bmi"] = df["weight"] / (df["height"] / 100.0) ** 2

    # Pulse pressure = systolic − diastolic 
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    # Hypertension flag (used for non-IID split, NOT as a model feature)
    df["hypertension_flag"] = (
        (df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)
    ).astype(int)

    # Gender: recode 1 -> 0 (female), 2 -> 1 (male)
    df["gender"] = df["gender"].map({1: 0, 2: 1})    

    return df


# -- Step 4: Preprocess (scale) -------------------------------------------------

def _scale(df: pd.DataFrame, scaler: StandardScaler,
           fit: bool = False) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply StandardScaler to numeric columns and return (X, y, scaler).
    If fit=True, the scaler is fitted on df before transforming.
    """
    df = df.copy()
    if fit:
        df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])
    else:
        df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])

    X = df[ALL_FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    return X, y


# -- Step 5: Hospital Splits ----------------------------------------------------

def _make_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def _hospital_indices(
    df: pd.DataFrame,
    num_clients: int,
    non_iid: bool,
    seed: int,
) -> List[np.ndarray]:
    """
    Return a list of index arrays, one per hospital.
    `df` must already have `age_years` and `hypertension_flag` columns,
    and must be reset-indexed (0 ... N-1).
    """
    n = len(df)
    rng = np.random.default_rng(seed)

    if not non_iid or num_clients != 3:
        # IID: equal random partitions
        indices = rng.permutation(n)
        return [arr for arr in np.array_split(indices, num_clients)]

    # -- Non-IID split --------------------------------------------------------
    old_mask = (df["age_years"] >= 55).values
    hyp_mask = (df["hypertension_flag"] == 1).values

    old_idx      = np.where(old_mask)[0]
    hyp_only_idx = np.where(hyp_mask & ~old_mask)[0]           # hyp but NOT old
    rest_idx     = np.where(~old_mask & ~hyp_mask)[0]          # residual

    rng.shuffle(old_idx)
    rng.shuffle(hyp_only_idx)
    rng.shuffle(rest_idx)

    target = n // 3   # approximate equal size target

    # Hospital 0: all older patients + padding from rest if needed
    h0_extra_n = max(0, target - len(old_idx))
    h0_extra   = rest_idx[:h0_extra_n]
    h0_idx     = np.concatenate([old_idx, h0_extra])

    # Hospital 1: all hypertensive-only + padding from rest if needed
    used_rest  = h0_extra_n
    h1_extra_n = max(0, target - len(hyp_only_idx))
    h1_extra   = rest_idx[used_rest : used_rest + h1_extra_n]
    h1_idx     = np.concatenate([hyp_only_idx, h1_extra])

    # Hospital 2: the remainder (balanced / general)
    used_set = set(h0_idx.tolist()) | set(h1_idx.tolist())
    h2_idx   = np.array([i for i in range(n) if i not in used_set])

    hospital_indices = [h0_idx.astype(int), h1_idx.astype(int), h2_idx.astype(int)]

    labels = ["H0 (older, age>=55)", "H1 (hypertensive)", "H2 (balanced)"]
    print("\n[Data] Non-IID hospital split:")
    for label, idx in zip(labels, hospital_indices):
        cardio_rate = df.iloc[idx][TARGET].mean()
        print(f"       {label:25s} | {len(idx):6,} samples | cardio rate: {cardio_rate:.3f}")

    return hospital_indices


# -- Full pipeline --------------------------------------------------------------

def prepare_data() -> Tuple[
    List[TensorDataset],   # hospital_datasets
    TensorDataset,          # test_dataset
    StandardScaler,         # fitted scaler (save for inference)
    int,                    # n_features
]:
    """
    Full pipeline: load -> clean -> engineer -> train/test split ->
    fit scaler -> non-IID hospital split -> return TensorDatasets.

    Returns
    -------
    hospital_datasets : List[TensorDataset]  -- one per hospital client
    test_dataset      : TensorDataset        -- shared held-out test set
    scaler            : StandardScaler       -- fitted on training split
    n_features        : int
    """
    # Create output dirs
    os.makedirs(config.RUNS_DIR, exist_ok=True)

    # 1. Load & clean & engineer
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)

    # 2. Train / test split (stratified on cardio)
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[TARGET],
        random_state=config.SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"[Data] Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"[Data] Train class balance  cardio=0: {(train_df[TARGET]==0).sum():,}  "
          f"cardio=1: {(train_df[TARGET]==1).sum():,}")

    # 3. Fit scaler on training split only, then transform both splits
    scaler = StandardScaler()
    X_train, y_train = _scale(train_df, scaler, fit=True)
    X_test,  y_test  = _scale(test_df,  scaler, fit=False)

    n_features = X_train.shape[1]
    print(f"[Data] n_features = {n_features}")

    # 4. Hospital splits (uses unscaled train_df for age/hypertension logic)
    hospital_indices = _hospital_indices(
        train_df,
        config.NUM_CLIENTS,
        config.NON_IID,
        config.SEED,
    )

    hospital_datasets = [
        _make_tensor_dataset(X_train[idx], y_train[idx])
        for idx in hospital_indices
    ]
    test_dataset = _make_tensor_dataset(X_test, y_test)

    # 5. Persist the scaler so inference scripts can reuse it
    scaler_path = os.path.join(config.RUNS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Data] Scaler saved -> {scaler_path}")

    return hospital_datasets, test_dataset, scaler, n_features


# -- Multimodal pipeline -------------------------------------------------------

def _make_multimodal_dataset(
    X_ehr: np.ndarray,
    X_vitals: np.ndarray,
    X_img: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
) -> TensorDataset:
    """Create a TensorDataset with 5-tuple: (x_ehr, x_vitals, x_img, mask, y)."""
    return TensorDataset(
        torch.tensor(X_ehr, dtype=torch.float32),
        torch.tensor(X_vitals, dtype=torch.float32),
        torch.tensor(X_img, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def prepare_multimodal_data() -> Tuple[
    List[TensorDataset],   # hospital multimodal datasets
    TensorDataset,          # test multimodal dataset
    StandardScaler,         # fitted scaler
    int,                    # n_ehr_features
]:
    """
    Full multimodal pipeline: load -> clean -> engineer -> split ->
    generate synthetic vitals & imaging -> apply per-hospital modality masks ->
    return TensorDatasets with (x_ehr, x_vitals, x_img, mask, y) tuples.

    Missing modalities are zero-filled and the presence mask indicates
    which modalities are available, per §4.2.2 of the Methodology.

    Returns
    -------
    hospital_datasets : List[TensorDataset]  -- per-hospital 5-tuple datasets
    test_dataset      : TensorDataset        -- test set with all modalities
    scaler            : StandardScaler       -- fitted on training split
    n_ehr_features    : int
    """
    from src.synthetic_data import generate_vitals, generate_imaging

    # Create output dirs
    os.makedirs(config.RUNS_DIR, exist_ok=True)

    # 1. Load & clean & engineer
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)

    # 2. Train / test split (stratified on cardio)
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[TARGET],
        random_state=config.SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"[Data] Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"[Data] Train class balance  cardio=0: {(train_df[TARGET]==0).sum():,}  "
          f"cardio=1: {(train_df[TARGET]==1).sum():,}")

    # 3. Fit scaler on training split only, then transform both splits
    scaler = StandardScaler()
    X_train, y_train = _scale(train_df, scaler, fit=True)
    X_test,  y_test  = _scale(test_df,  scaler, fit=False)

    n_ehr_features = X_train.shape[1]
    print(f"[Data] n_ehr_features = {n_ehr_features}")

    # 4. Generate synthetic vitals & imaging from scaled tabular features
    print("[Data] Generating synthetic vitals time-series ...")
    vitals_train = generate_vitals(X_train, seed=config.SEED)
    vitals_test  = generate_vitals(X_test,  seed=config.SEED + 1)

    print("[Data] Generating synthetic medical images ...")
    img_train = generate_imaging(X_train, seed=config.SEED)
    img_test  = generate_imaging(X_test,  seed=config.SEED + 1)

    print(f"[Data] Vitals shape: train={vitals_train.shape}, test={vitals_test.shape}")
    print(f"[Data] Imaging shape: train={img_train.shape}, test={img_test.shape}")

    # 5. Hospital splits (uses unscaled train_df for age/hypertension logic)
    hospital_indices = _hospital_indices(
        train_df,
        config.NUM_CLIENTS,
        config.NON_IID,
        config.SEED,
    )

    # 6. Build per-hospital datasets with modality-specific masks
    hospital_datasets = []
    for cid, idx in enumerate(hospital_indices):
        has_ehr, has_vit, has_img = config.HOSPITAL_MODALITIES.get(
            cid, (True, False, False)
        )

        h_ehr    = X_train[idx]
        h_vitals = vitals_train[idx] if has_vit else np.zeros_like(vitals_train[idx])
        h_img    = img_train[idx]    if has_img else np.zeros_like(img_train[idx])

        # Build mask: (n_samples, 3)
        n = len(idx)
        h_mask = np.zeros((n, 3), dtype=np.float32)
        h_mask[:, 0] = float(has_ehr)
        h_mask[:, 1] = float(has_vit)
        h_mask[:, 2] = float(has_img)

        h_y = y_train[idx]
        hospital_datasets.append(
            _make_multimodal_dataset(h_ehr, h_vitals, h_img, h_mask, h_y)
        )

        modality_str = " + ".join(
            m for m, avail in [("EHR", has_ehr), ("Vitals", has_vit), ("Imaging", has_img)]
            if avail
        )
        print(f"[Data] Hospital {cid}: {len(idx):,} samples | modalities: {modality_str}")

    # 7. Test set gets ALL modalities (full mask)
    n_test = len(X_test)
    test_mask = np.ones((n_test, 3), dtype=np.float32)
    test_dataset = _make_multimodal_dataset(
        X_test, vitals_test, img_test, test_mask, y_test
    )

    # 8. Persist the scaler
    scaler_path = os.path.join(config.RUNS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Data] Scaler saved -> {scaler_path}")

    return hospital_datasets, test_dataset, scaler, n_ehr_features


# -- DataLoader helper ----------------------------------------------------------

def make_dataloader(
    dataset: TensorDataset,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)


