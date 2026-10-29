"""Real MIMIC-IV + MIMIC-CXR multimodal data preparation.

This module uses the admission linkage in MIMIC-CXR-DB.  It never attempts to
infer an admission from a CXR subject identifier alone: a radiograph without a
``hadm_id`` is excluded.  Vitals are read from MIMIC-IV ``chartevents`` in
chunks and joined through that same admission identifier.
"""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import src.config as config


def _require_columns(frame: pd.DataFrame, columns: set, name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _cxr_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    subject = str(int(subject_id))
    return os.path.join(
        config.MIMIC_CXR_ROOT,
        "files",
        f"p{subject[:2]}",
        f"p{subject}",
        f"s{int(study_id)}",
        f"{dicom_id}.jpg",
    )


def build_mimic_cohort() -> pd.DataFrame:
    """Create a linked EHR/image cohort manifest from MIMIC-CXR-DB tables.

    The target is configured by ``MIMIC_TARGET_COLUMN`` and must be a binary
    radiograph finding. Uncertain (-1) and missing labels are excluded instead
    of silently treating them as negatives.
    """
    for path in (config.MIMIC_ADMISSIONS_PATH, config.MIMIC_RADIOGRAPHS_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required MIMIC-CXR-DB table not found: {path}")

    admissions = pd.read_csv(config.MIMIC_ADMISSIONS_PATH)
    radiographs = pd.read_csv(config.MIMIC_RADIOGRAPHS_PATH)
    _require_columns(admissions, {"hadm_id", "subject_id"}, "admissions.csv")
    _require_columns(
        radiographs,
        {"hadm_id", "subject_id", "study_id", "dicom_id", config.MIMIC_TARGET_COLUMN},
        "radiographs.csv",
    )

    image_columns = ["hadm_id", "subject_id", "study_id", "dicom_id", config.MIMIC_TARGET_COLUMN]
    linked = radiographs[image_columns].copy()
    linked[config.MIMIC_TARGET_COLUMN] = pd.to_numeric(
        linked[config.MIMIC_TARGET_COLUMN], errors="coerce"
    )
    linked = linked[linked["hadm_id"].notna()]
    linked = linked[linked[config.MIMIC_TARGET_COLUMN].isin([0, 1])]
    linked["hadm_id"] = linked["hadm_id"].astype("int64")
    linked["subject_id"] = linked["subject_id"].astype("int64")

    admission_columns = [column for column in ("hadm_id", "subject_id", "anchor_age", "gender", "admission_type") if column in admissions]
    linked = linked.merge(admissions[admission_columns], on=["hadm_id", "subject_id"], how="inner")
    if linked.empty:
        raise ValueError("No admission-linked radiographs with an unambiguous binary target were found.")

    # EHR features available before using the image label. Categorical features
    # are one-hot encoded once and written into the manifest for reproducibility.
    if "anchor_age" not in linked:
        linked["anchor_age"] = 0
    if "gender" not in linked:
        linked["gender"] = ""
    linked["ehr_anchor_age"] = pd.to_numeric(linked["anchor_age"], errors="coerce").fillna(0)
    linked["ehr_gender_male"] = linked["gender"].astype(str).str.upper().eq("M").astype(float)
    if "admission_type" in linked:
        admission_type = pd.get_dummies(linked["admission_type"].fillna("UNKNOWN"), prefix="ehr_admission_type", dtype=float)
        linked = pd.concat([linked, admission_type], axis=1)

    linked["image_path"] = [
        _cxr_path(subject, study, dicom)
        for subject, study, dicom in linked[["subject_id", "study_id", "dicom_id"]].itertuples(index=False, name=None)
    ]
    linked = linked.rename(columns={config.MIMIC_TARGET_COLUMN: "label"})
    linked = linked.sort_values(["subject_id", "hadm_id", "study_id", "dicom_id"])

    if config.MIMIC_MAX_SAMPLES is not None and len(linked) > config.MIMIC_MAX_SAMPLES:
        linked = linked.sample(n=config.MIMIC_MAX_SAMPLES, random_state=config.SEED)

    manifest_columns = [
        "subject_id", "hadm_id", "study_id", "dicom_id", "image_path", "label",
        *sorted(column for column in linked.columns if column.startswith("ehr_")),
    ]
    cohort = linked[manifest_columns].reset_index(drop=True)
    os.makedirs(os.path.dirname(config.MIMIC_COHORT_PATH), exist_ok=True)
    cohort.to_csv(config.MIMIC_COHORT_PATH, index=False)
    print(f"[MIMIC] Cohort saved: {len(cohort):,} linked radiographs -> {config.MIMIC_COHORT_PATH}")
    return cohort


def _vital_channel(itemid: int) -> int:
    for channel, ids in enumerate(config.MIMIC_VITAL_ITEM_IDS.values()):
        if itemid in ids:
            return channel
    return -1


def _resample_vitals(events: pd.DataFrame) -> np.ndarray:
    """Convert one admission's irregular chart events to (T, 4) real values."""
    result = np.full((config.VITALS_SEQ_LEN, config.VITALS_CHANNELS), np.nan, dtype=np.float32)
    events = events.dropna(subset=["valuenum", "charttime"]).sort_values("charttime")
    for channel in range(config.VITALS_CHANNELS):
        values = events[events["channel"] == channel]
        if values.empty:
            continue
        seconds = (values["charttime"] - values["charttime"].iloc[0]).dt.total_seconds().to_numpy()
        measurements = values["valuenum"].to_numpy(dtype=np.float32)
        unique_seconds, unique_indices = np.unique(seconds, return_index=True)
        measurements = measurements[unique_indices]
        if len(measurements) == 1:
            result[:, channel] = measurements[0]
        else:
            grid = np.linspace(unique_seconds[0], unique_seconds[-1], config.VITALS_SEQ_LEN)
            result[:, channel] = np.interp(grid, unique_seconds, measurements)
    return result


def build_vitals_cache(cohort: pd.DataFrame | None = None) -> None:
    """Extract and resample real ICU vital measurements for the cohort.

    ``chartevents`` is large, so it is streamed in chunks and only rows for
    linked admissions and the four configured vital-sign channels are retained.
    Admissions without all required vital channels are excluded later.
    """
    if cohort is None:
        if not os.path.isfile(config.MIMIC_COHORT_PATH):
            cohort = build_mimic_cohort()
        else:
            cohort = pd.read_csv(config.MIMIC_COHORT_PATH)
    if not os.path.isfile(config.MIMIC_CHARTEVENTS_PATH):
        raise FileNotFoundError(f"MIMIC-IV chartevents file not found: {config.MIMIC_CHARTEVENTS_PATH}")

    hadm_ids = set(cohort["hadm_id"].astype("int64"))
    wanted_item_ids = {item for ids in config.MIMIC_VITAL_ITEM_IDS.values() for item in ids}
    parts = []
    for chunk in pd.read_csv(
        config.MIMIC_CHARTEVENTS_PATH,
        compression="infer",
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        chunksize=config.MIMIC_CHARTEVENTS_CHUNK_SIZE,
    ):
        filtered = chunk[
            chunk["hadm_id"].isin(hadm_ids)
            & chunk["itemid"].isin(wanted_item_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if not filtered.empty:
            parts.append(filtered)
    if not parts:
        raise ValueError("No configured vital-sign observations were found for the linked cohort.")

    events = pd.concat(parts, ignore_index=True)
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["channel"] = events["itemid"].map(_vital_channel)
    events = events[(events["channel"] >= 0) & events["charttime"].notna()]
    sequences, cached_hadm_ids = [], []
    for hadm_id, group in events.groupby("hadm_id", sort=True):
        sequence = _resample_vitals(group)
        if not np.isnan(sequence).any():
            cached_hadm_ids.append(hadm_id)
            sequences.append(sequence)
    if not sequences:
        raise ValueError("No linked admissions contained all four configured vital-sign channels.")
    os.makedirs(os.path.dirname(config.MIMIC_VITALS_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        config.MIMIC_VITALS_CACHE_PATH,
        hadm_ids=np.asarray(cached_hadm_ids, dtype=np.int64),
        vitals=np.stack(sequences).astype(np.float32),
    )
    print(f"[MIMIC] Vitals cache saved: {len(sequences):,} admissions -> {config.MIMIC_VITALS_CACHE_PATH}")


class MimicMultimodalDataset(Dataset):
    """Patient-linked EHR, vital-sign sequence, chest X-ray, mask, and label."""

    def __init__(self, ehr: np.ndarray, vitals: np.ndarray, image_paths: List[str], labels: np.ndarray):
        self.ehr = torch.tensor(ehr, dtype=torch.float32)
        self.vitals = torch.tensor(vitals, dtype=torch.float32)
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("L").resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.BILINEAR)
                image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read MIMIC-CXR image {path!r}") from exc
        mask = torch.ones(3, dtype=torch.float32)
        return self.ehr[index], self.vitals[index], image_tensor, mask, self.labels[index]


def prepare_mimic_multimodal_data() -> Tuple[List[Dataset], Dataset, StandardScaler, int]:
    """Load the real linked cohort, fit preprocessing on training patients only, and split by patient."""
    if not os.path.isfile(config.MIMIC_COHORT_PATH) or not os.path.isfile(config.MIMIC_VITALS_CACHE_PATH):
        raise FileNotFoundError(
            "Real MIMIC artifacts are missing. Run `python3 prepare_mimic_data.py` after "
            "placing the credentialed files in the configured data directories."
        )
    cohort = pd.read_csv(config.MIMIC_COHORT_PATH)
    _require_columns(cohort, {"subject_id", "hadm_id", "image_path", "label"}, "MIMIC cohort manifest")
    ehr_columns = sorted(column for column in cohort.columns if column.startswith("ehr_"))
    if not ehr_columns:
        raise ValueError("MIMIC cohort manifest has no EHR feature columns.")
    cache = np.load(config.MIMIC_VITALS_CACHE_PATH)
    sequence_by_hadm = dict(zip(cache["hadm_ids"].astype(int), cache["vitals"]))
    keep = cohort["hadm_id"].astype(int).isin(sequence_by_hadm) & cohort["image_path"].map(os.path.isfile)
    cohort = cohort.loc[keep].reset_index(drop=True)
    if cohort["label"].nunique() != 2 or cohort["subject_id"].nunique() < config.NUM_CLIENTS + 1:
        raise ValueError("Cohort needs both labels and at least NUM_CLIENTS + 1 distinct patients.")

    indices = np.arange(len(cohort))
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.SEED)
    train_idx, test_idx = next(splitter.split(indices, cohort["label"], groups=cohort["subject_id"]))
    ehr = cohort[ehr_columns].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(ehr[train_idx])
    ehr = scaler.transform(ehr).astype(np.float32)
    vitals = np.stack([sequence_by_hadm[int(hadm)] for hadm in cohort["hadm_id"]]).astype(np.float32)
    vitals_mean = vitals[train_idx].mean(axis=(0, 1), keepdims=True)
    vitals_std = vitals[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
    vitals = (vitals - vitals_mean) / vitals_std

    patient_ids = cohort["subject_id"].to_numpy()
    train_patients = np.random.default_rng(config.SEED).permutation(np.unique(patient_ids[train_idx]))
    hospital_patients = np.array_split(train_patients, config.NUM_CLIENTS)
    hospitals = []
    for patients in hospital_patients:
        hospital_idx = train_idx[np.isin(patient_ids[train_idx], patients)]
        hospitals.append(MimicMultimodalDataset(
            ehr[hospital_idx], vitals[hospital_idx], cohort["image_path"].iloc[hospital_idx].tolist(),
            cohort["label"].iloc[hospital_idx].to_numpy(dtype=np.float32),
        ))
    test_dataset = MimicMultimodalDataset(
        ehr[test_idx], vitals[test_idx], cohort["image_path"].iloc[test_idx].tolist(),
        cohort["label"].iloc[test_idx].to_numpy(dtype=np.float32),
    )
    print(f"[MIMIC] Real multimodal split: {len(train_idx):,} train | {len(test_idx):,} test")
    return hospitals, test_dataset, scaler, len(ehr_columns)
